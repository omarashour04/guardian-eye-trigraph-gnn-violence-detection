"""FAISS + SQLite index layer for Legal Consequences RAG chunks.

This module stores LegalChunk vectors and metadata for future retrieval. It
does not implement retrieval ranking, summarization, orchestration, evaluation,
guardrails, or LLM calls.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from rag_service.legal_chunker import LegalChunk


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
LARGE_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]] | np.ndarray:
        """Return one embedding vector per input text."""


class SentenceTransformerEmbeddingProvider:
    """Lazy sentence-transformers provider for local indexing outside tests."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        passages = [f"passage: {text}" for text in texts]
        return self._model.encode(
            passages,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


@dataclass(frozen=True)
class LegalMetadataRow:
    vector_row: int
    chunk_id: str
    country: str
    source_language: str
    law_title: str
    article_number: str | None
    section_title: str | None
    violence_category: str | None
    source_url: str
    official_source: bool
    text: str
    matched_keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LegalVectorIndex:
    """In-memory handle for a FAISS index plus aligned SQLite metadata rows."""

    def __init__(
        self,
        *,
        faiss_index: Any,
        metadata_rows: list[LegalMetadataRow],
        embedding_dimension: int,
    ) -> None:
        self.faiss_index = faiss_index
        self.metadata_rows = metadata_rows
        self.embedding_dimension = embedding_dimension

    @classmethod
    def build(
        cls,
        chunks: list[LegalChunk | dict[str, Any]],
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> "LegalVectorIndex":
        normalized_chunks = [_coerce_chunk(chunk) for chunk in chunks]
        if not normalized_chunks:
            raise ValueError("Cannot build legal index without chunks.")

        provider = embedding_provider or SentenceTransformerEmbeddingProvider()
        embeddings = _as_float32_matrix(
            provider.embed_texts([chunk["text"] for chunk in normalized_chunks])
        )
        if embeddings.shape[0] != len(normalized_chunks):
            raise ValueError("Embedding count must match chunk count.")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        metadata_rows = [
            _metadata_row_from_chunk(vector_row, chunk)
            for vector_row, chunk in enumerate(normalized_chunks)
        ]
        return cls(
            faiss_index=index,
            metadata_rows=metadata_rows,
            embedding_dimension=embeddings.shape[1],
        )

    def save(self, index_path: str | Path, metadata_db_path: str | Path) -> None:
        index_path = Path(index_path)
        metadata_db_path = Path(metadata_db_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_db_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(index_path))
        _write_metadata_db(metadata_db_path, self.metadata_rows)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_db_path: str | Path,
    ) -> "LegalVectorIndex":
        index = faiss.read_index(str(index_path))
        metadata_rows = _read_metadata_db(Path(metadata_db_path))
        return cls(
            faiss_index=index,
            metadata_rows=metadata_rows,
            embedding_dimension=index.d,
        )

    def index_count(self) -> int:
        return int(self.faiss_index.ntotal)

    def metadata_count(self) -> int:
        return len(self.metadata_rows)

    def get_metadata_by_vector_row(self, vector_row: int) -> LegalMetadataRow | None:
        for row in self.metadata_rows:
            if row.vector_row == vector_row:
                return row
        return None

    def get_metadata_by_chunk_id(self, chunk_id: str) -> LegalMetadataRow | None:
        for row in self.metadata_rows:
            if row.chunk_id == chunk_id:
                return row
        return None


def build_legal_index(
    chunks: list[LegalChunk | dict[str, Any]],
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> LegalVectorIndex:
    return LegalVectorIndex.build(chunks, embedding_provider=embedding_provider)


def save_legal_index(
    legal_index: LegalVectorIndex,
    index_path: str | Path,
    metadata_db_path: str | Path,
) -> None:
    legal_index.save(index_path, metadata_db_path)


def load_legal_index(
    index_path: str | Path,
    metadata_db_path: str | Path,
) -> LegalVectorIndex:
    return LegalVectorIndex.load(index_path, metadata_db_path)


def _coerce_chunk(chunk: LegalChunk | dict[str, Any]) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    if is_dataclass(chunk):
        return asdict(chunk)
    raise TypeError("Legal index expects LegalChunk objects or dicts.")


def _as_float32_matrix(embeddings: list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")
    if matrix.shape[1] == 0:
        raise ValueError("Embeddings must have at least one dimension.")
    return np.ascontiguousarray(matrix)


def _metadata_row_from_chunk(vector_row: int, chunk: dict[str, Any]) -> LegalMetadataRow:
    return LegalMetadataRow(
        vector_row=vector_row,
        chunk_id=chunk["chunk_id"],
        country=chunk["country"],
        source_language=chunk["source_language"],
        law_title=chunk["law_title"],
        article_number=chunk.get("article_number"),
        section_title=chunk.get("section_title"),
        violence_category=chunk.get("violence_category"),
        source_url=chunk["source_url"],
        official_source=bool(chunk["official_source"]),
        text=chunk["text"],
        matched_keywords=list(chunk.get("matched_keywords") or []),
    )


def _write_metadata_db(
    metadata_db_path: Path,
    metadata_rows: list[LegalMetadataRow],
) -> None:
    with sqlite3.connect(metadata_db_path) as connection:
        connection.execute("DROP TABLE IF EXISTS legal_chunk_metadata")
        connection.execute(
            """
            CREATE TABLE legal_chunk_metadata (
                vector_row INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                country TEXT NOT NULL,
                source_language TEXT NOT NULL,
                law_title TEXT NOT NULL,
                article_number TEXT,
                section_title TEXT,
                violence_category TEXT,
                source_url TEXT NOT NULL,
                official_source INTEGER NOT NULL,
                text TEXT NOT NULL,
                matched_keywords TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO legal_chunk_metadata (
                vector_row,
                chunk_id,
                country,
                source_language,
                law_title,
                article_number,
                section_title,
                violence_category,
                source_url,
                official_source,
                text,
                matched_keywords
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.vector_row,
                    row.chunk_id,
                    row.country,
                    row.source_language,
                    row.law_title,
                    row.article_number,
                    row.section_title,
                    row.violence_category,
                    row.source_url,
                    int(row.official_source),
                    row.text,
                    json.dumps(row.matched_keywords, ensure_ascii=False),
                )
                for row in metadata_rows
            ],
        )
        connection.commit()


def _read_metadata_db(metadata_db_path: Path) -> list[LegalMetadataRow]:
    with sqlite3.connect(metadata_db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                vector_row,
                chunk_id,
                country,
                source_language,
                law_title,
                article_number,
                section_title,
                violence_category,
                source_url,
                official_source,
                text,
                matched_keywords
            FROM legal_chunk_metadata
            ORDER BY vector_row ASC
            """
        ).fetchall()

    return [
        LegalMetadataRow(
            vector_row=int(row["vector_row"]),
            chunk_id=row["chunk_id"],
            country=row["country"],
            source_language=row["source_language"],
            law_title=row["law_title"],
            article_number=row["article_number"],
            section_title=row["section_title"],
            violence_category=row["violence_category"],
            source_url=row["source_url"],
            official_source=bool(row["official_source"]),
            text=row["text"],
            matched_keywords=json.loads(row["matched_keywords"]),
        )
        for row in rows
    ]
