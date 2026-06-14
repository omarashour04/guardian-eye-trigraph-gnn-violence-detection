"""Deterministic retrieval over the Legal Consequences FAISS + SQLite index.

This module retrieves and ranks legal chunk metadata. It does not summarize,
call an LLM, run guardrails, evaluate, or orchestrate a demo flow.
"""

from __future__ import annotations

import re
import logging
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import faiss
import numpy as np

from rag_service.legal_index import (
    EmbeddingProvider,
    LegalMetadataRow,
    LegalVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from rag_service.schemas import LegalConsequencesInput


DEFAULT_TOP_K = 5
logger = logging.getLogger("guardian_eye.legal_retrieval")


@dataclass(frozen=True)
class RankedLegalReference:
    law_title: str
    article_number: str | None
    section_title: str | None
    source_url: str
    snippet: str
    score: float
    country: str
    violence_category: str | None
    official_source: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LegalRetrievalResult:
    query: str
    country: str
    references: list[RankedLegalReference]
    fallback_used: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "country": self.country,
            "references": [reference.to_dict() for reference in self.references],
            "fallback_used": self.fallback_used,
            "status": self.status,
        }


def build_legal_retrieval_query(input_data: LegalConsequencesInput | dict[str, Any]) -> str:
    """Build a retrieval query from the safe Legal RAG input fields."""

    data = _coerce_input(input_data)
    parts = [
        data.get("verdict", ""),
        data.get("packet_summary", ""),
        data.get("narrative", ""),
    ]
    if data.get("weapon_flag"):
        parts.append("weapon dangerous object")
        if data.get("weapon_class"):
            parts.append(str(data["weapon_class"]))
    return _normalize_text(" ".join(part for part in parts if part))


def retrieve_legal_references(
    input_data: LegalConsequencesInput | dict[str, Any],
    legal_index: LegalVectorIndex,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int | None = None,
) -> LegalRetrievalResult:
    """Retrieve and deterministically rerank legal references for one query."""

    data = _coerce_input(input_data)
    query = build_legal_retrieval_query(data)
    selected_country = str(data.get("country", "")).strip()

    if not query or legal_index.index_count() == 0 or legal_index.metadata_count() == 0:
        return LegalRetrievalResult(
            query=query,
            country=selected_country,
            references=[],
            fallback_used=False,
            status="empty",
        )

    provider = embedding_provider or SentenceTransformerEmbeddingProvider()
    query_vector = _embed_query(query, provider)
    # Country is a hard jurisdiction boundary. Search all rows so filtering is
    # complete, then rank only the selected country's rows.
    pool_size = legal_index.index_count()
    similarities, vector_rows = legal_index.faiss_index.search(query_vector, pool_size)

    candidates = _collect_candidates(
        legal_index,
        vector_rows=vector_rows[0],
        similarities=similarities[0],
    )
    if not candidates:
        return LegalRetrievalResult(
            query=query,
            country=selected_country,
            references=[],
            fallback_used=False,
            status="empty",
        )

    country_matches = [
        candidate
        for candidate in candidates
        if _same_country(candidate["metadata"].country, selected_country)
    ]
    _log_retrieval_filter(
        requested_country=selected_country,
        candidates=candidates,
        country_matches=country_matches,
    )
    if not country_matches:
        return LegalRetrievalResult(
            query=query,
            country=selected_country,
            references=[],
            fallback_used=True,
            status="no_jurisdiction_references",
        )

    context_terms = _context_terms(data, query)
    scored = [
        (
            _combined_score(
                similarity=candidate["similarity"],
                metadata=candidate["metadata"],
                selected_country=selected_country,
                context_terms=context_terms,
                fallback_used=False,
            ),
            candidate["metadata"],
        )
        for candidate in country_matches
    ]
    scored.sort(key=lambda item: (-item[0], item[1].vector_row, item[1].chunk_id))

    references = [
        _reference_from_metadata(metadata, score)
        for score, metadata in scored[: max(0, top_k)]
        if score > 0
    ]
    return LegalRetrievalResult(
        query=query,
        country=selected_country,
        references=references,
        fallback_used=False,
        status="ok" if references else "empty",
    )


def _coerce_input(input_data: LegalConsequencesInput | dict[str, Any]) -> dict[str, Any]:
    if isinstance(input_data, dict):
        return dict(input_data)
    if hasattr(input_data, "model_dump"):
        return input_data.model_dump()
    if is_dataclass(input_data):
        return asdict(input_data)
    raise TypeError("Legal retrieval expects LegalConsequencesInput or dict input.")


def _embed_query(query: str, provider: EmbeddingProvider) -> np.ndarray:
    vector = np.asarray(provider.embed_texts([f"query: {query}"]), dtype=np.float32)
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError("Query embedding provider must return one 2D embedding row.")
    vector = np.ascontiguousarray(vector)
    faiss.normalize_L2(vector)
    return vector


def _collect_candidates(
    legal_index: LegalVectorIndex,
    *,
    vector_rows: np.ndarray,
    similarities: np.ndarray,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    for vector_row, similarity in zip(vector_rows, similarities):
        row_id = int(vector_row)
        if row_id < 0 or row_id in seen_rows:
            continue
        metadata = legal_index.get_metadata_by_vector_row(row_id)
        if metadata is None:
            continue
        seen_rows.add(row_id)
        candidates.append({"metadata": metadata, "similarity": float(similarity)})
    return candidates


def _combined_score(
    *,
    similarity: float,
    metadata: LegalMetadataRow,
    selected_country: str,
    context_terms: set[str],
    fallback_used: bool,
) -> float:
    similarity_score = _normalize_similarity(similarity)
    country_score = 1.0 if _same_country(metadata.country, selected_country) else 0.0
    keyword_score = _keyword_overlap_score(context_terms, metadata)
    official_score = 1.0 if metadata.official_source else 0.0
    article_score = 1.0 if metadata.article_number else 0.0

    score = (
        0.45 * similarity_score
        + 0.25 * keyword_score
        + 0.12 * country_score
        + 0.10 * official_score
        + 0.08 * article_score
    )
    if fallback_used:
        score *= 0.55
    return round(max(0.0, min(score, 1.0)), 6)


def _normalize_similarity(similarity: float) -> float:
    if similarity < -1.0:
        return 0.0
    if similarity > 1.0:
        return 1.0
    return (similarity + 1.0) / 2.0


def _keyword_overlap_score(context_terms: set[str], metadata: LegalMetadataRow) -> float:
    if not context_terms:
        return 0.0

    searchable = " ".join(
        [
            metadata.text,
            metadata.law_title,
            metadata.section_title or "",
            metadata.article_number or "",
            metadata.violence_category or "",
            " ".join(metadata.matched_keywords),
        ]
    ).casefold()
    matches = {term for term in context_terms if term in searchable}
    denominator = min(max(len(context_terms), 1), 8)
    return min(1.0, len(matches) / denominator)


def _context_terms(data: dict[str, Any], query: str) -> set[str]:
    terms = set(_significant_terms(query))
    verdict = str(data.get("verdict", "")).casefold()
    if verdict:
        terms.add(verdict)
    if "violence" in verdict or "violent" in query.casefold():
        terms.update({"violence", "violent", "assault", "harm"})

    if data.get("weapon_flag"):
        terms.update({"weapon", "dangerous object", "weapon_or_dangerous_object"})
        weapon_class = data.get("weapon_class")
        if weapon_class:
            terms.add(str(weapon_class).casefold())
    return {term for term in terms if len(term) >= 3}


def _significant_terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "person",
        "people",
        "may",
        "under",
        "legal",
        "conduct",
    }
    words = re.findall(r"[\w\u0600-\u06FF]+", text.casefold())
    return {word for word in words if len(word) >= 4 and word not in stopwords}


def _same_country(candidate_country: str, selected_country: str) -> bool:
    return _canonical_country(candidate_country) == _canonical_country(selected_country)


def _canonical_country(country: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", country.casefold()).strip()
    aliases = {
        "uae": "uae",
        "united arab emirates": "uae",
        "uk": "uk",
        "united kingdom": "uk",
        "great britain": "uk",
        "britain": "uk",
        "england": "uk",
        "canada": "canada",
        "ksa": "ksa",
        "saudi arabia": "ksa",
        "kingdom of saudi arabia": "ksa",
        "egypt": "egypt",
        "usa california": "usa california",
        "us california": "usa california",
        "united states california": "usa california",
        "california": "usa california",
    }
    return aliases.get(normalized, normalized)


def _log_retrieval_filter(
    *,
    requested_country: str,
    candidates: list[dict[str, Any]],
    country_matches: list[dict[str, Any]],
) -> None:
    retrieved_countries = sorted(
        {
            candidate["metadata"].country
            for candidate in country_matches
        }
    )
    candidate_countries = sorted(
        {
            candidate["metadata"].country
            for candidate in candidates
        }
    )
    logger.info(
        "Legal retrieval country filter requested_country=%s "
        "retrieved_countries=%s number_of_chunks_before_filter=%s "
        "number_after_filter=%s candidate_countries=%s",
        requested_country,
        retrieved_countries,
        len(candidates),
        len(country_matches),
        candidate_countries,
    )


def _reference_from_metadata(
    metadata: LegalMetadataRow,
    score: float,
) -> RankedLegalReference:
    return RankedLegalReference(
        law_title=metadata.law_title,
        article_number=metadata.article_number,
        section_title=metadata.section_title,
        source_url=metadata.source_url,
        snippet=_snippet(metadata.text),
        score=score,
        country=metadata.country,
        violence_category=metadata.violence_category,
        official_source=metadata.official_source,
    )


def _snippet(text: str, max_chars: int = 420) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
