import sqlite3

import numpy as np

from rag_service.legal_chunker import LegalChunk
from rag_service.legal_index import build_legal_index, load_legal_index


class MockEmbeddingProvider:
    def __init__(self, dimensions=4):
        self.dimensions = dimensions
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for index, text in enumerate(texts):
            base = float(index + 1)
            vectors.append(
                [
                    base,
                    float(len(text) % 11),
                    float(text.count("assault")),
                    float(text.count("weapon")),
                ][: self.dimensions]
            )
        return np.asarray(vectors, dtype=np.float32)


def _chunks():
    return [
        LegalChunk(
            chunk_id="legal-001",
            country="UK",
            source_language="en",
            law_title="Mock UK Law",
            article_number="Article 1",
            section_title="Assault",
            violence_category="assault",
            source_url="https://example.test/uk",
            official_source=True,
            text="Article 1 Assault A person who commits assault is liable.",
            matched_keywords=["assault"],
        ),
        LegalChunk(
            chunk_id="legal-002",
            country="Canada",
            source_language="en",
            law_title="Mock Canada Law",
            article_number="Section 2",
            section_title="Weapon",
            violence_category="weapon_or_dangerous_object",
            source_url="https://example.test/canada",
            official_source=False,
            text="Section 2 A person must not use a weapon.",
            matched_keywords=["weapon"],
        ),
    ]


def test_index_builds_from_mocked_legal_chunks():
    provider = MockEmbeddingProvider()

    legal_index = build_legal_index(_chunks(), embedding_provider=provider)

    assert legal_index.index_count() == 2
    assert legal_index.metadata_count() == 2
    assert legal_index.embedding_dimension == 4


def test_mock_embeddings_work_without_downloading_model():
    provider = MockEmbeddingProvider()

    build_legal_index(_chunks(), embedding_provider=provider)

    assert len(provider.calls) == 1
    assert provider.calls[0] == [chunk.text for chunk in _chunks()]


def test_faiss_index_count_matches_number_of_chunks():
    chunks = _chunks()

    legal_index = build_legal_index(chunks, embedding_provider=MockEmbeddingProvider())

    assert legal_index.faiss_index.ntotal == len(chunks)


def test_sqlite_metadata_count_matches_number_of_chunks(tmp_path):
    chunks = _chunks()
    index_path = tmp_path / "legal.faiss"
    metadata_path = tmp_path / "legal.sqlite"
    legal_index = build_legal_index(chunks, embedding_provider=MockEmbeddingProvider())

    legal_index.save(index_path, metadata_path)

    with sqlite3.connect(metadata_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM legal_chunk_metadata").fetchone()[0]
    assert count == len(chunks)


def test_saved_index_can_be_loaded_again(tmp_path):
    index_path = tmp_path / "legal.faiss"
    metadata_path = tmp_path / "legal.sqlite"
    legal_index = build_legal_index(_chunks(), embedding_provider=MockEmbeddingProvider())
    legal_index.save(index_path, metadata_path)

    loaded = load_legal_index(index_path, metadata_path)

    assert loaded.index_count() == legal_index.index_count()
    assert loaded.metadata_count() == legal_index.metadata_count()
    assert loaded.embedding_dimension == legal_index.embedding_dimension


def test_metadata_can_be_fetched_by_vector_row_and_chunk_id(tmp_path):
    index_path = tmp_path / "legal.faiss"
    metadata_path = tmp_path / "legal.sqlite"
    legal_index = build_legal_index(_chunks(), embedding_provider=MockEmbeddingProvider())
    legal_index.save(index_path, metadata_path)
    loaded = load_legal_index(index_path, metadata_path)

    by_row = loaded.get_metadata_by_vector_row(1)
    by_chunk_id = loaded.get_metadata_by_chunk_id("legal-002")

    assert by_row is not None
    assert by_row.chunk_id == "legal-002"
    assert by_chunk_id is not None
    assert by_chunk_id.vector_row == 1


def test_country_and_source_metadata_are_preserved(tmp_path):
    index_path = tmp_path / "legal.faiss"
    metadata_path = tmp_path / "legal.sqlite"
    legal_index = build_legal_index(_chunks(), embedding_provider=MockEmbeddingProvider())
    legal_index.save(index_path, metadata_path)
    loaded = load_legal_index(index_path, metadata_path)

    row = loaded.get_metadata_by_chunk_id("legal-001")

    assert row is not None
    assert row.country == "UK"
    assert row.source_language == "en"
    assert row.law_title == "Mock UK Law"
    assert row.article_number == "Article 1"
    assert row.section_title == "Assault"
    assert row.source_url == "https://example.test/uk"
    assert row.official_source is True
    assert row.matched_keywords == ["assault"]


def test_detected_actions_is_not_required_or_used():
    legal_index = build_legal_index(_chunks(), embedding_provider=MockEmbeddingProvider())
    row = legal_index.get_metadata_by_vector_row(0)

    assert row is not None
    assert "detected_actions" not in row.to_dict()
