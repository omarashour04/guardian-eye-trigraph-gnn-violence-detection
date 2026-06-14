import numpy as np

from rag_service.legal_chunker import LegalChunk
from rag_service.legal_index import build_legal_index
from rag_service.legal_retrieval import (
    RankedLegalReference,
    build_legal_retrieval_query,
    retrieve_legal_references,
)


class KeywordEmbeddingProvider:
    vocabulary = ["assault", "weapon", "bottle", "threat", "family", "fee"]

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append([float(lowered.count(term)) for term in self.vocabulary])
        return np.asarray(vectors, dtype=np.float32)


def _chunk(
    chunk_id,
    *,
    country="UAE",
    law_title="Mock UAE Law",
    article_number="Article 1",
    section_title="Assault",
    violence_category="assault",
    source_url="https://example.test/uae",
    official_source=True,
    text="Article 1 Assault A person who commits assault is liable.",
    matched_keywords=None,
):
    return LegalChunk(
        chunk_id=chunk_id,
        country=country,
        source_language="en",
        law_title=law_title,
        article_number=article_number,
        section_title=section_title,
        violence_category=violence_category,
        source_url=source_url,
        official_source=official_source,
        text=text,
        matched_keywords=matched_keywords or ["assault"],
    )


def _safe_input(**overrides):
    values = {
        "country": "UAE",
        "verdict": "violence",
        "confidence": 0.91,
        "packet_summary": "The incident includes assault and physical harm.",
        "narrative": "A person was attacked during the incident.",
        "weapon_flag": False,
        "weapon_class": None,
        "language": "en",
    }
    values.update(overrides)
    return values


def _index(chunks):
    return build_legal_index(chunks, embedding_provider=KeywordEmbeddingProvider())


def test_retrieves_only_selected_country_chunks_when_available():
    legal_index = _index(
        [
            _chunk("uae-1", country="UAE", law_title="UAE Assault Law"),
            _chunk("canada-1", country="Canada", law_title="Canada Assault Law"),
        ]
    )

    result = retrieve_legal_references(
        _safe_input(country="UAE"),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=5,
    )

    assert result.status == "ok"
    assert result.references
    assert {reference.country for reference in result.references} == {"UAE"}


def test_uk_query_returns_uk_only_references():
    legal_index = _index(
        [
            _chunk("uk-1", country="UK", law_title="UK Assault Law"),
            _chunk("uae-1", country="UAE", law_title="UAE Assault Law"),
            _chunk("canada-1", country="Canada", law_title="Canada Assault Law"),
        ]
    )

    result = retrieve_legal_references(
        _safe_input(country="UK"),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=5,
    )

    assert result.status == "ok"
    assert result.references
    assert {reference.country for reference in result.references} == {"UK"}


def test_canada_query_returns_canada_only_references():
    legal_index = _index(
        [
            _chunk("uk-1", country="UK", law_title="UK Assault Law"),
            _chunk("uae-1", country="UAE", law_title="UAE Assault Law"),
            _chunk("canada-1", country="Canada", law_title="Canada Assault Law"),
        ]
    )

    result = retrieve_legal_references(
        _safe_input(country="Canada"),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=5,
    )

    assert result.status == "ok"
    assert result.references
    assert {reference.country for reference in result.references} == {"Canada"}


def test_country_filter_is_hard_even_when_other_countries_score_higher():
    legal_index = _index(
        [
            _chunk(
                "uae-low",
                country="UAE",
                law_title="UAE General Law",
                text="Article 1 Public safety provision.",
                matched_keywords=[],
                violence_category="general_violence",
            ),
            _chunk(
                "uk-high",
                country="UK",
                law_title="UK High Match Assault Law",
                text="Article 2 Assault assault weapon bottle violence harm threat.",
                matched_keywords=["assault", "weapon", "bottle", "violence", "harm"],
            ),
            _chunk(
                "canada-high",
                country="Canada",
                law_title="Canada High Match Assault Law",
                text="Article 3 Assault assault weapon bottle violence harm threat.",
                matched_keywords=["assault", "weapon", "bottle", "violence", "harm"],
            ),
        ]
    )

    result = retrieve_legal_references(
        _safe_input(
            country="UAE",
            weapon_flag=True,
            weapon_class="bottle",
            packet_summary="assault weapon bottle violence harm threat",
            narrative="assault weapon bottle violence harm threat",
        ),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=5,
    )

    assert result.references
    assert {reference.country for reference in result.references} == {"UAE"}
    assert all(reference.country != "UK" for reference in result.references)
    assert all(reference.country != "Canada" for reference in result.references)


def test_ranking_prefers_higher_keyword_overlap():
    legal_index = _index(
        [
            _chunk(
                "general",
                text="Article 1 Violence General violent conduct is prohibited.",
                matched_keywords=["violence"],
                violence_category="general_violence",
            ),
            _chunk(
                "specific",
                article_number="Article 2",
                text="Article 2 Assault Assault causing physical harm is prohibited.",
                matched_keywords=["assault", "harm"],
                violence_category="battery_or_physical_harm",
            ),
        ]
    )

    result = retrieve_legal_references(
        _safe_input(packet_summary="assault physical harm", narrative="assault harm"),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=2,
    )

    assert result.references[0].article_number == "Article 2"


def test_official_sources_receive_scoring_boost():
    legal_index = _index(
        [
            _chunk("unofficial", official_source=False, law_title="Unofficial Law"),
            _chunk("official", official_source=True, law_title="Official Law"),
        ]
    )

    result = retrieve_legal_references(
        _safe_input(),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=2,
    )

    assert result.references[0].law_title == "Official Law"
    assert result.references[0].score > result.references[1].score


def test_article_level_chunks_receive_scoring_boost():
    legal_index = _index(
        [
            _chunk("paragraph", article_number=None, section_title=None, law_title="Paragraph Law"),
            _chunk("article", article_number="Article 4", law_title="Article Law"),
        ]
    )

    result = retrieve_legal_references(
        _safe_input(),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=2,
    )

    assert result.references[0].law_title == "Article Law"
    assert result.references[0].score > result.references[1].score


def test_weapon_context_affects_ranking_when_weapon_flag_true():
    legal_index = _index(
        [
            _chunk(
                "assault",
                law_title="Assault Law",
                text="Article 1 Assault A person who commits assault is liable.",
                matched_keywords=["assault"],
                violence_category="assault",
            ),
            _chunk(
                "weapon",
                law_title="Weapon Law",
                article_number="Article 2",
                text="Article 2 Weapon Using a bottle or weapon during violence is prohibited.",
                matched_keywords=["weapon", "bottle"],
                violence_category="weapon_or_dangerous_object",
            ),
        ]
    )

    result = retrieve_legal_references(
        _safe_input(
            weapon_flag=True,
            weapon_class="bottle",
            packet_summary="A bottle was used during the violence.",
            narrative="The incident involved a weapon.",
        ),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=2,
    )

    assert result.references[0].law_title == "Weapon Law"


def test_detected_actions_is_not_required_or_used():
    legal_index = _index([_chunk("uae-1")])
    input_data = _safe_input()

    query = build_legal_retrieval_query(input_data)
    result = retrieve_legal_references(
        input_data,
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
    )

    assert "detected_actions" not in query
    assert result.references
    assert "detected_actions" not in result.references[0].to_dict()


def test_empty_or_no_match_retrieval_returns_safe_empty_jurisdiction_result():
    legal_index = _index([_chunk("uk-1", country="UK", law_title="UK Assault Law")])

    result = retrieve_legal_references(
        _safe_input(country="Egypt"),
        legal_index,
        embedding_provider=KeywordEmbeddingProvider(),
        top_k=1,
    )

    assert result.status == "no_jurisdiction_references"
    assert result.fallback_used is True
    assert result.references == []
