from rag_service.legal_chunker import LegalChunk, chunk_cleaned_section
from rag_service.legal_cleaner import CleanedLegalSection


def _cleaned_section(cleaned_text, **overrides):
    values = {
        "country": "UK",
        "source_url": "https://example.test/law",
        "law_title": "Mock Legal Law",
        "source_language": "en",
        "source_type": "html",
        "official_source": True,
        "cleaned_text": cleaned_text,
        "matched_keywords": ["assault"],
        "violence_category": "assault",
        "cleaning_status": "success",
        "error_message": None,
    }
    values.update(overrides)
    return CleanedLegalSection(**values)


def test_english_article_and_section_chunks_are_detected():
    section = _cleaned_section(
        """
        Article 240 Assault
        An assault is an unlawful attempt with present ability.

        Section 265 Assault
        A person commits assault by applying force intentionally.

        s. 4 Threats
        A person must not threaten another person.
        """
    )

    chunks = chunk_cleaned_section(section)

    assert [chunk.article_number for chunk in chunks] == [
        "Article 240",
        "Section 265",
        "s. 4",
    ]
    assert chunks[0].section_title == "Assault"
    assert chunks[2].section_title == "Threats"
    assert all(isinstance(chunk, LegalChunk) for chunk in chunks)


def test_arabic_article_chunks_are_detected():
    arabic_text = (
        "\u0627\u0644\u0645\u0627\u062f\u0629 1\n"
        "\u064a\u0639\u0627\u0642\u0628 \u0643\u0644 \u0645\u0646 \u064a\u0631\u062a\u0643\u0628 \u0627\u0639\u062a\u062f\u0627\u0621.\n\n"
        "\u0645\u0627\u062f\u0629 \u0662\n"
        "\u064a\u062d\u0638\u0631 \u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0633\u0644\u0627\u062d."
    )
    section = _cleaned_section(
        arabic_text,
        country="UAE",
        source_language="ar",
        matched_keywords=["\u0627\u0639\u062a\u062f\u0627\u0621"],
        violence_category="assault",
    )

    chunks = chunk_cleaned_section(section)

    assert [chunk.article_number for chunk in chunks] == [
        "\u0627\u0644\u0645\u0627\u062f\u0629 1",
        "\u0645\u0627\u062f\u0629 \u0662",
    ]
    assert "\u0627\u0639\u062a\u062f\u0627\u0621" in chunks[0].text
    assert chunks[0].source_language == "ar"


def test_metadata_is_preserved():
    section = _cleaned_section(
        "Article 9 Threats\nA person shall not threaten another person with a weapon.",
        country="Canada",
        source_url="https://example.test/canada",
        law_title="Canadian Mock Law",
        source_language="en",
        source_type="pdf",
        official_source=False,
        matched_keywords=["threaten", "weapon"],
        violence_category="weapon_or_dangerous_object",
    )

    chunk = chunk_cleaned_section(section)[0]

    assert chunk.country == "Canada"
    assert chunk.source_language == "en"
    assert chunk.law_title == "Canadian Mock Law"
    assert chunk.source_url == "https://example.test/canada"
    assert chunk.official_source is False
    assert chunk.violence_category == "weapon_or_dangerous_object"
    assert chunk.matched_keywords == ["threaten", "weapon"]


def test_fallback_paragraph_chunking_works_without_article_markers():
    section = _cleaned_section(
        """
        A person who commits assault may be liable under this law.

        A person who uses a weapon during violent conduct may face further consequences.
        """
    )

    chunks = chunk_cleaned_section(section, max_chars=70)

    assert len(chunks) == 2
    assert chunks[0].article_number is None
    assert "commits assault" in chunks[0].text
    assert "uses a weapon" in chunks[1].text


def test_chunk_ids_are_deterministic():
    section = _cleaned_section(
        "Article 12 Assault\nA person who commits assault is liable."
    )

    first = chunk_cleaned_section(section)
    second = chunk_cleaned_section(section)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_detected_actions_field_is_not_required_or_used():
    section = {
        "country": "UK",
        "source_url": "https://example.test/law",
        "law_title": "Mock Legal Law",
        "source_language": "en",
        "source_type": "html",
        "official_source": True,
        "cleaned_text": "Section 1 Assault\nA person who commits assault is liable.",
        "matched_keywords": ["assault"],
        "violence_category": "assault",
        "cleaning_status": "success",
        "error_message": None,
    }

    chunk = chunk_cleaned_section(section)[0]

    assert chunk.article_number == "Section 1"
    assert "detected_actions" not in chunk.to_dict()
