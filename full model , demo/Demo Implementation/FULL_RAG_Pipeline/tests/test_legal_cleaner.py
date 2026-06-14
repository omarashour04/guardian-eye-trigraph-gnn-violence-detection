from rag_service.legal_cleaner import (
    NO_RELEVANT_CONTENT,
    SUCCESS,
    clean_legal_document,
)
from rag_service.legal_scraper import RawLegalDocument


def _raw_document(raw_text, **overrides):
    values = {
        "country": "UK",
        "source_url": "https://example.test/law",
        "law_title": "Mock Legal Law",
        "source_language": "en",
        "source_type": "html",
        "official_source": True,
        "raw_text": raw_text,
        "extraction_method": "html_mock",
        "extraction_status": "success",
        "error_message": None,
    }
    values.update(overrides)
    return RawLegalDocument(**values)


def test_english_violence_text_is_retained():
    document = _raw_document(
        """
        Article 12 Assault
        A person who commits assault or causes bodily harm is liable under this Act.

        Article 13 Records
        The registrar may maintain administrative records.
        """
    )

    cleaned = clean_legal_document(document)

    assert len(cleaned) == 1
    assert cleaned[0].cleaning_status == SUCCESS
    assert "Article 12 Assault" in cleaned[0].cleaned_text
    assert "bodily harm" in cleaned[0].cleaned_text
    assert cleaned[0].violence_category in {"assault", "battery_or_physical_harm"}
    assert cleaned[0].matched_keywords


def test_arabic_violence_text_is_retained():
    document = _raw_document(
        """
        المادة 1
        يعاقب كل من يرتكب اعتداء أو إيذاء جسدي ضد شخص آخر.

        المادة 2
        تحفظ السجلات الإدارية لدى الجهة المختصة.
        """,
        country="UAE",
        source_language="ar",
    )

    cleaned = clean_legal_document(document)

    assert len(cleaned) == 1
    assert cleaned[0].cleaning_status == SUCCESS
    assert "المادة 1" in cleaned[0].cleaned_text
    assert "اعتداء" in cleaned[0].cleaned_text
    assert cleaned[0].matched_keywords


def test_non_violence_unrelated_legal_text_is_filtered_out():
    document = _raw_document(
        """
        Article 20 Filing Fee
        The applicant shall pay the prescribed filing fee before registration.
        """
    )

    cleaned = clean_legal_document(document)

    assert len(cleaned) == 1
    assert cleaned[0].cleaning_status == NO_RELEVANT_CONTENT
    assert cleaned[0].cleaned_text == ""
    assert cleaned[0].matched_keywords == []


def test_metadata_from_raw_legal_document_is_preserved():
    document = _raw_document(
        "Article 9 A person shall not threaten another person with a weapon.",
        country="Canada",
        source_url="https://example.test/canada",
        law_title="Canadian Mock Law",
        source_language="en",
        source_type="pdf",
        official_source=False,
    )

    cleaned = clean_legal_document(document)[0]

    assert cleaned.country == "Canada"
    assert cleaned.source_url == "https://example.test/canada"
    assert cleaned.law_title == "Canadian Mock Law"
    assert cleaned.source_language == "en"
    assert cleaned.source_type == "pdf"
    assert cleaned.official_source is False


def test_excessive_whitespace_is_normalized():
    document = _raw_document(
        """
        Official Header
        Article   5     Assault
        A     person      who     commits     assault     is liable.
        Official Header
        Official Header
        """
    )

    cleaned = clean_legal_document(document)[0]

    assert "Article 5 Assault" in cleaned.cleaned_text
    assert "commits assault is liable" in cleaned.cleaned_text
    assert "Official Header" not in cleaned.cleaned_text
    assert "     " not in cleaned.cleaned_text


def test_detected_actions_is_not_used_or_required():
    document = {
        "country": "UK",
        "source_url": "https://example.test/law",
        "law_title": "Mock Legal Law",
        "source_language": "en",
        "source_type": "html",
        "official_source": True,
        "raw_text": "Section 1 Assault A person who commits assault is liable.",
        "extraction_method": "html_mock",
        "extraction_status": "success",
        "error_message": None,
    }

    cleaned = clean_legal_document(document)

    assert cleaned[0].cleaning_status == SUCCESS
    assert "detected_actions" not in cleaned[0].to_dict()


def test_empty_or_non_relevant_text_returns_safe_no_relevant_content_result():
    document = _raw_document("")

    cleaned = clean_legal_document(document)

    assert len(cleaned) == 1
    assert cleaned[0].cleaning_status == NO_RELEVANT_CONTENT
    assert cleaned[0].cleaned_text == ""
    assert cleaned[0].violence_category is None
    assert cleaned[0].error_message is None
