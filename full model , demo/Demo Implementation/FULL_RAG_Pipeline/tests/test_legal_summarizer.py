from rag_service.legal_retrieval import RankedLegalReference
from rag_service.legal_summarizer import summarize_legal_consequences


class MockSummaryProvider:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def summarize(self, *, input_data, references, legal_context):
        self.calls.append(
            {
                "input_data": input_data,
                "references": references,
                "legal_context": legal_context,
            }
        )
        return self.text


def _safe_input(**overrides):
    values = {
        "country": "UAE",
        "verdict": "violence",
        "confidence": 0.88,
        "packet_summary": "The packet describes violent conduct.",
        "narrative": "The report describes a confrontation.",
        "weapon_flag": True,
        "weapon_class": "bottle",
        "language": "en",
    }
    values.update(overrides)
    return values


def _reference(**overrides):
    values = {
        "law_title": "UAE Crimes and Penalties Law",
        "article_number": "Article 1",
        "section_title": "Assault",
        "source_url": "https://uaelegislation.gov.ae/ar/legislations/1529",
        "snippet": "Retrieved text addresses assault and possible penalties.",
        "score": 0.87,
        "country": "UAE",
        "violence_category": "assault",
        "official_source": True,
    }
    values.update(overrides)
    return RankedLegalReference(**values)


def _summary_text():
    return (
        "According to the retrieved regulation, this reported violent conduct "
        "may be subject to statutory consequences. Possible legal consequences "
        "include penalties or protective measures depending on judicial determination."
    )


def test_summary_includes_cautious_legal_wording():
    output = summarize_legal_consequences(_safe_input(), [_reference()])
    summary = output.legal_consequences.summary.casefold()

    assert "according to the retrieved regulation" in summary
    assert "may be subject to" in summary
    assert "possible legal consequences include" in summary
    assert "depending on judicial determination" in summary


def test_summary_includes_retrieved_law_article_and_source_references():
    output = summarize_legal_consequences(_safe_input(), [_reference()])
    payload = output.legal_consequences

    assert payload.retrieved_legal_references[0].law_title == "UAE Crimes and Penalties Law"
    assert payload.retrieved_legal_references[0].article_number == "Article 1"
    assert payload.retrieved_legal_references[0].source_url.startswith("https://")
    assert "UAE Crimes and Penalties Law" in payload.summary
    assert "Article 1" in payload.summary


def test_no_guilt_declaration():
    provider = MockSummaryProvider(
        "According to the retrieved regulation, the person is guilty and may be subject to penalties "
        "depending on judicial determination. Possible legal consequences include review."
    )

    output = summarize_legal_consequences(
        _safe_input(),
        [_reference()],
        summarization_provider=provider,
    )

    assert "is guilty" not in output.legal_consequences.summary.casefold()


def test_no_guaranteed_punishment_wording():
    provider = MockSummaryProvider(
        "According to the retrieved regulation, punishment is guaranteed and the person will be punished. "
        "Possible legal consequences include penalties depending on judicial determination."
    )

    output = summarize_legal_consequences(
        _safe_input(),
        [_reference()],
        summarization_provider=provider,
    )
    summary = output.legal_consequences.summary.casefold()

    assert "punishment is guaranteed" not in summary
    assert "will be punished" not in summary


def test_no_legal_advice_wording_in_summary():
    output = summarize_legal_consequences(_safe_input(), [_reference()])
    summary = output.legal_consequences.summary.casefold()

    assert "you should" not in summary
    assert "my advice" not in summary
    assert "i advise" not in summary


def test_no_unsupported_action_claims_are_invented():
    provider = MockSummaryProvider(
        "According to the retrieved regulation, the person punched, slapped, and kicked someone. "
        "The conduct may be subject to review depending on judicial determination. "
        "Possible legal consequences include penalties."
    )

    output = summarize_legal_consequences(
        _safe_input(),
        [_reference()],
        summarization_provider=provider,
    )
    summary = output.legal_consequences.summary.casefold()

    assert "punched" not in summary
    assert "slapped" not in summary
    assert "kicked" not in summary
    assert "detected_actions" not in output.model_dump_json()


def test_empty_retrieval_returns_safe_fallback():
    output = summarize_legal_consequences(_safe_input(), [])
    payload = output.legal_consequences

    assert payload.guardrail_status == "needs_review"
    assert payload.retrieved_legal_references == []
    assert "not enough retrieved legal references" in payload.summary.casefold()


def test_arabic_mode_does_not_translate_but_is_translate_wrapper_compatible_later():
    output = summarize_legal_consequences(
        _safe_input(language="ar"),
        [_reference()],
    )
    payload = output.legal_consequences

    assert payload.summary.startswith("According to the retrieved regulation")
    assert "TranslateGemma wrapper can translate" in payload.limitations_note
    assert payload.country == "UAE"


def test_provider_receives_retrieved_context_only():
    provider = MockSummaryProvider(_summary_text())

    summarize_legal_consequences(
        _safe_input(),
        [_reference()],
        summarization_provider=provider,
    )

    assert provider.calls
    assert "Retrieved text addresses assault" in provider.calls[0]["legal_context"]
    assert "packet describes" not in provider.calls[0]["legal_context"].casefold()
