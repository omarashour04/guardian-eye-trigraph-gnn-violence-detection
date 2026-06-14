from rag_service.legal_guardrails import validate_legal_consequences_output
from rag_service.legal_summarizer import summarize_legal_consequences


def _safe_input(**overrides):
    values = {
        "country": "UAE",
        "verdict": "violence",
        "confidence": 0.9,
        "packet_summary": "The packet describes violent conduct.",
        "narrative": "The report describes a confrontation.",
        "weapon_flag": False,
        "weapon_class": None,
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
    return values


_DEFAULT_LIMITATIONS_NOTE = (
    "This is not legal advice and does not determine guilt or predict court outcome."
)


def _payload(summary=None, references=None, limitations_note=None):
    return {
        "legal_consequences": {
            "country": "UAE",
            "query_basis": {
                "verdict": "violence",
                "weapon_flag": False,
                "weapon_class": None,
            },
            "retrieved_legal_references": [_reference()] if references is None else references,
            "summary": summary
            or (
                "According to the retrieved regulation, the reported conduct may be subject to "
                "legal consequences. Possible legal consequences include penalties or protective "
                "measures depending on judicial determination."
            ),
            "guardrail_status": "passed",
            "limitations_note": (
                _DEFAULT_LIMITATIONS_NOTE if limitations_note is None else limitations_note
            ),
        }
    }


def _codes(result):
    return {issue.code for issue in result.issues}


def test_passing_cautious_summary_passes():
    output = summarize_legal_consequences(_safe_input(), [_reference()])

    result = validate_legal_consequences_output(output, legal_input=_safe_input())

    assert result.status == "passed"
    assert result.issues == []
    assert result.recommended_action == "pass"


def test_guilt_declaration_fails():
    result = validate_legal_consequences_output(
        _payload(
            "According to the retrieved regulation, the person is guilty and may be subject to "
            "penalties. Possible legal consequences include penalties depending on judicial determination."
        ),
        legal_input=_safe_input(),
    )

    assert result.status == "failed"
    assert "guilt_declaration" in _codes(result)
    assert result.corrected_summary


def test_guaranteed_punishment_wording_fails():
    result = validate_legal_consequences_output(
        _payload(
            "According to the retrieved regulation, punishment is guaranteed and the person will be punished. "
            "Possible legal consequences include penalties depending on judicial determination."
        ),
        legal_input=_safe_input(),
    )

    assert "guaranteed_punishment" in _codes(result)
    assert result.recommended_action == "regenerate"


def test_legal_advice_wording_fails():
    result = validate_legal_consequences_output(
        _payload(
            "According to the retrieved regulation, you should file a complaint and may be subject to "
            "procedures. Possible legal consequences include review depending on judicial determination."
        ),
        legal_input=_safe_input(),
    )

    assert "legal_advice" in _codes(result)


def test_court_prediction_wording_fails():
    result = validate_legal_consequences_output(
        _payload(
            "According to the retrieved regulation, the court will convict the person and may be subject to "
            "penalties. Possible legal consequences include penalties depending on judicial determination."
        ),
        legal_input=_safe_input(),
    )

    assert "court_outcome_prediction" in _codes(result)


def test_unsupported_punch_slap_kick_fails_when_absent_from_narrative_and_references():
    result = validate_legal_consequences_output(
        _payload(
            "According to the retrieved regulation, the reported person punched and kicked someone and "
            "may be subject to review. Possible legal consequences include penalties depending on judicial determination."
        ),
        legal_input=_safe_input(),
    )

    assert "unsupported_exact_action_claim" in _codes(result)


def test_supported_action_words_allowed_when_present_in_narrative_or_references():
    result = validate_legal_consequences_output(
        _payload(
            "According to the retrieved regulation, the narrative says punched conduct may be subject to "
            "review. Possible legal consequences include penalties depending on judicial determination."
        ),
        legal_input=_safe_input(narrative="The report says punched conduct was alleged."),
    )

    assert "unsupported_exact_action_claim" not in _codes(result)
    assert result.status == "passed"


def test_missing_limitations_note_fails():
    result = validate_legal_consequences_output(
        _payload(limitations_note=""),
        legal_input=_safe_input(),
    )

    assert "missing_limitations_note" in _codes(result)


def test_missing_retrieved_references_fails_unless_fallback_mode():
    no_refs = _payload(references=[])

    failed = validate_legal_consequences_output(no_refs, legal_input=_safe_input())
    fallback = validate_legal_consequences_output(
        no_refs,
        legal_input=_safe_input(),
        fallback_mode=True,
    )

    assert "missing_retrieved_references" in _codes(failed)
    assert failed.recommended_action == "fallback"
    assert "missing_retrieved_references" not in _codes(fallback)


def test_summary_must_not_use_detected_actions():
    payload = _payload()
    payload["legal_consequences"]["detected_actions"] = ["punch"]

    result = validate_legal_consequences_output(payload, legal_input=_safe_input())

    assert "detected_actions_used" in _codes(result)
