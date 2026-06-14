from rag_service.legal_orchestrator import generate_legal_consequences
from rag_service.legal_retrieval import RankedLegalReference


def _safe_input(**overrides):
    values = {
        "country": "UAE",
        "verdict": "violence",
        "confidence": 0.94,
        "packet_summary": "The packet describes violent conduct involving a bottle.",
        "narrative": "The narrative describes a confrontation involving a bottle.",
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
        "snippet": "Retrieved text addresses assault, weapons, bottle use, and possible penalties.",
        "score": 0.91,
        "country": "UAE",
        "violence_category": "weapon_or_dangerous_object",
        "official_source": True,
    }
    values.update(overrides)
    return RankedLegalReference(**values)


class MockRetriever:
    def __init__(self, references):
        self.references = references
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.references


class UnsafeSummaryProvider:
    def summarize(self, *, input_data, references, legal_context):
        return (
            "According to the retrieved regulation, the person is guilty and will be punished. "
            "Possible legal consequences include punishment depending on judicial determination."
        )


def _assert_legal_consequences_shape(payload):
    assert payload["country"]
    assert payload["query_basis"]
    assert "retrieved_legal_references" in payload
    assert payload["summary"]
    assert payload["guardrail_status"]
    assert payload["limitations_note"]


def test_mock_e2e_safe_input_to_final_legal_consequences_output():
    result = generate_legal_consequences(
        _safe_input(),
        retriever=MockRetriever([_reference()]),
    )

    payload = result.legal_consequences
    _assert_legal_consequences_shape(payload)
    assert payload["country"] == "UAE"
    assert payload["query_basis"]["verdict"] == "violence"
    assert payload["retrieved_legal_references"][0]["law_title"] == "UAE Crimes and Penalties Law"
    assert "According to the retrieved regulation" in payload["summary"]
    assert payload["guardrail_status"] == "passed"
    assert result.evaluation is not None
    assert "overall_score" in result.evaluation


def test_mock_e2e_weapon_flag_true_path():
    result = generate_legal_consequences(
        _safe_input(weapon_flag=True, weapon_class="bottle"),
        retriever=MockRetriever([_reference()]),
    )

    payload = result.legal_consequences
    assert payload["query_basis"]["weapon_flag"] is True
    assert payload["query_basis"]["weapon_class"] == "bottle"
    assert "bottle" in payload["summary"].casefold()


def test_mock_e2e_no_retrieval_fallback_path():
    result = generate_legal_consequences(
        _safe_input(),
        retriever=MockRetriever([]),
    )

    payload = result.legal_consequences
    _assert_legal_consequences_shape(payload)
    assert payload["retrieved_legal_references"] == []
    assert "not enough" in payload["summary"].casefold()
    assert payload["guardrail_status"] == "needs_review"
    assert result.evaluation is not None


def test_mock_e2e_guardrail_correction_path():
    result = generate_legal_consequences(
        _safe_input(),
        retriever=MockRetriever([_reference()]),
        summarization_provider=UnsafeSummaryProvider(),
    )

    summary = result.legal_consequences["summary"].casefold()
    assert "is guilty" not in summary
    assert "will be punished" not in summary
    assert "according to the retrieved regulation" in summary
    assert result.legal_consequences["guardrail_status"] == "passed"


def test_mock_e2e_arabic_wrapper_compatibility_path():
    result = generate_legal_consequences(
        _safe_input(language="ar"),
        retriever=MockRetriever([_reference()]),
    )

    payload = result.legal_consequences
    assert payload["summary"].startswith("According to the retrieved regulation")
    assert "TranslateGemma wrapper can translate" in payload["limitations_note"]


def test_mock_e2e_does_not_require_or_use_detected_actions():
    result = generate_legal_consequences(
        _safe_input(detected_actions=["punch", "kick"]),
        retriever=MockRetriever([_reference()]),
    )

    assert "detected_actions" not in str(result.legal_consequences)
    assert result.guardrails["status"] == "passed"
    assert "generation:detected_actions_used" not in (result.evaluation or {}).get("issues", [])
