from dataclasses import dataclass

from rag_service.legal_guardrails import LegalGuardrailResult
from rag_service.legal_orchestrator import generate_legal_consequences
from rag_service.legal_retrieval import RankedLegalReference


def _safe_input(**overrides):
    values = {
        "country": "UAE",
        "verdict": "violence",
        "confidence": 0.94,
        "packet_summary": "The packet describes violent conduct with a bottle.",
        "narrative": "The narrative describes a confrontation.",
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


def _summary():
    return (
        "According to the retrieved regulation, including UAE Crimes and Penalties Law "
        "Article 1, the reported conduct may be subject to legal consequences. "
        "Possible legal consequences include penalties or protective measures "
        "depending on judicial determination."
    )


class StaticRetriever:
    def __init__(self, references):
        self.references = references
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.references


class StaticSummarizer:
    def __init__(self, summary=None):
        self.summary = summary or _summary()

    def summarize(self, *, input_data, retrieved_references, **kwargs):
        return {
            "legal_consequences": {
                "country": input_data.country,
                "query_basis": {
                    "verdict": input_data.verdict,
                    "weapon_flag": input_data.weapon_flag,
                    "weapon_class": input_data.weapon_class,
                },
                "retrieved_legal_references": [
                    ref.to_dict() if hasattr(ref, "to_dict") else ref
                    for ref in retrieved_references
                ],
                "summary": self.summary,
                "guardrail_status": "passed",
                "limitations_note": (
                    "This is not legal advice and does not determine guilt or predict court outcome."
                ),
            }
        }


@dataclass(frozen=True)
class StaticEvaluation:
    retrieval_score: float = 0.9
    generation_score: float = 0.9
    overall_score: float = 0.9
    passed: bool = True
    issues: list[str] = None
    metric_breakdown: dict = None

    def to_dict(self):
        return {
            "retrieval_score": self.retrieval_score,
            "generation_score": self.generation_score,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "issues": self.issues or [],
            "metric_breakdown": self.metric_breakdown or {},
        }


def _evaluator(*args, **kwargs):
    return StaticEvaluation()


def test_happy_path_returns_legal_consequences_with_references_summary_status_and_note():
    result = generate_legal_consequences(
        _safe_input(),
        retriever=StaticRetriever([_reference()]),
        summarizer=StaticSummarizer(),
        evaluator=_evaluator,
    )

    payload = result.legal_consequences
    assert payload["country"] == "UAE"
    assert payload["retrieved_legal_references"]
    assert "According to the retrieved regulation" in payload["summary"]
    assert payload["guardrail_status"] == "passed"
    assert "not legal advice" in payload["limitations_note"]


def test_no_detected_actions_is_required_or_used():
    input_data = _safe_input(detected_actions=["punch"])

    result = generate_legal_consequences(
        input_data,
        retriever=StaticRetriever([_reference()]),
        summarizer=StaticSummarizer(),
        evaluator=_evaluator,
    )

    assert "detected_actions" not in str(result.to_dict())


def test_no_retrieval_path_returns_safe_fallback():
    result = generate_legal_consequences(
        _safe_input(),
        retriever=StaticRetriever([]),
        evaluator=_evaluator,
    )

    payload = result.legal_consequences
    assert payload["retrieved_legal_references"] == []
    assert "not enough" in payload["summary"].casefold()
    assert payload["guardrail_status"] == "needs_review"


def test_guardrail_failure_applies_corrected_summary_when_available():
    def failing_guardrail(*args, **kwargs):
        return LegalGuardrailResult(
            status="failed",
            issues=[],
            corrected_summary=_summary(),
            recommended_action="regenerate",
        )

    result = generate_legal_consequences(
        _safe_input(),
        retriever=StaticRetriever([_reference()]),
        summarizer=StaticSummarizer("The person is guilty and will be punished."),
        guardrail_validator=failing_guardrail,
        evaluator=_evaluator,
    )

    assert result.legal_consequences["summary"] == _summary()


def test_guardrail_failure_without_correction_returns_fallback():
    def failing_guardrail(*args, **kwargs):
        return LegalGuardrailResult(
            status="failed",
            issues=[],
            corrected_summary=None,
            recommended_action="regenerate",
        )

    result = generate_legal_consequences(
        _safe_input(),
        retriever=StaticRetriever([_reference()]),
        summarizer=StaticSummarizer("Unsafe summary."),
        guardrail_validator=failing_guardrail,
        evaluator=_evaluator,
    )

    assert "could not be produced" in result.legal_consequences["summary"]
    assert result.legal_consequences["guardrail_status"] == "needs_review"


def test_evaluation_metadata_is_returned():
    result = generate_legal_consequences(
        _safe_input(),
        retriever=StaticRetriever([_reference()]),
        summarizer=StaticSummarizer(),
        evaluator=_evaluator,
    )

    assert result.evaluation is not None
    assert result.evaluation["overall_score"] == 0.9
    assert result.evaluation["passed"] is True


def test_arabic_mode_does_not_translate_but_remains_wrapper_compatible():
    result = generate_legal_consequences(
        _safe_input(language="ar"),
        retriever=StaticRetriever([_reference()]),
        evaluator=_evaluator,
    )

    payload = result.legal_consequences
    assert payload["summary"].startswith("According to the retrieved regulation")
    assert "TranslateGemma wrapper can translate" in payload["limitations_note"]
