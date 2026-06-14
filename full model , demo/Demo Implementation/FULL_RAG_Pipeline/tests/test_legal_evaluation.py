from rag_service.legal_evaluation import (
    evaluate_generation,
    evaluate_legal_rag,
    evaluate_retrieval,
)
from rag_service.legal_summarizer import summarize_legal_consequences


def _safe_input(**overrides):
    values = {
        "country": "UAE",
        "verdict": "violence",
        "confidence": 0.9,
        "packet_summary": "The packet describes assault and physical harm.",
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
        "snippet": "Retrieved text addresses assault, physical harm, and possible penalties.",
        "score": 0.9,
        "country": "UAE",
        "violence_category": "assault",
        "official_source": True,
    }
    values.update(overrides)
    return values


def _output(summary=None, references=None, language="en"):
    input_data = _safe_input(language=language)
    output = summarize_legal_consequences(input_data, references if references is not None else [_reference()])
    if summary is not None:
        data = output.model_dump()
        data["legal_consequences"]["summary"] = summary
        return data
    return output


def test_retrieval_score_rewards_correct_country():
    good_score, good_metrics, _ = evaluate_retrieval(_safe_input(country="UAE"), [_reference(country="UAE")])
    bad_score, bad_metrics, _ = evaluate_retrieval(_safe_input(country="UAE"), [_reference(country="Canada")])

    assert good_metrics["country_filter_correctness"] == 1.0
    assert bad_metrics["country_filter_correctness"] == 0.0
    assert good_score > bad_score


def test_retrieval_score_rewards_official_source():
    official_score, official_metrics, _ = evaluate_retrieval(_safe_input(), [_reference(official_source=True)])
    unofficial_score, unofficial_metrics, _ = evaluate_retrieval(_safe_input(), [_reference(official_source=False)])

    assert official_metrics["source_officialness"] == 1.0
    assert unofficial_metrics["source_officialness"] == 0.0
    assert official_score > unofficial_score


def test_retrieval_score_rewards_article_level_match():
    article_score, article_metrics, _ = evaluate_retrieval(_safe_input(), [_reference(article_number="Article 1")])
    paragraph_score, paragraph_metrics, _ = evaluate_retrieval(_safe_input(), [_reference(article_number=None)])

    assert article_metrics["article_level_match"] == 1.0
    assert paragraph_metrics["article_level_match"] == 0.0
    assert article_score > paragraph_score


def test_retrieval_score_rewards_keyword_overlap():
    matched_score, matched_metrics, _ = evaluate_retrieval(
        _safe_input(packet_summary="assault physical harm"),
        [_reference(snippet="This article addresses assault and physical harm.")],
    )
    weak_score, weak_metrics, _ = evaluate_retrieval(
        _safe_input(packet_summary="assault physical harm"),
        [_reference(snippet="This article addresses filing fees.")],
    )

    assert matched_metrics["keyword_overlap"] > weak_metrics["keyword_overlap"]
    assert matched_score > weak_score


def test_generation_score_passes_grounded_cautious_summary():
    output = _output()

    score, metrics, issues = evaluate_generation(_safe_input(), output)
    combined = evaluate_legal_rag(_safe_input(), [_reference()], output)

    assert score >= 0.75
    assert metrics["groundedness"] > 0.65
    assert "generation:missing_reference_citation" not in issues
    assert combined.passed is True


def test_generation_score_fails_unsupported_claims():
    output = _output(
        summary=(
            "According to the retrieved regulation, deportation may be subject to review. "
            "Possible legal consequences include deportation depending on judicial determination."
        )
    )

    score, metrics, issues = evaluate_generation(_safe_input(), output)

    assert metrics["groundedness"] < 0.65
    assert "generation:low_groundedness" in issues
    assert score < 0.75


def test_generation_score_fails_missing_reference_citation():
    output = _output(
        summary=(
            "According to the retrieved regulation, the reported conduct may be subject to review. "
            "Possible legal consequences include penalties depending on judicial determination."
        )
    )

    score, metrics, issues = evaluate_generation(_safe_input(), output)

    assert metrics["citation_presence"] == 0.0
    assert "generation:missing_reference_citation" in issues
    assert score < 0.9


def test_generation_score_integrates_guardrail_failures():
    output = _output(
        summary=(
            "According to the retrieved regulation, the person is guilty and punishment is guaranteed. "
            "Possible legal consequences include penalties depending on judicial determination."
        )
    )

    score, metrics, issues = evaluate_generation(_safe_input(), output)

    assert metrics["guardrail_compliance"] == 0.0
    assert "guardrail:guilt_declaration" in issues
    assert "guardrail:guaranteed_punishment" in issues
    assert score < 0.75


def test_language_compatibility_supports_english_and_arabic_wrapper_modes():
    english_score, english_metrics, _ = evaluate_generation(_safe_input(language="en"), _output(language="en"))
    arabic_score, arabic_metrics, _ = evaluate_generation(_safe_input(language="ar"), _output(language="ar"))

    assert english_metrics["language_compatibility"] == 1.0
    assert arabic_metrics["language_compatibility"] == 1.0
    assert english_score > 0
    assert arabic_score > 0


def test_detected_actions_use_fails_generation():
    output = _output()
    data = output.model_dump()
    data["legal_consequences"]["detected_actions"] = ["punch"]

    score, metrics, issues = evaluate_generation(_safe_input(), data)

    assert metrics["detected_actions_absence"] == 0.0
    assert "generation:detected_actions_used" in issues
    assert score < 0.75
