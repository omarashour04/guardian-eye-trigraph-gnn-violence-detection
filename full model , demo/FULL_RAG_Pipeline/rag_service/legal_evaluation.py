"""Deterministic evaluation helpers for Legal Consequences RAG.

This module scores retrieved references and generated legal summaries offline.
It does not call an LLM, use real embeddings, modify demos, or orchestrate the
final pipeline.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from rag_service.legal_guardrails import validate_legal_consequences_output


RETRIEVAL_WEIGHTS = {
    "top_k_similarity": 0.30,
    "country_filter_correctness": 0.25,
    "keyword_overlap": 0.20,
    "source_officialness": 0.15,
    "article_level_match": 0.10,
}
GENERATION_WEIGHTS = {
    "groundedness": 0.35,
    "citation_presence": 0.20,
    "guardrail_compliance": 0.25,
    "language_compatibility": 0.10,
    "unsupported_action_safety": 0.05,
    "detected_actions_absence": 0.05,
}


@dataclass(frozen=True)
class LegalEvaluationResult:
    retrieval_score: float
    generation_score: float
    overall_score: float
    passed: bool
    issues: list[str]
    metric_breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_retrieval(
    legal_input: dict[str, Any] | Any,
    retrieved_references: list[Any] | Any,
    *,
    top_k: int | None = None,
) -> tuple[float, dict[str, float], list[str]]:
    """Score retrieved references against Legal RAG input context."""

    input_data = _coerce_mapping(legal_input)
    references = _coerce_references(retrieved_references)
    if top_k is not None:
        references = references[: max(0, top_k)]

    if not references:
        return (
            0.0,
            {
                "top_k_similarity": 0.0,
                "country_filter_correctness": 0.0,
                "keyword_overlap": 0.0,
                "source_officialness": 0.0,
                "article_level_match": 0.0,
            },
            ["retrieval:no_references"],
        )

    metrics = {
        "top_k_similarity": _mean(_clamp(float(ref.get("score", 0.0))) for ref in references),
        "country_filter_correctness": _country_score(input_data, references),
        "keyword_overlap": _retrieval_keyword_overlap(input_data, references),
        "source_officialness": _mean(1.0 if ref.get("official_source") else 0.0 for ref in references),
        "article_level_match": _mean(1.0 if ref.get("article_number") else 0.0 for ref in references),
    }
    score = _weighted_score(metrics, RETRIEVAL_WEIGHTS)
    issues: list[str] = []
    if metrics["country_filter_correctness"] < 1.0:
        issues.append("retrieval:country_mismatch")
    if metrics["keyword_overlap"] == 0.0:
        issues.append("retrieval:no_keyword_overlap")
    return score, metrics, issues


def evaluate_generation(
    legal_input: dict[str, Any] | Any,
    legal_output: dict[str, Any] | Any,
    *,
    fallback_mode: bool = False,
) -> tuple[float, dict[str, float], list[str]]:
    """Score generated legal_consequences output."""

    input_data = _coerce_mapping(legal_input)
    payload = _extract_payload(legal_output)
    references = _coerce_references(payload.get("retrieved_legal_references") or [])
    summary = _normalize_text(str(payload.get("summary") or ""))
    limitations_note = _normalize_text(str(payload.get("limitations_note") or ""))

    guardrail_result = validate_legal_consequences_output(
        legal_output,
        legal_input=input_data,
        fallback_mode=fallback_mode,
    )
    guardrail_codes = [issue.code for issue in guardrail_result.issues]

    metrics = {
        "groundedness": _groundedness_score(summary, references),
        "citation_presence": _citation_presence_score(summary, references),
        "guardrail_compliance": 1.0 if guardrail_result.status == "passed" else 0.0,
        "language_compatibility": _language_compatibility_score(input_data, summary, limitations_note),
        "unsupported_action_safety": (
            0.0 if "unsupported_exact_action_claim" in guardrail_codes else 1.0
        ),
        "detected_actions_absence": (
            0.0
            if _contains_detected_actions(payload) or _contains_detected_actions(input_data)
            else 1.0
        ),
    }
    score = _weighted_score(metrics, GENERATION_WEIGHTS)

    issues: list[str] = [f"guardrail:{code}" for code in guardrail_codes]
    if metrics["groundedness"] < 0.65:
        issues.append("generation:low_groundedness")
    if metrics["citation_presence"] == 0.0:
        issues.append("generation:missing_reference_citation")
    if metrics["language_compatibility"] == 0.0:
        issues.append("generation:language_incompatible")
    if metrics["detected_actions_absence"] == 0.0:
        issues.append("generation:detected_actions_used")
    return score, metrics, _dedupe(issues)


def evaluate_legal_rag(
    legal_input: dict[str, Any] | Any,
    retrieved_references: list[Any] | Any,
    legal_output: dict[str, Any] | Any,
    *,
    fallback_mode: bool = False,
    top_k: int | None = None,
) -> LegalEvaluationResult:
    """Evaluate retrieval and generation together."""

    retrieval_score, retrieval_metrics, retrieval_issues = evaluate_retrieval(
        legal_input,
        retrieved_references,
        top_k=top_k,
    )
    generation_score, generation_metrics, generation_issues = evaluate_generation(
        legal_input,
        legal_output,
        fallback_mode=fallback_mode,
    )
    overall_score = round((0.45 * retrieval_score) + (0.55 * generation_score), 6)
    issues = _dedupe(retrieval_issues + generation_issues)
    passed = (
        overall_score >= 0.75
        and generation_score >= 0.75
        and "guardrail:missing_retrieved_references" not in issues
        and not any(issue.startswith("guardrail:") for issue in generation_issues)
    )

    return LegalEvaluationResult(
        retrieval_score=retrieval_score,
        generation_score=generation_score,
        overall_score=overall_score,
        passed=passed,
        issues=issues,
        metric_breakdown={
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
        },
    )


def _country_score(input_data: dict[str, Any], references: list[dict[str, Any]]) -> float:
    country = str(input_data.get("country") or "").casefold().strip()
    if not country:
        return 0.0
    return _mean(
        1.0 if str(ref.get("country") or "").casefold().strip() == country else 0.0
        for ref in references
    )


def _retrieval_keyword_overlap(
    input_data: dict[str, Any],
    references: list[dict[str, Any]],
) -> float:
    terms = _context_terms(input_data)
    if not terms:
        return 0.0

    searchable = " ".join(_reference_context(ref) for ref in references).casefold()
    matched = {term for term in terms if term in searchable}
    denominator = min(max(len(terms), 1), 8)
    return round(min(1.0, len(matched) / denominator), 6)


def _groundedness_score(summary: str, references: list[dict[str, Any]]) -> float:
    if not references or not summary:
        return 0.0

    reference_terms = _significant_terms(" ".join(_reference_context(ref) for ref in references))
    claim_terms = _claim_terms(summary)
    if not claim_terms:
        return 1.0

    matched = claim_terms & reference_terms
    return round(len(matched) / len(claim_terms), 6)


def _citation_presence_score(summary: str, references: list[dict[str, Any]]) -> float:
    if not references:
        return 0.0

    lowered = summary.casefold()
    for ref in references:
        law_title = str(ref.get("law_title") or "").casefold()
        article_number = str(ref.get("article_number") or "").casefold()
        source_url = str(ref.get("source_url") or "").casefold()
        if law_title and law_title in lowered:
            return 1.0
        if article_number and article_number in lowered:
            return 1.0
        if source_url and source_url in lowered:
            return 1.0
    return 0.0


def _language_compatibility_score(
    input_data: dict[str, Any],
    summary: str,
    limitations_note: str,
) -> float:
    language = str(input_data.get("language") or "en").casefold()
    if language.startswith("ar"):
        note = limitations_note.casefold()
        return 1.0 if "translategemma" in note or "translate" in note else 0.0
    return 1.0 if summary else 0.0


def _context_terms(input_data: dict[str, Any]) -> set[str]:
    parts = [
        str(input_data.get("verdict") or ""),
        str(input_data.get("packet_summary") or ""),
        str(input_data.get("narrative") or ""),
    ]
    if input_data.get("weapon_flag"):
        parts.extend(["weapon", "dangerous object"])
        if input_data.get("weapon_class"):
            parts.append(str(input_data["weapon_class"]))
    return _significant_terms(" ".join(parts))


def _reference_context(reference: dict[str, Any]) -> str:
    fields = (
        "law_title",
        "article_number",
        "section_title",
        "source_url",
        "snippet",
        "country",
        "violence_category",
    )
    return " ".join(str(reference.get(field) or "") for field in fields)


def _claim_terms(summary: str) -> set[str]:
    terms = _significant_terms(summary)
    return terms - _GENERIC_SUMMARY_TERMS


def _significant_terms(text: str) -> set[str]:
    words = re.findall(r"[\w\u0600-\u06FF]+", text.casefold())
    return {word for word in words if len(word) >= 4 and word not in _STOPWORDS}


def _extract_payload(legal_output: dict[str, Any] | Any) -> dict[str, Any]:
    data = _coerce_mapping(legal_output)
    if "legal_consequences" in data:
        return _coerce_mapping(data["legal_consequences"])
    return data


def _coerce_references(references: list[Any] | Any) -> list[dict[str, Any]]:
    if hasattr(references, "references"):
        references = references.references
    result = []
    for reference in references or []:
        result.append(_coerce_mapping(reference))
    return result


def _coerce_mapping(value: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return {}


def _contains_detected_actions(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key == "detected_actions" or _contains_detected_actions(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_detected_actions(item) for item in value)
    return False


def _weighted_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return round(
        sum(_clamp(metrics[name]) * weight for name, weight in weights.items()),
        6,
    )


def _mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _dedupe(issues: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for issue in issues:
        if issue not in seen:
            result.append(issue)
            seen.add(issue)
    return result


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_STOPWORDS = {
    "according",
    "retrieved",
    "regulation",
    "reported",
    "conduct",
    "legal",
    "consequences",
    "include",
    "possible",
    "depending",
    "judicial",
    "determination",
    "subject",
    "summary",
    "source",
    "from",
    "with",
    "that",
    "this",
    "they",
    "them",
    "person",
    "people",
    "court",
    "facts",
    "available",
    "there",
    "were",
    "not",
    "enough",
    "produce",
    "grounded",
    "specific",
    "review",
    "need",
    "further",
    "change",
    "under",
    "into",
    "only",
    "here",
    "record",
    "text",
    "statutory",
    "measures",
    "outcomes",
    "established",
    "competent",
    "authorities",
    "exposure",
    "reflected",
}

_GENERIC_SUMMARY_TERMS = {
    "penalties",
    "penalty",
    "protective",
    "provisions",
    "category",
    "categories",
    "including",
    "violent",
    "violence",
}
