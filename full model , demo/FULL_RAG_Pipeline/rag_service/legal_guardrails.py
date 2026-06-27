"""Guardrails for Legal Consequences RAG outputs.

This module validates generated legal_consequences payloads. It does not run
an evaluation framework or orchestrate a demo flow.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


CAUTIOUS_TERMS = (
    "may be subject to",
    "according to the retrieved regulation",
    "depending on judicial determination",
    "possible legal consequences include",
)
LIMITATIONS_CHECKS = (
    (
        "not legal advice",
        re.compile(r"\bnot\s+legal\s+advice\b", re.IGNORECASE),
    ),
    (
        "does not determine guilt",
        re.compile(r"\bdoes\s+not\s+determine\s+guilt\b", re.IGNORECASE),
    ),
    (
        "does not predict court outcome",
        re.compile(r"\bdoes\s+not\b.*\bpredict\s+court\s+outcome\b", re.IGNORECASE),
    ),
)
EXACT_ACTION_WORDS = ("punch", "punched", "slap", "slapped", "kick", "kicked")


@dataclass(frozen=True)
class GuardrailIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LegalGuardrailResult:
    status: str
    issues: list[GuardrailIssue]
    corrected_summary: str | None
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
            "corrected_summary": self.corrected_summary,
            "recommended_action": self.recommended_action,
        }


def validate_legal_consequences_output(
    legal_output: dict[str, Any] | Any,
    *,
    legal_input: dict[str, Any] | Any | None = None,
    fallback_mode: bool = False,
) -> LegalGuardrailResult:
    """Validate a generated legal_consequences output."""

    payload = _extract_payload(legal_output)
    input_data = _coerce_mapping(legal_input or {})
    references = payload.get("retrieved_legal_references") or []
    summary = _normalize_text(str(payload.get("summary") or ""))
    limitations_note = _normalize_text(str(payload.get("limitations_note") or ""))

    issues: list[GuardrailIssue] = []
    _append_forbidden_summary_issues(summary, issues)
    _append_cautious_wording_issues(summary, issues)
    _append_exact_action_issues(summary, input_data, references, issues)
    _append_reference_issues(references, fallback_mode, issues)
    _append_limitations_issues(limitations_note, issues)

    if _contains_detected_actions(payload) or _contains_detected_actions(input_data):
        issues.append(
            GuardrailIssue(
                "detected_actions_used",
                "Legal consequences output must not use detected_actions.",
            )
        )

    corrected_summary = _correct_summary(summary, input_data, references)
    if corrected_summary == summary:
        corrected_summary = None

    status = "passed" if not issues else "failed"
    recommended_action = _recommended_action(issues)
    return LegalGuardrailResult(
        status=status,
        issues=issues,
        corrected_summary=corrected_summary,
        recommended_action=recommended_action,
    )


def _append_forbidden_summary_issues(
    summary: str,
    issues: list[GuardrailIssue],
) -> None:
    checks = (
        (
            "guilt_declaration",
            r"\b(?:is|are|was|were|be|been)\s+guilty\b|\bguilty\s+of\b",
            "Summary must not declare guilt.",
        ),
        (
            "legal_advice",
            r"\b(?:you should|you must|my advice|i advise|we advise|legal advice)\b",
            "Summary must not provide legal advice.",
        ),
        (
            "guaranteed_punishment",
            r"\b(?:will be punished|shall be punished|punishment is guaranteed|guaranteed punishment|will face)\b",
            "Summary must not state punishment is guaranteed.",
        ),
        (
            "court_outcome_prediction",
            r"\b(?:court will|judge will|will be convicted|will be acquitted|will sentence|will impose)\b",
            "Summary must not predict court outcomes.",
        ),
    )
    for code, pattern, message in checks:
        if re.search(pattern, summary, flags=re.IGNORECASE):
            issues.append(GuardrailIssue(code, message))


def _append_cautious_wording_issues(
    summary: str,
    issues: list[GuardrailIssue],
) -> None:
    lowered = summary.casefold()
    missing = [term for term in CAUTIOUS_TERMS if term not in lowered]
    if missing:
        issues.append(
            GuardrailIssue(
                "missing_cautious_wording",
                f"Summary is missing cautious wording: {', '.join(missing)}.",
            )
        )


def _append_exact_action_issues(
    summary: str,
    input_data: dict[str, Any],
    references: list[Any],
    issues: list[GuardrailIssue],
) -> None:
    summary_words = _action_words(summary)
    if not summary_words:
        return

    support_text = " ".join(
        [
            str(input_data.get("narrative") or ""),
            str(input_data.get("packet_summary") or ""),
            _reference_text(references),
        ]
    )
    supported_words = _action_words(support_text)
    unsupported = sorted(summary_words - supported_words)
    if unsupported:
        issues.append(
            GuardrailIssue(
                "unsupported_exact_action_claim",
                "Summary contains unsupported exact action words: "
                + ", ".join(unsupported)
                + ".",
            )
        )


def _append_reference_issues(
    references: list[Any],
    fallback_mode: bool,
    issues: list[GuardrailIssue],
) -> None:
    if not references and not fallback_mode:
        issues.append(
            GuardrailIssue(
                "missing_retrieved_references",
                "At least one retrieved legal reference is required unless fallback mode is used.",
            )
        )


def _append_limitations_issues(
    limitations_note: str,
    issues: list[GuardrailIssue],
) -> None:
    if not limitations_note:
        issues.append(
            GuardrailIssue(
                "missing_limitations_note",
                "Limitations note must be present.",
            )
        )
        return

    missing = [
        label for label, pattern in LIMITATIONS_CHECKS if not pattern.search(limitations_note)
    ]
    if missing:
        issues.append(
            GuardrailIssue(
                "incomplete_limitations_note",
                "Limitations note is missing required wording: "
                + ", ".join(missing)
                + ".",
            )
        )


def _correct_summary(
    summary: str,
    input_data: dict[str, Any],
    references: list[Any],
) -> str:
    corrected = summary
    replacements = {
        r"\b(?:is|are|was|were)\s+guilty\b": "has not been legally determined guilty",
        r"\bguilty\s+of\b": "subject to judicial determination for",
        r"\bwill be punished\b": "may be subject to legal consequences",
        r"\bshall be punished\b": "may be subject to legal consequences",
        r"\bpunishment is guaranteed\b": "punishment is not guaranteed",
        r"\bguaranteed punishment\b": "possible legal consequence",
        r"\bcourt will\b": "court may",
        r"\bjudge will\b": "judge may",
        r"\bwill be convicted\b": "could face judicial determination",
        r"\bwill face\b": "may be subject to",
        r"\byou should\b": "a person may consider seeking qualified legal guidance before deciding to",
        r"\byou must\b": "a person may need to",
        r"\bmy advice\b": "this non-advisory summary",
        r"\bi advise\b": "this summary does not advise",
        r"\blegal advice\b": "legal information",
    }
    for pattern, replacement in replacements.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

    support_text = " ".join(
        [
            str(input_data.get("narrative") or ""),
            str(input_data.get("packet_summary") or ""),
            _reference_text(references),
        ]
    )
    supported_words = _action_words(support_text)
    for action in EXACT_ACTION_WORDS:
        if action in _action_words(corrected) and action not in supported_words:
            corrected = re.sub(
                rf"\b{action}\b",
                "reported conduct",
                corrected,
                flags=re.IGNORECASE,
            )

    lowered = corrected.casefold()
    if any(term not in lowered for term in CAUTIOUS_TERMS):
        corrected = (
            "According to the retrieved regulation, "
            + corrected
            + " Possible legal consequences include outcomes that may be subject to "
            "judicial determination depending on the established facts."
        )
    return _normalize_text(corrected)


def _recommended_action(issues: list[GuardrailIssue]) -> str:
    if not issues:
        return "pass"
    codes = {issue.code for issue in issues}
    if "missing_retrieved_references" in codes:
        return "fallback"
    return "regenerate"


def _extract_payload(legal_output: dict[str, Any] | Any) -> dict[str, Any]:
    data = _coerce_mapping(legal_output)
    if "legal_consequences" in data:
        payload = data["legal_consequences"]
        return _coerce_mapping(payload)
    return data


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


def _reference_text(references: list[Any]) -> str:
    parts: list[str] = []
    for reference in references:
        data = _coerce_mapping(reference)
        parts.extend(
            str(data.get(field) or "")
            for field in ("law_title", "article_number", "section_title", "snippet", "source_url")
        )
    return " ".join(parts)


def _action_words(text: str) -> set[str]:
    lowered = text.casefold()
    return {word for word in EXACT_ACTION_WORDS if re.search(rf"\b{word}\b", lowered)}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
