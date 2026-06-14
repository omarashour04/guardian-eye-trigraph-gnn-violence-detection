"""Grounded legal-consequence summarization for Legal Consequences RAG.

This module turns retrieved legal references into a cautious schema-compatible
summary. It does not orchestrate the demo, evaluate, or require a real LLM in
tests.
"""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from rag_service.legal_retrieval import RankedLegalReference
from rag_service.schemas import LegalConsequencesInput, LegalConsequencesOutput


PLANNED_PRIMARY_MODEL = "Qwen/Qwen3-8B-Instruct"
PLANNED_FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LIMITATIONS_NOTE = (
    "This is not legal advice and does not determine guilt or predict court outcome."
)
ARABIC_TRANSLATION_NOTE = (
    " Arabic translation is not performed in this summarizer; the existing "
    "TranslateGemma wrapper can translate the final legal_consequences output later."
)


class SummarizationProvider(Protocol):
    def summarize(
        self,
        *,
        input_data: dict[str, Any],
        references: list[dict[str, Any]],
        legal_context: str,
    ) -> str:
        """Return a grounded English summary from retrieved legal references only."""


class GroundedTemplateSummarizationProvider:
    """Deterministic provider used by default and in offline tests."""

    def summarize(
        self,
        *,
        input_data: dict[str, Any],
        references: list[dict[str, Any]],
        legal_context: str,
    ) -> str:
        if not references:
            return _empty_reference_summary()

        first = references[0]
        reference_phrase = _reference_phrase(first)
        weapon_phrase = ""
        if input_data.get("weapon_flag"):
            weapon_class = input_data.get("weapon_class")
            weapon_phrase = (
                f" involving a potential weapon or dangerous object"
                f"{' such as ' + str(weapon_class) if weapon_class else ''}"
            )

        categories = sorted(
            {
                str(reference.get("violence_category"))
                for reference in references
                if reference.get("violence_category")
            }
        )
        category_phrase = (
            f" Retrieved categories include {', '.join(categories)}." if categories else ""
        )

        return (
            f"According to the retrieved regulation, including {reference_phrase}, "
            f"the reported {input_data.get('verdict', 'incident')}{weapon_phrase} "
            "may be subject to legal provisions reflected in the retrieved text. "
            "Possible legal consequences include exposure to statutory penalties, "
            "protective measures, or other court-determined outcomes depending on "
            "judicial determination and the facts established by competent authorities."
            f"{category_phrase}"
        )


class QwenSummarizationProvider:
    """Lazy planned provider for future local model use.

    Tests should inject a mock provider and should not instantiate this class.
    """

    def __init__(
        self,
        model_name: str = PLANNED_PRIMARY_MODEL,
        fallback_model_name: str = PLANNED_FALLBACK_MODEL,
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name

    def summarize(
        self,
        *,
        input_data: dict[str, Any],
        references: list[dict[str, Any]],
        legal_context: str,
    ) -> str:
        raise NotImplementedError(
            "Qwen summarization is planned for a later local runtime integration. "
            "Inject a SummarizationProvider for tests or use "
            "GroundedTemplateSummarizationProvider."
        )


def summarize_legal_consequences(
    input_data: LegalConsequencesInput | dict[str, Any],
    retrieved_references: list[RankedLegalReference | dict[str, Any]] | Any,
    *,
    summarization_provider: SummarizationProvider | None = None,
) -> LegalConsequencesOutput:
    """Create schema-compatible legal_consequences output from retrieved references."""

    safe_input = _coerce_input(input_data)
    references = _coerce_references(retrieved_references)
    output_references = [_schema_reference(reference) for reference in references]

    if output_references:
        provider = summarization_provider or GroundedTemplateSummarizationProvider()
        summary = provider.summarize(
            input_data=safe_input,
            references=output_references,
            legal_context=_legal_context(output_references),
        )
        guardrail_status = "passed"
    else:
        summary = _empty_reference_summary()
        guardrail_status = "needs_review"

    summary = _ensure_cautious_grounded_summary(summary, output_references)
    limitations_note = DEFAULT_LIMITATIONS_NOTE
    if str(safe_input.get("language", "")).casefold().startswith("ar"):
        limitations_note += ARABIC_TRANSLATION_NOTE

    return LegalConsequencesOutput(
        legal_consequences={
            "country": safe_input["country"],
            "query_basis": {
                "verdict": safe_input["verdict"],
                "weapon_flag": safe_input["weapon_flag"],
                "weapon_class": safe_input.get("weapon_class"),
            },
            "retrieved_legal_references": output_references,
            "summary": summary,
            "guardrail_status": guardrail_status,
            "limitations_note": limitations_note,
        }
    )


def _coerce_input(input_data: LegalConsequencesInput | dict[str, Any]) -> dict[str, Any]:
    if isinstance(input_data, dict):
        data = dict(input_data)
    elif hasattr(input_data, "model_dump"):
        data = input_data.model_dump()
    elif is_dataclass(input_data):
        data = asdict(input_data)
    else:
        raise TypeError("Legal summarizer expects LegalConsequencesInput or dict input.")

    if "detected_actions" in data:
        data.pop("detected_actions")
    return data


def _coerce_references(
    retrieved_references: list[RankedLegalReference | dict[str, Any]] | Any,
) -> list[dict[str, Any]]:
    if hasattr(retrieved_references, "references"):
        retrieved_references = retrieved_references.references

    references: list[dict[str, Any]] = []
    for reference in retrieved_references or []:
        if isinstance(reference, dict):
            data = dict(reference)
        elif hasattr(reference, "to_dict"):
            data = reference.to_dict()
        elif is_dataclass(reference):
            data = asdict(reference)
        else:
            raise TypeError("Retrieved references must be dict-like or RankedLegalReference.")
        data.pop("detected_actions", None)
        references.append(data)
    return references


def _schema_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "law_title": reference["law_title"],
        "article_number": reference.get("article_number"),
        "section_title": reference.get("section_title"),
        "source_url": reference["source_url"],
        "snippet": reference.get("snippet", ""),
        "score": float(reference.get("score", 0.0)),
        "country": reference.get("country"),
        "violence_category": reference.get("violence_category"),
        "official_source": reference.get("official_source"),
    }


def _legal_context(references: list[dict[str, Any]]) -> str:
    lines = []
    for reference in references:
        lines.append(
            " | ".join(
                part
                for part in [
                    reference.get("law_title"),
                    reference.get("article_number"),
                    reference.get("section_title"),
                    reference.get("source_url"),
                    reference.get("snippet"),
                ]
                if part
            )
        )
    return "\n".join(lines)


def _ensure_cautious_grounded_summary(
    summary: str,
    references: list[dict[str, Any]],
) -> str:
    summary = _remove_unsupported_action_claims(_normalize_text(summary))
    summary = _remove_forbidden_certainty(summary)

    cautious_terms = [
        "according to the retrieved regulation",
        "may be subject to",
        "depending on judicial determination",
        "possible legal consequences include",
    ]
    lowered = summary.casefold()
    missing_terms = [term for term in cautious_terms if term not in lowered]

    if missing_terms and references:
        summary = (
            f"According to the retrieved regulation, {summary} "
            "Possible legal consequences include measures that may be subject to "
            "judicial determination depending on the established facts."
        )
    elif missing_terms:
        summary = _empty_reference_summary()

    return _normalize_text(summary)


def _remove_unsupported_action_claims(summary: str) -> str:
    unsupported_actions = ("punch", "punched", "slap", "slapped", "kick", "kicked")
    for action in unsupported_actions:
        summary = re.sub(rf"\b{action}\b", "reported conduct", summary, flags=re.IGNORECASE)
    return summary


def _remove_forbidden_certainty(summary: str) -> str:
    replacements = {
        r"\bis guilty\b": "has not been legally determined guilty",
        r"\bare guilty\b": "have not been legally determined guilty",
        r"\bguilty\b": "subject to judicial determination",
        r"\bwill be punished\b": "may be subject to legal consequences",
        r"\bpunishment is guaranteed\b": "punishment is not guaranteed",
        r"\bguaranteed punishment\b": "possible legal consequence",
        r"\bcourt will\b": "court may",
    }
    for pattern, replacement in replacements.items():
        summary = re.sub(pattern, replacement, summary, flags=re.IGNORECASE)
    return summary


def _empty_reference_summary() -> str:
    return (
        "No legal references available for selected jurisdiction. There were not "
        "enough retrieved legal references to produce a grounded legal consequences "
        "summary; possible legal consequences require further jurisdiction-specific "
        "retrieval and review and may be subject to judicial determination."
    )


def _reference_phrase(reference: dict[str, Any]) -> str:
    parts = [reference.get("law_title")]
    if reference.get("article_number"):
        parts.append(str(reference["article_number"]))
    if reference.get("source_url"):
        parts.append(f"source {reference['source_url']}")
    return " ".join(part for part in parts if part)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
