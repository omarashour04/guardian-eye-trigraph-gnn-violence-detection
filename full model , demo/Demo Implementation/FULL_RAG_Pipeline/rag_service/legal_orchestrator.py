"""Integration-safe orchestrator for Legal Consequences RAG.

This module wires retrieval, summarization, guardrails, and evaluation for the
Legal Consequences feature. It does not modify demos, require UI changes, fetch
legal sources, build indexes, download models, or call an LLM in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from rag_service.legal_evaluation import evaluate_legal_rag
from rag_service.legal_guardrails import validate_legal_consequences_output
from rag_service.legal_retrieval import retrieve_legal_references
from rag_service.legal_summarizer import (
    ARABIC_TRANSLATION_NOTE,
    DEFAULT_LIMITATIONS_NOTE,
    summarize_legal_consequences,
)
from rag_service.schemas import LegalConsequencesInput, LegalConsequencesOutput


SAFE_FALLBACK_SUMMARY = (
    "No legal references available for selected jurisdiction. There were not "
    "enough safe retrieved legal references to produce a grounded legal "
    "consequences summary; possible legal consequences require further "
    "jurisdiction-specific retrieval and review and may be subject to judicial "
    "determination."
)


@dataclass(frozen=True)
class LegalOrchestratorResult:
    legal_consequences: dict[str, Any]
    evaluation: dict[str, Any] | None
    guardrails: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LegalConsequencesRAG:
    """Small injectable service for the Legal Consequences RAG path."""

    def __init__(
        self,
        *,
        legal_index: Any | None = None,
        retriever: Any | None = None,
        summarizer: Any | None = None,
        summarization_provider: Any | None = None,
        guardrail_validator: Any | None = None,
        evaluator: Any | None = None,
        embedding_provider: Any | None = None,
        top_k: int = 5,
        include_evaluation: bool = True,
    ) -> None:
        self.legal_index = legal_index
        self.retriever = retriever
        self.summarizer = summarizer
        self.summarization_provider = summarization_provider
        self.guardrail_validator = guardrail_validator or validate_legal_consequences_output
        self.evaluator = evaluator or evaluate_legal_rag
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.include_evaluation = include_evaluation

    def run(self, input_data: LegalConsequencesInput | dict[str, Any]) -> LegalOrchestratorResult:
        safe_input = _validate_safe_input(input_data)
        references = self._retrieve(safe_input)
        fallback_mode = not references

        output = self._summarize(safe_input, references)
        guardrail_result = _call_guardrail_validator(
            self.guardrail_validator,
            output,
            safe_input,
            fallback_mode=fallback_mode,
        )
        final_output = _apply_guardrail_result(
            output,
            guardrail_result,
            safe_input=safe_input,
            references=references,
            fallback_mode=fallback_mode,
        )

        final_guardrail_result = _call_guardrail_validator(
            self.guardrail_validator,
            final_output,
            safe_input,
            fallback_mode=fallback_mode or _is_fallback_summary(final_output),
        )
        final_output = _set_guardrail_status(final_output, final_guardrail_result)

        evaluation = None
        if self.include_evaluation:
            evaluation = _call_evaluator(
                self.evaluator,
                safe_input,
                references,
                final_output,
                fallback_mode=fallback_mode or _is_fallback_summary(final_output),
                top_k=self.top_k,
            )

        return LegalOrchestratorResult(
            legal_consequences=_extract_legal_consequences(final_output),
            evaluation=evaluation,
            guardrails=_to_dict(final_guardrail_result),
        )

    def _retrieve(self, safe_input: LegalConsequencesInput) -> list[Any]:
        if self.retriever is not None:
            retrieval_result = _call_dependency(
                self.retriever,
                "retrieve",
                input_data=safe_input,
                legal_input=safe_input,
                top_k=self.top_k,
            )
        else:
            if self.legal_index is None:
                raise ValueError("LegalConsequencesRAG requires a legal_index or retriever.")
            retrieval_result = retrieve_legal_references(
                safe_input,
                self.legal_index,
                embedding_provider=self.embedding_provider,
                top_k=self.top_k,
            )
        return _extract_references(retrieval_result)

    def _summarize(
        self,
        safe_input: LegalConsequencesInput,
        references: list[Any],
    ) -> LegalConsequencesOutput | dict[str, Any]:
        if self.summarizer is not None:
            return _call_dependency(
                self.summarizer,
                "summarize",
                input_data=safe_input,
                legal_input=safe_input,
                retrieved_references=references,
                references=references,
                summarization_provider=self.summarization_provider,
            )
        return summarize_legal_consequences(
            safe_input,
            references,
            summarization_provider=self.summarization_provider,
        )


def generate_legal_consequences(
    input_data: LegalConsequencesInput | dict[str, Any],
    *,
    legal_index: Any | None = None,
    retriever: Any | None = None,
    summarizer: Any | None = None,
    summarization_provider: Any | None = None,
    guardrail_validator: Any | None = None,
    evaluator: Any | None = None,
    embedding_provider: Any | None = None,
    top_k: int = 5,
    include_evaluation: bool = True,
) -> LegalOrchestratorResult:
    service = LegalConsequencesRAG(
        legal_index=legal_index,
        retriever=retriever,
        summarizer=summarizer,
        summarization_provider=summarization_provider,
        guardrail_validator=guardrail_validator,
        evaluator=evaluator,
        embedding_provider=embedding_provider,
        top_k=top_k,
        include_evaluation=include_evaluation,
    )
    return service.run(input_data)


def _validate_safe_input(input_data: LegalConsequencesInput | dict[str, Any]) -> LegalConsequencesInput:
    if isinstance(input_data, LegalConsequencesInput):
        return input_data
    data = _to_dict(input_data)
    data.pop("detected_actions", None)
    return LegalConsequencesInput(**data)


def _call_dependency(
    dependency: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    target = getattr(dependency, method_name, dependency)
    try:
        return target(**kwargs)
    except TypeError:
        compact_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"input_data", "retrieved_references", "references"}
        }
        try:
            return target(**compact_kwargs)
        except TypeError:
            if "retrieved_references" in compact_kwargs:
                return target(
                    compact_kwargs.get("input_data"),
                    compact_kwargs["retrieved_references"],
                )
            if "references" in compact_kwargs:
                return target(compact_kwargs.get("input_data"), compact_kwargs["references"])
            return target(compact_kwargs.get("input_data"))


def _call_guardrail_validator(
    validator: Any,
    output: Any,
    safe_input: LegalConsequencesInput,
    *,
    fallback_mode: bool,
) -> Any:
    try:
        return validator(output, legal_input=safe_input, fallback_mode=fallback_mode)
    except TypeError:
        return validator(output)


def _call_evaluator(
    evaluator: Any,
    safe_input: LegalConsequencesInput,
    references: list[Any],
    output: Any,
    *,
    fallback_mode: bool,
    top_k: int,
) -> dict[str, Any]:
    try:
        result = evaluator(
            safe_input,
            references,
            output,
            fallback_mode=fallback_mode,
            top_k=top_k,
        )
    except TypeError:
        result = evaluator(safe_input, references, output)
    return _to_dict(result)


def _apply_guardrail_result(
    output: LegalConsequencesOutput | dict[str, Any],
    guardrail_result: Any,
    *,
    safe_input: LegalConsequencesInput,
    references: list[Any],
    fallback_mode: bool,
) -> LegalConsequencesOutput:
    payload = _extract_legal_consequences(output)
    guardrail_data = _to_dict(guardrail_result)
    if guardrail_data.get("status") == "passed":
        return LegalConsequencesOutput(legal_consequences=payload)

    corrected_summary = guardrail_data.get("corrected_summary")
    payload["summary"] = corrected_summary or _fallback_summary(references)
    payload["guardrail_status"] = "needs_review" if fallback_mode or not corrected_summary else "passed"
    payload["limitations_note"] = payload.get("limitations_note") or _limitations_note(safe_input)
    return LegalConsequencesOutput(legal_consequences=payload)


def _set_guardrail_status(
    output: LegalConsequencesOutput | dict[str, Any],
    guardrail_result: Any,
) -> LegalConsequencesOutput:
    payload = _extract_legal_consequences(output)
    guardrail_data = _to_dict(guardrail_result)
    if not payload.get("retrieved_legal_references"):
        payload["guardrail_status"] = "needs_review"
    elif guardrail_data.get("status") == "passed":
        payload["guardrail_status"] = "passed"
    else:
        payload["guardrail_status"] = "needs_review"
    return LegalConsequencesOutput(legal_consequences=payload)


def _extract_references(retrieval_result: Any) -> list[Any]:
    if retrieval_result is None:
        return []
    if isinstance(retrieval_result, list):
        return retrieval_result
    if isinstance(retrieval_result, tuple):
        return list(retrieval_result)
    if isinstance(retrieval_result, dict):
        if "references" in retrieval_result:
            return list(retrieval_result["references"] or [])
        if "retrieved_legal_references" in retrieval_result:
            return list(retrieval_result["retrieved_legal_references"] or [])
    if hasattr(retrieval_result, "references"):
        return list(retrieval_result.references or [])
    return []


def _extract_legal_consequences(output: LegalConsequencesOutput | dict[str, Any]) -> dict[str, Any]:
    data = _to_dict(output)
    if "legal_consequences" in data:
        return dict(data["legal_consequences"])
    return dict(data)


def _fallback_summary(references: list[Any]) -> str:
    if references:
        return (
            "A safe grounded legal consequences summary could not be produced from "
            "the generated text. According to the retrieved regulation record "
            "available here, possible legal consequences include only further "
            "review of the retrieved references and may be subject to change "
            "depending on judicial determination."
        )
    return SAFE_FALLBACK_SUMMARY


def _is_fallback_summary(output: LegalConsequencesOutput | dict[str, Any]) -> bool:
    summary = str(_extract_legal_consequences(output).get("summary") or "").casefold()
    return "not enough safe retrieved legal references" in summary or (
        "could not be produced" in summary and "safe grounded" in summary
    )


def _limitations_note(safe_input: LegalConsequencesInput) -> str:
    note = DEFAULT_LIMITATIONS_NOTE
    if safe_input.language.casefold().startswith("ar"):
        note += ARABIC_TRANSLATION_NOTE
    return note


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return {}
