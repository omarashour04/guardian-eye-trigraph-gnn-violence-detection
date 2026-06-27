"""
Guardian Eye Ask RAG service.

This module is used only by POST /ask. It routes the question, retrieves
available context, asks a Qwen text LLM when available, and falls back to the
current deterministic QA path when context or model generation is unavailable.
"""

from __future__ import annotations

import datetime as _dt
import gc
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from text_quality import ARABIC_ONLY_INSTRUCTION, polish_generated_text


DEFAULT_ASK_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
AskRoute = Literal[
    "current_incident",
    "history_memory",
    "reference_corpus",
    "legal_rag",
]


@dataclass(frozen=True)
class AskContext:
    route: AskRoute
    query: str
    language: str
    context_items: list[dict[str, Any]]
    warnings: list[str]


@dataclass(frozen=True)
class AskResult:
    answer: str
    ask_mode: Literal["llm", "fallback", "grounded"]
    selected_route: AskRoute | None
    retrieved_context_count: int
    reason_if_fallback: str | None = None
    related_incident_ids: tuple[str, ...] = ()
    grounding_label: str | None = None
    vlm_summary_used: bool = False
    summary_source: str | None = None
    vlm_people_count: int | None = None
    vlm_violence_type: str | None = None


REFERENCE_CORPUS: tuple[dict[str, Any], ...] = (
    {
        "id": "system-role",
        "title": "Guardian Eye role",
        "tags": {"system", "guardian", "scope", "security"},
        "text_en": (
            "Guardian Eye is a decision-support system. Answers should explain "
            "model evidence and operational next steps without declaring guilt."
        ),
        "text_ar": (
            "Guardian Eye نظام لدعم القرار. يجب أن تشرح الإجابات أدلة النموذج "
            "والخطوات التشغيلية دون إعلان الذنب."
        ),
    },
    {
        "id": "current-incident-evidence",
        "title": "Current incident evidence",
        "tags": {"current", "incident", "verdict", "confidence", "gates", "peak"},
        "text_en": (
            "For current incident questions, answer only from saved verdict, "
            "confidence, gate weights, peak window, people count, weapon flag, "
            "packet summary, and narrative."
        ),
        "text_ar": (
            "في أسئلة الحادثة الحالية، أجب فقط من الحكم المحفوظ والثقة وأوزان "
            "البوابات ونافذة الذروة وعدد الأشخاص ومؤشر السلاح وملخص الدليل والسرد."
        ),
    },
    {
        "id": "history-memory-guidance",
        "title": "History memory guidance",
        "tags": {"history", "previous", "compare", "memory", "trend"},
        "text_en": (
            "History answers may compare saved incidents by date, source, verdict, "
            "confidence, and telemetry. Do not infer unseen details across incidents."
        ),
        "text_ar": (
            "يمكن لإجابات السجل مقارنة الحوادث المحفوظة حسب التاريخ والمصدر والحكم "
            "والثقة والقياسات. لا تستنتج تفاصيل غير مرئية عبر الحوادث."
        ),
    },
    {
        "id": "legal-boundary",
        "title": "Legal answer boundary",
        "tags": {"legal", "law", "consequence", "country", "liability"},
        "text_en": (
            "Legal answers must use retrieved legal context only, must not provide "
            "legal advice, and must not identify attacker, victim, guilt, or court outcome."
        ),
        "text_ar": (
            "يجب أن تستخدم الإجابات القانونية السياق القانوني المسترجع فقط، وألا "
            "تقدم استشارة قانونية أو تحدد مهاجما أو ضحية أو ذنبا أو نتيجة محكمة."
        ),
    },
    {
        "id": "non-violence-boundary",
        "title": "Non-violence boundary",
        "tags": {"non-violence", "threshold", "legal", "calm"},
        "text_en": (
            "If an incident is classified as non-violent, say that no violence-specific "
            "legal or security escalation should be inferred from the model alone."
        ),
        "text_ar": (
            "إذا صنفت الحادثة على أنها غير عنيفة، فاذكر أنه لا ينبغي استنتاج تصعيد "
            "قانوني أو أمني متعلق بالعنف من النموذج وحده."
        ),
    },
)


def answer_question_rag(
    *,
    question: str,
    language: str,
    db,
    clip_id: str | None,
    incident_id: str | None,
    country: str | None,
    fallback: Callable[[], str],
) -> str:
    return answer_question_rag_result(
        question=question,
        language=language,
        db=db,
        clip_id=clip_id,
        incident_id=incident_id,
        country=country,
        fallback=fallback,
    ).answer


def answer_question_rag_result(
    *,
    question: str,
    language: str,
    db,
    clip_id: str | None,
    incident_id: str | None,
    country: str | None,
    fallback: Callable[[], str],
) -> AskResult:
    from history_qa_service import (
        detect_question_language,
        format_history_answer,
    )

    language = detect_question_language(question, language)

    try:
        route = route_question(question)
        context = retrieve_ask_context(
            route=route,
            question=question,
            language=language,
            db=db,
            clip_id=clip_id,
            incident_id=incident_id,
            country=country,
        )
    except Exception as exc:
        print(f"[ask-rag] retrieval failed: {exc}")
        if route_question(question) == "history_memory":
            return AskResult(
                answer=format_history_answer([], language, question),
                ask_mode="grounded",
                selected_route="history_memory",
                retrieved_context_count=0,
                reason_if_fallback="Stored history retrieval unavailable",
            )
        return _fallback_result(fallback, None, 0, "Ask context retrieval unavailable")

    if context.route == "history_memory":
        history_items = [
            item for item in context.context_items
            if item.get("source") == "stored_history_record"
        ]
        matches = _history_matches_from_context(history_items, db)
        related_ids = tuple(
            str(item["incident_id"]) for item in history_items if item.get("incident_id")
        )
        if not history_items:
            return AskResult(
                answer=format_history_answer([], language, question),
                ask_mode="grounded",
                selected_route=context.route,
                retrieved_context_count=0,
                related_incident_ids=(),
                grounding_label="GROUNDED HISTORY",
            )
        if os.getenv("GUARDIAN_ASK_RAG_ENABLED", "1") == "0":
            return AskResult(
                answer=format_history_answer(matches, language, question),
                ask_mode="grounded",
                selected_route=context.route,
                retrieved_context_count=len(history_items),
                reason_if_fallback="Ask LLM disabled",
                related_incident_ids=related_ids,
                grounding_label="GROUNDED HISTORY",
            )
        try:
            answer = _sanitize_ask_answer(_run_ask_llm(context), context)
            violations = validate_history_answer(answer, context)
            if violations:
                retry_context = AskContext(
                    route=context.route,
                    query=context.query,
                    language=context.language,
                    context_items=context.context_items + [{
                        "source": "grounding_validation_feedback",
                        "read_only": True,
                        "rejected_claims": violations,
                        "instruction": "Correct every rejected claim using stored evidence only.",
                    }],
                    warnings=context.warnings + ["previous_history_answer_rejected"],
                )
                answer = _sanitize_ask_answer(_run_ask_llm(retry_context), retry_context)
                violations = validate_history_answer(answer, retry_context)
            if violations:
                raise ValueError("History grounding validation failed: " + "; ".join(violations))
            return AskResult(
                answer=answer,
                ask_mode="llm",
                selected_route=context.route,
                retrieved_context_count=len(history_items),
                related_incident_ids=related_ids,
                grounding_label="LLM + GROUNDED HISTORY",
            )
        except Exception as exc:
            print(f"[ask-rag] grounded history LLM unavailable/rejected: {exc!r}")
            return AskResult(
                answer=format_history_answer(matches, language, question),
                ask_mode="grounded",
                selected_route=context.route,
                retrieved_context_count=len(history_items),
                reason_if_fallback=_safe_ask_reason(exc),
                related_incident_ids=related_ids,
                grounding_label="GROUNDED HISTORY",
            )

    if os.getenv("GUARDIAN_ASK_RAG_ENABLED", "1") == "0":
        return _fallback_result(fallback, context.route, len(context.context_items), "Ask LLM disabled")

    if not context.context_items:
        print("[ask-rag] no retrieved context; using deterministic fallback")
        return _fallback_result(
            fallback,
            context.route,
            0,
            "No retrieved context available",
        )

    people_answer = _current_people_answer_if_requested(context)
    if people_answer is not None:
        return AskResult(
            answer=people_answer,
            ask_mode="llm",
            selected_route=context.route,
            retrieved_context_count=len(context.context_items),
            reason_if_fallback=None,
            **_ask_vlm_diagnostics(context.context_items),
        )

    gate_answer = _current_gate_answer_if_requested(context)
    if gate_answer is not None:
        return AskResult(
            answer=gate_answer,
            ask_mode="llm",
            selected_route=context.route,
            retrieved_context_count=len(context.context_items),
            reason_if_fallback=None,
            **_ask_vlm_diagnostics(context.context_items),
        )

    try:
        return AskResult(
            answer=_sanitize_ask_answer(_run_ask_llm(context), context),
            ask_mode="llm",
            selected_route=context.route,
            retrieved_context_count=len(context.context_items),
            reason_if_fallback=None,
            **_ask_vlm_diagnostics(context.context_items),
        )
    except Exception as exc:
        print(f"[ask-rag] LLM unavailable; using deterministic fallback: {exc}")
        return _fallback_result(
            fallback,
            context.route,
            len(context.context_items),
            _safe_ask_reason(exc),
            context.context_items,
        )


def _fallback_result(
    fallback: Callable[[], str],
    route: AskRoute | None,
    context_count: int,
    reason: str,
    context_items: list[dict[str, Any]] | None = None,
) -> AskResult:
    if route == "legal_rag":
        legal_answer = _deterministic_legal_answer_from_context(context_items or [])
        if legal_answer:
            return AskResult(
                answer=legal_answer,
                ask_mode="fallback",
                selected_route=route,
                retrieved_context_count=context_count,
                reason_if_fallback=reason,
                **_ask_vlm_diagnostics(context_items or []),
            )

    return AskResult(
        answer=fallback(),
        ask_mode="fallback",
        selected_route=route,
        retrieved_context_count=context_count,
        reason_if_fallback=reason,
        **_ask_vlm_diagnostics(context_items or []),
    )


def _deterministic_legal_answer_from_context(context_items: list[dict[str, Any]]) -> str:
    for item in context_items:
        if item.get("source") != "legal_rag":
            continue
        legal_output = item.get("legal_output") or {}
        summary = str(legal_output.get("summary") or "").strip()
        if summary:
            return summary
        references = legal_output.get("retrieved_legal_references") or []
        if not references:
            return ""
        primary = references[0]
        country = legal_output.get("country") or item.get("country") or "selected country"
        act = primary.get("section_title") or primary.get("violence_category") or "violent act"
        law = primary.get("law_title") or "retrieved legal reference"
        source_url = primary.get("source_url") or "source unavailable"
        article = primary.get("article_number") or "article-level citation unavailable"
        return (
            f"Country: {country}\n"
            f"Relevant act: {act}\n"
            "Possible legal consequence: Possible consequences depend on the cited legal context, "
            "confirmed facts, and competent authority review.\n"
            f"Relevant law/article/section: {law}; {article}\n"
            f"Source: {source_url}\n"
            "Limitation: Legal context only, not legal advice. Do not infer guilt, exact penalty, "
            "or court outcome from Guardian Eye output."
        )
    return ""


def _safe_ask_reason(exc: Exception) -> str:
    text = str(exc).strip()
    if "local_files_only" in text or "not the path to a directory" in text:
        return "Ask LLM model unavailable locally"
    if "out of memory" in text.lower():
        return "Ask LLM ran out of GPU memory"
    return "Ask LLM unavailable"


def route_question(question: str) -> AskRoute:
    q = question.lower()
    if _contains_any(
        q,
        [
            "legal",
            "law",
            "consequence",
            "penalty",
            "punishment",
            "court",
            "liability",
            "what happens",
            "assault",
            "battery",
            "domestic violence",
            "weapon during violence",
        ],
    ):
        return "legal_rag"
    if _contains_any(q, ["قانون", "عقوب", "محكمة", "مسؤولية"]):
        return "legal_rag"
    if _is_history_question(q):
        return "history_memory"
    if _is_current_incident_question(q):
        return "current_incident"
    if _contains_any(q, ["what is guardian", "how does", "what should security", "policy", "explain the system"]):
        return "reference_corpus"
    if _contains_any(q, ["ما هو", "كيف يعمل", "ماذا يجب على الأمن"]):
        return "reference_corpus"
    return "current_incident"


def _is_current_incident_question(q_lower: str) -> bool:
    return _contains_any(
        q_lower,
        [
            "current video",
            "this video",
            "current incident",
            "this incident",
            "uploaded video",
            "model stream",
            "stream contributed",
            "streams contributed",
            "gate contribution",
            "gate contributions",
            "skeleton",
            "interaction",
            "object",
            "vit",
            "rgb",
        ],
    )


def _is_history_question(q_lower: str) -> bool:
    if (
        _contains_any(q_lower, ["incident", "incidents", "حادث", "حوادث"])
        and _contains_any(q_lower, ["how many", "count", "number of", "كم عدد", "عدد الحوادث"])
    ):
        return True
    if (
        _contains_any(q_lower, ["حادث", "حوادث", "incident"])
        and _contains_any(q_lower, ["confidence", "الثقة"])
        and _contains_any(q_lower, ["highest", "maximum", "أعلى", "اعلى"])
    ):
        return True
    return _contains_any(
        q_lower,
        [
            "previous incidents",
            "previous incident",
            "previous",
            "old incidents",
            "old incident",
            "history",
            "last week",
            "week ago",
            "yesterday",
            "compare incidents",
            "compare previous",
            "past events",
            "past event",
            "past",
            "similar incident",
            "similar cases",
            "latest incident",
            "latest incidents",
            "most severe incident",
            "سجل",
            "السابق",
            "سابقة",
            "قديم",
            "آخر الحوادث",
            "الحوادث التي تم رصدها",
            "الحوادث العنيفة",
            "الأعلى خطورة",
            "قارن",
            "مقارنة",
            "مشابه",
            "متشابه",
            "مماثل",
            "نمط",
            "أنماط",
            "متكرر",
            "أمس",
            "الاسبوع",
            "الأسبوع",
        ],
    )


def retrieve_ask_context(
    *,
    route: AskRoute,
    question: str,
    language: str,
    db,
    clip_id: str | None,
    incident_id: str | None,
    country: str | None,
) -> AskContext:
    warnings: list[str] = []
    items: list[dict[str, Any]] = []

    if route == "current_incident":
        record = _current_record(db, clip_id, incident_id)
        if record is not None:
            vlm_item = _vlm_summary_context(record)
            if vlm_item:
                items.append(vlm_item)
            items.append(_record_context(record, "current_incident"))
        else:
            warnings.append("current_incident_not_found")
        items.extend(_reference_context(question, language, extra_terms={"current", "incident"}))

    elif route == "history_memory":
        from history_qa_service import (
            history_computation_item,
            history_context_items,
            retrieve_history_matches,
        )

        current_record = _explicit_record(db, clip_id, incident_id)
        matches = retrieve_history_matches(
            db,
            question,
            current_record=current_record,
        )
        items.append(history_computation_item(matches, question))
        items.extend(history_context_items(matches))
        if not matches:
            warnings.append("history_context_not_found")

    elif route == "legal_rag":
        record = _current_record(db, clip_id, incident_id)
        if record is not None:
            vlm_item = _vlm_summary_context(record)
            if vlm_item:
                items.append(vlm_item)
            items.append(_record_context(record, "legal_incident_basis"))
            legal_item = _legal_context(record, country, language, question)
            if legal_item:
                items.append(legal_item)
            else:
                warnings.append("legal_context_unavailable")
        else:
            warnings.append("legal_incident_not_found")
        items.extend(_reference_context(question, language, extra_terms={"legal", "country"}))

    else:
        items.extend(_reference_context(question, language, extra_terms={"system", "guardian", "security"}))

    return AskContext(
        route=route,
        query=question,
        language=language,
        context_items=items if route == "history_memory" else items[:8],
        warnings=warnings,
    )


def build_ask_prompt(context: AskContext) -> list[dict[str, str]]:
    target_language = "Arabic" if context.language == "ar" else "English"
    language_guidance = ARABIC_ONLY_INSTRUCTION if context.language == "ar" else ""
    context_text = json.dumps(
        {
            "route": context.route,
            "query": context.query,
            "warnings": context.warnings,
            "retrieved_context": context.context_items,
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    history_guidance = (
        "For history questions, treat grounded_history_computation as the read-only "
        "answer to ranking/counting and use stored_history_record items as the only "
        "incident evidence. Do not invent incidents, IDs, filenames, timestamps, "
        "confidence values, verdicts, people, or dates. Copy stored values exactly. "
        "If a field is null, say it is unavailable. Never say 'incident number' or "
        "'حادث رقم' unless the exact stored incident_id immediately follows it. "
        "Do not use general knowledge, personal history, or model memory. "
        if context.route == "history_memory"
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You answer questions for Guardian Eye using retrieved context only. "
                "Do not invent unavailable information. If context is incomplete, say so. "
                "Do not reclassify the video, identify attacker/victim roles, assign guilt, "
                "or predict court outcomes. For current-video questions, use "
                "Guardian Eye visual analysis first when it is present, then telemetry, "
                "then memory/reference/legal context. For stream, gate, contribution, "
                "skeleton, interaction, object, ViT, or RGB questions, answer from gate "
                "telemetry percentages before visual scene content. Do not say VLM summary, "
                "current_vlm_summary, summary_source, or retrieved VLM context to the user. "
                "Do not use markdown headings, bullets, bold markers, or raw section labels. "
                f"{history_guidance}"
                f"{language_guidance}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Answer in {target_language}.\n\n"
                f"QUESTION:\n{context.query}\n\n"
                f"RETRIEVED CONTEXT ONLY:\n{context_text}\n\n"
                "Give a concise answer. Mention limitations when the retrieved context "
                f"does not contain enough information. {language_guidance}"
            ),
        },
    ]


def _run_ask_llm(context: AskContext) -> str:
    model_id = os.getenv("GUARDIAN_ASK_LLM_MODEL_ID", DEFAULT_ASK_LLM_MODEL)
    local_only = os.getenv("GUARDIAN_MODEL_LOCAL_ONLY", "1") == "1"
    model = None
    tokenizer = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_only,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_only,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()

        messages = build_ask_prompt(context)

        def generate_once(prompt_messages: list[dict[str, str]]) -> str:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in prompt_messages)
                text += "\n\nassistant:"

            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=260,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
            return tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        answer = generate_once(messages)
        if not answer:
            raise RuntimeError("empty LLM answer")
        try:
            return polish_generated_text(answer, context.language, label="ask answer")
        except ValueError as exc:
            if context.language != "ar":
                raise
            print(f"[ask-rag] arabic_validation_retry reason={exc}")
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Rewrite the answer in Modern Standard Arabic only. "
                        "Remove any Chinese or mixed-language text. Keep the same retrieved facts."
                    ),
                },
            ]
            retry_answer = generate_once(retry_messages)
            if not retry_answer:
                raise RuntimeError("empty LLM answer after Arabic retry")
            return polish_generated_text(retry_answer, context.language, label="ask answer")
    finally:
        if model is not None:
            del model
        gc.collect()
        _empty_cuda_cache()
        del tokenizer
        gc.collect()


def _current_gate_answer_if_requested(context: AskContext) -> str | None:
    if context.route != "current_incident" or not _is_stream_question(context.query.lower()):
        return None
    for item in context.context_items:
        gate = item.get("gate")
        if isinstance(gate, dict) and gate:
            return _format_gate_answer(gate, context.language)
    return None


def _current_people_answer_if_requested(context: AskContext) -> str | None:
    if context.route != "current_incident" or not _is_people_question(context.query.lower()):
        return None
    count: int | None = None
    for item in context.context_items:
        if item.get("source") == "current_vlm_summary":
            summary = item.get("vlm_summary") or {}
            value = summary.get("people_count")
            if isinstance(value, int) and value > 0:
                count = value
                break
    if count is None:
        for item in context.context_items:
            value = item.get("people_count")
            if isinstance(value, int) and value > 0:
                count = value
                break
    if count is None:
        return (
            "لا يمكن تأكيد عدد الأشخاص الظاهرين من تحليل الحادثة الحالي."
            if context.language == "ar"
            else "Guardian Eye cannot confirm the number of visible people from the current incident analysis."
        )
    if context.language == "ar":
        return f"رصد Guardian Eye عدد {count} من الأشخاص الظاهرين في الحادثة، لكن لا يمكن تأكيد الأدوار الفردية."
    noun = "person" if count == 1 else "people"
    return f"Guardian Eye detected {count} {noun} in the incident."


def _is_people_question(q_lower: str) -> bool:
    return _contains_any(
        q_lower,
        [
            "how many",
            "people",
            "person",
            "involved",
            "number of people",
            "count",
            "كم",
            "عدد",
            "أشخاص",
            "اشخاص",
        ],
    )


def _is_stream_question(q_lower: str) -> bool:
    return _contains_any(
        q_lower,
        [
            "stream",
            "gate",
            "contribution",
            "contributed most",
            "skeleton",
            "interaction",
            "object",
            "vit",
            "rgb",
            "تيار",
            "بوابة",
            "مساهمة",
            "ساهم",
            "الأقوى",
            "اقوى",
        ],
    )


def _sanitize_ask_answer(answer: str, context: AskContext) -> str:
    clean = str(answer or "")
    replacements = {
        "VLM observations": "visual analysis",
        "VLM observation": "visual analysis",
        "VLM summary": "visual analysis",
        "current_vlm_summary": "visual analysis",
        "retrieved VLM context": "current incident analysis",
        "selected evidence frames": "analyzed video",
        "selected frame": "analyzed video",
        "selected frames": "analyzed video",
        "retrieved context": "current incident analysis",
        "retrieved_context": "current incident analysis",
        "legal responsibility": "responsibility",
    }
    for old, new in replacements.items():
        clean = clean.replace(old, new)
    if context.route != "legal_rag":
        clean = clean.replace(" Responsibility remains for human review.", "")
        clean = clean.replace(" responsibility remains for human review.", "")
    return clean.strip()


def validate_history_answer(answer: str, context: AskContext) -> list[str]:
    """Reject unsupported identifiers, filenames, and confidence percentages."""
    if context.route != "history_memory":
        return []
    records = [
        item for item in context.context_items
        if item.get("source") == "stored_history_record"
    ]
    allowed_ids = {str(item.get("incident_id")) for item in records if item.get("incident_id")}
    allowed_filenames = {str(item.get("filename")) for item in records if item.get("filename")}
    violations: list[str] = []

    numbered_claims = re.findall(
        r"(?:حادث\s+رقم|incident\s+number)\s*[:#-]?\s*([A-Za-z0-9_-]+)",
        answer,
        flags=re.IGNORECASE,
    )
    for claimed in numbered_claims:
        if claimed not in allowed_ids:
            violations.append(f"unsupported incident number/id: {claimed}")

    mentioned_files = set(
        re.findall(r"[^\s،,:;]+\.(?:avi|mp4|mov|mkv|webm)", answer, flags=re.IGNORECASE)
    )
    matched_allowed_files = {filename for filename in allowed_filenames if filename in answer}
    for filename in mentioned_files:
        normalized_filename = filename
        if filename not in allowed_filenames and filename.startswith("و"):
            normalized_filename = filename[1:]
        is_suffix_of_visible_allowed_file = any(
            allowed.endswith(normalized_filename) and allowed in answer
            for allowed in allowed_filenames
        )
        if normalized_filename not in allowed_filenames and not is_suffix_of_visible_allowed_file:
            violations.append(f"unsupported filename: {filename}")
    if records and not matched_allowed_files:
        violations.append("answer does not cite a retrieved filename")

    mentioned_ids = set(
        re.findall(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            answer,
            flags=re.IGNORECASE,
        )
    )
    for incident_id in mentioned_ids:
        if incident_id not in allowed_ids:
            violations.append(f"unsupported incident id: {incident_id}")

    allowed_percentages: list[float] = []
    allowed_raw: list[float] = []
    for item in records:
        for field in ("confidence", "similarity"):
            value = item.get(field)
            if isinstance(value, (int, float)):
                allowed_raw.append(float(value))
                allowed_percentages.append(float(value) * 100.0)
    for token in re.findall(r"([0-9٠-٩]+(?:[.,][0-9٠-٩]+)?)\s*[%٪]", answer):
        value = _localized_number(token)
        if value is not None and not any(abs(value - allowed) <= 0.11 for allowed in allowed_percentages):
            violations.append(f"unsupported percentage: {token}%")
    for token in re.findall(
        r"(?:confidence|نسبة\s+الثقة|الثقة)\s*(?:هي|:|=|بلغت|تبلغ)?\s*([0-9٠-٩]+(?:[.,][0-9٠-٩]+)?)",
        answer,
        flags=re.IGNORECASE,
    ):
        value = _localized_number(token)
        if value is None:
            continue
        allowed = allowed_raw if value <= 1.0 else allowed_percentages
        tolerance = 0.001 if value <= 1.0 else 0.11
        if not any(abs(value - candidate) <= tolerance for candidate in allowed):
            violations.append(f"unsupported confidence value: {token}")
    return sorted(set(violations))


def _localized_number(value: str) -> float | None:
    translation = str.maketrans("٠١٢٣٤٥٦٧٨٩,", "0123456789.")
    try:
        return float(value.translate(translation))
    except (TypeError, ValueError):
        return None


def _format_gate_answer(gate: dict[str, Any], language: str) -> str:
    ordered = sorted(
        ((str(name), float(value or 0.0)) for name, value in gate.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ordered:
        return (
            "لا تتوفر أوزان مساهمة المسارات لهذا الحادث."
            if language == "ar"
            else "Gate contribution telemetry is unavailable for this incident."
        )
    if language == "ar":
        parts = "، ".join(f"{_gate_label_ar(name)} {value:.0%}" for name, value in ordered)
        strongest = ordered[0]
        raw_note = " وتظهر نسبة ViT/RGB كنسبة بوابة خام عند عدم توفر التضمين."
        return f"أعلى مساهمة كانت من {_gate_label_ar(strongest[0])} بنسبة {strongest[1]:.0%}. أوزان المسارات: {parts}.{raw_note}"

    parts = ", ".join(f"{_gate_label_en(name)} {value:.0%}" for name, value in ordered)
    strongest = ordered[0]
    raw_note = " ViT/RGB is shown as a raw gate percentage if its embedding is unavailable."
    return f"The {_gate_label_en(strongest[0])} stream contributed most with {strongest[1]:.0%}. Full gate weights: {parts}.{raw_note}"


def _gate_label_en(name: str) -> str:
    labels = {
        "vit": "ViT/RGB",
        "video": "ViT/RGB",
        "skeleton": "Skeleton",
        "interaction": "Interaction",
        "object": "Object",
    }
    return labels.get(name, name)


def _gate_label_ar(name: str) -> str:
    labels = {
        "vit": "ViT/RGB",
        "video": "ViT/RGB",
        "skeleton": "الهيكل",
        "interaction": "التفاعل",
        "object": "الأجسام",
    }
    return labels.get(name, name)


def _current_record(db, clip_id: str | None, incident_id: str | None):
    from incident_service import get_by_clip, get_incident, query_incidents

    record = get_incident(db, incident_id) if incident_id else None
    if record is None and clip_id:
        record = get_by_clip(db, clip_id)
    if record is None:
        _, latest = query_incidents(db, limit=1)
        record = latest[0] if latest else None
    return record


def _explicit_record(db, clip_id: str | None, incident_id: str | None):
    from incident_service import get_by_clip, get_incident

    record = get_incident(db, incident_id) if incident_id else None
    if record is None and clip_id:
        record = get_by_clip(db, clip_id)
    return record


def _history_records(db, question: str):
    from incident_service import query_incidents

    q = question.lower()
    verdict = None
    weapon = None
    from_ts = None
    to_ts = None
    now = _dt.datetime.utcnow()
    if _contains_any(q, ["fight", "violen", "attack", "assault", "عنف", "عنيف"]):
        verdict = "violence"
    elif _contains_any(q, ["peaceful", "normal", "calm", "non-viol", "هادئ", "غير عنيف"]):
        verdict = "non-violence"
    if _contains_any(q, ["weapon", "object", "bottle", "knife", "سلاح", "جسم"]):
        weapon = True
    if _contains_any(q, ["week ago", "last week", "الأسبوع", "الاسبوع"]):
        from_ts = now - _dt.timedelta(days=10)
        to_ts = now - _dt.timedelta(days=5)
    elif _contains_any(q, ["yesterday", "أمس"]):
        from_ts = now - _dt.timedelta(days=2)
        to_ts = now - _dt.timedelta(days=1)
    elif _contains_any(q, ["today", "اليوم"]):
        from_ts = now - _dt.timedelta(hours=24)

    _, records = query_incidents(
        db,
        verdict=verdict,
        weapon=weapon,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=4,
    )
    return records


def _incident_memory_context(
    db,
    question: str,
    *,
    current_record,
    language: str,
) -> dict[str, Any] | None:
    try:
        from incident_memory_service import memory_context_item

        return memory_context_item(
            db,
            question,
            current_record=current_record,
            language=language,
        )
    except Exception as exc:
        print(f"[ask-rag] incident memory retrieval failed: {exc}")
        return None


def _record_context(record, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "incident_id": record.incident_id,
        "clip_id": record.clip_id,
        "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        "camera_or_source": record.source,
        "verdict": record.verdict,
        "confidence": record.confidence,
        "threshold": record.threshold,
        "gate": record.gate_dict(),
        "gqs": record.gqs_dict(),
        "people_count": record.people_count,
        "peak_window": record.peak_window(),
        "weapon_flag": record.weapon_flag,
        "weapon_class": record.weapon_class,
        "packet_summary": record.packet_summary,
        "narrative": record.narrative,
        "vlm_summary": record.vlm_summary(),
        "grounding_type": "saved_incident_record",
    }


def _history_matches_from_context(items: list[dict[str, Any]], db):
    """Reattach ORM rows while preserving deterministic retrieval order/reasons."""
    from history_qa_service import HistoryMatch
    from incident_service import get_incident

    matches = []
    for item in items:
        record = get_incident(db, str(item.get("incident_id") or ""))
        if record is None:
            continue
        matches.append(
            HistoryMatch(
                record=record,
                similarity=item.get("similarity"),
                reason_en=str(item.get("match_reason_en") or "stored record matched the query"),
                reason_ar=str(item.get("match_reason_ar") or "السجل المحفوظ طابق السؤال"),
            )
        )
    return matches


def _vlm_summary_context(record) -> dict[str, Any] | None:
    summary = record.vlm_summary()
    if not summary:
        return None
    return {
        "source": "current_vlm_summary",
        "summary_source": summary.get("summary_source") or "current_vlm",
        "incident_id": record.incident_id,
        "clip_id": record.clip_id,
        "vlm_summary": summary,
        "grounding_type": "vlm_generated_incident_understanding",
        "priority": 1,
    }


def _legal_context(record, country: str | None, language: str, question: str) -> dict[str, Any] | None:
    try:
        from schemas import GateWeights, GQSScores, PredictResponse, Telemetry, WeaponInfo
        from services.rag_adapter import build_legal_response

        pred = PredictResponse(
            clip_id=record.clip_id,
            verdict=record.verdict,
            confidence=record.confidence,
            threshold=record.threshold or 0.0,
            gate=GateWeights(**record.gate_dict()),
            gqs=GQSScores(**record.gqs_dict()),
            telemetry=Telemetry(
                people=record.people_count,
                peak_window=record.peak_window(),
                weapon=WeaponInfo(flag=record.weapon_flag, cls=record.weapon_class),
            ),
        )
        legal_output, legal_scores = build_legal_response(
            pred=pred,
            packet_summary=f"{question}\n{record.packet_summary or ''}".strip(),
            narrative=f"{question}\n{record.narrative or ''}".strip(),
            vlm_summary=record.vlm_summary(),
            country=country,
            language=language,
        )
        if hasattr(legal_output, "model_dump"):
            legal_output = legal_output.model_dump()
        return {
            "source": "legal_rag",
            "country": country,
            "legal_output": legal_output,
            "legal_scores": legal_scores.model_dump() if hasattr(legal_scores, "model_dump") else legal_scores,
            "grounding_type": "retrieved_legal_context",
        }
    except Exception as exc:
        print(f"[ask-rag] legal retrieval failed: {exc}")
        return None


def _reference_context(
    question: str,
    language: str,
    *,
    extra_terms: set[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    terms = set(question.lower().replace("-", " ").split()).union(extra_terms)
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in REFERENCE_CORPUS:
        score = len(terms.intersection(item["tags"]))
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    text_key = "text_ar" if language == "ar" else "text_en"
    return [
        {
            "source": "reference_corpus",
            "id": item["id"],
            "title": item["title"],
            "text": item[text_key],
            "grounding_type": "reference_guidance_not_incident_fact",
        }
        for _, item in scored[:top_k]
    ]


def _ask_vlm_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    for item in items:
        if item.get("source") == "current_vlm_summary":
            summary = item.get("vlm_summary") or {}
            return {
                "vlm_summary_used": True,
                "summary_source": str(summary.get("summary_source") or "current_vlm"),
                "vlm_people_count": summary.get("people_count"),
                "vlm_violence_type": summary.get("violence_type"),
            }
    return {
        "vlm_summary_used": False,
        "summary_source": None,
        "vlm_people_count": None,
        "vlm_violence_type": None,
    }


def _empty_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)
