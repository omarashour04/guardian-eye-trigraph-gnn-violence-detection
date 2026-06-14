from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from schemas import PredictResponse


logger = logging.getLogger("guardian_eye.rag")

# Real Legal RAG enablement:
#   GUARDIAN_RAG_MODE=real
#   GUARDIAN_RAG_PIPELINE_PATH=D:\sara\graduation\Demo Implementation\FULL_RAG_Pipeline
#   GUARDIAN_LEGAL_INDEX_PATH=D:\...\FULL_RAG_Pipeline\data\legal_faiss.index
#   GUARDIAN_LEGAL_METADATA_DB_PATH=D:\...\FULL_RAG_Pipeline\data\legal_metadata.db
#
# GUARDIAN_RAG_MODE=auto attempts real Legal RAG only when both legal index
# paths are configured and exist. GUARDIAN_RAG_MODE=mock is explicit mock-only.
RAG_MODE = os.getenv("GUARDIAN_RAG_MODE", "auto").strip().lower()
DEFAULT_RAG_PIPELINE_PATH = Path(__file__).resolve().parents[2] / "FULL_RAG_Pipeline"
RAG_PIPELINE_PATH = Path(
    os.getenv("GUARDIAN_RAG_PIPELINE_PATH", str(DEFAULT_RAG_PIPELINE_PATH))
)
LEGAL_INDEX_PATH = os.getenv("GUARDIAN_LEGAL_INDEX_PATH", "").strip()
LEGAL_METADATA_DB_PATH = os.getenv("GUARDIAN_LEGAL_METADATA_DB_PATH", "").strip()

LEGAL_LIMITATIONS_NOTE = (
    "This is not legal advice, does not determine guilt, and does not predict court outcome."
)
INDEX_NOT_CONFIGURED_WARNING = (
    "Legal RAG is running in fallback mode because the legal index or metadata "
    "database is not configured."
)


def build_full_rag_response(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    country: str | None,
    language: str,
) -> dict[str, Any]:
    explanation_rag = _mock_explanation_rag(pred, packet_summary)
    incident_memory_rag = _mock_incident_memory_rag(pred)
    legal_consequences_rag, legal_scores = build_legal_response(
        pred=pred,
        packet_summary=packet_summary,
        narrative=narrative,
        country=country,
        language=language,
    )

    return {
        "explanation_rag": explanation_rag,
        "incident_memory_rag": incident_memory_rag,
        "legal_consequences_rag": legal_consequences_rag,
        "legal_scores": legal_scores,
    }


def build_legal_response(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    country: str | None,
    language: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    clean_country = (country or "").strip()
    if not clean_country:
        _log_legal_mode("fallback", "country is missing")
        return _missing_country_legal_output(pred, language), None

    if RAG_MODE == "mock":
        _log_legal_mode("mock", "GUARDIAN_RAG_MODE=mock")
        return _mock_legal_output(pred, clean_country, language), _mock_legal_scores()

    index_warning = _legal_index_warning()
    if index_warning:
        _log_legal_mode("fallback", index_warning)
        return _fallback_legal_output(pred, clean_country, index_warning, language), None

    try:
        legal_output, legal_scores = _real_legal_output(
            pred=pred,
            packet_summary=packet_summary,
            narrative=narrative,
            country=clean_country,
            language=language,
        )
        _log_legal_mode("real", f"country={clean_country}")
        return legal_output, legal_scores
    except Exception as exc:
        warning = (
            "Legal RAG is running in fallback mode because the real Legal RAG "
            f"path failed: {exc.__class__.__name__}: {exc}"
        )
        _log_legal_mode("fallback", warning)
        return _fallback_legal_output(pred, clean_country, warning, language), None


def _mock_explanation_rag(
    pred: PredictResponse,
    packet_summary: str,
) -> dict[str, Any]:
    return {
        "status": "mocked",
        "verdict": pred.verdict,
        "confidence": pred.confidence,
        "explanation": (
            "The mock explanation RAG summarizes why the classifier produced this "
            "result while preserving the original verdict and confidence."
        ),
        "evidence_basis": _evidence_basis(pred, packet_summary),
        "limitations": "This mock explanation is deterministic and does not use a heavy model.",
    }


def _mock_incident_memory_rag(pred: PredictResponse) -> dict[str, Any]:
    return {
        "status": "mocked",
        "query_basis": _legal_query_basis(pred),
        "similar_incidents": [
            {
                "incident_id": "mock-memory-001",
                "summary": "Prior mock incident with a similar classification pattern.",
                "similarity": 0.82,
            },
            {
                "incident_id": "mock-memory-002",
                "summary": "Prior mock incident involving a confrontation classified as violence.",
                "similarity": 0.76,
            },
        ],
        "memory_note": "This mock memory output does not query a real vector store.",
    }


def _mock_legal_output(
    pred: PredictResponse,
    country: str,
    language: str,
) -> dict[str, Any]:
    return _finalize_legal_output(
        {
            "country": country,
            "query_basis": _legal_query_basis(pred),
            "retrieved_legal_references": [
                {
                    "law_title": f"{country} MOCK legal placeholder",
                    "article_number": "MOCK",
                    "section_title": "Mock-only demo reference",
                    "source_url": "https://example.com/mock-legal-reference",
                    "snippet": (
                        "Mock-only placeholder reference. This was not retrieved "
                        "from a real country-specific legal index."
                    ),
                    "score": 0.0,
                    "country": country,
                    "violence_category": "mock_only",
                    "official_source": False,
                }
            ],
            "summary": (
                "Mock-only Legal RAG output: no real country-specific legal index "
                "was queried. Use this only to verify the UI shape, not as grounded "
                "legal retrieval."
            ),
            "guardrail_status": "needs_review",
            "limitations_note": LEGAL_LIMITATIONS_NOTE,
        },
        source="mock",
        warning="GUARDIAN_RAG_MODE=mock; real Legal RAG was not used.",
        language=language,
    )


def _missing_country_legal_output(
    pred: PredictResponse,
    language: str,
) -> dict[str, Any]:
    return _finalize_legal_output(
        {
            "country": "",
            "query_basis": _legal_query_basis(pred),
            "retrieved_legal_references": [],
            "summary": "Select a country to request possible legal consequences.",
            "guardrail_status": "needs_review",
            "limitations_note": LEGAL_LIMITATIONS_NOTE,
        },
        source="fallback",
        warning="Country is required before Legal RAG can retrieve references.",
        legacy_warning="country_required",
        language=language,
    )


def _fallback_legal_output(
    pred: PredictResponse,
    country: str,
    warning: str,
    language: str,
) -> dict[str, Any]:
    return _finalize_legal_output(
        {
            "country": country,
            "query_basis": _legal_query_basis(pred),
            "retrieved_legal_references": [],
            "summary": warning,
            "guardrail_status": "needs_review",
            "limitations_note": LEGAL_LIMITATIONS_NOTE,
        },
        source="fallback",
        warning=warning,
        language=language,
    )


def _real_legal_output(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    country: str,
    language: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _ensure_pipeline_path()
    from rag_service.legal_index import load_legal_index
    from rag_service.legal_orchestrator import generate_legal_consequences

    legal_index = load_legal_index(LEGAL_INDEX_PATH, LEGAL_METADATA_DB_PATH)
    result = generate_legal_consequences(
        {
            "country": country,
            "verdict": pred.verdict,
            "confidence": pred.confidence,
            "packet_summary": packet_summary,
            "narrative": narrative,
            "weapon_flag": pred.telemetry.weapon.flag,
            "weapon_class": pred.telemetry.weapon.cls,
            "language": language,
        },
        legal_index=legal_index,
    )

    legal_output = dict(result.legal_consequences)
    return (
        _finalize_legal_output(
            legal_output,
            source="real",
            warning=None,
            language=language,
        ),
        _compact_scores(result.evaluation),
    )


def _finalize_legal_output(
    output: dict[str, Any],
    *,
    source: str,
    warning: str | None,
    language: str,
    legacy_warning: str | None = None,
) -> dict[str, Any]:
    final_output = dict(output)
    final_warning = warning
    if _is_arabic_language(language):
        final_output, translation_warning = _translate_legal_output_to_arabic(final_output)
        final_warning = final_warning or translation_warning
        if final_warning:
            final_warning = _fallback_arabic_warning(final_warning)

    return _with_legal_debug_fields(
        final_output,
        source=source,
        warning=final_warning,
        legacy_warning=legacy_warning,
    )


def _is_arabic_language(language: str) -> bool:
    return str(language or "").casefold().startswith("ar")


def _translate_legal_output_to_arabic(
    output: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    translated = _strip_arabic_translation_note(dict(output))
    if os.getenv("GUARDIAN_LEGAL_AR_TRANSLATION", "auto").strip().lower() == "fallback":
        return _fallback_translate_legal_output_to_arabic(translated), None

    try:
        _ensure_pipeline_path()
        from translation_service import translation_session
    except Exception:
        return _fallback_translate_legal_output_to_arabic(translated), None

    try:
        with translation_session() as translator:
            for key in ("summary", "limitations_note"):
                value = translated.get(key)
                if isinstance(value, str):
                    translated[key] = translator.translate(value, "English", "Arabic")

            references = []
            for reference in translated.get("retrieved_legal_references", []):
                item = dict(reference)
                for key in ("law_title", "section_title", "snippet"):
                    value = item.get(key)
                    if isinstance(value, str):
                        item[key] = translator.translate(value, "English", "Arabic")
                references.append(item)
            translated["retrieved_legal_references"] = references
    except Exception:
        return _fallback_translate_legal_output_to_arabic(translated), None
    return translated, None


def _strip_arabic_translation_note(output: dict[str, Any]) -> dict[str, Any]:
    note = (
        " Arabic translation is not performed in this summarizer; the existing "
        "TranslateGemma wrapper can translate the final legal_consequences output later."
    )
    limitations = output.get("limitations_note")
    if isinstance(limitations, str):
        output["limitations_note"] = limitations.replace(note, "").strip()
    return output


def _fallback_translate_legal_output_to_arabic(output: dict[str, Any]) -> dict[str, Any]:
    translated = dict(output)
    translated["summary"] = _fallback_arabic_legal_summary(translated)
    translated["limitations_note"] = (
        "هذه ليست استشارة قانونية، ولا تحدد الإدانة، ولا تتنبأ بنتيجة المحكمة."
    )

    references = []
    for reference in translated.get("retrieved_legal_references", []):
        item = dict(reference)
        item["law_title"] = _fallback_arabic_law_title(str(item.get("law_title") or ""))
        section_title = item.get("section_title")
        if isinstance(section_title, str):
            item["section_title"] = _fallback_arabic_section_title(section_title)
        snippet = item.get("snippet")
        if isinstance(snippet, str):
            item["snippet"] = _fallback_arabic_snippet(snippet)
        references.append(item)
    translated["retrieved_legal_references"] = references
    return translated


def _fallback_arabic_legal_summary(output: dict[str, Any]) -> str:
    references = output.get("retrieved_legal_references") or []
    country = output.get("country") or "الدولة المحددة"
    if not references:
        return (
            "لا توجد مراجع قانونية متاحة للولاية القضائية المحددة. لا يمكن إنتاج "
            "ملخص قانوني موثوق دون مراجع مسترجعة خاصة بالدولة المختارة."
        )

    first = references[0]
    law_title = _fallback_arabic_law_title(str(first.get("law_title") or "المرجع القانوني"))
    article = first.get("article_number")
    article_phrase = f" {article}" if article else ""
    return (
        f"استنادا إلى المراجع القانونية المسترجعة الخاصة بـ {country}، بما في ذلك "
        f"{law_title}{article_phrase}، قد تكون الواقعة المبلغ عنها خاضعة لأحكام "
        "قانونية مرتبطة بالعنف أو الإيذاء أو استخدام أداة خطرة. وتشمل العواقب "
        "القانونية المحتملة عقوبات أو تدابير حماية أو نتائج أخرى تحددها الجهات "
        "المختصة وفقا للوقائع المثبتة والإجراءات القانونية."
    )


def _fallback_arabic_law_title(title: str) -> str:
    lowered = title.casefold()
    if "uae" in lowered:
        return "مرجع قانوني تجريبي لدولة الإمارات"
    if "ksa" in lowered:
        return "مرجع قانوني تجريبي للمملكة العربية السعودية"
    if "canada" in lowered:
        return "القانون الجنائي الكندي"
    if "public order" in lowered:
        return "قانون النظام العام"
    if "criminal justice" in lowered:
        return "قانون العدالة الجنائية"
    if "crime and disorder" in lowered:
        return "قانون الجريمة والاضطراب"
    if "offences against the person" in lowered:
        return "قانون الجرائم ضد الأشخاص"
    return title


def _fallback_arabic_section_title(section_title: str) -> str:
    lowered = section_title.casefold()
    if "weapon" in lowered or "dangerous object" in lowered:
        return "سياق السلاح أو الأداة الخطرة"
    if "assault" in lowered or "physical harm" in lowered or "abuse" in lowered:
        return "الاعتداء أو الإيذاء الجسدي"
    if "protective" in lowered or "aggravating" in lowered:
        return "السياق الوقائي أو المشدد"
    if "demo legal-source fixture" in lowered:
        return "مرجع قانوني تجريبي"
    return section_title


def _fallback_arabic_snippet(snippet: str) -> str:
    lowered = snippet.casefold()
    if "weapon" in lowered or "dangerous object" in lowered:
        return (
            "يشير هذا المقتطف إلى أن استخدام سلاح أو زجاجة أو أداة خطرة في واقعة "
            "عنيفة مزعومة قد يكون عاملا ذا صلة عند تقييم العواقب القانونية المحتملة."
        )
    if "assault" in lowered or "physical harm" in lowered or "abuse" in lowered:
        return (
            "يشير هذا المقتطف إلى أن الاعتداء أو الإيذاء الجسدي أو التهديد قد "
            "يؤدي إلى عواقب قانونية محتملة، مع بقاء النتيجة رهنا بتقدير الجهات "
            "المختصة والوقائع المثبتة."
        )
    if "not official legal advice" in lowered:
        return (
            "هذا مقتطف من مرجع تجريبي مخصص للعرض وليس بديلا عن النص القانوني "
            "الرسمي أو الاستشارة القانونية."
        )
    return "مقتطف قانوني مسترجع خاص بالولاية القضائية المحددة."


def _fallback_arabic_warning(warning: str) -> str:
    lowered = warning.casefold()
    if "country is required" in lowered:
        return "يجب اختيار دولة قبل استرجاع المراجع القانونية."
    if "mock" in lowered:
        return "وضع المحاكاة مفعل؛ لم يتم استخدام Legal RAG الحقيقي."
    if "index" in lowered or "metadata" in lowered:
        return "يعمل Legal RAG في وضع احتياطي لأن الفهرس القانوني أو قاعدة بيانات البيانات الوصفية غير مهيأة."
    if "failed" in lowered:
        return "يعمل Legal RAG في وضع احتياطي لأن مسار Legal RAG الحقيقي تعذر تشغيله."
    return "تعذر إكمال مسار Legal RAG الحقيقي، وتم استخدام إخراج احتياطي."


def _with_legal_debug_fields(
    output: dict[str, Any],
    *,
    source: str,
    warning: str | None,
    legacy_warning: str | None = None,
) -> dict[str, Any]:
    enriched = dict(output)
    enriched["rag_mode"] = RAG_MODE
    enriched["legal_rag_source"] = source
    enriched["legal_rag_warning"] = warning
    enriched["warning"] = legacy_warning or warning
    return enriched


def _legal_index_warning() -> str | None:
    if not LEGAL_INDEX_PATH or not LEGAL_METADATA_DB_PATH:
        return INDEX_NOT_CONFIGURED_WARNING

    missing_paths = [
        path
        for path in (LEGAL_INDEX_PATH, LEGAL_METADATA_DB_PATH)
        if not Path(path).exists()
    ]
    if missing_paths:
        return (
            "Legal RAG is running in fallback mode because the legal index or "
            f"metadata database does not exist: {', '.join(missing_paths)}"
        )
    return None


def _log_legal_mode(source: str, detail: str) -> None:
    message = (
        f"Legal RAG source={source} mode={RAG_MODE} "
        f"pipeline={RAG_PIPELINE_PATH} index={LEGAL_INDEX_PATH or '<unset>'} "
        f"metadata={LEGAL_METADATA_DB_PATH or '<unset>'} detail={detail}"
    )
    if source == "real":
        logger.info(message)
    else:
        logger.warning(message)


def _ensure_pipeline_path() -> None:
    pipeline_path = str(RAG_PIPELINE_PATH)
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)


def _compact_scores(scores: dict[str, Any] | None) -> dict[str, Any] | None:
    if not scores:
        return None
    return {
        "retrieval_score": scores.get("retrieval_score"),
        "generation_score": scores.get("generation_score"),
        "overall_score": scores.get("overall_score"),
        "passed": scores.get("passed"),
    }


def _mock_legal_scores() -> dict[str, Any]:
    return {
        "retrieval_score": None,
        "generation_score": None,
        "overall_score": None,
        "passed": False,
    }


def _legal_query_basis(pred: PredictResponse) -> dict[str, Any]:
    return {
        "verdict": pred.verdict,
        "weapon_flag": pred.telemetry.weapon.flag,
        "weapon_class": pred.telemetry.weapon.cls,
    }


def _evidence_basis(pred: PredictResponse, packet_summary: str) -> list[str]:
    basis = [
        f"classifier confidence {pred.confidence:.2f}",
        f"verdict {pred.verdict}",
        f"people tracked {pred.telemetry.people}",
    ]
    if pred.telemetry.weapon.flag:
        basis.append(f"object context {pred.telemetry.weapon.cls or 'present'}")
    if packet_summary:
        basis.append("packet summary available")
    return basis
