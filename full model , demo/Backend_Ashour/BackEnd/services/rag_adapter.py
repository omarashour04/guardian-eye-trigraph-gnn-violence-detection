from __future__ import annotations

import logging
import os
import re
import sqlite3
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
RAG_PIPELINE_PATH = None
LEGAL_INDEX_PATH = ""
LEGAL_METADATA_DB_PATH = ""

LEGAL_LIMITATIONS_NOTE = (
    "This is not legal advice, does not determine guilt, and does not predict court outcome."
)
INDEX_NOT_CONFIGURED_WARNING = (
    "Legal RAG is running in fallback mode because the legal index or metadata "
    "database is not configured."
)
SUPPORTED_LEGAL_COUNTRIES = {
    "canada": "Canada",
    "uk": "UK",
    "united kingdom": "UK",
    "great britain": "UK",
    "britain": "UK",
    "england": "UK",
    "usa california": "USA California",
    "us california": "USA California",
    "united states california": "USA California",
    "california": "USA California",
    "uae": "UAE",
    "united arab emirates": "UAE",
    "emirates": "UAE",
    "ksa": "KSA",
    "saudi arabia": "KSA",
    "kingdom of saudi arabia": "KSA",
    "egypt": "Egypt",
}
UNVERIFIED_COUNTRY_WARNING = (
    "Legal RAG has no curated local references for this country. Choose Canada, "
    "UK, USA California, UAE, KSA, or Egypt for grounded curated retrieval."
)
PERSON_OBJECT_CLASSES = {"person", "people", "human", "individual", "individuals"}


def _resolve_rag_pipeline_path() -> Path:
    configured_path = os.getenv("GUARDIAN_RAG_PIPELINE_PATH", "").strip()
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "FULL_RAG_Pipeline"
        if candidate.exists():
            return candidate.resolve()

    return (current_file.parents[2] / "FULL_RAG_Pipeline").resolve()


def _resolve_legal_artifact_path(env_name: str, filename: str) -> str:
    configured_path = os.getenv(env_name, "").strip()
    if configured_path:
        return str(Path(configured_path).expanduser().resolve())

    candidate = RAG_PIPELINE_PATH / "data" / filename
    return str(candidate.resolve()) if candidate.exists() else ""


def configure_rag_paths() -> None:
    global RAG_PIPELINE_PATH, LEGAL_INDEX_PATH, LEGAL_METADATA_DB_PATH

    RAG_PIPELINE_PATH = _resolve_rag_pipeline_path()
    LEGAL_INDEX_PATH = _resolve_legal_artifact_path(
        "GUARDIAN_LEGAL_INDEX_PATH",
        "legal_faiss.index",
    )
    LEGAL_METADATA_DB_PATH = _resolve_legal_artifact_path(
        "GUARDIAN_LEGAL_METADATA_DB_PATH",
        "legal_metadata.db",
    )
    logger.info(
        "RAG path configured mode=%s pipeline=%s index=%s metadata=%s",
        RAG_MODE,
        RAG_PIPELINE_PATH,
        LEGAL_INDEX_PATH or "<unset>",
        LEGAL_METADATA_DB_PATH or "<unset>",
    )
    print(
        "[Guardian Eye] RAG path "
        f"mode={RAG_MODE} pipeline={RAG_PIPELINE_PATH} "
        f"index={LEGAL_INDEX_PATH or '<unset>'} "
        f"metadata={LEGAL_METADATA_DB_PATH or '<unset>'}"
    )


configure_rag_paths()


def build_full_rag_response(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    vlm_summary: dict[str, Any] | None = None,
    country: str | None,
    language: str,
) -> dict[str, Any]:
    explanation_rag = _mock_explanation_rag(pred, packet_summary)
    incident_memory_rag = _mock_incident_memory_rag(pred)
    legal_consequences_rag, legal_scores = build_legal_response(
        pred=pred,
        packet_summary=packet_summary,
        narrative=narrative,
        vlm_summary=vlm_summary,
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
    vlm_summary: dict[str, Any] | None = None,
    country: str | None,
    language: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if RAG_MODE == "mock":
        clean_country = _supported_country(country or "") or (country or "").strip()
        if not clean_country:
            return _missing_country_legal_output(pred, language), None
        return _mock_legal_output(pred, clean_country, language), _mock_legal_scores()

    if _is_non_violent_prediction(pred):
        try:
            from services.curated_legal_service import build_curated_legal_response

            return build_curated_legal_response(
                pred=pred,
                packet_summary=packet_summary,
                narrative=narrative,
                vlm_summary=vlm_summary,
                country=country,
                language=language,
            )
        except Exception:
            clean_country = _supported_country(country or "") or (country or "").strip()
            return _fallback_legal_output(
                pred,
                clean_country,
                "Non-violent legal fallback used because curated legal service failed.",
                language,
            ), None

    clean_country = _supported_country(country or "")
    if clean_country is None:
        if not (country or "").strip():
            return _missing_country_legal_output(pred, language), None
        return _fallback_legal_output(
            pred,
            (country or "").strip(),
            UNVERIFIED_COUNTRY_WARNING,
            language,
        ), None

    index_warning = _legal_index_warning()
    if not index_warning:
        try:
            legal_output, legal_scores = _metadata_legal_output(
                pred=pred,
                packet_summary=packet_summary,
                narrative=narrative,
                country=clean_country,
                language=language,
            )
            _log_legal_mode("real", "curated markdown legal metadata KB")
            return legal_output, legal_scores
        except Exception as exc:
            index_warning = (
                "Curated legal metadata retrieval failed; using legal fallback. "
                f"Root cause: {exc.__class__.__name__}: {exc}"
            )

    # Compatibility backup: keep the older curated service available if the
    # rebuilt index is missing during local development.
    try:
        from services.curated_legal_service import build_curated_legal_response

        legal_output, legal_scores = build_curated_legal_response(
            pred=pred,
            packet_summary=packet_summary,
            narrative=narrative,
            vlm_summary=vlm_summary,
            country=country,
            language=language,
        )
        _log_legal_mode(
            str(legal_output.get("legal_rag_source") or "real"),
            f"backup curated local legal KB after metadata warning: {index_warning}",
        )
        return legal_output, legal_scores
    except Exception as exc:
        warning = (
            f"{index_warning} Backup curated legal retrieval also failed; "
            "using deterministic legal fallback. "
            f"Root cause: {exc.__class__.__name__}: {exc}"
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
            "incident_context": _incident_context(pred),
            "retrieved_legal_references": [],
            "summary": (
                f"{_incident_context_sentence(pred)} "
                f"Legal RAG could not use verified references for {country}. "
                "The deterministic fallback can only say that possible legal "
                "consequences depend on jurisdiction-specific law, the established "
                "facts, and review by competent authorities."
            ),
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
            "weapon_flag": _legal_weapon_flag(pred),
            "weapon_class": _legal_weapon_class(pred),
            "language": language,
        },
        legal_index=legal_index,
    )

    legal_output = dict(result.legal_consequences)
    legal_output = _ensure_current_incident_context(legal_output, pred, language)
    return (
        _finalize_legal_output(
            legal_output,
            source="real",
            warning=None,
            language=language,
        ),
        _compact_scores(result.evaluation),
    )


def _metadata_legal_output(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    country: str,
    language: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _load_official_legal_rows(country)
    if not rows:
        raise ValueError(f"No curated legal metadata rows for {country}")

    ranked_rows = _rank_legal_rows(
        rows,
        pred=pred,
        packet_summary=packet_summary,
        narrative=narrative,
    )
    references = [
        _reference_from_metadata_dict(row, score)
        for row, score, _intent_priority in ranked_rows[:3]
        if score > 0
    ]
    if not references:
        references = [_reference_from_metadata_dict(ranked_rows[0][0], ranked_rows[0][1])]

    deterministic_summary = _metadata_legal_summary(pred, country, references)
    summary, legal_mode, fallback_reason, generation_score = _metadata_legal_summary_after_retrieval(
        pred=pred,
        packet_summary=packet_summary,
        narrative=narrative,
        country=country,
        language=language,
        references=references,
        deterministic_summary=deterministic_summary,
    )
    output = {
        "country": country,
        "query_basis": _legal_query_basis(pred),
        "incident_context": _incident_context(pred),
        "retrieved_legal_references": references,
        "summary": summary,
        "guardrail_status": "passed",
        "limitations_note": LEGAL_LIMITATIONS_NOTE,
    }
    scores = {
        "retrieval_score": round(
            sum(float(ref["score"]) for ref in references) / max(len(references), 1),
            6,
        ),
        "generation_score": generation_score,
        "overall_score": round(
            (
                sum(float(ref["score"]) for ref in references) / max(len(references), 1)
                + generation_score
            )
            / 2,
            6,
        ),
        "passed": True,
    }
    return (
        _finalize_legal_output(
            output,
            source="real",
            warning="Curated legal references retrieved successfully.",
            language=language,
            legal_mode=legal_mode,
            reason_if_fallback=fallback_reason,
        ),
        scores,
    )


def _metadata_legal_summary_after_retrieval(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    country: str,
    language: str,
    references: list[dict[str, Any]],
    deterministic_summary: str,
) -> tuple[str, str, str | None, float]:
    llm_env_enabled = os.getenv("GUARDIAN_LEGAL_LLM_ENABLED", "1")
    model_id = _legal_llm_model_id()
    attempted_llm = llm_env_enabled != "0"
    print(f"[legal] env_enabled={llm_env_enabled}")
    print(f"[legal] model_id={model_id}")
    print(f"[legal] attempted_llm={str(attempted_llm).lower()}")
    print(
        "[legal] config "
        f"GUARDIAN_LEGAL_LLM_ENABLED={llm_env_enabled!r} "
        f"GUARDIAN_LEGAL_LLM_MODEL_ID={model_id!r} "
        f"local_path_valid={Path(model_id).is_dir()}"
    )
    try:
        if not attempted_llm:
            raise RuntimeError("GUARDIAN_LEGAL_LLM_ENABLED=0")
        print(
            "[legal] legal_llm_load_start "
            f"model_id={model_id!r} local_path_valid={Path(model_id).is_dir()}"
        )
        summary = _run_metadata_legal_llm(
            pred=pred,
            packet_summary=packet_summary,
            narrative=narrative,
            country=country,
            language=language,
            references=references,
        )
        print("[legal] guardrail_status=passed")
        print("[legal] mode=llm")
        print("[legal] legal_llm_success")
        print("[legal] fallback_reason=None")
        return summary, "llm", None, 0.9
    except Exception as exc:
        fallback_reason = _metadata_legal_fallback_reason(exc)
        print(f"[legal] mode=curated_fallback reason={fallback_reason!r}")
        print(
            f"[legal] legal_fallback_reason={fallback_reason!r} "
            f"exception={exc.__class__.__name__}: {exc}"
        )
        print("[legal] guardrail_status=passed")
        print(f"[legal] fallback_reason={fallback_reason!r}")
        return deterministic_summary, "curated_fallback", fallback_reason, 0.72


def _run_metadata_legal_llm(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    country: str,
    language: str,
    references: list[dict[str, Any]],
) -> str:
    from services.curated_legal_service import (
        CuratedLegalRecord,
        _ensure_required_legal_language,
        _guardrail_ok,
        _run_legal_llm,
    )

    records = [
        CuratedLegalRecord(
            country=str(reference.get("country") or country),
            language=language,
            act_type=_metadata_reference_act_type(reference),
            severity=_metadata_reference_severity(pred),
            consequence=_metadata_reference_consequence(reference),
            reference=_metadata_reference_label(reference),
            disclaimer=LEGAL_LIMITATIONS_NOTE,
        )
        for reference in references
    ]
    summary = _run_legal_llm(
        pred=pred,
        packet_summary=packet_summary,
        narrative=narrative,
        country=country,
        language=language,
        act_type=records[0].act_type if records else "violent_conduct",
        severity=records[0].severity if records else _metadata_reference_severity(pred),
        vlm_summary={},
        records=records,
    )
    summary = _ensure_required_legal_language(summary, language)
    if not _guardrail_ok(summary):
        raise RuntimeError("legal LLM output failed safety guardrail")
    return summary


def _metadata_reference_act_type(reference: dict[str, Any]) -> str:
    text = " ".join(
        str(reference.get(key) or "")
        for key in ("violence_category", "section_title", "law_title")
    ).casefold()
    if "weapon" in text or "dangerous object" in text:
        return "weapon_or_dangerous_object"
    return "violent_conduct"


def _metadata_reference_severity(pred: PredictResponse) -> str:
    if _legal_weapon_flag(pred) or float(pred.confidence or 0.0) >= 0.82:
        return "high"
    if int(pred.telemetry.people or 0) >= 4:
        return "high"
    return "medium"


def _metadata_reference_consequence(reference: dict[str, Any]) -> str:
    snippet = str(reference.get("snippet") or "")
    consequence = _extract_field_from_snippet(snippet, "Possible legal consequence")
    if consequence:
        return consequence
    return _snippet(snippet, 280) or "Possible consequences depend on the facts established by competent authorities."


def _metadata_reference_label(reference: dict[str, Any]) -> str:
    parts = [
        str(reference.get("law_title") or "").strip(),
        str(reference.get("article_number") or "").strip(),
        str(reference.get("section_title") or "").strip(),
    ]
    return "; ".join(part for part in parts if part) or "Retrieved legal reference"


def _legal_llm_model_id() -> str:
    try:
        from services.curated_legal_service import DEFAULT_LEGAL_LLM_MODEL
    except Exception:
        DEFAULT_LEGAL_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
    return os.getenv(
        "GUARDIAN_LEGAL_LLM_MODEL_ID",
        os.getenv("GUARDIAN_LLM_MODEL_ID", DEFAULT_LEGAL_LLM_MODEL),
    )


def _metadata_legal_fallback_reason(exc: Exception) -> str:
    try:
        from services.curated_legal_service import _safe_fallback_reason

        return _safe_fallback_reason(exc)
    except Exception:
        text = str(exc)
        if "GUARDIAN_LEGAL_LLM_ENABLED=0" in text:
            return "Legal LLM disabled"
        return "Legal LLM unavailable"


def _load_official_legal_rows(country: str) -> list[dict[str, Any]]:
    if not LEGAL_METADATA_DB_PATH or not Path(LEGAL_METADATA_DB_PATH).exists():
        raise FileNotFoundError(LEGAL_METADATA_DB_PATH or "legal_metadata.db")

    with sqlite3.connect(LEGAL_METADATA_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                vector_row,
                country,
                law_title,
                article_number,
                section_title,
                source_url,
                official_source,
                violence_category,
                text,
                matched_keywords
            FROM legal_chunk_metadata
            ORDER BY vector_row ASC
            """
        ).fetchall()

    supported = _supported_country(country)
    return [
        dict(row)
        for row in rows
        if supported is not None and _supported_country(str(row["country"])) == supported
    ]


def _rank_legal_rows(
    rows: list[dict[str, Any]],
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
) -> list[tuple[dict[str, Any], float, int]]:
    terms = _legal_context_terms(pred, packet_summary, narrative)
    query_text = _primary_query_text(packet_summary, narrative)
    has_legal_weapon = _legal_weapon_flag(pred)
    query_intent = {
        "domestic": _contains_domestic_terms(query_text),
        "weapon": has_legal_weapon and _contains_weapon_terms(query_text),
        "battery": "battery" in query_text or "ضرب" in query_text,
        "sexual": "sexual" in query_text,
    }
    ranked: list[tuple[dict[str, Any], float, int]] = []
    for row in rows:
        searchable = " ".join(
            str(row.get(key) or "")
            for key in (
                "law_title",
                "section_title",
                "violence_category",
                "text",
                "matched_keywords",
            )
        ).casefold()
        matches = {term for term in terms if term in searchable}
        keyword_score = len(matches) / max(min(len(terms), 8), 1)
        focus_text = " ".join(
            str(row.get(key) or "")
            for key in ("section_title", "violence_category", "matched_keywords")
        ).casefold()
        focus_bonus = 0.18 if any(term in focus_text for term in matches) else 0.0
        exact_intent_match = False
        intent_priority = 0
        if "domestic" in terms and "domestic" in focus_text:
            focus_bonus += 0.22
            exact_intent_match = True
            intent_priority += 30 if query_intent["domestic"] else 8
        if "weapon" in terms and "weapon" in focus_text:
            focus_bonus += 0.22
            exact_intent_match = True
            category_text = str(row.get("violence_category") or "").casefold()
            section_text = str(row.get("section_title") or "").casefold()
            priority = 14 if "weapon" in category_text or "weapon" in section_text else 3
            intent_priority += priority + (20 if query_intent["weapon"] else 0)
        if "battery" in terms and "battery" in focus_text:
            focus_bonus += 0.22
            exact_intent_match = True
            intent_priority += 24 if query_intent["battery"] else 4
        if "sexual" in terms and "sexual" in focus_text:
            focus_bonus += 0.22
            exact_intent_match = True
            intent_priority += 25 if query_intent["sexual"] else 5
        weapon_bonus = (
            0.18
            if has_legal_weapon and ("weapon" in focus_text or "dangerous" in focus_text)
            else 0.08
            if has_legal_weapon and "weapon" in searchable
            else 0.0
        )
        official_bonus = 0.06 if row.get("official_source") else 0.0
        article_bonus = 0.05 if row.get("article_number") else 0.0
        cap = 1.0 if exact_intent_match else 0.985
        score = round(
            min(cap, 0.38 + keyword_score * 0.42 + focus_bonus + weapon_bonus + official_bonus + article_bonus),
            6,
        )
        ranked.append((row, score, intent_priority))
    ranked.sort(key=lambda item: (-item[1], -item[2], int(item[0].get("vector_row") or 0)))
    return ranked


def _primary_query_text(packet_summary: str, narrative: str) -> str:
    packet_first = str(packet_summary or "").splitlines()[0] if packet_summary else ""
    narrative_first = str(narrative or "").splitlines()[0] if narrative else ""
    return f"{packet_first} {narrative_first}".casefold()


def _contains_domestic_terms(text: str) -> bool:
    return any(
        term in text
        for term in ("domestic", "family abuse", "intimate partner", "منزلي", "اسري", "أسري", "عائلي", "العائلة", "الأسرة", "الاسرة")
    )


def _contains_weapon_terms(text: str) -> bool:
    return any(
        term in text
        for term in ("weapon", "dangerous object", "knife", "bottle", "سلاح", "سكين", "أداة", "اداة", "خطر", "خطير")
    )


def _legal_weapon_flag(pred: PredictResponse) -> bool:
    weapon = pred.telemetry.weapon
    if not bool(weapon.flag):
        return False
    return not _is_person_object_class(weapon.cls)


def _legal_weapon_class(pred: PredictResponse) -> str | None:
    if not _legal_weapon_flag(pred):
        return None
    weapon_class = pred.telemetry.weapon.cls
    return str(weapon_class).strip() if weapon_class else None


def _is_person_object_class(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    return normalized in PERSON_OBJECT_CLASSES


def _reference_from_metadata_dict(row: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "law_title": row.get("law_title") or "Legal reference",
        "article_number": row.get("article_number"),
        "section_title": row.get("section_title"),
        "source_url": row.get("source_url") or "",
        "snippet": _snippet(str(row.get("text") or ""), max_chars=850),
        "score": float(score),
        "country": row.get("country"),
        "violence_category": row.get("violence_category"),
        "official_source": bool(row.get("official_source")),
    }


def _metadata_legal_summary(
    pred: PredictResponse,
    country: str,
    references: list[dict[str, Any]],
) -> str:
    primary = references[0] if references else {}
    act = str(primary.get("section_title") or primary.get("violence_category") or "violent act")
    consequence = _extract_field_from_snippet(
        str(primary.get("snippet") or ""),
        "Possible legal consequence",
    )
    if not consequence:
        consequence = "Possible consequences depend on the facts established by competent authorities."
    weapon_phrase = (
        "Weapon/object context: A weapon or dangerous object was identified."
        if _legal_weapon_flag(pred)
        else "Weapon/object context: No weapon or dangerous object was identified."
    )
    return (
        f"Country: {country}\n"
        f"Relevant act: {_legal_act_label(act, has_real_weapon=_legal_weapon_flag(pred))}\n"
        f"{weapon_phrase}\n"
        f"Possible consequence: If responsibility is confirmed by authorities, {consequence[:1].lower() + consequence[1:]}\n"
        "Limitation: This is legal context only, not legal advice."
    )


def _extract_field_from_snippet(snippet: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}:\s*(.*?)(?:\s+[A-Z][A-Za-z /-]+:|$)"
    match = re.search(pattern, snippet, flags=re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _metadata_limitation_note(reference: dict[str, Any]) -> str:
    snippet = str(reference.get("snippet") or "")
    notes = _extract_field_from_snippet(snippet, "Notes")
    article = reference.get("article_number")
    source_url = str(reference.get("source_url") or "")
    if source_url.startswith("unavailable://"):
        return notes or "Reliable article-level source is unavailable in the local verified source set."
    if not article:
        return notes or "Official article-level verification may be required before relying on this section."
    return notes or "Review the cited source and confirmed incident facts before relying on this section."


def _clean_legal_excerpt(row: dict[str, Any]) -> str:
    law_title = str(row.get("law_title") or "This official legal reference")
    category = str(row.get("violence_category") or "").replace("_", " ").strip()
    section = str(row.get("section_title") or "").strip()
    country = str(row.get("country") or "the selected jurisdiction")

    focus = section or category or "violent incidents"
    sentences = [
        f"{law_title} is an official source for {country} relevant to {focus}.",
    ]

    lowered = " ".join(
        str(row.get(key) or "")
        for key in ("law_title", "section_title", "violence_category", "text")
    ).casefold()
    if "weapon" in lowered or "dangerous" in lowered:
        sentences.append(
            "It is relevant when a violent incident may involve a weapon or dangerous object."
        )
    elif "assault" in lowered or "battery" in lowered or "harm" in lowered:
        sentences.append(
            "It is relevant when reviewing alleged assault, battery, or physical harm."
        )
    else:
        sentences.append(
            "It should be reviewed by qualified authorities against the established facts."
        )

    sentences.append("This excerpt is a concise demo summary, not legal advice.")
    return " ".join(sentences)


def _ensure_current_incident_context(
    legal_output: dict[str, Any],
    pred: PredictResponse,
    language: str,
) -> dict[str, Any]:
    output = dict(legal_output)
    output["incident_context"] = _incident_context(pred)
    summary = str(output.get("summary") or "")
    context = _incident_context_sentence(pred)
    if context not in summary:
        output["summary"] = f"{context} {summary}".strip()
    return output


def _incident_context(pred: PredictResponse) -> dict[str, Any]:
    return {
        "verdict": pred.verdict,
        "confidence": pred.confidence,
        "people": pred.telemetry.people,
        "peak_window": pred.telemetry.peak_window,
        "weapon_flag": _legal_weapon_flag(pred),
        "weapon_class": _legal_weapon_class(pred),
        "dangerous_object_context": _legal_weapon_flag(pred),
    }


def _incident_context_sentence(pred: PredictResponse) -> str:
    weapon = (
        f"weapon/object context is present ({_legal_weapon_class(pred)})"
        if _legal_weapon_flag(pred) and _legal_weapon_class(pred)
        else "weapon/object context is present"
        if _legal_weapon_flag(pred)
        else "no weapon or dangerous object was identified"
    )
    return (
        "For the current incident prediction, "
        f"the classifier verdict is {pred.verdict} with {pred.confidence:.0%} confidence, "
        f"{pred.telemetry.people} tracked people, peak window "
        f"{pred.telemetry.peak_window}, and {weapon}."
    )


def _legal_context_terms(
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
) -> set[str]:
    text = " ".join(
        [
            pred.verdict,
            packet_summary or "",
            narrative or "",
            _legal_weapon_class(pred) or "",
            "weapon dangerous object" if _legal_weapon_flag(pred) else "",
        ]
    ).casefold()
    terms = {
        word
        for word in re.findall(r"[\w\u0600-\u06FF]+", text)
        if len(word) >= 4
    }
    if not _legal_weapon_flag(pred):
        terms.difference_update({"weapon", "weapons", "dangerous", "object", "objects"})
    if pred.verdict == "violence":
        terms.update({"violence", "violent", "assault", "harm"})
    if _legal_weapon_flag(pred):
        terms.update({"weapon", "dangerous", "object"})
    if "domestic" in text or "partner" in text or "family" in text:
        terms.update({"domestic", "partner", "family", "abuse"})
    if "robbery" in text or "theft" in text or "steal" in text:
        terms.update({"robbery", "theft", "force", "fear"})
    if "threat" in text or "intimidat" in text:
        terms.update({"threat", "threats", "intimidation"})
    if "sexual" in text:
        terms.update({"sexual", "consent", "assault"})
    if _contains_domestic_terms(text):
        terms.update({"domestic", "partner", "family", "abuse"})
    if _legal_weapon_flag(pred) and _contains_weapon_terms(text):
        terms.update({"weapon", "dangerous", "object"})
    if any(term in text for term in ("اعتداء", "ضرب", "إيذاء", "ايذاء", "عنف")):
        terms.update({"assault", "battery", "harm", "violence"})
    if any(term in text for term in ("تهديد", "تهدد", "تخويف")):
        terms.update({"threat", "threats", "intimidation"})
    return terms


def _is_non_violent_prediction(pred: PredictResponse) -> bool:
    normalized = str(pred.verdict or "").casefold().replace("_", "-")
    return normalized in {"non-violence", "nonviolent", "non-violent"}


def _snippet(text: str, max_chars: int = 420) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _finalize_legal_output(
    output: dict[str, Any],
    *,
    source: str,
    warning: str | None,
    language: str,
    legacy_warning: str | None = None,
    legal_mode: str = "curated_fallback",
    reason_if_fallback: str | None = None,
) -> dict[str, Any]:
    final_output = dict(output)
    final_output["legal_mode"] = legal_mode
    final_output["reason_if_fallback"] = reason_if_fallback
    final_output = _normalize_serialized_legal_output(final_output)
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
        legal_mode=legal_mode,
        reason_if_fallback=reason_if_fallback,
    )


def _normalize_serialized_legal_output(output: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(output)
    has_person_weapon_leak = _serialized_person_weapon_leak(normalized)
    has_real_weapon = _serialized_has_real_weapon(normalized) and not has_person_weapon_leak

    if has_person_weapon_leak:
        for key in ("query_basis", "incident_context"):
            value = normalized.get(key)
            if isinstance(value, dict):
                clean_value = dict(value)
                clean_value.update(
                    {
                        "weapon_flag": False,
                        "weapon_class": None,
                        "dangerous_object_context": False,
                    }
                )
                normalized[key] = clean_value
        normalized["retrieved_legal_references"] = [
            reference
            for reference in normalized.get("retrieved_legal_references", [])
            if not _serialized_weapon_reference(reference)
        ]

    normalized["summary"] = _compact_serialized_legal_summary(
        normalized,
        has_real_weapon=has_real_weapon,
    )
    return normalized


def _serialized_person_weapon_leak(output: dict[str, Any]) -> bool:
    summary = str(output.get("summary") or "").casefold()
    if "(person)" in summary or "present (person)" in summary:
        return True
    for key in ("query_basis", "incident_context"):
        value = output.get(key)
        if isinstance(value, dict) and _is_person_object_class(value.get("weapon_class")):
            return True
    return False


def _serialized_has_real_weapon(output: dict[str, Any]) -> bool:
    for key in ("query_basis", "incident_context"):
        value = output.get(key)
        if isinstance(value, dict):
            weapon_class = value.get("weapon_class")
            if bool(value.get("weapon_flag")) and not _is_person_object_class(weapon_class):
                return True
    return False


def _serialized_weapon_reference(reference: dict[str, Any]) -> bool:
    category = str(reference.get("violence_category") or "").casefold()
    if category == "weapon_or_dangerous_object":
        return True
    text = " ".join(
        str(reference.get(key) or "")
        for key in ("law_title", "section_title", "snippet")
    ).casefold()
    return any(
        term in text
        for term in (
            "offensive weapon",
            "weapon or dangerous",
            "dangerous weapon",
            "dangerous object",
        )
    )


def _compact_serialized_legal_summary(output: dict[str, Any], *, has_real_weapon: bool) -> str:
    country = str(output.get("country") or "")
    summary = str(output.get("summary") or "")
    references = output.get("retrieved_legal_references") or []
    primary = references[0] if references else {}
    act = str(primary.get("section_title") or primary.get("violence_category") or "violent conduct")
    consequence = ""
    if output.get("legal_mode") == "llm":
        consequence = _llm_consequence_from_summary(summary)
    if not consequence:
        consequence = _extract_field_from_snippet(str(primary.get("snippet") or ""), "Possible legal consequence")
    if not consequence:
        consequence = _extract_line_value(summary, "Possible consequence") or _extract_line_value(
            summary,
            "Possible legal consequence",
        )
    consequence = _clean_legal_possible_consequence(consequence)
    if not consequence:
        consequence = (
            "the incident may be reviewed under assault-related provisions. "
            "The outcome depends on evidence, harm, context, and the competent authority."
        )
    consequence = re.sub(r"https?://\S+", "", consequence).strip()
    if "if responsibility is confirmed by authorities" not in consequence.casefold():
        consequence = f"If responsibility is confirmed by authorities, {consequence[:1].lower() + consequence[1:]}"
    consequence = _clean_legal_possible_consequence(consequence) or (
        "If responsibility is confirmed by authorities, the incident may be reviewed under "
        "assault-related provisions. The outcome depends on evidence, harm, context, and "
        "the competent authority."
    )
    return "\n".join(
        [
            f"Country: {country}",
            f"Relevant act: {_legal_act_label(act, has_real_weapon=has_real_weapon)}",
            (
                "Weapon/object context: A weapon or dangerous object was identified."
                if has_real_weapon
                else "Weapon/object context: No weapon or dangerous object was identified."
            ),
            f"Possible consequence: {consequence.rstrip('.')}.",
            "Limitation: This is legal context only, not legal advice.",
        ]
    )


def _extract_line_value(text: str, label: str) -> str:
    pattern = rf"^\s*{re.escape(label)}\s*:\s*(.+)$"
    for line in str(text or "").splitlines():
        match = re.match(pattern, line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _llm_consequence_from_summary(summary: str) -> str:
    clean = re.sub(r"https?://\S+", "", str(summary or ""))
    value = _extract_line_value(clean, "Possible consequence") or _extract_line_value(
        clean,
        "Possible legal consequence",
    )
    if value:
        return value
    clean = re.sub(r"\bThis is not legal advice\.?", "", clean, flags=re.IGNORECASE)
    for sentence in re.findall(r"[^.!?\n]+[.!?]?", clean):
        stripped = re.sub(r"\s+", " ", sentence).strip()
        lowered = stripped.casefold()
        if not stripped:
            continue
        if lowered.startswith(("country:", "relevant act:", "weapon/object context:", "limitation:", "source:")):
            continue
        return stripped
    return ""


def _clean_legal_possible_consequence(text: str) -> str:
    clean = re.sub(r"https?://\S+", "", str(text or ""))
    clean = re.sub(r"\([^)]*$", "", clean).strip()
    clean = re.sub(
        r"\b(?:with\s+)?(?:high|medium|low)?\s*confidence\s*\(?\d+(?:\.\d*)?\)?",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\bconfidence\s*\(?\d+(?:\.\d*)?\)?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    sentences = [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in re.findall(r"[^.!?؟۔]+[.!?؟۔]", clean)
    ]
    sentences = [
        sentence
        for sentence in sentences
        if sentence
        and "confidence" not in sentence.casefold()
        and not re.search(r"\(\s*\d+(?:\.\d*)?[.!?؟۔]?$", sentence)
        and not re.search(r"\b\d+(?:\.\d*)?[.!?؟۔]?$", sentence)
        and not any(term in sentence.casefold() for term in ("attacker", "victim", "guilty", "guilt"))
    ]
    return " ".join(sentences).strip()


def _legal_act_label(act: str, *, has_real_weapon: bool) -> str:
    lowered = str(act or "").replace("_", " ").casefold()
    if has_real_weapon and ("weapon" in lowered or "dangerous object" in lowered):
        return "Weapon or dangerous-object related violent conduct"
    if "weapon" in lowered or "dangerous object" in lowered:
        return "Common assault / violent conduct"
    if "assault" in lowered or "battery" in lowered or "harm" in lowered or "violent" in lowered:
        return "Common assault / violent conduct"
    return str(act or "violent conduct").replace("_", " ")


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
    return _basic_arabic_legal_output(output)


def _basic_arabic_legal_output(output: dict[str, Any]) -> dict[str, Any]:
    translated = dict(output)
    country = translated.get("country") or "الدولة المحددة"
    references = []
    original_references = translated.get("retrieved_legal_references", [])
    for reference in original_references:
        item = dict(reference)
        section_title = item.get("section_title") or "مرجع قانوني"
        item["section_title"] = f"مرجع قانوني: {section_title}"
        item["snippet"] = (
            "هذا مرجع قانوني مسترجع خاص بالدولة المحددة. يجب مراجعته مع النص "
            "القانوني الرسمي والوقائع المثبتة، ولا يعد استشارة قانونية."
        )
        references.append(item)

    translated["retrieved_legal_references"] = references
    primary = original_references[0] if original_references else {}
    act = primary.get("section_title") or primary.get("violence_category") or "الفعل العنيف"
    law = primary.get("law_title") or "مرجع قانوني مسترجع"
    source_url = primary.get("source_url") or "غير متاح"
    article = primary.get("article_number") or "رقم المادة غير متاح في المصدر المحلي المتحقق منه"
    limitation = (
        "المصدر يوضح أن التحقق الرسمي من رقم المادة مطلوب."
        if str(source_url).startswith("unavailable://") or not primary.get("article_number")
        else "يجب مراجعة المصدر القانوني والوقائع المثبتة قبل الاعتماد على النتيجة."
    )
    translated["summary"] = (
        f"الدولة: {country}\n"
        f"الفعل ذو الصلة: {act}\n"
        "العاقبة القانونية المحتملة: قد تشمل تحقيقا أو شروط حماية أو عقوبات "
        "تحددها الجهة المختصة حسب الوقائع المثبتة والمصدر المسترجع.\n"
        f"القانون أو المادة: {law}; {article}\n"
        f"المصدر: {source_url}\n"
        f"القيد: {limitation} هذا سياق قانوني فقط وليس نصيحة قانونية، ولا يحدد الإدانة أو نتيجة المحكمة."
    )
    translated["limitations_note"] = (
        "هذه ليست استشارة قانونية، ولا تحدد الإدانة، ولا تتنبأ بنتيجة المحكمة."
    )
    return translated


def _legacy_fallback_translate_legal_output_to_arabic(output: dict[str, Any]) -> dict[str, Any]:
    translated = dict(output)
    translated["summary"] = _fallback_arabic_legal_summary(translated)
    context_sentence = _fallback_arabic_incident_context(translated)
    if context_sentence and context_sentence not in translated["summary"]:
        translated["summary"] = f"{context_sentence} {translated['summary']}"
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


def _fallback_arabic_incident_context(output: dict[str, Any]) -> str:
    context = output.get("incident_context") or {}
    query_basis = output.get("query_basis") or {}
    verdict = context.get("verdict") or query_basis.get("verdict")
    if not verdict:
        return ""
    verdict_ar = _fallback_arabic_label(str(verdict))
    confidence = context.get("confidence")
    confidence_text = f" بنسبة ثقة {float(confidence):.0%}" if isinstance(confidence, (int, float)) else ""
    people = context.get("people")
    people_text = f"، وعدد الأشخاص المتتبعين {people}" if people is not None else ""
    peak = context.get("peak_window")
    peak_text = f"، ونافذة الذروة {peak}" if peak else ""
    weapon_flag = context.get("weapon_flag", query_basis.get("weapon_flag"))
    weapon_class = context.get("weapon_class") or query_basis.get("weapon_class")
    weapon_class_ar = _fallback_arabic_label(str(weapon_class)) if weapon_class else ""
    if weapon_flag:
        weapon_text = f"، مع إشارة إلى سلاح أو جسم قريب{f' ({weapon_class_ar})' if weapon_class_ar else ''}"
    else:
        weapon_text = "، ودون إشارة مؤكدة إلى سلاح أو جسم قريب"
    return (
        f"بالنسبة للتنبؤ الحالي، صنف النظام الواقعة كـ {verdict_ar}{confidence_text}"
        f"{people_text}{peak_text}{weapon_text}."
    )


def _fallback_arabic_label(label: str) -> str:
    mapping = {
        "violence": "عنف",
        "non-violence": "غير عنيفة",
        "non violence": "غير عنيفة",
        "bottle": "زجاجة",
        "knife": "سكين",
        "person": "شخص",
        "weapon": "سلاح",
    }
    return mapping.get(label.casefold(), label)


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
    if "retrieved successfully" in warning.casefold():
        return "تم استرجاع مراجع قانونية منسقة بنجاح. راجع النص القانوني الرسمي قبل الاعتماد على أي نتيجة."
    return "تم استخدام مسار قانوني احتياطي أو معلوماتي. راجع المراجع المسترجعة والنص القانوني الرسمي."


def _legacy_fallback_arabic_warning(warning: str) -> str:
    lowered = warning.casefold()
    if "official legal metadata retrieval fallback" in lowered:
        return "تعذر تشغيل البحث المتجهي القانوني، لذلك تم استخدام المراجع القانونية الرسمية المخزنة محليا."
    if "no verified official-source references" in lowered:
        return "لا توجد مراجع قانونية رسمية موثوقة لهذه الدولة في فهرس العرض المحلي."
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
    legal_mode: str = "curated_fallback",
    reason_if_fallback: str | None = None,
) -> dict[str, Any]:
    enriched = dict(output)
    enriched["rag_mode"] = RAG_MODE
    enriched["legal_rag_source"] = source
    enriched["legal_mode"] = legal_mode
    enriched["legal_rag_warning"] = warning
    enriched["reason_if_fallback"] = reason_if_fallback
    enriched["warning"] = legacy_warning or warning
    return enriched


def _legal_index_warning() -> str | None:
    if not RAG_PIPELINE_PATH.exists():
        return (
            "Legal RAG is running in fallback mode because the shared RAG "
            f"pipeline path does not exist: {RAG_PIPELINE_PATH}"
        )

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


def _supported_country(country: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(country or "").casefold()).strip()
    return SUPPORTED_LEGAL_COUNTRIES.get(normalized)


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
        "weapon_flag": _legal_weapon_flag(pred),
        "weapon_class": _legal_weapon_class(pred),
        "dangerous_object_context": _legal_weapon_flag(pred),
    }


def _evidence_basis(pred: PredictResponse, packet_summary: str) -> list[str]:
    basis = [
        f"classifier confidence {pred.confidence:.2f}",
        f"verdict {pred.verdict}",
        f"people tracked {pred.telemetry.people}",
    ]
    if _legal_weapon_flag(pred):
        basis.append(f"object context {_legal_weapon_class(pred) or 'present'}")
    if packet_summary:
        basis.append("packet summary available")
    return basis
