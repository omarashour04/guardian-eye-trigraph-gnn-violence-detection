from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from schemas import PredictResponse
from text_quality import ARABIC_ONLY_INSTRUCTION, polish_generated_text


def _default_legal_llm_model() -> str:
    local_model = Path(__file__).resolve().parents[3] / "models" / "qwen2.5-1.5b-instruct"
    if local_model.exists():
        return str(local_model)
    return "Qwen/Qwen2.5-3B-Instruct-AWQ"


DEFAULT_LEGAL_LLM_MODEL = _default_legal_llm_model()
LEGAL_LIMITATIONS_NOTE = (
    "This is not legal advice, does not identify attacker or victim roles, "
    "does not determine guilt, and does not predict court outcome."
)
NON_VIOLENT_LEGAL_MESSAGE = (
    "No legal consequences are suggested because the incident was classified as non-violent."
)
NON_VIOLENT_LEGAL_MESSAGE_AR = (
    "لا توجد عواقب قانونية مقترحة لأن الحادثة صُنفت على أنها غير عنيفة."
)


@dataclass(frozen=True)
class CuratedLegalRecord:
    country: str
    language: str
    act_type: str
    severity: str
    consequence: str
    reference: str
    disclaimer: str


CURATED_LEGAL_KB: tuple[CuratedLegalRecord, ...] = (
    CuratedLegalRecord(
        country="Canada",
        language="en",
        act_type="violent_conduct",
        severity="low",
        consequence=(
            "Lower-severity physical contact or threatening conduct may lead to a police "
            "report, warning, diversion review, peace-bond conditions, or other proportionate "
            "measures depending on confirmed facts."
        ),
        reference="Canada Criminal Code: lower-severity assault and peace-bond context overview",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="UK",
        language="en",
        act_type="violent_conduct",
        severity="low",
        consequence=(
            "Lower-severity public confrontation or minor physical contact may be reviewed "
            "as common assault or public-order conduct, with possible warning, caution, "
            "community resolution, or protective conditions."
        ),
        reference="UK: common assault and public-order lower-severity overview",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="USA California",
        language="en",
        act_type="violent_conduct",
        severity="low",
        consequence=(
            "Lower-severity battery or disorderly-conduct concerns may lead to law-enforcement "
            "documentation, citation review, restraining-order consideration, diversion, or "
            "probation-related conditions if responsibility is confirmed."
        ),
        reference="California: lower-severity battery and disorderly-conduct consequence overview",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="UAE",
        language="ar",
        act_type="violent_conduct",
        severity="low",
        consequence=(
            "إذا ثبتت المسؤولية من الجهات المختصة، فقد تؤدي واقعة احتكاك أو تهديد محدود "
            "إلى بلاغ أو مراجعة إدارية أو تدابير تعهد وحماية بحسب الوقائع المثبتة."
        ),
        reference="الإمارات: إرشاد موجز حول الوقائع العنيفة منخفضة الخطورة",
        disclaimer="هذا ليس استشارة قانونية، ولا يحدد الذنب أو نتيجة أي إجراء.",
    ),
    CuratedLegalRecord(
        country="KSA",
        language="ar",
        act_type="violent_conduct",
        severity="low",
        consequence=(
            "إذا ثبتت المسؤولية من الجهات المختصة، فقد ترتبط الواقعة منخفضة الخطورة "
            "بتوثيق رسمي أو تعهد أو إجراء صلح أو تدبير حماية وفق الأنظمة والإجراءات المعمول بها."
        ),
        reference="السعودية: إرشاد موجز حول المخالفات العنيفة منخفضة الخطورة",
        disclaimer="هذا ليس استشارة قانونية، ولا يحدد الذنب أو نتيجة المحكمة.",
    ),
    CuratedLegalRecord(
        country="Egypt",
        language="ar",
        act_type="violent_conduct",
        severity="low",
        consequence=(
            "إذا ثبتت المسؤولية من الجهات المختصة، فقد تؤدي واقعة مشادة أو احتكاك محدود "
            "إلى محضر أو تصالح أو تدبير احترازي بحسب ما تثبته الجهات المختصة."
        ),
        reference="مصر: إرشاد موجز حول الوقائع العنيفة منخفضة الخطورة",
        disclaimer="هذا ليس استشارة قانونية، ولا يحدد الذنب أو نتيجة أي إجراء.",
    ),
    CuratedLegalRecord(
        country="Canada",
        language="en",
        act_type="violent_conduct",
        severity="medium",
        consequence=(
            "Possible consequences may include police review, assault-related charges, "
            "release conditions, protective conditions, fines, probation, or court-imposed "
            "penalties depending on proven facts."
        ),
        reference="Canada Criminal Code: assault and sentencing principles overview",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="Canada",
        language="en",
        act_type="weapon_or_dangerous_object",
        severity="high",
        consequence=(
            "A weapon or dangerous-object context may increase legal seriousness and may "
            "lead authorities to consider weapon-related, assault-related, or public safety "
            "measures if supported by evidence."
        ),
        reference="Canada Criminal Code: assault, weapons, and public safety considerations",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="UK",
        language="en",
        act_type="violent_conduct",
        severity="medium",
        consequence=(
            "Possible consequences may include investigation for assault, public order "
            "offences, protective conditions, community penalties, fines, or custody "
            "depending on proven harm and court assessment."
        ),
        reference="UK: assault and public order consequences overview",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="UK",
        language="en",
        act_type="weapon_or_dangerous_object",
        severity="high",
        consequence=(
            "A dangerous-object context may be treated as an aggravating factor and may "
            "affect police response, charging review, bail conditions, or sentencing."
        ),
        reference="UK: weapon or dangerous-object aggravating context overview",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="USA California",
        language="en",
        act_type="violent_conduct",
        severity="medium",
        consequence=(
            "Possible consequences may include law-enforcement review for assault or "
            "battery-related offences, protective orders, probation, fines, or custody "
            "depending on confirmed facts and prosecutorial decisions."
        ),
        reference="California: assault, battery, and protective-order consequences overview",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="USA California",
        language="en",
        act_type="weapon_or_dangerous_object",
        severity="high",
        consequence=(
            "A weapon or dangerous-object context may increase seriousness and may affect "
            "charging review, protective orders, bail conditions, or sentencing exposure."
        ),
        reference="California: dangerous weapon and assault-related consequence overview",
        disclaimer="Use only as curated guidance for demonstration, not legal advice.",
    ),
    CuratedLegalRecord(
        country="UAE",
        language="ar",
        act_type="violent_conduct",
        severity="medium",
        consequence=(
            "قد تشمل العواقب المحتملة مراجعة من الجهات المختصة، أو تدابير حماية، "
            "أو غرامات، أو عقوبات تحددها السلطة المختصة حسب الوقائع المثبتة."
        ),
        reference="الإمارات: إرشاد موجز حول وقائع العنف والاعتداء",
        disclaimer="هذا سياق إرشادي منسق للعرض وليس استشارة قانونية.",
    ),
    CuratedLegalRecord(
        country="UAE",
        language="ar",
        act_type="weapon_or_dangerous_object",
        severity="high",
        consequence=(
            "وجود سلاح أو جسم خطر قد يزيد من جدية المراجعة القانونية وقد يرتبط "
            "بتدابير حماية أو عقوبات تحددها الجهات المختصة إذا ثبتت المسؤولية."
        ),
        reference="الإمارات: إرشاد موجز حول استخدام أداة خطرة في واقعة عنف",
        disclaimer="هذا سياق إرشادي منسق للعرض وليس استشارة قانونية.",
    ),
    CuratedLegalRecord(
        country="KSA",
        language="ar",
        act_type="violent_conduct",
        severity="medium",
        consequence=(
            "قد تؤدي الوقائع العنيفة المثبتة إلى مراجعة من الجهات المختصة، "
            "وتدابير حماية أو عقوبات تقديرية وفق الأنظمة والإجراءات المعمول بها."
        ),
        reference="السعودية: إرشاد موجز حول الاعتداء والعنف",
        disclaimer="هذا سياق إرشادي منسق للعرض وليس استشارة قانونية.",
    ),
    CuratedLegalRecord(
        country="KSA",
        language="ar",
        act_type="weapon_or_dangerous_object",
        severity="high",
        consequence=(
            "وجود أداة خطرة أو سلاح قد يجعل الواقعة أشد خطورة عند التقييم، "
            "وقد يترتب عليه تدابير أو عقوبات تقررها الجهات المختصة إذا ثبتت المسؤولية."
        ),
        reference="السعودية: إرشاد موجز حول الأداة الخطرة في وقائع العنف",
        disclaimer="هذا سياق إرشادي منسق للعرض وليس استشارة قانونية.",
    ),
    CuratedLegalRecord(
        country="Egypt",
        language="ar",
        act_type="violent_conduct",
        severity="medium",
        consequence=(
            "قد تشمل العواقب المحتملة محضرا أو تحقيقا، وتدابير حماية أو عقوبات "
            "تحددها الجهات المختصة وفقا للوقائع المثبتة والإجراءات القانونية."
        ),
        reference="مصر: إرشاد موجز حول الاعتداء ووقائع العنف",
        disclaimer="هذا سياق إرشادي منسق للعرض وليس استشارة قانونية.",
    ),
    CuratedLegalRecord(
        country="Egypt",
        language="ar",
        act_type="weapon_or_dangerous_object",
        severity="high",
        consequence=(
            "استخدام أداة خطرة أو سلاح في واقعة عنف قد يزيد من خطورة التقييم "
            "وقد يؤدي إلى تدابير أو عقوبات أشد إذا أكدت السلطات المسؤولية."
        ),
        reference="مصر: إرشاد موجز حول الأداة الخطرة في وقائع العنف",
        disclaimer="هذا سياق إرشادي منسق للعرض وليس استشارة قانونية.",
    ),
)


COUNTRY_ALIASES = {
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

PERSON_OBJECT_CLASSES = {"person", "people", "human", "individual", "individuals"}
DANGEROUS_OBJECT_TERMS = {
    "weapon",
    "knife",
    "bottle",
    "gun",
    "firearm",
    "blade",
    "stick",
    "bat",
    "dangerous object",
    "سلاح",
    "سكين",
    "زجاجة",
}
CONCRETE_DANGEROUS_OBJECT_TERMS = DANGEROUS_OBJECT_TERMS.difference({"weapon", "object"})


def build_curated_legal_response(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    vlm_summary: dict[str, Any] | None = None,
    country: str | None,
    language: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if _is_non_violent(pred):
        return _non_violent_output(pred, country or "", language), None

    clean_country = normalize_country(country)
    if clean_country is None:
        return _missing_country_output(pred, language), None

    vlm_summary = vlm_summary if isinstance(vlm_summary, dict) else {}
    act_type = infer_act_type(pred, packet_summary, narrative, vlm_summary=vlm_summary)
    severity = infer_severity(pred, packet_summary, narrative, vlm_summary=vlm_summary)
    records = retrieve_curated_legal_context(
        country=clean_country,
        language=language,
        act_type=act_type,
        severity=severity,
    )
    print(
        "[legal] retrieved_refs="
        f"{len(records)} country={clean_country!r} act_type={act_type!r} severity={severity!r}"
    )
    llm_env_enabled = os.getenv("GUARDIAN_LEGAL_LLM_ENABLED", "1")
    legal_model_id = os.getenv(
        "GUARDIAN_LEGAL_LLM_MODEL_ID",
        os.getenv("GUARDIAN_LLM_MODEL_ID", DEFAULT_LEGAL_LLM_MODEL),
    )
    attempted_llm = llm_env_enabled != "0"
    print(f"[legal] env_enabled={llm_env_enabled}")
    print(f"[legal] model_id={legal_model_id}")
    print(f"[legal] attempted_llm={str(attempted_llm).lower()}")
    print(
        "[legal] config "
        f"GUARDIAN_LEGAL_LLM_ENABLED={llm_env_enabled!r} "
        f"GUARDIAN_LEGAL_LLM_MODEL_ID={legal_model_id!r} "
        f"local_path_valid={Path(legal_model_id).is_dir()}"
    )
    if not records:
        warning = f"No curated legal reference is available for {clean_country}."
        print(f"[legal] mode=curated_fallback reason={warning!r}")
        print(f"[legal] fallback_reason={warning!r}")
        print(f"[legal] legal_fallback_reason={warning!r}")
        return _fallback_output(pred, clean_country, warning, language), None

    summary = ""
    generation_score = 0.72
    legal_mode = "curated_fallback"
    fallback_reason: str | None = None
    try:
        if not attempted_llm:
            raise RuntimeError("GUARDIAN_LEGAL_LLM_ENABLED=0")
        print(
            "[legal] legal_llm_load_start "
            f"model_id={legal_model_id!r} local_path_valid={Path(legal_model_id).is_dir()}"
        )
        summary = _run_legal_llm(
            pred=pred,
            packet_summary=packet_summary,
            narrative=narrative,
            country=clean_country,
            language=language,
            act_type=act_type,
            severity=severity,
            vlm_summary=vlm_summary,
            records=records,
        )
        summary = _ensure_required_legal_language(summary, language)
        if not _guardrail_ok(summary):
            print("[legal] guardrail_status=failed")
            raise RuntimeError("legal LLM output failed safety guardrail")
        print("[legal] guardrail_status=passed")
        print("[legal] mode=llm")
        print("[legal] legal_llm_success")
        generation_score = 0.9
        legal_mode = "llm"
    except Exception as exc:
        fallback_reason = _safe_fallback_reason(exc)
        print(f"[legal] mode=curated_fallback reason={fallback_reason!r}")
        print(
            f"[legal] legal_fallback_reason={fallback_reason!r} "
            f"exception={exc.__class__.__name__}: {exc}"
        )
        summary = _deterministic_summary(
            pred=pred,
            country=clean_country,
            language=language,
            act_type=act_type,
            severity=severity,
            records=records,
        )
        print("[legal] guardrail_status=passed")
    print(f"[legal] fallback_reason={fallback_reason!r}")

    references = [
        _reference_from_record(record, rank=index)
        for index, record in enumerate(records[:3])
    ]
    retrieval_score = round(
        sum(float(reference["score"]) for reference in references) / max(len(references), 1),
        6,
    )
    payload = _finalize_curated_legal_payload(
        {
            "country": clean_country,
            "query_basis": _legal_query_basis(pred),
            "incident_context": _incident_context(pred, act_type, severity),
            "vlm_summary_used": bool(vlm_summary),
            "summary_source": "current_vlm" if vlm_summary else "telemetry",
            "vlm_people_count": vlm_summary.get("people_count") if vlm_summary else None,
            "vlm_violence_type": vlm_summary.get("violence_type") if vlm_summary else None,
            "retrieved_legal_references": references,
            "summary": summary,
            "guardrail_status": "passed",
            "limitations_note": _limitations_note(language),
        },
        pred=pred,
        act_type=act_type,
        severity=severity,
        records=records,
        language=language,
        legal_mode=legal_mode,
    )
    return (
        _with_debug_fields(
            payload,
            source="real",
            warning=None,
            legal_mode=legal_mode,
            reason_if_fallback=fallback_reason,
        ),
        {
            "retrieval_score": retrieval_score,
            "generation_score": generation_score,
            "overall_score": round((retrieval_score + generation_score) / 2, 6),
            "passed": True,
        },
    )


def normalize_country(country: str | None) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(country or "").casefold()).strip()
    return COUNTRY_ALIASES.get(normalized)


def _finalize_curated_legal_payload(
    output: dict[str, Any],
    *,
    pred: PredictResponse,
    act_type: str,
    severity: str,
    records: list[CuratedLegalRecord],
    language: str,
    legal_mode: str,
) -> dict[str, Any]:
    finalized = dict(output)
    real_weapon = legal_weapon_flag_for_prediction(pred)
    person_object = _person_object_leaked_into_legal_context(pred, finalized)
    if person_object and not real_weapon:
        finalized["retrieved_legal_references"] = [
            reference
            for reference in finalized.get("retrieved_legal_references", [])
            if not _is_weapon_reference(reference)
        ]
        query_basis = dict(finalized.get("query_basis") or {})
        query_basis.update(
            {
                "weapon_flag": False,
                "weapon_class": None,
                "dangerous_object_context": False,
            }
        )
        finalized["query_basis"] = query_basis
        incident_context = dict(finalized.get("incident_context") or {})
        incident_context.update(
            {
                "weapon_flag": False,
                "weapon_class": None,
                "dangerous_object_context": False,
                "act_type": "violent_conduct" if act_type == "weapon_or_dangerous_object" else act_type,
            }
        )
        finalized["incident_context"] = incident_context
        act_type = "violent_conduct" if act_type == "weapon_or_dangerous_object" else act_type

    finalized["summary"] = _compact_top_legal_summary(
        summary=str(finalized.get("summary") or ""),
        country=str(finalized.get("country") or ""),
        act_type=act_type,
        severity=severity,
        records=records,
        has_real_weapon=real_weapon and not person_object,
        language=language,
        legal_mode=legal_mode,
    )
    return finalized


def _person_object_leaked_into_legal_context(pred: PredictResponse, output: dict[str, Any]) -> bool:
    weapon = pred.telemetry.weapon
    if _is_person_object_class(weapon.cls):
        return True
    summary = str(output.get("summary") or "").casefold()
    if "(person)" in summary or "present (person)" in summary:
        return True
    for container_name in ("query_basis", "incident_context"):
        container = output.get(container_name)
        if isinstance(container, dict) and _is_person_object_class(container.get("weapon_class")):
            return True
    return False


def _is_weapon_reference(reference: dict[str, Any]) -> bool:
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


def _compact_top_legal_summary(
    *,
    summary: str,
    country: str,
    act_type: str,
    severity: str,
    records: list[CuratedLegalRecord],
    has_real_weapon: bool,
    language: str,
    legal_mode: str,
) -> str:
    if language == "ar":
        return _ensure_required_legal_language(summary, language)

    possible = ""
    if legal_mode == "llm" and (has_real_weapon or not _contains_weapon_offense_text(summary)):
        possible = _first_clean_legal_sentence(summary)
    if not possible and records:
        possible = records[0].consequence
    if not possible:
        possible = (
            "the incident may be reviewed under assault-related provisions. "
            "The outcome depends on evidence, harm, context, and the competent authority."
        )
    if "not legal advice" in possible.casefold():
        possible = re.sub(r"\bThis is not legal advice\.?", "", possible, flags=re.IGNORECASE).strip()
    possible = _clean_legal_possible_consequence(possible)
    if not possible:
        possible = (
            "the incident may be reviewed under assault-related provisions. "
            "The outcome depends on evidence, harm, context, and the competent authority."
        )
    if "if responsibility is confirmed by authorities" not in possible.casefold():
        possible = f"If responsibility is confirmed by authorities, {possible[:1].lower() + possible[1:]}"
    possible = _clean_legal_possible_consequence(possible) or (
        "If responsibility is confirmed by authorities, the incident may be reviewed under "
        "assault-related provisions. The outcome depends on evidence, harm, context, and "
        "the competent authority."
    )

    weapon_line = (
        "Weapon/object context: A weapon or dangerous object was identified."
        if has_real_weapon
        else "Weapon/object context: No weapon or dangerous object was identified."
    )
    return "\n".join(
        [
            f"Country: {country}",
            f"Relevant act: {_legal_act_label(act_type, has_real_weapon=has_real_weapon)}",
            weapon_line,
            f"Possible consequence: {possible}",
            "Limitation: This is legal context only, not legal advice.",
        ]
    )


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


def _first_clean_legal_sentence(summary: str) -> str:
    clean = re.sub(r"https?://\S+", "", str(summary or ""))
    clean = re.sub(r"\bSource\s*:\s*\S+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bWeapon/object context\s*:\s*present\s*\(person\)\.?", "", clean, flags=re.IGNORECASE)
    for sentence in re.findall(r"[^.!?\n]+[.!?]?", clean):
        stripped = re.sub(r"\s+", " ", sentence).strip()
        if stripped and "source:" not in stripped.casefold():
            return stripped
    return ""


def _contains_weapon_offense_text(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(
        term in lowered
        for term in (
            "offensive weapon",
            "weapon in a public place",
            "weapon-specific",
            "weapon/object context: present",
            "dangerous object",
        )
    )


def _legal_act_label(act_type: str, *, has_real_weapon: bool) -> str:
    if has_real_weapon and act_type == "weapon_or_dangerous_object":
        return "Weapon or dangerous-object related violent conduct"
    if act_type == "weapon_or_dangerous_object":
        return "Common assault / violent conduct"
    if act_type == "violent_conduct":
        return "Common assault / violent conduct"
    return act_type.replace("_", " ")


def legal_weapon_flag_for_prediction(pred: PredictResponse) -> bool:
    weapon = pred.telemetry.weapon
    if not bool(weapon.flag):
        return False
    return not _is_person_object_class(weapon.cls)


def legal_weapon_class_from_prediction(pred: PredictResponse) -> str | None:
    if not legal_weapon_flag_for_prediction(pred):
        return None
    weapon_class = pred.telemetry.weapon.cls
    return str(weapon_class).strip() if weapon_class else None


def _is_person_object_class(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    return normalized in PERSON_OBJECT_CLASSES


def retrieve_curated_legal_context(
    *,
    country: str,
    language: str,
    act_type: str,
    severity: str,
    top_k: int = 3,
) -> list[CuratedLegalRecord]:
    scored: list[tuple[float, CuratedLegalRecord]] = []
    preferred_language = "ar" if country in {"UAE", "KSA", "Egypt"} else "en"
    for record in CURATED_LEGAL_KB:
        if record.country != country:
            continue
        if act_type != "weapon_or_dangerous_object" and record.act_type == "weapon_or_dangerous_object":
            continue
        score = 0.45
        if record.act_type == act_type:
            score += 0.25
        if record.severity == severity:
            score += 0.2
        elif record.severity == "medium" and severity == "high":
            score += 0.08
        if record.language == preferred_language:
            score += 0.08
        if record.language == language:
            score += 0.02
        scored.append((min(score, 1.0), record))
    scored.sort(key=lambda item: (-item[0], item[1].act_type, item[1].severity))
    return [record for _, record in scored[:top_k]]


def infer_act_type(
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    *,
    vlm_summary: dict[str, Any] | None = None,
) -> str:
    vlm_summary = vlm_summary if isinstance(vlm_summary, dict) else {}
    legal_weapon_class = legal_weapon_class_from_prediction(pred)
    vlm_text = " ".join(
        [
            str(vlm_summary.get("violence_type") or ""),
            " ".join(map(str, vlm_summary.get("observed_actions") or [])),
            " ".join(
                str(obj)
                for obj in (vlm_summary.get("objects") or [])
                if not _is_person_object_class(obj)
            ),
            str(vlm_summary.get("visual_summary") or ""),
        ]
    ).casefold()
    if legal_weapon_class or any(term in vlm_text for term in DANGEROUS_OBJECT_TERMS):
        return "weapon_or_dangerous_object"
    if any(term in vlm_text for term in ("physical altercation", "confrontation", "pushing", "striking", "fight")):
        return "violent_conduct"
    text = f"{packet_summary} {narrative} {pred.telemetry.weapon.cls or ''}".casefold()
    if legal_weapon_class or any(term in text for term in CONCRETE_DANGEROUS_OBJECT_TERMS):
        return "weapon_or_dangerous_object"
    return "violent_conduct"


def infer_severity(
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    *,
    vlm_summary: dict[str, Any] | None = None,
) -> str:
    vlm_summary = vlm_summary if isinstance(vlm_summary, dict) else {}
    vlm_severity = str(vlm_summary.get("severity_estimate") or "").casefold()
    if vlm_severity in {"low", "medium", "high"}:
        return vlm_severity
    violence_type = str(vlm_summary.get("violence_type") or "").casefold()
    actions = " ".join(map(str, vlm_summary.get("observed_actions") or [])).casefold()
    if any(term in f"{violence_type} {actions}" for term in ("striking", "fight", "physical altercation")):
        return "medium"
    text = f"{packet_summary} {narrative}".casefold()
    if legal_weapon_flag_for_prediction(pred) or pred.confidence >= 0.82:
        return "high"
    if pred.telemetry.people >= 4 or any(term in text for term in ("high", "severe")):
        return "high"
    return "medium"


def _run_legal_llm(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    country: str,
    language: str,
    act_type: str,
    severity: str,
    vlm_summary: dict[str, Any],
    records: list[CuratedLegalRecord],
) -> str:
    if os.getenv("GUARDIAN_LEGAL_LLM_ENABLED", "1") == "0":
        raise RuntimeError("GUARDIAN_LEGAL_LLM_ENABLED=0")

    model_id = os.getenv(
        "GUARDIAN_LEGAL_LLM_MODEL_ID",
        os.getenv("GUARDIAN_LLM_MODEL_ID", DEFAULT_LEGAL_LLM_MODEL),
    )
    local_only = os.getenv("GUARDIAN_MODEL_LOCAL_ONLY", "1") == "1"
    max_new_tokens = int(os.getenv("GUARDIAN_LEGAL_LLM_MAX_NEW_TOKENS", "140"))
    model = None
    tokenizer = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[legal] loading_llm model_id={model_id!r} local_files_only={local_only}")
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
            low_cpu_mem_usage=True,
        )
        model.eval()

        messages = build_legal_prompt(
            pred=pred,
            packet_summary=packet_summary,
            narrative=narrative,
            country=country,
            language=language,
            act_type=act_type,
            severity=severity,
            vlm_summary=vlm_summary,
            records=records,
        )

        def generate_once(prompt_messages: list[dict[str, str]]) -> str:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = "\n\n".join(f"{message['role']}: {message['content']}" for message in prompt_messages)
                text += "\n\nassistant:"

            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
            return tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        answer = generate_once(messages)
        if not answer:
            raise RuntimeError("empty legal LLM answer")
        try:
            return polish_generated_text(answer, language, label="legal answer")
        except ValueError as exc:
            if language != "ar":
                raise
            print(f"[legal] arabic_validation_retry reason={exc}")
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Rewrite the legal summary in Modern Standard Arabic only. "
                        "Remove any Chinese or mixed-language text. Keep only the curated legal facts."
                    ),
                },
            ]
            retry_answer = generate_once(retry_messages)
            if not retry_answer:
                raise RuntimeError("empty legal LLM answer after Arabic retry")
            return polish_generated_text(retry_answer, language, label="legal answer")
    finally:
        if model is not None:
            del model
        gc.collect()
        _empty_cuda_cache()
        if tokenizer is not None:
            del tokenizer
        gc.collect()


def build_legal_prompt(
    *,
    pred: PredictResponse,
    packet_summary: str,
    narrative: str,
    country: str,
    language: str,
    act_type: str,
    severity: str,
    records: list[CuratedLegalRecord],
    vlm_summary: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    target_language = "Arabic" if language == "ar" else "English"
    language_guidance = ARABIC_ONLY_INSTRUCTION if language == "ar" else ""
    cautious_phrase = (
        "إذا أكدت السلطات المسؤولية"
        if language == "ar"
        else "If responsibility is confirmed by authorities"
    )
    context = {
        "country": country,
        "language": language,
        "act_type": act_type,
        "severity": severity,
        "incident_legal_factors": _legal_generation_factors(pred, act_type, severity),
        "vlm_incident_understanding": _legal_vlm_summary(vlm_summary),
        "curated_legal_context": [asdict(record) for record in records],
        "grounding_boundary": (
            "Curated legal context is the only legal source. The incident narrative "
            "is intentionally excluded so the legal answer does not repeat it."
        ),
    }
    context_text = json.dumps(context, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "You generate Guardian Eye legal consequence summaries from retrieved "
                "curated legal context only. Do not scrape websites. Do not identify "
                "attacker, victim, guilt, intent, or legal certainty. Do not invent law "
                "names, penalties, identities, injuries, exact sentencing ranges, or "
                "court outcomes. Do not repeat the incident narration. Do not use markdown "
                f"headings, bullets, bold markers, or raw section labels. {language_guidance}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Answer in {target_language}.\n\n"
                f"RETRIEVED CONTEXT ONLY:\n{context_text}\n\n"
                "Write 2 to 4 concise sentences. Include the cautious wording "
                f"'{cautious_phrase}'. Say this is not legal advice. Mention only "
                "the selected country, classifier verdict/confidence, act type, severity, "
                "weapon/object flag when present, and retrieved consequence guidance. "
                "Do not mention people count, frame numbers, dates, timing, or scene narration. "
                "Do not invent exact penalties unless they are explicitly present in "
                f"curated_legal_context. {language_guidance}"
            ),
        },
    ]


def _deterministic_summary(
    *,
    pred: PredictResponse,
    country: str,
    language: str,
    act_type: str,
    severity: str,
    records: list[CuratedLegalRecord],
) -> str:
    record = records[0]
    weapon_context = (
        "Weapon/object context: present."
        if legal_weapon_flag_for_prediction(pred)
        else "Weapon/object context: No weapon or dangerous object was identified."
    )
    if language == "ar":
        return (
            f"إذا أكدت السلطات المسؤولية، فقد تكون واقعة {country} المصنفة كـ "
            f"{_label_ar(act_type)} وبدرجة خطورة {_label_ar(severity)} مرتبطة بهذه "
            f"العواقب المحتملة: {record.consequence} يستند ذلك إلى سياق قانوني منسق: "
            f"{record.reference}. هذا ليس استشارة قانونية ولا يحدد هوية أي طرف أو الذنب أو نتيجة المحكمة."
        )
    return (
        f"If responsibility is confirmed by authorities, the {country} incident classified "
        f"as {act_type.replace('_', ' ')} with {severity} severity may be associated with "
        f"these possible consequences: {record.consequence} {weapon_context} This is grounded in curated "
        f"context: {record.reference}. This is not legal advice and does not identify any "
        "party, determine guilt, or predict a court outcome."
    )


def _non_violent_output(pred: PredictResponse, country: str, language: str) -> dict[str, Any]:
    summary = NON_VIOLENT_LEGAL_MESSAGE_AR if language == "ar" else NON_VIOLENT_LEGAL_MESSAGE
    return _with_debug_fields(
        {
            "country": normalize_country(country) or country.strip() if country else "",
            "query_basis": _legal_query_basis(pred),
            "incident_context": _incident_context(pred, "none", "none"),
            "retrieved_legal_references": [],
            "summary": summary,
            "guardrail_status": "passed",
            "limitations_note": _limitations_note(language),
            "vlm_summary_used": False,
            "summary_source": None,
            "vlm_people_count": None,
            "vlm_violence_type": None,
        },
        source="fallback",
        warning=None,
    )


def _missing_country_output(pred: PredictResponse, language: str) -> dict[str, Any]:
    summary = (
        "اختر دولة لطلب ملخص قانوني منسق."
        if language == "ar"
        else "Select a country to request a curated legal consequence summary."
    )
    return _with_debug_fields(
        {
            "country": "",
            "query_basis": _legal_query_basis(pred),
            "retrieved_legal_references": [],
            "summary": summary,
            "guardrail_status": "needs_review",
            "limitations_note": _limitations_note(language),
            "vlm_summary_used": False,
            "summary_source": None,
            "vlm_people_count": None,
            "vlm_violence_type": None,
        },
        source="fallback",
        warning="Country is required before curated legal retrieval can run.",
        legacy_warning="country_required",
    )


def _fallback_output(
    pred: PredictResponse,
    country: str,
    warning: str,
    language: str,
) -> dict[str, Any]:
    if language == "ar":
        summary = (
            f"لا يوجد سياق قانوني منسق متاح لـ {country}. لا يمكن اقتراح عواقب "
            "قانونية موثوقة دون مرجع محلي منسق. هذا ليس استشارة قانونية."
        )
    else:
        summary = (
            f"No curated legal context is available for {country}. Guardian Eye cannot "
            "suggest grounded legal consequences without a local curated reference. "
            "This is not legal advice."
        )
    return _with_debug_fields(
        {
            "country": country,
            "query_basis": _legal_query_basis(pred),
            "incident_context": _incident_context(pred, "unknown", "unknown"),
            "retrieved_legal_references": [],
            "summary": summary,
            "guardrail_status": "needs_review",
            "limitations_note": _limitations_note(language),
            "vlm_summary_used": False,
            "summary_source": None,
            "vlm_people_count": None,
            "vlm_violence_type": None,
        },
        source="fallback",
        warning=warning,
    )


def _reference_from_record(record: CuratedLegalRecord, rank: int) -> dict[str, Any]:
    score = max(0.55, 0.92 - rank * 0.08)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{record.country}-{record.act_type}-{record.severity}".casefold()).strip("-")
    return {
        "law_title": record.reference,
        "article_number": None,
        "section_title": record.act_type.replace("_", " "),
        "source_url": f"curated-legal-kb://{slug}",
        "snippet": f"{record.consequence} {record.disclaimer}",
        "score": round(score, 6),
        "country": record.country,
        "violence_category": record.act_type,
        "official_source": False,
    }


def _legal_query_basis(pred: PredictResponse) -> dict[str, Any]:
    return {
        "verdict": pred.verdict,
        "weapon_flag": legal_weapon_flag_for_prediction(pred),
        "weapon_class": legal_weapon_class_from_prediction(pred),
        "dangerous_object_context": legal_weapon_flag_for_prediction(pred),
    }


def _incident_context(pred: PredictResponse, act_type: str, severity: str) -> dict[str, Any]:
    return {
        "verdict": pred.verdict,
        "confidence": pred.confidence,
        "people": pred.telemetry.people,
        "peak_window": pred.telemetry.peak_window,
        "weapon_flag": legal_weapon_flag_for_prediction(pred),
        "weapon_class": legal_weapon_class_from_prediction(pred),
        "dangerous_object_context": legal_weapon_flag_for_prediction(pred),
        "act_type": act_type,
        "severity": severity,
    }


def _legal_generation_factors(pred: PredictResponse, act_type: str, severity: str) -> dict[str, Any]:
    weapon_flag = legal_weapon_flag_for_prediction(pred)
    return {
        "verdict": pred.verdict,
        "confidence": pred.confidence,
        "weapon_flag": weapon_flag,
        "weapon_class": legal_weapon_class_from_prediction(pred),
        "dangerous_object_context": weapon_flag,
        "weapon_object_context": (
            "Weapon/object context: present."
            if weapon_flag
            else "Weapon/object context: No weapon or dangerous object was identified."
        ),
        "act_type": act_type,
        "severity": severity,
    }


def _legal_vlm_summary(vlm_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(vlm_summary, dict) or not vlm_summary:
        return None
    return {
        "summary_source": vlm_summary.get("summary_source") or "current_vlm",
        "people_count": vlm_summary.get("people_count"),
        "observed_actions": vlm_summary.get("observed_actions") or [],
        "objects": vlm_summary.get("objects") or [],
        "violence_type": vlm_summary.get("violence_type"),
        "severity_estimate": vlm_summary.get("severity_estimate"),
        "visual_summary": vlm_summary.get("visual_summary"),
    }


def _limitations_note(language: str) -> str:
    if language == "ar":
        return "هذا ليس استشارة قانونية، ولا يحدد الذنب، ولا يتنبأ بنتيجة المحكمة."
    return LEGAL_LIMITATIONS_NOTE


def _with_debug_fields(
    output: dict[str, Any],
    *,
    source: str,
    warning: str | None,
    legal_mode: str = "curated_fallback",
    reason_if_fallback: str | None = None,
    legacy_warning: str | None = None,
) -> dict[str, Any]:
    enriched = dict(output)
    enriched["rag_mode"] = os.getenv("GUARDIAN_RAG_MODE", "auto").strip().lower()
    enriched["legal_rag_source"] = source
    enriched["legal_mode"] = legal_mode
    enriched["legal_rag_warning"] = warning
    enriched["reason_if_fallback"] = reason_if_fallback or warning
    enriched["warning"] = legacy_warning or warning
    return enriched


def _safe_fallback_reason(exc: Exception) -> str:
    text = str(exc).strip()
    if "GUARDIAN_LEGAL_LLM_ENABLED=0" in text:
        return "Legal LLM disabled"
    if "paging file is too small" in text.lower() or "os error 1455" in text.lower():
        return "Legal LLM could not load because Windows virtual memory is too low"
    if "local_files_only" in text or "not the path to a directory" in text:
        return "Legal LLM model unavailable locally"
    if "guardrail" in text.lower():
        return "Legal LLM output did not pass safety guardrails"
    return "Legal LLM unavailable"


def _ensure_required_legal_language(summary: str, language: str) -> str:
    clean = re.sub(r"\s+", " ", summary).strip()
    if language == "ar":
        cautious = "إذا أكدت السلطات المسؤولية"
        disclaimer = "هذا ليس استشارة قانونية."
        if cautious not in clean:
            clean = f"{cautious}، {clean}"
        if "استشارة قانونية" not in clean:
            clean = f"{clean} {disclaimer}"
        return clean

    cautious = "If responsibility is confirmed by authorities"
    if cautious.casefold() not in clean.casefold():
        clean = f"{cautious}, {clean[:1].lower() + clean[1:] if clean else clean}"
    if "not legal advice" not in clean.casefold():
        clean = f"{clean} This is not legal advice."
    return clean


def _guardrail_ok(summary: str) -> bool:
    text = summary.casefold()
    blocked = [
        "the attacker is",
        "the victim is",
        "is guilty",
        "was guilty",
        "will be convicted",
        "will be punished",
        "legal certainty",
        "هو المهاجم",
        "هي الضحية",
        "مذنب",
        "سيتم إدانته",
    ]
    return not any(term in text for term in blocked)


def _is_non_violent(pred: PredictResponse) -> bool:
    return str(pred.verdict).casefold().replace("_", "-") in {"non-violence", "nonviolent", "non-violent"}


def _label_ar(value: str) -> str:
    labels = {
        "weapon_or_dangerous_object": "سلاح أو أداة خطرة",
        "violent_conduct": "سلوك عنيف",
        "high": "مرتفعة",
        "medium": "متوسطة",
        "none": "غير منطبقة",
    }
    return labels.get(value, value)


def _empty_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
