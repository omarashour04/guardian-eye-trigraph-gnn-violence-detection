"""Deterministic, source-grounded question answering over saved incidents."""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import Any

from database import IncidentRecord


ARABIC_EMPTY_HISTORY_ANSWER = (
    "لا توجد حوادث سابقة كافية في السجل للإجابة بدقة. "
    "لا يمكنني اختراع حوادث غير موجودة في سجل التحليل."
)
ENGLISH_EMPTY_HISTORY_ANSWER = (
    "There is not enough stored history to answer accurately. "
    "I cannot invent incidents that are not in the analysis history."
)


@dataclass(frozen=True)
class HistoryMatch:
    record: IncidentRecord
    similarity: float | None
    reason_en: str
    reason_ar: str


def detect_question_language(question: str, requested_language: str = "en") -> str:
    """Arabic text wins over a stale UI language toggle."""
    if re.search(r"[\u0600-\u06FF]", question or ""):
        return "ar"
    return "ar" if requested_language == "ar" else "en"


def retrieve_history_matches(
    db,
    question: str,
    *,
    current_record: IncidentRecord | None = None,
    limit: int = 4,
) -> list[HistoryMatch]:
    """Retrieve only persisted records and rank similar-case queries structurally."""
    candidates = (
        db.query(IncidentRecord)
        .order_by(IncidentRecord.timestamp.desc())
        .limit(500)
        .all()
    )
    if current_record is not None:
        candidates = [
            record for record in candidates
            if record.incident_id != current_record.incident_id
        ]

    requested_verdict = _requested_verdict(question)
    if requested_verdict:
        candidates = [record for record in candidates if record.verdict == requested_verdict]

    from_ts, to_ts = _requested_time_range(question)
    if from_ts is not None:
        candidates = [
            record for record in candidates
            if record.timestamp is not None and record.timestamp >= from_ts
        ]
    if to_ts is not None:
        candidates = [
            record for record in candidates
            if record.timestamp is not None and record.timestamp < to_ts
        ]

    if _is_highest_confidence_query(question):
        ranked = [record for record in candidates if getattr(record, "confidence", None) is not None]
        ranked.sort(
            key=lambda record: (
                -float(record.confidence),
                -(record.timestamp.timestamp() if record.timestamp else 0.0),
            )
        )
        if not ranked:
            return []
        reason_en = "it has the highest stored confidence among the retrieved history records"
        reason_ar = "يمتلك أعلى نسبة ثقة محفوظة بين سجلات الحوادث المسترجعة"
        return [HistoryMatch(ranked[0], None, reason_en, reason_ar)]

    if _is_similar_query(question):
        if current_record is None:
            return []
        scored: list[HistoryMatch] = []
        for record in candidates:
            # A contrary classifier verdict is not presented as a similar incident.
            if record.verdict != current_record.verdict:
                continue
            score, reason_en, reason_ar = _similarity(record, current_record)
            if score >= 0.50:
                scored.append(HistoryMatch(record, score, reason_en, reason_ar))
        scored.sort(
            key=lambda match: (
                -(match.similarity or 0.0),
                -(match.record.timestamp.timestamp() if match.record.timestamp else 0.0),
            )
        )
        return scored if _is_count_query(question) else scored[:limit]

    reason_en, reason_ar = _filter_reason(requested_verdict, from_ts, to_ts)
    selected = candidates if _is_count_query(question) else candidates[:limit]
    return [HistoryMatch(record, None, reason_en, reason_ar) for record in selected]


def history_context_items(matches: list[HistoryMatch]) -> list[dict[str, Any]]:
    """Serialize exactly the fields an LLM would be allowed to see."""
    return [
        {
            "source": "stored_history_record",
            "incident_id": match.record.incident_id,
            "filename": match.record.source,
            "camera_or_source": match.record.source,
            "clip_id": match.record.clip_id,
            "verdict": match.record.verdict,
            "confidence": match.record.confidence,
            "timestamp": match.record.timestamp.isoformat() if match.record.timestamp else None,
            "peak_window": _safe_peak(match.record),
            "narration_or_summary": match.record.narrative or match.record.packet_summary,
            "similarity": match.similarity,
            "match_reason_en": match.reason_en,
            "match_reason_ar": match.reason_ar,
            "grounding_type": "saved_incident_record",
        }
        for match in matches
    ]


def history_computation_item(matches: list[HistoryMatch], question: str) -> dict[str, Any]:
    """Read-only deterministic result supplied to the history LLM."""
    intent = _history_intent(question)
    result: dict[str, Any] = {
        "intent": intent,
        "matching_record_count": len(matches),
        "matching_incident_ids": [match.record.incident_id for match in matches],
    }
    if intent == "highest_confidence" and matches:
        record = matches[0].record
        result["highest_confidence_record"] = {
            "incident_id": record.incident_id,
            "filename": record.source,
            "verdict": record.verdict,
            "confidence": getattr(record, "confidence", None),
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        }
    elif intent == "count_violence":
        result["violent_incident_count"] = len(matches)
    elif intent == "count_non_violence":
        result["non_violent_incident_count"] = len(matches)
    elif intent == "similar_incidents":
        result["ranked_similar_incidents"] = [
            {
                "incident_id": match.record.incident_id,
                "filename": match.record.source,
                "similarity": match.similarity,
            }
            for match in matches
        ]
    elif intent == "latest_incidents":
        result["latest_incident_ids"] = [match.record.incident_id for match in matches]
    return {
        "source": "grounded_history_computation",
        "read_only": True,
        "result": result,
        "grounding_type": "deterministic_history_statistic",
    }


def format_history_answer(matches: list[HistoryMatch], language: str, question: str = "") -> str:
    if not matches:
        return ARABIC_EMPTY_HISTORY_ANSWER if language == "ar" else ENGLISH_EMPTY_HISTORY_ANSWER
    return (
        _format_arabic(matches, question)
        if language == "ar"
        else _format_english(matches, question)
    )


def _format_arabic(matches: list[HistoryMatch], question: str) -> str:
    displayed = matches[:4]
    lines = [
        "الإجابة المختصرة",
        f"نعم. عدد الحوادث المطابقة في السجل: {len(matches)}.",
        "",
        "الحوادث المطابقة من السجل",
    ]
    for index, match in enumerate(displayed, 1):
        record = match.record
        lines.extend(
            [
                f"{index}. الملف: {_filename(record)}",
                f"   الحكم: {_verdict(record, 'ar')}",
                f"   الثقة: {_confidence(record, 'ar')}",
                f"   الطابع الزمني: {_timestamp(record, 'ar')}",
                f"   إطار/نطاق الذروة: {_peak_text(record, 'ar')}",
                f"   السرد/الملخص: {_summary(record, 'ar')}",
            ]
        )
    lines.extend(["", "سبب المطابقة"])
    for index, match in enumerate(displayed, 1):
        similarity = (
            f" درجة التشابه المحسوبة: {match.similarity:.0%}."
            if match.similarity is not None
            else ""
        )
        lines.append(f"{index}. {_filename(match.record)}: {match.reason_ar}{similarity}")
    if len(matches) > len(displayed):
        lines.append(f"عُرضت أول {len(displayed)} سجلات من أصل {len(matches)} سجلاً مطابقاً.")
    lines.extend(
        [
            "",
            "ملاحظة الدقة",
            "هذه الإجابة مبنية حصراً على سجلات الفيديو المحللة والمحفوظة أعلاه. "
            "أي حقل غير محفوظ يظهر على أنه غير متاح، ولم تُفترض حوادث أو أشخاص أو تواريخ.",
        ]
    )
    return "\n".join(lines)


def _format_english(matches: list[HistoryMatch], question: str) -> str:
    displayed = matches[:4]
    lines = [
        "Short answer",
        f"Yes. I found {len(matches)} matching incident record(s) in stored history.",
        "",
        "Matching stored incidents",
    ]
    for index, match in enumerate(displayed, 1):
        record = match.record
        lines.extend(
            [
                f"{index}. Filename: {_filename(record)}",
                f"   Verdict: {_verdict(record, 'en')}",
                f"   Confidence: {_confidence(record, 'en')}",
                f"   Timestamp: {_timestamp(record, 'en')}",
                f"   Peak frame/range: {_peak_text(record, 'en')}",
                f"   Narration/summary: {_summary(record, 'en')}",
            ]
        )
    lines.extend(["", "Why these match"])
    for index, match in enumerate(displayed, 1):
        similarity = (
            f" Computed similarity: {match.similarity:.0%}."
            if match.similarity is not None
            else ""
        )
        lines.append(f"{index}. {_filename(match.record)}: {match.reason_en}{similarity}")
    if len(matches) > len(displayed):
        lines.append(f"Showing the first {len(displayed)} of {len(matches)} matching stored records.")
    lines.extend(
        [
            "",
            "Accuracy note",
            "This answer uses only the stored analyzed-video records listed above. "
            "Missing fields are marked unavailable; no incidents, people, or dates were inferred.",
        ]
    )
    return "\n".join(lines)


def _similarity(record: IncidentRecord, current: IncidentRecord) -> tuple[float, str, str]:
    components: list[tuple[str, str, float]] = []
    score = 0.0
    if record.verdict == current.verdict:
        score += 0.50
        components.append(("same saved verdict", "نفس الحكم المحفوظ", 0.50))
    if bool(record.weapon_flag) == bool(current.weapon_flag):
        score += 0.12
        components.append(("same object/weapon flag", "نفس حالة مؤشر الجسم/السلاح", 0.12))
    people_delta = abs(int(record.people_count or 0) - int(current.people_count or 0))
    people_score = max(0.0, 0.10 - 0.025 * people_delta)
    if people_score:
        score += people_score
        components.append(("similar tracked-person count", "تقارب عدد الأشخاص المرصودين", people_score))
    gate_score = 0.18 * _cosine(record.gate_dict(), current.gate_dict())
    if gate_score:
        score += gate_score
        components.append(("similar model-stream pattern", "تقارب نمط مساهمات مسارات النموذج", gate_score))
    text_score = 0.10 * _jaccard(_record_words(record), _record_words(current))
    if text_score:
        score += text_score
        components.append(("overlapping saved summary terms", "تداخل كلمات الملخص المحفوظ", text_score))
    strongest = sorted(components, key=lambda item: item[2], reverse=True)[:3]
    return (
        min(1.0, score),
        ", ".join(item[0] for item in strongest) or "stored fields provide limited similarity evidence",
        "، ".join(item[1] for item in strongest) or "أدلة التشابه في الحقول المحفوظة محدودة",
    )


def _requested_verdict(question: str) -> str | None:
    q = (question or "").casefold()
    if any(term in q for term in ("non-violent", "nonviolent", "non-violence", "calm", "peaceful", "غير عنيف", "غير عنيفة", "غير العنيفة", "سلمي", "هادئ")):
        return "non-violence"
    if any(term in q for term in ("violent", "violence", "fight", "fighting", "attack", "assault", "عنف", "عنيف", "عنيفة", "شجار", "مشاجرة", "اعتداء")):
        return "violence"
    return None


def _is_similar_query(question: str) -> bool:
    q = (question or "").casefold()
    return any(term in q for term in ("similar", "like this", "matching current", "مشابه", "متشابه", "مماثل"))


def _is_count_query(question: str) -> bool:
    q = (question or "").casefold()
    return any(term in q for term in ("how many", "count", "number of", "كم عدد", "عدد الحوادث"))


def _is_highest_confidence_query(question: str) -> bool:
    q = (question or "").casefold()
    confidence = any(term in q for term in ("confidence", "نسبة الثقة", "الثقة"))
    highest = any(term in q for term in ("highest", "maximum", "top", "أعلى", "اعلى"))
    return confidence and highest


def _history_intent(question: str) -> str:
    if _is_highest_confidence_query(question):
        return "highest_confidence"
    verdict = _requested_verdict(question)
    if _is_count_query(question) and verdict == "violence":
        return "count_violence"
    if _is_count_query(question) and verdict == "non-violence":
        return "count_non_violence"
    if _is_similar_query(question):
        return "similar_incidents"
    q = (question or "").casefold()
    if any(term in q for term in ("latest", "most recent", "آخر الحوادث", "أحدث", "احدث")):
        return "latest_incidents"
    return "history_lookup"


def _requested_time_range(question: str) -> tuple[dt.datetime | None, dt.datetime | None]:
    q = (question or "").casefold()
    now = dt.datetime.utcnow()
    if any(term in q for term in ("this week", "هذا الأسبوع", "هذا الاسبوع")):
        return now - dt.timedelta(days=7), None
    if any(term in q for term in ("last week", "week ago", "الأسبوع الماضي", "الاسبوع الماضي")):
        return now - dt.timedelta(days=14), now - dt.timedelta(days=7)
    if any(term in q for term in ("yesterday", "أمس", "امس")):
        today = dt.datetime.combine(now.date(), dt.time.min)
        return today - dt.timedelta(days=1), today
    if any(term in q for term in ("today", "اليوم")):
        return dt.datetime.combine(now.date(), dt.time.min), None
    return None, None


def _filter_reason(verdict: str | None, from_ts: dt.datetime | None, to_ts: dt.datetime | None) -> tuple[str, str]:
    if verdict:
        return (
            f"the stored verdict is exactly {verdict}, as requested",
            f"الحكم المحفوظ هو {verdict} كما طلب السؤال",
        )
    if from_ts is not None or to_ts is not None:
        return ("the timestamp is inside the requested period", "الطابع الزمني داخل الفترة المطلوبة")
    return ("it is a recent saved analysis record", "هو سجل تحليل محفوظ وحديث")


def _filename(record: IncidentRecord) -> str:
    return str(record.source or record.clip_id or "unavailable")


def _verdict(record: IncidentRecord, language: str) -> str:
    value = str(record.verdict or "").strip()
    if not value:
        return "غير متاح" if language == "ar" else "unavailable"
    if language == "ar":
        translation = "عنيف" if value == "violence" else "غير عنيف" if value == "non-violence" else ""
        return f"{value} ({translation})" if translation else value
    return value


def _confidence(record: IncidentRecord, language: str) -> str:
    value = getattr(record, "confidence", None)
    if value is None:
        return "غير متاحة" if language == "ar" else "unavailable"
    numeric = float(value)
    return f"{numeric:.1%} ({numeric:g})"


def _timestamp(record: IncidentRecord, language: str) -> str:
    if record.timestamp is None:
        return "غير متاح" if language == "ar" else "unavailable"
    return record.timestamp.isoformat(sep=" ", timespec="seconds")


def _safe_peak(record: IncidentRecord) -> list[int] | None:
    try:
        peak = record.peak_window()
        return peak if isinstance(peak, list) and len(peak) >= 2 else None
    except Exception:
        return None


def _peak_text(record: IncidentRecord, language: str) -> str:
    peak = _safe_peak(record)
    if not peak:
        return "غير متاح" if language == "ar" else "unavailable"
    return f"{peak[0]}-{peak[1]}"


def _summary(record: IncidentRecord, language: str) -> str:
    value = " ".join(str(record.narrative or record.packet_summary or "").split())
    if not value:
        return "غير متاح" if language == "ar" else "unavailable"
    return value[:240] + ("…" if len(value) > 240 else "")


def _record_words(record: IncidentRecord) -> set[str]:
    text = " ".join((record.narrative or "", record.packet_summary or ""))
    return set(re.findall(r"[\w\u0600-\u06FF]{3,}", text.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    left_values = [float(left.get(key) or 0.0) for key in keys]
    right_values = [float(right.get(key) or 0.0) for key in keys]
    denominator = math.sqrt(sum(value * value for value in left_values)) * math.sqrt(
        sum(value * value for value in right_values)
    )
    if denominator <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left_values, right_values)) / denominator))
