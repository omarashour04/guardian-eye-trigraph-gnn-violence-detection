"""
Guardian Eye — Explanation Service
Phase 3: returns mock narratives in EN/AR.
Phase 5: replace _real_explain() with Qwen2.5-VL-7B-Instruct 4-bit + ChromaDB RAG.
"""

from __future__ import annotations
import os

MOCK_MODE: bool = os.getenv("GUARDIAN_MOCK", "1") == "1"

_MOCK_EN = {
    "violence": (
        "The system detected a violent altercation with high confidence. "
        "Two individuals were tracked throughout the clip; they closed distance rapidly "
        "around frames 14–22 (~44–69% of the clip), with a sharp relative-speed spike "
        "consistent with rapid physical contact. An object resembling a bottle was "
        "detected near person A's wrist during frames 16–20. The decision was driven "
        "primarily by the interaction stream (41%) and skeleton stream (34%)."
    ),
    "non-violence": (
        "The system classified this clip as non-violent with high confidence. "
        "Two individuals are present and moving at normal speeds throughout; no rapid "
        "closing distance, speed spikes, or weapon proximity were detected. "
        "Both the interaction and skeleton streams showed calm, routine movement patterns."
    ),
}

_MOCK_AR = {
    "violence": (
        "اكتشف النظام مشادة عنيفة بنسبة ثقة عالية. "
        "تمّ تتبع شخصين طوال المقطع؛ اقتربا من بعضهما بسرعة في الإطارات 14–22، "
        "مع ارتفاع حاد في السرعة النسبية يتسق مع احتكاك جسدي مباشر. "
        "رُصد جسم يشبه الزجاجة بالقرب من معصم الشخص الأول خلال الإطارات 16–20. "
        "استندت قرارات النظام بشكل رئيسي إلى تيار التفاعل (41٪) وتيار الهيكل العظمي (34٪)."
    ),
    "non-violence": (
        "صنّف النظام هذا المقطع على أنه غير عنيف بثقة عالية. "
        "يظهر في المقطع شخصان يتحركان بسرعات طبيعية دون أي اقتراب مفاجئ "
        "أو ارتفاع في السرعة أو تقارب من أسلحة. "
        "أظهر كلا تياري التفاعل والهيكل العظمي أنماط حركة هادئة واعتيادية."
    ),
}


def _mock_explain(verdict: str, language: str) -> str:
    pool = _MOCK_AR if language == "ar" else _MOCK_EN
    return pool.get(verdict, pool["non-violence"])


def _real_explain(clip_id: str, verdict: str, language: str) -> str:
    """
    TODO (Phase 5): Integrate Qwen2.5-VL-7B-Instruct 4-bit.
      1. Build evidence packet (done by incident_service.build_packet_summary).
      2. Retrieve reference snippets from ChromaDB (RAG store #1).
      3. Load 16 frames_vit from NPZ (uint8 [16,224,224,3]).
      4. Run VLM with prompt from EXPLANATION_RAG_SYSTEM.md §8.
      5. Validate guardrails: narrative must not flip verdict.
    """
    raise NotImplementedError("VLM not wired yet — set GUARDIAN_MOCK=1")


def generate_explanation(clip_id: str, verdict: str, language: str = "en") -> str:
    if MOCK_MODE:
        return _mock_explain(verdict, language)
    return _real_explain(clip_id, verdict, language)


# ── /ask ─────────────────────────────────────────────────────────────────────

_MOCK_ASK_EN = (
    "About a week ago (camera 3), the system flagged a violent incident with 94% confidence. "
    "Two people were involved; they closed distance rapidly around the middle of the clip, "
    "and an object resembling a bottle was detected near one person's hand. "
    "The interaction stream was the dominant decision driver."
)

_MOCK_ASK_AR = (
    "منذ حوالي أسبوع (الكاميرا 3)، رصد النظام حادثة عنف بنسبة ثقة 94٪. "
    "تورّط شخصان في الحادثة؛ اقتربا من بعضهما بسرعة في منتصف المقطع تقريباً، "
    "ورُصد جسم يشبه الزجاجة بالقرب من يد أحدهما. "
    "كان تيار التفاعل هو العامل الرئيسي في اتخاذ القرار."
)


def answer_question(question: str, language: str = "en") -> str:
    if MOCK_MODE:
        return _MOCK_ASK_AR if language == "ar" else _MOCK_ASK_EN
    raise NotImplementedError("RAG QA not wired yet — set GUARDIAN_MOCK=1")
