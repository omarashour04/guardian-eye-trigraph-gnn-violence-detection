"""
Guardian Eye — Explanation Service
Phase 3 (MOCK_MODE=1): returns deterministic template narratives in EN/AR.
Phase 4 (MOCK_MODE=0): Qwen2.5-VL-7B-Instruct 4-bit + RAG store #1 retrieval.

Env vars (real mode):
  GUARDIAN_QWEN_CKPT   path or HF model ID (default: "Qwen/Qwen2.5-VL-7B-Instruct")
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

MOCK_MODE: bool = os.getenv("GUARDIAN_MOCK", "1") == "1"

# ── VLM singletons (loaded on first /explain call in real mode) ───────────────
_qwen_model     = None
_qwen_processor = None


def _resolve_rag_pipeline_path() -> Path:
    configured_path = os.getenv("GUARDIAN_RAG_PIPELINE_PATH", "").strip()
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "FULL_RAG_Pipeline"
        if candidate.exists():
            return candidate.resolve()

    return (Path(__file__).resolve().parent.parent / "FULL_RAG_Pipeline").resolve()

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
        "Guardian Eye classified this clip as non-violent with high confidence; "
        "the visible activity appears calm, and no clear violent behavior is visible."
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


def _fallback_explain(verdict: str, language: str, pred=None) -> str:
    if pred is not None:
        try:
            return _deterministic_explain_from_pred(pred, language)
        except Exception as e:
            print(f"[explanation] deterministic fallback failed: {e}")
    return _mock_explain(verdict, language)


def _deterministic_explain_from_pred(pred, language: str) -> str:
    verdict = pred.verdict
    confidence = round(_display_confidence_for_verdict(verdict, float(pred.confidence)) * 100)
    people = int(pred.telemetry.people)
    peak_window = pred.telemetry.peak_window
    weapon = pred.telemetry.weapon
    gate_items = [
        ("interaction", float(pred.gate.interaction)),
        ("skeleton", float(pred.gate.skeleton)),
        ("object", float(pred.gate.object)),
        ("video", float(pred.gate.vit)),
    ]
    gate_items.sort(key=lambda item: item[1], reverse=True)
    strongest = gate_items[:2]

    if language == "ar":
        return _deterministic_explain_ar(
            verdict, confidence, people, peak_window, weapon, strongest
        )
    return _deterministic_explain_en(
        verdict, confidence, people, peak_window, weapon, strongest
    )


def _display_confidence_for_verdict(verdict: str, confidence: float) -> float:
    bounded = max(0.0, min(1.0, confidence))
    if verdict == "non-violence":
        return 1.0 - bounded
    return bounded


def _deterministic_explain_en(
    verdict: str,
    confidence: int,
    people: int,
    peak_window: list[int],
    weapon,
    strongest: list[tuple[str, float]],
) -> str:
    people_word = "person" if people == 1 else "people"
    gate_text = _format_gate_contributors(strongest)
    window_text = f"frames {peak_window[0]}-{peak_window[1]}"

    if verdict == "violence":
        weapon_text = (
            f" A nearby object/proximity signal was also flagged"
            f"{f' (class: {weapon.cls})' if weapon.cls else ''}."
            if weapon.flag
            else " No weapon/object proximity was flagged."
        )
        return (
            f"The system classified this clip as violent with {confidence}% confidence. "
            f"It tracked {people} {people_word}, with the strongest activity in "
            f"{window_text}. The decision was driven mainly by {gate_text}, which "
            "were the strongest classifier signal streams "
            "from the current analyzed clip."
            f"{weapon_text}"
        )

    return (
        f"Guardian Eye classified this clip as non-violent with {confidence}% confidence; "
        "the visible activity appears calm, and no clear violent behavior is visible."
    )


def _deterministic_explain_ar(
    verdict: str,
    confidence: int,
    people: int,
    peak_window: list[int],
    weapon,
    strongest: list[tuple[str, float]],
) -> str:
    gate_text = _format_gate_contributors(strongest)
    window_text = f"الإطارات {peak_window[0]}-{peak_window[1]}"

    if verdict == "violence":
        weapon_text = (
            f" وتم أيضا رصد مؤشر قرب جسم/سلاح"
            f"{f' ({weapon.cls})' if weapon.cls else ''}."
            if weapon.flag
            else " ولم يتم رصد مؤشر قرب سلاح أو جسم خطر."
        )
        return (
            f"صنف النظام هذا المقطع على أنه عنيف بثقة {confidence}%. "
            f"تم تتبع {people} أشخاص، وكانت أقوى نافذة نشاط في {window_text}. "
            f"اعتمد القرار بشكل أساسي على {gate_text}، وهي إشارات مستخرجة من "
            f"المقطع الحالي نفسه.{weapon_text}"
        )

    return (
        f"صنف النظام هذا المقطع على أنه غير عنيف بثقة {confidence}%. "
        f"تم تتبع {people} أشخاص، وكانت نافذة التحليل الأبرز في {window_text}. "
        f"أقوى إشارات النموذج كانت {gate_text}، لكن الأدلة المجمعة لم تتجاوز "
        "عتبة العنف لهذا المقطع."
    )


def _format_gate_contributors(gate_items: list[tuple[str, float]]) -> str:
    labels = {
        "interaction": "interaction",
        "skeleton": "skeleton",
        "object": "object",
        "video": "video",
    }
    return " and ".join(
        f"{labels.get(name, name)} ({value:.0%})"
        for name, value in gate_items
    )


def _load_qwen() -> bool:
    """Load Qwen2.5-VL-7B-Instruct 4-bit into module singletons. Returns True on success."""
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return True

    model_id = os.getenv("GUARDIAN_QWEN_CKPT", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"[explanation] Loading Qwen2.5-VL from {model_id} (4-bit) ...")

    try:
        from transformers import AutoProcessor, BitsAndBytesConfig
        from qwen_vl_utils import process_vision_info

        # Try Qwen2_5_VLForConditionalGeneration first, fall back to AutoModel
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenCls
        except ImportError:
            from transformers import AutoModelForVision2Seq as QwenCls

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _qwen_model = QwenCls.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype="auto",
        )
        _qwen_model.eval()
        _qwen_processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=224*224, max_pixels=1280*28*28
        )
        print("[explanation] Qwen2.5-VL ready.")
        return True

    except Exception as e:
        print(f"[explanation] WARN: Could not load Qwen: {e}. Falling back to mock narrative.")
        return False


def _retrieve_rag_snippets(verdict: str, weapon_cls: str | None,
                            people: int, top_k: int = 3) -> str:
    """
    Retrieve reference snippets from RAG store #1 (reference_corpus via g2_reference_store).
    Falls back gracefully to empty string if the RAG pipeline is unavailable.
    """
    try:
        # Add the FULL_RAG_Pipeline to path so g2_reference_store is importable
        rag_pipeline_path = str(_resolve_rag_pipeline_path())
        print(f"[explanation] RAG pipeline path: {rag_pipeline_path}")
        if rag_pipeline_path not in sys.path:
            sys.path.insert(0, rag_pipeline_path)

        from g2_reference_store import retrieve_reference  # type: ignore

        query  = f"{verdict} {weapon_cls or ''} {people} people"
        hits   = retrieve_reference(query, top_k=top_k)
        snippets = "\n\n".join(
            f"[{h.get('title', 'Reference')}]\n{h.get('snippet', '')}"
            for h in hits
        )
        return snippets
    except Exception as e:
        print(f"[explanation] WARN: RAG retrieval failed: {e}")
        return ""


def _guardrail_ok(narrative: str, verdict: str) -> bool:
    """Return False if narrative explicitly contradicts the classifier verdict."""
    nl = narrative.lower()
    if verdict == "violence":
        contradiction_terms = ["non-violent", "non-violence", "no violence",
                                "peaceful", "not violent", "not a fight"]
        return not any(t in nl for t in contradiction_terms)
    else:
        contradiction_terms = ["violent", "fight", "attack", "assault", "weapon"]
        # Allow hedged mentions but flag if confident contradictions appear
        return not any(t in nl for t in contradiction_terms[:3])


def _run_qwen(frames_vit: "np.ndarray",
              packet_summary: str,
              rag_snippets: str,
              verdict: str,
              confidence: float) -> str:
    """
    Run Qwen2.5-VL-7B inference with the spec prompt template.
    frames_vit: [16, 224, 224, 3] uint8 RGB
    """
    import numpy as np
    import PIL.Image
    from qwen_vl_utils import process_vision_info  # type: ignore

    system_prompt = (
        "You explain a violence-detection result. You are a NARRATOR, not the judge. "
        f"The VERDICT is '{verdict.upper()}' and the CONFIDENCE is {confidence:.0%}. "
        "These are FINAL and come from a separate classifier — never contradict or re-decide them. "
        "Describe only what the frames and the evidence support. "
        "If person-level or timing detail is geometry-derived, hedge with 'appears to' or 'around frames X–Y'. "
        "Use the reference vocabulary. Do not invent objects or people not in the evidence. "
        "Keep your explanation to 3–5 sentences."
    )

    # Convert frames to PIL images for the VLM
    pil_images = [PIL.Image.fromarray(frames_vit[i]) for i in range(frames_vit.shape[0])]

    ref_section = f"\n\nREFERENCE (retrieved):\n{rag_snippets}" if rag_snippets else ""
    user_content = [
        {
            "type": "text",
            "text": (
                f"EVIDENCE PACKET:\n{packet_summary}"
                f"{ref_section}\n\n"
                "IMAGES (16 frames from the clip):"
            ),
        }
    ]
    for img in pil_images:
        user_content.append({"type": "image", "image": img})

    user_content.append({
        "type": "text",
        "text": "Explain what is happening in this clip, consistent with the verdict.",
    })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    text = _qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_qwen_model.device)

    with __import__("torch").no_grad():
        generated_ids = _qwen_model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    output_text = _qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def _real_explain(clip_id: str, verdict: str, language: str,
                  pred=None, return_status: bool = False):
    """
    Full real explanation:
      1. Prediction is already complete before this function is called.
      2. Export 16 evidence frames and build a structured evidence packet.
      3. Run VLM visual understanding, release it, then run the text LLM.
      4. Fall back to deterministic narration if any model step is unavailable.
      5. Guardrail check — regenerate once if narrative contradicts verdict
      6. Fallback to mock template if VLM unavailable or guardrail fails twice
    """
    if pred is None:
        narrative = _fallback_explain(verdict, language, pred=pred)
        if return_status:
            return {
                "narrative": narrative,
                "narration_mode": "fallback",
                "model_status": {"vlm": "not_loaded", "llm": "not_loaded"},
                "reason_if_fallback": "Prediction output unavailable",
                "vlm_summary": None,
            }
        return narrative

    try:
        from narration_service import generate_grounded_narration_result

        result = generate_grounded_narration_result(
            clip_id=clip_id,
            pred=pred,
            language=language,
            fallback=lambda: _fallback_explain(verdict, language, pred=pred),
        )
        if return_status:
            return {
                "narrative": result.narrative,
                "narration_mode": result.narration_mode,
                "model_status": result.model_status,
                "reason_if_fallback": result.reason_if_fallback,
                "vlm_summary": result.vlm_summary,
            }
        return result.narrative
    except Exception as e:
        print(f"[explanation] grounded narration failed: {e}")
        narrative = _fallback_explain(verdict, language, pred=pred)
        if return_status:
            return {
                "narrative": narrative,
                "narration_mode": "fallback",
                "model_status": {"vlm": "unavailable", "llm": "not_loaded"},
                "reason_if_fallback": "Narration runtime unavailable",
                "vlm_summary": None,
            }
        return narrative

    import numpy as np

    # Load NPZ for frames and packet context
    npz_path = Path("cache_npz") / f"{clip_id}.npz"
    frames_vit = None
    if npz_path.exists():
        try:
            d = np.load(str(npz_path), allow_pickle=False)
            frames_vit = d["frames_vit"]   # [16, 224, 224, 3] uint8 RGB
        except Exception as e:
            print(f"[explanation] WARN: Could not load NPZ for {clip_id}: {e}")

    if frames_vit is None:
        # No frames available — fall back to mock
        return _fallback_explain(verdict, language, pred=pred)

    # Build evidence packet
    packet_summary = ""
    if pred is not None:
        try:
            from incident_service import build_packet_summary
            packet_summary = build_packet_summary(pred)
        except Exception:
            pass

    # RAG snippet retrieval
    weapon_cls = None
    people     = 2
    if pred is not None:
        try:
            weapon_cls = pred.telemetry.weapon.cls
            people     = pred.telemetry.people
        except Exception:
            pass
    rag_snippets = _retrieve_rag_snippets(verdict, weapon_cls, people)

    # Load Qwen
    if not _load_qwen():
        return _fallback_explain(verdict, language, pred=pred)

    # Run VLM
    confidence = getattr(pred, "confidence", 0.0) if pred is not None else 0.0
    try:
        narrative = _run_qwen(frames_vit, packet_summary, rag_snippets,
                               verdict, confidence)
    except Exception as e:
        print(f"[explanation] VLM inference failed: {e}")
        return _fallback_explain(verdict, language, pred=pred)

    # Guardrail check
    if not _guardrail_ok(narrative, verdict):
        print("[explanation] Guardrail triggered — regenerating with stricter prompt.")
        try:
            stricter_packet = (
                f"IMPORTANT: The verdict is {verdict.upper()} and must NOT be contradicted.\n\n"
                + packet_summary
            )
            narrative = _run_qwen(frames_vit, stricter_packet, rag_snippets,
                                   verdict, confidence)
        except Exception:
            pass
        if not _guardrail_ok(narrative, verdict):
            print("[explanation] Guardrail failed twice — using mock narrative.")
            return _fallback_explain(verdict, language, pred=pred)

    # Arabic translation: Qwen supports Arabic natively — re-run in AR if needed
    if language == "ar":
        try:
            ar_prompt = f"Translate the following to Arabic:\n\n{narrative}"
            messages_tr = [{"role": "user", "content": ar_prompt}]
            text_tr = _qwen_processor.apply_chat_template(
                messages_tr, tokenize=False, add_generation_prompt=True
            )
            inputs_tr = _qwen_processor(text=[text_tr], return_tensors="pt")
            inputs_tr = inputs_tr.to(_qwen_model.device)
            with __import__("torch").no_grad():
                ids_tr = _qwen_model.generate(**inputs_tr, max_new_tokens=300)
            ids_tr_trimmed = [out[len(inp):] for inp, out in zip(inputs_tr.input_ids, ids_tr)]
            narrative = _qwen_processor.batch_decode(
                ids_tr_trimmed, skip_special_tokens=True
            )[0].strip()
        except Exception as e:
            print(f"[explanation] Arabic translation failed: {e}")
            # Return English narrative rather than falling back to mock

    return narrative


def generate_explanation(clip_id: str, verdict: str,
                         language: str = "en", pred=None,
                         return_status: bool = False):
    if MOCK_MODE:
        narrative = _fallback_explain(verdict, language, pred=pred)
        if return_status:
            return {
                "narrative": narrative,
                "narration_mode": "fallback",
                "model_status": {"vlm": "disabled", "llm": "disabled"},
                "reason_if_fallback": "Mock backend mode",
                "vlm_summary": None,
            }
        return narrative
    return _real_explain(
        clip_id,
        verdict,
        language,
        pred=pred,
        return_status=return_status,
    )


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


def answer_question(
    question: str,
    language: str = "en",
    db=None,
    clip_id: str | None = None,
    incident_id: str | None = None,
    country: str | None = None,
    return_status: bool = False,
):
    if db is not None:
        try:
            from ask_rag_service import answer_question_rag_result

            result = answer_question_rag_result(
                question=question,
                language=language,
                db=db,
                clip_id=clip_id,
                incident_id=incident_id,
                country=country,
                fallback=lambda: _answer_question_deterministic(
                    question,
                    language=language,
                    db=db,
                    clip_id=clip_id,
                    incident_id=incident_id,
                    country=country,
                ),
            )
            if return_status:
                return {
                    "answer": result.answer,
                    "ask_mode": result.ask_mode,
                    "selected_route": result.selected_route,
                    "retrieved_context_count": result.retrieved_context_count,
                    "reason_if_fallback": result.reason_if_fallback,
                    "related_incident_ids": list(result.related_incident_ids),
                    "grounding_label": result.grounding_label,
                    "vlm_summary_used": result.vlm_summary_used,
                    "summary_source": result.summary_source,
                    "vlm_people_count": result.vlm_people_count,
                    "vlm_violence_type": result.vlm_violence_type,
                }
            return result.answer
        except Exception as e:
            print(f"[explanation] ask RAG wrapper failed: {e}")

    answer = _answer_question_deterministic(
        question,
        language=language,
        db=db,
        clip_id=clip_id,
        incident_id=incident_id,
        country=country,
    )
    if return_status:
        return {
            "answer": answer,
            "ask_mode": "fallback",
            "selected_route": None,
            "retrieved_context_count": 0,
            "reason_if_fallback": "Deterministic Ask fallback",
            "related_incident_ids": [],
            "grounding_label": None,
            "vlm_summary_used": False,
            "summary_source": None,
            "vlm_people_count": None,
            "vlm_violence_type": None,
        }
    return answer


def _answer_question_deterministic(
    question: str,
    language: str = "en",
    db=None,
    clip_id: str | None = None,
    incident_id: str | None = None,
    country: str | None = None,
) -> str:
    if db is None:
        return _MOCK_ASK_AR if language == "ar" else _MOCK_ASK_EN

    # Real path: simple keyword → filter → retrieve → summarise
    try:
        from incident_service import get_by_clip, get_incident, query_incidents
        import datetime as _dt

        q_lower = question.lower()
        try:
            from ask_rag_service import route_question

            wants_history = route_question(question) == "history_memory"
        except Exception:
            wants_history = _asks_for_history(q_lower)

        if not wants_history:
            record = None
            if incident_id:
                record = get_incident(db, incident_id)
            if record is None and clip_id:
                record = get_by_clip(db, clip_id)
            if record is None:
                _, latest_records = query_incidents(db, limit=1)
                record = latest_records[0] if latest_records else None
            if record is not None:
                return _format_current_answer(question, record, language, country)
            return _no_current_incident_answer(language)

        from history_qa_service import (
            detect_question_language,
            format_history_answer,
            retrieve_history_matches,
        )

        current_record = get_incident(db, incident_id) if incident_id else None
        if current_record is None and clip_id:
            current_record = get_by_clip(db, clip_id)
        effective_language = detect_question_language(question, language)
        matches = retrieve_history_matches(
            db,
            question,
            current_record=current_record,
        )
        return format_history_answer(matches, effective_language, question)

        # Legacy Qwen summarization path is intentionally bypassed for the demo:
        # the answer above is deterministic and comes directly from SQLite.

        # Build a simple textual summary from the retrieved records
        lines = []
        for r in records:
            pw = r.peak_window()
            lines.append(
                f"On {r.timestamp.strftime('%Y-%m-%d %H:%M')} (source: {r.source}), "
                f"verdict={r.verdict}, confidence={r.confidence:.0%}, "
                f"people={r.people_count}, peak frames {pw[0]}–{pw[1]}, "
                f"weapon={'yes ('+r.weapon_class+')' if r.weapon_flag and r.weapon_class else 'none'}. "
                f"Narrative: {(r.narrative or '')[:200]}"
            )
        summary = " | ".join(lines)

        # If Qwen is already loaded, use it to produce a fluent answer
        if _qwen_model is not None:
            messages = [{
                "role": "user",
                "content": (
                    f"The user asked: \"{question}\"\n\n"
                    f"Here are the relevant incidents from the database:\n{summary}\n\n"
                    "Provide a concise, natural-language answer in "
                    f"{'Arabic' if language == 'ar' else 'English'}."
                ),
            }]
            text = _qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = _qwen_processor(text=[text], return_tensors="pt").to(_qwen_model.device)
            with __import__("torch").no_grad():
                ids = _qwen_model.generate(**inputs, max_new_tokens=200)
            ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, ids)]
            return _qwen_processor.batch_decode(ids_trimmed, skip_special_tokens=True)[0].strip()

        return summary  # plain text fallback

    except Exception as e:
        print(f"[explanation] answer_question failed: {e}")
        return _no_matching_incidents_answer(language)


def _asks_for_history(q_lower: str) -> bool:
    if _asks_current_incident_question(q_lower):
        return False
    triggers = [
        "history",
        "previous incident",
        "previous incidents",
        "last week",
        "week ago",
        "yesterday",
        "old incident",
        "old incidents",
        "compare incidents",
        "compare previous",
        "past events",
        "past event",
        "سجل",
        "السابقة",
        "سابق",
        "قديم",
        "القديمة",
        "قارن",
        "مقارنة",
        "كل الحوادث",
        "كل الفيديوهات",
        "الأسبوع الماضي",
        "اسبوع",
        "أمس",
    ]
    return any(trigger in q_lower for trigger in triggers)


def _asks_current_incident_question(q_lower: str) -> bool:
    triggers = [
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
    ]
    return any(trigger in q_lower for trigger in triggers)


def _no_current_incident_answer(language: str) -> str:
    if language == "ar":
        return (
            "لا أجد حادثة مرتبطة بالفيديو الحالي بعد. حلل الفيديو أولا أو اختر حادثة من السجل، "
            "ثم أعد السؤال."
        )
    return (
        "I do not have a saved incident for the current video yet. Analyze the video "
        "or select an incident from history, then ask again."
    )


def _format_current_answer(
    question: str,
    record,
    language: str,
    country: str | None,
) -> str:
    q_lower = question.lower()
    if language == "ar":
        return _format_current_answer_ar(q_lower, record, country)
    return _format_current_answer_en(q_lower, record, country)


def _format_current_answer_en(q_lower: str, record, country: str | None) -> str:
    peak_window = record.peak_window()
    confidence_pct = round(record.confidence * 100)
    gate_items = _ordered_gate_items(record)
    strongest = gate_items[0]
    secondary = gate_items[1] if len(gate_items) > 1 else gate_items[0]
    vlm_summary = _record_vlm_summary(record)

    if _asks_legal_question(q_lower):
        return _current_legal_answer(record, country, "en")

    if vlm_summary and _asks_people_question(q_lower):
        count = vlm_summary.get("people_count")
        if isinstance(count, int) and count > 0:
            noun = "person" if count == 1 else "people"
            return f"Guardian Eye detected {count} {noun} in the incident."

    if vlm_summary and _asks_initiator_question(q_lower):
        roles = vlm_summary.get("possible_roles") or []
        if roles:
            return (
                "Guardian Eye's visual analysis supports only hedged role descriptions: "
                f"{'; '.join(map(str, roles[:2]))}. Individual roles cannot be confirmed."
            )
        return "The frames do not clearly support identifying individual roles."

    if vlm_summary and _asks_what_happened_question(q_lower):
        return _format_vlm_summary_answer_en(vlm_summary, record)

    if _asks_weapon_question(q_lower):
        if record.weapon_flag:
            weapon = record.weapon_class or "object"
            return (
                f"Yes. The current video has a {weapon} object-proximity flag "
                f"around frames {peak_window[0]}-{peak_window[1]}. This is a "
                "model telemetry flag, not a legal conclusion."
            )
        return "No weapon/object proximity was flagged in the current video."

    if _asks_people_question(q_lower):
        noun = "person" if record.people_count == 1 else "people"
        return f"Guardian Eye detected {record.people_count} {noun} in the incident."

    if _asks_peak_question(q_lower):
        return (
            f"The current video's peak window is frames {peak_window[0]}-{peak_window[1]}. "
            "That is the highest-activity window saved by the analysis; timing details "
            "are model telemetry, not an independent confirmation of intent."
        )

    if _asks_stream_question(q_lower):
        return (
            f"The strongest current-video stream is {strongest[0]} ({strongest[1]:.0%}), "
            f"followed by {secondary[0]} ({secondary[1]:.0%}). Full gate weights are "
            f"{_format_gate_items_en(gate_items)}."
        )

    if _asks_security_question(q_lower):
        if record.verdict == "violence":
            return (
                "Security should treat this as a decision-support alert: review the "
                "current clip, verify the scene context, separate people only if it is "
                "safe and policy allows, preserve the video and incident ID, and escalate "
                "to a trained supervisor. Do not treat the model output as proof of guilt "
                "or a confirmed crime."
            )
        return (
            "Security should keep monitoring and verify context before taking action. "
            "The current clip was not classified as violent, so escalation should be "
            "based on human review, site policy, and any additional evidence."
        )

    if _asks_why_question(q_lower):
        if record.verdict == "violence":
            return (
                f"The current video was classified as violent with {confidence_pct}% "
                f"confidence. The strongest evidence was activity during frames "
                f"{peak_window[0]}-{peak_window[1]}, with {strongest[0]} "
                f"({strongest[1]:.0%}) and {secondary[0]} ({secondary[1]:.0%}) "
                f"signals contributing most. {_current_evidence_sentence(record)}"
            )
        return (
            f"The current video was classified as non-violent with {confidence_pct}% "
            f"confidence. The saved evidence did not show a strong violent interaction "
            f"pattern during the analyzed frames; {strongest[0]} was the strongest "
            f"stream at {strongest[1]:.0%}. {_current_evidence_sentence(record)}"
        )

    return (
        f"The current video verdict is {record.verdict} with {confidence_pct}% "
        f"confidence. It tracked {record.people_count} people, peak frames "
        f"{peak_window[0]}-{peak_window[1]}, and "
        f"{'did' if record.weapon_flag else 'did not'} flag weapon/object proximity."
    )


def _format_current_answer_ar(q_lower: str, record, country: str | None) -> str:
    peak_window = record.peak_window()
    confidence_pct = round(record.confidence * 100)
    gate_items = _ordered_gate_items(record)
    strongest = gate_items[0]
    secondary = gate_items[1] if len(gate_items) > 1 else gate_items[0]
    vlm_summary = _record_vlm_summary(record)

    if _asks_legal_question(q_lower):
        return _current_legal_answer(record, country, "ar")

    if vlm_summary and _asks_people_question(q_lower):
        count = vlm_summary.get("people_count")
        if isinstance(count, int) and count > 0:
            return f"يشير التحليل البصري من Guardian Eye إلى وجود {count} أفراد في الفيديو الحالي."

    if vlm_summary and _asks_initiator_question(q_lower):
        roles = vlm_summary.get("possible_roles") or []
        if roles:
            return (
                "يدعم التحليل البصري من Guardian Eye أوصافا حذرة فقط للأدوار المرئية: "
                f"{'؛ '.join(map(str, roles[:2]))}. وتبقى المسؤولية القانونية للمراجعة البشرية."
            )
        return "لا تدعم الإطارات بوضوح تحديد أدوار الأفراد."

    if vlm_summary and _asks_what_happened_question(q_lower):
        return _format_vlm_summary_answer_ar(vlm_summary, record)

    if _asks_weapon_question(q_lower):
        if record.weapon_flag:
            weapon = record.weapon_class or "جسم"
            return (
                f"نعم. يوجد في الفيديو الحالي مؤشر قرب جسم/سلاح ({weapon}) "
                f"حول الإطارات {peak_window[0]}-{peak_window[1]}. هذا مؤشر من "
                "النموذج وليس استنتاجا قانونيا."
            )
        return "لم يتم رصد مؤشر قرب سلاح أو جسم خطير في الفيديو الحالي."

    if _asks_people_question(q_lower):
        return (
            f"الفيديو الحالي يحتوي على {record.people_count} أشخاص تم تتبعهم "
            "أثناء التحليل."
        )

    if _asks_peak_question(q_lower):
        return (
            f"نافذة الذروة في الفيديو الحالي هي الإطارات {peak_window[0]}-{peak_window[1]}. "
            "هذه نافذة النشاط الأعلى المحفوظة من التحليل، وليست تأكيدا مستقلا للنية."
        )

    if _asks_stream_question(q_lower):
        return (
            f"أقوى تيار في الفيديو الحالي هو {_gate_label_ar(strongest[0])} "
            f"({strongest[1]:.0%})، يليه {_gate_label_ar(secondary[0])} "
            f"({secondary[1]:.0%}). أوزان البوابة هي "
            f"{_format_gate_items_ar(gate_items)}."
        )

    if _asks_security_question(q_lower):
        if record.verdict == "violence":
            return (
                "على الأمن التعامل مع النتيجة كتنبيه مساعد للقرار: مراجعة المقطع "
                "الحالي، التحقق من سياق المشهد، فصل الأشخاص فقط إذا كان ذلك آمنا "
                "ومتوافقا مع السياسة، حفظ الفيديو ورقم الحادثة، ثم التصعيد لمشرف "
                "مختص. لا تعتبر نتيجة النموذج دليلا على الذنب أو جريمة مؤكدة."
            )
        return (
            "على الأمن الاستمرار في المراقبة والتحقق من السياق قبل اتخاذ إجراء. "
            "الفيديو الحالي لم يصنف كعنيف، لذلك يجب أن يعتمد التصعيد على مراجعة "
            "بشرية وسياسة الموقع وأي أدلة إضافية."
        )

    if _asks_why_question(q_lower):
        verdict = "عنيف" if record.verdict == "violence" else "غير عنيف"
        if record.verdict == "violence":
            return (
                f"تم تصنيف الفيديو الحالي على أنه {verdict} بثقة {confidence_pct}%. "
                f"كانت نافذة النشاط الأعلى في الإطارات {peak_window[0]}-{peak_window[1]}، "
                f"مع مساهمة {_gate_label_ar(strongest[0])} ({strongest[1]:.0%}) "
                f"و{_gate_label_ar(secondary[0])} ({secondary[1]:.0%}) بشكل أكبر. "
                f"{_current_evidence_sentence(record)}"
            )
        return (
            f"تم تصنيف الفيديو الحالي على أنه {verdict} بثقة {confidence_pct}%. "
            f"الأدلة المحفوظة لم تظهر نمط تفاعل عنيف قوي في الإطارات المحللة؛ "
            f"أقوى تيار كان {_gate_label_ar(strongest[0])} بنسبة {strongest[1]:.0%}. "
            f"{_current_evidence_sentence(record)}"
        )

    if "weapon" in q_lower or "سلاح" in q_lower or "object" in q_lower:
        if record.weapon_flag:
            weapon = record.weapon_class or "جسم"
            return (
                f"نعم. تم رصد مؤشر قرب جسم/سلاح ({weapon}) في الفيديو الحالي "
                f"حول الإطارات {peak_window[0]}-{peak_window[1]}."
            )
        return "لم يتم رصد مؤشر قرب سلاح أو جسم خطر في الفيديو الحالي."

    if "legal" in q_lower or "قانون" in q_lower or "عقوب" in q_lower:
        return _current_legal_answer(record, country, "ar")

    if "why" in q_lower or "لماذا" in q_lower or "violent" in q_lower or "عنيف" in q_lower:
        verdict = "عنيف" if record.verdict == "violence" else "غير عنيف"
        return (
            f"تم تصنيف الفيديو الحالي على أنه {verdict} بثقة {confidence_pct}%. "
            f"أقوى دليل كان في الإطارات {peak_window[0]}-{peak_window[1]}، "
            "مع مساهمة إشارات التفاعل والهيكل العظمي بشكل أساسي."
        )

    if _asks_people_question(q_lower):
        return f"الفيديو الحالي يحتوي على {record.people_count} أشخاص تم تتبعهم أثناء التحليل."

    verdict = "عنيف" if record.verdict == "violence" else "غير عنيف"
    weapon_text = "مع مؤشر سلاح/جسم" if record.weapon_flag else "بدون مؤشر سلاح/جسم"
    return (
        f"نتيجة الفيديو الحالي هي {verdict} بثقة {confidence_pct}%. "
        f"تم تتبع {record.people_count} أشخاص، ونافذة الذروة هي الإطارات "
        f"{peak_window[0]}-{peak_window[1]}، {weapon_text}."
    )


def _current_legal_answer(record, country: str | None, language: str) -> str:
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
                weapon=WeaponInfo(
                    flag=record.weapon_flag,
                    cls=record.weapon_class,
                ),
            ),
        )
        legal_output, _ = build_legal_response(
            pred=pred,
            packet_summary=record.packet_summary or "",
            narrative=record.narrative or "",
            vlm_summary=record.vlm_summary(),
            country=country,
            language=language,
        )
        if isinstance(legal_output, dict):
            summary = legal_output.get("summary", "")
        else:
            summary = getattr(legal_output, "summary", "")
        return _compact_text(summary, 620)
    except Exception as e:
        print(f"[explanation] current legal answer failed: {e}")
        if language == "ar":
            return (
                "لا يمكن توليد ملخص قانوني مفصل الآن. قد تكون هذه الحادثة مرتبطة "
                "بمراجعة قانونية محتملة حسب البلد، وهذا ليس استشارة قانونية."
            )
        return (
            "A legal summary is not available right now. Based on the current incident, "
            "the legal path should be reviewed under the selected country's rules. "
            "This is not legal advice."
        )


def _record_vlm_summary(record) -> dict:
    try:
        summary = record.vlm_summary()
        return summary if isinstance(summary, dict) else {}
    except Exception:
        return {}


def _format_vlm_summary_answer_en(summary: dict, record) -> str:
    count = summary.get("people_count")
    people = f"{count} individuals" if isinstance(count, int) and count > 0 else "the visible individuals"
    violence_type = summary.get("violence_type") or "the observed event"
    actions = summary.get("observed_actions") or []
    action_text = ", ".join(map(str, actions[:3])) if actions else "visible interaction"
    visual = summary.get("visual_summary")
    if visual:
        return (
            f"Based on Guardian Eye's visual analysis, {people} are shown in {violence_type}. "
            f"Observed actions include {action_text}. {_compact_text(str(visual), 260)}"
        )
    return (
        f"Based on Guardian Eye's visual analysis, {people} are shown in {violence_type}, "
        f"with observed actions including {action_text}."
    )


def _format_vlm_summary_answer_ar(summary: dict, record) -> str:
    count = summary.get("people_count")
    people = f"{count} أفراد" if isinstance(count, int) and count > 0 else "الأفراد الظاهرون"
    violence_type = summary.get("violence_type") or "الحدث المرصود"
    actions = summary.get("observed_actions") or []
    action_text = "، ".join(map(str, actions[:3])) if actions else "تفاعل مرئي"
    visual = summary.get("visual_summary")
    if visual:
        return (
            f"وفقا للتحليل البصري من Guardian Eye، يظهر {people} في {violence_type}. "
            f"تشمل الأفعال المرصودة: {action_text}. {_compact_text(str(visual), 260)}"
        )
    return (
        f"وفقا للتحليل البصري من Guardian Eye، يظهر {people} في {violence_type}، "
        f"مع أفعال مرصودة تشمل: {action_text}."
    )


def _asks_people_question(q_lower: str) -> bool:
    terms = [
        "how many",
        "involved",
        "people",
        "person",
        "tracked",
        "count",
        "number of people",
        "اشخاص",
        "أشخاص",
        "كم",
        "عدد",
    ]
    return any(term in q_lower for term in terms)


def _asks_what_happened_question(q_lower: str) -> bool:
    terms = [
        "what happened",
        "describe",
        "summary",
        "summarize",
        "incident",
        "event",
        "Ù…Ø§Ø°Ø§ Ø­Ø¯Ø«",
        "Ù…Ø§ Ø§Ù„Ø°ÙŠ Ø­Ø¯Ø«",
        "ØµÙ",
        "Ù„Ø®Øµ",
    ]
    return any(term in q_lower for term in terms)


def _asks_initiator_question(q_lower: str) -> bool:
    terms = [
        "initiate",
        "initiated",
        "started",
        "who began",
        "who appears",
        "dominant",
        "defensive",
        "role",
        "roles",
        "Ø¨Ø¯Ø£",
        "Ù…Ù† Ø¨Ø¯Ø£",
        "Ø¯ÙˆØ±",
        "Ø£Ø¯ÙˆØ§Ø±",
    ]
    return any(term in q_lower for term in terms)


def _asks_weapon_question(q_lower: str) -> bool:
    terms = [
        "weapon",
        "object",
        "knife",
        "bottle",
        "armed",
        "سلاح",
        "جسم",
        "سكين",
        "زجاج",
    ]
    return any(term in q_lower for term in terms)


def _asks_legal_question(q_lower: str) -> bool:
    terms = [
        "legal",
        "consequence",
        "consequences",
        "law",
        "penalty",
        "liability",
        "court",
        "charge",
        "قانون",
        "قانوني",
        "عقوب",
        "مسؤولية",
        "محكمة",
    ]
    return any(term in q_lower for term in terms)


def _asks_peak_question(q_lower: str) -> bool:
    terms = [
        "peak",
        "window",
        "frame",
        "frames",
        "when",
        "time",
        "timestamp",
        "ذروة",
        "نافذة",
        "إطار",
        "اطار",
        "متى",
        "وقت",
    ]
    return any(term in q_lower for term in terms)


def _asks_stream_question(q_lower: str) -> bool:
    terms = [
        "stream",
        "contributed",
        "contribution",
        "contributor",
        "gate",
        "driver",
        "drivers",
        "signal",
        "signals",
        "dominant",
        "most",
        "تيار",
        "ساهم",
        "مساهمة",
        "بوابة",
        "إشارة",
        "اشارة",
        "الأقوى",
        "اقوى",
    ]
    return any(term in q_lower for term in terms)


def _asks_security_question(q_lower: str) -> bool:
    terms = [
        "security",
        "do next",
        "next step",
        "next steps",
        "respond",
        "response",
        "action",
        "recommend",
        "recommendation",
        "should do",
        "what should",
        "أمن",
        "الامن",
        "الأمن",
        "التالي",
        "ماذا نفعل",
        "يتصرف",
        "إجراء",
        "اجراء",
        "توصية",
    ]
    return any(term in q_lower for term in terms)


def _asks_why_question(q_lower: str) -> bool:
    terms = [
        "why",
        "classified",
        "classification",
        "violent",
        "violence",
        "non-violent",
        "nonviolent",
        "explain",
        "reason",
        "evidence",
        "لماذا",
        "صنف",
        "تصنيف",
        "عنيف",
        "العنف",
        "اشرح",
        "سبب",
        "دليل",
    ]
    return any(term in q_lower for term in terms)


def _ordered_gate_items(record) -> list[tuple[str, float]]:
    gate = record.gate_dict()
    items = [(name, float(value or 0.0)) for name, value in gate.items()]
    if not items:
        return [("unknown", 0.0)]
    return sorted(items, key=lambda item: item[1], reverse=True)


def _format_gate_items_en(items: list[tuple[str, float]]) -> str:
    return ", ".join(f"{name}={value:.0%}" for name, value in items)


def _format_gate_items_ar(items: list[tuple[str, float]]) -> str:
    return "، ".join(
        f"{_gate_label_ar(name)}={value:.0%}" for name, value in items
    )


def _gate_label_ar(name: str) -> str:
    labels = {
        "skeleton": "الهيكل",
        "interaction": "التفاعل",
        "object": "الأجسام",
        "vit": "الصورة/RGB",
        "video": "الصورة/RGB",
        "unknown": "غير معروف",
    }
    return labels.get(name, name)


def _current_evidence_sentence(record) -> str:
    if record.narrative:
        return f"Saved narrative: {_compact_text(record.narrative, 220)}"
    if record.packet_summary:
        return f"Saved evidence packet: {_compact_text(record.packet_summary, 220)}"
    return "No additional narrative is saved for this incident yet."


def _dominant_gate_label(record) -> str:
    gate = record.gate_dict()
    return max(gate.items(), key=lambda item: item[1])[0]


def _secondary_gate_label(record) -> str:
    gate = record.gate_dict()
    ordered = sorted(gate.items(), key=lambda item: item[1], reverse=True)
    return ordered[1][0] if len(ordered) > 1 else ordered[0][0]


def _no_matching_incidents_answer(language: str) -> str:
    if language == "ar":
        return (
            "لم أجد حوادث مطابقة في سجل العرض الحالي. "
            "حلل فيديو أولا أو استخدم بيانات العرض التجريبية ثم أعد السؤال."
        )
    return (
        "I could not find matching incidents in the current demo history. "
        "Analyze a video or use the seeded demo history, then ask again."
    )


def _format_history_answer(question: str, records, language: str) -> str:
    q_lower = question.lower()
    wants_why = "why" in q_lower
    wants_legal = "legal" in q_lower
    wants_weapon = "weapon" in q_lower

    if language == "ar":
        lines = [f"وجدت {len(records)} حادثة مطابقة في سجل العرض."]
        for record in records:
            lines.append(_format_record_ar(record, include_packet=wants_why or wants_legal))
        if wants_weapon and not any(record.weapon_flag for record in records):
            lines.append("لا توجد حادثة مطابقة تحتوي على مؤشر سلاح في النتائج الحالية.")
        if wants_legal:
            lines.append(
                "المسار القانوني يعرض ملخصا حذرا فقط عند اختيار دولة، ولا يعد استشارة قانونية."
            )
        return " ".join(lines)

    lines = [f"I found {len(records)} matching incident(s) in the demo history."]
    for record in records:
        lines.append(_format_record_en(record, include_packet=wants_why or wants_legal))
    if wants_weapon and not any(record.weapon_flag for record in records):
        lines.append("None of the matching incidents has a weapon flag.")
    if wants_legal:
        lines.append(
            "For legal consequences, select a country so the Legal RAG path can "
            "use retrieved references or clearly report fallback mode. This is "
            "not legal advice."
        )
    return " ".join(lines)


def _format_record_en(record, include_packet: bool) -> str:
    peak_window = record.peak_window()
    weapon = (
        f"weapon/object flag yes ({record.weapon_class})"
        if record.weapon_flag and record.weapon_class
        else "weapon/object flag no"
    )
    sentence = (
        f"On {record.timestamp.strftime('%Y-%m-%d %H:%M')} from {record.source}, "
        f"the verdict was {record.verdict} with {record.confidence:.0%} confidence; "
        f"{record.people_count} people were tracked, peak frames {peak_window[0]}-{peak_window[1]}, "
        f"{weapon}."
    )
    if include_packet and record.packet_summary:
        sentence += f" Evidence packet: {_compact_text(record.packet_summary, 220)}"
    elif record.narrative:
        sentence += f" Narrative: {_compact_text(record.narrative, 180)}"
    return sentence


def _format_record_ar(record, include_packet: bool) -> str:
    peak_window = record.peak_window()
    weapon = (
        f"يوجد مؤشر سلاح أو جسم ({record.weapon_class})"
        if record.weapon_flag and record.weapon_class
        else "لا يوجد مؤشر سلاح"
    )
    verdict = "عنف" if record.verdict == "violence" else "غير عنيف"
    sentence = (
        f"في {record.timestamp.strftime('%Y-%m-%d %H:%M')} من {record.source}، "
        f"كانت النتيجة {verdict} بثقة {record.confidence:.0%}; "
        f"تم تتبع {record.people_count} أشخاص، ونافذة الذروة هي الإطارات "
        f"{peak_window[0]}-{peak_window[1]}، و{weapon}."
    )
    if include_packet and record.packet_summary:
        sentence += f" ملخص الدليل: {_compact_text(record.packet_summary, 220)}"
    elif record.narrative:
        sentence += f" السرد: {_compact_text(record.narrative, 180)}"
    return sentence


def _compact_text(value: str, max_chars: int) -> str:
    clean = " ".join(str(value).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."
