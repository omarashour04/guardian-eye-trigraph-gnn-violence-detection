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
        rag_pipeline_path = os.getenv(
            "GUARDIAN_RAG_PIPELINE_PATH",
            str(Path(__file__).parent.parent / "FULL_RAG_Pipeline"),
        )
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
                  pred=None) -> str:
    """
    Full real explanation:
      1. Load frames_vit from NPZ
      2. Get evidence packet (from pred if available)
      3. Retrieve RAG snippets from reference corpus
      4. Run Qwen2.5-VL-7B-Instruct 4-bit
      5. Guardrail check — regenerate once if narrative contradicts verdict
      6. Fallback to mock template if VLM unavailable or guardrail fails twice
    """
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
        return _mock_explain(verdict, language)

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
        return _mock_explain(verdict, language)

    # Run VLM
    confidence = getattr(pred, "confidence", 0.0) if pred is not None else 0.0
    try:
        narrative = _run_qwen(frames_vit, packet_summary, rag_snippets,
                               verdict, confidence)
    except Exception as e:
        print(f"[explanation] VLM inference failed: {e}")
        return _mock_explain(verdict, language)

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
            return _mock_explain(verdict, language)

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
                         language: str = "en", pred=None) -> str:
    if MOCK_MODE:
        return _mock_explain(verdict, language)
    return _real_explain(clip_id, verdict, language, pred=pred)


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


def answer_question(question: str, language: str = "en",
                    db=None) -> str:
    if MOCK_MODE:
        return _MOCK_ASK_AR if language == "ar" else _MOCK_ASK_EN

    # Real path: simple keyword → filter → retrieve → summarise
    try:
        from incident_service import query_incidents
        import datetime as _dt

        q_lower = question.lower()
        verdict_filter = None
        if any(w in q_lower for w in ["fight", "violen", "attack", "assault"]):
            verdict_filter = "violence"
        elif any(w in q_lower for w in ["peaceful", "normal", "calm", "non-viol"]):
            verdict_filter = "non-violence"

        # Rough time-window parsing
        from_ts, to_ts = None, None
        now = _dt.datetime.utcnow()
        if "week" in q_lower:
            from_ts = now - _dt.timedelta(days=8)
            to_ts   = now - _dt.timedelta(days=6)
        elif "yesterday" in q_lower:
            from_ts = now - _dt.timedelta(days=2)
            to_ts   = now - _dt.timedelta(days=1)
        elif "today" in q_lower:
            from_ts = now - _dt.timedelta(hours=24)

        if db is not None:
            _, records = query_incidents(
                db,
                verdict=verdict_filter,
                from_ts=from_ts,
                to_ts=to_ts,
                limit=3,
            )
        else:
            records = []

        if not records:
            fallback = (_MOCK_ASK_AR if language == "ar" else _MOCK_ASK_EN)
            return fallback

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
        return _MOCK_ASK_AR if language == "ar" else _MOCK_ASK_EN
