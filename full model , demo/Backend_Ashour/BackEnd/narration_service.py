"""
Guardian Eye grounded narration service.

This module runs only after prediction has completed. It exports the 16 cached
evidence frames and asks Qwen-VL for the English narration. A text LLM is used only
when that VLM output is invalid. Arabic output uses the local Qwen translator after
the narration models have been released.
"""

from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from text_quality import ARABIC_ONLY_INSTRUCTION, polish_generated_text


def _default_local_model(directory_name: str, remote_fallback: str) -> str:
    local_model = Path(__file__).resolve().parents[2] / "models" / directory_name
    if local_model.is_dir():
        return str(local_model)
    return remote_fallback


DEFAULT_VLM_MODEL = _default_local_model(
    "Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
)
DEFAULT_LLM_MODEL = _default_local_model(
    "qwen2.5-1.5b-instruct",
    "Qwen/Qwen2.5-3B-Instruct-AWQ",
)

ROLE_DESCRIPTION_GUIDANCE_EN = (
    "Visible people may be described only by clearly visible appearance, location, "
    "and observable action. Always hedge with phrases such as 'appears to', "
    "'seems to', or 'may'. If individual roles are unclear, say: 'The frames do "
    "not clearly support identifying individual roles.' Never label anyone as "
    "attacker, victim, guilty, legally responsible, or as having started the fight."
)

ROLE_DESCRIPTION_GUIDANCE_AR = (
    "يمكن وصف الأشخاص الظاهرين فقط من خلال المظهر أو الموقع أو الفعل المرئي بوضوح. "
    "استخدم دائما صياغة حذرة مثل: يبدو أن، أو قد، أو ربما. إذا كانت الأدوار الفردية "
    "غير واضحة فقل: لا تدعم الإطارات بوضوح تحديد أدوار الأفراد. لا تصف أي شخص "
    "كمهاجم أو ضحية أو مذنب أو مسؤول قانونيا أو بأنه بدأ الشجار."
)


@dataclass(frozen=True)
class EvidenceFrames:
    indices: list[int]
    paths: list[str]
    peak_vit_indices: list[int]


@dataclass(frozen=True)
class ReferenceEntry:
    ref_id: str
    title: str
    tags: tuple[str, ...]
    guidance_en: str
    guidance_ar: str


@dataclass(frozen=True)
class NarrationResult:
    narrative: str
    narration_mode: Literal["vlm_llm", "vlm_only", "fallback"]
    model_status: dict[str, str]
    reason_if_fallback: str | None = None
    vlm_summary: dict[str, Any] | None = None


REFERENCE_KB: tuple[ReferenceEntry, ...] = (
    ReferenceEntry(
        ref_id="classifier-finality",
        title="Classifier finality",
        tags=("all", "verdict", "confidence", "grounding"),
        guidance_en=(
        "Treat the Guardian Eye classifier verdict and confidence as final decision-support "
            "metadata. Do not reclassify the clip from visual impressions."
        ),
        guidance_ar=(
            "اعتبر حكم المصنف ونسبة الثقة بيانات دعم قرار نهائية. لا تعد تصنيف "
            "المقطع اعتمادا على الانطباع البصري فقط."
        ),
    ),
    ReferenceEntry(
        ref_id="visual-grounding",
        title="Visual grounding",
        tags=("all", "frames", "grounding", "visual"),
        guidance_en=(
            "Describe only what is visible in the selected evidence frames or what "
            "is explicitly present in telemetry. Use uncertain wording for unclear visuals."
        ),
        guidance_ar=(
            "صف فقط ما يظهر في إطارات الدليل المختارة أو ما يوجد صراحة في القياسات. "
            "استخدم صياغة حذرة عند عدم وضوح المشهد."
        ),
    ),
    ReferenceEntry(
        ref_id="violent-alert-high-confidence",
        title="High-confidence violent alert",
        tags=("violence", "high", "confidence", "severity"),
        guidance_en=(
            "For a high-confidence violence verdict, state that the system classified "
            "the clip as violent, then ground the explanation in visible people "
            "count, visible motion, and strongest model streams."
        ),
        guidance_ar=(
            "عند وجود حكم عنف بثقة عالية، اذكر أن النظام صنف المقطع كعنيف، ثم اربط "
            "التفسير بنافذة الذروة وعدد الأشخاص والحركة المرئية وأقوى مسارات النموذج."
        ),
    ),
    ReferenceEntry(
        ref_id="violent-alert-low-confidence",
        title="Lower-confidence violent alert",
        tags=("violence", "low", "medium", "confidence", "cautious"),
        guidance_en=(
            "For a lower-confidence violence verdict, keep language cautious. Say the "
            "system flagged the clip as violent, but avoid strong claims about exact actions."
        ),
        guidance_ar=(
            "عند وجود حكم عنف بثقة أقل، اجعل اللغة حذرة. قل إن النظام أشار إلى "
            "المقطع كعنيف، وتجنب الجزم بأفعال دقيقة."
        ),
    ),
    ReferenceEntry(
        ref_id="non-violence-threshold",
        title="Non-violence threshold explanation",
        tags=("non-violence", "threshold", "confidence", "calm"),
        guidance_en=(
            "For a non-violence verdict, explain that the combined evidence did not "
            "pass the violence threshold. Do not describe assault, attack, or weapon use "
            "unless the packet explicitly supports it."
        ),
        guidance_ar=(
            "عند حكم عدم العنف، وضح أن الأدلة المجمعة لم تتجاوز عتبة العنف. لا تصف "
            "اعتداء أو هجوما أو استخدام سلاح إلا إذا دعمت الحزمة ذلك صراحة."
        ),
    ),
    ReferenceEntry(
        ref_id="object-proximity",
        title="Object proximity caveat",
        tags=("weapon", "object", "proximity", "telemetry", "cautious"),
        guidance_en=(
            "Object or weapon telemetry is a proximity flag, not proof of use. Say "
            "an object was flagged near a person only when the packet says so."
        ),
        guidance_ar=(
            "قياس الجسم أو السلاح هو إشارة قرب وليس دليلا على الاستخدام. اذكر قرب "
            "جسم من شخص فقط عندما تنص الحزمة على ذلك."
        ),
    ),
    ReferenceEntry(
        ref_id="multi-person-scene",
        title="Multi-person scene",
        tags=("people", "crowd", "multiple", "telemetry"),
        guidance_en=(
            "When several people are tracked, avoid assigning attacker or victim roles. "
            "Refer to tracked people or visible individuals instead."
        ),
        guidance_ar=(
            "عند تتبع عدة أشخاص، تجنب تعيين أدوار مهاجم أو ضحية. استخدم عبارات مثل "
            "الأشخاص المتتبعين أو الأفراد الظاهرين."
        ),
    ),
    ReferenceEntry(
        ref_id="gate-contributions",
        title="Gate contribution language",
        tags=("gate", "stream", "skeleton", "interaction", "object", "vit"),
        guidance_en=(
            "Gate weights describe model contribution streams, not independent facts. "
            "Use them to explain why the model leaned toward the verdict."
        ),
        guidance_ar=(
            "أوزان البوابات تصف مساهمة مسارات النموذج وليست حقائق مستقلة. استخدمها "
            "لتفسير سبب ميل النموذج إلى الحكم."
        ),
    ),
    ReferenceEntry(
        ref_id="peak-window-caveat",
        title="Peak window caveat",
        tags=("peak", "window", "timing", "telemetry"),
        guidance_en=(
            "Peak-window timing is telemetry. Phrase it as the highest-activity window, "
            "not as confirmed start or end time of an incident."
        ),
        guidance_ar=(
            "توقيت نافذة الذروة هو قياس آلي. صغه كنافذة أعلى نشاط، وليس كتأكيد لبداية "
            "أو نهاية حادثة."
        ),
    ),
)


def generate_grounded_narration(
    *,
    clip_id: str,
    pred,
    language: str,
    fallback: Callable[[], str],
) -> str:
    return generate_grounded_narration_result(
        clip_id=clip_id,
        pred=pred,
        language=language,
        fallback=fallback,
    ).narrative


def generate_grounded_narration_result(
    *,
    clip_id: str,
    pred,
    language: str,
    fallback: Callable[[], str],
) -> NarrationResult:
    """
    Return VLM + LLM grounded narration plus demo-safe runtime status.

    The fallback callable is intentionally supplied by explanation_service so the
    old narration path remains the source of truth when models are unavailable.
    """
    print(f"[narration] requested_language={language}")
    narration_enabled = os.getenv("GUARDIAN_NARRATION_ENABLED", "1")
    vlm_enabled = os.getenv("GUARDIAN_NARRATION_VLM_ENABLED", "1")
    vlm_model_id = os.getenv("GUARDIAN_VLM_MODEL_ID", DEFAULT_VLM_MODEL)
    print(
        "[narration] config "
        f"GUARDIAN_NARRATION_ENABLED={narration_enabled!r} "
        f"GUARDIAN_NARRATION_VLM_ENABLED={vlm_enabled!r} "
        f"vlm_model_id={vlm_model_id!r} "
        f"local_path_valid={Path(vlm_model_id).is_dir()}"
    )
    if narration_enabled == "0":
        print("[narration] mode=fallback reason='Narration AI disabled' vlm=disabled llm=not_loaded")
        return _fallback_result(fallback, "Narration AI disabled", {"vlm": "disabled", "llm": "not_loaded"})

    requested_mode = os.getenv("GUARDIAN_NARRATION_MODE", "vlm_llm").strip().lower()
    if requested_mode not in {"vlm_llm", "vlm_only"}:
        print(f"[narration] invalid_mode={requested_mode!r} using='vlm_llm'")
        requested_mode = "vlm_llm"
    if vlm_enabled != "1":
        print("[narration] mode=fallback reason='VLM narration disabled' vlm=disabled llm=not_loaded")
        return _fallback_result(
            fallback,
            "VLM narration disabled",
            {"vlm": "disabled", "llm": "not_loaded"},
        )

    try:
        frames_vit = _load_frames_vit(clip_id)
        evidence_frames = _export_evidence_frames(clip_id, frames_vit, pred)
        packet = build_evidence_packet(pred, evidence_frames)
        print(
            "[narration] evidence_ready "
            f"clip_id={clip_id!r} frames_vit_shape={getattr(frames_vit, 'shape', None)} "
            f"exported_frames={len(evidence_frames.paths)} "
            f"peak_vit_indices={evidence_frames.peak_vit_indices}"
        )
    except Exception as exc:
        print(
            "[narration] mode=fallback stage=evidence "
            f"exception={exc.__class__.__name__}: {exc}"
        )
        return _fallback_result(
            fallback,
            "Evidence frames unavailable",
            {"vlm": "not_loaded", "llm": "not_loaded"},
        )

    try:
        print(
            "[narration] narration_vlm_load_start "
            f"model_id={vlm_model_id!r} local_path_valid={Path(vlm_model_id).is_dir()}"
        )
        visual_observations = _run_vlm(
            evidence_frames.paths,
            packet,
            "en",
            output_mode="final_narration",
        )
        if not str(visual_observations or "").strip():
            raise RuntimeError("empty VLM narration")
        print("[narration] narration_vlm_success")
    except Exception as exc:
        print(
            "[narration] mode=fallback stage=vlm "
            f"reason={_safe_model_reason(exc, 'VLM')!r} "
            f"exception={exc.__class__.__name__}: {exc}"
        )
        return _fallback_result(
            fallback,
            _safe_model_reason(exc, "VLM"),
            {"vlm": "unavailable", "llm": "not_loaded"},
        )

    if _valid_vlm_narration(visual_observations, packet):
        narrative = polish_generated_text(
            visual_observations,
            "en",
            label="VLM narration",
        )
        model_status = {"vlm": "released", "llm": "skipped"}
        narration_mode: Literal["vlm_llm", "vlm_only"] = "vlm_only"
        print("[narration] narration_llm=skipped reason='valid VLM narration'")
    elif requested_mode == "vlm_only":
        return _fallback_result(
            fallback,
            "VLM narration was invalid",
            {"vlm": "released", "llm": "skipped"},
        )
    else:
        try:
            narrative = _run_llm(visual_observations, packet, "en")
        except Exception as exc:
            print(
                "[narration] mode=fallback stage=llm "
                f"reason={_safe_model_reason(exc, 'LLM')!r} "
                f"exception={exc.__class__.__name__}: {exc}"
            )
            return _fallback_result(
                fallback,
                _safe_model_reason(exc, "LLM"),
                {"vlm": "released", "llm": "unavailable"},
            )
        model_status = {"vlm": "released", "llm": "released"}
        narration_mode = "vlm_llm"

    narrative = _format_narrative_for_display(
        narrative,
        verdict=str(packet.get("verdict")),
        language="en",
        packet=packet,
        visual_observations=visual_observations,
    )

    if language == "ar":
        try:
            narrative = _translate_narration_to_arabic(narrative)
            narrative = polish_generated_text(
                narrative,
                "ar",
                label="final narration",
                require_arabic_only=True,
                allow_chinese=False,
            )
            print("[narration] Arabic translation done")
        except ValueError as exc:
            print(
                "[narration] mode=fallback stage=arabic_language_guard "
                f"reason={exc!s}"
            )
            return _fallback_result(
                fallback,
                "Generated narration was not fully Arabic",
                model_status,
            )
        except Exception as exc:
            print(
                "[narration] mode=fallback stage=arabic_translation "
                f"exception={exc.__class__.__name__}: {exc}"
            )
            return _fallback_result(
                fallback,
                "Arabic translation failed",
                model_status,
            )

    if not _guardrail_ok(narrative, str(packet["verdict"])):
        print("[narration] mode=fallback stage=guardrail reason='Generated narration did not pass safety guardrails'")
        return _fallback_result(
            fallback,
            "Generated narration did not pass safety guardrails",
            model_status,
        )

    narrative = narrative.strip()
    if not narrative:
        print("[narration] mode=fallback stage=empty_generation reason='Generated narration was empty'")
        return _fallback_result(
            fallback,
            "Generated narration was empty",
            model_status,
        )
    vlm_summary = build_vlm_summary(
        visual_observations=visual_observations,
        narrative=narrative,
        packet=packet,
        language=language,
    )
    print(f"[narration] mode={narration_mode} model_status={model_status}")
    return NarrationResult(
        narrative=narrative,
        narration_mode=narration_mode,
        model_status=model_status,
        reason_if_fallback=None,
        vlm_summary=vlm_summary,
    )


def _fallback_result(
    fallback: Callable[[], str],
    reason: str,
    model_status: dict[str, str],
) -> NarrationResult:
    print(f"[narration] narration_fallback_reason={reason!r}")
    return NarrationResult(
        narrative=fallback(),
        narration_mode="fallback",
        model_status=model_status,
        reason_if_fallback=reason,
    )


def _safe_model_reason(exc: Exception, label: str) -> str:
    text = str(exc).strip()
    lowered = text.lower()
    if "local_files_only" in text or "not the path to a directory" in text:
        return f"{label} model unavailable locally"
    if "qwen-vl-utils" in text:
        return "VLM image dependency unavailable"
    if "autoawq" in lowered or "awq" in lowered:
        return f"{label} AWQ dependency unavailable"
    if "paging file is too small" in lowered or "os error 1455" in lowered:
        return f"{label} could not load because Windows virtual memory is too low"
    if "out of memory" in lowered:
        return f"{label} ran out of GPU memory"
    return f"{label} unavailable"


def build_evidence_packet(pred, evidence_frames: EvidenceFrames) -> dict[str, Any]:
    gate = {
        "skeleton": float(pred.gate.skeleton),
        "interaction": float(pred.gate.interaction),
        "object": float(pred.gate.object),
        "vit": float(pred.gate.vit),
    }
    gqs = {
        "q_skel": float(pred.gqs.q_skel),
        "q_int": float(pred.gqs.q_int),
        "q_obj": float(pred.gqs.q_obj),
        "q_po": float(pred.gqs.q_po),
        "valid_ratio": float(pred.gqs.valid_ratio),
    }
    dominant = sorted(gate.items(), key=lambda item: item[1], reverse=True)
    weapon = pred.telemetry.weapon
    severity = _estimate_severity(pred, gate)
    reference_context = retrieve_reference_context(pred, gate, severity)
    return {
        "clip_id": pred.clip_id,
        "verdict": pred.verdict,
        "confidence": float(pred.confidence),
        "threshold": float(pred.threshold),
        "severity": severity,
        "peak_window": list(pred.telemetry.peak_window),
        "people_count": int(pred.telemetry.people),
        "weapon": {
            "flag": bool(weapon.flag),
            "class": weapon.cls,
        },
        "gate": gate,
        "gqs": gqs,
        "dominant_contributors": [
            {"stream": name, "weight": value} for name, value in dominant[:3]
        ],
        "selected_frames": [
            {"vit_index": idx, "path": path}
            for idx, path in zip(evidence_frames.indices, evidence_frames.paths)
        ],
        "peak_vit_indices": evidence_frames.peak_vit_indices,
        "reference_context": reference_context,
        "caveats": [
            "The verdict and confidence come from the Guardian Eye classifier and must not be re-decided.",
            "Frame-level timing and object proximity are telemetry, not legal conclusions.",
            "Only visible observations from selected evidence frames may be described.",
            "Reference context is narration guidance, not observed incident evidence.",
            ROLE_DESCRIPTION_GUIDANCE_EN,
            ROLE_DESCRIPTION_GUIDANCE_AR,
        ],
    }


def retrieve_reference_context(
    pred,
    gate: dict[str, float] | None = None,
    severity: str | None = None,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    gate = gate or {
        "skeleton": float(pred.gate.skeleton),
        "interaction": float(pred.gate.interaction),
        "object": float(pred.gate.object),
        "vit": float(pred.gate.vit),
    }
    severity = severity or _estimate_severity(pred, gate)
    terms = _reference_query_terms(pred, gate, severity)
    scored: list[tuple[int, ReferenceEntry]] = []
    for entry in REFERENCE_KB:
        score = len(terms.intersection(entry.tags))
        if "all" in entry.tags:
            score += 1
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1].ref_id))

    selected = scored[:top_k]
    return [
        {
            "id": entry.ref_id,
            "title": entry.title,
            "score": score,
            "applies_to": sorted(terms.intersection(entry.tags)),
            "guidance_en": entry.guidance_en,
            "guidance_ar": entry.guidance_ar,
            "grounding_type": "narration_guidance_not_incident_fact",
        }
        for score, entry in selected
    ]


def _reference_query_terms(pred, gate: dict[str, float], severity: str) -> set[str]:
    terms = {"all", "verdict", "confidence", "gate", "peak", "window", severity}
    terms.add(str(pred.verdict))
    if str(pred.verdict) == "non-violence":
        terms.update({"threshold", "calm"})
    else:
        terms.add("violence")

    peak = list(pred.telemetry.peak_window)
    if peak and (peak[0] != 0 or peak[-1] != 0):
        terms.update({"timing", "telemetry"})

    if int(pred.telemetry.people) > 1:
        terms.update({"people", "multiple"})
    if int(pred.telemetry.people) >= 4:
        terms.add("crowd")

    weapon = pred.telemetry.weapon
    if bool(weapon.flag):
        terms.update({"weapon", "object", "proximity", "telemetry"})

    for stream, value in gate.items():
        if value >= 0.2:
            terms.update({"stream", stream})
    return terms


def build_vlm_summary(
    *,
    visual_observations: str,
    narrative: str,
    packet: dict[str, Any],
    language: str = "en",
) -> dict[str, Any]:
    combined = " ".join(str(part or "") for part in (visual_observations, narrative))
    people_count = _extract_people_count(combined)
    if people_count is None:
        people_count = int(packet.get("people_count") or 0)

    actions = _extract_observed_actions(combined)
    objects = _extract_objects(combined, packet)
    possible_roles = _extract_possible_roles(combined)
    violence_type = _estimate_violence_type(combined, str(packet.get("verdict") or ""))
    return {
        "summary_source": "current_vlm",
        "language": language,
        "people_count": people_count,
        "environment": _extract_environment(combined),
        "observed_actions": actions,
        "objects": objects,
        "possible_roles": possible_roles,
        "violence_type": violence_type,
        "severity_estimate": str(packet.get("severity") or "unknown"),
        "visual_summary": _compact_summary_text(combined, 700),
        "limitations": (
            "Descriptions are VLM observations from selected evidence frames. "
            "They do not identify guilt, legal responsibility, or fixed person roles."
        ),
    }


def _extract_people_count(text: str) -> int | None:
    lowered = text.lower()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
    }
    for match in re.finditer(r"\b(\d+)\s+(?:people|persons|individuals|men|women)\b", lowered):
        return int(match.group(1))
    for word, value in words.items():
        if re.search(rf"\b{word}\s+(?:people|persons|individuals|men|women)\b", lowered):
            return value
    return None


def _extract_observed_actions(text: str) -> list[str]:
    lowered = text.lower()
    candidates = [
        ("pushing", ["push", "pushing"]),
        ("striking", ["strike", "striking", "hit", "hitting", "punch", "punching"]),
        ("physical confrontation", ["physical confrontation", "altercation", "fight", "fighting"]),
        ("moving toward another person", ["move toward", "moves toward", "moving toward", "approach", "approaches"]),
        ("stepping back or falling", ["step back", "steps back", "fall", "falls", "falling"]),
        ("close interaction", ["close interaction", "close contact", "grappling", "holding"]),
    ]
    return [label for label, terms in candidates if any(term in lowered for term in terms)]


def _extract_objects(text: str, packet: dict[str, Any]) -> list[str]:
    lowered = text.lower()
    objects: list[str] = []
    weapon = packet.get("weapon") or {}
    weapon_class = weapon.get("class")
    if weapon.get("flag") and weapon_class:
        objects.append(str(weapon_class))
    for term in ("vehicle", "car", "bottle", "knife", "object", "bag"):
        if term in lowered and term not in objects:
            objects.append(term)
    return objects


def _extract_possible_roles(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    allowed = []
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(term in lowered for term in ("appears", "seems", "may", "looks")):
            continue
        if not any(term in lowered for term in ("person", "individual", "shirt", "defensive", "dominant", "moves", "push", "step")):
            continue
        if any(term in lowered for term in ("attacker", "victim", "guilty", "criminal")):
            continue
        allowed.append(sentence.strip())
    if allowed:
        return allowed[:3]
    return ["The frames do not clearly support identifying individual roles."]


def _estimate_violence_type(text: str, verdict: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("push", "strike", "hit", "punch", "altercation", "fight", "confrontation")):
        return "physical altercation"
    if verdict == "violence":
        return "violent interaction"
    return "no clear physical altercation"


def _extract_environment(text: str) -> str:
    lowered = text.lower()
    if "street" in lowered and "night" in lowered:
        return "street at night"
    if "street" in lowered:
        return "street"
    if "parking" in lowered:
        return "parking area"
    if "indoor" in lowered or "room" in lowered:
        return "indoor area"
    if "night" in lowered or "dark" in lowered:
        return "low-light scene"
    return "unspecified scene"


def _compact_summary_text(text: str, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return _truncate_at_sentence_boundary(clean, max_chars)


def _estimate_severity(pred, gate: dict[str, float]) -> str:
    confidence = float(pred.confidence)
    if str(pred.verdict) == "non-violence":
        return "low"
    if confidence >= 0.75 or bool(pred.telemetry.weapon.flag) or int(pred.telemetry.people) >= 4:
        return "high"
    if confidence >= float(pred.threshold) or max(gate.values()) >= 0.35:
        return "medium"
    return "low"


def _load_frames_vit(clip_id: str):
    import numpy as np

    npz_path = Path("cache_npz") / f"{clip_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing NPZ: {npz_path}")

    with np.load(str(npz_path), allow_pickle=False) as npz:
        if "frames_vit" not in npz.files:
            raise KeyError("frames_vit missing from NPZ")
        frames = npz["frames_vit"]

    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
        raise ValueError(f"unexpected frames_vit shape: {frames.shape}")
    return frames


def _export_evidence_frames(clip_id: str, frames_vit, pred) -> EvidenceFrames:
    from PIL import Image

    total = int(frames_vit.shape[0])
    if total >= 16:
        indices = _representative_indices(total, 16)
    else:
        indices = list(range(total))
        indices.extend([indices[-1]] * (16 - total))

    peak_vit_indices = _peak_vit_indices(pred.telemetry.peak_window, total)
    stem = _safe_stem(clip_id)
    out_dir = Path("static") / "evidence_frames" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for output_idx, frame_idx in enumerate(indices):
        img = Image.fromarray(frames_vit[frame_idx])
        path = out_dir / f"evidence_{output_idx:02d}_vit_{frame_idx:02d}.jpg"
        img.save(path, quality=90)
        paths.append(str(path))

    return EvidenceFrames(indices=indices, paths=paths, peak_vit_indices=peak_vit_indices)


def _representative_indices(total: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def _peak_vit_indices(peak_window: list[int], total_vit: int) -> list[int]:
    if total_vit <= 1:
        return [0]
    start = int(peak_window[0]) if peak_window else 0
    end = int(peak_window[1]) if len(peak_window) > 1 else start
    mapped = {
        max(0, min(total_vit - 1, round(t / 31 * (total_vit - 1))))
        for t in range(max(0, start), max(start, end) + 1)
    }
    return sorted(mapped)


def _valid_vlm_narration(text: str, packet: dict[str, Any]) -> bool:
    candidate = str(text or "").strip()
    if len(candidate.split()) < 8:
        return False
    lowered = candidate.casefold()
    if any(term in lowered for term in ("evidence packet", "prompt instruction", "as an ai")):
        return False
    return _guardrail_ok(candidate, str(packet.get("verdict") or ""))


def _complete_vlm_unload() -> None:
    """Collect released VLM resources and report post-unload GPU memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        memory = _narration_gpu_memory_snapshot(torch)
    except Exception as exc:
        memory = f"unavailable:{exc!r}"
    print("[narration] VLM unload done")
    print(f"[narration] GPU memory after unload={memory}")


def _narration_gpu_memory_snapshot(torch_module: Any) -> str:
    if not torch_module.cuda.is_available():
        return "cuda_unavailable"
    try:
        device = torch_module.cuda.current_device()
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
        allocated = torch_module.cuda.memory_allocated(device)
        reserved = torch_module.cuda.memory_reserved(device)
        mb = 1024 * 1024
        return (
            f"device={device},free_mb={free_bytes / mb:.0f},total_mb={total_bytes / mb:.0f},"
            f"allocated_mb={allocated / mb:.0f},reserved_mb={reserved / mb:.0f}"
        )
    except Exception as exc:
        return f"unavailable:{exc!r}"


def _run_vlm(
    frame_paths: list[str],
    packet: dict[str, Any],
    language: str,
    *,
    output_mode: Literal["observations", "final_narration"] = "observations",
) -> str:
    model_id = os.getenv("GUARDIAN_VLM_MODEL_ID", DEFAULT_VLM_MODEL)
    local_only = os.getenv("GUARDIAN_MODEL_LOCAL_ONLY", "1") == "1"
    model = None
    processor = None
    inputs = None
    generated = None
    image_inputs = None
    video_inputs = None
    images: list[Any] = []
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, BitsAndBytesConfig

        if torch.cuda.is_available():
            device_map: str = "auto"
            max_memory: dict[Any, str] | None = {
                0: os.getenv("GUARDIAN_NARRATION_VLM_GPU_MAX_MEMORY", "5GiB"),
                "cpu": os.getenv("GUARDIAN_NARRATION_VLM_CPU_MAX_MEMORY", "10GiB"),
            }
        else:
            device_map = "cpu"
            max_memory = None
        use_4bit = (
            torch.cuda.is_available()
            and os.getenv("GUARDIAN_NARRATION_VLM_4BIT", "1") == "1"
        )
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if use_4bit
            else None
        )

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass
        except Exception:
            from transformers import AutoModelForVision2Seq as VLMClass

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:
            raise RuntimeError("qwen-vl-utils is required for VLM image inputs") from exc

        print(
            "[narration] VLM load start "
            f"model_id={model_id!r} local_files_only={local_only} "
            f"device_map={device_map!r} max_memory={max_memory!r} "
            f"load_in_4bit={use_4bit} "
            f"frame_count={len(frame_paths)} output_mode={output_mode!r} "
            f"first_frame={frame_paths[0] if frame_paths else None!r}"
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_only,
            trust_remote_code=True,
            min_pixels=224 * 224,
            max_pixels=224 * 224,
        )
        model = VLMClass.from_pretrained(
            model_id,
            local_files_only=local_only,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        model.eval()
        print(f"[narration] vlm_loaded device={getattr(model, 'device', None)}")

        images = [Image.open(path).convert("RGB") for path in frame_paths]
        packet_text = json.dumps(packet, ensure_ascii=False, indent=2)
        output_language = "Arabic" if language == "ar" else "English"
        role_guidance = _role_description_guidance(language)
        language_guidance = ARABIC_ONLY_INSTRUCTION if language == "ar" else ""
        if output_mode == "final_narration":
            is_nonviolent = str(packet.get("verdict")) == "non-violence"
            display_confidence = round(
                (1.0 - max(0.0, min(1.0, float(packet.get("confidence", 0.0))))) * 100
            )
            task_text = (
                (
                    "Return the final user-facing Incident Narrative, not analysis notes. "
                    "Write exactly one concise sentence with "
                    f"{display_confidence}% non-violence confidence and one visible observation. "
                    "Do not use section labels, markdown symbols, bullets, bold markers, "
                    "or raw telemetry."
                )
                if is_nonviolent
                else (
                    "Return the final user-facing Incident Narrative, not analysis notes. "
                    "Use exactly these readable sections: Scene, People observed, "
                    "Observed activity, and Guardian Eye assessment. Under People observed, "
                    "include Person 1:, Person 2:, Person 3: only when supported by visible "
                    "evidence. Give each person one cautious sentence about clothing, "
                    "location, posture, or movement. Put each main observation on a new line. "
                    "Target 100 to 160 words. Use Arabic labels when the requested language "
                    "is Arabic. Do not use markdown symbols, bullets, bold markers, model "
                    "internals, file names, frame indices, or raw telemetry."
                )
            )
        else:
            task_text = (
                "Return concise visual observations. Mention only what is visible in "
                "the analyzed video and what the evidence packet explicitly provides. "
                "When possible, include Person 1:, Person 2:, and Person 3: with one "
                "cautious visible description each."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a visual evidence analyst. Describe only visible facts "
                    "from the provided frames. Do not decide whether violence occurred; "
                    "the classifier verdict in the packet is final. If an object or "
                    "action is unclear, say it is unclear. Reference context in the "
                    "packet is narration guidance only, not visual evidence. "
                    f"{role_guidance} {language_guidance}"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"EVIDENCE PACKET:\n{packet_text}\n\n"
                            f"Answer in {output_language}. {task_text} Use reference "
                            "context only for cautious wording rules; do not convert it "
                            "into incident facts. Describe visible people by clothing, "
                            "position, or observable movement only when clearly visible. "
                            f"{role_guidance} {language_guidance}"
                        ),
                    },
                    *[{"type": "image", "image": image} for image in images],
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        print(
            "[narration] vlm_inputs "
            f"image_inputs={len(image_inputs) if image_inputs is not None else 0} "
            f"video_inputs={len(video_inputs) if video_inputs is not None else 0}"
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=220)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        narration = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        print("[narration] VLM narration generated")
        return narration
    finally:
        print("[narration] VLM unload start")
        for image in images:
            try:
                image.close()
            except Exception:
                pass
        del generated
        del inputs
        del image_inputs
        del video_inputs
        del images
        del model
        del processor
        _complete_vlm_unload()


def _run_llm(visual_observations: str, packet: dict[str, Any], language: str) -> str:
    model_id = os.getenv("GUARDIAN_LLM_MODEL_ID", DEFAULT_LLM_MODEL)
    local_only = os.getenv("GUARDIAN_MODEL_LOCAL_ONLY", "1") == "1"
    model = None
    tokenizer = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[narration] loading_llm model_id={model_id!r} local_files_only={local_only}")
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

        packet_text = json.dumps(packet, ensure_ascii=False, indent=2)
        target_language = "English"
        role_guidance = _role_description_guidance("en")
        language_guidance = ""
        is_nonviolent = str(packet.get("verdict")) == "non-violence"
        display_confidence = round(
            (1.0 - max(0.0, min(1.0, float(packet.get("confidence", 0.0))))) * 100
        )
        verdict_guidance = (
            "For a non-violence verdict, write exactly one concise sentence. "
            f"Use {display_confidence}% as the displayed confidence because it is "
            "the non-violence confidence, include one visible observation from the VLM "
            "when available, and say no clear violent behavior is visible. Do not include "
            "long caveats, frame telemetry, legal wording, or attacker/victim wording."
            if is_nonviolent
            else "For a violence verdict, keep the wording cautious and grounded."
        )
        narration_constraints = (
            "Constraints: exactly one concise sentence. Include the non-violent verdict, "
            f"{display_confidence}% confidence, one visible observation from the VLM when "
            "available, and that no clear violent behavior is visible. Do not mention "
            "frame numbers, telemetry, legal review, attacker, or victim."
            if is_nonviolent
            else (
                "Constraints: target 100 to 160 words. Use exactly these readable plain "
                "section labels on separate lines: Scene, People observed, Observed "
                "activity, and Guardian Eye assessment. Under People observed, include "
                "Person 1:, Person 2:, Person 3: only when supported by visual observations, "
                "with one cautious sentence each about clothing, location, posture, or "
                "movement. Keep activity cautious with appears to, seems to, or may. "
                "Include verdict and confidence. If visual evidence is uncertain, say so."
            )
        )
        unsupported_action_constraints = (
            "Do not mention unsupported actions, intent, or guilt."
            if is_nonviolent
            else (
                "Do not mention unsupported actions, intent, guilt, legal responsibility, or legal conclusions."
            )
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You write Guardian Eye incident narration. The Guardian Eye classifier verdict "
                    "and confidence are final. Do not add facts beyond the visual "
                    "observations and evidence packet. Use cautious wording. Reference "
                    "context is grounding guidance only, not observed incident evidence. "
                    "Write user-facing prose only. Do not expose frame numbers, selected "
                    "frame lists, peak-frame indices, raw evidence windows, file paths, "
                    "or prompt instructions. Use Guardian Eye classifier as the system label. "
                    f"{role_guidance} {language_guidance} {verdict_guidance}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Write the final narration in {target_language}.\n\n"
                    f"EVIDENCE PACKET:\n{packet_text}\n\n"
                    f"VLM VISUAL OBSERVATIONS:\n{visual_observations}\n\n"
                    f"{narration_constraints} {unsupported_action_constraints} Use reference_context only "
                    "to choose safe wording; do not present it as something seen in "
                    "the video. "
                    "Do not use markdown heading symbols, bullets, or bold markers. "
                    "Plain section labels are allowed for readable structure. "
                    "Do not expose frame numbers, selected frame lists, peak-frame indices, "
                    "raw evidence windows, file paths, or prompt instructions. "
                    "Use Guardian Eye classifier as the system label. "
                    f"{role_guidance} {language_guidance} {verdict_guidance}"
                ),
            },
        ]

        def generate_once(prompt_messages: list[dict[str, str]]) -> str:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = "\n\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)
                text += "\n\nassistant:"

            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=420,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            return tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        answer = generate_once(messages)
        english_answer = polish_generated_text(answer, "en", label="narration")
        if is_nonviolent:
            english_answer = _single_sentence(english_answer)
        if language != "ar":
            return english_answer

        # Translation uses its own provider lifecycle. Release the English Qwen
        # narrator first so TranslateGemma can claim VRAM; Qwen reloads only as fallback.
        if model is not None:
            del model
            model = None
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
        gc.collect()
        _empty_cuda_cache()
        return _translate_narration_to_arabic(english_answer)

    finally:
        if model is not None:
            del model
        gc.collect()
        _empty_cuda_cache()
        del tokenizer
        gc.collect()


def _translate_narration_to_arabic(english_answer: str) -> str:
    from arabic_translation_service import translate_texts_to_arabic

    translation_source, required_markers = _prepare_structured_translation_source(
        english_answer
    )
    allowed_latin_terms = tuple(
        dict.fromkeys(
            re.findall(
                r"[^\s,;:]+\.(?:avi|mp4|mov|mkv|webm)|"
                r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+|"
                r"\b[A-Za-z][A-Za-z0-9.]*(?:-[A-Za-z0-9.]+){2,}\b",
                english_answer,
                flags=re.IGNORECASE,
            )
        )
    )

    def validate_arabic_segment(raw_translation: str) -> str:
        return polish_generated_text(
            raw_translation,
            "ar",
            label="narration translation segment",
            require_arabic_only=True,
            allow_chinese=False,
            allowed_latin_terms=allowed_latin_terms,
        )

    print("[narration] translation_to_arabic=started")
    segments = _structured_translation_segments(
        translation_source,
        required_markers=required_markers,
    )
    nonempty_sources = [source for _, source in segments if source]
    results = translate_texts_to_arabic(
        nonempty_sources,
        validator=validate_arabic_segment,
    )
    result_iterator = iter(results)
    translated_parts: list[str] = []
    for marker, source_segment in segments:
        translated_parts.append(
            f"{marker}\n{next(result_iterator).text}" if source_segment else marker
        )

    restored = _restore_arabic_translation_structure(
        "\n".join(translated_parts),
        required_markers=required_markers,
    )
    translated = polish_generated_text(
        restored,
        "ar",
        label="narration translation",
        require_arabic_only=True,
        allow_chinese=False,
        allowed_latin_terms=allowed_latin_terms,
    )
    print(
        "[narration] translation_to_arabic=done "
        f"provider=qwen attempts={sum(result.attempts for result in results)}"
    )
    return translated


_TRANSLATION_SECTION_MARKERS = (
    (re.compile(r"\bGuardian\s+Eye\s+assessment\s*:\s*", re.IGNORECASE), "[[GE_ASSESSMENT]]"),
    (re.compile(r"\bPeople\s+observed\s*:\s*", re.IGNORECASE), "[[GE_PEOPLE]]"),
    (re.compile(r"\bObserved\s+activity\s*:\s*", re.IGNORECASE), "[[GE_ACTIVITY]]"),
    (re.compile(r"\bScene\s*:\s*", re.IGNORECASE), "[[GE_SCENE]]"),
)
_TRANSLATION_PERSON_RE = re.compile(r"\bPerson\s+(?P<number>[1-4])\s*:\s*", re.IGNORECASE)


def _prepare_structured_translation_source(text: str) -> tuple[str, tuple[str, ...]]:
    """Replace English display labels with stable tokens before Arabic translation."""
    structured = str(text or "")
    for pattern, marker in _TRANSLATION_SECTION_MARKERS:
        structured = pattern.sub(f"{marker}\n", structured)

    def replace_person(match: re.Match[str]) -> str:
        marker = f"[[GE_PERSON_{match.group('number')}]]"
        return f"{marker}\n"

    structured = _TRANSLATION_PERSON_RE.sub(replace_person, structured)
    required = tuple(
        re.findall(
            r"\[\[GE_(?:ASSESSMENT|PEOPLE|ACTIVITY|SCENE|PERSON_[1-4])\]\]",
            structured,
        )
    )
    return structured.strip(), required


def _structured_translation_segments(
    text: str,
    *,
    required_markers: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Return marker/content pairs while keeping every marker out of model input."""
    marker_pattern = re.compile(r"(\[\[GE_(?:ASSESSMENT|PEOPLE|ACTIVITY|SCENE|PERSON_[1-4])\]\])")
    parts = marker_pattern.split(str(text or ""))
    if parts[0].strip():
        raise ValueError("narration translation contains text before its first marker")

    segments: list[tuple[str, str]] = []
    for index in range(1, len(parts), 2):
        marker = parts[index]
        content = parts[index + 1].strip() if index + 1 < len(parts) else ""
        segments.append((marker, content))

    found_markers = tuple(marker for marker, _ in segments)
    if found_markers != required_markers:
        raise ValueError(
            "narration translation section order changed: "
            f"expected={required_markers!r} found={found_markers!r}"
        )
    return tuple(segments)


def _restore_arabic_translation_structure(
    text: str,
    *,
    required_markers: tuple[str, ...],
) -> str:
    """Validate protected tokens and render their user-facing Arabic labels."""
    translated = str(text or "").strip()
    malformed = [marker for marker in required_markers if translated.count(marker) != 1]
    if malformed:
        raise ValueError(
            "narration translation did not preserve structure markers: "
            + ", ".join(malformed)
        )

    labels = _narrative_labels("ar")
    replacements = {
        "[[GE_SCENE]]": f"{labels['scene']}:",
        "[[GE_PEOPLE]]": f"{labels['people']}:",
        "[[GE_ACTIVITY]]": f"{labels['activity']}:",
        "[[GE_ASSESSMENT]]": f"{labels['assessment']}:",
        **{
            f"[[GE_PERSON_{number}]]": f"{_person_label('ar', number)}:"
            for number in range(1, 5)
        },
    }
    for marker, label in replacements.items():
        translated = translated.replace(marker, label)
    return translated


def _empty_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _role_description_guidance(language: str) -> str:
    if language == "ar":
        return ROLE_DESCRIPTION_GUIDANCE_AR
    return ROLE_DESCRIPTION_GUIDANCE_EN


def _guardrail_ok(narrative: str, verdict: str) -> bool:
    text = narrative.lower()
    if verdict == "violence":
        blocked = ["non-violent", "non violence", "not violent", "peaceful scene"]
    else:
        blocked = [
            "classified as violent",
            "the verdict is violence",
            "confirmed fight",
            "frame",
            "frames",
            "peak",
            "telemetry",
            "legal",
            "court",
            "attacker",
            "victim",
        ]
    forbidden_role_claims = [
        "the guilty person",
        "guilty person",
        "the victim is",
        "victim is",
        "the attacker is",
        "attacker is",
        "committed assault",
        "this person committed assault",
        "started the fight",
        "this person started the fight",
        "legal responsibility is",
        "legally responsible",
        "الشخص المذنب",
        "المذنب",
        "الضحية هي",
        "الضحية هو",
        "المهاجم هو",
        "المهاجم هي",
        "ارتكب اعتداء",
        "بدأ الشجار",
        "مسؤول قانونيا",
    ]
    forbidden_role_claims.extend(
        [
            "الشخص المذنب",
            "المذنب",
            "الضحية هي",
            "الضحية هو",
            "المهاجم هو",
            "المهاجم هي",
            "ارتكب اعتداء",
            "بدأ الشجار",
            "مسؤول قانونيا",
        ]
    )
    return not any(term in text for term in [*blocked, *forbidden_role_claims])


def _format_narrative_for_display(
    text: str,
    *,
    verdict: str,
    language: str,
    packet: dict[str, Any] | None = None,
    visual_observations: str = "",
) -> str:
    clean = str(text or "").strip()
    if verdict == "non-violence":
        return _single_sentence(clean)

    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\s*\n+\s*", "\n", clean)
    clean = re.sub(r"\bCaveats?\s*:\s*", "Limitations: ", clean, flags=re.IGNORECASE)
    clean = _remove_narrative_internal_terms(clean)
    clean, artifact_count = _sanitize_narrative_artifacts(clean)

    labels = _narrative_labels(language)
    sections = _extract_labeled_narrative_sections(clean)
    if not sections:
        sections = _infer_narrative_sections(clean)

    ordered = [
        ("scene", labels["scene"], 1),
        ("people", labels["people"], 2),
        ("activity", labels["activity"], 2),
        ("assessment", labels["assessment"], 1),
    ]
    rendered: list[str] = []
    for key, label, max_sentences in ordered:
        if key == "people":
            value = _clean_people_section_value(sections.get(key, ""), language=language)
            visual_people = ""
            if language != "ar" or _is_strict_arabic_narration(visual_observations):
                visual_people = _clean_people_section_value(
                    visual_observations,
                    language=language,
                )
            if visual_people and (not value or not _has_person_label(value)):
                value = visual_people
        else:
            value = _clean_narrative_section_value(
                sections.get(key, ""),
                max_sentences=max_sentences,
            )
        if value:
            rendered.append(f"{label}:\n{value}")

    formatted = "\n\n".join(rendered) if rendered else clean
    formatted, remaining_artifacts = _sanitize_narrative_artifacts(formatted, preserve_layout=True)
    formatted = _clean_narrative_section_output(formatted)
    if _narrative_needs_clean_fallback(
        formatted,
        artifact_count=artifact_count + remaining_artifacts,
    ):
        return _deterministic_clean_narrative(
            packet=packet,
            visual_observations=visual_observations or clean,
            language=language,
        )
    return formatted


def _is_strict_arabic_narration(text: str) -> bool:
    try:
        polish_generated_text(
            text,
            "ar",
            label="narration",
            require_arabic_only=True,
            allow_chinese=False,
        )
    except ValueError:
        return False
    return True


def _narrative_labels(language: str) -> dict[str, str]:
    if language == "ar":
        return {
            "scene": "المشهد",
            "people": "الأشخاص المرصودون",
            "activity": "النشاط المرصود",
            "assessment": "تقييم Guardian Eye",
            "limitations": "القيود",
            "limitations_text": "لا يحدد النظام الذنب أو أدوار المهاجم أو الضحية.",
        }
    if language == "ar":
        return {
            "scene": "المشهد",
            "people": "الأشخاص المرصودون",
            "activity": "النشاط المرصود",
            "assessment": "تقييم Guardian Eye",
            "limitations": "القيود",
            "limitations_text": "لا يحدد النظام الذنب أو أدوار المهاجم أو الضحية.",
        }
    return {
        "scene": "Scene",
        "people": "People observed",
        "activity": "Observed activity",
        "assessment": "Guardian Eye assessment",
        "limitations": "Limitations",
        "limitations_text": "The system does not identify guilt, attacker, or victim roles.",
    }


def _extract_labeled_narrative_sections(text: str) -> dict[str, str]:
    label_map = {
        "scene": "scene",
        "المشهد": "scene",
        "visible scene entities": "people",
        "people observed": "people",
        "الأشخاص المرصودون": "people",
        "person 1": "people",
        "person 2": "people",
        "person 3": "people",
        "person 4": "people",
        "الشخص 1": "people",
        "الشخص 2": "people",
        "الشخص 3": "people",
        "الشخص 4": "people",
        "action": "activity",
        "interaction": "activity",
        "observed activity": "activity",
        "النشاط المرصود": "activity",
        "pointing gesture": "activity",
        "lying individual": "activity",
        "guardian eye assessment": "assessment",
        "guardian eye classifier": "assessment",
        "assessment": "assessment",
        "تقييم guardian eye": "assessment",
        "limitations": "limitations",
        "القيود": "limitations",
    }
    pattern = re.compile(
        r"(?P<label>Visible Scene Entities|People observed|Guardian Eye assessment|"
        r"Guardian Eye classifier|Observed activity|Scene|Person\s+\d+|Action|"
        r"Interaction|Pointing Gesture|Lying Individual|Assessment|Limitations|"
        r"المشهد|الأشخاص المرصودون|الشخص\s+\d+|النشاط المرصود|"
        r"تقييم\s+Guardian\s+Eye|القيود)\s*:",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    sections: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_label = re.sub(r"\s+", " ", match.group("label")).strip()
        label = raw_label.casefold()
        key = "people" if _is_person_section_label(label) else label_map.get(label)
        value = text[start:end].strip()
        if key and value:
            if _is_person_section_label(label):
                person_number = re.search(r"\d+", raw_label)
                display_label = raw_label if person_number else raw_label
                value = f"{display_label}:\n{value}"
            sections.setdefault(key, []).append(value)
    return {key: "\n\n".join(values) for key, values in sections.items()}


def _infer_narrative_sections(text: str) -> dict[str, str]:
    sentences = _sentences(text)
    sections: dict[str, str] = {}
    for sentence in sentences:
        lowered = sentence.casefold()
        if "caveat" in lowered or "telemetry" in lowered or "legal" in lowered:
            continue
        if "guardian eye" in lowered or "classifier" in lowered or "classified" in lowered:
            sections.setdefault("assessment", sentence)
        elif any(term in lowered for term in ("person", "people", "individual")):
            if "appears" in lowered or "seems" in lowered or "floor" in lowered or "stand" in lowered:
                current = sections.get("activity", "")
                sections["activity"] = f"{current} {sentence}".strip() if current else sentence
            else:
                sections.setdefault("people", sentence)
        elif any(term in lowered for term in ("indoor", "outdoor", "area", "scene", "room", "street")):
            sections.setdefault("scene", sentence)
        else:
            current = sections.get("activity", "")
            sections["activity"] = f"{current} {sentence}".strip() if current else sentence
    return sections


def _remove_narrative_internal_terms(text: str) -> str:
    clean = re.sub(r"\bV9\s+classifier\b", "Guardian Eye classifier", text, flags=re.IGNORECASE)
    clean = re.sub(r"\bV9\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bpeak[- ]?frames?\b", "frames", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bmodel internals?\b", "analysis details", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bstrongest model evidence\b", "visible observations", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bselected evidence frames?\b", "the analyzed video", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bselected frames?\b", "the analyzed video", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\braw telemetry\b", "analysis data", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bframe-level timing\b", "timing details", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^.!?\n]*legal responsibility[^.!?\n]*[.!?]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^.!?\n]*\btelemetry\b[^.!?\n]*[.!?]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^.!?\n]*\btiming details\b[^.!?\n]*[.!?]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


_NARRATIVE_ARTIFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[[^\]]*(?:evidence|frame|\.jpe?g|\.png|\.mp4)[^\]]*\]", re.IGNORECASE),
    re.compile(r"\b[\w.-]*evidence_[\w.-]*\.(?:jpe?g|png|mp4)\b", re.IGNORECASE),
    re.compile(r"\b[\w.-]+\.(?:jpe?g|png|mp4)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:Gate|Interaction|Object|Skeletal|Skeleton|ViT|RGB)\s+Contribution\s*:\s*[^.!?\n]*(?:[.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\bSummary\s*:\s*", re.IGNORECASE),
)

_NARRATIVE_DROP_SENTENCE_TERMS = (
    "gate contribution",
    "interaction contribution",
    "object contribution",
    "skeletal contribution",
    "skeleton contribution",
    "summary:",
    "telemetry",
    "selected evidence frames",
    "selected frames",
    "frame-level timing",
    "frame index",
    "peak frame",
    "peak frames",
    "model internals",
    "strongest model evidence",
    "v9",
    "should not be reconsidered",
    "prompt instruction",
    "reference_context",
    "evidence packet",
    "vlm visual observations",
)


def _sanitize_narrative_artifacts(text: str, *, preserve_layout: bool = False) -> tuple[str, int]:
    clean = str(text or "")
    artifact_count = 0
    for pattern in _NARRATIVE_ARTIFACT_PATTERNS:
        clean, count = pattern.subn("", clean)
        artifact_count += count

    if preserve_layout:
        kept_lines: list[str] = []
        for line in clean.splitlines():
            stripped = line.strip()
            lowered = stripped.casefold()
            if stripped and any(term in lowered for term in _NARRATIVE_DROP_SENTENCE_TERMS):
                artifact_count += 1
                continue
            kept_lines.append(stripped)
        clean = "\n".join(kept_lines)
        clean = re.sub(r"\b(?:gate|interaction|object|skeletal|skeleton|vit|rgb)\s*=\s*\d+(?:\.\d+)?\b", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\(\s*\d+(?:\.\d+)?\s*\)", "", clean)
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
        return clean, artifact_count

    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", clean):
        stripped = sentence.strip()
        if not stripped:
            continue
        lowered = stripped.casefold()
        if any(term in lowered for term in _NARRATIVE_DROP_SENTENCE_TERMS):
            artifact_count += 1
            continue
        kept.append(stripped)

    clean = " ".join(kept)
    clean = re.sub(r"\b(?:gate|interaction|object|skeletal|skeleton|vit|rgb)\s*=\s*\d+(?:\.\d+)?\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\(\s*\d+(?:\.\d+)?\s*\)", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean, artifact_count


def _narrative_needs_clean_fallback(text: str, *, artifact_count: int) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return True
    lowered = clean.casefold()
    if any(term in lowered for term in _NARRATIVE_DROP_SENTENCE_TERMS):
        return True
    if re.search(r"\b[\w.-]*evidence_[\w.-]*\.(?:jpe?g|png|mp4)\b", clean, flags=re.IGNORECASE):
        return True
    required = ("Scene:", "People observed:", "Observed activity:", "Guardian Eye assessment:")
    labels_ar = _narrative_labels("ar")
    required_ar = (
        f"{labels_ar['scene']}:",
        f"{labels_ar['people']}:",
        f"{labels_ar['activity']}:",
        f"{labels_ar['assessment']}:",
    )
    if artifact_count >= 2 and not (
        all(label in clean for label in required)
        or all(label in clean for label in required_ar)
    ):
        return True
    return False


_PERSON_DESCRIPTION_FORBIDDEN_TERMS = (
    "attacker",
    "victim",
    "guilty",
    "guilt",
    "culprit",
    "criminal",
    "legally responsible",
    "legal responsibility",
    "started the fight",
    "committed assault",
    "weapon offence",
    "weapon offense",
    "مذنب",
    "المذنب",
    "ضحية",
    "الضحية",
    "مهاجم",
    "المهاجم",
    "مسؤول قانونيا",
    "مسؤولية قانونية",
    "بدأ الشجار",
    "ارتكب اعتداء",
)


def _clean_people_section_value(value: str, *, language: str = "en") -> str:
    people = _extract_safe_person_descriptions(value)
    if people:
        return "\n".join(
            f"{_person_label(language, number)}:\n{description}"
            for number, description in people
        )
    return _clean_narrative_section_value(value, max_sentences=2)


def _extract_safe_person_descriptions(text: str, *, max_people: int = 4) -> list[tuple[int, str]]:
    clean = str(text or "").strip()
    if not clean:
        return []

    person_pattern = re.compile(
        r"(?P<label>\bPerson\s+(?P<number_en>\d+)|الشخص\s+(?P<number_ar>\d+))\s*[:\-]\s*"
        r"(?P<body>.*?)(?=(?:\bPerson\s+\d+|الشخص\s+\d+)\s*[:\-]|"
        r"\b(?:Scene|People observed|Observed activity|Action|Interaction|"
        r"Guardian Eye assessment|Assessment|Limitations)\s*:|"
        r"(?:المشهد|الأشخاص المرصودون|النشاط المرصود|تقييم\s+Guardian\s+Eye|القيود)\s*:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    descriptions: list[tuple[int, str]] = []
    seen_numbers: set[int] = set()
    for match in person_pattern.finditer(clean):
        try:
            number = int(match.group("number_en") or match.group("number_ar"))
        except Exception:
            continue
        if number in seen_numbers or number < 1:
            continue
        description = _clean_person_description(match.group("body"))
        if not description:
            continue
        seen_numbers.add(number)
        descriptions.append((number, description))
        if len(descriptions) >= max_people:
            break
    return sorted(descriptions)


def _person_label(language: str, number: int) -> str:
    return f"الشخص {number}" if language == "ar" else f"Person {number}"


def _has_person_label(text: str) -> bool:
    return bool(re.search(r"(?:\bPerson\s+\d+|الشخص\s+\d+)\s*:", str(text or ""), flags=re.IGNORECASE))


def _is_person_section_label(label: str) -> bool:
    return bool(re.match(r"^(?:person|الشخص)\s+\d+$", str(label or "").strip(), flags=re.IGNORECASE))


def _clean_person_description(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    clean = _remove_narrative_internal_terms(clean)
    clean, _ = _sanitize_narrative_artifacts(clean)
    clean = re.sub(r"\b(?:Visual Observations|Summary|Strongest Model Evidence)\s*:\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"(?:\bPerson\s+\d+|الشخص\s+\d+|People observed|الأشخاص المرصودون)\s*[:\-]\s*", "", clean, flags=re.IGNORECASE)
    lowered = clean.casefold()
    if any(term in lowered for term in _PERSON_DESCRIPTION_FORBIDDEN_TERMS):
        return ""
    if any(term in lowered for term in _NARRATIVE_DROP_SENTENCE_TERMS):
        return ""
    clean = _complete_sentences_only(_ensure_sentence_punctuation(clean), max_sentences=1)
    if not clean:
        return ""
    lowered = clean.casefold()
    if any(term in lowered for term in _PERSON_DESCRIPTION_FORBIDDEN_TERMS):
        return ""
    return _soften_person_description(clean)


def _ensure_sentence_punctuation(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if clean and not clean.endswith(tuple(_TERMINAL_PUNCTUATION)):
        clean = f"{clean}."
    return clean


def _soften_person_description(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    lowered = clean.casefold()
    if re.search(r"[\u0600-\u06ff]", clean):
        return clean
    if any(term in lowered for term in ("appears", "seems", "may", "unclear", "possibly")):
        return clean
    replacements = (
        (r"\bis standing\b", "appears to be standing"),
        (r"\bis sitting\b", "appears to be sitting"),
        (r"\bis lying\b", "appears to be lying"),
        (r"\bis wearing\b", "appears to be wearing"),
        (r"\bstands\b", "appears to stand"),
        (r"\bsits\b", "appears to sit"),
        (r"\blies\b", "appears to lie"),
        (r"\bmoves\b", "appears to move"),
        (r"\bwears\b", "appears to wear"),
        (r"\bwearing\b", "appears to be wearing"),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, clean, count=1, flags=re.IGNORECASE)
        if updated != clean:
            return updated
    return f"Appears to show {clean[0].lower()}{clean[1:]}" if clean else ""


def _deterministic_clean_narrative(
    *,
    packet: dict[str, Any] | None,
    visual_observations: str,
    language: str,
) -> str:
    labels = _narrative_labels(language)
    packet = packet or {}
    people_count = int(packet.get("people_count") or 0)
    confidence = packet.get("confidence")
    try:
        confidence_text = f"{float(confidence) * 100:.1f}%"
    except Exception:
        confidence_text = "the recorded"

    scene = _extract_environment(visual_observations) or (
        "منطقة مرصودة" if language == "ar" else "the monitored area"
    )
    if language == "ar":
        scene_sentence = "يبدو أن الفيديو يعرض منطقة مرصودة."
    elif scene == "the visible scene":
        scene_sentence = "The analyzed video appears to show the monitored scene."
    else:
        scene_sentence = f"The analyzed video appears to show {scene}."

    people_descriptions = _extract_safe_person_descriptions(visual_observations)
    if people_descriptions:
        people_sentence = "\n".join(
            f"{_person_label(language, number)}:\n{description}"
            for number, description in people_descriptions
        )
    elif people_count > 0:
        people_sentence = (
            f"رصد Guardian Eye {people_count} أشخاص ظاهرين في الحادثة."
            if language == "ar"
            else f"Guardian Eye detected {people_count} visible people in the incident."
        )
    else:
        people_sentence = (
            "رصد Guardian Eye أشخاصا ظاهرين في الحادثة."
            if language == "ar"
            else "Guardian Eye detected visible people in the incident."
        )

    actions = _extract_observed_actions(visual_observations)
    if language == "ar":
        activity = "يبدو أن النشاط المرئي يتضمن تفاعلا قريبا أو مواجهة محتملة. لا يمكن تحديد التسلسل أو القصد بدقة."
    elif actions:
        activity = (
            "The visible activity appears to involve "
            f"{', '.join(actions[:2])}. The exact sequence is unclear."
        )
    else:
        activity = (
            "One or more people appear to be involved in a possible confrontation or unsafe "
            "interaction. The exact sequence is unclear."
        )

    assessment = (
        f"صنف Guardian Eye المقطع على أنه عنيف بثقة {confidence_text}."
        if language == "ar"
        else f"Guardian Eye classified the clip as violent with {confidence_text} confidence."
    )
    return "\n\n".join(
        [
            f"{labels['scene']}:\n{scene_sentence}",
            f"{labels['people']}:\n{people_sentence}",
            f"{labels['activity']}:\n{activity}",
            f"{labels['assessment']}:\n{assessment}",
        ]
    )


_TERMINAL_PUNCTUATION = ".!?؟۔"


def _clean_narrative_section_value(value: str, *, max_sentences: int) -> str:
    clean = _remove_broken_sentence_artifacts(value)
    clean = re.sub(r"(?i)\b(?:Visual Observations|Summary|Strongest Model Evidence)\s*:\s*", "", clean)
    clean = re.sub(r"[^.!?؟۔\n]*\bframe index\b[^.!?؟۔\n]*[.!?؟۔]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^.!?؟۔\n]*\breference context\b[^.!?؟۔\n]*[.!?؟۔]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^.!?؟۔\n]*\bselected evidence frames?\b[^.!?؟۔\n]*[.!?؟۔]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^.!?؟۔\n]*\btelemetry\b[^.!?؟۔\n]*[.!?؟۔]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^.!?؟۔\n]*\blegal responsibility\b[^.!?؟۔\n]*[.!?؟۔]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[^.!?؟۔\n]*\b(?:attacker|victim|guilt|guilty)\b[^.!?؟۔\n]*[.!?؟۔]?", "", clean, flags=re.IGNORECASE)
    return _complete_sentences_only(clean, max_sentences=max_sentences)


def _clean_narrative_section_output(text: str) -> str:
    blocks = re.split(r"\n{2,}", str(text or "").strip())
    cleaned_blocks: list[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        label = lines[0]
        if label.casefold().rstrip(":") == "limitations":
            continue
        if label.endswith(":"):
            raw_body = "\n".join(lines[1:])
            body = (
                _clean_people_section_value(
                    raw_body,
                    language="ar" if _is_arabic_people_section_label(label) else "en",
                )
                if _is_people_section_label(label)
                else _complete_sentences_only(" ".join(lines[1:]), max_sentences=2)
            )
            if body:
                cleaned_blocks.append(f"{label}\n{body}")
        else:
            body = _complete_sentences_only(" ".join(lines), max_sentences=2)
            if body:
                cleaned_blocks.append(body)
    return "\n\n".join(cleaned_blocks)


def _is_people_section_label(label: str) -> bool:
    normalized = str(label or "").strip().rstrip(":").casefold()
    return normalized in {
        "people observed",
        _narrative_labels("ar")["people"].casefold(),
    }


def _is_arabic_people_section_label(label: str) -> bool:
    return str(label or "").strip().rstrip(":").casefold() == _narrative_labels("ar")["people"].casefold()


def _complete_sentences_only(text: str, *, max_sentences: int | None = None) -> str:
    clean = _remove_broken_sentence_artifacts(text)
    if _is_dangling_fragment(clean):
        return ""
    sentences = [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in re.findall(r"[^.!?؟۔]+[.!?؟۔]", clean)
    ]
    sentences = [
        sentence
        for sentence in sentences
        if sentence
        and not _is_dangling_fragment(sentence)
        and "frame index" not in sentence.casefold()
    ]
    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    return " ".join(sentences).strip()


def _remove_broken_sentence_artifacts(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    clean = re.sub(r"\([^)]*$", "", clean).strip()
    clean = re.sub(r"\b(?:with\s+)?(?:high|low|medium)?\s*confidence\s*\(?\d+(?:\.\d*)?$", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\b(?:at|index|frame)\s+\d+(?:\s+and\s+\d+)?\.?$", "", clean, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"[\d.() ]+", clean):
        clean = re.sub(r"\b\d+(?:\.\d*)?$", "", clean).strip()
    return clean


def _is_dangling_fragment(text: str) -> bool:
    clean = str(text or "").strip().strip(":")
    if not clean:
        return True
    if clean.casefold() in {"wearing", "visual observations", "summary", "strongest model evidence"}:
        return True
    if clean.endswith("("):
        return True
    if re.fullmatch(r"[\d.() ]+", clean):
        return True
    words = re.findall(r"[\w\u0600-\u06FF]+", clean)
    return len(words) <= 1 and not clean.endswith(tuple(_TERMINAL_PUNCTUATION))


def _truncate_at_sentence_boundary(text: str, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    clipped = clean[: max(0, max_chars - 3)].rstrip()
    matches = list(re.finditer(r"[.!?؟۔](?=\s|$)", clipped))
    if matches:
        return clipped[: matches[-1].end()].strip()
    return clipped.rstrip(".!?؟۔") + "."


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.findall(r"[^.!?]+[.!?]?", text)
        if sentence.strip()
    ]


def _single_sentence(text: str) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return clean
    match = re.search(r"^.+?[.!?](?=\s|$)", clean)
    if match:
        return match.group(0).strip()
    return clean.rstrip(".!?") + "."


def _safe_stem(clip_id: str) -> str:
    chars = []
    for ch in Path(clip_id).stem:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("_")
    return ("".join(chars).strip("._-") or "clip")[:80]
