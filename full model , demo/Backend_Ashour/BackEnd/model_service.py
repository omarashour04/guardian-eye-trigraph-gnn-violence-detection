"""
Guardian Eye — Model Service
Controlled by env var GUARDIAN_MOCK=1 (default).

Phase 3  (current): returns deterministic mock data so Sara can build the
         frontend immediately against a stable contract.
Phase 4  (next):    replace _real_predict() stub with the real ST-HGT classifier.
Phase 5  (optional): enable real Qwen2.5-VL-7B + ChromaDB RAG.
"""

from __future__ import annotations
import os, random, hashlib
from schemas import GateWeights, GQSScores, Telemetry, WeaponInfo, PredictResponse, ModelRoute

MOCK_MODE: bool = os.getenv("GUARDIAN_MOCK", "1") == "1"


# ── Mock prediction ───────────────────────────────────────────────────────────

_MOCK_RESPONSES: list[dict] = [
    {
        "verdict": "violence",
        "confidence": 0.94,
        "threshold": 0.51,
        "gate": {"skeleton": 0.34, "interaction": 0.41, "object": 0.07, "vit": 0.18},
        "active_modalities": ["skeleton", "interaction", "object"],
        "inactive_modalities": ["vit"],
        "gate_validity": {
            "status": "partial",
            "message": (
                "Mock response marks ViT as inactive to demonstrate the gate "
                "validity warning when VideoMAE embeddings are unavailable."
            ),
            "raw_gate_sum": 1.0,
            "active_gate_sum": 0.82,
            "inactive_gate_sum": 0.18,
            "unavailable_contributions": ["vit"],
        },
        "gqs": {"q_skel": 0.91, "q_int": 0.88, "q_obj": 0.40, "q_po": 0.33, "valid_ratio": 0.97},
        "telemetry": {
            "people": 2,
            "peak_window": [14, 22],
            "weapon": {"flag": True, "cls": "bottle"},
        },
    },
    {
        "verdict": "violence",
        "confidence": 0.87,
        "threshold": 0.51,
        "gate": {"skeleton": 0.45, "interaction": 0.32, "object": 0.12, "vit": 0.11},
        "active_modalities": ["skeleton", "interaction", "object"],
        "inactive_modalities": ["vit"],
        "gate_validity": {
            "status": "partial",
            "message": (
                "Mock response marks ViT as inactive to demonstrate the gate "
                "validity warning when VideoMAE embeddings are unavailable."
            ),
            "raw_gate_sum": 1.0,
            "active_gate_sum": 0.89,
            "inactive_gate_sum": 0.11,
            "unavailable_contributions": ["vit"],
        },
        "gqs": {"q_skel": 0.82, "q_int": 0.79, "q_obj": 0.55, "q_po": 0.50, "valid_ratio": 0.93},
        "telemetry": {
            "people": 3,
            "peak_window": [8, 18],
            "weapon": {"flag": False, "cls": None},
        },
    },
    {
        "verdict": "non-violence",
        "confidence": 0.12,
        "threshold": 0.51,
        "gate": {"skeleton": 0.28, "interaction": 0.35, "object": 0.15, "vit": 0.22},
        "active_modalities": ["skeleton", "interaction", "object"],
        "inactive_modalities": ["vit"],
        "gate_validity": {
            "status": "partial",
            "message": (
                "Mock response marks ViT as inactive to demonstrate the gate "
                "validity warning when VideoMAE embeddings are unavailable."
            ),
            "raw_gate_sum": 1.0,
            "active_gate_sum": 0.78,
            "inactive_gate_sum": 0.22,
            "unavailable_contributions": ["vit"],
        },
        "gqs": {"q_skel": 0.95, "q_int": 0.70, "q_obj": 0.20, "q_po": 0.15, "valid_ratio": 0.98},
        "telemetry": {
            "people": 2,
            "peak_window": [0, 0],
            "weapon": {"flag": False, "cls": None},
        },
    },
]


def _mock_predict(clip_id: str) -> dict:
    """Deterministic mock — same clip_id always returns the same response."""
    idx = int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % len(_MOCK_RESPONSES)
    return _MOCK_RESPONSES[idx]


# ── Real predict (Phase 4) ────────────────────────────────────────────────────

def _real_predict(
    video_path: str,
    clip_id: str,
    force_reprocess: bool = False,
    model_route=None,
) -> dict:
    """Run YOLO preprocessing → V9 forward → calibrated result dict."""
    from inference_preprocess import preprocess_video
    from inference_classifier import classifier_forward
    print(f"[predict] real inference video_path={video_path!r} clip_id={clip_id!r}")
    npz_path = preprocess_video(
        video_path,
        clip_id,
        force_reprocess=force_reprocess,
        videomae_ckpt=getattr(model_route, "selected_videomae_checkpoint", None),
    )
    print(f"[predict] classifier input npz_path={npz_path!r}")
    return classifier_forward(
        npz_path,
        clip_id,
        checkpoint_path=getattr(model_route, "selected_v9_checkpoint", None),
    )


# ── Public interface ──────────────────────────────────────────────────────────

def run_predict(
    video_path: str,
    clip_id: str,
    force_reprocess: bool = False,
    source: str | None = None,
) -> PredictResponse:
    model_route = None
    try:
        from dataset_router import route_video
        model_route = route_video(video_path)
        if model_route is not None:
            print(f"[predict][router] {model_route.model_dump()}")
    except Exception as exc:
        print(f"[predict][router] fallback_disabled reason={exc.__class__.__name__}: {exc}")
        model_route = None

    if MOCK_MODE:
        raw = _mock_predict(clip_id)
    else:
        raw = _real_predict(
            video_path,
            clip_id,
            force_reprocess=force_reprocess,
            model_route=model_route,
        )

    t = raw["telemetry"]
    return PredictResponse(
        verdict=raw["verdict"],
        confidence=raw["confidence"],
        threshold=raw["threshold"],
        gate=GateWeights(**raw["gate"]),
        active_modalities=raw.get("active_modalities", ["skeleton", "interaction", "object", "vit"]),
        inactive_modalities=raw.get("inactive_modalities", []),
        gate_validity=raw.get("gate_validity") or {
            "status": "unknown",
            "message": "Modality activity was not checked for this prediction.",
            "raw_gate_sum": sum(float(v) for v in raw["gate"].values()),
            "active_gate_sum": sum(float(v) for v in raw["gate"].values()),
            "inactive_gate_sum": 0.0,
            "unavailable_contributions": [],
        },
        gqs=GQSScores(**raw["gqs"]),
        telemetry=Telemetry(
            people=t["people"],
            peak_window=t["peak_window"],
            weapon=WeaponInfo(flag=t["weapon"]["flag"], cls=t["weapon"]["cls"]),
        ),
        clip_id=clip_id,
        source=source,
        model_route=ModelRoute(**model_route.model_dump()) if model_route else None,
    )
