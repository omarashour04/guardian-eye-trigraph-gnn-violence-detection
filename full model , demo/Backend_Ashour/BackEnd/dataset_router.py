"""
Guardian Eye dataset router.

This module deliberately uses lightweight OpenCV frame features only. It does
not load Qwen/VLM models and it does not decide violence, confidence, or final
verdict. Its only job is to choose the dataset-specific checkpoints that should
feed the existing V9 classifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

DatasetName = Literal["RLVS", "HF", "NTU"]

DATASETS: tuple[DatasetName, ...] = ("RLVS", "HF", "NTU")

V9_CHECKPOINTS: dict[DatasetName, str] = {
    "RLVS": "models/v9/RLVS_best.pt",
    "HF": "models/v9/HF_best.pt",
    "NTU": "models/v9/NTU_best.pt",
}

VIDEOMAE_CHECKPOINTS: dict[DatasetName, str] = {
    "RLVS": "models/videomae/RLVS_videomae_best.pt",
    "HF": "models/videomae/HF_videomae_best.pt",
    "NTU": "models/videomae/NTU-videomae_best.pt",
}


@dataclass(frozen=True)
class DatasetRoute:
    selected_dataset: DatasetName
    routing_confidence: float
    routing_reason: str
    dataset_similarity: dict[str, float]
    selected_v9_checkpoint: str | None
    selected_videomae_checkpoint: str | None
    route_source: str
    fallback_used: bool = False
    requested_dataset: DatasetName | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "selected_dataset": self.selected_dataset,
            "routing_confidence": self.routing_confidence,
            "routing_reason": self.routing_reason,
            "dataset_similarity": self.dataset_similarity,
            "selected_v9_checkpoint": _checkpoint_label(self.selected_v9_checkpoint),
            "selected_videomae_checkpoint": _checkpoint_label(self.selected_videomae_checkpoint),
            "route_source": self.route_source,
            "fallback_used": self.fallback_used,
            "requested_dataset": self.requested_dataset,
        }


def routing_enabled() -> bool:
    return os.getenv("GUARDIAN_MODEL_ROUTING_ENABLED", "0") == "1"


def route_video(video_path: str, *, sample_count: int = 6) -> DatasetRoute | None:
    if not routing_enabled():
        return None

    forced = _forced_dataset()
    if forced is not None:
        return _finalize_route(
            forced,
            {name: 1.0 if name == forced else 0.0 for name in DATASETS},
            f"Dataset route forced by GUARDIAN_FORCE_DATASET_ROUTE={forced}.",
            route_source="forced",
        )

    frames = _sample_frames(video_path, max(4, min(sample_count, 8)))
    if not frames:
        return _finalize_route(
            "RLVS",
            {"RLVS": 0.60, "HF": 0.05, "NTU": 0.25},
            "Frame sampling failed; defaulting to the broad real-life RLVS model.",
            route_source="fallback",
        )

    features = _extract_features(frames)
    raw_scores = _score_features(features)
    similarity = _normalize_scores(raw_scores)
    selected = max(DATASETS, key=lambda name: similarity[name])
    reason = _reason(selected, features)
    confidence = float(similarity[selected])
    return _finalize_route(
        selected,
        similarity,
        reason,
        route_source="rules",
        routing_confidence=confidence,
    )


def _forced_dataset() -> DatasetName | None:
    value = os.getenv("GUARDIAN_FORCE_DATASET_ROUTE", "").strip().upper()
    if not value:
        return None
    aliases = {
        "HOCKEY": "HF",
        "HOCKEYFIGHT": "HF",
        "HOCKEY_FIGHT": "HF",
        "CCTV": "NTU",
        "NTU_CCTV": "NTU",
    }
    value = aliases.get(value, value)
    if value not in DATASETS:
        print(f"[router] ignoring invalid GUARDIAN_FORCE_DATASET_ROUTE={value!r}")
        return None
    return value  # type: ignore[return-value]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_root() -> Path:
    configured = os.getenv("GUARDIAN_MODEL_ROOT", "").strip()
    return Path(configured) if configured else _project_root()


def _resolve(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = _model_root() / p
    return str(p)


def _existing(path: str | None) -> str | None:
    resolved = _resolve(path)
    if resolved and Path(resolved).exists():
        return resolved
    return None


def checkpoint_for_dataset(dataset: DatasetName) -> tuple[str | None, bool]:
    selected = _existing(V9_CHECKPOINTS[dataset])
    if selected:
        return selected, False
    rlvs = _existing(V9_CHECKPOINTS["RLVS"])
    if rlvs:
        return rlvs, True
    env = _existing(os.getenv("GUARDIAN_V9_CKPT", ""))
    return env, True


def videomae_for_dataset(dataset: DatasetName) -> tuple[str | None, bool]:
    selected = _existing(VIDEOMAE_CHECKPOINTS[dataset])
    if selected:
        return selected, False
    rlvs = _existing(VIDEOMAE_CHECKPOINTS["RLVS"])
    if rlvs:
        return rlvs, True
    env = _existing(os.getenv("GUARDIAN_VIDEOMAE_CKPT", ""))
    return env, True


def _finalize_route(
    selected: DatasetName,
    similarity: dict[str, float],
    reason: str,
    *,
    route_source: str,
    routing_confidence: float | None = None,
) -> DatasetRoute:
    v9_path, v9_fallback = checkpoint_for_dataset(selected)
    videomae_path, videomae_fallback = videomae_for_dataset(selected)
    fallback_used = bool(v9_fallback or videomae_fallback)
    if fallback_used:
        reason = (
            f"{reason} One or more selected checkpoints were unavailable, so "
            "the router used the configured fallback priority."
        )
    confidence = float(routing_confidence if routing_confidence is not None else similarity[selected])
    return DatasetRoute(
        selected_dataset=selected,
        routing_confidence=round(max(0.0, min(1.0, confidence)), 4),
        routing_reason=reason,
        dataset_similarity={name: round(float(similarity.get(name, 0.0)), 4) for name in DATASETS},
        selected_v9_checkpoint=v9_path,
        selected_videomae_checkpoint=videomae_path,
        route_source=route_source,
        fallback_used=fallback_used,
        requested_dataset=selected,
    )


def _checkpoint_label(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).name


def _sample_frames(video_path: str, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        indices = set(range(count))
    else:
        indices = {int(round(i * (total - 1) / max(count - 1, 1))) for i in range(count)}

    frames: list[np.ndarray] = []
    idx = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in indices:
            frames.append(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA))
        idx += 1
    cap.release()
    return frames


def _extract_features(frames: list[np.ndarray]) -> dict[str, float]:
    hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames]
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    ice_ratios = []
    green_ratios = []
    dark_ratios = []
    edge_ratios = []
    vertical_ratios = []
    for hsv, gray in zip(hsv_frames, gray_frames):
        h, s, v = cv2.split(hsv)
        ice = (s < 45) & (v > 155)
        green = (h >= 35) & (h <= 95) & (s > 45) & (v > 45)
        dark = v < 55
        edges = cv2.Canny(gray, 60, 140)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        vertical = np.abs(sobel_x) > (np.abs(sobel_y) * 1.25 + 10)

        ice_ratios.append(float(ice.mean()))
        green_ratios.append(float(green.mean()))
        dark_ratios.append(float(dark.mean()))
        edge_ratios.append(float((edges > 0).mean()))
        vertical_ratios.append(float(vertical.mean()))

    motion = []
    for left, right in zip(gray_frames, gray_frames[1:]):
        diff = cv2.absdiff(left, right)
        motion.append(float(diff.mean() / 255.0))

    return {
        "ice_ratio": float(np.mean(ice_ratios)),
        "green_ratio": float(np.mean(green_ratios)),
        "dark_ratio": float(np.mean(dark_ratios)),
        "edge_ratio": float(np.mean(edge_ratios)),
        "vertical_ratio": float(np.mean(vertical_ratios)),
        "camera_motion": float(np.mean(motion)) if motion else 0.0,
        "motion_variance": float(np.var(motion)) if motion else 0.0,
    }


def _score_features(features: dict[str, float]) -> dict[str, float]:
    ice = features["ice_ratio"]
    green = features["green_ratio"]
    edge = features["edge_ratio"]
    vertical = features["vertical_ratio"]
    motion = features["camera_motion"]
    motion_var = features["motion_variance"]
    dark = features["dark_ratio"]

    hf = 0.08 + 2.6 * ice + 0.30 * edge
    ntu = 0.18 + 1.35 * max(0.0, 0.055 - motion) + 0.75 * vertical + 0.25 * edge
    rlvs = 0.28 + 1.05 * motion + 0.65 * motion_var + 0.25 * dark + 0.20 * green

    if ice > 0.45:
        hf += 0.55
        rlvs -= 0.10
        ntu -= 0.08
    if motion < 0.025 and vertical > 0.18:
        ntu += 0.35
        rlvs -= 0.08
    if motion > 0.075:
        rlvs += 0.30
        ntu -= 0.12

    return {
        "RLVS": max(0.01, rlvs),
        "HF": max(0.01, hf),
        "NTU": max(0.01, ntu),
    }


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(scores.get(name, 0.0))) for name in DATASETS)
    if total <= 0:
        return {"RLVS": 1.0, "HF": 0.0, "NTU": 0.0}
    return {name: max(0.0, float(scores.get(name, 0.0))) / total for name in DATASETS}


def _reason(dataset: DatasetName, features: dict[str, float]) -> str:
    motion = features["camera_motion"]
    ice = features["ice_ratio"]
    vertical = features["vertical_ratio"]
    edge = features["edge_ratio"]
    if dataset == "HF":
        return (
            "Sampled frames contain a large bright low-saturation region consistent "
            "with ice or rink-like footage."
        )
    if dataset == "NTU":
        return (
            "Sampled frames look comparatively static with strong structured edges, "
            "matching a CCTV or fixed-camera environment."
        )
    if motion > 0.075:
        return (
            "Sampled frames show stronger camera or scene motion, matching real-world "
            "handheld/public RLVS-style footage."
        )
    return (
        "No strong hockey-rink or CCTV-specific cue dominated the sampled frames, "
        f"so the broad RLVS route was selected (motion={motion:.3f}, "
        f"ice={ice:.3f}, vertical_edges={vertical:.3f}, edge_density={edge:.3f})."
    )
