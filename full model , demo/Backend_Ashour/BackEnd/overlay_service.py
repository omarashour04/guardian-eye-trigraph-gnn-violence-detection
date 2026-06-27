"""
Guardian Eye — Overlay Service
Draws COCO-17 skeleton lines, per-person bounding boxes, and optional
object boxes onto video frames, exports an MP4, and saves a thumbnail.

Data shapes (from EXPLANATION_RAG_SYSTEM.md §4.4 / DEMO_APP.md §7):
  frames_vit  [16, 224, 224, 3]  uint8  RGB  (T_vit=16 uniformly-sampled frames)
  skeleton    [32,  6, 17, 3]    f32    (x, y, conf) normalised by max(H,W)
  int_nodes   [32,  6,  7]       f32    (cx, cy, w, h, conf, speed, continuity)
  int_node_mask [32, 6]          bool
  obj_nodes   [32,  8,  6]       f32    (cx, cy, w, h, conf, cls_norm)   — optional
  obj_node_mask [32, 8]          bool                                     — optional

Alignment: skeleton/int_nodes are T=32; frames_vit is T=16.
Strategy: render on the 32-frame grid.  For each skeleton frame t (0..31),
find the nearest frames_vit frame by fractional index:
    vit_idx = round(t / 31 * 15)   (maps [0,31] → [0,15])

Public API
----------
render_overlay(
    clip_id,
    frames_vit,          # np.ndarray [16,224,224,3] uint8
    skeleton=None,       # np.ndarray [32,6,17,3]   float32
    int_nodes=None,      # np.ndarray [32,6,7]      float32
    int_node_mask=None,  # np.ndarray [32,6]        bool
    obj_nodes=None,      # np.ndarray [32,8,6]      float32   (optional)
    obj_node_mask=None,  # np.ndarray [32,8]        bool      (optional)
    orig_hw=None,        # (H, W) original frame size before preprocessing
    fps=8,               # output video frame-rate  (32 frames @ 8 fps = 4 s)
    verdict=None,        # str — drawn as a banner on every frame
    confidence=None,     # float
) -> OverlayResult(overlay_path, thumbnail_path)

Mock entry-point (Phase 3)
--------------------------
render_mock_overlay(clip_id) -> OverlayResult
  Generates a solid-colour placeholder MP4 + thumbnail so the frontend
  can work before Phase 4 wires real NPZ data.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Output directories ────────────────────────────────────────────────────────

OVERLAYS_DIR   = Path("static/overlays")
THUMBNAILS_DIR = Path("static/thumbnails")
OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

MOCK_MODE: bool = os.getenv("GUARDIAN_MOCK", "1") == "1"

# ── COCO-17 skeleton edge list ────────────────────────────────────────────────
# Joint indices: 0=nose 1=left_eye 2=right_eye 3=left_ear 4=right_ear
#   5=left_shoulder 6=right_shoulder 7=left_elbow 8=right_elbow
#   9=left_wrist 10=right_wrist 11=left_hip 12=right_hip
#   13=left_knee 14=right_knee 15=left_ankle 16=right_ankle

COCO17_EDGES: list[tuple[int, int]] = [
    # face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # left arm
    (5, 7), (7, 9),
    # right arm
    (6, 8), (8, 10),
    # left leg
    (11, 13), (13, 15),
    # right leg
    (12, 14), (14, 16),
]

# ── Per-person colours (BGR) — stable across frames via ByteTrack index ──────
_PERSON_COLOURS: list[tuple[int, int, int]] = [
    (0, 200, 255),   # 0 — yellow-orange
    (255, 80,  80),  # 1 — blue
    (80,  255, 80),  # 2 — green
    (200, 80,  255), # 3 — purple
    (255, 200, 0),   # 4 — cyan
    (80,  80,  255), # 5 — red
]

# Object box colour (BGR)
_OBJ_COLOUR  = (0, 0, 220)      # red
_OBJ_LABEL_COLOUR = (255, 255, 255)

CONF_THRESH  = 0.25             # minimum joint confidence to draw
FRAME_SIZE   = (224, 224)       # output frame dimensions (matches frames_vit)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OverlayResult:
    overlay_path:   str   # relative path served at /static/overlays/<clip_id>.mp4
    thumbnail_path: str   # relative path served at /static/thumbnails/<clip_id>.jpg
    overlays: dict[str, Optional[str]] = field(default_factory=dict)
    overlay_status: dict[str, str] = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _person_colour(m: int) -> tuple[int, int, int]:
    return _PERSON_COLOURS[m % len(_PERSON_COLOURS)]


def _denorm_point(x_n: float, y_n: float, orig_hw: tuple[int, int],
                  frame_hw: tuple[int, int]) -> tuple[int, int]:
    """
    Convert a normalised coordinate back to pixel space.

    Preprocessing normalised by max(H, W).  We need to:
      1. Multiply by max(H, W) to get original-resolution pixels.
      2. Scale to the rendered frame size (224×224).
    """
    H, W   = orig_hw
    fH, fW = frame_hw
    max_dim = max(H, W)
    px = int(x_n * max_dim * fW / W)
    py = int(y_n * max_dim * fH / H)
    return px, py


def _denorm_box(
    cx_n: float,
    cy_n: float,
    w_n: float,
    h_n: float,
    orig_hw: tuple[int, int],
    frame_hw: tuple[int, int],
) -> tuple[int, int, int, int, int, int]:
    """Convert a normalized center box to rendered pixel coordinates."""
    H, W = orig_hw
    fH, fW = frame_hw
    max_dim = max(H, W)
    scale_x = fW / W
    scale_y = fH / H

    cx = int(cx_n * max_dim * scale_x)
    cy = int(cy_n * max_dim * scale_y)
    bw = int(w_n * max_dim * scale_x)
    bh = int(h_n * max_dim * scale_y)

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)
    return x1, y1, x2, y2, cx, cy


def _vit_idx(t: int, T: int = 32, T_vit: int = 16) -> int:
    """Map skeleton frame index t ∈ [0, T-1] to nearest frames_vit index."""
    return round(t / (T - 1) * (T_vit - 1))


def _draw_skeleton(
    canvas: np.ndarray,
    skel_frame: np.ndarray,   # [6, 17, 3]  float32
    orig_hw: tuple[int, int],
    frame_hw: tuple[int, int],
) -> np.ndarray:
    """Draw COCO-17 skeleton for all valid people onto canvas (in-place)."""
    M = skel_frame.shape[0]
    for m in range(M):
        joints = skel_frame[m]        # [17, 3]
        colour = _person_colour(m)

        # Collect valid joint pixel positions
        pts: dict[int, tuple[int, int]] = {}
        for j in range(17):
            x_n, y_n, conf = joints[j]
            if conf > CONF_THRESH:
                pts[j] = _denorm_point(x_n, y_n, orig_hw, frame_hw)

        # Draw edges
        for (a, b) in COCO17_EDGES:
            if a in pts and b in pts:
                cv2.line(canvas, pts[a], pts[b], colour, 2, cv2.LINE_AA)

        # Draw joints
        for j, pt in pts.items():
            cv2.circle(canvas, pt, 3, colour, -1, cv2.LINE_AA)

    return canvas


def _draw_person_boxes(
    canvas: np.ndarray,
    nodes_frame: np.ndarray,   # [6, 7]   float32  (cx,cy,w,h,conf,speed,cont)
    mask_frame: np.ndarray,    # [6]      bool
    orig_hw: tuple[int, int],
    frame_hw: tuple[int, int],
) -> np.ndarray:
    """Draw per-person bounding boxes (one colour per track index)."""
    M = nodes_frame.shape[0]

    for m in range(M):
        if not mask_frame[m]:
            continue
        cx_n, cy_n, w_n, h_n = nodes_frame[m, :4]
        x1, y1, x2, y2, _, _ = _denorm_box(
            cx_n, cy_n, w_n, h_n, orig_hw, frame_hw
        )

        colour = _person_colour(m)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"P{m}", (x1, max(y1 - 4, 10)),
                    FONT, 0.45, colour, 1, cv2.LINE_AA)

    return canvas


def _draw_object_boxes(
    canvas: np.ndarray,
    obj_frame: np.ndarray,     # [8, 6]  float32 (cx,cy,w,h,conf,cls_norm)
    obj_mask: np.ndarray,      # [8]     bool
    orig_hw: tuple[int, int],
    frame_hw: tuple[int, int],
) -> np.ndarray:
    """Draw optional object bounding boxes in red."""
    N  = obj_frame.shape[0]

    for n in range(N):
        if not obj_mask[n]:
            continue
        cx_n, cy_n, w_n, h_n, conf, cls_n = obj_frame[n]
        if conf < CONF_THRESH:
            continue

        x1, y1, x2, y2, _, _ = _denorm_box(
            cx_n, cy_n, w_n, h_n, orig_hw, frame_hw
        )

        cv2.rectangle(canvas, (x1, y1), (x2, y2), _OBJ_COLOUR, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"obj{n}", (x1, max(y1 - 4, 10)),
                    FONT, 0.40, _OBJ_LABEL_COLOUR, 1, cv2.LINE_AA)

    return canvas


def _person_centers(
    nodes_frame: np.ndarray,
    mask_frame: np.ndarray,
    orig_hw: tuple[int, int],
    frame_hw: tuple[int, int],
) -> dict[int, tuple[int, int]]:
    """Return rendered centers for valid person nodes."""
    centers: dict[int, tuple[int, int]] = {}
    for m in range(nodes_frame.shape[0]):
        if not mask_frame[m]:
            continue
        cx_n, cy_n, w_n, h_n = nodes_frame[m, :4]
        _, _, _, _, cx, cy = _denorm_box(cx_n, cy_n, w_n, h_n, orig_hw, frame_hw)
        centers[m] = (cx, cy)
    return centers


def _object_centers(
    obj_frame: np.ndarray,
    obj_mask: np.ndarray,
    orig_hw: tuple[int, int],
    frame_hw: tuple[int, int],
) -> dict[int, tuple[int, int]]:
    """Return rendered centers for valid object nodes."""
    centers: dict[int, tuple[int, int]] = {}
    for n in range(obj_frame.shape[0]):
        if not obj_mask[n]:
            continue
        cx_n, cy_n, w_n, h_n, conf, _ = obj_frame[n]
        if conf < CONF_THRESH:
            continue
        _, _, _, _, cx, cy = _denorm_box(cx_n, cy_n, w_n, h_n, orig_hw, frame_hw)
        centers[n] = (cx, cy)
    return centers


def _draw_person_nodes(
    canvas: np.ndarray,
    nodes_frame: np.ndarray,
    mask_frame: np.ndarray,
    orig_hw: tuple[int, int],
    frame_hw: tuple[int, int],
    draw_boxes: bool = True,
) -> dict[int, tuple[int, int]]:
    """Draw simple person graph nodes and return their centers."""
    centers = _person_centers(nodes_frame, mask_frame, orig_hw, frame_hw)
    if draw_boxes:
        canvas = _draw_person_boxes(canvas, nodes_frame, mask_frame, orig_hw, frame_hw)

    for m, center in centers.items():
        colour = _person_colour(m)
        cv2.circle(canvas, center, 5, colour, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"P{m}", (center[0] + 5, center[1] - 5),
                    FONT, 0.38, colour, 1, cv2.LINE_AA)
    return centers


def _draw_interaction_links(
    canvas: np.ndarray,
    centers: dict[int, tuple[int, int]],
    edge_frame: np.ndarray,
    edge_mask_frame: np.ndarray,
) -> np.ndarray:
    """Draw proximity/interaction links between person nodes."""
    for i, center_i in centers.items():
        for j, center_j in centers.items():
            if i >= j or not edge_mask_frame[i, j]:
                continue
            dist, iou, close, rel_spd = edge_frame[i, j]
            if close < 0.5 and iou <= 0.0 and dist >= 0.25:
                continue
            width = 2 if close >= 0.5 else 1
            colour = (30, 120, 255) if close >= 0.5 else (120, 120, 120)
            cv2.line(canvas, center_i, center_j, colour, width, cv2.LINE_AA)
            mid = ((center_i[0] + center_j[0]) // 2, (center_i[1] + center_j[1]) // 2)
            cv2.circle(canvas, mid, 3, colour, -1, cv2.LINE_AA)
    return canvas


def _draw_person_object_links(
    canvas: np.ndarray,
    person_centers: dict[int, tuple[int, int]],
    object_centers: dict[int, tuple[int, int]],
    po_frame: np.ndarray,
    po_mask_frame: np.ndarray,
) -> np.ndarray:
    """Draw person-object links where the PO graph indicates useful proximity."""
    for p_idx, p_center in person_centers.items():
        for o_idx, o_center in object_centers.items():
            if not po_mask_frame[p_idx, o_idx]:
                continue
            wrist_d, body_d, po_iou, near_wrist, near_body = po_frame[p_idx, o_idx]
            if (
                near_wrist < 0.5
                and near_body < 0.5
                and po_iou <= 0.01
                and wrist_d >= 0.2
                and body_d >= 0.3
            ):
                continue
            colour = (0, 80, 220) if near_wrist >= 0.5 else (0, 140, 180)
            cv2.line(canvas, p_center, o_center, colour, 2, cv2.LINE_AA)
            mid = ((p_center[0] + o_center[0]) // 2, (p_center[1] + o_center[1]) // 2)
            cv2.circle(canvas, mid, 3, colour, -1, cv2.LINE_AA)
    return canvas


def _draw_verdict_banner(
    canvas: np.ndarray,
    verdict: str,
    confidence: float,
) -> np.ndarray:
    """Stamp a verdict + confidence banner in the top-left corner."""
    label    = f"{verdict.upper()}  {confidence:.0%}"
    bg_color = (0, 0, 180) if verdict == "violence" else (0, 140, 0)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(canvas, (4, 4), (tw + 10, th + 12), bg_color, -1)
    cv2.putText(canvas, label, (7, th + 6),
                FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    """Write BGR frames to a browser-compatible MP4 file."""
    if not frames:
        raise ValueError("No frames to write")
    H, W = frames[0].shape[:2]
    raw_path = out_path.with_name(f"{out_path.stem}.opencv{out_path.suffix}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raw_path.replace(out_path)
        return

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_path),
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        raw_path.replace(out_path)
    else:
        raw_path.unlink(missing_ok=True)


def _save_thumbnail(frame_bgr: np.ndarray, thumb_path: Path) -> None:
    """Save the middle frame as a JPEG thumbnail."""
    cv2.imwrite(str(thumb_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ── Public: real render ───────────────────────────────────────────────────────

def _write_stream(
    clip_id: str,
    stream: str,
    frames: list[np.ndarray],
    fps: int,
) -> str:
    stem = Path(clip_id).stem
    overlay_path = OVERLAYS_DIR / f"{stem}_{stream}.mp4"
    _write_video(frames, overlay_path, fps)
    return str(overlay_path)


def overlay_status_from_paths(
    overlays: dict[str, Optional[str]],
    *,
    fallback_placeholder: bool = False,
) -> dict[str, str]:
    """Return demo-safe per-stream availability labels."""
    status: dict[str, str] = {}
    for stream in ("skeleton", "interaction", "object", "vit"):
        path = overlays.get(stream)
        if path and Path(path).exists():
            status[stream] = "fallback_placeholder" if fallback_placeholder else "available"
        else:
            status[stream] = "missing"
    return status


def _warn_missing_stream(stream: str, keys: list[str]) -> None:
    missing = ", ".join(keys)
    print(f"[overlay] stream={stream} skipped; missing {missing}")


def render_stream_overlays(
    clip_id: str,
    frames_vit: Optional[np.ndarray] = None,
    skeleton: Optional[np.ndarray] = None,
    int_nodes: Optional[np.ndarray] = None,
    int_edges: Optional[np.ndarray] = None,
    int_node_mask: Optional[np.ndarray] = None,
    int_edge_mask: Optional[np.ndarray] = None,
    obj_nodes: Optional[np.ndarray] = None,
    obj_node_mask: Optional[np.ndarray] = None,
    po_edges: Optional[np.ndarray] = None,
    po_edge_mask: Optional[np.ndarray] = None,
    orig_hw: tuple[int, int] = (224, 224),
    fps: int = 8,
) -> dict[str, Optional[str]]:
    """Render stream-specific graph videos on white canvases."""
    overlays: dict[str, Optional[str]] = {
        "skeleton": None,
        "interaction": None,
        "object": None,
        "vit": None,
    }

    frame_hw = FRAME_SIZE
    if frames_vit is not None and frames_vit.size:
        frame_hw = (frames_vit.shape[1], frames_vit.shape[2])

    if skeleton is not None:
        T = skeleton.shape[0]
        frames: list[np.ndarray] = []
        for t in range(T):
            canvas = np.full((frame_hw[0], frame_hw[1], 3), 255, dtype=np.uint8)
            canvas = _draw_skeleton(canvas, skeleton[t], orig_hw, frame_hw)
            cv2.putText(canvas, f"{t+1}/{T}", (frame_hw[1] - 42, frame_hw[0] - 6),
                        FONT, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
            frames.append(canvas)
        overlays["skeleton"] = _write_stream(clip_id, "skeleton", frames, fps)
    else:
        _warn_missing_stream("skeleton", ["skeleton"])

    interaction_missing = [
        key for key, value in {
            "int_nodes": int_nodes,
            "int_edges": int_edges,
            "int_node_mask": int_node_mask,
            "int_edge_mask": int_edge_mask,
        }.items()
        if value is None
    ]
    if not interaction_missing:
        T = int_nodes.shape[0]
        frames = []
        for t in range(T):
            canvas = np.full((frame_hw[0], frame_hw[1], 3), 255, dtype=np.uint8)
            centers = _draw_person_nodes(
                canvas, int_nodes[t], int_node_mask[t], orig_hw, frame_hw
            )
            canvas = _draw_interaction_links(
                canvas, centers, int_edges[t], int_edge_mask[t]
            )
            cv2.putText(canvas, f"{t+1}/{T}", (frame_hw[1] - 42, frame_hw[0] - 6),
                        FONT, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
            frames.append(canvas)
        overlays["interaction"] = _write_stream(clip_id, "interaction", frames, fps)
    else:
        _warn_missing_stream("interaction", interaction_missing)

    object_missing = [
        key for key, value in {
            "obj_nodes": obj_nodes,
            "obj_node_mask": obj_node_mask,
            "po_edges": po_edges,
            "po_edge_mask": po_edge_mask,
        }.items()
        if value is None
    ]
    if not object_missing:
        T = obj_nodes.shape[0]
        frames = []
        for t in range(T):
            canvas = np.full((frame_hw[0], frame_hw[1], 3), 255, dtype=np.uint8)
            object_centers = _object_centers(
                obj_nodes[t], obj_node_mask[t], orig_hw, frame_hw
            )
            canvas = _draw_object_boxes(
                canvas, obj_nodes[t], obj_node_mask[t], orig_hw, frame_hw
            )
            person_centers: dict[int, tuple[int, int]] = {}
            if int_nodes is not None and int_node_mask is not None:
                person_centers = _draw_person_nodes(
                    canvas,
                    int_nodes[t],
                    int_node_mask[t],
                    orig_hw,
                    frame_hw,
                    draw_boxes=False,
                )
            canvas = _draw_person_object_links(
                canvas, person_centers, object_centers, po_edges[t], po_edge_mask[t]
            )
            cv2.putText(canvas, f"{t+1}/{T}", (frame_hw[1] - 42, frame_hw[0] - 6),
                        FONT, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
            frames.append(canvas)
        overlays["object"] = _write_stream(clip_id, "object", frames, fps)
    else:
        _warn_missing_stream("object", object_missing)

    if frames_vit is not None and frames_vit.size:
        T = 32
        T_vit = frames_vit.shape[0]
        frames = []
        for t in range(T):
            vit_t = _vit_idx(t, T, T_vit)
            frames.append(cv2.cvtColor(frames_vit[vit_t], cv2.COLOR_RGB2BGR).copy())
        overlays["vit"] = _write_stream(clip_id, "vit", frames, fps)
    else:
        _warn_missing_stream("vit", ["frames_vit"])

    return overlays


def render_overlay(
    clip_id: str,
    frames_vit: np.ndarray,               # [16, 224, 224, 3] uint8 RGB
    skeleton: Optional[np.ndarray] = None,      # [32, 6, 17, 3]
    int_nodes: Optional[np.ndarray] = None,     # [32, 6, 7]
    int_node_mask: Optional[np.ndarray] = None, # [32, 6]
    obj_nodes: Optional[np.ndarray] = None,     # [32, 8, 6]
    obj_node_mask: Optional[np.ndarray] = None, # [32, 8]
    orig_hw: tuple[int, int] = (224, 224),      # original frame dims before preprocessing
    fps: int = 8,
    verdict: Optional[str] = None,
    confidence: Optional[float] = None,
) -> OverlayResult:
    """
    Render skeleton + bounding-box overlay onto frames_vit and export:
      - MP4  → static/overlays/<clip_id_stem>.mp4
      - JPEG → static/thumbnails/<clip_id_stem>.jpg

    Uses T=32 skeleton grid; maps each skeleton frame to nearest frames_vit
    frame by fractional index so overlay aligns with appearance frames.

    All skeleton/box coordinates are de-normalised from max(H,W) space back
    to 224×224 pixel space before drawing.
    """
    T     = 32
    T_vit = frames_vit.shape[0]   # should be 16
    frame_hw = (frames_vit.shape[1], frames_vit.shape[2])  # (224, 224)

    rendered: list[np.ndarray] = []

    for t in range(T):
        vit_t = _vit_idx(t, T, T_vit)
        # frames_vit is RGB; OpenCV works in BGR
        rgb   = frames_vit[vit_t]
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

        # Person bounding boxes (drawn first, under skeleton)
        if int_nodes is not None and int_node_mask is not None:
            canvas = _draw_person_boxes(
                canvas, int_nodes[t], int_node_mask[t], orig_hw, frame_hw
            )

        # Object boxes
        if obj_nodes is not None and obj_node_mask is not None:
            canvas = _draw_object_boxes(
                canvas, obj_nodes[t], obj_node_mask[t], orig_hw, frame_hw
            )

        # Skeleton (drawn last, on top)
        if skeleton is not None:
            canvas = _draw_skeleton(canvas, skeleton[t], orig_hw, frame_hw)

        # Verdict banner
        if verdict is not None and confidence is not None:
            canvas = _draw_verdict_banner(canvas, verdict, confidence)

        # Frame index counter (bottom-right)
        cv2.putText(canvas, f"{t+1}/{T}", (frame_hw[1] - 42, frame_hw[0] - 6),
                    FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        rendered.append(canvas)

    # ── Save MP4 ──────────────────────────────────────────────────────────────
    stem         = Path(clip_id).stem
    overlay_path = OVERLAYS_DIR / f"{stem}.mp4"
    _write_video(rendered, overlay_path, fps)

    # ── Save thumbnail (middle frame) ─────────────────────────────────────────
    thumb_path = THUMBNAILS_DIR / f"{stem}.jpg"
    _save_thumbnail(rendered[len(rendered) // 2], thumb_path)

    stream_overlays = {
        "skeleton": None,
        "interaction": None,
        "object": None,
        "vit": None,
    }
    stream_status = overlay_status_from_paths(stream_overlays)

    return OverlayResult(
        overlay_path   = str(overlay_path),
        thumbnail_path = str(thumb_path),
        overlays       = stream_overlays,
        overlay_status = stream_status,
    )


# ── Public: mock render (Phase 3) ─────────────────────────────────────────────

def render_mock_overlay(
    clip_id: str,
    verdict: str = "violence",
    confidence: float = 0.94,
) -> OverlayResult:
    """
    Generate a placeholder overlay without real NPZ data.
    Produces a 32-frame MP4 of animated skeleton on a dark background
    so the frontend has a real video to display immediately.
    """
    T   = 32
    fps = 8
    H, W = 224, 224
    frames: list[np.ndarray] = []

    for t in range(T):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)   # dark grey background

        # Animated stick-figure (two people walking toward each other)
        phase = t / T

        for m, (cx_base, direction) in enumerate([(60, 1), (164, -1)]):
            colour = _person_colour(m)
            cx = int(cx_base + direction * phase * 20)

            # Torso
            shoulder_y, hip_y = 80, 120
            cv2.line(canvas, (cx, shoulder_y), (cx, hip_y), colour, 2)
            # Head
            cv2.circle(canvas, (cx, 65), 12, colour, 2)
            # Arms (swinging)
            swing = int(15 * np.sin(phase * np.pi * 4 + m * np.pi))
            cv2.line(canvas, (cx, shoulder_y), (cx - 20, shoulder_y + 25 + swing), colour, 2)
            cv2.line(canvas, (cx, shoulder_y), (cx + 20, shoulder_y + 25 - swing), colour, 2)
            # Legs
            step = int(15 * np.sin(phase * np.pi * 4 + m * np.pi))
            cv2.line(canvas, (cx, hip_y), (cx - 12 + step, hip_y + 35), colour, 2)
            cv2.line(canvas, (cx, hip_y), (cx + 12 - step, hip_y + 35), colour, 2)
            # Bounding box
            cv2.rectangle(canvas, (cx - 28, 50), (cx + 28, 160), colour, 1)
            cv2.putText(canvas, f"P{m}", (cx - 12, 45), FONT, 0.38, colour, 1)

        # Verdict banner
        canvas = _draw_verdict_banner(canvas, verdict, confidence)

        # Frame counter
        cv2.putText(canvas, f"MOCK {t+1}/{T}", (W - 72, H - 6),
                    FONT, 0.35, (120, 120, 120), 1)

        frames.append(canvas)

    stem         = Path(clip_id).stem
    overlay_path = OVERLAYS_DIR / f"{stem}.mp4"
    _write_video(frames, overlay_path, fps)

    thumb_path = THUMBNAILS_DIR / f"{stem}.jpg"
    _save_thumbnail(frames[T // 2], thumb_path)

    stream_overlays = {
        "skeleton": str(overlay_path),
        "interaction": str(overlay_path),
        "object": str(overlay_path),
        "vit": str(overlay_path),
    }
    return OverlayResult(
        overlay_path   = str(overlay_path),
        thumbnail_path = str(thumb_path),
        overlays       = stream_overlays,
        overlay_status = overlay_status_from_paths(
            stream_overlays,
            fallback_placeholder=True,
        ),
    )
