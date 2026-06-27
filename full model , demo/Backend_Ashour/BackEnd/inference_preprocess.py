"""
Guardian Eye — Preprocessing Pipeline (inference-time)

Converts a raw video file into a V9-compatible NPZ file by:
  1. Uniformly sampling T=32 graph frames + T_vit=16 VIT frames
  2. Running YOLO11x-pose + ByteTrack (persist=True) for skeleton/interaction graphs
  3. Running YOLO11x object detector for object/person-object graphs
  4. Computing GQS quality scores
  5. Running the VideoMAE encoder to get vit_embedding [768]
  6. Saving all arrays to cache_npz/<clip_id>.npz

All constants (T, M, N, V, thresholds, normalization) are locked to match
training in trigraph_v9_rlvs_preprocess.py. Do not change them.

Env vars:
  GUARDIAN_YOLO_POSE_PATH   path to yolo11x-pose.pt  (ultralytics auto-downloads)
  GUARDIAN_YOLO_OBJ_PATH    path to yolo11x.pt
  GUARDIAN_VIDEOMAE_CKPT    path to videomae_best.pt  (optional — zeros if missing)
"""

from __future__ import annotations
import gc
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── Constants (locked to training) ────────────────────────────────────────────
T         = 32      # graph frames
T_VIT     = 16      # VideoMAE frames
M         = 6       # max persons
N         = 8       # max objects
V         = 17      # COCO-17 joints
VIT_SIZE  = 224
POSE_CONF = 0.25
OBJ_CONF  = 0.25
OBJ_IOU   = 0.45
MIN_JOINTS = 5      # for q_skel

# ── Singleton models (loaded once per process) ─────────────────────────────────
_yolo_pose   = None
_yolo_obj    = None
_vit_encoder = None   # VideoMAEModel encoder
_vit_proc    = None   # AutoImageProcessor
_vit_ckpt_path: str | None = None


def _cuda_memory_snapshot(label: str) -> dict[str, float | str]:
    if not torch.cuda.is_available():
        snapshot: dict[str, float | str] = {"label": label, "device": "cpu"}
    else:
        snapshot = {
            "label": label,
            "device": torch.cuda.get_device_name(0),
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        }
    print(f"[preprocess][gpu] {snapshot}")
    return snapshot


def _move_to_cpu_if_possible(model, label: str) -> None:
    if model is None:
        return
    try:
        model.to("cpu")
        print(f"[preprocess][gpu] moved {label} to CPU")
        return
    except Exception:
        pass
    try:
        inner = getattr(model, "model", None)
        if inner is not None:
            inner.to("cpu")
            print(f"[preprocess][gpu] moved {label}.model to CPU")
    except Exception as exc:
        print(f"[preprocess][gpu] could not move {label} to CPU: {exc}")


def _release_preprocess_models() -> None:
    """Release preprocessing models so V9/VLM/LLM can use GPU sequentially."""
    global _yolo_pose, _yolo_obj, _vit_encoder, _vit_proc, _vit_ckpt_path
    if os.getenv("GUARDIAN_RELEASE_PREPROCESS_MODELS", "1") == "0":
        print("[preprocess][gpu] model release disabled by GUARDIAN_RELEASE_PREPROCESS_MODELS=0")
        return

    _move_to_cpu_if_possible(_yolo_pose, "yolo_pose")
    _move_to_cpu_if_possible(_yolo_obj, "yolo_obj")
    _move_to_cpu_if_possible(_vit_encoder, "videomae")
    _yolo_pose = None
    _yolo_obj = None
    _vit_encoder = None
    _vit_proc = None
    _vit_ckpt_path = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _cuda_memory_snapshot("after_preprocess_cleanup")


def _load_yolo_singletons() -> None:
    global _yolo_pose, _yolo_obj
    if _yolo_pose is not None and _yolo_obj is not None:
        return

    from ultralytics import YOLO

    pose_path = os.getenv("GUARDIAN_YOLO_POSE_PATH", "yolo11x-pose.pt")
    obj_path  = os.getenv("GUARDIAN_YOLO_OBJ_PATH",  "yolo11x.pt")

    print(f"[preprocess] Loading pose model: {pose_path}")
    _yolo_pose = YOLO(pose_path)

    print(f"[preprocess] Loading object model: {obj_path}")
    _yolo_obj = YOLO(obj_path)


def _load_vit_singleton(ckpt_path_override: str | None = None) -> bool:
    """Returns True if VideoMAE encoder was loaded successfully."""
    global _vit_encoder, _vit_proc, _vit_ckpt_path

    requested = ckpt_path_override or os.getenv("GUARDIAN_VIDEOMAE_CKPT", "")
    ckpt_path = str(Path(requested).resolve()) if requested else ""
    if _vit_encoder is not None and _vit_ckpt_path == ckpt_path:
        print("[preprocess][VideoMAE] model_loaded=yes cached_singleton=yes fallback_used=no")
        return True
    if _vit_encoder is not None and _vit_ckpt_path != ckpt_path:
        _move_to_cpu_if_possible(_vit_encoder, "videomae")
        _vit_encoder = None
        _vit_proc = None
        _vit_ckpt_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[preprocess][VideoMAE] checkpoint={ckpt_path or '<unset>'}")
    if not ckpt_path or not Path(ckpt_path).exists():
        warnings.warn(
            "[preprocess] GUARDIAN_VIDEOMAE_CKPT not set or file missing. "
            "vit_embedding will be zeros — ViT stream will not contribute.",
            RuntimeWarning, stacklevel=2,
        )
        print("[preprocess][VideoMAE] model_loaded=no fallback_used=yes reason=checkpoint_missing")
        return False

    try:
        from transformers import VideoMAEModel, AutoImageProcessor
        print(f"[preprocess] Loading VideoMAE encoder from {ckpt_path}")
        _vit_proc    = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        base_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        # Load fine-tuned weights — checkpoint was saved as EMA state dict
        ck = torch.load(ckpt_path, map_location="cpu")
        if "ema_state_dict" in ck:
            # The fine-tuned VideoMAEClassifier wraps the encoder under self.encoder
            encoder_state = {
                k[len("encoder."):]: v
                for k, v in ck["ema_state_dict"].items()
                if k.startswith("encoder.")
            }
            missing, unexpected = base_encoder.load_state_dict(encoder_state, strict=False)
            if missing:
                print(f"[preprocess] VideoMAE missing keys: {missing[:5]}")
        elif "state_dict" in ck:
            encoder_state = {
                k[len("encoder."):]: v
                for k, v in ck["state_dict"].items()
                if k.startswith("encoder.")
            }
            base_encoder.load_state_dict(encoder_state, strict=False)

        _vit_encoder = base_encoder.eval()
        _vit_ckpt_path = ckpt_path
        print("[preprocess] VideoMAE encoder ready.")
        print("[preprocess][VideoMAE] model_loaded=yes fallback_used=no")
        return True

    except Exception as e:
        warnings.warn(
            f"[preprocess] Failed to load VideoMAE: {e}. "
            "Falling back to zeros for vit_embedding.",
            RuntimeWarning, stacklevel=2,
        )
        print(f"[preprocess][VideoMAE] model_loaded=no fallback_used=yes reason={e.__class__.__name__}: {e}")
        return False


def _extract_vit_embedding(frames_vit: np.ndarray) -> np.ndarray:
    """
    frames_vit: [T_VIT, H, W, 3] uint8 RGB
    Returns: [768] float32
    """
    if _vit_encoder is None:
        emb = np.zeros(768, dtype=np.float32)
        _log_vit_embedding_stats(emb, fallback_used=True)
        return emb

    # AutoImageProcessor expects list of PIL images or numpy HWC
    import PIL.Image
    pil_frames = [PIL.Image.fromarray(frames_vit[i]) for i in range(T_VIT)]
    inputs = _vit_proc(images=pil_frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # [1, T_VIT, 3, H, W] or [T_VIT, 3, H, W]

    # Ensure batch dim
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)

    device = next(_vit_encoder.parameters()).device
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        out = _vit_encoder(pixel_values=pixel_values)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0)  # [768]

    emb_np = emb.cpu().numpy().astype(np.float32)
    _log_vit_embedding_stats(emb_np, fallback_used=False)
    return emb_np


def _log_vit_embedding_stats(embedding: np.ndarray, *, fallback_used: bool) -> None:
    emb = np.asarray(embedding, dtype=np.float32)
    print(
        "[preprocess][VideoMAE] embedding "
        f"shape={tuple(emb.shape)} "
        f"mean={float(emb.mean()):.6f} "
        f"std={float(emb.std()):.6f} "
        f"fallback_used={fallback_used}"
    )


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess_video(video_path: str,
                     clip_id: str,
                     cache_dir: str = "cache_npz",
                     force_reprocess: bool = False,
                     videomae_ckpt: str | None = None) -> str:
    """
    Run full preprocessing on video_path, save NPZ to cache_dir/<clip_id>.npz.
    Returns the path to the saved NPZ file.

    If an NPZ already exists for this clip_id it is returned immediately
    (re-upload of the same clip skips re-processing).
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    npz_path = str(Path(cache_dir) / f"{clip_id}.npz")
    npz_exists = Path(npz_path).exists()
    input_size = Path(video_path).stat().st_size if Path(video_path).exists() else 0
    print(
        "[preprocess] request "
        f"video_path={video_path!r} "
        f"clip_id={clip_id!r} "
        f"npz_path={npz_path!r} "
        f"npz_exists={npz_exists} "
        f"force_reprocess={force_reprocess} "
        f"videomae_ckpt={Path(videomae_ckpt).name if videomae_ckpt else '<env>'} "
        f"input_size_bytes={input_size}"
    )

    if npz_exists and not force_reprocess:
        print(f"[preprocess] cache hit; reusing NPZ: {npz_path}")
        return npz_path
    if npz_exists and force_reprocess:
        print(f"[preprocess] cache bypass; regenerating NPZ: {npz_path}")
    else:
        print(f"[preprocess] cache miss; generating NPZ: {npz_path}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _cuda_memory_snapshot("before_yolo")
    _load_yolo_singletons()
    _cuda_memory_snapshot("after_yolo_load")
    vit_available = _load_vit_singleton(videomae_ckpt)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_frames_all: List[np.ndarray] = []
    first_frame_read = False
    last_frame_read = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not raw_frames_all:
            first_frame_read = True
        last_frame_read = True
        raw_frames_all.append(frame)
    cap.release()

    n_frames = len(raw_frames_all)
    print(
        "[preprocess] video read "
        f"video_path={video_path!r} "
        f"first_frame_read={first_frame_read} "
        f"last_frame_read={last_frame_read} "
        f"total_frames={n_frames}"
    )
    if n_frames < 2:
        raise RuntimeError(f"Video too short ({n_frames} frames): {video_path}")

    # ── Frame index sets (same formula as training) ────────────────────────────
    graph_idx = np.linspace(0, n_frames - 1, T,     dtype=int)
    vit_idx   = np.linspace(0, n_frames - 1, T_VIT, dtype=int)

    needed     = set(graph_idx.tolist()) | set(vit_idx.tolist())
    raw_frames = {i: raw_frames_all[i] for i in needed}
    del raw_frames_all

    # ── Allocate output arrays ────────────────────────────────────────────────
    skeleton      = np.zeros((T, M, V, 3),    dtype=np.float32)
    int_nodes     = np.zeros((T, M, 7),        dtype=np.float32)
    int_edges     = np.zeros((T, M, M, 4),     dtype=np.float32)
    int_node_mask = np.zeros((T, M),            dtype=bool)
    int_edge_mask = np.zeros((T, M, M),         dtype=bool)
    obj_nodes     = np.zeros((T, N, 6),         dtype=np.float32)
    obj_node_mask = np.zeros((T, N),            dtype=bool)
    po_edges      = np.zeros((T, M, N, 5),      dtype=np.float32)
    po_edge_mask  = np.zeros((T, M, N),         dtype=bool)
    frames_vit    = np.zeros((T_VIT, VIT_SIZE, VIT_SIZE, 3), dtype=np.uint8)

    prev_centers: Dict[int, np.ndarray] = {}
    track_ages:   Dict[int, int]        = {}

    # Reset ByteTrack state between clips
    try:
        for tracker in _yolo_pose.predictor.trackers:
            tracker.reset()
    except Exception:
        pass

    # ── Detection pass over graph frames ──────────────────────────────────────
    graph_set    = set(graph_idx.tolist())
    frame_buffer: Dict[int, Dict] = {}

    for fi in tqdm(sorted(raw_frames.keys()),
                   desc=f"[preprocess] Detecting {clip_id}", unit="frame", leave=False):
        if fi not in graph_set:
            continue

        frame_bgr = raw_frames[fi]
        h, w      = frame_bgr.shape[:2]
        denom     = float(max(h, w))

        # ── Pose + ByteTrack ─────────────────────────────────────────────────
        persons = []
        try:
            res_pose = _yolo_pose.track(
                frame_bgr, conf=POSE_CONF, persist=True,
                tracker="bytetrack.yaml", verbose=False,
            )[0]
            if res_pose.keypoints is not None and res_pose.boxes is not None:
                boxes_t  = res_pose.boxes
                kps_data = res_pose.keypoints.data.cpu().numpy()
                ids_raw  = (boxes_t.id.int().cpu().tolist()
                            if boxes_t.id is not None
                            else list(range(len(boxes_t))))
                for pi in range(len(ids_raw)):
                    tid             = int(ids_raw[pi])
                    x1, y1, x2, y2 = boxes_t.xyxy[pi].cpu().numpy()
                    bx   = (x1 + x2) * 0.5 / denom
                    by   = (y1 + y2) * 0.5 / denom
                    bw   = (x2 - x1)       / denom
                    bh   = (y2 - y1)       / denom
                    conf = float(boxes_t.conf[pi].item())
                    speed = 0.0
                    if tid in prev_centers:
                        speed = float(np.linalg.norm(
                            np.array([bx, by]) - prev_centers[tid]))
                    prev_centers[tid] = np.array([bx, by])
                    track_ages[tid]   = track_ages.get(tid, 0) + 1
                    continuity = min(track_ages[tid] / (len(frame_buffer) + 1), 1.0)
                    kps_n = kps_data[pi].copy()
                    kps_n[:, 0] /= denom
                    kps_n[:, 1] /= denom
                    persons.append({
                        "id": tid, "conf": conf,
                        "box": (bx, by, bw, bh),
                        "speed": speed, "continuity": continuity,
                        "kps": kps_n,
                    })
        except Exception as e:
            print(f"[preprocess] WARN pose frame {fi}: {e}")

        # ── Object detection ─────────────────────────────────────────────────
        objects = []
        try:
            res_obj = _yolo_obj.predict(
                frame_bgr, conf=OBJ_CONF, iou=OBJ_IOU, verbose=False,
            )[0]
            for oi in range(len(res_obj.boxes)):
                x1, y1, x2, y2 = res_obj.boxes.xyxy[oi].cpu().numpy()
                objects.append({
                    "cx":   (x1 + x2) * 0.5 / denom,
                    "cy":   (y1 + y2) * 0.5 / denom,
                    "w":    (x2 - x1)       / denom,
                    "h":    (y2 - y1)       / denom,
                    "conf": float(res_obj.boxes.conf[oi].item()),
                    "cls":  float(res_obj.boxes.cls[oi].item()) / 79.0,
                })
        except Exception as e:
            print(f"[preprocess] WARN obj frame {fi}: {e}")

        frame_buffer[fi] = {"persons": persons, "objects": objects}

    # ── Fill structured arrays ────────────────────────────────────────────────
    for ti, fi in enumerate(graph_idx.tolist()):
        fi = int(fi)
        fb = frame_buffer.get(fi)
        if fb is None:
            # carry-forward from previous time step
            if ti > 0:
                skeleton[ti]      = skeleton[ti - 1]
                int_nodes[ti]     = int_nodes[ti - 1]
                int_edges[ti]     = int_edges[ti - 1]
                int_node_mask[ti] = int_node_mask[ti - 1]
                int_edge_mask[ti] = int_edge_mask[ti - 1]
                obj_nodes[ti]     = obj_nodes[ti - 1]
                obj_node_mask[ti] = obj_node_mask[ti - 1]
                po_edges[ti]      = po_edges[ti - 1]
                po_edge_mask[ti]  = po_edge_mask[ti - 1]
            continue

        persons = sorted(fb["persons"], key=lambda p: p["conf"], reverse=True)[:M]
        objects = fb["objects"][:N]

        for pi, p in enumerate(persons):
            bx, by, bw, bh = p["box"]
            int_nodes[ti, pi] = [bx, by, bw, bh,
                                  p["conf"], p["speed"], p["continuity"]]
            int_node_mask[ti, pi] = True
            if p["kps"].shape[0] == V:
                skeleton[ti, pi] = p["kps"]

        # Interaction edges (person-person)
        for pi in range(len(persons)):
            bxi, byi, bwi, bhi = persons[pi]["box"]
            for pj in range(len(persons)):
                if pi == pj:
                    continue
                bxj, byj, bwj, bhj = persons[pj]["box"]
                dist  = float(np.hypot(bxi - bxj, byi - byj))
                ix    = max(0.0, min(bxi + bwi/2, bxj + bwj/2)
                                 - max(bxi - bwi/2, bxj - bwj/2))
                iy    = max(0.0, min(byi + bhi/2, byj + bhj/2)
                                 - max(byi - bhi/2, byj - bhj/2))
                inter = ix * iy
                union = bwi * bhi + bwj * bhj - inter + 1e-6
                iou   = inter / union
                close = 1.0 if dist < 0.15 else 0.0
                rel_spd = abs(persons[pi]["speed"] - persons[pj]["speed"])
                int_edges[ti, pi, pj]     = [dist, iou, close, rel_spd]
                int_edge_mask[ti, pi, pj] = True

        # Object nodes
        for oi, obj in enumerate(objects):
            obj_nodes[ti, oi] = [obj["cx"], obj["cy"],
                                  obj["w"],  obj["h"],
                                  obj["conf"], obj["cls"]]
            obj_node_mask[ti, oi] = True

        # Person-object edges
        for pi, p in enumerate(persons):
            kps    = p["kps"]
            bx, by, bw, bh = p["box"]
            wrist_pts = [kps[wi, :2] for wi in (9, 10) if kps[wi, 2] > 0.25]
            for oi, obj in enumerate(objects):
                ox, oy, ow, oh = obj["cx"], obj["cy"], obj["w"], obj["h"]
                body_d  = float(np.hypot(bx - ox, by - oy))
                wrist_d = (float(min(np.hypot(wp[0] - ox, wp[1] - oy)
                                     for wp in wrist_pts))
                           if wrist_pts else 2.0)
                ix    = max(0.0, min(bx + bw/2, ox + ow/2) - max(bx - bw/2, ox - ow/2))
                iy    = max(0.0, min(by + bh/2, oy + oh/2) - max(by - bh/2, oy - oh/2))
                inter = ix * iy
                union = bw * bh + ow * oh - inter + 1e-6
                po_iou     = inter / union
                near_wrist = 1.0 if wrist_d < 0.10 else 0.0
                near_body  = 1.0 if body_d  < 0.20 else 0.0
                po_edges[ti, pi, oi]     = [wrist_d, body_d, po_iou,
                                             near_wrist, near_body]
                po_edge_mask[ti, pi, oi] = True

    # ── VideoMAE frames ───────────────────────────────────────────────────────
    for vi, fi in enumerate(vit_idx.tolist()):
        fi = int(fi)
        if fi in raw_frames:
            bgr = raw_frames[fi]
            rgb = cv2.cvtColor(
                cv2.resize(bgr, (VIT_SIZE, VIT_SIZE),
                           interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_BGR2RGB,
            )
            frames_vit[vi] = rgb
        elif vi > 0:
            frames_vit[vi] = frames_vit[vi - 1]

    # ── GQS ──────────────────────────────────────────────────────────────────
    valid_skel = sum(
        1 for ti in range(T)
        if any(
            int_node_mask[ti, pi]
            and int((skeleton[ti, pi, :, 2] > 0.25).sum()) >= MIN_JOINTS
            for pi in range(M)
        )
    )
    q_skel      = valid_skel / T
    q_int       = sum(int(int_node_mask[ti].sum() >= 2) for ti in range(T)) / T
    q_obj       = sum(int(obj_node_mask[ti].sum() >= 1) for ti in range(T)) / T
    q_po        = sum(int(po_edge_mask[ti].any())       for ti in range(T)) / T
    valid_ratio = sum(int(int_node_mask[ti].any())      for ti in range(T)) / T
    gqs = np.array([q_skel, q_int, q_obj, q_po, valid_ratio], dtype=np.float32)

    # ── VideoMAE embedding ────────────────────────────────────────────────────
    if vit_available:
        vit_embedding = _extract_vit_embedding(frames_vit)
    else:
        vit_embedding = np.zeros(768, dtype=np.float32)
        _log_vit_embedding_stats(vit_embedding, fallback_used=True)

    # ── Save NPZ ──────────────────────────────────────────────────────────────
    np.savez_compressed(
        npz_path,
        skeleton      = skeleton,
        int_nodes     = int_nodes,
        int_edges     = int_edges,
        int_node_mask = int_node_mask,
        int_edge_mask = int_edge_mask,
        obj_nodes     = obj_nodes,
        obj_node_mask = obj_node_mask,
        po_edges      = po_edges,
        po_edge_mask  = po_edge_mask,
        gqs           = gqs,
        frames_vit    = frames_vit,
        vit_embedding = vit_embedding,
    )

    print(f"[preprocess] Saved NPZ: {npz_path}  "
          f"(q_skel={q_skel:.2f} q_int={q_int:.2f} "
          f"q_obj={q_obj:.2f} valid={valid_ratio:.2f})")
    _cuda_memory_snapshot("after_preprocessing")
    _release_preprocess_models()
    return npz_path
