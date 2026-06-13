"""
trigraph_v9_rlvs_preprocess.py
Guardian Eye — V9  |  Phase 1: Preprocessing
Dataset : RLVS (Real Life Violence Situations)
          Soliman et al., IJCI 2019.
          Kaggle: mohamedmustafa/real-life-violence-situations-dataset
Hardware: Local workstation — RTX 3050 (4 GB VRAM), 64 GB RAM

What this script does
─────────────────────
1. Scans the RLVS dataset from a local directory:
     <DATA_ROOT>/Violence/      — 1,000 fight clips (label=1)
     <DATA_ROOT>/NonViolence/   — 1,000 non-fight clips (label=0)
2. Applies a seeded 80/10/10 train/val/test split (seed=42), stratified
   per class so both label=0 and label=1 are split independently.
   There is NO official RLVS split — document this protocol in any paper.
3. For every clip:
   a. Uniformly samples T=32 frames across the full clip duration.
   b. Runs YOLO11x-pose + ByteTrack (persist=True) for identity-stable tracks.
   c. Runs YOLO11x object detector on the same frames.
   d. Builds V9-compatible graph arrays + GQS.
   e. Captures T_vit=16 uniformly-sampled frames at 224×224 uint8 RGB for VideoMAE.
4. Saves one NPZ per clip to OUTPUT_DIR/cache_v9/.
5. Saves split_rlvs.csv and gqs_summary_rlvs.csv to OUTPUT_DIR/.

NPZ schema (V9 — identical to RWF-2000 V9 for model compatibility)
───────────────────────────────────────────────────────────────────
  skeleton      [T, M, V, 3]     float32   joint (x,y,conf), normalised
  int_nodes     [T, M, 7]        float32   cx,cy,w,h,conf,speed,continuity
  int_edges     [T, M, M, 4]     float32   dist,iou,close,rel_speed
  int_node_mask [T, M]           bool
  int_edge_mask [T, M, M]        bool
  obj_nodes     [T, N, 6]        float32   cx,cy,w,h,conf,cls_norm
  obj_node_mask [T, N]           bool
  po_edges      [T, M, N, 5]     float32   wrist_d,body_d,iou,near_wrist,near_body
  po_edge_mask  [T, M, N]        bool
  gqs           [5]              float32   q_skel,q_int,q_obj,q_po,valid_ratio
  frames_vit    [T_vit, H, W, 3] uint8     224x224 RGB for VideoMAE
  label         scalar           int64     1=fight, 0=non-fight
  split         scalar           str       train / val / test

Constants: T=32, T_vit=16, M=6, N=8, V=17

Note: CUE-Net (2024) identified mislabelled samples in RLVS.
Report the eval protocol and any detected anomalies in the paper.

Usage
─────
  python trigraph_v9_rlvs_preprocess.py
  python trigraph_v9_rlvs_preprocess.py --force-reextract
  python trigraph_v9_rlvs_preprocess.py --data-root /path/to/RLVS
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
# DATA_ROOT : folder that contains Violence/ and NonViolence/ subdirectories.
# OUTPUT_DIR: where NPZs, CSVs, and checkpoints are written.
DATA_ROOT  = Path(r"C:\Violence detection\datasets\RLVSzip")
OUTPUT_DIR = Path(r"C:\Violence detection\ashour\fresh\RLVS\preproc_output")

CACHE_DIR = OUTPUT_DIR / "cache_v9"
SPLIT_CSV = OUTPUT_DIR / "split_rlvs.csv"
GQS_CSV   = OUTPUT_DIR / "gqs_summary_rlvs.csv"


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    seed: int = 42

    # Split fractions — no official RLVS split exists
    train_frac: float = 0.80
    val_frac:   float = 0.10
    # test_frac = 0.10 (implicit)

    # Graph constants (locked — must match train.py and videomae.py)
    T:          int   = 32
    M:          int   = 6
    N:          int   = 8
    V:          int   = 17

    # Detection thresholds
    pose_conf:  float = 0.25
    obj_conf:   float = 0.25
    obj_iou:    float = 0.45
    min_joints: int   = 5

    # VideoMAE frame extraction
    T_vit:    int = 16
    vit_size: int = 224

    # YOLO weights (downloaded automatically by ultralytics on first run)
    pose_weights:  str            = "yolo11x-pose.pt"
    obj_weights:   str            = "yolo11x.pt"
    pose_fallback: Tuple[str,...] = ("yolo11l-pose.pt", "yolo11m-pose.pt",
                                     "yolo11s-pose.pt")
    obj_fallback:  Tuple[str,...] = ("yolo11l.pt", "yolo11m.pt", "yolo11s.pt")

    force_reextract: bool = False


cfg = CFG()


# ── Helpers ───────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yolo(primary: str, fallbacks: Tuple[str, ...], task: str):
    from ultralytics import YOLO
    for candidate in (primary,) + fallbacks:
        try:
            print(f"  [{task}] Trying {candidate} ...")
            m = YOLO(candidate)
            print(f"  [{task}] Loaded {candidate}")
            return m
        except Exception as e:
            print(f"  [{task}] Failed: {e}")
    raise RuntimeError(f"All YOLO candidates failed for task={task}")


def find_class_dir(root: Path, names: List[str]) -> Path:
    """Return first directory under root whose name matches any of names (case-insensitive)."""
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name.lower() in [n.lower() for n in names]:
            return p
    raise RuntimeError(
        f"Could not find a directory named one of {names} under {root}.\n"
        f"Check that DATA_ROOT is set correctly (currently: {root}).\n"
        f"Expected structure:\n"
        f"  {root}/\n"
        f"    Violence/      (1,000 fight videos)\n"
        f"    NonViolence/   (1,000 non-fight videos)"
    )


def assign_splits(clips: List[Dict], seed: int,
                  train_frac: float, val_frac: float) -> List[Dict]:
    """Shuffle and label each clip with train/val/test. Applied per-class for stratification."""
    rng = random.Random(seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    for i, c in enumerate(shuffled):
        if i < n_train:
            c["split"] = "train"
        elif i < n_train + n_val:
            c["split"] = "val"
        else:
            c["split"] = "test"
    return shuffled


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args) -> None:
    if args.data_root:
        global DATA_ROOT
        DATA_ROOT = Path(args.data_root)

    force = args.force_reextract or cfg.force_reextract

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Data   : {DATA_ROOT}")
    print(f"Output : {OUTPUT_DIR}")

    # ── Step 1: Locate Violence/ and NonViolence/ ──────────────────────────
    fight_dir    = find_class_dir(DATA_ROOT, ["Violence",    "violence"])
    nonfight_dir = find_class_dir(DATA_ROOT, ["NonViolence", "nonviolence", "non-violence"])
    print(f"Fight dir    : {fight_dir}")
    print(f"Non-fight dir: {nonfight_dir}")

    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

    def collect_clips(folder: Path, label: int) -> List[Dict]:
        clips = []
        for vf in sorted(folder.iterdir()):
            if vf.is_file() and vf.suffix.lower() in VIDEO_EXTS:
                clips.append({"clip_id": vf.stem, "video_path": str(vf), "label": label})
        return clips

    fight_clips    = collect_clips(fight_dir,    label=1)
    nonfight_clips = collect_clips(nonfight_dir, label=0)
    print(f"\nFight clips    : {len(fight_clips)}")
    print(f"Non-fight clips: {len(nonfight_clips)}")

    # Guard: clip_id is the filename stem and NPZs are saved as {clip_id}.npz.
    # If a fight and non-fight clip share a stem, one NPZ would silently
    # overwrite the other (and the split CSV would carry two conflicting labels).
    # RLVS uses V_* / NV_* prefixes so this should never fire, but a collision
    # is a label-corrupting bug we must catch before processing, not after.
    fight_ids    = {c["clip_id"] for c in fight_clips}
    nonfight_ids = {c["clip_id"] for c in nonfight_clips}
    collisions   = fight_ids & nonfight_ids
    if collisions:
        raise RuntimeError(
            f"{len(collisions)} clip_id(s) appear in BOTH Violence and "
            f"NonViolence folders, e.g. {sorted(collisions)[:5]}. "
            "NPZ filenames would collide and corrupt labels. "
            "Rename the offending files or add a per-class prefix to clip_id."
        )

    # ── Step 2: Seeded 80/10/10 split, stratified per class ───────────────
    fight_clips    = assign_splits(fight_clips,    cfg.seed, cfg.train_frac, cfg.val_frac)
    nonfight_clips = assign_splits(nonfight_clips, cfg.seed, cfg.train_frac, cfg.val_frac)

    all_clips = fight_clips + nonfight_clips
    all_df    = pd.DataFrame(all_clips)

    print(f"\nClip inventory (total={len(all_df)}):")
    print(all_df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False))

    all_df.to_csv(str(SPLIT_CSV), index=False)
    print(f"\nSplit CSV saved -> {SPLIT_CSV}")

    # ── Step 3: Load YOLO models ───────────────────────────────────────────
    print("\nLoading detectors ...")
    pose_model = load_yolo(cfg.pose_weights, cfg.pose_fallback, "pose")
    obj_model  = load_yolo(cfg.obj_weights,  cfg.obj_fallback,  "object")
    pose_model.to(DEVICE)
    obj_model.to(DEVICE)

    # ── Step 4: Per-clip processing loop ──────────────────────────────────
    T  = cfg.T
    M  = cfg.M
    N  = cfg.N
    V  = cfg.V
    Tv = cfg.T_vit
    SZ = cfg.vit_size

    gqs_rows: List[Dict] = []
    skipped = 0
    processed_count = 0
    t_start = time.time()

    for _, row in tqdm(all_df.iterrows(), total=len(all_df),
                       desc="Preprocessing", unit="clip"):

        clip_id  = str(row["clip_id"])
        label    = int(row["label"])
        split    = str(row["split"])
        vid_path = str(row["video_path"])
        npz_path = CACHE_DIR / f"{clip_id}.npz"

        # Skip already-done clips
        if npz_path.exists() and not force:
            try:
                with np.load(str(npz_path), allow_pickle=False) as d:
                    if "frames_vit" in d:
                        g = d["gqs"]
                        gqs_rows.append({
                            "clip_id": clip_id, "split": split, "label": label,
                            "q_skel": float(g[0]), "q_int": float(g[1]),
                            "q_obj":  float(g[2]), "q_po":  float(g[3]),
                            "valid_ratio": float(g[4]),
                        })
                        continue
            except Exception:
                pass  # corrupt NPZ — reprocess

        # Open video
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  WARN: cannot open {vid_path}")
            skipped += 1
            continue

        # Read all frames into RAM (RLVS clips are short — fits in 64 GB easily)
        raw_frames_all: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames_all.append(frame)
        cap.release()

        n_frames = len(raw_frames_all)
        if n_frames < 2:
            print(f"  WARN: {clip_id} has only {n_frames} frame(s), skipping.")
            skipped += 1
            continue

        # Uniform frame index sets
        graph_idx = np.linspace(0, n_frames - 1, T, dtype=int)
        vit_idx   = np.linspace(0, n_frames - 1, Tv, dtype=int)

        needed = set(graph_idx.tolist()) | set(vit_idx.tolist())
        raw_frames: Dict[int, np.ndarray] = {i: raw_frames_all[i] for i in needed}
        del raw_frames_all  # free before allocating NPZ arrays

        # Allocate output arrays
        skeleton      = np.zeros((T, M, V, 3),   dtype=np.float32)
        int_nodes     = np.zeros((T, M, 7),       dtype=np.float32)
        int_edges     = np.zeros((T, M, M, 4),    dtype=np.float32)
        int_node_mask = np.zeros((T, M),           dtype=bool)
        int_edge_mask = np.zeros((T, M, M),        dtype=bool)
        obj_nodes     = np.zeros((T, N, 6),        dtype=np.float32)
        obj_node_mask = np.zeros((T, N),           dtype=bool)
        po_edges      = np.zeros((T, M, N, 5),     dtype=np.float32)
        po_edge_mask  = np.zeros((T, M, N),        dtype=bool)
        frames_vit    = np.zeros((Tv, SZ, SZ, 3),  dtype=np.uint8)

        prev_centers: Dict[int, np.ndarray] = {}
        track_ages:   Dict[int, int]        = {}

        # Reset ByteTrack state between clips
        try:
            for tracker in pose_model.predictor.trackers:
                tracker.reset()
        except Exception:
            pass

        # Sequential tracking pass over graph frames
        graph_set    = set(graph_idx.tolist())
        frame_buffer: Dict[int, Dict] = {}

        for fi in sorted(raw_frames.keys()):
            if fi not in graph_set:
                continue

            frame_bgr = raw_frames[fi]
            h, w      = frame_bgr.shape[:2]
            denom     = float(max(h, w))

            persons = []
            try:
                res_pose = pose_model.track(
                    frame_bgr, conf=cfg.pose_conf, persist=True,
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
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                raise
            except Exception:
                pass

            objects = []
            try:
                res_obj = obj_model.predict(
                    frame_bgr, conf=cfg.obj_conf,
                    iou=cfg.obj_iou, verbose=False,
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
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                raise
            except Exception:
                pass

            frame_buffer[fi] = {"persons": persons, "objects": objects}

        # Fill structured arrays
        for ti, fi in enumerate(graph_idx.tolist()):
            fi = int(fi)
            fb = frame_buffer.get(fi)
            if fb is None:
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

            for oi, obj in enumerate(objects):
                obj_nodes[ti, oi] = [obj["cx"], obj["cy"],
                                     obj["w"],  obj["h"],
                                     obj["conf"], obj["cls"]]
                obj_node_mask[ti, oi] = True

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

        # VideoMAE frames
        for vi, fi in enumerate(vit_idx.tolist()):
            fi = int(fi)
            if fi in raw_frames:
                bgr = raw_frames[fi]
                rgb = cv2.cvtColor(
                    cv2.resize(bgr, (SZ, SZ), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_BGR2RGB,
                )
                frames_vit[vi] = rgb
            elif vi > 0:
                frames_vit[vi] = frames_vit[vi - 1]

        # GQS — q_skel denominator is T (frames), not T×M
        valid_skel = sum(
            1 for ti in range(T)
            if any(
                int_node_mask[ti, pi]
                and int((skeleton[ti, pi, :, 2] > 0.25).sum()) >= cfg.min_joints
                for pi in range(M)
            )
        )
        q_skel      = valid_skel / T
        q_int       = sum(int(int_node_mask[ti].sum() >= 2) for ti in range(T)) / T
        q_obj       = sum(int(obj_node_mask[ti].sum() >= 1) for ti in range(T)) / T
        q_po        = sum(int(po_edge_mask[ti].any())       for ti in range(T)) / T
        valid_ratio = sum(int(int_node_mask[ti].any())      for ti in range(T)) / T
        gqs = np.array([q_skel, q_int, q_obj, q_po, valid_ratio], dtype=np.float32)

        # Save NPZ
        np.savez_compressed(
            str(npz_path),
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
            label         = np.array(label, dtype=np.int64),
            split         = np.array(split),
        )

        gqs_rows.append({
            "clip_id": clip_id, "split": split, "label": label,
            "q_skel": float(q_skel), "q_int": float(q_int),
            "q_obj":  float(q_obj),  "q_po":  float(q_po),
            "valid_ratio": float(valid_ratio),
        })

        processed_count += 1
        if processed_count % 50 == 0:
            elapsed = time.time() - t_start
            rate    = processed_count / elapsed
            remaining = (len(all_df) - processed_count - skipped) / max(rate, 1e-6)
            print(f"  [progress] {processed_count} done, {skipped} skipped "
                  f"| {rate:.1f} clips/s | ETA {remaining/60:.0f} min")

    # ── Step 5: GQS summary CSV ────────────────────────────────────────────
    gqs_df = pd.DataFrame(gqs_rows)
    gqs_df.to_csv(str(GQS_CSV), index=False)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete.")
    print(f"  Processed : {processed_count}")
    print(f"  Skipped   : {skipped}")
    print(f"  NPZ files : {len(list(CACHE_DIR.glob('*.npz')))}")
    print(f"  Split CSV : {SPLIT_CSV}")
    print(f"  GQS CSV   : {GQS_CSV}")

    cols = ["q_skel", "q_int", "q_obj", "q_po", "valid_ratio"]
    print("\nGQS means by split:")
    print(gqs_df.groupby("split")[cols].mean().round(3).to_string())
    print(f"\nOverall GQS means:")
    for c in cols:
        print(f"  {c:<15}: {gqs_df[c].mean():.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLVS V9 Preprocessing")
    parser.add_argument("--data-root",       type=str,  default=None,
                        help="Path to RLVS dataset folder containing Violence/ and NonViolence/")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Reprocess all clips, even if NPZ already exists")
    args = parser.parse_args()
    main(args)
