"""
trigraph_v9_hf_preprocess.py
Guardian Eye — V9  |  Phase 1: Preprocessing
Dataset : Hockey Fights (Bermejo Nievas et al., CAIP 2011)
Hardware: Local workstation — RTX 3090 (24 GB VRAM), 64 GB RAM

What this script does
─────────────────────
1. Scans the Hockey Fights dataset from a local directory:
     <DATA_ROOT>/fights/    — 500 fight clips   (label=1)
     <DATA_ROOT>/nofights/  — 500 non-fight clips (label=0)
2. Applies a seeded 70/10/20 train/val/test split (seed=42), stratified
   per class independently.
3. For every clip:
   a. Uniformly samples T=32 frames across the full clip duration.
   b. Runs YOLO11x-pose + ByteTrack (persist=True) for identity-stable tracks.
   c. Runs YOLO11x object detector on the same frames.
   d. Builds V9-compatible graph arrays + GQS.
   e. Captures T_vit=16 uniformly-sampled frames at 224x224 uint8 RGB for VideoMAE.
4. Saves one NPZ per clip to OUTPUT_DIR/cache_v9/.
5. Saves split_hf.csv and gqs_summary_hf.csv to OUTPUT_DIR/.

CRITICAL label bug
──────────────────
The folder name "nofights" contains the substring "fights".  Any check that
tests for "fights" first will silently assign label=1 to non-fight clips.
All label resolution in this script tests for "nofights" BEFORE "fights".

NPZ schema (V9 — identical to RLVS/RWF/NTU V9)
────────────────────────────────────────────────
  skeleton      [T, M, V, 3]     float32
  int_nodes     [T, M, 7]        float32
  int_edges     [T, M, M, 4]     float32
  int_node_mask [T, M]           bool
  int_edge_mask [T, M, M]        bool
  obj_nodes     [T, N, 6]        float32
  obj_node_mask [T, N]           bool
  po_edges      [T, M, N, 5]     float32
  po_edge_mask  [T, M, N]        bool
  gqs           [5]              float32   q_skel,q_int,q_obj,q_po,valid_ratio
  frames_vit    [T_vit, H, W, 3] uint8     224x224 RGB for VideoMAE
  label         scalar           int64     1=fight, 0=non-fight
  split         scalar           str       train / val / test

Constants: T=32, T_vit=16, M=6, N=8, V=17

Citation
────────
Bermejo Nievas et al., "Violence Detection in Video using Computer Vision
Techniques", CAIP 2011.

Usage
─────
  python trigraph_v9_hf_preprocess.py
  python trigraph_v9_hf_preprocess.py --force-reextract
  python trigraph_v9_hf_preprocess.py --data-root /path/to/HockeyFights
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
# WS layout (confirmed against V8 script + GQS CSV):
#   DATA_ROOT/fights/    — fight clips    (e.g. fi410_xvid.avi),  label=1
#   DATA_ROOT/nofights/  — non-fight clips (e.g. no476_xvid.avi), label=0
DATA_ROOT  = Path(r"C:\Violence detection\datasets\Hockey fight\data")
OUTPUT_DIR = Path(r"C:\Violence detection\ashour\fresh\HF\preproc_output")

CACHE_DIR = OUTPUT_DIR / "cache_v9"
SPLIT_CSV = OUTPUT_DIR / "split_hf.csv"
GQS_CSV   = OUTPUT_DIR / "gqs_summary_hf.csv"


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    seed: int = 42

    # Split: 70/10/20 — balanced dataset, no official split
    train_frac: float = 0.70
    val_frac:   float = 0.10
    # test_frac = 0.20 (implicit)

    # Graph constants (locked — must match train.py and videomae.py)
    T:  int = 32
    M:  int = 6
    N:  int = 8
    V:  int = 17

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


def assign_splits(clips: List[Dict], seed: int,
                  train_frac: float, val_frac: float) -> List[Dict]:
    """Shuffle and assign train/val/test split labels. Applied per-class for stratification."""
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

    # ── Step 1: Locate fights/ and nofights/ ──────────────────────────────
    # CRITICAL: resolve nofights BEFORE fights.  The folder name "nofights"
    # contains "fights" as a substring; reversing the order would assign
    # label=1 to the non-fight class.
    nofight_dir = DATA_ROOT / "nofights"
    fight_dir   = DATA_ROOT / "fights"

    # Case-insensitive fallback scan
    if not nofight_dir.exists() or not fight_dir.exists():
        found_dirs = {p.name.lower(): p for p in DATA_ROOT.iterdir() if p.is_dir()}
        if "nofights" not in found_dirs:
            raise RuntimeError(
                f"Could not find 'nofights' directory under {DATA_ROOT}.\n"
                f"Found: {list(found_dirs.keys())}\n"
                "Expected layout: DATA_ROOT/fights/ and DATA_ROOT/nofights/"
            )
        if "fights" not in found_dirs:
            raise RuntimeError(
                f"Could not find 'fights' directory under {DATA_ROOT}.\n"
                f"Found: {list(found_dirs.keys())}"
            )
        # Use the case-insensitive matches
        nofight_dir = found_dirs["nofights"]
        fight_dir   = found_dirs["fights"]

    print(f"Fight dir     : {fight_dir}")
    print(f"Non-fight dir : {nofight_dir}")

    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

    def collect_clips(folder: Path, label: int, prefix: str) -> List[Dict]:
        clips = []
        for vf in sorted(folder.iterdir()):
            if vf.is_file() and vf.suffix.lower() in VIDEO_EXTS:
                # Prefix clip_id with the class-folder name to (a) guarantee
                # global uniqueness across the two folders and (b) match the V8
                # video_id convention "{class}_{stem}" (e.g. fights_fi410_xvid,
                # nofights_no476_xvid) so V9 clip_ids cross-reference cleanly
                # with the V8 CSVs. The raw filenames already carry unique
                # no*/fi* stems (verified: zero stem collisions), so this prefix
                # is belt-and-suspenders, not strictly required for uniqueness.
                clips.append({
                    "clip_id":    f"{prefix}_{vf.stem}",
                    "video_path": str(vf),
                    "label":      label,
                })
        return clips

    # Collect non-fights FIRST, then fights — safe ordering that avoids any
    # accidental "fights" substring match on the nofights folder. Prefixes match
    # the V8 class-folder names exactly (nofights_ / fights_).
    nofight_clips = collect_clips(nofight_dir, label=0, prefix="nofights")
    fight_clips   = collect_clips(fight_dir,   label=1, prefix="fights")
    print(f"\nFight clips    : {len(fight_clips)}")
    print(f"Non-fight clips: {len(nofight_clips)}")

    # Collision guard: clip_ids must be globally unique to prevent NPZ overwrites
    fight_ids    = {c["clip_id"] for c in fight_clips}
    nofight_ids  = {c["clip_id"] for c in nofight_clips}
    collisions   = fight_ids & nofight_ids
    if collisions:
        raise RuntimeError(
            f"{len(collisions)} clip_id(s) appear in BOTH classes after prefixing "
            f"(e.g. {sorted(collisions)[:5]}). This should not happen — check "
            "the prefix logic."
        )

    # ── Step 2: Seeded 70/10/20 split, stratified per class ───────────────
    fight_clips   = assign_splits(fight_clips,   cfg.seed, cfg.train_frac, cfg.val_frac)
    nofight_clips = assign_splits(nofight_clips, cfg.seed, cfg.train_frac, cfg.val_frac)

    all_clips = fight_clips + nofight_clips
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

        # Skip already-done clips (resume-safe)
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

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  WARN: cannot open {vid_path}")
            skipped += 1
            continue

        # Read all frames into RAM (Hockey Fights clips are very short, 2-4s)
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

        needed     = set(graph_idx.tolist()) | set(vit_idx.tolist())
        raw_frames: Dict[int, np.ndarray] = {i: raw_frames_all[i] for i in needed}
        del raw_frames_all  # free RAM before allocating NPZ arrays

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

        # Reset ByteTrack state between clips so identity IDs do not leak
        try:
            for tracker in pose_model.predictor.trackers:
                tracker.reset()
        except Exception:
            pass

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
                # Propagate last valid frame (avoids zero gaps in short clips)
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

        # VideoMAE frames (224x224 uint8 RGB)
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
            elapsed   = time.time() - t_start
            rate      = processed_count / elapsed
            remaining = (len(all_df) - processed_count - skipped) / max(rate, 1e-6)
            print(f"  [progress] {processed_count} done, {skipped} skipped "
                  f"| {rate:.1f} clips/s | ETA {remaining/60:.0f} min")

    # ── Step 5: GQS summary CSV ────────────────────────────────────────────
    gqs_df = pd.DataFrame(gqs_rows)
    gqs_df.to_csv(str(GQS_CSV), index=False)

    print(f"\n{'='*60}")
    print("Preprocessing complete.")
    print(f"  Processed : {processed_count}")
    print(f"  Skipped   : {skipped}")
    print(f"  NPZ files : {len(list(CACHE_DIR.glob('*.npz')))}")
    print(f"  Split CSV : {SPLIT_CSV}")
    print(f"  GQS CSV   : {GQS_CSV}")

    cols = ["q_skel", "q_int", "q_obj", "q_po", "valid_ratio"]
    print("\nGQS means by split:")
    print(gqs_df.groupby("split")[cols].mean().round(3).to_string())
    print("\nOverall GQS means:")
    for c in cols:
        print(f"  {c:<15}: {gqs_df[c].mean():.3f}")
    # Note: PO stream (q_po) expected to be near 0.111 on Hockey Fights
    # (object GQS mean from V8.1 was ~0.111 — near-empty signal on most clips).
    # Watch for this in diagnostics; PO stream added FPs in V8.1.
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hockey Fights V9 Preprocessing")
    parser.add_argument("--data-root",       type=str, default=None,
                        help="Path to Hockey Fights dataset folder "
                             "(containing fights/ and nofights/)")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Reprocess all clips, even if NPZ already exists")
    args = parser.parse_args()
    main(args)
