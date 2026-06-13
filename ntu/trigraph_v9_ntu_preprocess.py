"""
trigraph_v9_ntu_preprocess.py
Guardian Eye - V9  |  Phase 1: Preprocessing
Dataset : NTU CCTV-Fights (Perez et al., temporal-localization fight dataset)
Hardware: Local workstation - RTX 3090 (24 GB VRAM), 64 GB RAM

What this script does
---------------------
NTU CCTV-Fights is a TEMPORAL LOCALIZATION dataset, NOT clip-level binary:
  - 1,000 fight videos, ALL annotated as containing >=1 fight (no normal videos).
  - groundtruth.json marks each fight as a [start_sec, end_sec] segment.
  - Official train/val/test split is in the per-video "subset" field.

We convert it to clip-level binary classification by SLIDING-WINDOW windowing,
identical downstream to RLVS once each window is treated as one "clip":

1. Parse groundtruth.json: for each video read duration, frame_rate, source,
   subset, and the list of fight segments.
2. Slide a fixed WINDOW_SEC window over each video (length scaled to the
   per-video fps so the temporal span is always WINDOW_SEC):
     - POSITIVE window (label=1): overlaps a fight segment by >= POS_OVERLAP.
       Positive stride = POS_STRIDE_SEC (overlapping; fights are scarce).
     - NEGATIVE window (label=0): 0%% overlap with ANY fight segment AND
       >= GUARD_SEC away from every fight boundary.
       Negative stride = NEG_STRIDE_SEC (non-overlapping; negatives abundant).
     - Ambiguous windows (0 < overlap < POS_OVERLAP, or inside the guard gap)
       are discarded.
3. Per-video cap: at most MAX_POS_PER_VIDEO positives and MAX_NEG_PER_VIDEO
   negatives, so one long video (NTU has a 703 s video) cannot dominate.
4. Per split, downsample negatives to match the positive count (1:1 balance).
5. For every surviving window, run the IDENTICAL V9 per-clip extraction as RLVS:
   a. Uniformly sample T=32 frames across the WINDOW (not the whole video).
   b. YOLO11x-pose + ByteTrack for identity-stable tracks.
   c. YOLO11x object detector on the same frames.
   d. Build V9 graph arrays + GQS.
   e. Capture T_vit=16 frames at 224x224 uint8 RGB for VideoMAE.
6. Save one NPZ per window to OUTPUT_DIR/cache_v9/{clip_id}.npz.
7. Save split_ntu.csv and gqs_summary_ntu.csv to OUTPUT_DIR/.

clip_id scheme (locked):
  clip_id = "{video_id}_w{idx:02d}"  e.g. fight_0001_w03
  NPZ filename = {clip_id}.npz
  split CSV columns: clip_id, video_id, source, start_frame, end_frame,
                     label, split

NPZ schema (V9 - identical to RWF / RLVS / UBI for model compatibility)
-----------------------------------------------------------------------
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
  frames_vit    [T_vit, H, W, 3] uint8
  label         scalar           int64     1=fight, 0=non-fight
  split         scalar           str       train / val / test
  source        scalar           str       CCTV / Mobile / Other / Car

Constants: T=32, T_vit=16, M=6, N=8, V=17

Usage
-----
  python trigraph_v9_ntu_preprocess.py
  python trigraph_v9_ntu_preprocess.py --force-reextract
  python trigraph_v9_ntu_preprocess.py --data-root "C:\\path\\to\\NTU-CCTV fights"
  python trigraph_v9_ntu_preprocess.py --plan-only   # window the videos, write the
                                                     # split CSV, then stop (no YOLO)
"""

import argparse
import json
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


# -- Paths ----------------------------------------------------------------------
# DATA_ROOT : folder that contains the fight_XXXX.mpeg videos AND groundtruth.json.
# OUTPUT_DIR: where NPZs, CSVs, and checkpoints are written.
DATA_ROOT  = Path(r"C:\Violence detection\datasets\NTU-CCTV fights")
OUTPUT_DIR = Path(r"C:\Violence detection\ashour\fresh\ntu\preproc_output")

GROUNDTRUTH = DATA_ROOT / "groundtruth.json"
CACHE_DIR   = OUTPUT_DIR / "cache_v9"
SPLIT_CSV   = OUTPUT_DIR / "split_ntu.csv"
GQS_CSV     = OUTPUT_DIR / "gqs_summary_ntu.csv"

# Video container extensions to search for, in case some clips are not .mpeg.
VIDEO_EXTS = (".mpeg", ".mpg", ".mp4", ".avi", ".mkv", ".mov", ".webm")


# -- Configuration --------------------------------------------------------------
@dataclass
class CFG:
    seed: int = 42

    # -- Windowing (LOCKED 2026-06-07) -----------------------------------------
    window_sec:     float = 5.0    # temporal span of every clip
    pos_stride_sec: float = 2.5    # 50%% overlap for positives (fights scarce)
    neg_stride_sec: float = 5.0    # no overlap for negatives (negatives abundant)
    pos_overlap:    float = 0.50   # window is positive if >=50%% inside a fight seg
    guard_sec:      float = 2.0    # negatives must be >=2 s from any fight boundary
    max_pos_per_video: int = 6     # per-video clip cap (flattens the 703 s video)
    max_neg_per_video: int = 6
    default_fps:    float = 30.0   # fallback if a video's frame_rate is missing/0

    # Split id column name in groundtruth.json subset field -> our split name
    subset_map: Tuple[Tuple[str, str], ...] = (
        ("training",   "train"),
        ("validation", "val"),
        ("testing",    "test"),
    )

    # -- Graph constants (locked - must match train.py and videomae.py) --------
    T:          int = 32
    M:          int = 6
    N:          int = 8
    V:          int = 17

    # -- Detection thresholds --------------------------------------------------
    pose_conf:  float = 0.25
    obj_conf:   float = 0.25
    obj_iou:    float = 0.45
    min_joints: int   = 5

    # -- VideoMAE frame extraction ---------------------------------------------
    T_vit:    int = 16
    vit_size: int = 224

    # -- YOLO weights (downloaded automatically by ultralytics on first run) ---
    pose_weights:  str            = "yolo11x-pose.pt"
    obj_weights:   str            = "yolo11x.pt"
    pose_fallback: Tuple[str,...] = ("yolo11l-pose.pt", "yolo11m-pose.pt",
                                     "yolo11s-pose.pt")
    obj_fallback:  Tuple[str,...] = ("yolo11l.pt", "yolo11m.pt", "yolo11s.pt")

    force_reextract: bool = False


cfg = CFG()


# -- Helpers --------------------------------------------------------------------
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


def find_video_file(data_root: Path, video_id: str) -> Path:
    """Locate the video file for a groundtruth key (e.g. 'fight_0001').
    Files are flat in data_root, named fight_XXXX.mpeg. Try each known
    extension; return the first that exists."""
    for ext in VIDEO_EXTS:
        cand = data_root / f"{video_id}{ext}"
        if cand.is_file():
            return cand
    return None  # caller decides how to handle a missing file


def _overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length (seconds) of the intersection of intervals [a0,a1] and [b0,b1]."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_video(video_id: str, meta: Dict, split: str) -> List[Dict]:
    """
    Convert one groundtruth video entry into a list of labelled windows.

    meta keys used: duration (sec), frame_rate (fps), source (str),
                    annotations (list of {segment:[s,e], label:"Fight"}).

    Returns a list of dicts (already capped + ready for balancing):
      {clip_id, video_id, source, start_frame, end_frame, label, split}
    start_frame/end_frame are integer frame indices INTO THE SOURCE VIDEO.
    """
    duration = float(meta.get("duration", 0.0))
    fps      = float(meta.get("frame_rate", 0.0)) or cfg.default_fps
    source   = str(meta.get("source", "Unknown"))
    segments = [(float(a["segment"][0]), float(a["segment"][1]))
                for a in meta.get("annotations", [])
                if a.get("label", "").lower() == "fight"]

    win = cfg.window_sec
    if duration < win:
        # Video shorter than one window: cannot form a clean clip. Skip.
        return []

    # -- Candidate positives (overlapping stride) -----------------------------
    pos_cands: List[Tuple[float, float]] = []
    t = 0.0
    while t + win <= duration + 1e-6:
        w0, w1 = t, t + win
        # fraction of the WINDOW that lies inside ANY single fight segment
        best_frac = 0.0
        for s0, s1 in segments:
            ov = _overlap_sec(w0, w1, s0, s1)
            best_frac = max(best_frac, ov / win)
        if best_frac >= cfg.pos_overlap:
            pos_cands.append((w0, w1))
        t += cfg.pos_stride_sec

    # -- Candidate negatives (non-overlapping stride, guard gap) --------------
    neg_cands: List[Tuple[float, float]] = []
    t = 0.0
    while t + win <= duration + 1e-6:
        w0, w1 = t, t + win
        # A negative must have ZERO overlap with any fight, AND keep a guard gap:
        # the window (expanded by guard on both sides) must not touch any segment.
        g0, g1 = w0 - cfg.guard_sec, w1 + cfg.guard_sec
        touches = any(_overlap_sec(g0, g1, s0, s1) > 0.0 for s0, s1 in segments)
        if not touches:
            neg_cands.append((w0, w1))
        t += cfg.neg_stride_sec

    # -- Per-video cap (deterministic: evenly spaced subsample, not random) ----
    def cap(cands: List[Tuple[float, float]], k: int) -> List[Tuple[float, float]]:
        if len(cands) <= k:
            return cands
        idx = np.linspace(0, len(cands) - 1, k, dtype=int)
        return [cands[i] for i in idx]

    pos_sel = cap(pos_cands, cfg.max_pos_per_video)
    neg_sel = cap(neg_cands, cfg.max_neg_per_video)

    # -- Assemble window records ----------------------------------------------
    records: List[Dict] = []
    widx = 0
    for (w0, w1), label in ([(w, 1) for w in pos_sel] + [(w, 0) for w in neg_sel]):
        start_frame = int(round(w0 * fps))
        end_frame   = int(round(w1 * fps))
        records.append({
            "clip_id":     f"{video_id}_w{widx:02d}",
            "video_id":    video_id,
            "source":      source,
            "start_frame": start_frame,
            "end_frame":   end_frame,
            "label":       label,
            "split":       split,
        })
        widx += 1
    return records


def balance_per_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample negatives to match positives WITHIN each split (1:1)."""
    rng = np.random.RandomState(seed)
    kept = []
    for split_name, grp in df.groupby("split"):
        pos = grp[grp["label"] == 1]
        neg = grp[grp["label"] == 0]
        n_keep = min(len(pos), len(neg))
        if n_keep == 0:
            # Degenerate split (no positives or no negatives) - keep as-is and warn.
            print(f"  WARN: split '{split_name}' has pos={len(pos)} neg={len(neg)}; "
                  "cannot balance, keeping all.")
            kept.append(grp)
            continue
        pos_k = pos.sample(n=n_keep, random_state=rng) if len(pos) > n_keep else pos
        neg_k = neg.sample(n=n_keep, random_state=rng) if len(neg) > n_keep else neg
        kept.append(pd.concat([pos_k, neg_k]))
    out = pd.concat(kept).reset_index(drop=True)
    return out


# -- Main -----------------------------------------------------------------------
def main(args) -> None:
    if args.data_root:
        global DATA_ROOT, GROUNDTRUTH
        DATA_ROOT   = Path(args.data_root)
        GROUNDTRUTH = DATA_ROOT / "groundtruth.json"

    force = args.force_reextract or cfg.force_reextract

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Data   : {DATA_ROOT}")
    print(f"GT     : {GROUNDTRUTH}")
    print(f"Output : {OUTPUT_DIR}")

    # -- Step 1: Parse groundtruth.json and window every video ----------------
    if not GROUNDTRUTH.exists():
        raise RuntimeError(
            f"groundtruth.json not found at {GROUNDTRUTH}. "
            f"Check --data-root (currently {DATA_ROOT})."
        )
    with open(str(GROUNDTRUTH), "r") as f:
        gt = json.load(f)
    database = gt["database"]
    print(f"\nVideos in groundtruth: {len(database)}")

    subset_to_split = dict(cfg.subset_map)

    all_records: List[Dict] = []
    missing_files = 0
    skipped_short = 0
    video_file_map: Dict[str, str] = {}  # video_id -> resolved file path

    for video_id, meta in tqdm(database.items(), total=len(database),
                               desc="Windowing", unit="video"):
        split = subset_to_split.get(str(meta.get("subset", "")).lower())
        if split is None:
            # Unknown subset label - skip rather than silently mislabel.
            continue

        vf = find_video_file(DATA_ROOT, video_id)
        if vf is None:
            missing_files += 1
            continue
        video_file_map[video_id] = str(vf)

        recs = window_video(video_id, meta, split)
        if not recs:
            skipped_short += 1
        all_records.extend(recs)

    if missing_files:
        print(f"  WARN: {missing_files} videos in groundtruth had no matching "
              f"video file in {DATA_ROOT} (skipped).")
    if skipped_short:
        print(f"  Note: {skipped_short} videos produced no windows "
              "(shorter than one window, or no clean pos/neg).")

    raw_df = pd.DataFrame(all_records)
    if len(raw_df) == 0:
        raise RuntimeError("No windows produced. Check groundtruth + video paths.")

    print(f"\nRaw windows (pre-balance, total={len(raw_df)}):")
    print(raw_df.groupby(["split", "label"]).size()
          .rename("count").reset_index().to_string(index=False))

    # -- Step 2: Balance negatives to positives per split ---------------------
    bal_df = balance_per_split(raw_df, cfg.seed)
    # Attach the resolved source video path for the extraction loop.
    bal_df["video_path"] = bal_df["video_id"].map(video_file_map)

    print(f"\nBalanced windows (total={len(bal_df)}):")
    print(bal_df.groupby(["split", "label"]).size()
          .rename("count").reset_index().to_string(index=False))
    print("\nPer-source window counts:")
    print(bal_df.groupby(["source", "label"]).size()
          .rename("count").reset_index().to_string(index=False))

    bal_df.to_csv(str(SPLIT_CSV), index=False)
    print(f"\nSplit CSV saved -> {SPLIT_CSV}")

    if args.plan_only:
        print("\n--plan-only: stopping before YOLO extraction.")
        return

    # -- Step 3: Load YOLO models ---------------------------------------------
    print("\nLoading detectors ...")
    pose_model = load_yolo(cfg.pose_weights, cfg.pose_fallback, "pose")
    obj_model  = load_yolo(cfg.obj_weights,  cfg.obj_fallback,  "object")
    pose_model.to(DEVICE)
    obj_model.to(DEVICE)

    # -- Step 4: Per-window processing loop -----------------------------------
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

    # Group rows by source video so we open + decode each video only ONCE,
    # then slice the needed frame ranges per window. NTU videos are long
    # (mean 64 s) so re-opening per window would be very wasteful.
    for video_id, vid_rows in tqdm(bal_df.groupby("video_id"),
                                   total=bal_df["video_id"].nunique(),
                                   desc="Preprocessing", unit="video"):
        vid_path = str(vid_rows.iloc[0]["video_path"])

        # Determine which windows of this video still need processing.
        pending = []
        for _, row in vid_rows.iterrows():
            npz_path = CACHE_DIR / f"{row['clip_id']}.npz"
            if npz_path.exists() and not force:
                try:
                    with np.load(str(npz_path), allow_pickle=False) as d:
                        if "frames_vit" in d:
                            g = d["gqs"]
                            gqs_rows.append({
                                "clip_id": str(row["clip_id"]),
                                "video_id": video_id,
                                "source": str(row["source"]),
                                "split": str(row["split"]),
                                "label": int(row["label"]),
                                "q_skel": float(g[0]), "q_int": float(g[1]),
                                "q_obj":  float(g[2]), "q_po":  float(g[3]),
                                "valid_ratio": float(g[4]),
                            })
                            continue
                except Exception:
                    pass  # corrupt NPZ - reprocess
            pending.append(row)

        if not pending:
            continue

        # Open video once.
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  WARN: cannot open {vid_path}")
            skipped += len(pending)
            continue

        # We need the union of frame indices across all pending windows. Rather
        # than decode the entire (possibly 700 s) video, compute per-window the
        # T graph indices and T_vit vit indices, take their union, then do a
        # single sequential decode pass keeping only the needed frames.
        per_window_idx: Dict[str, Dict] = {}
        needed_global = set()
        for row in pending:
            sf = int(row["start_frame"])
            ef = int(row["end_frame"])
            ef = max(ef, sf + 1)  # guard against zero-length window
            graph_idx = np.linspace(sf, ef - 1, T,  dtype=int)
            vit_idx   = np.linspace(sf, ef - 1, Tv, dtype=int)
            per_window_idx[str(row["clip_id"])] = {
                "row": row, "graph_idx": graph_idx, "vit_idx": vit_idx,
            }
            needed_global.update(graph_idx.tolist())
            needed_global.update(vit_idx.tolist())

        max_needed = max(needed_global)
        # Sequential decode, capturing only needed frames up to max_needed.
        frame_store: Dict[int, np.ndarray] = {}
        fpos = 0
        while fpos <= max_needed:
            ret, frame = cap.read()
            if not ret:
                break
            if fpos in needed_global:
                frame_store[fpos] = frame
            fpos += 1
        cap.release()

        # -- Process each pending window for this video -----------------------
        for clip_id, info in per_window_idx.items():
            row       = info["row"]
            graph_idx = info["graph_idx"]
            vit_idx   = info["vit_idx"]
            label     = int(row["label"])
            split     = str(row["split"])
            source    = str(row["source"])
            npz_path  = CACHE_DIR / f"{clip_id}.npz"

            # Frames available for this window (some may be missing if the video
            # ended early or a decode hiccup dropped them).
            avail = [fi for fi in set(graph_idx.tolist()) | set(vit_idx.tolist())
                     if fi in frame_store]
            if len(avail) < 2:
                print(f"  WARN: {clip_id} has <2 decoded frames, skipping.")
                skipped += 1
                continue

            raw_frames = {fi: frame_store[fi] for fi in avail}

            # Allocate output arrays (identical schema to RLVS)
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

            # Reset ByteTrack state between windows
            try:
                for tracker in pose_model.predictor.trackers:
                    tracker.reset()
            except Exception:
                pass

            graph_set    = set(graph_idx.tolist())
            frame_buffer: Dict[int, Dict] = {}

            # Sequential tracking pass over graph frames (in ascending order)
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

            # Fill structured arrays (graph_idx may not be contiguous frame ids,
            # so map by position ti -> the actual frame index fi).
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

            # GQS - q_skel denominator is T (frames), not T x M
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

            # Save NPZ (schema identical to RLVS + a `source` scalar for NTU)
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
                source        = np.array(source),
            )

            gqs_rows.append({
                "clip_id": clip_id, "video_id": video_id, "source": source,
                "split": split, "label": label,
                "q_skel": float(q_skel), "q_int": float(q_int),
                "q_obj":  float(q_obj),  "q_po":  float(q_po),
                "valid_ratio": float(valid_ratio),
            })

            processed_count += 1
            if processed_count % 50 == 0:
                elapsed = time.time() - t_start
                rate    = processed_count / elapsed
                print(f"  [progress] {processed_count} clips done, {skipped} skipped "
                      f"| {rate:.1f} clips/s")

    # -- Step 5: GQS summary CSV ----------------------------------------------
    gqs_df = pd.DataFrame(gqs_rows)
    gqs_df.to_csv(str(GQS_CSV), index=False)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete.")
    print(f"  Processed : {processed_count}")
    print(f"  Skipped   : {skipped}")
    print(f"  NPZ files : {len(list(CACHE_DIR.glob('*.npz')))}")
    print(f"  Split CSV : {SPLIT_CSV}")
    print(f"  GQS CSV   : {GQS_CSV}")

    if len(gqs_df):
        cols = ["q_skel", "q_int", "q_obj", "q_po", "valid_ratio"]
        print("\nGQS means by split:")
        print(gqs_df.groupby("split")[cols].mean().round(3).to_string())
        print("\nGQS means by source (expect Mobile q_skel < CCTV):")
        print(gqs_df.groupby("source")[cols].mean().round(3).to_string())
        print(f"\nOverall GQS means:")
        for c in cols:
            print(f"  {c:<15}: {gqs_df[c].mean():.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTU CCTV-Fights V9 Preprocessing")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to NTU folder containing fight_XXXX.mpeg + groundtruth.json")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Reprocess all windows, even if NPZ already exists")
    parser.add_argument("--plan-only", action="store_true",
                        help="Window the videos and write the split CSV, then stop "
                             "(no YOLO, no NPZ) - lets you inspect the clip plan first")
    args = parser.parse_args()
    main(args)
