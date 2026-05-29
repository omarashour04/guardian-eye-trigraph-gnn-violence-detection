"""
trigraph_v9_ubi_preprocess.py
Guardian Eye — V9  |  Phase 1: Preprocessing
Dataset : UBI-Fights  (intissarziani/ubi-fightsall)
GPU     : L4  |  ETA ~4 hours for ~1,000 source videos

What this script does
─────────────────────
1. Downloads UBI-Fights from Kaggle into the raw Modal volume.
2. Reads per-frame annotation files to locate fight intervals.
3. Approach B clip extraction:
   - Positive clips: 32-frame windows centred on annotated fight intervals.
   - Negative clips: 32-frame windows sampled from explicitly non-fight intervals.
4. For every clip:
   a. Runs YOLO11x-pose + ByteTrack (persist=True) to extract identity-stable tracks.
   b. Runs YOLO11x object detector on the same frames.
   c. Builds V9-compatible graph arrays (skeleton, interaction, object, PO edges, masks) + GQS.
   d. Captures T_vit=16 uniformly-sampled frames at 224×224 uint8 RGB for VideoMAE.
5. Saves one NPZ per clip to CACHE_DIR.
6. Saves split_ubi.csv and gqs_summary_ubi.csv to PROC_MOUNT.

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
  frames_vit    [T_vit, H, W, 3] uint8     224×224 RGB for VideoMAE
  label         scalar           int64     1=fight, 0=non-fight
  split         scalar           str       train / test

Constants: T=32, T_vit=16, M=6, N=8, V=17

Usage
─────
  modal run trigraph_v9_ubi_preprocess.py::preprocess
  modal run trigraph_v9_ubi_preprocess.py::preprocess --force-reextract
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import random
import zipfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import modal

# ── Modal primitives ──────────────────────────────────────────────────────────
APP_NAME      = "guardian-eye-v9-ubi-preprocess"
VOL_NAME_RAW  = "ubi-fights-raw"
VOL_NAME_PROC = "ubi-fights-processed"
SECRET_NAME   = "KAGGLE_TOKEN"

app      = modal.App(APP_NAME)
vol_raw  = modal.Volume.from_name(VOL_NAME_RAW,  create_if_missing=True)
vol_proc = modal.Volume.from_name(VOL_NAME_PROC, create_if_missing=True)

RAW_MOUNT  = Path("/data/raw")
PROC_MOUNT = Path("/data/proc")

RAW_DIR   = RAW_MOUNT  / "videos"        # extracted Kaggle dataset root
CACHE_DIR = PROC_MOUNT / "cache_v9"      # one NPZ per clip
SPLIT_CSV = PROC_MOUNT / "split_ubi.csv"

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "unzip", "curl")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "ultralytics==8.3.40",
        "lapx==0.5.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "tqdm==4.66.4",
        "kaggle==1.6.12",
        "opencv-python-headless==4.9.0.80",
    )
)


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    kaggle_slug: str = "intissarziani/ubi-fightsall"

    seed: int = 42

    # ── Graph preprocessing ───────────────────────────────────────────────────
    T:          int   = 32
    M:          int   = 6
    N:          int   = 8
    V:          int   = 17

    pose_conf:  float = 0.25
    obj_conf:   float = 0.25
    obj_iou:    float = 0.45
    min_joints: int   = 5

    # ── VideoMAE ──────────────────────────────────────────────────────────────
    T_vit:    int = 16
    vit_size: int = 224

    # ── Clip extraction (Approach B) ──────────────────────────────────────────
    # Positive: tile non-overlapping T-frame windows across each fight interval.
    #           Intervals shorter than T//2 are discarded (no coherent motion).
    # Negative: sample T-frame windows from non-fight intervals; at most
    #           neg_per_video windows per source video to keep imbalance mild.
    # neg_per_video=4 targets ~2.5:1 neg:pos with max_windows=3 tiling
    # (~1,500 train positives across 933 source videos → ~3,700 negatives).
    neg_per_video: int = 4

    # ── Quality-stratified negative selection (SOURCE FIX, 2026-05-30) ─────────
    # Root cause of the UBI shortcut: the original random negative sampler made
    # train negatives systematically LOWER skeleton-quality than the well-framed
    # fight intervals (q_skel 0.67 vs 0.91), so VideoMAE learned "clean -> fight".
    # On the official test split the negatives are also clean, the shortcut breaks
    # and test AUC collapses. Fix: oversample negative CANDIDATES per video, run
    # YOLO on all of them, then KEEP only the neg_per_video candidates whose q_skel
    # matches the fight (positive) target distribution — destroying the
    # "clean -> fight" correlation at the data source.
    #
    # neg_oversample: how many candidates to propose per kept negative. The only
    #   added GPU cost vs the original run is YOLO on the dropped candidates, so
    #   keep this small (1.5-2x). 2 => propose 8 candidates, keep 4 per video.
    neg_oversample: float = 2.0
    # quality_match: how to choose which candidates to keep from the scored pool.
    #   "fight_quantile" — match the kept negatives' q_skel distribution to the
    #     global fight q_skel distribution (quantile match). Default: removes the
    #     correlation rather than merely inverting it.
    #   "high"          — simply keep the highest-q_skel candidates (stronger
    #     decorrelation if test negatives are very clean; can overshoot).
    quality_match: str = "fight_quantile"

    # ── YOLO ─────────────────────────────────────────────────────────────────
    pose_weights:  str            = "yolo11x-pose.pt"
    obj_weights:   str            = "yolo11x.pt"
    pose_fallback: Tuple[str,...] = ("yolo11l-pose.pt", "yolo11m-pose.pt",
                                     "yolo11s-pose.pt")
    obj_fallback:  Tuple[str,...] = ("yolo11l.pt", "yolo11m.pt", "yolo11s.pt")

    force_reextract: bool = False


cfg = CFG()


# ── Helpers ───────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    import torch, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yolo(primary: str, fallbacks: Tuple[str, ...], task: str):
    from ultralytics import YOLO
    for candidate in (primary,) + fallbacks:
        try:
            print(f"  [{task}] Trying {candidate} …")
            m = YOLO(candidate)
            print(f"  [{task}] Loaded {candidate}")
            return m
        except Exception as e:
            print(f"  [{task}] Failed: {e}")
    raise RuntimeError(f"All YOLO candidates failed for task={task}")


def parse_annotation(ann_path: Path) -> List[int]:
    """
    Read a UBI-Fights annotation file.
    Returns a list of frame indices where label == 1 (fight).
    Format: one integer per line (0 or 1).
    """
    labels = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels


def fight_intervals(frame_labels: List[int]) -> List[Tuple[int, int]]:
    """
    Convert a list of per-frame binary labels to contiguous [start, end) intervals
    where label == 1.
    """
    intervals = []
    in_fight = False
    start = 0
    for i, lbl in enumerate(frame_labels):
        if lbl == 1 and not in_fight:
            in_fight = True
            start = i
        elif lbl == 0 and in_fight:
            in_fight = False
            intervals.append((start, i))
    if in_fight:
        intervals.append((start, len(frame_labels)))
    return intervals


def clips_for_interval(start: int, end: int,
                        n_frames: int, T: int,
                        max_windows: int = 3) -> List[List[int]]:
    """
    Return up to max_windows T-frame clips covering [start, end).
    Drops intervals shorter than T//2 (no coherent motion).

    Strategy:
      - dur < T//2 : drop entirely.
      - dur <= T   : one centred window.
      - dur > T    : evenly space up to max_windows start-points across the
                     interval using linspace, so the beginning, middle, and
                     end of every fight are always represented regardless of
                     how long the interval is.  This caps the clip count at
                     max_windows per interval even for multi-minute fights.
    """
    import numpy as np
    dur = end - start
    if dur < T // 2:
        return []

    if dur <= T:
        mid     = (start + end) // 2
        w_start = max(0, min(mid - T // 2, n_frames - T))
        starts  = [w_start]
    else:
        n_win  = min(max_windows, dur // T)   # never more than fit non-overlapping
        starts = np.linspace(start, end - T, n_win, dtype=int).tolist()

    result = []
    for ws in starts:
        ws = int(max(0, min(ws, n_frames - T)))
        we = ws + T
        result.append(np.linspace(ws, we - 1, T, dtype=int).tolist())
    return result


def negative_clip_indices(frame_labels: List[int],
                           n_frames: int, T: int,
                           neg_per_video: int,
                           rng: random.Random,
                           n_candidates: int = None) -> List[List[int]]:
    """
    Sample up to n_candidates non-overlapping T-frame windows from explicitly
    non-fight frames (label == 0 throughout window).

    n_candidates defaults to neg_per_video (original behaviour). The source fix
    passes n_candidates = ceil(neg_per_video * neg_oversample) so we propose a
    larger candidate pool; the per-video quality-matched selection downstream
    keeps only neg_per_video of them (those whose q_skel matches the fight
    target). The window-proposal logic itself is UNCHANGED (random anchor +
    all-non-fight + non-overlap validation) — only the count differs.
    """
    import numpy as np
    if n_candidates is None:
        n_candidates = neg_per_video
    non_fight = [i for i, l in enumerate(frame_labels) if l == 0]
    if len(non_fight) < T:
        return []

    clips = []
    attempts = 0
    used: List[Tuple[int, int]] = []

    # More candidates need more attempts to find non-overlapping windows.
    max_attempts = max(200, n_candidates * 50)
    while len(clips) < n_candidates and attempts < max_attempts:
        attempts += 1
        anchor = rng.choice(non_fight)
        half   = T // 2
        s = max(0, anchor - half)
        e = min(n_frames, s + T)
        s = max(0, e - T)
        # Check all frames in window are non-fight
        window_labels = frame_labels[s:e]
        if len(window_labels) < T or any(l == 1 for l in window_labels):
            continue
        # Check no overlap with existing negative clips
        overlap = any(not (e <= us or s >= ue) for us, ue in used)
        if overlap:
            continue
        used.append((s, e))
        clips.append(np.linspace(s, e - 1, T, dtype=int).tolist())

    return clips


# ── Main preprocessing function ───────────────────────────────────────────────
@app.function(
    image=image,
    gpu="L4",        # SOURCE FIX cost knob: L4 (cheaper/hr) — YOLO11x fits in 24GB
    cpu=2,
    memory=12288,
    timeout=25200,   # 7 hours
    volumes={
        str(RAW_MOUNT):  vol_raw,
        str(PROC_MOUNT): vol_proc,
    },
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
def preprocess(force_reextract: bool = True) -> None:
    # SOURCE FIX: default force_reextract=True. The negative-clip WINDOWS differ
    # from the original buggy run (oversampled candidate pool, new RNG sequence),
    # so reusing old NPZs by filename would mix old/new frame windows. A clean
    # re-extract guarantees every kept clip's frames match its inventory window.
    import numpy as np
    import pandas as pd
    import cv2
    import torch
    from tqdm import tqdm

    seed_everything(cfg.seed)
    rng    = random.Random(cfg.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    for d in [RAW_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Clean slate (SOURCE FIX) ──────────────────────────────────────────────
    # When re-extracting, purge ALL old clip NPZs so no stale window from the
    # buggy run survives under a colliding clip_id. Only touches CACHE_DIR NPZs;
    # raw videos and the Kaggle zip are untouched.
    if force_reextract or cfg.force_reextract:
        old_npz = list(CACHE_DIR.glob("*.npz"))
        for p in old_npz:
            try:
                p.unlink()
            except Exception as e:
                print(f"  WARN: could not remove old NPZ {p.name}: {e}")
        if old_npz:
            print(f"Clean slate: removed {len(old_npz)} old NPZ files from "
                  f"{CACHE_DIR} before re-extraction.")
            vol_proc.commit()

    # ═════════════════════════════════════════════════════════════════════════
    # Step 1 — Kaggle download & extraction
    # ═════════════════════════════════════════════════════════════════════════
    import json as _json
    _token    = os.environ.get("KAGGLE_TOKEN",    "").strip()
    _key      = os.environ.get("KAGGLE_KEY",      "").strip()
    _username = os.environ.get("KAGGLE_USERNAME", "").strip()

    if _token.startswith("{"):
        kaggle_json = _token
    elif _username and (_key or _token):
        kaggle_json = _json.dumps({"username": _username, "key": _key or _token})
    else:
        kaggle_json = ""

    zips = list(RAW_MOUNT.glob("*.zip"))

    if not zips:
        if not kaggle_json:
            raise RuntimeError(
                "No zip found in ubi-fights-raw and no usable Kaggle credentials.\n"
                "Options:\n"
                "  (a) Set KAGGLE_TOKEN secret to full kaggle.json contents\n"
                "  (b) Set KAGGLE_TOKEN=<api_key> and KAGGLE_USERNAME=<user>\n"
                "  (c) Upload zip manually:\n"
                "      modal volume put ubi-fights-raw ubi-fightsall.zip /ubi-fightsall.zip"
            )
        kdir = Path("/root/.kaggle")
        kdir.mkdir(exist_ok=True)
        (kdir / "kaggle.json").write_text(kaggle_json)
        os.chmod(str(kdir / "kaggle.json"), 0o600)

        print("Downloading UBI-Fights from Kaggle …")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", cfg.kaggle_slug,
             "-p", str(RAW_MOUNT), "--quiet"],
            check=True,
        )
        vol_raw.commit()
        zips = list(RAW_MOUNT.glob("*.zip"))

    zip_path = zips[0]
    print(f"Zip: {zip_path}")

    marker = RAW_MOUNT / ".extracted"
    if not marker.exists():
        print("Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(RAW_MOUNT))
        marker.touch()
        vol_raw.commit()
        print("Extraction done.")

    # Locate dataset root — handle both flat and nested zip layouts
    candidates = [
        RAW_MOUNT / "UBI_FIGHTS",
        RAW_MOUNT / "UBI-FIGHTS",
        RAW_MOUNT / "ubi_fights",
        RAW_MOUNT,
    ]
    dataset_root = next(
        (p for p in candidates
         if (p / "annotation").exists() or (p / "videos").exists()),
        RAW_MOUNT,
    )
    print(f"Dataset root: {dataset_root}")

    ann_dir    = dataset_root / "annotation"
    video_root = dataset_root / "videos"
    test_csv   = dataset_root / "test_videos.csv"

    if not ann_dir.exists():
        raise RuntimeError(f"annotation/ dir not found under {dataset_root}")
    if not video_root.exists():
        raise RuntimeError(f"videos/ dir not found under {dataset_root}")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 2 — Load test split list
    # ═════════════════════════════════════════════════════════════════════════
    test_stems: set = set()
    if test_csv.exists():
        import pandas as pd
        tc = pd.read_csv(str(test_csv), header=None)
        # Column 0 expected to be video filename or stem
        for val in tc.iloc[:, 0]:
            test_stems.add(Path(str(val)).stem)
        print(f"Test split: {len(test_stems)} videos from test_videos.csv")
    else:
        print("WARNING: test_videos.csv not found — all clips assigned to train")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 3 — Build clip inventory (Approach B)
    # ═════════════════════════════════════════════════════════════════════════
    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

    # Map annotation stem → annotation file
    ann_map: Dict[str, Path] = {}
    for af in sorted(ann_dir.rglob("*")):
        if af.is_file() and af.suffix in (".txt", ".csv", ""):
            ann_map[af.stem] = af

    # Map video stem → video file (search fight/ and non-fight/ subdirs + root)
    vid_map: Dict[str, Path] = {}
    for vf in sorted(video_root.rglob("*")):
        if vf.is_file() and vf.suffix.lower() in VIDEO_EXTS:
            vid_map[vf.stem] = vf

    print(f"Annotation files: {len(ann_map)}  |  Video files: {len(vid_map)}")

    # Reserve 20% of non-test video stems as a validation split.
    # Seeded shuffle ensures the same val/train boundary on every run.
    # Must come after ann_map is built (uses its keys).
    train_candidate_stems = [s for s in sorted(ann_map.keys())
                              if s not in test_stems]
    rng_val = random.Random(cfg.seed)
    rng_val.shuffle(train_candidate_stems)
    n_val = max(1, int(len(train_candidate_stems) * 0.20))
    val_stems: set = set(train_candidate_stems[:n_val])
    print(f"Val split: {len(val_stems)} videos (20% of non-test)")

    rows: List[Dict] = []

    for stem, ann_path in tqdm(sorted(ann_map.items()),
                                desc="Building clip inventory", unit="video"):
        vid_path = vid_map.get(stem)
        if vid_path is None:
            print(f"  WARN: no video for annotation {stem}, skipping.")
            continue

        if stem in test_stems:
            split = "test"
        elif stem in val_stems:
            split = "val"
        else:
            split = "train"

        frame_labels = parse_annotation(ann_path)
        n_ann        = len(frame_labels)
        if n_ann == 0:
            continue

        has_fight = any(l == 1 for l in frame_labels)

        if has_fight:
            # Positive clips — tile non-overlapping T-frame windows across each
            # annotated fight interval.  Intervals < T//2 frames are dropped.
            for (fs, fe) in fight_intervals(frame_labels):
                windows = clips_for_interval(fs, fe, n_ann, cfg.T)
                for wi, _ in enumerate(windows):
                    # clip_id encodes the original interval + window index so
                    # NPZ filenames remain stable across re-runs.
                    clip_id = f"{stem}_fight_{fs}_{fe}_w{wi}"
                    rows.append({
                        "clip_id":      clip_id,
                        "video_path":   str(vid_path),
                        "ann_path":     str(ann_path),
                        "label":        1,
                        "split":        split,
                        "frame_start":  fs,
                        "frame_end":    fe,
                        "clip_type":    "positive",
                        "n_ann_frames": n_ann,
                        "window_idx":   wi,
                        "src_stem":     stem,
                        "is_neg_cand":  False,
                    })

        # Negative clips — sampled from non-fight intervals.
        # SOURCE FIX: propose an OVERSAMPLED candidate pool; the post-loop
        # quality-matched selection keeps only neg_per_video per video. All
        # candidates are processed (YOLO) so we know each one's q_skel before
        # selecting; dropped candidates' NPZs are deleted afterwards.
        import math as _math
        n_cand = max(cfg.neg_per_video,
                     int(_math.ceil(cfg.neg_per_video * cfg.neg_oversample)))
        neg_clips = negative_clip_indices(
            frame_labels, n_ann, cfg.T, cfg.neg_per_video, rng,
            n_candidates=n_cand)
        for ni, idxs in enumerate(neg_clips):
            # clip_id carries the candidate index so every candidate NPZ has a
            # unique stable filename. Selection later prunes a subset of these.
            clip_id = f"{stem}_nonfight_{ni}"
            rows.append({
                "clip_id":      clip_id,
                "video_path":   str(vid_path),
                "ann_path":     str(ann_path),
                "label":        0,
                "split":        split,
                "frame_start":  int(idxs[0]),
                "frame_end":    int(idxs[-1]) + 1,
                "clip_type":    "negative",     # all candidates are real negatives
                "n_ann_frames": n_ann,
                "src_stem":     stem,           # group key for per-video selection
                "is_neg_cand":  True,           # marks rows subject to pruning
            })

    all_df = pd.DataFrame(rows)
    # Fill window_idx=0 for negative clips (they have no window tiling)
    if "window_idx" not in all_df.columns:
        all_df["window_idx"] = 0
    else:
        all_df["window_idx"] = all_df["window_idx"].fillna(0).astype(int)
    # Ensure the source-fix grouping columns exist for every row.
    if "src_stem" not in all_df.columns:
        all_df["src_stem"] = ""
    if "is_neg_cand" not in all_df.columns:
        all_df["is_neg_cand"] = False
    all_df["is_neg_cand"] = all_df["is_neg_cand"].fillna(False).astype(bool)

    n_neg_cand = int(all_df["is_neg_cand"].sum())
    print(f"\nClip inventory (with oversampled negative candidates):")
    print(f"  Total clips           : {len(all_df)}")
    print(f"  Negative CANDIDATES   : {n_neg_cand} "
          f"(oversample={cfg.neg_oversample}, keep {cfg.neg_per_video}/video)")
    print(all_df.groupby(["split", "label"]).size().to_string())

    # SOURCE FIX: this pre-loop CSV is PROVISIONAL — it lists every candidate
    # (more negatives than we will keep). The authoritative split_ubi.csv is
    # rewritten AFTER quality-matched selection (Step 6b) so it lists only kept
    # clips. We do NOT overwrite split_ubi.csv here, to avoid leaving a stale
    # candidate-inflated split file if the run is interrupted mid-loop.
    candidate_csv = PROC_MOUNT / "neg_candidates_ubi.csv"
    all_df.to_csv(str(candidate_csv), index=False)
    vol_proc.commit()

    # ═════════════════════════════════════════════════════════════════════════
    # Step 4 — Load YOLO models
    # ═════════════════════════════════════════════════════════════════════════
    print("\nLoading detectors …")
    pose_model = load_yolo(cfg.pose_weights, cfg.pose_fallback, "pose")
    obj_model  = load_yolo(cfg.obj_weights,  cfg.obj_fallback,  "object")
    pose_model.to(DEVICE)
    obj_model.to(DEVICE)

    # ═════════════════════════════════════════════════════════════════════════
    # Step 5 — Per-clip processing loop
    # ═════════════════════════════════════════════════════════════════════════
    T  = cfg.T
    M  = cfg.M
    N  = cfg.N
    V  = cfg.V
    Tv = cfg.T_vit
    SZ = cfg.vit_size

    gqs_rows: List[Dict] = []
    skipped       = 0
    processed_count = 0
    COMMIT_EVERY  = 100

    for _, row in tqdm(all_df.iterrows(), total=len(all_df),
                       desc="Preprocessing clips", unit="clip"):

        clip_id  = str(row["clip_id"])
        label    = int(row["label"])
        split    = str(row["split"])
        vid_path = str(row["video_path"])
        f_start  = int(row["frame_start"])
        f_end    = int(row["frame_end"])
        # Source-fix grouping metadata (carried into gqs_rows for selection).
        src_stem    = str(row.get("src_stem", ""))
        is_neg_cand = bool(row.get("is_neg_cand", False))
        npz_path = CACHE_DIR / f"{clip_id}.npz"

        # ── Skip already-done clips ───────────────────────────────────────
        if npz_path.exists() and not (force_reextract or cfg.force_reextract):
            try:
                with np.load(str(npz_path), allow_pickle=False) as d:
                    if "frames_vit" in d:
                        g = d["gqs"]
                        gqs_rows.append({
                            "clip_id": clip_id, "split": split, "label": label,
                            "q_skel": float(g[0]), "q_int": float(g[1]),
                            "q_obj":  float(g[2]), "q_po":  float(g[3]),
                            "valid_ratio": float(g[4]),
                            "src_stem": src_stem, "is_neg_cand": is_neg_cand,
                        })
                        continue
            except Exception:
                pass

        # ── Read relevant frames from source video ────────────────────────
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  WARN: cannot open {vid_path}")
            skipped += 1
            continue

        # Determine frame indices for this clip
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if row["clip_type"] == "positive":
            # Use the specific tiled window for this row (window_idx into
            # clips_for_interval output). Recompute to avoid storing all
            # frame indices in the CSV.
            wi       = int(row.get("window_idx", 0))
            windows  = clips_for_interval(f_start, f_end, n_total, T)
            if wi >= len(windows):
                # Stale window_idx (shouldn't happen, but guard against it)
                cap.release()
                skipped += 1
                continue
            graph_idx = windows[wi]
        else:
            import numpy as np_local
            graph_idx = np_local.linspace(f_start, max(f_start, f_end - 1),
                                          T, dtype=int).tolist()

        vit_idx = graph_idx  # same window; T_vit subset taken below

        needed = set(graph_idx)
        raw_frames: Dict[int, np.ndarray] = {}
        fi_cursor = 0

        for target_fi in sorted(needed):
            # Seek only when needed (forward seek is cheap, backward is not)
            if target_fi < fi_cursor:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
                fi_cursor = target_fi
            else:
                while fi_cursor < target_fi:
                    cap.read()   # skip frames
                    fi_cursor += 1

            ret, frame = cap.read()
            if ret:
                raw_frames[target_fi] = frame
            fi_cursor += 1

        cap.release()

        if len(raw_frames) == 0:
            skipped += 1
            continue

        # ── Allocate output arrays ────────────────────────────────────────
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

        frame_buffer: Dict[int, Dict] = {}

        for ti, fi in enumerate(graph_idx):
            fi = int(fi)
            frame_bgr = raw_frames.get(fi)
            if frame_bgr is None:
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

            h, w  = frame_bgr.shape[:2]
            denom = float(max(h, w))

            # ── Pose + ByteTrack ──────────────────────────────────────────
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
                        bx  = (x1 + x2) * 0.5 / denom
                        by  = (y1 + y2) * 0.5 / denom
                        bw  = (x2 - x1)       / denom
                        bh  = (y2 - y1)       / denom
                        conf = float(boxes_t.conf[pi].item())

                        speed = 0.0
                        if tid in prev_centers:
                            speed = float(np.linalg.norm(
                                np.array([bx, by]) - prev_centers[tid]))
                        prev_centers[tid] = np.array([bx, by])

                        track_ages[tid]  = track_ages.get(tid, 0) + 1
                        frames_elapsed   = ti + 1
                        continuity       = min(track_ages[tid] / frames_elapsed, 1.0)

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

            # ── Object detection ──────────────────────────────────────────
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

            # ── Fill structured arrays ────────────────────────────────────
            persons_s = sorted(persons, key=lambda p: p["conf"], reverse=True)[:M]
            objects_s = objects[:N]

            for pi, p in enumerate(persons_s):
                bx, by, bw, bh = p["box"]
                int_nodes[ti, pi] = [bx, by, bw, bh,
                                     p["conf"], p["speed"], p["continuity"]]
                int_node_mask[ti, pi] = True
                if p["kps"].shape[0] == V:
                    skeleton[ti, pi] = p["kps"]

            for pi in range(len(persons_s)):
                bxi, byi, bwi, bhi = persons_s[pi]["box"]
                for pj in range(len(persons_s)):
                    if pi == pj:
                        continue
                    bxj, byj, bwj, bhj = persons_s[pj]["box"]
                    dist  = float(np.hypot(bxi - bxj, byi - byj))
                    xi1, yi1 = bxi - bwi / 2, byi - bhi / 2
                    xi2, yi2 = bxi + bwi / 2, byi + bhi / 2
                    xj1, yj1 = bxj - bwj / 2, byj - bhj / 2
                    xj2, yj2 = bxj + bwj / 2, byj + bhj / 2
                    ix    = max(0.0, min(xi2, xj2) - max(xi1, xj1))
                    iy    = max(0.0, min(yi2, yj2) - max(yi1, yj1))
                    inter = ix * iy
                    union = bwi * bhi + bwj * bhj - inter + 1e-6
                    iou   = inter / union
                    close = 1.0 if dist < 0.15 else 0.0
                    rel_spd = abs(persons_s[pi]["speed"] - persons_s[pj]["speed"])
                    int_edges[ti, pi, pj]     = [dist, iou, close, rel_spd]
                    int_edge_mask[ti, pi, pj] = True

            for oi, obj in enumerate(objects_s):
                obj_nodes[ti, oi] = [obj["cx"], obj["cy"],
                                     obj["w"],  obj["h"],
                                     obj["conf"], obj["cls"]]
                obj_node_mask[ti, oi] = True

            for pi, p in enumerate(persons_s):
                kps    = p["kps"]
                bx, by, bw, bh = p["box"]
                wrist_pts = [kps[wi, :2]
                              for wi in (9, 10)
                              if kps[wi, 2] > 0.25]
                for oi, obj in enumerate(objects_s):
                    ox, oy, ow, oh = obj["cx"], obj["cy"], obj["w"], obj["h"]
                    body_d  = float(np.hypot(bx - ox, by - oy))
                    wrist_d = (float(min(np.hypot(wp[0] - ox, wp[1] - oy)
                                        for wp in wrist_pts))
                               if wrist_pts else 2.0)
                    xi1, yi1 = bx - bw / 2, by - bh / 2
                    xi2, yi2 = bx + bw / 2, by + bh / 2
                    xj1, yj1 = ox - ow / 2, oy - oh / 2
                    xj2, yj2 = ox + ow / 2, oy + oh / 2
                    ix    = max(0.0, min(xi2, xj2) - max(xi1, xj1))
                    iy    = max(0.0, min(yi2, yj2) - max(yi1, yj1))
                    inter = ix * iy
                    union = bw * bh + ow * oh - inter + 1e-6
                    po_iou     = inter / union
                    near_wrist = 1.0 if wrist_d < 0.10 else 0.0
                    near_body  = 1.0 if body_d  < 0.20 else 0.0
                    po_edges[ti, pi, oi]     = [wrist_d, body_d,
                                                po_iou, near_wrist, near_body]
                    po_edge_mask[ti, pi, oi] = True

        # ── VideoMAE frames ───────────────────────────────────────────────
        import numpy as np_vit
        vit_sample = np_vit.linspace(0, T - 1, Tv, dtype=int).tolist()
        for vi, ti in enumerate(vit_sample):
            fi = int(graph_idx[ti])
            bgr = raw_frames.get(fi)
            if bgr is not None:
                rgb = cv2.cvtColor(
                    cv2.resize(bgr, (SZ, SZ), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_BGR2RGB,
                )
                frames_vit[vi] = rgb
            elif vi > 0:
                frames_vit[vi] = frames_vit[vi - 1]

        # ── GQS ──────────────────────────────────────────────────────────
        # Count frames where at least one person has ≥ min_joints confident keypoints
        valid_skel = sum(
            1
            for ti in range(T)
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

        # ── Save NPZ ──────────────────────────────────────────────────────
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
            label         = np.array(label,  dtype=np.int64),
            split         = np.array(split),
        )

        gqs_rows.append({
            "clip_id": clip_id, "split": split, "label": label,
            "q_skel": float(q_skel), "q_int": float(q_int),
            "q_obj":  float(q_obj),  "q_po":  float(q_po),
            "valid_ratio": float(valid_ratio),
            "src_stem": src_stem, "is_neg_cand": is_neg_cand,
        })

        processed_count += 1
        if processed_count % COMMIT_EVERY == 0:
            vol_proc.commit()
            print(f"  [commit] {processed_count} clips processed — volume flushed.")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 6 — Quality-matched negative selection  (SOURCE FIX, CPU-only)
    # ═════════════════════════════════════════════════════════════════════════
    # Every clip (positives + ALL negative candidates) now has a measured q_skel
    # in gqs_rows. We keep neg_per_video negatives per source video whose q_skel
    # matches the FIGHT q_skel distribution, then drop the rest. This destroys the
    # train-only "clean skeleton -> fight" correlation at the data source.
    gqs_df = pd.DataFrame(gqs_rows)
    if "is_neg_cand" not in gqs_df.columns:
        gqs_df["is_neg_cand"] = False
    gqs_df["is_neg_cand"] = gqs_df["is_neg_cand"].fillna(False).astype(bool)
    if "src_stem" not in gqs_df.columns:
        gqs_df["src_stem"] = ""

    # Global fight q_skel target (used by the "fight_quantile" strategy). Fights
    # are the positive clips; we match negatives to this distribution so the two
    # classes become quality-indistinguishable.
    fight_q = gqs_df.loc[gqs_df["label"] == 1, "q_skel"].to_numpy()
    fight_q_med = float(np.median(fight_q)) if fight_q.size else 1.0
    print(f"\nQuality-matched negative selection "
          f"(strategy={cfg.quality_match}, fight q_skel median={fight_q_med:.3f})")

    def _select_keep_ids(cand: "pd.DataFrame") -> List[str]:
        """Return the clip_ids to KEEP from one video's negative candidates."""
        k = min(cfg.neg_per_video, len(cand))
        if k <= 0:
            return []
        if len(cand) <= cfg.neg_per_video:
            return cand["clip_id"].tolist()           # nothing to prune
        if cfg.quality_match == "high":
            # Keep the highest-q_skel candidates.
            ordered = cand.sort_values("q_skel", ascending=False)
            return ordered["clip_id"].head(k).tolist()
        # "fight_quantile": keep the k candidates whose q_skel is closest to the
        # global fight q_skel target (so kept-negative quality tracks fight
        # quality and the correlation with the label vanishes).
        cand = cand.assign(_d=(cand["q_skel"] - fight_q_med).abs())
        ordered = cand.sort_values("_d", ascending=True)
        return ordered["clip_id"].head(k).tolist()

    keep_neg_ids: set = set()
    cand_df = gqs_df[gqs_df["is_neg_cand"]]
    for stem, grp in cand_df.groupby("src_stem"):
        keep_neg_ids.update(_select_keep_ids(grp))

    # Drop = candidate negatives NOT selected. Positives and any non-candidate
    # rows are always kept.
    dropped_ids = [cid for cid in cand_df["clip_id"].tolist()
                   if cid not in keep_neg_ids]
    print(f"  Negative candidates : {len(cand_df)}")
    print(f"  Negatives KEPT      : {len(keep_neg_ids)}")
    print(f"  Negatives DROPPED   : {len(dropped_ids)}")

    # Delete dropped candidate NPZs so the volume holds only kept clips.
    # GUARD: only ever delete files whose clip_id is a confirmed dropped
    # candidate — never a positive, never a kept negative.
    keep_all_mask = (~gqs_df["is_neg_cand"]) | gqs_df["clip_id"].isin(keep_neg_ids)
    keep_ids_all  = set(gqs_df.loc[keep_all_mask, "clip_id"].tolist())
    deleted = 0
    for cid in dropped_ids:
        if cid in keep_ids_all:
            continue   # safety: never delete a kept clip
        p = CACHE_DIR / f"{cid}.npz"
        try:
            if p.exists():
                p.unlink()
                deleted += 1
        except Exception as e:
            print(f"  WARN: could not delete dropped candidate {cid}: {e}")
    print(f"  Dropped NPZs deleted: {deleted}")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 6b — Authoritative split_ubi.csv + gqs_summary_ubi.csv (kept clips)
    # ═════════════════════════════════════════════════════════════════════════
    kept_df = gqs_df[keep_all_mask].copy()

    # Rebuild the authoritative split CSV from the kept clips. Reload the
    # provisional candidate inventory to recover the full per-clip columns, then
    # filter to kept clip_ids. Column schema matches the original split_ubi.csv.
    candidate_csv = PROC_MOUNT / "neg_candidates_ubi.csv"
    inv_df = pd.read_csv(str(candidate_csv))
    split_out = inv_df[inv_df["clip_id"].isin(keep_ids_all)].copy()
    # Drop the source-fix-only helper columns from the public split CSV so it
    # stays schema-compatible with downstream scripts (clip_id, etc.).
    for c in ("src_stem", "is_neg_cand"):
        if c in split_out.columns:
            split_out = split_out.drop(columns=[c])
    split_out.to_csv(str(SPLIT_CSV), index=False)

    # Filtered GQS summary (kept clips only) — same columns as before, helper
    # columns dropped.
    gqs_out = kept_df.drop(columns=[c for c in ("src_stem", "is_neg_cand")
                                    if c in kept_df.columns])
    gqs_csv = PROC_MOUNT / "gqs_summary_ubi.csv"
    gqs_out.to_csv(str(gqs_csv), index=False)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete.  Skipped: {skipped}")
    print(f"NPZ files in {CACHE_DIR}: {len(list(CACHE_DIR.glob('*.npz')))}")
    print(f"Kept clips: {len(kept_df)}  (split_ubi.csv rows: {len(split_out)})")
    print("\nKept-clip counts by split x label:")
    print(kept_df.groupby(["split", "label"]).size().to_string())
    cols = ["q_skel", "q_int", "q_obj", "q_po", "valid_ratio"]
    print("\nGQS means by split x label (kept clips) — q_skel pos vs neg should "
          "now be CLOSE (shortcut removed):")
    print(kept_df.groupby(["split", "label"])[cols].mean().round(3).to_string())
    print(f"{'='*60}")

    overall = kept_df[cols].mean()
    print("\nOverall GQS means (kept clips, V9 YOLO11x):")
    for c in cols:
        print(f"  {c:<15}: {overall[c]:.3f}")

    vol_proc.commit()
    print("Volumes committed. Phase 1 complete.")
    print(f"  Raw        → ubi-fights-raw      ({RAW_MOUNT})")
    print(f"  Processed  → ubi-fights-processed ({PROC_MOUNT})")


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    print("Guardian Eye V9 — UBI-Fights Phase 1: Preprocessing")
    print("Run: modal run trigraph_v9_ubi_preprocess.py::preprocess")
    print("     modal run trigraph_v9_ubi_preprocess.py::preprocess --force-reextract")
