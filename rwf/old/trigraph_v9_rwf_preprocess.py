"""
trigraph_v9_rwf_preprocess.py
Guardian Eye — V9  |  Phase 1: Preprocessing
Dataset : RWF-2000  (vulamnguyen/rwf2000)
GPU     : L4  |  ETA ~6 hours for 4,000 videos

What this script does
─────────────────────
1. Downloads RWF-2000 from Kaggle into the Modal volume.
2. Carves a stratified val set (10%) from the official train split.
3. For every video:
   a. Runs YOLO11x-pose + ByteTrack (persist=True) sequentially to extract
      identity-stable person tracks.
   b. Runs YOLO11x object detector on the same frames.
   c. Builds the full V8-compatible graph arrays (skeleton, interaction,
      object, PO edges, masks) plus GQS.
   d. Additionally captures T_vit=16 uniformly-sampled frames resized to
      224×224 (uint8 RGB) for Phase 2 VideoMAE fine-tuning.
4. Saves one NPZ per video to CACHE_DIR.
5. Saves split.csv and gqs_summary.csv to MOUNT_PATH.

NPZ schema (V9, superset of V8)
────────────────────────────────
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
  frames_vit    [T_vit, H, W, 3] uint8     224×224 RGB for VideoMAE  ← NEW in V9
  label         scalar           int64
  split         scalar           str

Constants: T=32, T_vit=16, M=6, N=8, V=17

Usage
─────
  modal run trigraph_v9_rwf_preprocess.py::preprocess
  modal run trigraph_v9_rwf_preprocess.py::preprocess --force-reextract
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
APP_NAME         = "guardian-eye-v9-rwf-preprocess"
VOL_NAME_RAW     = "rwf-2000-raw"        # raw videos only
VOL_NAME_PROC    = "rwf-2000-processed"  # NPZ cache + split CSV + GQS
SECRET_NAME      = "KAGGLE_TOKEN"

app        = modal.App(APP_NAME)
vol_raw    = modal.Volume.from_name(VOL_NAME_RAW,  create_if_missing=True)
vol_proc   = modal.Volume.from_name(VOL_NAME_PROC, create_if_missing=True)

# Each volume is mounted at its own path inside the container.
# Raw volume  → /data/raw   (large, written once, never touched by Phase 2/3)
# Proc volume → /data/proc  (NPZ cache, split CSV, GQS — read by Phase 2/3)
RAW_MOUNT  = Path("/data/raw")
PROC_MOUNT = Path("/data/proc")

RAW_DIR   = RAW_MOUNT  / "videos"        # extracted Kaggle dataset
CACHE_DIR = PROC_MOUNT / "cache_v9"      # one NPZ per video
SPLIT_CSV = PROC_MOUNT / "split_v9.csv"

# ── Container image ───────────────────────────────────────────────────────────
# Pinned versions identical to V8.2/V8.3 for library parity, except
# ultralytics bumped to support YOLO11x weights.
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
        # YOLO11x requires ultralytics ≥ 8.3
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
    # ── Dataset ──────────────────────────────────────────────────────────────
    kaggle_slug:   str = "vulamnguyen/rwf2000"
    class_map: Dict[str, int] = field(
        default_factory=lambda: {"Fight": 1, "NonFight": 0})

    # ── Split ─────────────────────────────────────────────────────────────────
    seed: int = 42

    # ── Graph preprocessing ───────────────────────────────────────────────────
    T:          int   = 32       # graph frames (V8-compatible)
    M:          int   = 6        # max persons
    N:          int   = 8        # max objects
    V:          int   = 17       # COCO joints

    pose_conf:  float = 0.25
    obj_conf:   float = 0.25
    obj_iou:    float = 0.45
    min_joints: int   = 5        # GQS: minimum visible joints per person-frame

    # ── VideoMAE frame extraction (NEW in V9) ─────────────────────────────────
    T_vit:       int = 16        # frames for VideoMAE (standard ViT-S input)
    vit_size:    int = 224       # spatial resolution

    # ── Detector (upgraded to x for maximum keypoint recall on CCTV) ─────────
    pose_weights: str = "yolo11x-pose.pt"
    obj_weights:  str = "yolo11x.pt"
    # Fallback chain if x weights are unavailable
    pose_fallback: Tuple[str, ...] = ("yolo11l-pose.pt", "yolo11m-pose.pt",
                                       "yolo11s-pose.pt")
    obj_fallback:  Tuple[str, ...] = ("yolo11l.pt", "yolo11m.pt", "yolo11s.pt")

    force_reextract: bool = False


cfg = CFG()


# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── YOLO loader with fallback chain ──────────────────────────────────────────
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


# ── Preprocessing function ────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=12288,           # 12 GB — sufficient; preprocessing is GPU-bound
    timeout=25200,          # 7 hours
    volumes={
        str(RAW_MOUNT):  vol_raw,    # raw videos (read/write during Phase 1)
        str(PROC_MOUNT): vol_proc,   # NPZ cache + CSV outputs
    },
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
def preprocess(force_reextract: bool = False) -> None:
    """
    Full preprocessing pipeline for RWF-2000 V9.
    Produces one NPZ per video in CACHE_DIR with V9 schema.

    If the KAGGLE_TOKEN secret is empty (token not yet obtained), place the
    dataset zip manually in the raw volume first:
        modal volume put rwf-2000-raw rwf2000.zip /rwf2000.zip
    The script will skip the Kaggle download step automatically.
    """
    import numpy as np
    import pandas as pd
    import cv2
    import torch
    from tqdm import tqdm

    seed_everything(cfg.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # ── Directories ───────────────────────────────────────────────────────────
    for d in [RAW_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════════
    # Step 1 — Kaggle download & extraction
    # ═════════════════════════════════════════════════════════════════════════
    # Build kaggle.json content from whatever the secret contains.
    # Modal secrets inject each key as a separate env var.
    # Supported layouts:
    #   (a) KAGGLE_TOKEN = full JSON string {"username":"...","key":"..."}
    #   (b) KAGGLE_USERNAME + KAGGLE_KEY  (two separate keys in the secret)
    #   (c) KAGGLE_USERNAME + KAGGLE_TOKEN (token used as the key value)
    import json as _json
    _token    = os.environ.get("KAGGLE_TOKEN", "").strip()
    _key      = os.environ.get("KAGGLE_KEY",   "").strip()
    _username = os.environ.get("KAGGLE_USERNAME", "").strip()

    if _token.startswith("{"):
        # Layout (a): secret value is already a valid kaggle.json blob
        kaggle_json = _token
    elif _username and (_key or _token):
        # Layout (b/c): assemble JSON from separate username + key fields
        kaggle_json = _json.dumps({"username": _username, "key": _key or _token})
    else:
        kaggle_json = ""   # no credentials — will use zip upload path

    # Check for a pre-existing zip in the raw volume (manual upload path)
    zips = list(RAW_MOUNT.glob("*.zip"))

    if not zips:
        if not kaggle_json:
            raise RuntimeError(
                "No zip found in rwf-2000-raw and no usable Kaggle credentials.\n"
                "Options:\n"
                "  (a) Set KAGGLE_TOKEN secret to the full contents of kaggle.json\n"
                "      {\"username\": \"...\", \"key\": \"...\"}\n"
                "  (b) Set KAGGLE_TOKEN to just the API key and KAGGLE_USERNAME to your username\n"
                "  (c) Upload the zip manually:\n"
                "      modal volume put rwf-2000-raw rwf2000.zip /rwf2000.zip"
            )
        # Write kaggle.json from secret
        kdir = Path("/root/.kaggle")
        kdir.mkdir(exist_ok=True)
        (kdir / "kaggle.json").write_text(kaggle_json)
        os.chmod(str(kdir / "kaggle.json"), 0o600)

        print("Downloading RWF-2000 from Kaggle …")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", cfg.kaggle_slug,
             "-p", str(RAW_MOUNT), "--quiet"],
            check=True,
        )
        vol_raw.commit()
        zips = list(RAW_MOUNT.glob("*.zip"))

    zip_path = zips[0]
    print(f"Zip: {zip_path}")

    marker = RAW_DIR / ".extracted"
    if not marker.exists():
        import re
        print("Extracting …")

        def safe_filename(name: str, max_len: int = 120) -> str:
            """
            Sanitise broken/corrupted filenames from RWF-2000 zip.
            """

            name = os.path.basename(name)

            stem, ext = os.path.splitext(name)

            # remove broken unicode / illegal chars
            stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)

            # collapse repeated underscores
            stem = re.sub(r"_+", "_", stem)

            # trim long filenames
            stem = stem[:max_len]

            return stem + ext


        with zipfile.ZipFile(zip_path, "r") as zf:

            for member in zf.infolist():

                # skip folders
                if member.is_dir():
                    continue

                original = member.filename

                # preserve dataset structure
                # expected:
                # RWF-2000/train/Fight/xxx.avi
                parts = Path(original).parts

                if len(parts) >= 4:
                    split_name = parts[-3]
                    class_name = parts[-2]
                else:
                    split_name = "unknown"
                    class_name = "unknown"

                stem = Path(parts[-1]).stem
                suffix = Path(parts[-1]).suffix

                safe_stem = safe_filename(stem, max_len=100)

                # append zip CRC to guarantee uniqueness
                clean_name = f"{safe_stem}_{member.CRC:x}{suffix}"

                target_dir = (
                    RAW_DIR
                    / "RWF-2000"
                    / split_name
                    / class_name
                )

                target_dir.mkdir(parents=True, exist_ok=True)

                target_path = target_dir / clean_name

                try:
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())

                except Exception as e:
                    print(f"  WARN: failed extracting {original}: {e}")

        marker.touch()
        vol_raw.commit()
        print("Extraction done.")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 2 — Discover videos and build split CSV
    # ═════════════════════════════════════════════════════════════════════════
    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

    def discover(split_name: str) -> List[Dict]:
        """Walk RWF-2000 official split directory; return list of video dicts."""
        candidates = [
            RAW_DIR / "RWF-2000" / split_name,
            RAW_DIR / f"RWF-2000_{split_name}",
            RAW_DIR / split_name,
            RAW_MOUNT / "RWF-2000" / split_name,   # flat zip extraction
        ]
        base = next((p for p in candidates if p.exists()), None)
        if base is None:
            # Warn and return empty list rather than aborting the whole run
            print(
                f"  WARNING: Cannot find official '{split_name}' folder. "
                f"Tried: {[str(c) for c in candidates]}. Skipping split."
            )
            return []
        print(f"  {split_name} → {base}")
        rows = []
        for class_name, label in cfg.class_map.items():
            class_dir = base / class_name
            if not class_dir.exists():
                print(f"  WARNING: {class_dir} not found, skipping.")
                continue
            for p in sorted(class_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    rows.append({
                        "video_id":       f"{split_name}_{class_name}_{p.stem}",
                        "video_path":     str(p),
                        "class_name":     class_name,
                        "label":          int(label),
                        "official_split": split_name,
                    })
        return rows

    print("Discovering videos …")
    train_rows = discover("train")
    val_rows   = discover("val")
    print(f"  Official train={len(train_rows)}  val={len(val_rows)}")

    if not train_rows and not val_rows:
        raise RuntimeError(
            "No videos found in either split. Check the raw volume contents."
        )

    # Use official train/val splits from dataset
    train_df = pd.DataFrame(train_rows) if train_rows else pd.DataFrame()
    if not train_df.empty:
        train_df["split"] = "train"

    val_df = pd.DataFrame(val_rows) if val_rows else pd.DataFrame()
    if not val_df.empty:
        val_df["split"] = "val"

    all_df = pd.concat(
        [df for df in [train_df, val_df] if not df.empty],
        ignore_index=True,
    )
    print(f"  Split counts:\n{all_df['split'].value_counts().to_string()}")
    print(f"  Label counts:\n{all_df['label'].value_counts().to_string()}")

    all_df.to_csv(str(SPLIT_CSV), index=False)
    vol_proc.commit()

    # ═════════════════════════════════════════════════════════════════════════
    # Step 3 — Load YOLO models (YOLO11x primary, fallback chain)
    # ═════════════════════════════════════════════════════════════════════════
    print("Loading detectors …")
    pose_model = load_yolo(cfg.pose_weights, cfg.pose_fallback, "pose")
    obj_model  = load_yolo(cfg.obj_weights,  cfg.obj_fallback,  "object")
    pose_model.to(DEVICE)
    obj_model.to(DEVICE)

    # ═════════════════════════════════════════════════════════════════════════
    # Step 4 — Per-video processing loop
    # ═════════════════════════════════════════════════════════════════════════
    T  = cfg.T
    M  = cfg.M
    N  = cfg.N
    V  = cfg.V
    Tv = cfg.T_vit
    SZ = cfg.vit_size

    gqs_rows: List[Dict] = []
    skipped = 0
    processed_count = 0   # counts videos that completed NPZ write this run
    COMMIT_EVERY = 100    # flush to Modal volume every N processed videos

    for _, row in tqdm(all_df.iterrows(), total=len(all_df),
                       desc="Preprocessing", unit="video"):

        vid      = str(row["video_id"])
        label    = int(row["label"])
        split    = str(row["split"])
        vid_path = str(row["video_path"])
        npz_path = CACHE_DIR / f"{vid}.npz"

        # ── Skip if already done ──────────────────────────────────────────
        if npz_path.exists() and not (force_reextract or cfg.force_reextract):
            try:
                # allow_pickle=False avoids mmap handle leaks that block overwrite
                with np.load(str(npz_path), allow_pickle=False) as d:
                    g = d["gqs"]
                    # Guard: skip if V9 frame array is missing (incomplete V8 NPZ)
                    if "frames_vit" not in d:
                        pass   # fall through to re-extract
                    else:
                        gqs_rows.append({
                            "video_id": vid, "split": split, "label": label,
                            "q_skel": float(g[0]), "q_int": float(g[1]),
                            "q_obj":  float(g[2]), "q_po":  float(g[3]),
                            "valid_ratio": float(g[4]),
                        })
                        continue
            except Exception:
                pass

        # ── Open video, validate ──────────────────────────────────────────
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  WARN: cannot open {vid_path}")
            skipped += 1
            continue

        # CAP_PROP_FRAME_COUNT is unreliable for many AVI containers.
        # Read all frames first to get the true count, then derive indices.
        raw_frames_all: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames_all.append(frame)
        cap.release()

        n_frames = len(raw_frames_all)
        if n_frames < 2:
            skipped += 1
            continue

        # ── Frame index sets ──────────────────────────────────────────────
        # Graph frames: T=32 linspace (V8-compatible)
        graph_idx = np.linspace(0, n_frames - 1, T, dtype=int)
        # ViT frames:  T_vit=16 linspace (for VideoMAE; must be ≤ T indices)
        vit_idx   = np.linspace(0, n_frames - 1, Tv, dtype=int)

        # Build sparse dict only for indices we actually need
        needed = set(graph_idx.tolist()) | set(vit_idx.tolist())
        raw_frames: Dict[int, np.ndarray] = {
            i: raw_frames_all[i] for i in needed
        }
        del raw_frames_all   # free memory before allocating NPZ arrays

        if not raw_frames:
            skipped += 1
            continue

        # ── Allocate output arrays ────────────────────────────────────────
        skeleton      = np.zeros((T, M, V, 3),    dtype=np.float32)
        int_nodes     = np.zeros((T, M, 7),        dtype=np.float32)
        int_edges     = np.zeros((T, M, M, 4),     dtype=np.float32)
        int_node_mask = np.zeros((T, M),            dtype=bool)
        int_edge_mask = np.zeros((T, M, M),         dtype=bool)
        obj_nodes     = np.zeros((T, N, 6),         dtype=np.float32)
        obj_node_mask = np.zeros((T, N),            dtype=bool)
        po_edges      = np.zeros((T, M, N, 5),      dtype=np.float32)
        po_edge_mask  = np.zeros((T, M, N),         dtype=bool)
        # VideoMAE frames: [T_vit, H, W, C] uint8 RGB
        frames_vit    = np.zeros((Tv, SZ, SZ, 3),   dtype=np.uint8)

        prev_centers: Dict[int, np.ndarray] = {}
        track_ages:   Dict[int, int]        = {}

        # ── Sequential tracking pass ──────────────────────────────────────
        # ByteTrack requires sequential inference with persist=True to
        # maintain Kalman filter state. We iterate all required frames
        # in temporal order, running pose tracking at every graph frame
        # and storing results in frame_buffer.
        #
        # reset_tracker() is not a public Ultralytics API; the real reset
        # path is through the predictor's tracker list.  Silently skip if
        # the internals differ across versions — worst case is stale IDs
        # from the previous video, which only affects continuity features.
        try:
            for tracker in pose_model.predictor.trackers:
                tracker.reset()
        except Exception:
            pass

        frame_buffer: Dict[int, Dict] = {}
        graph_set = set(graph_idx.tolist())

        for fi in sorted(raw_frames.keys()):
            frame_bgr = raw_frames[fi]
            if fi not in graph_set:
                continue   # only track on graph-sampled frames

            h, w   = frame_bgr.shape[:2]
            denom  = float(max(h, w))

            # ── Pose + ByteTrack ──────────────────────────────────────────
            persons = []
            try:
                res_pose = pose_model.track(
                    frame_bgr, conf=cfg.pose_conf, persist=True,
                    tracker="bytetrack.yaml", verbose=False,
                )[0]
                if res_pose.keypoints is not None and res_pose.boxes is not None:
                    boxes_t  = res_pose.boxes
                    kps_data = res_pose.keypoints.data.cpu().numpy()   # [P, V, 3]
                    ids_raw  = (boxes_t.id.int().cpu().tolist()
                                if boxes_t.id is not None
                                else list(range(len(boxes_t))))
                    for pi in range(len(ids_raw)):
                        tid            = int(ids_raw[pi])
                        x1, y1, x2, y2 = boxes_t.xyxy[pi].cpu().numpy()
                        bx  = (x1 + x2) * 0.5 / denom
                        by  = (y1 + y2) * 0.5 / denom
                        bw  = (x2 - x1)       / denom
                        bh  = (y2 - y1)       / denom
                        conf = float(boxes_t.conf[pi].item())

                        # Speed: Euclidean displacement of normalised centre
                        speed = 0.0
                        if tid in prev_centers:
                            speed = float(np.linalg.norm(
                                np.array([bx, by]) - prev_centers[tid]))
                        prev_centers[tid] = np.array([bx, by])

                        # Track continuity: frames tracked / frames elapsed so far
                        track_ages[tid] = track_ages.get(tid, 0) + 1
                        frames_elapsed  = len(frame_buffer) + 1   # includes current
                        continuity      = min(track_ages[tid] / frames_elapsed, 1.0)

                        # Normalise keypoints to [0,1]
                        kps_n = kps_data[pi].copy()    # [V, 3]
                        kps_n[:, 0] /= denom
                        kps_n[:, 1] /= denom

                        persons.append({
                            "id": tid, "conf": conf,
                            "box": (bx, by, bw, bh),
                            "speed": speed, "continuity": continuity,
                            "kps": kps_n,
                        })
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                raise   # propagate OOM / CUDA errors; don't swallow silently
            except Exception:
                pass   # leave persons empty; masks stay False

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
                        # COCO has 80 classes (IDs 0–79); divide by 79 to normalise to [0,1]
                        "cls":  float(res_obj.boxes.cls[oi].item()) / 79.0,
                    })
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                raise
            except Exception:
                pass

            frame_buffer[fi] = {"persons": persons, "objects": objects}

        # ── Fill structured arrays from frame_buffer ──────────────────────
        for ti, fi in enumerate(graph_idx.tolist()):
            fi = int(fi)
            fb = frame_buffer.get(fi)
            if fb is None:
                # Repeat previous frame on read failure
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

            # Sort by confidence descending so the M most-confident detections fill slots
            persons = sorted(fb["persons"], key=lambda p: p["conf"], reverse=True)[:M]
            objects = fb["objects"][:N]

            # ── Skeleton and person nodes ─────────────────────────────────
            for pi, p in enumerate(persons):
                bx, by, bw, bh = p["box"]
                int_nodes[ti, pi] = [bx, by, bw, bh,
                                     p["conf"], p["speed"], p["continuity"]]
                int_node_mask[ti, pi] = True
                if p["kps"].shape[0] == V:
                    skeleton[ti, pi] = p["kps"]
                # If kps shape is wrong, skeleton slot stays zero and mask
                # is still True (person was detected, just no keypoints)

            # ── Interaction edges (person↔person within frame) ────────────
            for pi in range(len(persons)):
                bxi, byi, bwi, bhi = persons[pi]["box"]
                for pj in range(len(persons)):
                    if pi == pj:
                        continue
                    bxj, byj, bwj, bhj = persons[pj]["box"]
                    dist = float(np.hypot(bxi - bxj, byi - byj))
                    # IoU
                    xi1, yi1 = bxi - bwi / 2, byi - bhi / 2
                    xi2, yi2 = bxi + bwi / 2, byi + bhi / 2
                    xj1, yj1 = bxj - bwj / 2, byj - bhj / 2
                    xj2, yj2 = bxj + bwj / 2, byj + bhj / 2
                    ix = max(0.0, min(xi2, xj2) - max(xi1, xj1))
                    iy = max(0.0, min(yi2, yj2) - max(yi1, yj1))
                    inter = ix * iy
                    union = bwi * bhi + bwj * bhj - inter + 1e-6
                    iou   = inter / union
                    close = 1.0 if dist < 0.15 else 0.0
                    rel_spd = abs(persons[pi]["speed"] - persons[pj]["speed"])
                    int_edges[ti, pi, pj]     = [dist, iou, close, rel_spd]
                    int_edge_mask[ti, pi, pj] = True

            # ── Object nodes ──────────────────────────────────────────────
            for oi, obj in enumerate(objects):
                obj_nodes[ti, oi] = [obj["cx"], obj["cy"],
                                     obj["w"],  obj["h"],
                                     obj["conf"], obj["cls"]]
                obj_node_mask[ti, oi] = True

            # ── Person-object edges ───────────────────────────────────────
            for pi, p in enumerate(persons):
                kps    = p["kps"]         # [V, 3]
                bx, by, bw, bh = p["box"]
                # Wrist indices in COCO-17: 9 (left), 10 (right)
                wrist_pts = [kps[wi, :2]
                             for wi in (9, 10)
                             if kps[wi, 2] > 0.25]
                for oi, obj in enumerate(objects):
                    ox, oy, ow, oh = obj["cx"], obj["cy"], obj["w"], obj["h"]
                    body_d  = float(np.hypot(bx - ox, by - oy))
                    # Use large sentinel (2.0) when wrists are not visible so
                    # near_wrist stays 0 and does not inherit body proximity
                    wrist_d = (float(min(np.hypot(wp[0] - ox, wp[1] - oy)
                                        for wp in wrist_pts))
                               if wrist_pts else 2.0)
                    # Person-object IoU
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

        # ── VideoMAE frames: resize T_vit frames to 224×224 RGB ──────────
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
                frames_vit[vi] = frames_vit[vi - 1]   # repeat on read failure

        # ── GQS computation ───────────────────────────────────────────────
        # q_skel: fraction of person-frame slots with ≥ min_joints visible
        valid_skel = sum(
            1
            for ti in range(T)
            for pi in range(M)
            if int_node_mask[ti, pi]
            and int((skeleton[ti, pi, :, 2] > 0.25).sum()) >= cfg.min_joints
        )
        q_skel = valid_skel / (T * M + 1e-6)

        # q_int: fraction of frames with ≥ 2 valid persons
        q_int = sum(int(int_node_mask[ti].sum() >= 2)
                    for ti in range(T)) / T

        # q_obj: fraction of frames with ≥ 1 valid object
        q_obj = sum(int(obj_node_mask[ti].sum() >= 1)
                    for ti in range(T)) / T

        # q_po: fraction of frames with ≥ 1 valid PO edge
        q_po = sum(int(po_edge_mask[ti].any())
                   for ti in range(T)) / T

        # valid_ratio: fraction of frames with ≥ 1 detected person
        valid_ratio = sum(int(int_node_mask[ti].any())
                          for ti in range(T)) / T

        gqs = np.array([q_skel, q_int, q_obj, q_po, valid_ratio],
                       dtype=np.float32)

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
            label         = np.array(label, dtype=np.int64),
            split         = np.array(split),
        )

        gqs_rows.append({
            "video_id": vid, "split": split, "label": label,
            "q_skel": float(q_skel), "q_int": float(q_int),
            "q_obj":  float(q_obj),  "q_po":  float(q_po),
            "valid_ratio": float(valid_ratio),
        })

        processed_count += 1
        if processed_count % COMMIT_EVERY == 0:
            vol_proc.commit()
            print(f"  [commit] {processed_count} videos processed — volume flushed.")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 5 — GQS summary
    # ═════════════════════════════════════════════════════════════════════════
    gqs_df  = pd.DataFrame(gqs_rows)
    gqs_csv = PROC_MOUNT / "gqs_summary_v9.csv"
    gqs_df.to_csv(str(gqs_csv), index=False)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete.  Skipped: {skipped}")
    print(f"NPZ files in {CACHE_DIR}: "
          f"{len(list(CACHE_DIR.glob('*.npz')))}")
    print("\nGQS means by split:")
    cols = ["q_skel", "q_int", "q_obj", "q_po", "valid_ratio"]
    print(gqs_df.groupby("split")[cols].mean().round(3).to_string())
    print(f"{'='*60}")

    # Compare YOLO11x GQS to V8 YOLO11s benchmarks for reference
    overall = gqs_df[cols].mean()
    print("\nOverall GQS means (V9 YOLO11x):")
    for c in cols:
        print(f"  {c:<15}: {overall[c]:.3f}")

    vol_proc.commit()
    print("Volumes committed. Phase 1 complete.")
    print(f"  Raw videos  → rwf-2000-raw   ({RAW_MOUNT})")
    print(f"  Processed   → rwf-2000-processed ({PROC_MOUNT})")


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    print("Guardian Eye V9 — Phase 1: Preprocessing")
    print("Run: modal run trigraph_v9_rwf_preprocess.py::preprocess")
    print("     modal run trigraph_v9_rwf_preprocess.py::preprocess "
          "--force-reextract")
