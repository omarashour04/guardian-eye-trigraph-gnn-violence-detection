# -*- coding: utf-8 -*-
"""
TRI-GRAPH V8.1 — Hockey Fight
==============================
Single self-contained notebook: preprocessing + training in one run.

Architecture:
  Stream 1 — Skeleton  : CompactSTGCN on [C,T,V,M] graph tensor
  Stream 2 — Interaction: FrameGATv2 (per-frame person graph) + BiGRU + attention
  Stream 3 — Object    : FrameGINE  (per-frame object graph)  + BiGRU + attention
  Stream 4 — PO        : FrameGINE  (bipartite person-object)  + BiGRU + attention
  Fusion    : UniversalQualityGatedFusion with GQS conditioning + stream dropout
  Head      : Linear → ReLU → Dropout → Linear → logit

Fixes vs V8:
  - [BUG FIX] FrameGATv2 NaN propagation on empty-person frames:
      When all nodes in a frame are invalid (no persons detected), every row of
      the attention matrix is [-inf, -inf, …] and softmax returns NaN.  NaN
      then propagates through bmm → LayerNorm → BiGRU → fusion → loss,
      corrupting all weights in any experiment that includes the interaction
      branch.  Fix: rows where every column is -inf are replaced with 0.0
      before softmax, producing a uniform distribution that is cancelled by the
      downstream masked mean (mask_f = 0 for those nodes).
      Impact: 39.5% of Hockey Fight videos contain at least one empty-person
      frame.  Without this fix experiments C (skel+int) and E (full trigraph)
      both collapse to 50% accuracy at epoch 1.

  - [REGULARISATION] Label-smoothed BCE loss (α = 0.03):
      BCEWithLogitsLoss does not support label_smoothing natively.
      Targets are smoothed manually: y_smooth = y*(1-α) + α/2.
      Reduces overconfident fitting on the 700-sample training set.

  - [REGULARISATION] Per-stream learning rate:
      Single-stream experiments use cfg.lr (3e-4).
      Multi-stream experiments (2+ streams) use cfg.lr_multi_stream (1e-4).
      Rationale: additional streams increase effective capacity and accelerate
      memorisation.  A lower lr gives the model time to generalise.

  - [REGULARISATION] Minimum checkpoint epoch (min_checkpoint_epoch = 5):
      Checkpoints and patience counting are frozen for the first 4 epochs.
      Prevents a lucky warm-up epoch from being reported as the best result.

  - [REGULARISATION] Stream dropout raised 0.25 → 0.35:
      Stronger stream-level regularisation for the small Hockey Fights dataset.

  - [ROBUSTNESS] NaN batch detection in training loop:
      If a batch produces a non-finite loss (should not occur after the above
      fix, but guarded defensively), the backward pass is skipped and the batch
      is counted.  The number of skipped batches is logged per epoch.

Target: V4.2 D_skeleton_object → Acc=0.955, Macro-F1=0.9550, AUC=0.9658
"""

# ============================================================
# Cell 1 — Imports
# ============================================================
import os, cv2, json, math, time, random, warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not available:", e)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)
print("Torch  :", torch.__version__)

# ============================================================
# Cell 2 — Configuration
# ============================================================
@dataclass
class CFG:
    # ---- paths ----
    dataset_root: str = r"C:\Violence detection\datasets\Hockey fight\data"
    output_root:  str = r"C:\Violence detection\sara_elwakeel\we are really there\tri_graph\v8_hockey_fight"
    cache_dir_name: str = "v8_unified_cache_T32"
    split_dir_name: str = "splits"
    run_dir_name:   str = "runs"

    # ---- video sampling ----
    target_frames: int = 32
    max_persons:   int = 6      # M
    max_objects:   int = 8      # N
    num_joints:    int = 17     # V  (COCO-17)

    # ---- YOLO models: primary = YOLO11, fallbacks in order ----
    # Pose model fallback chain: yolo11s-pose → yolo11n-pose → yolov8s-pose (local)
    pose_model_candidates: tuple = (
        "yolo11s-pose.pt",                  # primary: YOLO11s pose
        "yolo11n-pose.pt",                  # fallback 1: YOLO11n pose (lighter)
        r"C:\Violence detection\sara_elwakeel\we are really there\skeleton_only\yolov8n-pose.pt",  # fallback 2: local YOLOv8
    )
    # Object model fallback chain: yolo11s → yolo11n → yolov8s
    object_model_candidates: tuple = (
        "yolo11s.pt",                       # primary: YOLO11s detect
        "yolo11n.pt",                       # fallback 1: YOLO11n detect (lighter)
        "yolov8s.pt",                       # fallback 2: YOLOv8s detect
    )

    # ---- detection thresholds (identical to V4.2) ----
    pose_conf:                   float = 0.25
    object_conf:                 float = 0.20
    object_iou:                  float = 0.45
    spatial_distance_threshold:  float = 0.30
    temporal_stability_kappa:    float = 1.50
    min_keypoint_conf:           float = 0.25
    min_valid_joints:            int   = 5
    skeleton_temporal_kappa:     float = 0.15
    wrist_alpha:                 float = 2.0
    body_distance_alpha:         float = 1.5
    min_iou_for_po_edge:         float = 0.01

    # ---- split ----
    seed:        int   = 42
    train_ratio: float = 0.70   # fixed three-way split
    val_ratio:   float = 0.10
    test_ratio:  float = 0.20

    # ---- graph dimensions (kept minimal to control param count) ----
    int_node_dim:  int = 7   # interaction node features
    int_edge_dim:  int = 4   # interaction edge features
    obj_node_dim:  int = 6   # object node features
    po_edge_dim:   int = 5   # PO edge features
    graph_hidden:  int = 64  # per-frame graph conv output dim

    # ---- training ----
    batch_size:           int   = 16
    epochs:               int   = 45
    lr:                   float = 3e-4   # single-stream experiments
    lr_multi_stream:      float = 1e-4   # 2+ streams: slower convergence on small dataset
    weight_decay:         float = 1e-3
    embed_dim:            int   = 128
    dropout:              float = 0.35
    stream_dropout_prob:  float = 0.35   # raised from 0.25; slows memorisation on Hockey Fights
    patience:             int   = 8
    lr_factor:            float = 0.5
    lr_patience:          int   = 4
    grad_clip:            float = 1.0
    use_weighted_sampler: bool  = True
    num_workers:          int   = 0

    # ---- convergence stability ----
    # Minimum number of epochs before any checkpoint is saved.
    # Prevents early-epoch noise from becoming the reported best result.
    # Patience counter is also frozen until this epoch is reached.
    min_checkpoint_epoch: int   = 5
    # Binary label smoothing: target 1 → (1 - α), target 0 → α/2.
    # Reduces overconfident fitting on the small Hockey Fights training set.
    label_smoothing:      float = 0.03

    # ---- misc ----
    force_reextract: bool = False

cfg = CFG()
CLASS_MAP = {"nofights": 0, "fights": 1}

OUT_ROOT  = Path(cfg.output_root)
CACHE_DIR = OUT_ROOT / cfg.cache_dir_name
SPLIT_DIR = OUT_ROOT / cfg.split_dir_name
RUN_DIR   = OUT_ROOT / cfg.run_dir_name
for p in [OUT_ROOT, CACHE_DIR, SPLIT_DIR, RUN_DIR]:
    p.mkdir(parents=True, exist_ok=True)

with open(OUT_ROOT / "v8_config.json", "w") as f:
    json.dump(asdict(cfg), f, indent=4)

print("Output root:", OUT_ROOT)
print("Cache dir  :", CACHE_DIR)
print("Split      : 70 / 10 / 20")

# ============================================================
# Cell 3 — Reproducibility
# ============================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

seed_everything(cfg.seed)

# ============================================================
# Cell 4 — Dataset discovery + fixed three-way split
# ============================================================
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}

def discover_videos(root: str) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for class_name, label in CLASS_MAP.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append({
                    "video_id":   f"{class_name}_{p.stem}",
                    "video_path": str(p),
                    "class_name": class_name,
                    "label":      int(label),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No videos found under dataset root.")
    print(f"Discovered {len(df)} videos")
    print(df.groupby(["class_name", "label"]).size())
    return df

def make_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Fixed three-way stratified split: 70% train / 10% val / 20% test.
    Saved to CSV so it is identical across every run.
    """
    split_path = SPLIT_DIR / f"split_seed{seed}.csv"
    if split_path.exists():
        print("Loading existing split:", split_path)
        return pd.read_csv(split_path)

    # first carve off test (20%)
    train_val, test = train_test_split(
        df, test_size=cfg.test_ratio,
        stratify=df["label"], random_state=seed
    )
    # then carve off val from the remainder (10 / 80 = 0.125)
    val_frac = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_frac,
        stratify=train_val["label"], random_state=seed
    )

    train = train.copy(); train["split"] = "train"
    val   = val.copy();   val["split"]   = "val"
    test  = test.copy();  test["split"]  = "test"

    split_df = pd.concat([train, val, test], ignore_index=True)
    split_df.to_csv(split_path, index=False)
    print("Saved split:", split_path)
    print(pd.crosstab(split_df["split"], split_df["label"]))
    return split_df

all_videos_df = discover_videos(cfg.dataset_root)
split_df      = make_split(all_videos_df, cfg.seed)

# ============================================================
# Cell 5 — Load YOLO models with fallback chain
# ============================================================
if not ULTRALYTICS_AVAILABLE:
    raise ImportError("Install ultralytics: pip install ultralytics")

def load_model_with_fallback(candidates: tuple, task_name: str) -> YOLO:
    """
    Try each candidate in order.
    A candidate that is an absolute path is tested for existence first —
    if the file does not exist it is skipped without attempting a download.
    A candidate that is a plain model name (e.g. "yolo11s-pose.pt") is
    attempted via Ultralytics auto-download; if that raises any exception
    the next candidate is tried.
    Returns the first successfully loaded YOLO model.
    Raises RuntimeError if all candidates fail.
    """
    last_err = None
    for candidate in candidates:
        p = Path(candidate)
        # absolute / local path: must exist on disk
        if p.is_absolute() or (len(p.parts) > 1):
            if not p.exists():
                print(f"  [{task_name}] Local path not found, skipping: {p}")
                continue
            candidate = str(p)
        try:
            print(f"  [{task_name}] Trying: {candidate}")
            model = YOLO(candidate)
            print(f"  [{task_name}] Loaded: {candidate}")
            return model
        except Exception as e:
            print(f"  [{task_name}] Failed ({e}), trying next fallback ...")
            last_err = e
    raise RuntimeError(
        f"All YOLO candidates failed for {task_name}. Last error: {last_err}"
    )

print("Loading pose model ...")
pose_model   = load_model_with_fallback(cfg.pose_model_candidates,   "pose")
print("Loading object model ...")
object_model = load_model_with_fallback(cfg.object_model_candidates, "object")
print("Both models ready.")

# ============================================================
# Cell 6 — Geometry helpers (identical to V4.2)
# ============================================================
def box_area_norm(box, W, H):
    x1, y1, x2, y2 = box
    return max(0, x2-x1) * max(0, y2-y1) / max(W*H, 1)

def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    ua = max(0,ax2-ax1)*max(0,ay2-ay1) + max(0,bx2-bx1)*max(0,by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

def center_distance_norm(a, b, W, H):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    diag = math.sqrt(W*W + H*H)
    return math.sqrt(((ax1+ax2)/2-(bx1+bx2)/2)**2 +
                     ((ay1+ay2)/2-(by1+by2)/2)**2) / max(diag,1e-6)

def sample_frame_indices(n: int, target: int) -> List[int]:
    if n <= 0: return []
    if n >= target:
        return np.linspace(0, n-1, target).round().astype(int).tolist()
    idx = list(range(n))
    while len(idx) < target: idx.append(n-1)
    return idx[:target]

def read_sampled_frames(video_path: str, target: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(n, target)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros_like(frames[-1]) if frames else np.zeros((224,224,3), dtype=np.uint8)
        frames.append(frame)
    cap.release()
    return frames

# ============================================================
# Cell 7 — Object category mapping (identical to V4.2)
# ============================================================
WEAPON_LIKE   = {"knife","scissors","baseball bat","sports ball","skateboard","snowboard"}
SPORTS_OBJ    = {"sports ball","baseball bat","skateboard","snowboard","tennis racket"}
BAG_NAMES     = {"backpack","handbag","suitcase"}
BOTTLE_NAMES  = {"bottle","cup","wine glass"}
FURNITURE     = {"chair","bench","couch","dining table","bed"}
VEHICLE_NAMES = {"bicycle","car","motorcycle","bus","truck","train"}
GROUP_TO_ID   = {
    "general_object": 0, "weapon_like": 1, "sports_object": 2,
    "vehicle": 3,        "bag": 4,         "bottle_cup": 5,    "furniture": 6,
}

def get_object_group(name: str) -> str:
    name = name.lower()
    if name in WEAPON_LIKE:   return "weapon_like"
    if name in SPORTS_OBJ:    return "sports_object"
    if name in VEHICLE_NAMES: return "vehicle"
    if name in BAG_NAMES:     return "bag"
    if name in BOTTLE_NAMES:  return "bottle_cup"
    if name in FURNITURE:     return "furniture"
    return "general_object"

# ============================================================
# Cell 8 — YOLO result parsers (identical to V4.2)
# ============================================================
def parse_pose_result(result, W, H) -> List[dict]:
    persons = []
    if result.boxes is None or len(result.boxes) == 0:
        return persons
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    kpts_xy, kpts_conf = None, None
    if result.keypoints is not None:
        try:
            kpts_xy   = result.keypoints.xy.detach().cpu().numpy()
            kpts_conf = result.keypoints.conf.detach().cpu().numpy()
        except Exception:
            pass
    for i, box in enumerate(boxes):
        conf = float(confs[i])
        if conf < cfg.pose_conf: continue
        kp = np.zeros((cfg.num_joints, 3), dtype=np.float32)
        if kpts_xy is not None and i < len(kpts_xy):
            kp[:,0] = kpts_xy[i][:,0] / max(W,1)
            kp[:,1] = kpts_xy[i][:,1] / max(H,1)
            kp[:,2] = kpts_conf[i] if kpts_conf is not None else 1.0
        vis   = kp[:,2] >= cfg.min_keypoint_conf
        area  = box_area_norm(box, W, H)
        mkc   = float(kp[:,2].mean())
        score = area*0.5 + mkc*0.5
        persons.append({
            "box": np.array(box, dtype=np.float32),
            "conf": conf, "keypoints": kp,
            "visible_count": int(vis.sum()),
            "mean_kpt_conf": mkc, "area": area, "score": score,
        })
    return sorted(persons, key=lambda x: x["score"], reverse=True)[:cfg.max_persons]

def parse_object_result(result, W, H) -> List[dict]:
    objects = []
    if result.boxes is None or len(result.boxes) == 0:
        return objects
    names = result.names if hasattr(result, "names") else object_model.names
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss  = result.boxes.cls.detach().cpu().numpy().astype(int)
    for box, conf, cls_id in zip(boxes, confs, clss):
        cls_name = str(names.get(int(cls_id), str(cls_id))).lower()
        if cls_name == "person": continue
        if float(conf) < cfg.object_conf: continue
        group = get_object_group(cls_name)
        objects.append({
            "box":      np.array(box, dtype=np.float32),
            "conf":     float(conf),
            "cls_name": cls_name,
            "group":    group,
            "group_id": GROUP_TO_ID[group],
            "area":     box_area_norm(box, W, H),
        })
    return sorted(objects, key=lambda x: x["conf"], reverse=True)[:cfg.max_objects]

# ============================================================
# Cell 9 — Tracking: ByteTrack via YOLO11 built-in tracker
# ============================================================
# ByteTrack is used via pose_model.track(..., persist=True,
# tracker="bytetrack.yaml") called frame-by-frame inside
# process_one_video. The persist=True flag is critical — it
# instructs the tracker to maintain Kalman filter state across
# consecutive .track() calls on the same model instance, giving
# stable identity assignments even through fast motion and
# occlusion.
#
# Track IDs are read from result.boxes.id (int tensor).
# result.boxes.id may be None on the very first frame or when
# no detections exist — this is handled with a null check.
#
# Speed computation uses a prev_centers dict {track_id: (cx,cy)}
# maintained inside process_one_video, updated each frame.
# Track continuity is estimated as min(track_age / T, 1.0)
# where track_age is the number of frames a given ID has been
# seen, also tracked in a track_ages dict.
#
# SimpleIoUTracker has been fully removed.

# ============================================================
# Cell 10 — Per-frame graph builders
# ============================================================

def build_skeleton_frame(persons: List[dict]) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
        arr   [M, V, 3]   skeleton keypoints, zeros for empty person slots
        gqs   float       frame-level skeleton quality score
        valid float       1.0 if at least one person with >=min_valid_joints
    """
    M, V = cfg.max_persons, cfg.num_joints
    arr = np.zeros((M, V, 3), dtype=np.float32)
    q_list = []
    for i, p in enumerate(persons[:M]):
        k = p["keypoints"].copy()
        k[k[:,2] < cfg.min_keypoint_conf] = 0.0
        arr[i] = k
        vis = p["keypoints"][:,2] >= cfg.min_keypoint_conf
        vc  = int(vis.sum())
        if vc < cfg.min_valid_joints:
            q_list.append(0.0)
        else:
            q_list.append(0.70*(vc/V) + 0.30*float(p["keypoints"][vis,2].mean()))
    gqs   = float(np.mean(q_list)) if q_list else 0.0
    valid = 1.0 if q_list and max(q_list) > 0 else 0.0
    return arr, gqs, valid


def build_interaction_frame(persons: List[dict],
                            prev_centers: Dict[int, Tuple[float, float]],
                            track_ages:   Dict[int, int],
                            W: int, H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
        nodes     [M, 7]      person node features
        edges     [M, M, 4]   pairwise edge features
        node_mask [M]         bool, True where person slot occupied
        edge_mask [M, M]      bool, True where both endpoints valid
        gqs       float       frame-level interaction quality
    Node features (7):
        cx_norm, cy_norm, w_norm, h_norm, detection_conf, speed, track_continuity
    Edge features (4):
        center_distance_norm, iou, close_flag, relative_speed

    prev_centers : {track_id: (cx, cy)} from the previous frame
    track_ages   : {track_id: age_in_frames} cumulative count
    Both dicts are maintained by process_one_video and passed in each frame.
    """
    M = cfg.max_persons
    nodes     = np.zeros((M, cfg.int_node_dim), dtype=np.float32)
    edges     = np.zeros((M, M, cfg.int_edge_dim), dtype=np.float32)
    node_mask = np.zeros(M, dtype=bool)
    edge_mask = np.zeros((M, M), dtype=bool)
    speeds    = {}

    n = len(persons)
    for i, p in enumerate(persons[:M]):
        box  = p["box"]
        cx   = (box[0]+box[2])/2/W
        cy   = (box[1]+box[3])/2/H
        w    = (box[2]-box[0])/W
        h    = (box[3]-box[1])/H
        conf = p["conf"]
        tid  = p.get("track_id", None)
        spd, cont = 0.0, 0.0
        if tid is not None:
            if tid in prev_centers:
                cx0, cy0 = prev_centers[tid]
                spd = math.sqrt((cx-cx0)**2+(cy-cy0)**2)
            if tid in track_ages:
                cont = min(track_ages[tid] / cfg.target_frames, 1.0)
        speeds[i] = spd
        nodes[i]     = [cx, cy, w, h, conf, spd, cont]
        node_mask[i] = True

    for i in range(min(n, M)):
        for j in range(min(n, M)):
            if i == j: continue
            dist  = center_distance_norm(persons[i]["box"], persons[j]["box"], W, H)
            ov    = iou_xyxy(persons[i]["box"], persons[j]["box"])
            close = 1.0 if dist < cfg.spatial_distance_threshold else 0.0
            rspd  = abs(speeds.get(i,0.0) - speeds.get(j,0.0))
            edges[i,j]     = [dist, ov, close, rspd]
            edge_mask[i,j] = True

    # GQS: mirror V4.2 interaction_gqs formula
    if n == 0:
        return nodes, edges, node_mask, edge_mask, 0.0
    confs      = np.array([p["conf"] for p in persons[:M]], dtype=np.float32)
    conts      = np.array([nodes[i,6] for i in range(min(n,M))], dtype=np.float32)
    close_vals = edges[:min(n,M), :min(n,M), 2]
    close_pair = float(close_vals.mean()) if close_vals.size > 0 else 0.0
    ov_vals    = edges[:min(n,M), :min(n,M), 1]
    ov_pair    = float(ov_vals.mean()) if ov_vals.size > 0 else 0.0
    count_ch   = 0.0  # no prev_count stored at frame level; kept at 0
    det_q      = float(confs.mean())
    conn_q     = min(close_pair + ov_pair, 1.0)
    t_stab     = math.exp(-count_ch / max(cfg.temporal_stability_kappa, 1e-6))
    track_cont = float(conts.mean())
    gqs = (0.30*det_q + 0.25*track_cont + 0.20*conn_q +
           0.15*t_stab + 0.10*close_pair)
    return nodes, edges, node_mask, edge_mask, gqs


def build_object_frame(objects: List[dict]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
        nodes     [N, 6]   object node features
        node_mask [N]      bool
        gqs       float
    Node features (6):
        cx_norm, cy_norm, w_norm, h_norm, detection_conf, group_id_norm
    """
    N = cfg.max_objects
    nodes     = np.zeros((N, cfg.obj_node_dim), dtype=np.float32)
    node_mask = np.zeros(N, dtype=bool)
    for i, o in enumerate(objects[:N]):
        box  = o["box"]
        # box is in pixel coords from parse_object_result — needs W,H normalization
        # W,H passed via closure captured in process_one_video
        nodes[i]     = [o["_cx"], o["_cy"], o["_w"], o["_h"],
                        o["conf"], o["group_id"]/6.0]
        node_mask[i] = True
    if len(objects) == 0:
        return nodes, node_mask, 0.0
    confs        = np.array([o["conf"] for o in objects[:N]])
    count_norm   = min(len(objects)/N, 1.0)
    weapon_ratio = float(np.mean([1 if o["group"] in ["weapon_like","sports_object"]
                                  else 0 for o in objects[:N]]))
    gqs = 0.60*float(confs.mean()) + 0.25*count_norm + 0.15*weapon_ratio
    return nodes, node_mask, gqs


def wrist_points(person: dict) -> List[np.ndarray]:
    k   = person["keypoints"]
    pts = []
    for idx in [9, 10]:  # left wrist, right wrist
        if k[idx, 2] >= cfg.min_keypoint_conf:
            pts.append(k[idx, :2])
    return pts


def build_po_frame(persons: List[dict], objects: List[dict],
                   W: int, H: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
        po_edges  [M, N, 5]   person-object edge features
        po_mask   [M, N]      bool, True where person and object both valid
        gqs       float
    Edge features (5):
        wrist_dist_norm, body_dist_norm, iou, near_wrist_flag, near_body_flag
    """
    M, N = cfg.max_persons, cfg.max_objects
    po_edges = np.zeros((M, N, cfg.po_edge_dim), dtype=np.float32)
    po_mask  = np.zeros((M, N), dtype=bool)
    diag     = math.sqrt(W*W + H*H)

    edge_exists, near_wrist_vals, near_body_vals, overlap_vals = [], [], [], []

    for i, p in enumerate(persons[:M]):
        pbox  = p["box"]
        px1,py1,px2,py2 = pbox
        pcx,pcy = (px1+px2)/2, (py1+py2)/2
        p_scale = max(px2-px1, py2-py1, 1.0)
        wpts    = wrist_points(p)

        for j, o in enumerate(objects[:N]):
            obox  = o["box"]
            ox1,oy1,ox2,oy2 = obox
            ocx,ocy = (ox1+ox2)/2, (oy1+oy2)/2
            obj_scale = max(ox2-ox1, oy2-oy1, 1.0)

            if wpts:
                wd = min(math.sqrt(((wp[0]*W)-ocx)**2+((wp[1]*H)-ocy)**2)
                         for wp in wpts)
            else:
                wd = diag
            bd  = math.sqrt((pcx-ocx)**2+(pcy-ocy)**2)
            ov  = iou_xyxy(pbox, obox)
            nw  = 1.0 if wd < cfg.wrist_alpha * obj_scale else 0.0
            nb  = 1.0 if bd < cfg.body_distance_alpha * p_scale else 0.0
            has_edge = 1.0 if (nw or nb or ov >= cfg.min_iou_for_po_edge) else 0.0

            po_edges[i,j] = [wd/max(diag,1e-6), bd/max(diag,1e-6), ov, nw, nb]
            po_mask[i,j]  = True

            edge_exists.append(has_edge)
            near_wrist_vals.append(nw)
            near_body_vals.append(nb)
            overlap_vals.append(1.0 if ov >= cfg.min_iou_for_po_edge else 0.0)

    if not edge_exists:
        return po_edges, po_mask, 0.0

    er  = float(np.mean(edge_exists))
    nwr = float(np.mean(near_wrist_vals))
    ovr = float(np.mean(overlap_vals))
    gqs = 0.50*er + 0.30*nwr + 0.20*ovr
    return po_edges, po_mask, gqs

# ============================================================
# Cell 11 — Process one video → NPZ (YOLO11 + ByteTrack)
# ============================================================
def process_one_video(video_path: str, video_id: str,
                      label: int, split: str) -> Tuple[str, str]:
    cache_path = CACHE_DIR / f"{video_id}.npz"
    if cache_path.exists() and not cfg.force_reextract:
        return str(cache_path), "cached"

    frames = read_sampled_frames(video_path, cfg.target_frames)
    T  = len(frames)
    M  = cfg.max_persons
    N  = cfg.max_objects
    V  = cfg.num_joints

    # pre-allocate all arrays
    skel_seq      = np.zeros((T, M, V, 3),                   dtype=np.float32)
    int_nodes_seq = np.zeros((T, M, cfg.int_node_dim),       dtype=np.float32)
    int_edges_seq = np.zeros((T, M, M, cfg.int_edge_dim),    dtype=np.float32)
    int_nmask_seq = np.zeros((T, M),                         dtype=bool)
    int_emask_seq = np.zeros((T, M, M),                      dtype=bool)
    obj_nodes_seq = np.zeros((T, N, cfg.obj_node_dim),       dtype=np.float32)
    obj_nmask_seq = np.zeros((T, N),                         dtype=bool)
    po_edges_seq  = np.zeros((T, M, N, cfg.po_edge_dim),     dtype=np.float32)
    po_mask_seq   = np.zeros((T, M, N),                      dtype=bool)

    skel_q_seq = np.zeros(T, dtype=np.float32)
    int_q_seq  = np.zeros(T, dtype=np.float32)
    obj_q_seq  = np.zeros(T, dtype=np.float32)
    po_q_seq   = np.zeros(T, dtype=np.float32)
    valid_seq  = np.zeros(T, dtype=np.float32)

    # ByteTrack state maintained across frames via prev_centers + track_ages.
    # pose_model.track(..., persist=True) keeps Kalman filter state internally
    # between consecutive calls on the same model instance.
    prev_centers: Dict[int, Tuple[float, float]] = {}
    track_ages:   Dict[int, int]                 = {}

    for t, frame in enumerate(frames):
        H_f, W_f = frame.shape[:2]

        # --- pose + ByteTrack (persist=True maintains tracker state) ---
        try:
            pose_res = pose_model.track(
                frame,
                persist     = True,
                tracker     = "bytetrack.yaml",
                conf        = cfg.pose_conf,
                verbose     = False,
            )[0]
        except Exception:
            # bytetrack.yaml may not be in PATH on some installs;
            # fall back to botsort which ships with ultralytics by default
            pose_res = pose_model.track(
                frame,
                persist  = True,
                conf     = cfg.pose_conf,
                verbose  = False,
            )[0]

        # --- object detection (no tracking needed for objects) ---
        obj_res = object_model.predict(
            frame,
            conf    = cfg.object_conf,
            iou     = cfg.object_iou,
            verbose = False,
        )[0]

        # --- parse pose result; attach ByteTrack IDs ---
        persons = parse_pose_result(pose_res, W_f, H_f)
        objects = parse_object_result(obj_res, W_f, H_f)

        # attach track IDs from ByteTrack to each person dict
        if (pose_res.boxes is not None and
                pose_res.boxes.id is not None and
                len(pose_res.boxes.id) > 0):
            track_ids_raw = pose_res.boxes.id.int().cpu().numpy()
            confs_raw     = pose_res.boxes.conf.cpu().numpy()
            # sort by conf descending (same order as parse_pose_result)
            order = np.argsort(-confs_raw)
            sorted_ids = track_ids_raw[order][:cfg.max_persons]
            for k, p in enumerate(persons):
                if k < len(sorted_ids):
                    p["track_id"] = int(sorted_ids[k])

        # update prev_centers and track_ages for this frame
        current_centers: Dict[int, Tuple[float, float]] = {}
        for p in persons:
            tid = p.get("track_id", None)
            if tid is None:
                continue
            box = p["box"]
            cx  = (box[0]+box[2])/2/W_f
            cy  = (box[1]+box[3])/2/H_f
            current_centers[tid] = (cx, cy)
            track_ages[tid]      = track_ages.get(tid, 0) + 1

        # attach normalized bbox coords to objects for build_object_frame
        for o in objects:
            box = o["box"]
            o["_cx"] = (box[0]+box[2])/2/W_f
            o["_cy"] = (box[1]+box[3])/2/H_f
            o["_w"]  = (box[2]-box[0])/W_f
            o["_h"]  = (box[3]-box[1])/H_f

        sk,  sk_q, sk_v = build_skeleton_frame(persons)
        in_n, in_e, in_nm, in_em, in_q = build_interaction_frame(
            persons, prev_centers, track_ages, W_f, H_f)
        ob_n, ob_nm, ob_q = build_object_frame(objects)
        po_e, po_m, po_q  = build_po_frame(persons, objects, W_f, H_f)

        skel_seq[t]      = sk
        int_nodes_seq[t] = in_n
        int_edges_seq[t] = in_e
        int_nmask_seq[t] = in_nm
        int_emask_seq[t] = in_em
        obj_nodes_seq[t] = ob_n
        obj_nmask_seq[t] = ob_nm
        po_edges_seq[t]  = po_e
        po_mask_seq[t]   = po_m

        skel_q_seq[t] = sk_q
        int_q_seq[t]  = in_q
        obj_q_seq[t]  = ob_q
        po_q_seq[t]   = po_q
        valid_seq[t]  = sk_v

        # advance prev_centers for next frame
        prev_centers = current_centers

    # GQS summary — same formula as V4.2
    if T > 1:
        delta = float(np.mean(np.abs(np.diff(skel_q_seq))))
    else:
        delta = 0.0
    skel_stab = math.exp(-delta / max(cfg.skeleton_temporal_kappa, 1e-6))
    skel_gqs  = 0.70*float(skel_q_seq.mean()) + 0.30*skel_stab
    clip_gqs  = np.array([
        skel_gqs,
        float(int_q_seq.mean()),
        float(obj_q_seq.mean()),
        float(po_q_seq.mean()),
        float(valid_seq.mean()),
    ], dtype=np.float32)

    np.savez_compressed(
        cache_path,
        # metadata
        video_id = video_id,
        label    = np.array(label, dtype=np.int64),
        split    = split,
        # skeleton graph: [T, M, V, 3]  — NO flattening
        skeleton      = skel_seq,
        # interaction graph
        int_nodes     = int_nodes_seq,   # [T, M, 7]
        int_edges     = int_edges_seq,   # [T, M, M, 4]
        int_node_mask = int_nmask_seq,   # [T, M]
        int_edge_mask = int_emask_seq,   # [T, M, M]
        # object graph
        obj_nodes     = obj_nodes_seq,   # [T, N, 6]
        obj_node_mask = obj_nmask_seq,   # [T, N]
        # PO graph (reuses person/object nodes from above)
        po_edges      = po_edges_seq,    # [T, M, N, 5]
        po_edge_mask  = po_mask_seq,     # [T, M, N]
        # quality
        gqs           = clip_gqs,        # [5]
    )
    return str(cache_path), "extracted"

# ============================================================
# Cell 12 — Run preprocessing for all videos
# ============================================================
cache_rows, errors = [], []
for row in tqdm(split_df.to_dict("records"), desc="V8 preprocessing"):
    try:
        path, status = process_one_video(
            row["video_path"], row["video_id"],
            int(row["label"]), row["split"])
        cache_rows.append({**row, "npz_path": path, "status": status})
    except Exception as e:
        errors.append({**row, "error": str(e)})

cache_df = pd.DataFrame(cache_rows)
cache_df.to_csv(OUT_ROOT / "v8_cache_index.csv", index=False)
if errors:
    pd.DataFrame(errors).to_csv(OUT_ROOT / "v8_cache_errors.csv", index=False)
    print(f"ERRORS: {len(errors)} videos failed — see v8_cache_errors.csv")

print(f"Cached : {(cache_df['status']=='cached').sum()}")
print(f"Extracted: {(cache_df['status']=='extracted').sum()}")
print(f"Errors : {len(errors)}")

# ============================================================
# Cell 13 — GQS diagnostics
# ============================================================
gqs_rows = []
for p in tqdm(cache_df["npz_path"], desc="GQS summary"):
    z = np.load(p, allow_pickle=True)
    g = z["gqs"]
    gqs_rows.append({
        "video_id":    str(z["video_id"]),
        "label":       int(z["label"]),
        "split":       str(z["split"]),
        "skel_gqs":    float(g[0]),
        "int_gqs":     float(g[1]),
        "obj_gqs":     float(g[2]),
        "po_gqs":      float(g[3]),
        "valid_ratio": float(g[4]),
    })
gqs_df = pd.DataFrame(gqs_rows)
gqs_df.to_csv(OUT_ROOT / "v8_gqs_summary.csv", index=False)
print("\nGQS statistics:")
print(gqs_df[["skel_gqs","int_gqs","obj_gqs","po_gqs","valid_ratio"]].describe().round(3))

# ============================================================
# Cell 14 — Dataset
# ============================================================
class V8Dataset(Dataset):
    """
    Loads one NPZ per video and returns all graph tensors.
    No in-memory caching — NPZ files are small enough to read on demand.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        z   = np.load(row["npz_path"], allow_pickle=True)
        return {
            "video_id":     str(z["video_id"]),
            "label":        torch.tensor(int(z["label"]), dtype=torch.float32),
            "gqs":          torch.tensor(z["gqs"].astype(np.float32)),
            # skeleton [T,M,V,3] → transposed to [3,T,V,M] here
            "skeleton":     torch.tensor(
                                z["skeleton"].transpose(3,0,2,1).astype(np.float32)),
            # interaction
            "int_nodes":    torch.tensor(z["int_nodes"].astype(np.float32)),
            "int_edges":    torch.tensor(z["int_edges"].astype(np.float32)),
            "int_node_mask":torch.tensor(z["int_node_mask"].astype(bool)),
            "int_edge_mask":torch.tensor(z["int_edge_mask"].astype(bool)),
            # object
            "obj_nodes":    torch.tensor(z["obj_nodes"].astype(np.float32)),
            "obj_node_mask":torch.tensor(z["obj_node_mask"].astype(bool)),
            # PO
            "po_edges":     torch.tensor(z["po_edges"].astype(np.float32)),
            "po_edge_mask": torch.tensor(z["po_edge_mask"].astype(bool)),
        }

def collate_fn(batch):
    """Stack all tensors along batch dimension. Shapes are fixed so no padding needed."""
    keys_tensor = [
        "label","gqs","skeleton",
        "int_nodes","int_edges","int_node_mask","int_edge_mask",
        "obj_nodes","obj_node_mask",
        "po_edges","po_edge_mask",
    ]
    out = {"video_id": [b["video_id"] for b in batch]}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

train_df = cache_df[cache_df["split"] == "train"].copy()
val_df   = cache_df[cache_df["split"] == "val"].copy()
test_df  = cache_df[cache_df["split"] == "test"].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

train_ds = V8Dataset(train_df)
val_ds   = V8Dataset(val_df)
test_ds  = V8Dataset(test_df)

if cfg.use_weighted_sampler:
    labels  = train_df["label"].values.astype(int)
    counts  = np.bincount(labels, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sw      = torch.tensor([weights[y] for y in labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True,
                                    generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, collate_fn=collate_fn,
                              num_workers=cfg.num_workers, pin_memory=True)
else:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=cfg.num_workers, pin_memory=True)

val_loader  = DataLoader(val_ds,  batch_size=cfg.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=cfg.num_workers)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=cfg.num_workers)

# ============================================================
# Cell 15 — Model: skeleton branch (CompactSTGCN, from V7)
# ============================================================
def coco17_adj(V=17) -> torch.Tensor:
    edges = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),
    ]
    A = np.eye(V, dtype=np.float32)
    for i, j in edges:
        A[i,j] = A[j,i] = 1.0
    deg = A.sum(1)
    D   = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-6)))
    return torch.tensor(D @ A @ D, dtype=torch.float32)


class GraphTemporalConv(nn.Module):
    """Spatial graph conv + temporal conv with residual. Input: [B,C,T,V]."""
    def __init__(self, cin, cout, A: torch.Tensor, k=9, dropout=0.35):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = nn.Conv2d(cin, cout, 1)
        pad      = (k-1)//2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=(k,1), padding=(pad,0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout),
        )
        self.res  = (nn.Identity() if cin == cout
                     else nn.Sequential(nn.Conv2d(cin,cout,1), nn.BatchNorm2d(cout)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xg = torch.einsum("bctv,vw->bctw", x, self.A)
        return self.relu(self.tcn(self.gcn(xg)) + self.res(x))


class CompactSTGCN(nn.Module):
    """
    Skeleton encoder.
    Input : [B, C=3, T=32, V=17, M=6]
    Output: [B, embed_dim]
    """
    def __init__(self, C=3, embed_dim=128, V=17, M=6, dropout=0.35):
        super().__init__()
        A = coco17_adj(V)
        self.bn = nn.BatchNorm1d(C*V)
        self.g1 = GraphTemporalConv(C,   64,  A, dropout=dropout)
        self.g2 = GraphTemporalConv(64,  64,  A, dropout=dropout)
        self.g3 = GraphTemporalConv(64,  128, A, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.M = M

    def forward(self, x):
        # x: [B, C, T, V, M]
        B, C, T, V, M = x.shape
        # normalise per person
        xp = x.permute(0,4,3,1,2).contiguous().view(B*M, V*C, T)
        xp = self.bn(xp)
        xp = xp.view(B, M, V, C, T).permute(0,1,3,4,2).contiguous().view(B*M, C, T, V)
        xp = self.g3(self.g2(self.g1(xp)))          # [B*M, 128, T, V]
        xp = xp.mean(dim=[2,3]).view(B, M, -1).mean(dim=1)  # [B, 128]
        return self.proj(xp)

# ============================================================
# Cell 16 — Model: per-frame graph conv layers
# ============================================================

class FrameGATv2(nn.Module):
    """
    Single GATv2 layer operating on one frame's person graph.
    Input nodes : [B, M, node_dim]
    Input edges : [B, M, M, edge_dim]
    Input mask  : [B, M]  bool (True = valid node)
    Output      : [B, hidden]  — masked mean over valid nodes
    """
    def __init__(self, node_dim, edge_dim, hidden=64, dropout=0.35):
        super().__init__()
        self.hidden = hidden
        # project nodes and edges into attention space
        self.Wq  = nn.Linear(node_dim, hidden, bias=False)
        self.Wk  = nn.Linear(node_dim, hidden, bias=False)
        self.We  = nn.Linear(edge_dim,  hidden, bias=False)
        self.att = nn.Linear(hidden, 1, bias=False)
        self.Wv  = nn.Linear(node_dim, hidden, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes, edges, node_mask):
        """
        nodes     : [B, M, node_dim]
        edges     : [B, M, M, edge_dim]
        node_mask : [B, M]  bool
        returns   : [B, hidden]
        """
        B, M, _ = nodes.shape
        q = self.Wq(nodes)                     # [B, M, H]
        k = self.Wk(nodes)                     # [B, M, H]
        e = self.We(edges)                     # [B, M, M, H]
        v = self.Wv(nodes)                     # [B, M, H]

        # GATv2 attention score: a( LeakyReLU( W[h_i || h_j || e_ij] ) )
        q_exp = q.unsqueeze(2).expand(B, M, M, self.hidden)  # [B,M,M,H] (source)
        k_exp = k.unsqueeze(1).expand(B, M, M, self.hidden)  # [B,M,M,H] (target)
        attn  = self.att(self.act(q_exp + k_exp + e)).squeeze(-1)  # [B, M, M]

        # mask invalid target nodes: set their attention score to -inf
        invalid = ~node_mask.unsqueeze(1).expand(B, M, M)   # [B, M, M]
        attn    = attn.masked_fill(invalid, float("-inf"))

        # ── NaN guard ────────────────────────────────────────────────────────
        # When ALL nodes in a frame are invalid (no persons detected) every row
        # of `attn` is [-inf, -inf, …], and softmax produces NaN.  NaN then
        # propagates through bmm → LayerNorm → BiGRU → fusion → loss, silently
        # corrupting all parameters.
        #
        # Fix: before softmax, identify rows where every column is -inf and
        # replace them with 0.  softmax(0, 0, …) gives a uniform distribution,
        # which is numerically safe.  The contribution of these invalid source
        # rows is cancelled in the final masked mean (mask_f = 0 for them).
        all_invalid_row = invalid.all(dim=2, keepdim=True)   # [B, M, 1]  True where no valid targets
        attn = attn.masked_fill(all_invalid_row.expand_as(attn), 0.0)
        # ─────────────────────────────────────────────────────────────────────

        attn    = torch.softmax(attn, dim=2)                 # [B, M, M]
        attn    = self.drop(attn)

        # aggregate: for each source node i, weighted sum of neighbour values
        agg = torch.bmm(attn, v)                             # [B, M, H]

        # residual + norm
        out = self.norm(agg + v)                             # [B, M, H]

        # masked mean over valid nodes → [B, H]
        # Invalid source nodes have mask_f = 0 so their output is zeroed out,
        # consistent with FrameGINE's handling of empty frames.
        mask_f = node_mask.float().unsqueeze(-1)             # [B, M, 1]
        pooled = (out * mask_f).sum(1) / mask_f.sum(1).clamp_min(1)
        return pooled                                        # [B, H]


class FrameGINE(nn.Module):
    """
    Single GINE layer for object or PO graphs.

    Object mode (bipartite=False):
        nodes     : [B, N, node_dim]
        edges     : not used (fully connected within frame)
        node_mask : [B, N]
        → [B, hidden]

    PO mode (bipartite=True):
        Aggregates object information into person nodes via PO edges.
        person_nodes : [B, M, person_node_dim]
        object_nodes : [B, N, obj_node_dim]
        po_edges     : [B, M, N, edge_dim]
        po_mask      : [B, M, N]
        person_mask  : [B, M]
        → [B, hidden]  (masked mean over valid person nodes)
    """
    def __init__(self, node_dim, edge_dim, hidden=64, dropout=0.35,
                 bipartite=False, src_node_dim=None):
        super().__init__()
        self.bipartite = bipartite
        if bipartite:
            # project person nodes, object nodes, and PO edges
            assert src_node_dim is not None
            self.Wp  = nn.Linear(src_node_dim, hidden)   # person (target)
            self.Wo  = nn.Linear(node_dim,     hidden)   # object (source)
            self.We  = nn.Linear(edge_dim,     hidden)   # PO edge
        else:
            self.Wn  = nn.Linear(node_dim, hidden)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, nodes, edges=None, node_mask=None,
                person_nodes=None, po_mask=None, person_mask=None):
        if self.bipartite:
            return self._forward_po(nodes, edges, node_mask,
                                    person_nodes, po_mask, person_mask)
        return self._forward_obj(nodes, node_mask)

    def _forward_obj(self, nodes, node_mask):
        """Fully connected object graph — aggregate neighbors."""
        B, N, _ = nodes.shape
        h = self.Wn(nodes)                                # [B, N, H]
        # simple mean aggregation from all valid neighbors
        mask_f = node_mask.float().unsqueeze(-1)          # [B, N, 1]
        agg    = (h * mask_f).sum(1, keepdim=True) / mask_f.sum(1, keepdim=True).clamp_min(1)
        agg    = agg.expand(B, N, -1)                     # [B, N, H]
        out    = self.norm(h + self.drop(self.mlp(h + agg)))
        pooled = (out * mask_f).sum(1) / mask_f.sum(1).clamp_min(1)
        return pooled                                     # [B, H]

    def _forward_po(self, obj_nodes, po_edges, po_mask,
                    person_nodes, _, person_mask):
        """
        Bipartite: aggregate from object nodes into person nodes via PO edges.
        obj_nodes   : [B, N, obj_node_dim]
        po_edges    : [B, M, N, edge_dim]
        po_mask     : [B, M, N]
        person_nodes: [B, M, person_node_dim]
        person_mask : [B, M]
        """
        B, M, N, _ = po_edges.shape
        hp = self.Wp(person_nodes)                        # [B, M, H]
        ho = self.Wo(obj_nodes)                           # [B, N, H]
        he = self.We(po_edges)                            # [B, M, N, H]

        # message: for each person i, sum over valid object neighbors j
        ho_exp = ho.unsqueeze(1).expand(B, M, N, -1)     # [B, M, N, H]
        msg    = torch.relu(ho_exp + he)                  # [B, M, N, H]
        mask_f = po_mask.float().unsqueeze(-1)            # [B, M, N, 1]
        agg    = (msg * mask_f).sum(2) / mask_f.sum(2).clamp_min(1)  # [B, M, H]

        out    = self.norm(hp + self.drop(self.mlp(agg))) # [B, M, H]
        pm     = person_mask.float().unsqueeze(-1)        # [B, M, 1]
        pooled = (out * pm).sum(1) / pm.sum(1).clamp_min(1)
        return pooled                                     # [B, H]

# ============================================================
# Cell 17 — Model: temporal BiGRU with attention pooling
# ============================================================
class TemporalGRUEncoder(nn.Module):
    """
    Projects [B, T, in_dim] → [B, embed_dim] using BiGRU + learned attention.
    Identical to V4.2's TemporalGRUEncoder.
    """
    def __init__(self, in_dim, embed_dim=128, dropout=0.35):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.gru  = nn.GRU(embed_dim, embed_dim//2,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.Tanh(),
            nn.Linear(embed_dim//2, 1),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, T, in_dim]
        h  = self.proj(x)                          # [B, T, embed_dim]
        y, _ = self.gru(h)                         # [B, T, embed_dim]
        a  = self.attn(y).squeeze(-1)              # [B, T]
        w  = torch.softmax(a, dim=1).unsqueeze(-1) # [B, T, 1]
        return self.norm((y * w).sum(dim=1))       # [B, embed_dim]

# ============================================================
# Cell 18 — Model: quality-gated fusion (restored from V4.2)
# ============================================================
class QualityGatedFusion(nn.Module):
    """
    Four-stream quality-gated fusion.
    Takes four embeddings [B, embed_dim] and gqs [B, 5].
    Returns fused [B, embed_dim].

    During training applies stream dropout: each stream zeroed with
    probability stream_dropout_prob independently.
    """
    def __init__(self, embed_dim=128, n_streams=4, dropout=0.35,
                 stream_dropout_prob=0.25):
        super().__init__()
        self.n   = n_streams
        self.sdp = stream_dropout_prob
        # gate MLP: (4 × embed_dim + 5) → hidden → n_streams
        self.gate = nn.Sequential(
            nn.Linear(n_streams * embed_dim + 5, embed_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, n_streams),
        )
        self.post = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, embs: List[torch.Tensor], gqs: torch.Tensor):
        """
        embs : list of n tensors each [B, embed_dim]
        gqs  : [B, 5]
        """
        B    = gqs.shape[0]
        dev  = gqs.device
        stack = torch.stack(embs, dim=1)       # [B, n, embed_dim]

        # stream dropout during training
        if self.training and self.sdp > 0 and self.n > 1:
            mask = (torch.rand(B, self.n, device=dev) >= self.sdp).float()
            # ensure at least one stream active per sample
            all_off = (mask.sum(dim=1) == 0)
            if all_off.any():
                mask[all_off, 0] = 1.0
            stack = stack * mask.unsqueeze(-1)

        gate_in = torch.cat([stack.reshape(B, -1), gqs], dim=1)
        gates   = torch.softmax(self.gate(gate_in), dim=1)        # [B, n]
        fused   = (stack * gates.unsqueeze(-1)).sum(dim=1)        # [B, embed_dim]
        return self.post(fused), gates

# ============================================================
# Cell 19 — Full V8 model
# ============================================================
class V8Model(nn.Module):
    """
    Four-stream tri-graph violence detector.

    Stream 1 — Skeleton  : CompactSTGCN
    Stream 2 — Interaction: FrameGATv2 → TemporalGRUEncoder
    Stream 3 — Object    : FrameGINE   → TemporalGRUEncoder
    Stream 4 — PO        : FrameGINE   → TemporalGRUEncoder
    Fusion    : QualityGatedFusion (GQS-conditioned, stream dropout)
    Head      : Linear → ReLU → Dropout → Linear
    """
    def __init__(self,
                 C=3, T=32, V=17, M=6, N=8,
                 int_node_dim=7, int_edge_dim=4,
                 obj_node_dim=6, po_edge_dim=5,
                 graph_hidden=64, embed_dim=128, dropout=0.35,
                 stream_dropout_prob=0.25,
                 active_streams=("skeleton","interaction","object","po")):
        super().__init__()
        self.active = list(active_streams)

        # --- skeleton ---
        if "skeleton" in self.active:
            self.skel_enc = CompactSTGCN(
                C=C, embed_dim=embed_dim, V=V, M=M, dropout=dropout)

        # --- interaction ---
        if "interaction" in self.active:
            self.int_gat  = FrameGATv2(
                node_dim=int_node_dim, edge_dim=int_edge_dim,
                hidden=graph_hidden, dropout=dropout)
            self.int_gru  = TemporalGRUEncoder(
                in_dim=graph_hidden, embed_dim=embed_dim, dropout=dropout)

        # --- object ---
        if "object" in self.active:
            self.obj_gine = FrameGINE(
                node_dim=obj_node_dim, edge_dim=0,
                hidden=graph_hidden, dropout=dropout, bipartite=False)
            self.obj_gru  = TemporalGRUEncoder(
                in_dim=graph_hidden, embed_dim=embed_dim, dropout=dropout)

        # --- PO ---
        if "po" in self.active:
            self.po_gine  = FrameGINE(
                node_dim=obj_node_dim, edge_dim=po_edge_dim,
                hidden=graph_hidden, dropout=dropout,
                bipartite=True, src_node_dim=int_node_dim)
            self.po_gru   = TemporalGRUEncoder(
                in_dim=graph_hidden, embed_dim=embed_dim, dropout=dropout)

        n_active = len(self.active)
        self.fusion = QualityGatedFusion(
            embed_dim=embed_dim, n_streams=n_active,
            dropout=dropout, stream_dropout_prob=stream_dropout_prob)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, 1),
        )

    def _encode_interaction(self, b):
        """Run FrameGATv2 on each frame, stack, feed BiGRU."""
        int_nodes = b["int_nodes"]      # [B, T, M, 7]
        int_edges = b["int_edges"]      # [B, T, M, M, 4]
        int_nmask = b["int_node_mask"]  # [B, T, M]
        B, T, M, _ = int_nodes.shape
        frame_embs = []
        for t in range(T):
            fe = self.int_gat(
                int_nodes[:, t, :, :],   # [B, M, 7]
                int_edges[:, t, :, :, :],# [B, M, M, 4]
                int_nmask[:, t, :],      # [B, M]
            )                            # [B, graph_hidden]
            frame_embs.append(fe)
        seq = torch.stack(frame_embs, dim=1)  # [B, T, graph_hidden]
        return self.int_gru(seq)              # [B, embed_dim]

    def _encode_object(self, b):
        """Run FrameGINE on each frame's object nodes, stack, feed BiGRU."""
        obj_nodes = b["obj_nodes"]      # [B, T, N, 6]
        obj_nmask = b["obj_node_mask"]  # [B, T, N]
        B, T, N, _ = obj_nodes.shape
        frame_embs = []
        for t in range(T):
            fe = self.obj_gine(
                obj_nodes[:, t, :, :],   # [B, N, 6]
                node_mask=obj_nmask[:, t, :],  # [B, N]
            )                            # [B, graph_hidden]
            frame_embs.append(fe)
        seq = torch.stack(frame_embs, dim=1)  # [B, T, graph_hidden]
        return self.obj_gru(seq)              # [B, embed_dim]

    def _encode_po(self, b):
        """Run bipartite FrameGINE on each frame's PO graph, stack, feed BiGRU."""
        int_nodes = b["int_nodes"]      # [B, T, M, 7]
        obj_nodes = b["obj_nodes"]      # [B, T, N, 6]
        po_edges  = b["po_edges"]       # [B, T, M, N, 5]
        po_mask   = b["po_edge_mask"]   # [B, T, M, N]
        int_nmask = b["int_node_mask"]  # [B, T, M]
        B, T, M, N, _ = po_edges.shape
        frame_embs = []
        for t in range(T):
            fe = self.po_gine(
                nodes=obj_nodes[:, t, :, :],      # obj as source nodes
                edges=po_edges[:, t, :, :, :],    # PO edges
                node_mask=po_mask[:, t, :, :],    # PO edge mask [B,M,N]
                person_nodes=int_nodes[:, t, :, :],  # persons as target
                person_mask=int_nmask[:, t, :],
            )                                     # [B, graph_hidden]
            frame_embs.append(fe)
        seq = torch.stack(frame_embs, dim=1)      # [B, T, graph_hidden]
        return self.po_gru(seq)                   # [B, embed_dim]

    def forward(self, b):
        embs = []
        if "skeleton"    in self.active:
            embs.append(self.skel_enc(b["skeleton"]))
        if "interaction" in self.active:
            embs.append(self._encode_interaction(b))
        if "object"      in self.active:
            embs.append(self._encode_object(b))
        if "po"          in self.active:
            embs.append(self._encode_po(b))

        fused, gates = self.fusion(embs, b["gqs"])
        logit = self.head(fused).squeeze(-1)
        return logit, gates

# ============================================================
# Cell 20 — Metrics and helpers
# ============================================================
def move_to_device(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}

def compute_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs  = np.clip(np.asarray(probs).astype(float), 0, 1)
    preds  = (probs >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, preds, labels=[0,1])
    return {
        "threshold": float(threshold),
        "accuracy":  float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "macro_f1":  float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "auc":       float(auc),
        "cm":        cm.tolist(),
    }

def best_threshold(y_true, probs,
                   grid=np.round(np.arange(0.05, 0.96, 0.01), 2)):
    best_t, best_m, best_r = None, -1, None
    for t in grid:
        r = compute_metrics(y_true, probs, t)
        if r["macro_f1"] > best_m:
            best_t, best_m, best_r = float(t), r["macro_f1"], r
    return best_t, best_r

@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps, vids, gate_rows = [], [], [], []
    for b in loader:
        b     = move_to_device(b)
        logit, gates = model(b)
        prob  = torch.sigmoid(logit).cpu().numpy()
        y     = b["label"].cpu().numpy().astype(int)
        g     = gates.cpu().numpy()
        ys.extend(y.tolist()); ps.extend(prob.tolist())
        vids.extend(b["video_id"])
        for i, vid in enumerate(b["video_id"]):
            gate_rows.append({"video_id": vid,
                               **{f"gate_{s}": float(g[i,k])
                                  for k,s in enumerate(model.active)}})
    return np.array(ys), np.array(ps), vids, pd.DataFrame(gate_rows)

# ============================================================
# Cell 21 — Training loop
# ============================================================
EXPERIMENTS = {
    "A_skeleton_only":   ("skeleton",),
    "B_skel_obj":        ("skeleton","object"),
    "C_skel_int":        ("skeleton","interaction"),
    "D_skel_obj_po":     ("skeleton","object","po"),
    "E_full_trigraph":   ("skeleton","interaction","object","po"),
}

def train_experiment(exp_name: str, active_streams: tuple, seed: int = 42):
    seed_everything(seed)
    run_dir = RUN_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = V8Model(
        C=3, T=cfg.target_frames, V=cfg.num_joints,
        M=cfg.max_persons, N=cfg.max_objects,
        int_node_dim=cfg.int_node_dim, int_edge_dim=cfg.int_edge_dim,
        obj_node_dim=cfg.obj_node_dim, po_edge_dim=cfg.po_edge_dim,
        graph_hidden=cfg.graph_hidden, embed_dim=cfg.embed_dim,
        dropout=cfg.dropout, stream_dropout_prob=cfg.stream_dropout_prob,
        active_streams=active_streams,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{exp_name}] Parameters: {total_params:,}")

    # ── Fix A: per-stream learning rate ──────────────────────────────────────
    # Multi-stream models have higher capacity and overfit faster on the 700-
    # sample Hockey Fights training set.  A lower lr slows memorisation without
    # changing model capacity.  Single-stream (skeleton-only) retains cfg.lr.
    n_streams = len(active_streams)
    effective_lr = cfg.lr if n_streams == 1 else cfg.lr_multi_stream
    print(f"[{exp_name}] Streams: {n_streams}  LR: {effective_lr:.1e}")

    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=effective_lr, weight_decay=cfg.weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor,
        patience=cfg.lr_patience, verbose=True)

    # ── Fix B: label-smoothed BCE loss ───────────────────────────────────────
    # BCEWithLogitsLoss does not support label_smoothing natively.
    # Smooth targets manually: y_smooth = y*(1-α) + α/2
    # α = cfg.label_smoothing = 0.03
    # Class-1 target: 1.0 → 0.985   Class-0 target: 0.0 → 0.015
    _bce_raw = nn.BCEWithLogitsLoss(reduction="mean")
    _alpha   = cfg.label_smoothing

    def loss_fn(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        smooth = label * (1.0 - _alpha) + 0.5 * _alpha
        return _bce_raw(logit, smooth)

    best_score = -1.0
    best_epoch = -1
    patience   = cfg.patience
    best_path  = run_dir / "best_model.pt"
    history    = []

    for epoch in range(1, cfg.epochs + 1):
        # ── Fix C: min_checkpoint_epoch ──────────────────────────────────────
        # Do not save checkpoints or decrement patience in the warm-up period.
        # This prevents a lucky first epoch from becoming the reported result.
        checkpoint_allowed = (epoch >= cfg.min_checkpoint_epoch)

        # --- train ---
        model.train()
        losses    = []
        nan_count = 0   # Fix D: NaN tracking
        for b in tqdm(train_loader, desc=f"[{exp_name}] epoch {epoch:03d}",
                      leave=False):
            b = move_to_device(b)
            optimizer.zero_grad(set_to_none=True)
            logit, _ = model(b)
            loss = loss_fn(logit, b["label"])

            # ── Fix D: NaN guard in training loop ────────────────────────────
            # If a corrupted batch somehow produces NaN loss (e.g., extreme
            # gradient values before the NaN fix propagated), skip it rather
            # than corrupting the model weights.
            if not torch.isfinite(loss):
                nan_count += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if nan_count > 0:
            print(f"[{exp_name}] epoch={epoch:03d}  WARNING: {nan_count} NaN/Inf "
                  f"batches skipped out of {len(train_loader)} total")

        avg_loss = float(np.mean(losses)) if losses else float("nan")

        # --- validate ---
        va_y, va_p, _, _ = predict(model, val_loader)
        va_m = compute_metrics(va_y, va_p, 0.5)
        score = va_m["macro_f1"]

        scheduler.step(score)

        row = {
            "epoch":              epoch,
            "loss":               avg_loss,
            "val_macro_f1":       va_m["macro_f1"],
            "val_auc":            va_m["auc"],
            "val_accuracy":       va_m["accuracy"],
            "score":              score,
            "lr":                 float(optimizer.param_groups[0]["lr"]),
            "checkpoint_allowed": checkpoint_allowed,
            "nan_batches":        nan_count,
        }
        history.append(row)
        print(f"[{exp_name}] epoch={epoch:03d} "
              f"loss={avg_loss:.4f} "
              f"val_macro_f1={va_m['macro_f1']:.4f} "
              f"val_auc={va_m['auc']:.4f} "
              f"lr={row['lr']:.2e} "
              f"ckpt={'YES' if checkpoint_allowed else f'frozen(<{cfg.min_checkpoint_epoch})'}")

        # ── Fix C continued: only checkpoint / count patience after warm-up ──
        if checkpoint_allowed:
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save({"state_dict": model.state_dict(),
                            "active":    active_streams,
                            "epoch":     epoch}, best_path)
                patience = cfg.patience
            else:
                patience -= 1
                if patience <= 0:
                    print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                    break

    # --- final evaluation ---
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    tr_y, tr_p, _,  _       = predict(model, train_loader)
    va_y, va_p, _,  _       = predict(model, val_loader)
    te_y, te_p, te_vids, te_gates = predict(model, test_loader)

    val_t,    val_best  = best_threshold(va_y, va_p)
    oracle_t, oracle_m  = best_threshold(te_y, te_p)

    result = {
        "experiment":           exp_name,
        "active_streams":       list(active_streams),
        "best_epoch":           int(best_epoch),
        "total_params":         total_params,
        "test_at_0_50":         compute_metrics(te_y, te_p, 0.50),
        "test_at_val_threshold":compute_metrics(te_y, te_p, val_t),
        "val_selected_threshold": float(val_t),
        "test_oracle_analysis_only": oracle_m,
        "test_oracle_threshold_analysis_only": float(oracle_t),
    }

    pd.DataFrame(history).to_csv(run_dir/"history.csv", index=False)
    pd.DataFrame({
        "video_id":        te_vids,
        "y_true":          te_y,
        "prob":            te_p,
        "pred_0_50":       (te_p >= 0.50).astype(int),
        "pred_val_thresh": (te_p >= val_t).astype(int),
    }).to_csv(run_dir/"test_predictions.csv", index=False)
    te_gates.to_csv(run_dir/"test_gates.csv", index=False)

    with open(run_dir/"result.json","w") as f:
        json.dump(result, f, indent=4)

    print(f"\n{'='*60}")
    print(f"DONE: {exp_name}")
    print(f"Test @0.50 : {result['test_at_0_50']}")
    print(f"Test @val_t: {result['test_at_val_threshold']}")
    print(f"{'='*60}\n")
    return result

# ============================================================
# Cell 22 — Run all experiments
# ============================================================
all_results = []
for exp_name, streams in EXPERIMENTS.items():
    print("="*70)
    print(f"Running: {exp_name}  streams={streams}")
    print("="*70)
    all_results.append(train_experiment(exp_name, streams, cfg.seed))

# ============================================================
# Cell 23 — Comparison table
# ============================================================
rows = []
for r in all_results:
    m05  = r["test_at_0_50"]
    mvt  = r["test_at_val_threshold"]
    rows.append({
        "experiment":           r["experiment"],
        "streams":              "+".join(r["active_streams"]),
        "params":               r["total_params"],
        "best_epoch":           r["best_epoch"],
        # official results at threshold 0.50
        "acc_0_50":             m05["accuracy"],
        "macro_f1_0_50":        m05["macro_f1"],
        "f1_0_50":              m05["f1"],
        "auc":                  m05["auc"],
        "cm_0_50":              str(m05["cm"]),
        # results at val-selected threshold
        "threshold_val":        r["val_selected_threshold"],
        "acc_val_t":            mvt["accuracy"],
        "macro_f1_val_t":       mvt["macro_f1"],
        "cm_val_t":             str(mvt["cm"]),
    })

cmp_df = pd.DataFrame(rows).sort_values("macro_f1_0_50", ascending=False)
cmp_df.to_csv(OUT_ROOT/"v8_experiment_comparison.csv", index=False)

print("\nV8 RESULTS SUMMARY")
print(cmp_df.to_string(index=False))
print("\nV4.2 target: Acc=0.955 | Macro-F1=0.9550 | AUC=0.9658 | CM=[[97,3],[6,94]]")
