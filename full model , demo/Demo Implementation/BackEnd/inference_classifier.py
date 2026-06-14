"""
Guardian Eye — V9 Classifier Inference

Loads the E_full_qgf checkpoint, runs one forward pass per clip, and returns
verdict + calibrated confidence + gate[4] + gqs[5] + telemetry.

VRAM lifecycle (RTX 3090):
  - Model lives on CPU between calls.
  - Moved to CUDA just before forward, moved back after, cache cleared.
  - This leaves VRAM free for Qwen2.5-VL-7B when /explain is called.

Env vars:
  GUARDIAN_V9_CKPT    path to E_full_qgf/best.pt  (required in real mode)
  GUARDIAN_CAL_TEMP   calibration temperature for RLVS (default 1.0 — update
                      after confirming the value from the WS training log)
  GUARDIAN_THRESHOLD  float override; falls back to ck["threshold"] (RLVS: 0.28)
"""

from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# COCO-80 class names (index → name) used for weapon_cls extraction
_COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

# ── Module-level singletons ────────────────────────────────────────────────────
_v9_model    = None
_v9_threshold: float = 0.28   # RLVS default; overridden by checkpoint
_cal_temp:     float = 1.0


def load_v9_model() -> None:
    """
    Load the V9 E_full_qgf checkpoint into the module singleton.
    Call once at FastAPI startup when GUARDIAN_MOCK=0.
    """
    global _v9_model, _v9_threshold, _cal_temp

    ckpt_path = os.getenv("GUARDIAN_V9_CKPT", "")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise RuntimeError(
            "GUARDIAN_V9_CKPT is not set or the file does not exist. "
            "Set it to the path of E_full_qgf/best.pt before starting in real mode."
        )

    _cal_temp = float(os.getenv("GUARDIAN_CAL_TEMP", "1.0"))

    from v9_model import V9Model, compute_bones

    model = V9Model(
        active_streams=("skeleton", "interaction", "object", "vit"),
        fusion_mode="qgf",
        C=12,   # multi-input: joint+bone+joint_motion+bone_motion (4×3)
        T=32, V=17, M=6, N=8,
        int_nd=7, int_ed=4, obj_nd=6, po_ed=5,
        vit_dim=768, hidden=64, embed_dim=128,
        dropout=0.35, stream_dropout=0.0,   # no stream dropout at inference
        n_heads_gat=4, n_heads_trf=4, n_layers_trf=2,
    )

    print(f"[classifier] Loading checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    thr_override = os.getenv("GUARDIAN_THRESHOLD", "")
    if thr_override:
        _v9_threshold = float(thr_override)
    elif "threshold" in ck:
        _v9_threshold = float(ck["threshold"])

    _v9_model = model
    print(f"[classifier] Ready  threshold={_v9_threshold:.3f}  cal_temp={_cal_temp:.3f}")


def _apply_calibration(raw_prob: float, temp: float) -> float:
    """Temperature scaling: re-calibrates sigmoid output."""
    if abs(temp - 1.0) < 1e-6:
        return raw_prob
    eps = 1e-7
    p   = max(eps, min(1 - eps, raw_prob))
    logit = math.log(p / (1 - p)) / max(temp, 1e-3)
    return 1.0 / (1.0 + math.exp(-logit))


def _build_batch(d: "np.lib.npyio.NpzFile") -> dict:
    """Convert NPZ arrays into a torch batch (B=1) ready for the model."""
    from v9_model import compute_bones

    skel  = d["skeleton"].astype(np.float32)          # [T, M, V, 3]
    bone  = compute_bones(skel)
    jm    = np.zeros_like(skel);  jm[:-1] = skel[1:] - skel[:-1]
    bm    = np.zeros_like(bone);  bm[:-1] = bone[1:] - bone[:-1]
    skel12 = np.concatenate([skel, bone, jm, bm], axis=-1).astype(np.float32)  # [T,M,V,12]

    if "vit_embedding" in d:
        vit_emb = torch.from_numpy(d["vit_embedding"].astype(np.float32))
    else:
        vit_emb = torch.zeros(768, dtype=torch.float32)

    return {
        "skeleton":      torch.from_numpy(skel12).unsqueeze(0),          # [1,T,M,V,12]
        "int_nodes":     torch.from_numpy(d["int_nodes"].astype(np.float32)).unsqueeze(0),
        "int_edges":     torch.from_numpy(d["int_edges"].astype(np.float32)).unsqueeze(0),
        "int_node_mask": torch.from_numpy(d["int_node_mask"]).unsqueeze(0),
        "int_edge_mask": torch.from_numpy(d["int_edge_mask"]).unsqueeze(0),
        "obj_nodes":     torch.from_numpy(d["obj_nodes"].astype(np.float32)).unsqueeze(0),
        "obj_node_mask": torch.from_numpy(d["obj_node_mask"]).unsqueeze(0),
        "po_edges":      torch.from_numpy(d["po_edges"].astype(np.float32)).unsqueeze(0),
        "po_edge_mask":  torch.from_numpy(d["po_edge_mask"]).unsqueeze(0),
        "gqs":           torch.from_numpy(d["gqs"].astype(np.float32)).unsqueeze(0),
        "vit_embedding": vit_emb.unsqueeze(0),
    }


def _derive_telemetry(d: "np.lib.npyio.NpzFile") -> dict:
    """
    Deterministic telemetry from NPZ arrays — no model inference.
    Matches the spec from EXPLANATION_RAG_SYSTEM.md §4:
      - people:       max simultaneously valid persons in any frame
      - peak_window:  4-frame window with highest sum of 'close' interaction flags
      - weapon_flag:  any frame where po_edge near_wrist flag fires
      - weapon_cls:   COCO class at the triggering (t, n) location
    """
    int_node_mask = d["int_node_mask"]    # [T, M] bool
    int_edges     = d["int_edges"]        # [T, M, M, 4]  close flag is index 2
    po_edges      = d["po_edges"]         # [T, M, N, 5]  near_wrist is index 3
    po_edge_mask  = d["po_edge_mask"]     # [T, M, N] bool
    obj_nodes     = d["obj_nodes"]        # [T, N, 6]  cls_norm is index 5

    T_local = int_node_mask.shape[0]

    # People count
    people = int(int_node_mask.sum(axis=1).max())

    # Peak window: slide 4-frame window over sum of valid 'close' flags
    close_flags = int_edges[:, :, :, 2]    # [T, M, M]
    window = 4
    best_start, best_score = 0, -1.0
    for start in range(T_local - window + 1):
        score = float(close_flags[start:start + window].sum())
        if score > best_score:
            best_score = score
            best_start = start
    peak_window = [best_start, min(best_start + window - 1, T_local - 1)]

    # Weapon detection: look for near_wrist flag in po_edges
    weapon_flag = False
    weapon_cls  = None
    near_wrist_mask = po_edges[:, :, :, 3] > 0.5  # [T, M, N] float → bool
    triggered = po_edge_mask & near_wrist_mask      # [T, M, N]
    if triggered.any():
        weapon_flag = True
        # Find the first (t, m, n) that fired
        t_idx, m_idx, n_idx = np.argwhere(triggered)[0]
        cls_norm = float(obj_nodes[t_idx, n_idx, 5])
        cls_id   = int(round(cls_norm * 79.0))
        if 0 <= cls_id < len(_COCO80):
            weapon_cls = _COCO80[cls_id]
        else:
            weapon_cls = "unknown"

    return {
        "people":       max(people, 0),
        "peak_window":  peak_window,
        "weapon": {
            "flag": weapon_flag,
            "cls":  weapon_cls,
        },
    }


def classifier_forward(npz_path: str, clip_id: str) -> dict:
    """
    Load NPZ, run V9 forward pass, derive telemetry.

    VRAM lifecycle:
      model.to("cuda") → forward → model.to("cpu") → empty_cache()

    Returns dict compatible with model_service._real_predict() contract.
    """
    if _v9_model is None:
        raise RuntimeError(
            "V9 model not loaded. Call load_v9_model() first, "
            "or set GUARDIAN_MOCK=1 to use mock mode."
        )

    # Load NPZ
    d = np.load(npz_path, allow_pickle=False)

    # Build batch
    batch = _build_batch(d)

    # Move model + batch to GPU, run forward, move model back
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _v9_model.to(device)

    batch_gpu = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.no_grad():
        logits, gates = _v9_model(batch_gpu, return_gates=True)

    raw_prob = torch.sigmoid(logits).item()
    gates_np = gates.squeeze(0).cpu().numpy()   # [4]

    # Move model back to CPU and free VRAM
    _v9_model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    # Calibrate confidence
    conf      = _apply_calibration(raw_prob, _cal_temp)
    threshold = _v9_threshold
    verdict   = "violence" if conf >= threshold else "non-violence"

    # Gate weights (order: skeleton, interaction, object, vit)
    gate_dict = {
        "skeleton":    float(gates_np[0]),
        "interaction": float(gates_np[1]),
        "object":      float(gates_np[2]),
        "vit":         float(gates_np[3]),
    }

    # GQS from NPZ
    gqs_arr = d["gqs"].astype(np.float32)
    gqs_dict = {
        "q_skel":      float(gqs_arr[0]),
        "q_int":       float(gqs_arr[1]),
        "q_obj":       float(gqs_arr[2]),
        "q_po":        float(gqs_arr[3]),
        "valid_ratio": float(gqs_arr[4]),
    }

    # Telemetry
    telemetry = _derive_telemetry(d)

    print(
        f"[classifier] {clip_id}  verdict={verdict}  "
        f"conf={conf:.3f} (raw={raw_prob:.3f})  thr={threshold:.2f}  "
        f"gate={[f'{v:.2f}' for v in gates_np.tolist()]}"
    )

    return {
        "verdict":    verdict,
        "confidence": conf,
        "threshold":  threshold,
        "gate":       gate_dict,
        "gqs":        gqs_dict,
        "telemetry":  telemetry,
    }
