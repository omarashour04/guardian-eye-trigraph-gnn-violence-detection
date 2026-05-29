"""
trigraph_v9_ubi_train.py
Guardian Eye — V9  |  Phase 3: Model Training + Ablation Study
Dataset : UBI-Fights
GPU     : L4  |  ETA ~4 hours (7 experiments × ~35 min each)

Architecture (V9)
──────────────────
Stream 1 — Enhanced Multi-Input STGCN
  Input    : skeleton [T, M, V, 3] → derive joint+bone+joint_motion+bone_motion
             → C=12 channels
  Spatial  : 3 × GraphTemporalConv [12→64→64→128]
             adaptive adjacency A = A_fixed(COCO-17) + softmax(A_learned)
  Pooling  : learned attention over M persons (suppresses bystanders)
  Output   : e_skel [B, 128]

Stream 2 — Improved Interaction (temporal transformer replaces BiGRU)
  Input    : int_nodes [T, M, 7], int_edges [T, M, M, 4], int_node_mask [T, M]
  Per-frame: 2-layer GATv2 → frame embedding [B, 128] per timestep
  Temporal : 2-layer transformer encoder (4 heads, learned pos encoding)
             over sequence [B, T, 128] → attention-pooled → [B, 128]
  Output   : e_int [B, 128]

Stream 3 — Object/PO (unchanged from V8)
  FrameGINE + BiGRU on obj_nodes, obj_node_mask
  Bipartite FrameGINE + BiGRU on po_edges
  Combined → [B, 128]

Stream 4 — VideoMAE Projection (frozen fine-tuned ViT embedding)
  Input    : vit_embedding [B, 768]  (precomputed, loaded from NPZ)
  MLP      : Linear(768→256) → LayerNorm → ReLU → Linear(256→128)
  Output   : e_vit [B, 128]

Fusion    — QualityGatedFusion
  Gate input: gqs [B, 5] → MLP(5→32→4) → softmax → stream weights
  Stream dropout p=0.25 during training
  Fused [B, 128] → Head Linear(128→128) → ReLU → Dropout → Linear(128→1)

Ablation experiments
─────────────────────
  A  — Enhanced STGCN only
  B  — STGCN + improved interaction
  C  — VideoMAE only            (standalone result from Phase 2)
  D  — STGCN + interaction + object/PO   (graph-only full model)
  E  — All 4 streams            (full V9 model)
  E-QGF — E with QualityGatedFusion      (GQS-conditioned gates)
  E-LW  — E with learned static weights  (no GQS input; control for E-QGF)

Usage
─────
  modal run trigraph_v9_ubi_train.py::train
  modal run trigraph_v9_ubi_train.py::results
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import modal

# ── Modal primitives ──────────────────────────────────────────────────────────
APP_NAME      = "guardian-eye-v9-ubi-train"
VOL_NAME_PROC = "ubi-fights-processed"

app      = modal.App(APP_NAME)
vol_proc = modal.Volume.from_name(VOL_NAME_PROC, create_if_missing=True)

PROC_MOUNT = Path("/data/proc")
CACHE_DIR  = PROC_MOUNT / "cache_v9"
SPLIT_CSV  = PROC_MOUNT / "split_ubi.csv"
RUN_DIR    = PROC_MOUNT / "runs_ubi"

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "torch-geometric==2.5.3",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "tqdm==4.66.4",
        "scipy==1.13.0",
    )
)


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class CFG:
    # ── Data dimensions (must match Phase 1 constants) ────────────────────────
    T:   int = 32     # graph frames
    M:   int = 6      # max persons
    N:   int = 8      # max objects
    V:   int = 17     # COCO joints
    C:   int = 12     # multi-input skeleton channels (4 streams × 3)

    # ── Model ─────────────────────────────────────────────────────────────────
    embed_dim:      int   = 128
    graph_hidden:   int   = 64
    n_heads_gat:    int   = 4
    n_heads_trf:    int   = 4
    n_layers_trf:   int   = 2
    vit_embed_dim:  int   = 768   # VideoMAE-Base CLS dimension (mean-pool over patches)
    dropout:        float = 0.35
    stream_dropout: float = 0.25

    # ── Training ──────────────────────────────────────────────────────────────
    lr:              float = 3e-4    # single-stream experiments
    lr_multi:        float = 1e-4    # multi-stream experiments
    weight_decay:    float = 1e-3
    grad_clip:       float = 1.0
    label_smoothing: float = 0.01
    use_label_smoothing: bool = True
    pos_weight:      float = 2.284   # UBI-Fights train imbalance 2957 neg / 1295 pos = 2.284 (post-tiling)
    batch_size:      int   = 64
    num_workers:     int   = 0      # in-memory dataset; workers waste CPU on 2-core container
    epochs:          int   = 60
    patience:        int   = 12
    min_ckpt_epoch:  int   = 5

    # ── Threshold search ──────────────────────────────────────────────────────
    thr_min:  float = 0.05   # widened lower bound — model logits skew negative
    thr_max:  float = 0.90
    thr_step: float = 0.01   # finer step: 0.02 missed the 0.12 optimum from diagnostics

    # ── Hard-negative exclusion ───────────────────────────────────────────────
    # Populate after first error_analysis run if OOD clips are identified.
    hard_negative_video_stems: tuple = ()

    # ── Resume ─────────────────────────────────────────────────────────────────
    # When True, any experiment whose test_metrics.json already exists on the
    # volume is skipped (its result is reloaded into the comparison table).
    # Lets a run that died mid-loop (e.g. local SSL drop) pick up where it left
    # off without retraining completed experiments. Set False to force a fresh
    # run of all experiments.
    resume: bool = True

    seed: int = 42


cfg = CFG()


# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset_class():
    """
    Returns UBIDataset.
    Loads all arrays from NPZ into RAM.
    Derives multi-input skeleton (bone, motion) at load time to avoid
    repeating this computation during every forward pass.
    """
    import torch
    from torch.utils.data import Dataset
    import numpy as np
    from tqdm import tqdm

    class UBIDataset(Dataset):
        def __init__(self, npz_paths: List[Path], labels: List[int]):
            self.samples = []
            print(f"  Loading {len(npz_paths)} samples …")
            missing_vit = 0
            for path, label in tqdm(zip(npz_paths, labels),
                                    total=len(npz_paths),
                                    desc="Loading", unit="file"):
                try:
                    d    = np.load(str(path), allow_pickle=True)
                    skel = d["skeleton"]   # [T, M, V, 3]

                    # ── Derive multi-input channels at load time ──────────
                    # Joint:        original (x, y, conf) → C=3
                    joint        = skel.copy()                     # [T,M,V,3]
                    # Bone:         edge vector + min-conf per COCO-17 edge
                    bone         = _compute_bones(skel)            # [T,M,V,3]
                    # Joint motion: Δjoint across frames (zero-pad at T-1)
                    jm           = np.zeros_like(skel)
                    jm[:-1]      = skel[1:] - skel[:-1]
                    # Bone motion:  Δbone across frames
                    bm           = np.zeros_like(bone)
                    bm[:-1]      = bone[1:] - bone[:-1]
                    # Stack → [T, M, V, 12]
                    skel12       = np.concatenate(
                        [joint, bone, jm, bm], axis=-1).astype(np.float32)

                    # ── ViT embedding ─────────────────────────────────────
                    if "vit_embedding" in d:
                        vit_emb = torch.from_numpy(
                            d["vit_embedding"].astype(np.float32))
                    else:
                        # Phase 2 not yet run; use zeros as placeholder
                        vit_emb = torch.zeros(cfg.vit_embed_dim,
                                              dtype=torch.float32)
                        missing_vit += 1

                    self.samples.append({
                        "skeleton":      torch.from_numpy(skel12),
                        "int_nodes":     torch.from_numpy(
                            d["int_nodes"].astype(np.float32)),
                        "int_edges":     torch.from_numpy(
                            d["int_edges"].astype(np.float32)),
                        "int_node_mask": torch.from_numpy(
                            d["int_node_mask"]),
                        "int_edge_mask": torch.from_numpy(
                            d["int_edge_mask"]),
                        "obj_nodes":     torch.from_numpy(
                            d["obj_nodes"].astype(np.float32)),
                        "obj_node_mask": torch.from_numpy(
                            d["obj_node_mask"]),
                        "po_edges":      torch.from_numpy(
                            d["po_edges"].astype(np.float32)),
                        "po_edge_mask":  torch.from_numpy(
                            d["po_edge_mask"]),
                        "gqs":           torch.from_numpy(
                            d["gqs"].astype(np.float32)),
                        "vit_embedding": vit_emb,
                        "label":         torch.tensor(float(label),
                                                      dtype=torch.float32),
                        "video_id":      str(path.stem),
                    })
                except Exception as e:
                    print(f"  WARN: {path}: {e}")

            if missing_vit > 0:
                print(f"  WARN: {missing_vit} samples have no vit_embedding "
                      f"(Phase 2 not yet run). ViT stream will produce zeros.")
            print(f"  Loaded {len(self.samples)} samples.")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    def _compute_bones(skel: np.ndarray) -> np.ndarray:
        """
        Compute bone vectors from COCO-17 kinematic tree.
        skel: [T, M, V, 3]  (x, y, conf)
        Returns: [T, M, V, 3]  (Δx, Δy, min_conf)  per joint's outgoing bone.
        Joints with no outgoing edge (terminal) get zeros.
        """
        # COCO-17 parent→child pairs (0-indexed)
        EDGES = [
            (0,1),(0,2),(1,3),(2,4),
            (5,7),(7,9),(6,8),(8,10),
            (5,6),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),
        ]
        T_, M_, V_, _ = skel.shape
        bones = np.zeros_like(skel)
        for (p, c) in EDGES:
            dx  = skel[:, :, c, 0] - skel[:, :, p, 0]   # [T, M]
            dy  = skel[:, :, c, 1] - skel[:, :, p, 1]
            cf  = np.minimum(skel[:, :, c, 2], skel[:, :, p, 2])
            bones[:, :, p, 0] = dx
            bones[:, :, p, 1] = dy
            bones[:, :, p, 2] = cf
        return bones

    return UBIDataset


# ══════════════════════════════════════════════════════════════════════════════
# Model components
# ══════════════════════════════════════════════════════════════════════════════
def build_model_components():
    """
    Returns a dict of all model component classes.
    All imports of torch/nn are local to this function to avoid top-level
    import conflicts on the Modal worker.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    # ─────────────────────────────────────────────────────────────────────────
    # Stream 1 — Enhanced Multi-Input STGCN
    # ─────────────────────────────────────────────────────────────────────────

    class GraphTemporalConv(nn.Module):
        """
        One STGCN block.
        Spatial: 1×1 GCN with adaptive adjacency.
        Temporal: depth-wise TCN across time axis.
        Residual connection with projection if channel dimensions differ.
        """
        def __init__(self, in_c: int, out_c: int,
                     t_kernel: int = 9, dropout: float = 0.35):
            super().__init__()
            self.gcn  = nn.Conv2d(in_c, out_c, 1)
            self.tcn  = nn.Conv2d(out_c, out_c,
                                  kernel_size=(t_kernel, 1),
                                  padding=(t_kernel // 2, 0),
                                  groups=out_c)    # depthwise over time
            self.tcn_pw = nn.Conv2d(out_c, out_c, 1)  # pointwise after dw
            self.bn   = nn.BatchNorm2d(out_c)
            self.drop = nn.Dropout(dropout)
            self.res  = (nn.Conv2d(in_c, out_c, 1)
                         if in_c != out_c else nn.Identity())

        def forward(self, x: torch.Tensor,
                    A: torch.Tensor) -> torch.Tensor:
            # x: [B, C, T, V]   A: [V, V]
            xs = torch.einsum("bctv,vw->bctw", x, A)
            xs = self.gcn(xs)
            xs = self.tcn_pw(F.relu(self.tcn(xs)))
            xs = self.bn(xs)
            return self.drop(F.relu(xs + self.res(x)))

    class EnhancedSTGCN(nn.Module):
        """
        Stream 1: Multi-input STGCN with adaptive adjacency
        and learned person-attention pooling.
        Input: [B, T, M, V, C=12]
        Output: e_skel [B, embed_dim]
        """
        def __init__(self, C: int, T: int, V: int, M: int,
                     hidden: int, embed_dim: int, dropout: float):
            super().__init__()
            # Fixed COCO-17 adjacency (degree-normalised)
            A_fix = self._make_coco17_adj(V)
            self.register_buffer("A_fixed", A_fix)
            # Learnable correction: A = A_fixed + softmax(A_learned)
            self.A_learned = nn.Parameter(torch.zeros(V, V))

            self.blocks = nn.ModuleList([
                GraphTemporalConv(C,      hidden,    dropout=dropout),
                GraphTemporalConv(hidden, hidden,    dropout=dropout),
                GraphTemporalConv(hidden, embed_dim, dropout=dropout),
            ])
            # Person attention: scalar importance score per person slot
            self.person_attn = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            self.out_norm = nn.LayerNorm(embed_dim)

        @staticmethod
        def _make_coco17_adj(V: int) -> torch.Tensor:
            edges = [
                (0,1),(0,2),(1,3),(2,4),
                (5,6),(5,7),(6,8),(7,9),(8,10),
                (5,11),(6,12),(11,12),
                (11,13),(12,14),(13,15),(14,16),
            ]
            A = torch.zeros(V, V)
            for i, j in edges:
                A[i, j] = A[j, i] = 1.0
            A.fill_diagonal_(1.0)
            D = A.sum(-1, keepdim=True).clamp(min=1.0).sqrt()
            return A / D / D.T

        def forward(self, x: torch.Tensor,
                    node_mask: torch.Tensor) -> torch.Tensor:
            # x: [B, T, M, V, C]   node_mask: [B, T, M]
            B, T, M, V, C = x.shape
            # Adaptive adjacency
            A = self.A_fixed + F.softmax(self.A_learned, dim=-1)
            # Per-person graph: treat each person independently
            # Reshape → [B*M, C, T, V]
            xr = x.permute(0, 2, 4, 1, 3).reshape(B * M, C, T, V)
            for blk in self.blocks:
                xr = blk(xr, A)
            # Mean-pool over T and V → [B*M, embed_dim]
            emb = xr.mean(dim=[2, 3])          # [B*M, embed_dim]
            emb = emb.view(B, M, -1)           # [B, M, embed_dim]
            # Mask invalid person slots
            valid = node_mask.any(dim=1)       # [B, M]  any frame valid
            emb_masked = emb * valid.unsqueeze(-1).float()
            # Learned attention pooling over persons
            scores = self.person_attn(emb_masked).squeeze(-1)  # [B, M]
            scores = scores.masked_fill(~valid, -1e9)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, M, 1]
            pooled  = (emb_masked * weights).sum(1)            # [B, embed]
            return self.out_norm(pooled)

    # ─────────────────────────────────────────────────────────────────────────
    # Stream 2 — Improved Interaction (2-layer GATv2 + temporal transformer)
    # ─────────────────────────────────────────────────────────────────────────

    class GATv2Layer(nn.Module):
        """
        Single GATv2 layer over a small person graph.
        Edge features are incorporated via additive attention.
        """
        def __init__(self, node_dim: int, edge_dim: int,
                     out_dim: int, heads: int = 4, dropout: float = 0.35):
            super().__init__()
            self.heads  = heads
            self.d      = out_dim // heads
            self.Wq     = nn.Linear(node_dim, out_dim)
            self.Wk     = nn.Linear(node_dim, out_dim)
            self.Wv     = nn.Linear(node_dim, out_dim)
            self.We     = nn.Linear(edge_dim,  out_dim)
            self.attn_v = nn.Linear(out_dim, heads)   # per-head scalar
            self.out    = nn.Linear(out_dim, out_dim)
            self.norm   = nn.LayerNorm(out_dim)
            self.drop   = nn.Dropout(dropout)

        def forward(self, nodes: torch.Tensor,
                    edges: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
            # nodes: [B, M, node_dim]
            # edges: [B, M, M, edge_dim]
            # mask:  [B, M]  True = valid node
            B, M, _ = nodes.shape
            Q = self.Wq(nodes)                       # [B, M, D]
            K = self.Wk(nodes)
            V = self.Wv(nodes)
            E = self.We(edges)                       # [B, M, M, D]
            # Attention: query i vs key j + edge i→j
            a = self.attn_v(F.leaky_relu(
                Q.unsqueeze(2) + K.unsqueeze(1) + E,
                negative_slope=0.2))                 # [B, M, M, heads]
            # Mask out invalid target nodes
            if mask is not None:
                a = a.masked_fill(
                    ~mask.unsqueeze(1).unsqueeze(-1), -1e9)
            a = F.softmax(a, dim=2)                  # [B, M, M, heads]
            # Per-head value weighting: each head attends its own slice of V
            V_heads = V.view(B, M, self.heads, self.d)           # [B, M, heads, d]
            agg = torch.einsum('bimh,bmhd->bihd', a, V_heads)   # [B, M, heads, d]
            agg = agg.reshape(B, M, -1)                          # [B, M, D]
            out = self.norm(self.drop(self.out(agg)) + V)
            return out                               # [B, M, D]

    class ImprovedInteraction(nn.Module):
        """
        Stream 2: 2-layer GATv2 per frame → temporal transformer.
        Input: int_nodes [B,T,M,7], int_edges [B,T,M,M,4], masks.
        Output: e_int [B, embed_dim]
        """
        def __init__(self, node_dim: int, edge_dim: int,
                     embed_dim: int, n_heads_gat: int,
                     n_heads_trf: int, n_layers_trf: int,
                     T: int, M: int, dropout: float):
            super().__init__()
            # Node input projection
            self.node_proj = nn.Sequential(
                nn.Linear(node_dim, embed_dim),
                nn.LayerNorm(embed_dim), nn.ReLU(),
            )
            # 2-layer GATv2
            self.gat1 = GATv2Layer(embed_dim, edge_dim,
                                   embed_dim, n_heads_gat, dropout)
            self.gat2 = GATv2Layer(embed_dim, edge_dim,
                                   embed_dim, n_heads_gat, dropout)
            # Learned temporal positional encoding
            self.pos_enc = nn.Parameter(torch.randn(1, T, embed_dim) * 0.02)
            # Transformer encoder
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads_trf,
                dim_feedforward=embed_dim * 2,
                dropout=dropout, batch_first=True,
                norm_first=True,          # Pre-LN for stability
            )
            self.transformer = nn.TransformerEncoder(
                enc_layer, num_layers=n_layers_trf)
            # Temporal attention pooling (learnable)
            self.tpool = nn.Linear(embed_dim, 1)
            self.out_norm = nn.LayerNorm(embed_dim)

        def forward(self, nodes: torch.Tensor,
                    edges: torch.Tensor,
                    node_mask: torch.Tensor) -> torch.Tensor:
            # nodes: [B,T,M,7]  edges: [B,T,M,M,4]  mask: [B,T,M]
            B, T, M, _ = nodes.shape
            frame_embs = []
            for t in range(T):
                n_t  = self.node_proj(nodes[:, t])         # [B, M, D]
                e_t  = edges[:, t]                          # [B, M, M, 4]
                mk_t = node_mask[:, t]                      # [B, M]
                # Two GATv2 passes with residual
                h    = self.gat1(n_t, e_t, mk_t)
                h    = self.gat2(h,   e_t, mk_t)
                # Masked mean over valid persons
                valid = mk_t.unsqueeze(-1).float()          # [B, M, 1]
                fe    = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
                frame_embs.append(fe)                       # [B, D]
            # Sequence: [B, T, D]
            seq = torch.stack(frame_embs, dim=1) + self.pos_enc
            # Transformer (no src_key_padding_mask needed; per-frame
            # masked mean already handles absent frames)
            out = self.transformer(seq)                     # [B, T, D]
            # Temporal attention pooling
            w   = F.softmax(self.tpool(out), dim=1)         # [B, T, 1]
            pooled = (out * w).sum(1)                       # [B, D]
            return self.out_norm(pooled)

    # ─────────────────────────────────────────────────────────────────────────
    # Stream 3 — Object/PO (V8-compatible FrameGINE + BiGRU)
    # ─────────────────────────────────────────────────────────────────────────

    class FrameGINE(nn.Module):
        """Per-frame GINE aggregation (V8-compatible)."""
        def __init__(self, node_dim: int, edge_dim: int,
                     hidden: int, layers: int = 2, dropout: float = 0.35):
            super().__init__()
            # Project nodes to hidden once so every subsequent residual
            # (x + agg) operates in the same dimensionality.
            self.node_in = nn.Linear(node_dim, hidden)
            # Edge encoder must also produce hidden-dim features so its
            # per-neighbour aggregate matches x's width after node_in.
            self.edge_enc = nn.Linear(edge_dim, hidden)
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                ) for _ in range(layers)
            ])
            self.out = nn.Linear(hidden, hidden)

        def forward(self, nodes: torch.Tensor,
                    edges: torch.Tensor,
                    node_mask: torch.Tensor,
                    src_mask: Optional[torch.Tensor] = None,
                    bipartite: bool = False) -> torch.Tensor:
            x   = self.node_in(nodes)   # [B, N, hidden]
            for mlp in self.mlps:
                e_e = self.edge_enc(edges)   # [..., hidden]
                if bipartite:
                    # e_e: [B, M, N, hidden]  sm: [B, M, 1, 1]
                    # broadcast then sum over M (source persons) → [B, N, hidden]
                    sm  = (src_mask.unsqueeze(-1).unsqueeze(-1).float()
                           if src_mask is not None
                           else torch.ones(
                               x.shape[0], e_e.shape[1], 1, 1,
                               device=nodes.device))
                    agg = (e_e * sm).sum(1)   # sum over M → [B, N, hidden]
                else:
                    # e_e: [B, N, N, hidden]  nm: [B, N, 1, 1]
                    # broadcast then sum over neighbour dim → [B, N, hidden]
                    nm  = (node_mask.unsqueeze(-1).unsqueeze(-1).float()
                           if node_mask is not None
                           else torch.ones(
                               x.shape[0], e_e.shape[1], 1, 1,
                               device=nodes.device))
                    agg = (e_e * nm).sum(2)   # sum over neighbour dim → [B, N, hidden]
                x = mlp(x + agg)
            nm2 = (node_mask.unsqueeze(-1).float()
                   if node_mask is not None
                   else torch.ones(x.shape[0], x.shape[1], 1,
                                   device=nodes.device))
            return (self.out(x) * nm2).sum(1) / nm2.sum(1).clamp(min=1)

    class TemporalBiGRU(nn.Module):
        """BiGRU with attention pooling (V8-compatible)."""
        def __init__(self, in_dim: int, embed_dim: int,
                     dropout: float = 0.35):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_dim, embed_dim),
                nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
            )
            self.gru  = nn.GRU(embed_dim, embed_dim // 2,
                               batch_first=True, bidirectional=True)
            self.attn = nn.Linear(embed_dim, 1)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, seq: torch.Tensor) -> torch.Tensor:
            # seq: [B, T, in_dim]
            x, _ = self.gru(self.proj(seq))
            w     = F.softmax(self.attn(x), dim=1)
            return self.norm((x * w).sum(1))

    class ObjectPOStream(nn.Module):
        """
        Stream 3: FrameGINE+BiGRU (object) + Bipartite FrameGINE+BiGRU (PO).
        Combines both sub-streams into a single [B, embed_dim] embedding.
        Identical to V8 design.
        """
        def __init__(self, obj_node_dim: int, int_edge_dim: int,
                     po_edge_dim: int, embed_dim: int, dropout: float):
            super().__init__()
            half = embed_dim // 2
            # obj_gine uses int_edge_dim; dummy zero edges must match this width
            self.obj_edge_dim = int_edge_dim
            self.obj_gine  = FrameGINE(obj_node_dim, int_edge_dim,
                                        half, layers=2, dropout=dropout)
            self.obj_gru   = TemporalBiGRU(half, half, dropout)
            self.po_gine   = FrameGINE(obj_node_dim, po_edge_dim,
                                        half, layers=2, dropout=dropout)
            self.po_gru    = TemporalBiGRU(half, half, dropout)
            self.proj      = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim), nn.ReLU(),
            )

        def forward(self, obj_n: torch.Tensor,
                    obj_nm: torch.Tensor,
                    po_e: torch.Tensor,
                    po_em: torch.Tensor,
                    int_nm: torch.Tensor) -> torch.Tensor:
            # obj_n:  [B, T, N, 6]   obj_nm: [B, T, N]
            # po_e:   [B, T, M, N, 5]  po_em: [B, T, M, N]
            # int_nm: [B, T, M]  (person mask, for bipartite src_mask)
            B, T, N, _ = obj_n.shape

            obj_frames = []
            po_frames  = []
            for t in range(T):
                on_t  = obj_n[:, t]           # [B, N, 6]
                onm_t = obj_nm[:, t]           # [B, N]
                pe_t  = po_e[:, t]             # [B, M, N, 5]
                inm_t = int_nm[:, t]           # [B, M]
                # Object stream: fully-connected within-frame graph.
                # dummy_edge last dim must equal obj_gine.edge_enc input dim
                # (int_edge_dim=4), NOT po_edge_dim=5.
                dummy_edge = torch.zeros(
                    B, N, N, self.obj_edge_dim, device=obj_n.device)
                obj_frames.append(
                    self.obj_gine(on_t, dummy_edge, onm_t))
                # PO stream: bipartite person→object
                po_frames.append(
                    self.po_gine(on_t, pe_t, onm_t,
                                  src_mask=inm_t, bipartite=True))

            e_obj = self.obj_gru(torch.stack(obj_frames, dim=1))  # [B,half]
            e_po  = self.po_gru(torch.stack(po_frames,  dim=1))   # [B,half]
            return self.proj(torch.cat([e_obj, e_po], dim=-1))     # [B,emb]

    # ─────────────────────────────────────────────────────────────────────────
    # Stream 4 — VideoMAE projection MLP
    # ─────────────────────────────────────────────────────────────────────────

    class VitProjection(nn.Module):
        """
        Linear projection from 768-d ViT [CLS] embedding to embed_dim.
        Two layers with LayerNorm for stable training.
        """
        def __init__(self, vit_dim: int, embed_dim: int,
                     dropout: float = 0.35):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(vit_dim, 256),
                nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)   # [B, embed_dim]

    # ─────────────────────────────────────────────────────────────────────────
    # Fusion — QualityGatedFusion (GQS-conditioned)
    # ─────────────────────────────────────────────────────────────────────────

    class QualityGatedFusion(nn.Module):
        """
        GQS-conditioned softmax gate over n_streams embeddings.
        gate_input: gqs[:, :n_streams] → MLP → softmax weights.
        Stream dropout: during training, zero one stream with probability p.
        gate_temperature: divides logits before softmax (annealed externally
            via current_temp buffer when temp_anneal=True).
        entropy_weight: coefficient for gate-entropy regularisation loss
            (accessed via model.get_gate_entropy() in the training loop).
        """
        def __init__(self, n_streams: int = 4,
                     stream_dropout: float = 0.25,
                     gate_temperature: float = 1.0,
                     entropy_weight: float = 0.0,
                     temp_anneal: bool = False):
            super().__init__()
            self.n              = n_streams
            self.p              = stream_dropout
            self.entropy_weight = entropy_weight
            self.temp_anneal    = temp_anneal
            # LayerNorm before the MLP stabilises gradient flow when GQS
            # values have low variance (all near 1.0 on clean RWF clips),
            # which otherwise makes gate logits near-uniform after softmax.
            self.gate = nn.Sequential(
                nn.LayerNorm(n_streams),
                nn.Linear(n_streams, 32), nn.ReLU(),
                nn.Linear(32, n_streams),
            )
            # Non-persistent: excluded from state_dict so old checkpoints load
            # without missing-key errors.
            self.register_buffer(
                "current_temp",
                torch.tensor(gate_temperature),
                persistent=False,
            )

        def forward(self, streams: List[torch.Tensor],
                    gqs: torch.Tensor) -> torch.Tensor:
            # streams: list of [B, D]   gqs: [B, ≥n_streams]
            stk        = torch.stack(streams, dim=1)           # [B, n, D]
            gate_logits = self.gate(gqs[:, :self.n])           # [B, n]
            gates       = F.softmax(
                gate_logits / self.current_temp, dim=-1)       # [B, n]
            if self.training and self.p > 0:
                B = stk.shape[0]
                mask = torch.ones(B, self.n, 1, device=stk.device)
                drop = (torch.rand(B, self.n, device=stk.device) < self.p)
                # Guarantee at least one stream survives per sample
                all_dead = drop.all(dim=1, keepdim=True)
                drop = drop & ~all_dead
                mask[drop] = 0.0
                gates = gates * mask.squeeze(-1)
                gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
            # Store per-batch entropy for optional regularisation in trainer
            if self.training:
                self.last_entropy = (
                    -(gates * (gates + 1e-8).log()).sum(-1).mean()
                )
            else:
                self.last_entropy = None
            return (stk * gates.unsqueeze(-1)).sum(1)          # [B, D]

    class LearnedWeightFusion(nn.Module):
        """
        Ablation control: fixed learned weights with no GQS input.
        Weights are a softmax over n_streams learned scalars
        (same for all samples in a batch — not conditioned on quality).
        """
        def __init__(self, n_streams: int = 4,
                     stream_dropout: float = 0.25):
            super().__init__()
            self.n = n_streams
            self.p = stream_dropout
            self.weights = nn.Parameter(torch.zeros(n_streams))

        def forward(self, streams: List[torch.Tensor],
                    gqs: torch.Tensor) -> torch.Tensor:
            # gqs ignored
            stk   = torch.stack(streams, dim=1)           # [B, n, D]
            gates = F.softmax(self.weights, dim=0)         # [n]
            gates = gates.unsqueeze(0).expand(stk.shape[0], -1)  # [B, n]
            if self.training and self.p > 0:
                B = stk.shape[0]
                mask = torch.ones(B, self.n, 1, device=stk.device)
                drop = (torch.rand(B, self.n, device=stk.device) < self.p)
                all_dead = drop.all(dim=1, keepdim=True)
                drop = drop & ~all_dead
                mask[drop] = 0.0
                gates = gates * mask.squeeze(-1)
                gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
            return (stk * gates.unsqueeze(-1)).sum(1)      # [B, D]

    # ─────────────────────────────────────────────────────────────────────────
    # ContentGatedFusion — gate derived from stream content, NOT from GQS.
    # This is ablation E_full_cgf: tests whether the QGF gate's advantage over
    # E_full_lw comes from quality conditioning or from content-adaptive gating.
    # If E_full_cgf ≥ E_full_qgf, the quality signal does not help beyond what
    # the content itself already provides — and the shortcut was the reason QGF
    # appeared competitive on UBI.
    # ─────────────────────────────────────────────────────────────────────────

    class ContentGatedFusion(nn.Module):
        """
        Gate logits derived from a mean-pool of the stream embeddings
        themselves, not from raw GQS. Structure mirrors QualityGatedFusion
        (same LayerNorm→Linear→ReLU→Linear, same stream-dropout path) so the
        comparison is apples-to-apples.
        """
        def __init__(self, n_streams: int = 4, embed_dim: int = 128,
                     stream_dropout: float = 0.25):
            super().__init__()
            self.n = n_streams
            self.p = stream_dropout
            # Input: mean-pool over streams → [B, embed_dim]
            self.gate = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 32), nn.ReLU(),
                nn.Linear(32, n_streams),
            )

        def forward(self, streams: List[torch.Tensor],
                    gqs: torch.Tensor) -> torch.Tensor:
            # gqs is intentionally ignored — gate derived from content only
            stk      = torch.stack(streams, dim=1)          # [B, n, D]
            ctx      = stk.mean(dim=1)                      # [B, D]
            gate_log = self.gate(ctx)                       # [B, n]
            gates    = F.softmax(gate_log, dim=-1)          # [B, n]
            if self.training and self.p > 0:
                B    = stk.shape[0]
                mask = torch.ones(B, self.n, 1, device=stk.device)
                drop = (torch.rand(B, self.n, device=stk.device) < self.p)
                all_dead = drop.all(dim=1, keepdim=True)
                drop = drop & ~all_dead
                mask[drop] = 0.0
                gates = gates * mask.squeeze(-1)
                gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
            return (stk * gates.unsqueeze(-1)).sum(1)       # [B, D]

    # ─────────────────────────────────────────────────────────────────────────
    # QualityGatedFusionAdv — QGF with an adversarial decorrelation head.
    # Keeps GQS as gate input (preserving the paper's original QGF design) but
    # adds a gradient-reversal penalty that prevents the gate from using raw
    # quality as a proxy for the fight label. The adversarial head tries to
    # predict the *label* from the gate logits; the gradient-reversal layer
    # (GRL) flips the gradient sign so the gate MLP is trained to make quality
    # non-predictive of the label while still being useful for weighting streams.
    # ─────────────────────────────────────────────────────────────────────────

    class _GRL(torch.autograd.Function):
        """Gradient Reversal Layer — passes forward, negates gradient."""
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            return -ctx.lam * grad, None

    class QualityGatedFusionAdv(nn.Module):
        """
        QGF variant with adversarial decorrelation (ablation E_full_qgf_adv).
        adv_weight: coefficient on the adversarial classification loss
            (accessed via model.get_adv_loss() from the training loop).
        """
        def __init__(self, n_streams: int = 4,
                     stream_dropout: float = 0.25,
                     adv_weight: float = 0.1,
                     adv_lam: float = 1.0):
            super().__init__()
            self.n          = n_streams
            self.p          = stream_dropout
            self.adv_weight = adv_weight
            self.adv_lam    = adv_lam
            # Same gate as standard QGF
            self.gate = nn.Sequential(
                nn.LayerNorm(n_streams),
                nn.Linear(n_streams, 32), nn.ReLU(),
                nn.Linear(32, n_streams),
            )
            # Adversarial head: predict label from gate logits via GRL so the
            # gate is penalised for being quality-predictive of the label.
            self.adv_head = nn.Linear(n_streams, 1)
            self.last_adv_loss: Optional[torch.Tensor] = None

        def forward(self, streams: List[torch.Tensor],
                    gqs: torch.Tensor) -> torch.Tensor:
            stk        = torch.stack(streams, dim=1)            # [B, n, D]
            gate_logits = self.gate(gqs[:, :self.n])            # [B, n]
            gates       = F.softmax(gate_logits, dim=-1)        # [B, n]

            if self.training:
                # Adversarial: push the gate so its logits do NOT predict label
                rev = _GRL.apply(gate_logits, self.adv_lam)     # [B, n]
                self.last_adv_loss = self.adv_head(rev).squeeze(-1)  # [B] raw logit
            else:
                self.last_adv_loss = None

            if self.training and self.p > 0:
                B    = stk.shape[0]
                mask = torch.ones(B, self.n, 1, device=stk.device)
                drop = (torch.rand(B, self.n, device=stk.device) < self.p)
                all_dead = drop.all(dim=1, keepdim=True)
                drop = drop & ~all_dead
                mask[drop] = 0.0
                gates = gates * mask.squeeze(-1)
                gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
            return (stk * gates.unsqueeze(-1)).sum(1)            # [B, D]

    # ─────────────────────────────────────────────────────────────────────────
    # Full model
    # ─────────────────────────────────────────────────────────────────────────

    class V9Model(nn.Module):
        """
        Full V9 model.
        active_streams: tuple of stream names in use, controls which
        forward paths are executed and which fusion mode is used.
        fusion_mode: 'qgf' | 'lw' | 'cgf' | 'qgf_adv'.
        gate_temperature / entropy_weight / temp_anneal forwarded to QGF only.
        adv_weight / adv_lam forwarded to QualityGatedFusionAdv only.
        """
        def __init__(self,
                     active_streams: Tuple[str, ...],
                     fusion_mode: str,
                     C: int, T: int, V: int, M: int, N: int,
                     int_nd: int, int_ed: int,
                     obj_nd: int, po_ed: int,
                     vit_dim: int,
                     hidden: int, embed_dim: int,
                     dropout: float, stream_dropout: float,
                     n_heads_gat: int, n_heads_trf: int,
                     n_layers_trf: int,
                     gate_temperature: float = 1.0,
                     entropy_weight: float = 0.0,
                     temp_anneal: bool = False,
                     adv_weight: float = 0.1,
                     adv_lam: float = 1.0):
            super().__init__()
            self.active   = active_streams
            self.fmode    = fusion_mode
            n_active = len(active_streams)

            if "skeleton" in active_streams:
                self.stgcn = EnhancedSTGCN(
                    C, T, V, M, hidden, embed_dim, dropout)

            if "interaction" in active_streams:
                self.interaction = ImprovedInteraction(
                    int_nd, int_ed, embed_dim,
                    n_heads_gat, n_heads_trf, n_layers_trf,
                    T, M, dropout)

            if "object" in active_streams:
                self.obj_po = ObjectPOStream(
                    obj_nd, int_ed, po_ed, embed_dim, dropout)

            if "vit" in active_streams:
                self.vit_proj = VitProjection(vit_dim, embed_dim, dropout)

            # Fusion (only instantiated for multi-stream experiments)
            if n_active > 1:
                if fusion_mode == "qgf":
                    self.fusion = QualityGatedFusion(
                        n_active, stream_dropout,
                        gate_temperature=gate_temperature,
                        entropy_weight=entropy_weight,
                        temp_anneal=temp_anneal,
                    )
                elif fusion_mode == "cgf":
                    self.fusion = ContentGatedFusion(
                        n_active, embed_dim, stream_dropout)
                elif fusion_mode == "qgf_adv":
                    self.fusion = QualityGatedFusionAdv(
                        n_active, stream_dropout,
                        adv_weight=adv_weight,
                        adv_lam=adv_lam,
                    )
                else:
                    self.fusion = LearnedWeightFusion(n_active, stream_dropout)

            # Classification head
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 1),
            )

        def get_gate_entropy(self):
            """Return last-batch gate entropy if QGF is active, else None."""
            if hasattr(self, "fusion") and hasattr(self.fusion, "last_entropy"):
                return self.fusion.last_entropy
            return None

        def get_adv_loss(self):
            """Return adversarial label-prediction logits [B] if qgf_adv, else None."""
            if hasattr(self, "fusion") and hasattr(self.fusion, "last_adv_loss"):
                return self.fusion.last_adv_loss
            return None

        def forward(self, batch: Dict) -> torch.Tensor:
            streams_out = []

            if "skeleton" in self.active:
                # int_node_mask [B,T,M] doubles as the person-presence mask for
                # the skeleton stream — both streams share the same M-person slots.
                e_skel = self.stgcn(
                    batch["skeleton"],
                    batch["int_node_mask"])
                streams_out.append(e_skel)

            if "interaction" in self.active:
                e_int = self.interaction(
                    batch["int_nodes"],
                    batch["int_edges"],
                    batch["int_node_mask"])
                streams_out.append(e_int)

            if "object" in self.active:
                e_obj = self.obj_po(
                    batch["obj_nodes"],
                    batch["obj_node_mask"],
                    batch["po_edges"],
                    batch["po_edge_mask"],
                    batch["int_node_mask"])
                streams_out.append(e_obj)

            if "vit" in self.active:
                e_vit = self.vit_proj(batch["vit_embedding"])
                streams_out.append(e_vit)

            if len(streams_out) == 1:
                fused = streams_out[0]
            else:
                fused = self.fusion(streams_out, batch["gqs"])

            return self.head(fused).squeeze(-1)   # [B]

    return {
        "V9Model": V9Model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training function
# ══════════════════════════════════════════════════════════════════════════════
@app.function(
    image=image,
    gpu="L4",               # ~740K param graph model fits comfortably; L4 ~1.8x T4 throughput at +35% price → ~25% cheaper overall
    cpu=2,                  # num_workers=0; data is in-RAM, no decode parallelism needed
    memory=12288,           # 12 GB — 3 in-memory datasets (~2 GB) + autograd activations + PyTorch overhead
    timeout=21600,          # 6 hours — finishes in ~2.5 h on L4 across 5 experiments
    volumes={str(PROC_MOUNT): vol_proc},
)
def train() -> None:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import (f1_score, roc_auc_score,
                                  accuracy_score, confusion_matrix)
    from scipy.optimize import minimize_scalar
    from tqdm import tqdm
    from torch.utils.data import WeightedRandomSampler

    vol_proc.reload()
    seed_everything(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # ── Load split ────────────────────────────────────────────────────────
    if not SPLIT_CSV.exists():
        raise RuntimeError("split_ubi.csv not found. Run Phase 1 first.")
    split_df = pd.read_csv(str(SPLIT_CSV))

    def get_split(name: str):
        sub    = split_df[split_df["split"] == name]
        paths  = [CACHE_DIR / f"{v}.npz" for v in sub["clip_id"]]
        labels = sub["label"].tolist()
        # Filter hard-negative OOD videos from training only.
        # A segment stem looks like "val_Fight_XM0NI7ZmwWM_0_<hash>"; we match
        # by checking whether the video stem appears anywhere in the path stem.
        if name == "train" and cfg.hard_negative_video_stems:
            filtered = [
                (p, l) for p, l in zip(paths, labels)
                if p.exists() and not any(
                    hn in p.stem for hn in cfg.hard_negative_video_stems
                )
            ]
            excluded = sum(
                1 for p in paths
                if p.exists() and any(
                    hn in p.stem for hn in cfg.hard_negative_video_stems
                )
            )
            if excluded:
                print(f"  Hard-negative exclusion: removed {excluded} train "
                      f"segments from {cfg.hard_negative_video_stems}")
            paths_out, labels_out = zip(*filtered) if filtered else ([], [])
            return list(paths_out), list(labels_out)
        return (
            [p for p in paths if p.exists()],
            [l for p, l in zip(paths, labels) if p.exists()],
        )

    UBIDataset  = build_dataset_class()
    components  = build_model_components()
    V9Model     = components["V9Model"]

    train_paths, train_labels = get_split("train")
    val_paths,   val_labels   = get_split("val")
    test_paths,  test_labels  = get_split("test")
    # UBI-Fights has an official 67-video test split in test_videos.csv.
    # A missing test split is a preprocessing error, not a fallback case.
    if len(test_paths) == 0:
        raise RuntimeError(
            "No test split found in split_ubi.csv. "
            "Ensure preprocessing completed and test_videos.csv was read correctly."
        )
    print(f"Train={len(train_paths)}  Val={len(val_paths)}  Test={len(test_paths)}")

    train_ds = UBIDataset(train_paths, train_labels)
    val_ds   = UBIDataset(val_paths,   val_labels)
    test_ds  = UBIDataset(test_paths,  test_labels)

    # ── Prior-matched, quality-decorrelated sampler ───────────────────────
    # Root cause (diagnosed 2026-05-29): train prior ~0.30, test prior ~0.62
    # (inverted). Also, in train high-skeleton-quality clips skew positive
    # (q_skel ~0.91 fights vs ~0.67 non-fights), giving the model a
    # non-transferable shortcut that breaks on the test set (test negatives
    # are also high quality). We fix both with one joint WeightedRandomSampler:
    #   - Target prior = test prior (~0.62 positive rate).
    #   - Within each quality tier, pos:neg ratio is equalised, breaking the
    #     quality-label correlation without discarding any clips.
    # pos_weight is then set so BCE loss matches the target prior (~0.62 pos
    # = 0.38/0.62 ≈ 0.613 pos_weight), rather than the raw train imbalance.
    TARGET_POS_RATE = 0.62   # UBI-Fights official test split positive rate

    gqs_path = PROC_MOUNT / "gqs_summary_ubi.csv"
    if not gqs_path.exists():
        raise RuntimeError("gqs_summary_ubi.csv not found. Run Phase 1 first.")
    gqs_df = pd.read_csv(str(gqs_path))
    # composite matches the formula used in train.py diagnostics
    gqs_df["gqs_composite"] = (
        0.4 * gqs_df["q_skel"] + 0.4 * gqs_df["q_int"] + 0.2 * gqs_df["q_po"]
    )
    # Build clip_id → composite score lookup
    gqs_lookup = gqs_df.set_index("clip_id")["gqs_composite"].to_dict()

    # Assign each train clip to a quality tier (3 equal-width tiers over [0,1])
    N_QTIERS = 3
    train_gqs = np.array([
        gqs_lookup.get(p.stem, 0.5) for p in train_paths
    ], dtype=np.float32)
    tier_edges  = np.linspace(0.0, 1.0 + 1e-6, N_QTIERS + 1)
    train_tiers = np.digitize(train_gqs, tier_edges) - 1   # 0-indexed

    train_labels_arr = np.array(train_labels, dtype=np.int32)
    sample_weights   = np.zeros(len(train_labels), dtype=np.float64)

    for tier in range(N_QTIERS):
        in_tier = (train_tiers == tier)
        for lbl in (0, 1):
            mask  = in_tier & (train_labels_arr == lbl)
            n_cell = mask.sum()
            if n_cell == 0:
                continue
            # Target proportion: in this tier, positives should occupy
            # TARGET_POS_RATE and negatives (1 - TARGET_POS_RATE).
            target_frac = TARGET_POS_RATE if lbl == 1 else (1.0 - TARGET_POS_RATE)
            # Weight is inversely proportional to cell size relative to target.
            sample_weights[mask] = target_frac / n_cell

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_labels),
        replacement=True,
    )

    def make_loader(ds, sampler=None, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            ds, batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=(shuffle if sampler is None else False),
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    train_loader = make_loader(train_ds, sampler=sampler)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    # ── Helpers ───────────────────────────────────────────────────────────
    # pos_weight targets the eval prior (TARGET_POS_RATE), not the raw train
    # imbalance. At TARGET_POS_RATE=0.62: pw = (1-0.62)/0.62 ≈ 0.613.
    # This keeps the loss scale aligned with the distribution the model is
    # evaluated on, so the decision boundary generalises to the test prior.
    n_pos    = sum(train_labels)
    n_neg    = len(train_labels) - n_pos
    pw_val   = (1.0 - TARGET_POS_RATE) / TARGET_POS_RATE
    pw       = torch.tensor([pw_val], device=DEVICE)
    print(f"  pos_weight = {pw_val:.3f}  (target prior {TARGET_POS_RATE:.2f}; "
          f"raw train: n_neg={n_neg}, n_pos={n_pos})")

    def smooth_bce(logits, targets, label_smoothing=None):
        # label_smoothing=None means use cfg defaults; pass 0.0 to disable.
        ls = cfg.label_smoothing if label_smoothing is None else label_smoothing
        if ls == 0.0 or not cfg.use_label_smoothing:
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw)
        t = targets * (1.0 - ls) + (1.0 - targets) * ls
        return F.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)

    def find_threshold(y_true, y_prob, target_pos_rate: float = TARGET_POS_RATE):
        # Resample (y_true, y_prob) to target_pos_rate before searching so the
        # chosen threshold matches the eval prior, not the val prior (~0.21).
        # We downsample the majority class to achieve the target rate exactly.
        rng     = np.random.default_rng(cfg.seed)
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos_v = len(pos_idx)
        n_neg_v = len(neg_idx)
        # n_pos / (n_pos + n_neg_keep) = target_pos_rate
        # → n_neg_keep = n_pos * (1 - target_pos_rate) / target_pos_rate
        n_neg_keep = max(1, int(round(n_pos_v * (1.0 - target_pos_rate)
                                      / target_pos_rate)))
        if n_neg_keep < n_neg_v:
            neg_idx = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        idx_use  = np.concatenate([pos_idx, neg_idx])
        yt, yp   = y_true[idx_use], y_prob[idx_use]

        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(cfg.thr_min, cfg.thr_max + 1e-6, cfg.thr_step):
            f1 = f1_score(yt, (yp >= thr).astype(int),
                          average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        return best_thr, best_f1

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        y_true, y_prob = [], []
        for batch in loader:
            batch  = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            probs  = torch.sigmoid(model(batch)).cpu().numpy()
            labels = batch["label"].cpu().numpy().astype(int)
            y_prob.extend(probs.tolist())
            y_true.extend(labels.tolist())
        return np.array(y_true), np.array(y_prob)

    def metrics_at_threshold(y_true, y_prob, thr):
        preds = (y_prob >= thr).astype(int)
        cm    = confusion_matrix(y_true, preds).tolist()
        # roc_auc_score raises if only one class is present in y_true
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        return {
            "accuracy":  float(accuracy_score(y_true, preds)),
            "macro_f1":  float(f1_score(y_true, preds, average="macro",
                                         zero_division=0)),
            "binary_f1": float(f1_score(y_true, preds, average="binary",
                                         zero_division=0)),
            "roc_auc":   auc,
            "threshold": float(thr),
            "confusion_matrix": cm,
        }

    # ── Read tensor dimensions from one sample ────────────────────────────
    s0      = train_ds[0]
    int_nd  = s0["int_nodes"].shape[-1]    # 7
    int_ed  = s0["int_edges"].shape[-1]    # 4
    obj_nd  = s0["obj_nodes"].shape[-1]    # 6
    po_ed   = s0["po_edges"].shape[-1]     # 5

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # Ablation experiment definitions
    # ══════════════════════════════════════════════════════════════════════
    # Each entry: (experiment_id, active_streams_tuple, fusion_mode)
    # fusion_mode is irrelevant for single-stream experiments.
    EXPERIMENTS = [
        ("A_skeleton_only",        ("skeleton",),
         "qgf"),
        ("B_skel_interaction",     ("skeleton", "interaction"),
         "qgf"),
        ("D_skel_int_obj",         ("skeleton", "interaction", "object"),
         "qgf"),
        ("E_full_qgf",             ("skeleton", "interaction", "object", "vit"),
         "qgf"),
        ("E_full_lw",              ("skeleton", "interaction", "object", "vit"),
         "lw"),
        ("E_full_qgf_fixed",       ("skeleton", "interaction", "object", "vit"),
         "qgf"),
        # ── New ablations (Phase-2b, 2026-05-29) ─────────────────────────────
        # E_full_cgf: content-gated fusion — gate from stream embeddings, not GQS.
        # Tests whether QGF's advantage came from content-adaptive gating or the
        # quality shortcut.
        ("E_full_cgf",             ("skeleton", "interaction", "object", "vit"),
         "cgf"),
        # E_full_qgf_adv: QGF + adversarial decorrelation head. Keeps GQS as gate
        # input but penalises the gate for being predictive of the fight label.
        ("E_full_qgf_adv",         ("skeleton", "interaction", "object", "vit"),
         "qgf_adv"),
    ]
    # Note: Experiment C (VideoMAE only) is reported directly from Phase 2
    # results (videomae_test_results.json). We do not re-run it here.

    # Per-experiment overrides: any subset of {entropy_weight, gate_temperature,
    # temp_anneal, label_smoothing, adv_weight, adv_lam}. Missing keys inherit
    # cfg defaults.
    EXPERIMENT_OVERRIDES = {
        "E_full_qgf_fixed": {
            "entropy_weight":  0.10,   # maximise gate spread during training
            "temp_anneal":     True,   # anneal temperature 2.0→0.5 over epochs
            "label_smoothing": 0.0,    # disable smoothing; entropy reg replaces it
        },
        "E_full_qgf_adv": {
            "adv_weight": 0.1,   # adversarial penalty coefficient
            "adv_lam":    1.0,   # GRL gradient scaling
        },
    }

    results_rows = []

    for exp_id, active_streams, fmode in EXPERIMENTS:
      try:
        seed_everything(cfg.seed)
        exp_dir = RUN_DIR / exp_id
        exp_dir.mkdir(exist_ok=True)

        # ── Resume: skip experiments already completed on the volume ───────
        # An experiment is "done" if its test_metrics.json was written (it is
        # the last artefact saved per experiment). Reload its result into the
        # comparison table so the final CSV stays complete, then skip training.
        done_marker = exp_dir / "test_metrics.json"
        ckpt_marker = exp_dir / "best.pt"
        if cfg.resume and done_marker.exists():
            try:
                with open(str(done_marker)) as f:
                    test_m_done = json.load(f)
                best_val_f1_done = float("nan")
                if ckpt_marker.exists():
                    ck_done = torch.load(str(ckpt_marker), map_location="cpu")
                    best_val_f1_done = float(ck_done.get("val_macro_f1",
                                                          float("nan")))
                results_rows.append({
                    "experiment":  exp_id,
                    "streams":     str(active_streams),
                    "fusion":      fmode,
                    "params":      None,   # not recomputed on resume
                    "best_val_f1": (round(best_val_f1_done, 4)
                                    if best_val_f1_done == best_val_f1_done
                                    else "resumed"),
                    **{f"test_{k}": v for k, v in test_m_done.items()
                       if k not in ("confusion_matrix",
                                    "confusion_matrix_thr050")},
                })
                print(f"\n[RESUME] {exp_id} already complete "
                      f"(test_metrics.json found) — skipping. "
                      f"Test macro-F1={test_m_done.get('macro_f1', float('nan')):.4f}")
                continue
            except Exception as e:
                # Corrupt/partial marker — fall through and retrain this one.
                print(f"[RESUME] {exp_id} marker unreadable ({e}); retraining.")

        # ── Resolve per-experiment overrides ──────────────────────────────
        ovr                 = EXPERIMENT_OVERRIDES.get(exp_id, {})
        exp_entropy_weight  = ovr.get("entropy_weight",  0.0)
        exp_gate_temp       = ovr.get("gate_temperature", 1.0)
        exp_temp_anneal     = ovr.get("temp_anneal",      False)
        exp_label_smoothing = ovr.get("label_smoothing",  cfg.label_smoothing)
        exp_adv_weight      = ovr.get("adv_weight",       0.1)
        exp_adv_lam         = ovr.get("adv_lam",          1.0)

        n_active = len(active_streams)
        eff_lr   = cfg.lr if n_active == 1 else cfg.lr_multi

        print(f"\n{'='*70}")
        print(f"Experiment : {exp_id}")
        print(f"Streams    : {active_streams}")
        print(f"Fusion     : {fmode}  |  LR: {eff_lr:.1e}")
        if ovr:
            print(f"Overrides  : {ovr}")
        print(f"{'='*70}")

        model = V9Model(
            active_streams  = active_streams,
            fusion_mode     = fmode,
            C=cfg.C, T=cfg.T, V=cfg.V, M=cfg.M, N=cfg.N,
            int_nd=int_nd, int_ed=int_ed,
            obj_nd=obj_nd, po_ed=po_ed,
            vit_dim=cfg.vit_embed_dim,
            hidden=cfg.graph_hidden, embed_dim=cfg.embed_dim,
            dropout=cfg.dropout, stream_dropout=cfg.stream_dropout,
            n_heads_gat=cfg.n_heads_gat,
            n_heads_trf=cfg.n_heads_trf,
            n_layers_trf=cfg.n_layers_trf,
            gate_temperature=exp_gate_temp,
            entropy_weight  =exp_entropy_weight,
            temp_anneal     =exp_temp_anneal,
            adv_weight      =exp_adv_weight,
            adv_lam         =exp_adv_lam,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print(f"Trainable params: {n_params:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=eff_lr, weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=4,
        )

        best_val_f1  = -1.0
        best_ckpt    = exp_dir / "best.pt"
        patience_cnt = 0
        history      = []

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(1, cfg.epochs + 1):
            # Anneal gate temperature from 2.0→0.5 across epochs when enabled
            if exp_temp_anneal and hasattr(model, "fusion"):
                T_val = max(
                    0.5,
                    2.0 - 1.5 * (epoch - 1) / max(cfg.epochs - 1, 1),
                )
                model.fusion.current_temp.fill_(T_val)

            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader,
                        desc=f"  Epoch {epoch:03d}", leave=False,
                        unit="batch")
            for batch in pbar:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor)
                           else v for k, v in batch.items()}
                optimizer.zero_grad()
                logits = model(batch)
                loss   = smooth_bce(logits, batch["label"],
                                    label_smoothing=exp_label_smoothing)
                # Entropy regularisation: subtract to maximise gate spread
                entropy = model.get_gate_entropy()
                if entropy is not None and exp_entropy_weight > 0:
                    loss = loss - exp_entropy_weight * entropy
                # Adversarial decorrelation: penalise gate for predicting label
                # from quality (qgf_adv only). GRL already reversed the gradient
                # inside the fusion module; we add BCE here so the adv head has
                # a proper classification signal to reverse against.
                adv_logits = model.get_adv_loss()
                if adv_logits is not None and exp_adv_weight > 0:
                    adv_loss = F.binary_cross_entropy_with_logits(
                        adv_logits, batch["label"])
                    loss = loss + exp_adv_weight * adv_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / max(len(train_loader), 1)

            # Validate
            y_tv, y_pv    = evaluate(model, val_loader)
            val_thr, val_f1 = find_threshold(y_tv, y_pv)
            try:
                val_auc = float(roc_auc_score(y_tv, y_pv))
            except Exception:
                val_auc = 0.0

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_f1)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                print(f"  LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")
            history.append({
                "epoch": epoch, "train_loss": avg_loss,
                "val_macro_f1": val_f1, "val_roc_auc": val_auc,
                "val_threshold": val_thr,
            })
            print(f"  Ep {epoch:03d} | Loss {avg_loss:.4f} | "
                  f"Val F1 {val_f1:.4f} | AUC {val_auc:.4f} | "
                  f"Thr {val_thr:.2f}")

            # Checkpoint
            if epoch >= cfg.min_ckpt_epoch and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_macro_f1": val_f1,
                    "val_roc_auc":  val_auc,
                    "threshold":    val_thr,
                }, str(best_ckpt))
                patience_cnt = 0
            elif epoch >= cfg.min_ckpt_epoch:
                patience_cnt += 1
                if patience_cnt >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # ── Test evaluation ───────────────────────────────────────────────
        if not best_ckpt.exists():
            # No checkpoint was ever saved (model never beat -1.0, e.g. NaN loss)
            print(f"  WARN: no checkpoint written for {exp_id}; skipping test.")
            continue
        ck = torch.load(str(best_ckpt), map_location=DEVICE)
        model.load_state_dict(ck["state_dict"])
        best_thr = float(ck["threshold"])
        print(f"\n  Loaded best ckpt ep={ck['epoch']} "
              f"val_f1={ck['val_macro_f1']:.4f}")

        def metrics_dual(y_true, y_prob, opt_thr):
            """Compute metrics at optimal threshold AND at 0.5 for comparison."""
            m   = metrics_at_threshold(y_true, y_prob, opt_thr)
            m50 = metrics_at_threshold(y_true, y_prob, 0.5)
            m["macro_f1_thr050"]         = m50["macro_f1"]
            m["accuracy_thr050"]         = m50["accuracy"]
            m["confusion_matrix_thr050"] = m50["confusion_matrix"]
            return m

        y_tt, y_pt = evaluate(model, test_loader)

        # ── Temperature calibration on val set, re-tuned threshold ───────
        # Fit temperature T* that minimises NLL on val probs, then find the
        # optimal decision threshold on the calibrated val probs.  The
        # calibrated threshold replaces best_thr for the test evaluation so
        # we never evaluate calibrated probs at the raw-trained threshold.
        y_tv_cal, y_pv_cal = evaluate(model, val_loader)
        eps_cal = 1e-7
        def nll_t(temp):
            p = np.clip(y_pv_cal, eps_cal, 1 - eps_cal)
            logit = np.log(p / (1 - p)) / max(temp, 1e-3)
            p_t   = 1 / (1 + np.exp(-logit))
            return -np.mean(
                y_tv_cal * np.log(p_t + eps_cal)
                + (1 - y_tv_cal) * np.log(1 - p_t + eps_cal)
            )
        opt_t = minimize_scalar(nll_t, bounds=(0.5, 3.0), method="bounded")
        cal_temp = float(opt_t.x)

        def apply_temp(probs, temp):
            p = np.clip(probs, eps_cal, 1 - eps_cal)
            logit = np.log(p / (1 - p)) / max(temp, 1e-3)
            return 1 / (1 + np.exp(-logit))

        y_pv_scaled = apply_temp(y_pv_cal, cal_temp)
        cal_thr, cal_f1 = find_threshold(y_tv_cal, y_pv_scaled)
        y_pt_scaled     = apply_temp(y_pt, cal_temp)

        print(f"  Calibration: T*={cal_temp:.4f}  "
              f"cal_thr={cal_thr:.2f}  cal_val_f1={cal_f1:.4f}  "
              f"(raw thr was {best_thr:.2f})")

        # Report both raw-threshold and calibrated-threshold test metrics
        test_m     = metrics_dual(y_tt, y_pt, best_thr)
        test_m_cal = metrics_at_threshold(y_tt, y_pt_scaled, cal_thr)
        test_m["cal_macro_f1"]  = test_m_cal["macro_f1"]
        test_m["cal_accuracy"]  = test_m_cal["accuracy"]
        test_m["cal_roc_auc"]   = test_m_cal["roc_auc"]
        test_m["cal_threshold"] = cal_thr
        test_m["cal_temp"]      = cal_temp

        print(f"\n  TEST (raw)  — Macro-F1={test_m['macro_f1']:.4f}  "
              f"ROC-AUC={test_m['roc_auc']:.4f}  "
              f"Acc={test_m['accuracy']:.4f}  "
              f"Thr={best_thr:.2f}")
        print(f"  TEST (cal)  — Macro-F1={test_m['cal_macro_f1']:.4f}  "
              f"ROC-AUC={test_m['cal_roc_auc']:.4f}  "
              f"Acc={test_m['cal_accuracy']:.4f}  "
              f"Thr={cal_thr:.2f}  T={cal_temp:.3f}")
        print(f"  CM: {test_m['confusion_matrix']}")

        # Logit-distribution diagnostics
        eps     = 1e-7
        p_clip  = np.clip(y_pt, eps, 1.0 - eps)
        logit_t = np.log(p_clip / (1.0 - p_clip))
        lg_pos  = float(logit_t[y_tt == 1].mean()) if (y_tt == 1).any() else float("nan")
        lg_neg  = float(logit_t[y_tt == 0].mean()) if (y_tt == 0).any() else float("nan")
        print(f"  Threshold analysis — {exp_id}")
        print(f"    Optimal thr : {best_thr:.4f}   F1@opt  : {test_m['macro_f1']:.4f}")
        print(f"    F1@thr=0.5  : {test_m['macro_f1_thr050']:.4f}")
        print(f"    Mean logit  pos={lg_pos:+.4f}   neg={lg_neg:+.4f}")
        print(f"    Mean sigmoid: {float(y_pt.mean()):.4f}")

        # GQS stratification analysis — composite quality score avoids the
        # valid_ratio saturation problem on RWF-2000 (all clips fully valid).
        gqs_df_path = PROC_MOUNT / "gqs_summary_ubi.csv"
        if gqs_df_path.exists() and ("qgf" in exp_id or "lw" in exp_id):
            try:
                gqs_df   = pd.read_csv(str(gqs_df_path))
                test_sub = gqs_df[gqs_df["split"] == "test"].copy()
                if len(test_sub) == 0:
                    test_sub = gqs_df[gqs_df["split"] == "val"].copy()
                if len(test_sub) == 0:
                    raise ValueError("no test/val rows in gqs_summary_ubi.csv")

                # Validate required columns exist before computing composite
                for col in ("q_skel", "q_int", "q_po"):
                    if col not in test_sub.columns:
                        raise ValueError(
                            f"gqs_summary_ubi.csv missing column {col}")
                test_sub["gqs_composite"] = (
                    0.4 * test_sub["q_skel"] +
                    0.4 * test_sub["q_int"]  +
                    0.2 * test_sub["q_po"]
                )

                # Primary: quartile cut; fall back to tertile cut if too few
                # unique values (common when quality is uniformly high)
                try:
                    test_sub["q_bucket"], _ = pd.qcut(
                        test_sub["gqs_composite"], q=4,
                        duplicates="drop", retbins=True)
                    n_bins = test_sub["q_bucket"].nunique()
                except Exception:
                    n_bins = 0

                if n_bins < 2:
                    edges = np.quantile(
                        test_sub["gqs_composite"].values,
                        [0, 0.33, 0.66, 1.0])
                    test_sub["q_bucket"] = pd.cut(
                        test_sub["gqs_composite"], bins=edges,
                        labels=["Q1_low", "Q2_mid", "Q3_high"],
                        include_lowest=True)
                    n_bins = test_sub["q_bucket"].nunique()

                if n_bins < 2:
                    print("  WARN: gqs_composite has no discriminative "
                          "variance; skipping GQS quartile CSV.")
                else:
                    vid_to_idx = {
                        test_ds.samples[i]["video_id"]: i
                        for i in range(len(test_ds))
                    }
                    quartile_rows = []
                    for bucket, grp in test_sub.groupby(
                            "q_bucket", observed=True):
                        vids = grp["clip_id"].tolist()
                        idxs = [vid_to_idx[v] for v in vids
                                if v in vid_to_idx]
                        if not idxs:
                            continue
                        yt  = y_tt[idxs]
                        yp  = y_pt[idxs]
                        pr  = (yp >= best_thr).astype(int)
                        f1q = float(f1_score(yt, pr, average="macro",
                                             zero_division=0))
                        try:
                            aucq = float(roc_auc_score(yt, yp))
                        except Exception:
                            aucq = float("nan")
                        quartile_rows.append({
                            "experiment": exp_id,
                            "bucket":     str(bucket),
                            "n":          len(idxs),
                            "macro_f1":   f1q,
                            "roc_auc":    aucq,
                        })
                    pd.DataFrame(quartile_rows).to_csv(
                        str(exp_dir / "gqs_quartile_f1.csv"), index=False)
                    print("  GQS quartile F1 saved.")
            except Exception as e:
                print(f"  WARN: GQS stratification failed: {e}")

        # Save artefacts
        pd.DataFrame(history).to_csv(
            str(exp_dir / "train_history.csv"), index=False)
        with open(str(exp_dir / "test_metrics.json"), "w") as f:
            json.dump(test_m, f, indent=2)

        results_rows.append({
            "experiment":    exp_id,
            "streams":       str(active_streams),
            "fusion":        fmode,
            "params":        n_params,
            "best_val_f1":   round(best_val_f1, 4),
            **{f"test_{k}": v for k, v in test_m.items()
               if k not in ("confusion_matrix", "confusion_matrix_thr050")},
        })
        vol_proc.commit()
      except Exception as exc:
        # Persist results from any experiments that completed before this failure
        print(f"  ERROR in experiment {exp_id}: {exc}")
        vol_proc.commit()
        raise

    # ── Final comparison table ────────────────────────────────────────────
    # Append VideoMAE standalone result from Phase 2 (Experiment C)
    vit_results_path = PROC_MOUNT / "videomae_test_results.json"
    if vit_results_path.exists():
        with open(str(vit_results_path)) as f:
            vit_r = json.load(f)
        # Only merge metric keys that align with the graph experiments'
        # columns; drop ad-hoc keys like "note" that would create stray cols.
        allowed = {"accuracy", "macro_f1", "roc_auc", "threshold"}
        results_rows.insert(2, {
            "experiment":  "C_videomae_only",
            "streams":     "('vit',)",
            "fusion":      "none",
            "params":      86_000_000,   # VideoMAE-Base ~86M (encoder) + small head
            "best_val_f1": "see_phase2_log",
            **{f"test_{k}": vit_r[k] for k in vit_r if k in allowed},
        })

    results_df  = pd.DataFrame(results_rows)
    results_csv = RUN_DIR / "experiment_comparison_ubi.csv"
    results_df.to_csv(str(results_csv), index=False)

    print(f"\n{'='*70}")
    print("UBI ABLATION COMPARISON")
    show_cols = ["experiment", "test_macro_f1", "test_roc_auc",
                 "test_accuracy", "test_threshold"]
    # cal columns only present for graph experiments (not C_videomae_only)
    cal_cols = ["test_cal_macro_f1", "test_cal_threshold", "test_cal_temp"]
    show_cols += [c for c in cal_cols if c in results_df.columns]
    print(results_df[show_cols].to_string(index=False))
    print(f"{'='*70}")
    print("\nKey comparison — GQS ablation:")
    subset = results_df[results_df["experiment"].isin(
        ["E_full_qgf", "E_full_lw"])]
    if len(subset):
        sub_cols = ["experiment", "fusion", "test_macro_f1", "test_roc_auc"]
        sub_cols += [c for c in cal_cols if c in subset.columns]
        print(subset[sub_cols].to_string(index=False))
    vol_proc.commit()
    print("UBI-Fights Phase 3 complete.")


# ── Results inspector ─────────────────────────────────────────────────────────
@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=300,
    volumes={str(PROC_MOUNT): vol_proc},
)
def results() -> None:
    """Print experiment comparison table and GQS stratification results."""
    import pandas as pd
    vol_proc.reload()

    cmp = RUN_DIR / "experiment_comparison_ubi.csv"
    if cmp.exists():
        df = pd.read_csv(str(cmp))
        print("\n=== V9 EXPERIMENT COMPARISON ===")
        base_cols = ["experiment", "fusion", "params",
                     "test_macro_f1", "test_roc_auc",
                     "test_accuracy", "test_threshold"]
        cal_cols  = ["test_cal_macro_f1", "test_cal_threshold", "test_cal_temp"]
        show = base_cols + [c for c in cal_cols if c in df.columns]
        print(df[show].to_string(index=False))
    else:
        print("No results yet. Run: modal run trigraph_v9_ubi_train.py::train")

    # GQS quartile F1 for E-QGF and E-LW
    for exp_id in ["E_full_qgf", "E_full_lw"]:
        q_csv = RUN_DIR / exp_id / "gqs_quartile_f1.csv"
        if q_csv.exists():
            print(f"\n=== GQS QUARTILE F1 — {exp_id} ===")
            print(pd.read_csv(str(q_csv)).to_string(index=False))

    # Overall GQS summary
    gqs_csv = PROC_MOUNT / "gqs_summary_ubi.csv"
    if gqs_csv.exists():
        gqs = pd.read_csv(str(gqs_csv))
        print("\n=== GQS MEANS BY SPLIT (V9 YOLO11x) ===")
        cols = ["q_skel","q_int","q_obj","q_po","valid_ratio"]
        print(gqs.groupby("split")[cols].mean().round(3).to_string())

    # VideoMAE standalone (Ablation C)
    vit_r = PROC_MOUNT / "videomae_test_results.json"
    if vit_r.exists():
        import json
        r = json.load(open(str(vit_r)))
        print("\n=== ABLATION C: VideoMAE standalone ===")
        for k, v in r.items():
            print(f"  {k:<15}: {v}")


# ── Download outputs to local machine ────────────────────────────────────────
# Runs inside Modal (volume mounted) then streams file bytes back to the
# local_entrypoint via return value — avoids any client-side volume path access.
@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=300,
    volumes={str(PROC_MOUNT): vol_proc},
)
def _collect_outputs() -> list:
    """
    Reads training artefacts from the volume and returns them as a list of
    (relative_path_str, bytes) tuples for the local entrypoint to write out.
    """
    vol_proc.reload()

    TOP_FILES = ["experiment_comparison_ubi.csv"]
    EXP_FILES = [
        "best.pt",
        "test_metrics.json",
        "train_history.csv",
        "gqs_quartile_f1.csv",
    ]
    EXPERIMENT_IDS = [
        "A_skeleton_only",
        "B_skel_interaction",
        "D_skel_int_obj",
        "E_full_qgf",
        "E_full_lw",
        "E_full_qgf_fixed",
    ]

    collected = []
    for fname in TOP_FILES:
        p = RUN_DIR / fname
        if p.exists():
            collected.append((fname, p.read_bytes()))

    for exp_id in EXPERIMENT_IDS:
        for fname in EXP_FILES:
            p = RUN_DIR / exp_id / fname
            if p.exists():
                collected.append((f"{exp_id}/{fname}", p.read_bytes()))

    return collected


def _download_to_local() -> None:
    """Pull outputs from Modal volume into ./training_output/ on local disk."""
    from pathlib import Path as LocalPath

    LOCAL_OUT = LocalPath(__file__).parent / "training_output"
    LOCAL_OUT.mkdir(exist_ok=True)

    print("Collecting files from Modal volume …")
    files = _collect_outputs.remote()

    if not files:
        print("No output files found. Has training completed?")
        return

    for rel_path, data in files:
        dst = LOCAL_OUT / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        size_kb = len(data) / 1024
        print(f"  OK  training_output/{rel_path}  ({size_kb:.1f} KB)")

    print(f"\nDone. {len(files)} files written to {LOCAL_OUT}")


# ── Local entrypoint — download only ─────────────────────────────────────────
@app.local_entrypoint()
def download() -> None:
    """Download all training outputs to ./training_output/ without re-running training."""
    print("Guardian Eye V9 — downloading training outputs ...")
    _download_to_local()


# ── Local entrypoint — train then auto-download ───────────────────────────────
@app.local_entrypoint()
def main() -> None:
    """
    Run training then automatically download all outputs to ./training_output/.

    Usage:
        modal run trigraph_v9_ubi_train.py
    """
    print("Guardian Eye V9 — UBI-Fights Phase 3: Training + auto-download")
    train.remote()

    print("\nTraining complete. Downloading outputs to training_output/ ...")
    _download_to_local()
