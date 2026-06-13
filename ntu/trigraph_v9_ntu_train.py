"""
trigraph_v9_ntu_train.py
Guardian Eye - V9  |  Phase 3: Model Training + Ablation Study
Dataset : NTU CCTV-Fights
Hardware: Local workstation - RTX 3090 (24 GB VRAM), 64 GB RAM

Architecture (V9) - IDENTICAL to the RLVS Phase 3 script (same V9Model, same
mean-over-heads GATv2, same 6 ablation experiments, same temperature calibration).
Only the data front-end changes:
  - paths -> NTU preproc/finetune/train output dirs
  - split CSV -> split_ntu.csv (one row per WINDOW; id column = clip_id)
  - NPZ carries an extra `source` scalar (CCTV/Mobile/Other/Car); the dataset
    reads it so the test loop can report source-stratified metrics (a paper
    contribution: pose-estimation degradation on Mobile vs CCTV).

NTU is balanced 1:1 by construction (Phase 1 downsamples negatives per split),
so pos_weight ~1.0 - still computed dynamically from train labels.

Resumability identical to RLVS: finished experiments (with test_metrics.json)
are skipped; an experiment killed mid-training resumes from last.pt.

Usage
-----
  python trigraph_v9_ntu_train.py
  python trigraph_v9_ntu_train.py --no-resume
"""

import argparse
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (f1_score, roc_auc_score,
                              accuracy_score, confusion_matrix)
from scipy.optimize import minimize_scalar
from tqdm import tqdm


# -- Paths ----------------------------------------------------------------------
PREPROC_DIR  = Path(r"C:\Violence detection\ashour\fresh\ntu\preproc_output")
FINETUNE_DIR = Path(r"C:\Violence detection\ashour\fresh\ntu\finetune_output")
OUTPUT_DIR   = Path(r"C:\Violence detection\ashour\fresh\ntu\train_output")

CACHE_DIR = PREPROC_DIR  / "cache_v9"
SPLIT_CSV = PREPROC_DIR  / "split_ntu.csv"
GQS_CSV   = PREPROC_DIR  / "gqs_summary_ntu.csv"
RUN_DIR   = OUTPUT_DIR   / "runs_ntu"


# -- Configuration --------------------------------------------------------------
@dataclass
class CFG:
    T:   int = 32
    M:   int = 6
    N:   int = 8
    V:   int = 17
    C:   int = 12   # multi-input skeleton channels

    embed_dim:      int   = 128
    graph_hidden:   int   = 64
    n_heads_gat:    int   = 4
    n_heads_trf:    int   = 4
    n_layers_trf:   int   = 2
    vit_embed_dim:  int   = 768
    dropout:        float = 0.35
    stream_dropout: float = 0.25

    lr:              float = 3e-4   # single-stream
    lr_multi:        float = 1e-4   # multi-stream
    weight_decay:    float = 1e-3
    grad_clip:       float = 1.0
    label_smoothing: float = 0.01
    use_label_smoothing: bool = True
    batch_size:      int   = 64
    num_workers:     int   = 0   # Windows multiprocessing spawn breaks with workers > 0
    epochs:          int   = 60
    patience:        int   = 12
    min_ckpt_epoch:  int   = 5

    thr_min:  float = 0.05
    thr_max:  float = 0.90
    thr_step: float = 0.01

    # Hard-negative exclusion (clip_id stems). NTU within-video negatives are
    # already hard by design; populate after first error_analysis if needed.
    hard_negative_clip_ids: tuple = ()

    resume: bool = True
    seed:   int  = 42


cfg = CFG()


# -- Reproducibility ------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


# ==============================================================================
# Dataset
# ==============================================================================
def _compute_bones(skel: np.ndarray) -> np.ndarray:
    """
    Compute bone vectors from COCO-17 kinematic tree.
    skel: [T, M, V, 3] (x, y, conf) -> returns [T, M, V, 3] (dx, dy, min_conf).
    """
    EDGES = [
        (0,1),(0,2),(1,3),(2,4),
        (5,7),(7,9),(6,8),(8,10),
        (5,6),(5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),
    ]
    bones = np.zeros_like(skel)
    for p, c in EDGES:
        bones[:, :, p, 0] = skel[:, :, c, 0] - skel[:, :, p, 0]
        bones[:, :, p, 1] = skel[:, :, c, 1] - skel[:, :, p, 1]
        bones[:, :, p, 2] = np.minimum(skel[:, :, c, 2], skel[:, :, p, 2])
    return bones


class NTUDataset(Dataset):
    """
    Loads all NPZ arrays into RAM.
    Derives multi-input skeleton channels (bone, motion) at load time.
    Carries `source` so the test loop can report CCTV-vs-Mobile metrics.
    """
    def __init__(self, npz_paths: List[Path], labels: List[int]):
        self.samples = []
        print(f"  Loading {len(npz_paths)} samples ...")
        missing_vit = 0
        for path, label in tqdm(zip(npz_paths, labels),
                                total=len(npz_paths),
                                desc="Loading", unit="file"):
            try:
                d    = np.load(str(path), allow_pickle=True)
                skel = d["skeleton"]   # [T, M, V, 3]

                joint = skel.copy()
                bone  = _compute_bones(skel)
                jm    = np.zeros_like(skel);  jm[:-1] = skel[1:] - skel[:-1]
                bm    = np.zeros_like(bone);  bm[:-1] = bone[1:] - bone[:-1]
                skel12 = np.concatenate([joint, bone, jm, bm],
                                        axis=-1).astype(np.float32)

                if "vit_embedding" in d:
                    vit_emb = torch.from_numpy(d["vit_embedding"].astype(np.float32))
                else:
                    vit_emb = torch.zeros(cfg.vit_embed_dim, dtype=torch.float32)
                    missing_vit += 1

                source = str(d["source"]) if "source" in d else "Unknown"

                self.samples.append({
                    "skeleton":      torch.from_numpy(skel12),
                    "int_nodes":     torch.from_numpy(d["int_nodes"].astype(np.float32)),
                    "int_edges":     torch.from_numpy(d["int_edges"].astype(np.float32)),
                    "int_node_mask": torch.from_numpy(d["int_node_mask"]),
                    "int_edge_mask": torch.from_numpy(d["int_edge_mask"]),
                    "obj_nodes":     torch.from_numpy(d["obj_nodes"].astype(np.float32)),
                    "obj_node_mask": torch.from_numpy(d["obj_node_mask"]),
                    "po_edges":      torch.from_numpy(d["po_edges"].astype(np.float32)),
                    "po_edge_mask":  torch.from_numpy(d["po_edge_mask"]),
                    "gqs":           torch.from_numpy(d["gqs"].astype(np.float32)),
                    "vit_embedding": vit_emb,
                    "label":         torch.tensor(float(label), dtype=torch.float32),
                    "clip_id":       str(path.stem),
                    "source":        source,
                })
            except Exception as e:
                print(f"  WARN: {path}: {e}")

        if missing_vit:
            print(f"  WARN: {missing_vit} samples missing vit_embedding "
                  "(run Phase 2 first). ViT stream will output zeros.")
        print(f"  Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_v9(batch):
    """Stack tensor fields; keep clip_id/source as python lists. Needed because
    the default collate cannot batch string fields, and the test loop wants
    per-sample source for stratified metrics."""
    out = {}
    tensor_keys = ["skeleton", "int_nodes", "int_edges", "int_node_mask",
                   "int_edge_mask", "obj_nodes", "obj_node_mask", "po_edges",
                   "po_edge_mask", "gqs", "vit_embedding", "label"]
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch])
    out["clip_id"] = [b["clip_id"] for b in batch]
    out["source"]  = [b["source"]  for b in batch]
    return out


# ==============================================================================
# Model components  (IDENTICAL to RLVS train.py - mean-over-heads GATv2)
# ==============================================================================

class GraphTemporalConv(nn.Module):
    def __init__(self, in_c: int, out_c: int,
                 t_kernel: int = 9, dropout: float = 0.35):
        super().__init__()
        self.gcn    = nn.Conv2d(in_c, out_c, 1)
        self.tcn    = nn.Conv2d(out_c, out_c,
                                kernel_size=(t_kernel, 1),
                                padding=(t_kernel // 2, 0),
                                groups=out_c)
        self.tcn_pw = nn.Conv2d(out_c, out_c, 1)
        self.bn     = nn.BatchNorm2d(out_c)
        self.drop   = nn.Dropout(dropout)
        self.res    = (nn.Conv2d(in_c, out_c, 1)
                       if in_c != out_c else nn.Identity())

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        xs = torch.einsum("bctv,vw->bctw", x, A)
        xs = self.gcn(xs)
        xs = self.tcn_pw(F.relu(self.tcn(xs)))
        xs = self.bn(xs)
        return self.drop(F.relu(xs + self.res(x)))


class EnhancedSTGCN(nn.Module):
    def __init__(self, C: int, T: int, V: int, M: int,
                 hidden: int, embed_dim: int, dropout: float):
        super().__init__()
        A_fix = self._make_coco17_adj(V)
        self.register_buffer("A_fixed", A_fix)
        self.A_learned  = nn.Parameter(torch.zeros(V, V))
        self.blocks     = nn.ModuleList([
            GraphTemporalConv(C,      hidden,    dropout=dropout),
            GraphTemporalConv(hidden, hidden,    dropout=dropout),
            GraphTemporalConv(hidden, embed_dim, dropout=dropout),
        ])
        self.person_attn = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.out_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _make_coco17_adj(V: int) -> torch.Tensor:
        edges = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(6,8),(7,9),(8,10),
                 (5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]
        A = torch.zeros(V, V)
        for i, j in edges:
            A[i, j] = A[j, i] = 1.0
        A.fill_diagonal_(1.0)
        D = A.sum(-1, keepdim=True).clamp(min=1.0).sqrt()
        return A / D / D.T

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        B, T, M, V, C = x.shape
        A   = self.A_fixed + F.softmax(self.A_learned, dim=-1)
        xr  = x.permute(0, 2, 4, 1, 3).reshape(B * M, C, T, V)
        for blk in self.blocks:
            xr = blk(xr, A)
        emb       = xr.mean(dim=[2, 3]).view(B, M, -1)       # [B, M, D]
        valid     = node_mask.any(dim=1)                       # [B, M]
        emb_m     = emb * valid.unsqueeze(-1).float()
        scores    = self.person_attn(emb_m).squeeze(-1)
        scores    = scores.masked_fill(~valid, -1e9)
        weights   = F.softmax(scores, dim=-1).unsqueeze(-1)
        return self.out_norm((emb_m * weights).sum(1))


class GATv2Layer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int,
                 out_dim: int, heads: int = 4, dropout: float = 0.35):
        super().__init__()
        self.heads  = heads
        self.d      = out_dim // heads
        self.Wq     = nn.Linear(node_dim, out_dim)
        self.Wk     = nn.Linear(node_dim, out_dim)
        self.Wv     = nn.Linear(node_dim, out_dim)
        self.We     = nn.Linear(edge_dim,  out_dim)
        self.attn_v = nn.Linear(out_dim, heads)
        self.out    = nn.Linear(out_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor,
                edges: torch.Tensor,
                mask:  torch.Tensor) -> torch.Tensor:
        B, M, _ = nodes.shape
        Q = self.Wq(nodes); K = self.Wk(nodes); V = self.Wv(nodes)
        E = self.We(edges)
        a = self.attn_v(F.leaky_relu(
            Q.unsqueeze(2) + K.unsqueeze(1) + E,
            negative_slope=0.2))                       # [B, M, M, heads]
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(1).unsqueeze(-1), -1e9)
        a   = F.softmax(a, dim=2)
        # Mean-over-heads aggregation (matches RLVS-trained checkpoints).
        agg = (a.mean(-1, keepdim=True) * V.unsqueeze(1)).sum(2)
        return self.norm(self.drop(self.out(agg)) + V)


class ImprovedInteraction(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int,
                 embed_dim: int, n_heads_gat: int,
                 n_heads_trf: int, n_layers_trf: int,
                 T: int, M: int, dropout: float):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, embed_dim),
            nn.LayerNorm(embed_dim), nn.ReLU())
        self.gat1    = GATv2Layer(embed_dim, edge_dim, embed_dim,
                                  n_heads_gat, dropout)
        self.gat2    = GATv2Layer(embed_dim, edge_dim, embed_dim,
                                  n_heads_gat, dropout)
        self.pos_enc = nn.Parameter(torch.randn(1, T, embed_dim) * 0.02)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads_trf,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                  num_layers=n_layers_trf)
        self.tpool    = nn.Linear(embed_dim, 1)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, nodes: torch.Tensor,
                edges: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        B, T, M, _ = nodes.shape
        frame_embs = []
        for t in range(T):
            n_t   = self.node_proj(nodes[:, t])
            e_t   = edges[:, t]
            mk_t  = node_mask[:, t]
            h     = self.gat1(n_t, e_t, mk_t)
            h     = self.gat2(h,   e_t, mk_t)
            valid = mk_t.unsqueeze(-1).float()
            fe    = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
            frame_embs.append(fe)
        seq    = torch.stack(frame_embs, dim=1) + self.pos_enc
        out    = self.transformer(seq)
        w      = F.softmax(self.tpool(out), dim=1)
        return self.out_norm((out * w).sum(1))


class FrameGINE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int,
                 hidden: int, layers: int = 2, dropout: float = 0.35):
        super().__init__()
        self.node_in  = nn.Linear(node_dim, hidden)
        self.edge_enc = nn.Linear(edge_dim,  hidden)
        self.mlps     = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden),
                          nn.LayerNorm(hidden), nn.ReLU(),
                          nn.Dropout(dropout))
            for _ in range(layers)])
        self.out = nn.Linear(hidden, hidden)

    def forward(self, nodes: torch.Tensor,
                edges: torch.Tensor,
                node_mask: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                bipartite: bool = False) -> torch.Tensor:
        x = self.node_in(nodes)
        for mlp in self.mlps:
            e_e = self.edge_enc(edges)
            if bipartite:
                sm  = (src_mask.unsqueeze(-1).unsqueeze(-1).float()
                       if src_mask is not None
                       else torch.ones(x.shape[0], e_e.shape[1], 1, 1,
                                       device=nodes.device))
                agg = (e_e * sm).sum(1)
            else:
                nm  = (node_mask.unsqueeze(-1).unsqueeze(-1).float()
                       if node_mask is not None
                       else torch.ones(x.shape[0], e_e.shape[1], 1, 1,
                                       device=nodes.device))
                agg = (e_e * nm).sum(2)
            x = mlp(x + agg)
        nm2 = (node_mask.unsqueeze(-1).float()
               if node_mask is not None
               else torch.ones(x.shape[0], x.shape[1], 1, device=nodes.device))
        return (self.out(x) * nm2).sum(1) / nm2.sum(1).clamp(min=1)


class TemporalBiGRU(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, dropout: float = 0.35):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gru  = nn.GRU(embed_dim, embed_dim // 2,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru(self.proj(seq))
        w     = F.softmax(self.attn(x), dim=1)
        return self.norm((x * w).sum(1))


class ObjectPOStream(nn.Module):
    def __init__(self, obj_node_dim: int, int_edge_dim: int,
                 po_edge_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        half = embed_dim // 2
        self.obj_edge_dim = int_edge_dim
        self.obj_gine = FrameGINE(obj_node_dim, int_edge_dim,
                                   half, layers=2, dropout=dropout)
        self.obj_gru  = TemporalBiGRU(half, half, dropout)
        self.po_gine  = FrameGINE(obj_node_dim, po_edge_dim,
                                   half, layers=2, dropout=dropout)
        self.po_gru   = TemporalBiGRU(half, half, dropout)
        self.proj     = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim), nn.ReLU())

    def forward(self, obj_n, obj_nm, po_e, po_em, int_nm) -> torch.Tensor:
        B, T, N, _ = obj_n.shape
        obj_frames, po_frames = [], []
        for t in range(T):
            on_t  = obj_n[:, t];   onm_t = obj_nm[:, t]
            pe_t  = po_e[:, t];    inm_t = int_nm[:, t]
            dummy = torch.zeros(B, N, N, self.obj_edge_dim, device=obj_n.device)
            obj_frames.append(self.obj_gine(on_t, dummy, onm_t))
            po_frames.append(self.po_gine(on_t, pe_t, onm_t,
                                          src_mask=inm_t, bipartite=True))
        e_obj = self.obj_gru(torch.stack(obj_frames, dim=1))
        e_po  = self.po_gru(torch.stack(po_frames,  dim=1))
        return self.proj(torch.cat([e_obj, e_po], dim=-1))


class VitProjection(nn.Module):
    def __init__(self, vit_dim: int, embed_dim: int, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QualityGatedFusion(nn.Module):
    def __init__(self, n_streams: int = 4, stream_dropout: float = 0.25,
                 gate_temperature: float = 1.0, entropy_weight: float = 0.0,
                 temp_anneal: bool = False):
        super().__init__()
        self.n = n_streams; self.p = stream_dropout
        self.entropy_weight = entropy_weight
        self.temp_anneal    = temp_anneal
        self.gate = nn.Sequential(
            nn.LayerNorm(n_streams),
            nn.Linear(n_streams, 32), nn.ReLU(),
            nn.Linear(32, n_streams))
        self.register_buffer("current_temp",
                             torch.tensor(gate_temperature), persistent=False)
        self.return_gates: bool = False

    def forward(self, streams: List[torch.Tensor],
                gqs: torch.Tensor):
        stk         = torch.stack(streams, dim=1)
        gate_logits = self.gate(gqs[:, :self.n])
        gates       = F.softmax(gate_logits / self.current_temp, dim=-1)
        if self.training and self.p > 0:
            B    = stk.shape[0]
            mask = torch.ones(B, self.n, 1, device=stk.device)
            drop = torch.rand(B, self.n, device=stk.device) < self.p
            drop = drop & ~drop.all(dim=1, keepdim=True)
            mask[drop] = 0.0
            gates = gates * mask.squeeze(-1)
            gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
        if self.training:
            self.last_entropy = -(gates * (gates + 1e-8).log()).sum(-1).mean()
        else:
            self.last_entropy = None
        fused = (stk * gates.unsqueeze(-1)).sum(1)
        if self.return_gates:
            return fused, gates
        return fused


class LearnedWeightFusion(nn.Module):
    def __init__(self, n_streams: int = 4, stream_dropout: float = 0.25):
        super().__init__()
        self.n = n_streams; self.p = stream_dropout
        self.weights = nn.Parameter(torch.zeros(n_streams))

    def forward(self, streams: List[torch.Tensor],
                gqs: torch.Tensor) -> torch.Tensor:
        stk   = torch.stack(streams, dim=1)
        gates = F.softmax(self.weights, dim=0).unsqueeze(0).expand(stk.shape[0], -1)
        if self.training and self.p > 0:
            B    = stk.shape[0]
            mask = torch.ones(B, self.n, 1, device=stk.device)
            drop = torch.rand(B, self.n, device=stk.device) < self.p
            drop = drop & ~drop.all(dim=1, keepdim=True)
            mask[drop] = 0.0
            gates = gates * mask.squeeze(-1)
            gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
        return (stk * gates.unsqueeze(-1)).sum(1)


class V9Model(nn.Module):
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
                 entropy_weight:   float = 0.0,
                 temp_anneal:      bool  = False):
        super().__init__()
        self.active = active_streams
        self.fmode  = fusion_mode
        n_active    = len(active_streams)

        if "skeleton"    in active_streams:
            self.stgcn       = EnhancedSTGCN(C, T, V, M, hidden, embed_dim, dropout)
        if "interaction" in active_streams:
            self.interaction = ImprovedInteraction(int_nd, int_ed, embed_dim,
                                                   n_heads_gat, n_heads_trf,
                                                   n_layers_trf, T, M, dropout)
        if "object"      in active_streams:
            self.obj_po      = ObjectPOStream(obj_nd, int_ed, po_ed,
                                              embed_dim, dropout)
        if "vit"         in active_streams:
            self.vit_proj    = VitProjection(vit_dim, embed_dim, dropout)

        if n_active > 1:
            if fusion_mode == "qgf":
                self.fusion = QualityGatedFusion(n_active, stream_dropout,
                                                 gate_temperature, entropy_weight,
                                                 temp_anneal)
            else:
                self.fusion = LearnedWeightFusion(n_active, stream_dropout)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def get_gate_entropy(self):
        if hasattr(self, "fusion") and hasattr(self.fusion, "last_entropy"):
            return self.fusion.last_entropy
        return None

    def forward(self, batch: Dict, return_gates: bool = False):
        streams_out = []
        if "skeleton"    in self.active:
            streams_out.append(self.stgcn(batch["skeleton"],
                                          batch["int_node_mask"]))
        if "interaction" in self.active:
            streams_out.append(self.interaction(batch["int_nodes"],
                                                batch["int_edges"],
                                                batch["int_node_mask"]))
        if "object"      in self.active:
            streams_out.append(self.obj_po(batch["obj_nodes"],
                                           batch["obj_node_mask"],
                                           batch["po_edges"],
                                           batch["po_edge_mask"],
                                           batch["int_node_mask"]))
        if "vit"         in self.active:
            streams_out.append(self.vit_proj(batch["vit_embedding"]))

        gate_weights_out = None
        if len(streams_out) == 1:
            fused = streams_out[0]
        else:
            if return_gates and hasattr(self.fusion, "return_gates"):
                self.fusion.return_gates = True
                result = self.fusion(streams_out, batch["gqs"])
                self.fusion.return_gates = False
                if isinstance(result, tuple):
                    fused, gate_weights_out = result
                else:
                    fused = result
            else:
                fused = self.fusion(streams_out, batch["gqs"])

        logits = self.head(fused).squeeze(-1)
        if return_gates:
            return logits, gate_weights_out
        return logits


# ==============================================================================
# Training helpers
# ==============================================================================
def find_threshold(y_true, y_prob) -> Tuple[float, float]:
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(cfg.thr_min, cfg.thr_max + 1e-6, cfg.thr_step):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


@torch.no_grad()
def evaluate(model, loader, device, want_meta=False):
    model.eval()
    y_true, y_prob = [], []
    sources, clip_ids = [], []
    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        probs  = torch.sigmoid(model(batch_dev)).cpu().numpy()
        labels = batch_dev["label"].cpu().numpy().astype(int)
        y_prob.extend(probs.tolist())
        y_true.extend(labels.tolist())
        if want_meta:
            sources.extend(batch["source"])
            clip_ids.extend(batch["clip_id"])
    if want_meta:
        return (np.array(y_true), np.array(y_prob),
                np.array(sources), np.array(clip_ids))
    return np.array(y_true), np.array(y_prob)


def metrics_at_threshold(y_true, y_prob, thr) -> Dict:
    preds = (y_prob >= thr).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy":        float(accuracy_score(y_true, preds)),
        "macro_f1":        float(f1_score(y_true, preds, average="macro",
                                           zero_division=0)),
        "binary_f1":       float(f1_score(y_true, preds, average="binary",
                                           zero_division=0)),
        "roc_auc":         auc,
        "threshold":       float(thr),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }


# ==============================================================================
# Main
# ==============================================================================
def main(args) -> None:
    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Preproc : {PREPROC_DIR}")
    print(f"Output  : {OUTPUT_DIR}")

    if not SPLIT_CSV.exists():
        raise RuntimeError(
            f"split_ntu.csv not found at {SPLIT_CSV}. "
            "Run Phase 1 (trigraph_v9_ntu_preprocess.py) first."
        )
    split_df = pd.read_csv(str(SPLIT_CSV))

    def get_split(name: str):
        sub    = split_df[split_df["split"] == name]
        paths  = [CACHE_DIR / f"{cid}.npz" for cid in sub["clip_id"]]
        labels = sub["label"].tolist()
        if name == "train" and cfg.hard_negative_clip_ids:
            before = len(paths)
            pairs  = [(p, l) for p, l in zip(paths, labels)
                      if p.exists() and not any(
                          hn in p.stem for hn in cfg.hard_negative_clip_ids)]
            excl   = before - len(pairs)
            if excl:
                print(f"  Hard-negative exclusion: removed {excl} train clips")
            paths, labels = zip(*pairs) if pairs else ([], [])
            return list(paths), list(labels)
        return ([p for p in paths if p.exists()],
                [l for p, l in zip(paths, labels) if p.exists()])

    train_paths, train_labels = get_split("train")
    val_paths,   val_labels   = get_split("val")
    test_paths,  test_labels  = get_split("test")

    if len(test_paths) == 0:
        raise RuntimeError(
            "No test split found in split_ntu.csv. "
            "Check that Phase 1 completed successfully."
        )
    print(f"Train={len(train_paths)}  Val={len(val_paths)}  "
          f"Test={len(test_paths)}")

    train_ds = NTUDataset(train_paths, train_labels)
    val_ds   = NTUDataset(val_paths,   val_labels)
    test_ds  = NTUDataset(test_paths,  test_labels)

    # Dynamic pos_weight - NTU is balanced 1:1 so this will be ~1.0
    n_pos   = sum(train_labels)
    n_neg   = len(train_labels) - n_pos
    pw_val  = n_neg / max(n_pos, 1)
    print(f"  pos_weight = {pw_val:.3f}  (n_neg={n_neg}, n_pos={n_pos})")

    sample_weights = torch.tensor(
        [pw_val if l == 1 else 1.0 for l in train_labels], dtype=torch.float64)
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    def make_loader(ds, sampler=None, shuffle=False) -> DataLoader:
        return DataLoader(ds, batch_size=cfg.batch_size,
                          sampler=sampler,
                          shuffle=(shuffle if sampler is None else False),
                          num_workers=cfg.num_workers,
                          pin_memory=True, collate_fn=collate_v9)

    train_loader = make_loader(train_ds, sampler=sampler)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    pw = torch.tensor([pw_val], device=DEVICE)

    def smooth_bce(logits, targets, ls=None):
        ls = cfg.label_smoothing if ls is None else ls
        if ls == 0.0 or not cfg.use_label_smoothing:
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        t = targets * (1.0 - ls) + (1.0 - targets) * ls
        return F.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)

    s0     = train_ds[0]
    int_nd = s0["int_nodes"].shape[-1]   # 7
    int_ed = s0["int_edges"].shape[-1]   # 4
    obj_nd = s0["obj_nodes"].shape[-1]   # 6
    po_ed  = s0["po_edges"].shape[-1]    # 5

    EXPERIMENTS = [
        ("A_skeleton_only",    ("skeleton",),
         "qgf", {}),
        ("B_skel_interaction", ("skeleton", "interaction"),
         "qgf", {}),
        ("D_skel_int_obj",     ("skeleton", "interaction", "object"),
         "qgf", {}),
        ("E_full_qgf",         ("skeleton", "interaction", "object", "vit"),
         "qgf", {}),
        ("E_full_lw",          ("skeleton", "interaction", "object", "vit"),
         "lw",  {}),
        ("E_full_qgf_fixed",   ("skeleton", "interaction", "object", "vit"),
         "qgf", {"entropy_weight":  0.10,
                 "temp_anneal":     True,
                 "label_smoothing": 0.0}),
    ]

    results_rows = []
    resume = args.resume or cfg.resume

    for exp_id, active_streams, fmode, ovr in EXPERIMENTS:
        try:
            seed_everything(cfg.seed)
            exp_dir = RUN_DIR / exp_id
            exp_dir.mkdir(exist_ok=True)

            done_marker = exp_dir / "test_metrics.json"
            ckpt_path   = exp_dir / "best.pt"
            if resume and done_marker.exists():
                try:
                    with open(str(done_marker)) as f:
                        test_m_done = json.load(f)
                    best_val_f1_done = float("nan")
                    if ckpt_path.exists():
                        ck_done = torch.load(str(ckpt_path), map_location="cpu")
                        best_val_f1_done = float(
                            ck_done.get("val_macro_f1", float("nan")))
                    results_rows.append({
                        "experiment":  exp_id,
                        "streams":     str(active_streams),
                        "fusion":      fmode,
                        "params":      None,
                        "best_val_f1": round(best_val_f1_done, 4)
                            if best_val_f1_done == best_val_f1_done
                            else "resumed",
                        **{f"test_{k}": v for k, v in test_m_done.items()
                           if k != "confusion_matrix"},
                    })
                    print(f"\n[RESUME] {exp_id} done - "
                          f"test_macro_f1={test_m_done.get('macro_f1', float('nan')):.4f}")
                    continue
                except Exception as e:
                    print(f"[RESUME] {exp_id} marker unreadable ({e}); retraining.")

            exp_entropy  = ovr.get("entropy_weight",  0.0)
            exp_gate_tmp = ovr.get("gate_temperature", 1.0)
            exp_t_anneal = ovr.get("temp_anneal",      False)
            exp_ls       = ovr.get("label_smoothing",  cfg.label_smoothing)

            n_active = len(active_streams)
            eff_lr   = cfg.lr if n_active == 1 else cfg.lr_multi

            print(f"\n{'='*70}")
            print(f"Experiment : {exp_id}")
            print(f"Streams    : {active_streams}")
            print(f"Fusion     : {fmode}  |  LR: {eff_lr:.1e}")
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
                gate_temperature=exp_gate_tmp,
                entropy_weight  =exp_entropy,
                temp_anneal     =exp_t_anneal,
            ).to(DEVICE)

            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
            print(f"Trainable params: {n_params:,}")

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=eff_lr,
                weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=4)

            best_val_f1  = -1.0
            patience_cnt = 0
            history      = []
            start_epoch  = 1
            last_path    = exp_dir / "last.pt"

            if resume and last_path.exists():
                try:
                    lck = torch.load(str(last_path), map_location=DEVICE)
                    model.load_state_dict(lck["model_state_dict"])
                    optimizer.load_state_dict(lck["optimizer_state_dict"])
                    scheduler.load_state_dict(lck["scheduler_state_dict"])
                    start_epoch  = lck["epoch"] + 1
                    best_val_f1  = lck["best_val_f1"]
                    patience_cnt = lck["patience_cnt"]
                    history      = lck.get("history", [])
                    print(f"  [RESUME] {exp_id}: continuing from epoch "
                          f"{start_epoch} (best_val_f1={best_val_f1:.4f}, "
                          f"patience={patience_cnt})")
                except Exception as e:
                    print(f"  [RESUME] {exp_id}: last.pt unreadable ({e}); "
                          "starting fresh.")

            for epoch in range(start_epoch, cfg.epochs + 1):
                if exp_t_anneal and hasattr(model, "fusion"):
                    T_val = max(0.5, 2.0 - 1.5 * (epoch - 1)
                                / max(cfg.epochs - 1, 1))
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
                    loss   = smooth_bce(logits, batch["label"], ls=exp_ls)
                    entropy = model.get_gate_entropy()
                    if entropy is not None and exp_entropy > 0:
                        loss = loss - exp_entropy * entropy
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                avg_loss = epoch_loss / max(len(train_loader), 1)
                y_tv, y_pv = evaluate(model, val_loader, DEVICE)
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

                if epoch >= cfg.min_ckpt_epoch and val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save({
                        "epoch": epoch,
                        "state_dict":   model.state_dict(),
                        "val_macro_f1": val_f1,
                        "val_roc_auc":  val_auc,
                        "threshold":    val_thr,
                    }, str(ckpt_path))
                    patience_cnt = 0
                elif epoch >= cfg.min_ckpt_epoch:
                    patience_cnt += 1

                tmp_last = last_path.with_suffix(".tmp.pt")
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1":          best_val_f1,
                    "patience_cnt":         patience_cnt,
                    "history":              history,
                }, str(tmp_last))
                tmp_last.replace(last_path)

                pd.DataFrame(history).to_csv(
                    str(exp_dir / "train_history.csv"), index=False)

                if patience_cnt >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

            if not ckpt_path.exists():
                print(f"  WARN: no checkpoint for {exp_id}; skipping test.")
                continue

            ck = torch.load(str(ckpt_path), map_location=DEVICE)
            model.load_state_dict(ck["state_dict"])
            best_thr = float(ck["threshold"])
            print(f"\n  Loaded best ckpt ep={ck['epoch']} "
                  f"val_f1={ck['val_macro_f1']:.4f}")

            # Test eval - also collect per-sample source for stratified metrics
            y_tt, y_pt, src_tt, cid_tt = evaluate(model, test_loader, DEVICE,
                                                  want_meta=True)

            # Temperature calibration on val set
            y_tv_cal, y_pv_cal = evaluate(model, val_loader, DEVICE)
            eps_cal = 1e-7

            def nll_t(temp):
                p = np.clip(y_pv_cal, eps_cal, 1 - eps_cal)
                logit = np.log(p / (1 - p)) / max(temp, 1e-3)
                p_t   = 1 / (1 + np.exp(-logit))
                return -np.mean(
                    y_tv_cal * np.log(p_t + eps_cal)
                    + (1 - y_tv_cal) * np.log(1 - p_t + eps_cal))

            opt_t    = minimize_scalar(nll_t, bounds=(0.5, 3.0), method="bounded")
            cal_temp = float(opt_t.x)

            def apply_temp(probs, temp):
                p     = np.clip(probs, eps_cal, 1 - eps_cal)
                logit = np.log(p / (1 - p)) / max(temp, 1e-3)
                return 1 / (1 + np.exp(-logit))

            y_pv_scaled = apply_temp(y_pv_cal, cal_temp)
            cal_thr, cal_f1 = find_threshold(y_tv_cal, y_pv_scaled)
            y_pt_scaled     = apply_temp(y_pt, cal_temp)

            print(f"  Calibration: T*={cal_temp:.4f}  "
                  f"cal_thr={cal_thr:.2f}  cal_val_f1={cal_f1:.4f}  "
                  f"(raw thr={best_thr:.2f})")

            test_m     = metrics_at_threshold(y_tt, y_pt, best_thr)
            test_m_cal = metrics_at_threshold(y_tt, y_pt_scaled, cal_thr)
            m50 = metrics_at_threshold(y_tt, y_pt, 0.5)
            test_m["macro_f1_thr050"]  = m50["macro_f1"]
            test_m["accuracy_thr050"]  = m50["accuracy"]
            test_m["cal_macro_f1"]     = test_m_cal["macro_f1"]
            test_m["cal_accuracy"]     = test_m_cal["accuracy"]
            test_m["cal_roc_auc"]      = test_m_cal["roc_auc"]
            test_m["cal_threshold"]    = cal_thr
            test_m["cal_temp"]         = cal_temp

            print(f"\n  TEST (raw) - Macro-F1={test_m['macro_f1']:.4f}  "
                  f"ROC-AUC={test_m['roc_auc']:.4f}  "
                  f"Acc={test_m['accuracy']:.4f}  Thr={best_thr:.2f}")
            print(f"  TEST (cal) - Macro-F1={test_m['cal_macro_f1']:.4f}  "
                  f"ROC-AUC={test_m['cal_roc_auc']:.4f}  "
                  f"Acc={test_m['cal_accuracy']:.4f}  Thr={cal_thr:.2f}")
            print(f"  CM: {test_m['confusion_matrix']}")

            # -- Source-stratified test metrics (NTU paper contribution) -------
            # Quantifies pose-estimation degradation on Mobile vs CCTV footage.
            src_rows = []
            for src_name in sorted(set(src_tt.tolist())):
                mask = (src_tt == src_name)
                if mask.sum() == 0:
                    continue
                yt_s = y_tt[mask]; yp_s = y_pt[mask]
                if len(set(yt_s.tolist())) < 2:
                    # single-class subset: AUC undefined, still report F1/acc
                    auc_s = float("nan")
                else:
                    try:
                        auc_s = float(roc_auc_score(yt_s, yp_s))
                    except ValueError:
                        auc_s = float("nan")
                preds_s = (yp_s >= best_thr).astype(int)
                src_rows.append({
                    "experiment": exp_id,
                    "source": src_name,
                    "n": int(mask.sum()),
                    "macro_f1": float(f1_score(yt_s, preds_s, average="macro",
                                               zero_division=0)),
                    "accuracy": float(accuracy_score(yt_s, preds_s)),
                    "roc_auc": auc_s,
                })
            if src_rows:
                pd.DataFrame(src_rows).to_csv(
                    str(exp_dir / "source_stratified_metrics.csv"), index=False)
                print("\n  Source-stratified test metrics:")
                print(pd.DataFrame(src_rows)[
                    ["source", "n", "macro_f1", "roc_auc"]].to_string(index=False))

            # -- Per-clip test predictions (for diagnostics / RAG handoff) -----
            pd.DataFrame({
                "clip_id": cid_tt, "source": src_tt,
                "true_label": y_tt, "pred_prob": y_pt,
                "pred_label": (y_pt >= best_thr).astype(int),
            }).to_csv(str(exp_dir / "test_predictions.csv"), index=False)

            # GQS stratification analysis
            if GQS_CSV.exists() and ("qgf" in exp_id or "lw" in exp_id):
                try:
                    gqs_df   = pd.read_csv(str(GQS_CSV))
                    test_sub = gqs_df[gqs_df["split"] == "test"].copy()
                    if len(test_sub) > 0:
                        test_sub["gqs_composite"] = (
                            0.4 * test_sub["q_skel"] +
                            0.4 * test_sub["q_int"]  +
                            0.2 * test_sub["q_po"])
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
                        if n_bins >= 2:
                            vid_to_idx = {
                                test_ds.samples[i]["clip_id"]: i
                                for i in range(len(test_ds))
                            }
                            # Map clip_id -> position in y_tt/y_pt arrays.
                            cid_to_pos = {c: i for i, c in enumerate(cid_tt)}
                            q_rows = []
                            for bucket, grp in test_sub.groupby(
                                    "q_bucket", observed=True):
                                poss = [cid_to_pos[v]
                                        for v in grp["clip_id"].tolist()
                                        if v in cid_to_pos]
                                if not poss:
                                    continue
                                yt  = y_tt[poss]; yp = y_pt[poss]
                                pr  = (yp >= best_thr).astype(int)
                                f1q = float(f1_score(yt, pr, average="macro",
                                                     zero_division=0))
                                try:
                                    aucq = float(roc_auc_score(yt, yp))
                                except Exception:
                                    aucq = float("nan")
                                q_rows.append({
                                    "experiment": exp_id,
                                    "bucket": str(bucket),
                                    "n": len(poss),
                                    "macro_f1": f1q,
                                    "roc_auc":  aucq,
                                })
                            pd.DataFrame(q_rows).to_csv(
                                str(exp_dir / "gqs_quartile_f1.csv"),
                                index=False)
                            print("  GQS quartile F1 saved.")
                except Exception as e:
                    print(f"  WARN: GQS stratification failed: {e}")

            pd.DataFrame(history).to_csv(
                str(exp_dir / "train_history.csv"), index=False)
            with open(str(done_marker), "w") as f:
                json.dump(test_m, f, indent=2)

            if last_path.exists():
                try:
                    last_path.unlink()
                except Exception:
                    pass

            results_rows.append({
                "experiment":  exp_id,
                "streams":     str(active_streams),
                "fusion":      fmode,
                "params":      n_params,
                "best_val_f1": round(best_val_f1, 4),
                **{f"test_{k}": v for k, v in test_m.items()
                   if k != "confusion_matrix"},
            })

        except Exception as exc:
            print(f"  ERROR in experiment {exp_id}: {exc}")
            raise

    # -- Final comparison table -----------------------------------------------
    vit_results_path = FINETUNE_DIR / "videomae_test_results.json"
    if vit_results_path.exists():
        with open(str(vit_results_path)) as f:
            vit_r = json.load(f)
        allowed = {"accuracy", "macro_f1", "roc_auc", "threshold"}
        results_rows.insert(2, {
            "experiment":  "C_videomae_only",
            "streams":     "('vit',)",
            "fusion":      "none",
            "params":      86_000_000,
            "best_val_f1": "see_phase2_log",
            **{f"test_{k}": vit_r[k] for k in vit_r if k in allowed},
        })

    results_df  = pd.DataFrame(results_rows)
    results_csv = RUN_DIR / "experiment_comparison_ntu.csv"
    results_df.to_csv(str(results_csv), index=False)

    print(f"\n{'='*70}")
    print("NTU CCTV-FIGHTS ABLATION COMPARISON")
    show_cols = ["experiment", "test_macro_f1", "test_roc_auc",
                 "test_accuracy", "test_threshold"]
    cal_cols  = ["test_cal_macro_f1", "test_cal_threshold", "test_cal_temp"]
    show_cols += [c for c in cal_cols if c in results_df.columns]
    print(results_df[show_cols].to_string(index=False))
    print(f"{'='*70}")
    print(f"\nFull results -> {results_csv}")
    print("Phase 3 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTU CCTV-Fights V9 Training")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments whose test_metrics.json already exists")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force retraining all experiments from scratch")
    args = parser.parse_args()
    if args.no_resume:
        cfg.resume = False
    main(args)
