"""
trigraph_v9_rwf_diagnostics.py
Guardian Eye — V9  |  Post-hoc Diagnostics
Dataset : RWF-2000
GPU     : T4  (inference only — no retraining)

Runs all four diagnostics in one command and downloads results locally:

  conda run -n dlcuda128 modal run trigraph_v9_rwf_diagnostics.py

Output files land in ./diagnostics_output/ on your machine:
  gate_weights_E_full_qgf.csv
  gqs_stratified_comparison.csv
  calibration_results.json
  error_analysis.csv

Individual functions can still be called separately if needed:
  conda run -n dlcuda128 modal run trigraph_v9_rwf_diagnostics.py::gate_weights
  conda run -n dlcuda128 modal run trigraph_v9_rwf_diagnostics.py::gqs_stratified
  conda run -n dlcuda128 modal run trigraph_v9_rwf_diagnostics.py::calibration
  conda run -n dlcuda128 modal run trigraph_v9_rwf_diagnostics.py::error_analysis
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import modal

# ── Modal primitives ──────────────────────────────────────────────────────────
APP_NAME      = "guardian-eye-v9-rwf-diagnostics"
VOL_NAME_PROC = "rwf-2000-processed"

app      = modal.App(APP_NAME)
vol_proc = modal.Volume.from_name(VOL_NAME_PROC, create_if_missing=True)

PROC_MOUNT  = Path("/data/proc")
CACHE_DIR   = PROC_MOUNT / "cache_v9"
SPLIT_CSV   = PROC_MOUNT / "split_v9.csv"
RUN_DIR     = PROC_MOUNT / "runs_v9"
DIAG_DIR    = RUN_DIR / "diagnostics"

# ── Container image (identical pip_install block to trigraph_v9_rwf_train.py) ─
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

# ── Shared function decorator kwargs ─────────────────────────────────────────
_FUNC_KWARGS = dict(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    timeout=900,
    volumes={str(PROC_MOUNT): vol_proc},
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration  (verbatim copy from trigraph_v9_rwf_train.py)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class CFG:
    # ── Data dimensions (must match Phase 1 constants) ────────────────────────
    T:   int = 32
    M:   int = 6
    N:   int = 8
    V:   int = 17
    C:   int = 12

    # ── Model ─────────────────────────────────────────────────────────────────
    embed_dim:      int   = 128
    graph_hidden:   int   = 64
    n_heads_gat:    int   = 4
    n_heads_trf:    int   = 4
    n_layers_trf:   int   = 2
    vit_embed_dim:  int   = 768
    dropout:        float = 0.35
    stream_dropout: float = 0.25

    # ── Training ──────────────────────────────────────────────────────────────
    lr:              float = 3e-4
    lr_multi:        float = 1e-4
    weight_decay:    float = 1e-3
    grad_clip:       float = 1.0
    label_smoothing: float = 0.01
    pos_weight:      float = 1.0
    batch_size:      int   = 64
    num_workers:     int   = 0
    epochs:          int   = 60
    patience:        int   = 12
    min_ckpt_epoch:  int   = 5

    # ── Threshold search ──────────────────────────────────────────────────────
    thr_min:  float = 0.10
    thr_max:  float = 0.90
    thr_step: float = 0.02

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
# Dataset  (verbatim copy from trigraph_v9_rwf_train.py)
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset_class():
    import torch
    from torch.utils.data import Dataset
    import numpy as np
    from tqdm import tqdm

    class RWFDataset(Dataset):
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

                    joint        = skel.copy()
                    bone         = _compute_bones(skel)
                    jm           = np.zeros_like(skel)
                    jm[:-1]      = skel[1:] - skel[:-1]
                    bm           = np.zeros_like(bone)
                    bm[:-1]      = bone[1:] - bone[:-1]
                    skel12       = np.concatenate(
                        [joint, bone, jm, bm], axis=-1).astype(np.float32)

                    if "vit_embedding" in d:
                        vit_emb = torch.from_numpy(
                            d["vit_embedding"].astype(np.float32))
                    else:
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
        EDGES = [
            (0,1),(0,2),(1,3),(2,4),
            (5,7),(7,9),(6,8),(8,10),
            (5,6),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),
        ]
        bones = np.zeros_like(skel)
        for (p, c) in EDGES:
            dx  = skel[:, :, c, 0] - skel[:, :, p, 0]
            dy  = skel[:, :, c, 1] - skel[:, :, p, 1]
            cf  = np.minimum(skel[:, :, c, 2], skel[:, :, p, 2])
            bones[:, :, p, 0] = dx
            bones[:, :, p, 1] = dy
            bones[:, :, p, 2] = cf
        return bones

    return RWFDataset


# ══════════════════════════════════════════════════════════════════════════════
# Model components  (verbatim copy from trigraph_v9_rwf_train.py,
# with QualityGatedFusion extended to support return_gates flag)
# ══════════════════════════════════════════════════════════════════════════════
def build_model_components():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    # ─────────────────────────────────────────────────────────────────────────
    # Stream 1 — Enhanced Multi-Input STGCN
    # ─────────────────────────────────────────────────────────────────────────

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

        def forward(self, x: torch.Tensor,
                    A: torch.Tensor) -> torch.Tensor:
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
            self.A_learned = nn.Parameter(torch.zeros(V, V))
            self.blocks = nn.ModuleList([
                GraphTemporalConv(C,      hidden,    dropout=dropout),
                GraphTemporalConv(hidden, hidden,    dropout=dropout),
                GraphTemporalConv(hidden, embed_dim, dropout=dropout),
            ])
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
            B, T, M, V, C = x.shape
            A  = self.A_fixed + F.softmax(self.A_learned, dim=-1)
            xr = x.permute(0, 2, 4, 1, 3).reshape(B * M, C, T, V)
            for blk in self.blocks:
                xr = blk(xr, A)
            emb       = xr.mean(dim=[2, 3])
            emb       = emb.view(B, M, -1)
            valid     = node_mask.any(dim=1)
            emb_masked = emb * valid.unsqueeze(-1).float()
            scores    = self.person_attn(emb_masked).squeeze(-1)
            scores    = scores.masked_fill(~valid, -1e9)
            weights   = F.softmax(scores, dim=-1).unsqueeze(-1)
            pooled    = (emb_masked * weights).sum(1)
            return self.out_norm(pooled)

    # ─────────────────────────────────────────────────────────────────────────
    # Stream 2 — Improved Interaction (2-layer GATv2 + temporal transformer)
    # ─────────────────────────────────────────────────────────────────────────

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
                    mask: torch.Tensor) -> torch.Tensor:
            B, M, _ = nodes.shape
            Q = self.Wq(nodes)
            K = self.Wk(nodes)
            V = self.Wv(nodes)
            E = self.We(edges)
            a = self.attn_v(F.leaky_relu(
                Q.unsqueeze(2) + K.unsqueeze(1) + E,
                negative_slope=0.2))
            if mask is not None:
                a = a.masked_fill(
                    ~mask.unsqueeze(1).unsqueeze(-1), -1e9)
            a   = F.softmax(a, dim=2)
            agg = (a.mean(-1, keepdim=True) * V.unsqueeze(1)).sum(2)
            out = self.norm(self.drop(self.out(agg)) + V)
            return out

    class ImprovedInteraction(nn.Module):
        def __init__(self, node_dim: int, edge_dim: int,
                     embed_dim: int, n_heads_gat: int,
                     n_heads_trf: int, n_layers_trf: int,
                     T: int, M: int, dropout: float):
            super().__init__()
            self.node_proj = nn.Sequential(
                nn.Linear(node_dim, embed_dim),
                nn.LayerNorm(embed_dim), nn.ReLU(),
            )
            self.gat1 = GATv2Layer(embed_dim, edge_dim,
                                   embed_dim, n_heads_gat, dropout)
            self.gat2 = GATv2Layer(embed_dim, edge_dim,
                                   embed_dim, n_heads_gat, dropout)
            self.pos_enc = nn.Parameter(torch.randn(1, T, embed_dim) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads_trf,
                dim_feedforward=embed_dim * 2,
                dropout=dropout, batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                enc_layer, num_layers=n_layers_trf)
            self.tpool    = nn.Linear(embed_dim, 1)
            self.out_norm = nn.LayerNorm(embed_dim)

        def forward(self, nodes: torch.Tensor,
                    edges: torch.Tensor,
                    node_mask: torch.Tensor) -> torch.Tensor:
            B, T, M, _ = nodes.shape
            frame_embs = []
            for t in range(T):
                n_t  = self.node_proj(nodes[:, t])
                e_t  = edges[:, t]
                mk_t = node_mask[:, t]
                h    = self.gat1(n_t, e_t, mk_t)
                h    = self.gat2(h,   e_t, mk_t)
                valid = mk_t.unsqueeze(-1).float()
                fe    = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
                frame_embs.append(fe)
            seq    = torch.stack(frame_embs, dim=1) + self.pos_enc
            out    = self.transformer(seq)
            w      = F.softmax(self.tpool(out), dim=1)
            pooled = (out * w).sum(1)
            return self.out_norm(pooled)

    # ─────────────────────────────────────────────────────────────────────────
    # Stream 3 — Object/PO (V8-compatible FrameGINE + BiGRU)
    # ─────────────────────────────────────────────────────────────────────────

    class FrameGINE(nn.Module):
        def __init__(self, node_dim: int, edge_dim: int,
                     hidden: int, layers: int = 2, dropout: float = 0.35):
            super().__init__()
            self.node_in  = nn.Linear(node_dim, hidden)
            self.edge_enc = nn.Linear(edge_dim, hidden)
            self.mlps     = nn.ModuleList([
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
            x = self.node_in(nodes)
            for mlp in self.mlps:
                e_e = self.edge_enc(edges)
                if bipartite:
                    sm  = (src_mask.unsqueeze(-1).unsqueeze(-1).float()
                           if src_mask is not None
                           else torch.ones(
                               x.shape[0], e_e.shape[1], 1, 1,
                               device=nodes.device))
                    agg = (e_e * sm).sum(1)
                else:
                    nm  = (node_mask.unsqueeze(-1).unsqueeze(-1).float()
                           if node_mask is not None
                           else torch.ones(
                               x.shape[0], e_e.shape[1], 1, 1,
                               device=nodes.device))
                    agg = (e_e * nm).sum(2)
                x = mlp(x + agg)
            nm2 = (node_mask.unsqueeze(-1).float()
                   if node_mask is not None
                   else torch.ones(x.shape[0], x.shape[1], 1,
                                   device=nodes.device))
            return (self.out(x) * nm2).sum(1) / nm2.sum(1).clamp(min=1)

    class TemporalBiGRU(nn.Module):
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
            x, _ = self.gru(self.proj(seq))
            w     = F.softmax(self.attn(x), dim=1)
            return self.norm((x * w).sum(1))

    class ObjectPOStream(nn.Module):
        def __init__(self, obj_node_dim: int, int_edge_dim: int,
                     po_edge_dim: int, embed_dim: int, dropout: float):
            super().__init__()
            half = embed_dim // 2
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
            B, T, N, _ = obj_n.shape
            obj_frames = []
            po_frames  = []
            for t in range(T):
                on_t  = obj_n[:, t]
                onm_t = obj_nm[:, t]
                pe_t  = po_e[:, t]
                inm_t = int_nm[:, t]
                dummy_edge = torch.zeros(
                    B, N, N, self.obj_edge_dim, device=obj_n.device)
                obj_frames.append(
                    self.obj_gine(on_t, dummy_edge, onm_t))
                po_frames.append(
                    self.po_gine(on_t, pe_t, onm_t,
                                  src_mask=inm_t, bipartite=True))
            e_obj = self.obj_gru(torch.stack(obj_frames, dim=1))
            e_po  = self.po_gru(torch.stack(po_frames,  dim=1))
            return self.proj(torch.cat([e_obj, e_po], dim=-1))

    # ─────────────────────────────────────────────────────────────────────────
    # Stream 4 — VideoMAE projection MLP
    # ─────────────────────────────────────────────────────────────────────────

    class VitProjection(nn.Module):
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
            return self.net(x)

    # ─────────────────────────────────────────────────────────────────────────
    # Fusion — QualityGatedFusion
    # Extended: when return_gates=True the forward returns (fused, gates [B,n])
    # ─────────────────────────────────────────────────────────────────────────

    class QualityGatedFusion(nn.Module):
        def __init__(self, n_streams: int = 4,
                     stream_dropout: float = 0.25):
            super().__init__()
            self.n = n_streams
            self.p = stream_dropout
            self.gate = nn.Sequential(
                nn.Linear(n_streams, 32), nn.ReLU(),
                nn.Linear(32, n_streams),
            )
            # Diagnostic flag: set to True before an eval forward pass to
            # also return the raw per-sample softmax gate weights [B, n].
            self.return_gates: bool = False

        def forward(self, streams: List[torch.Tensor],
                    gqs: torch.Tensor):
            stk   = torch.stack(streams, dim=1)              # [B, n, D]
            gates = F.softmax(
                self.gate(gqs[:, :self.n]), dim=-1)          # [B, n]
            if self.training and self.p > 0:
                B    = stk.shape[0]
                mask = torch.ones(B, self.n, 1, device=stk.device)
                drop = (torch.rand(B, self.n, device=stk.device) < self.p)
                all_dead = drop.all(dim=1, keepdim=True)
                drop = drop & ~all_dead
                mask[drop] = 0.0
                gates = gates * mask.squeeze(-1)
                gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
            fused = (stk * gates.unsqueeze(-1)).sum(1)       # [B, D]
            if self.return_gates:
                return fused, gates
            return fused

    class LearnedWeightFusion(nn.Module):
        def __init__(self, n_streams: int = 4,
                     stream_dropout: float = 0.25):
            super().__init__()
            self.n = n_streams
            self.p = stream_dropout
            self.weights = nn.Parameter(torch.zeros(n_streams))

        def forward(self, streams: List[torch.Tensor],
                    gqs: torch.Tensor) -> torch.Tensor:
            stk   = torch.stack(streams, dim=1)
            gates = F.softmax(self.weights, dim=0)
            gates = gates.unsqueeze(0).expand(stk.shape[0], -1)
            if self.training and self.p > 0:
                B    = stk.shape[0]
                mask = torch.ones(B, self.n, 1, device=stk.device)
                drop = (torch.rand(B, self.n, device=stk.device) < self.p)
                all_dead = drop.all(dim=1, keepdim=True)
                drop = drop & ~all_dead
                mask[drop] = 0.0
                gates = gates * mask.squeeze(-1)
                gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
            return (stk * gates.unsqueeze(-1)).sum(1)

    # ─────────────────────────────────────────────────────────────────────────
    # Full model  (verbatim copy, forward unchanged)
    # ─────────────────────────────────────────────────────────────────────────

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
                     n_layers_trf: int):
            super().__init__()
            self.active = active_streams
            self.fmode  = fusion_mode
            n_active    = len(active_streams)

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

            if n_active > 1:
                if fusion_mode == "qgf":
                    self.fusion = QualityGatedFusion(n_active, stream_dropout)
                else:
                    self.fusion = LearnedWeightFusion(n_active, stream_dropout)

            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 1),
            )

        def forward(self, batch: Dict,
                    return_gates: bool = False):
            streams_out = []

            if "skeleton" in self.active:
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

            gate_weights_out = None
            if len(streams_out) == 1:
                fused = streams_out[0]
            else:
                if return_gates and hasattr(self.fusion, "return_gates"):
                    # Temporarily enable gate return on QualityGatedFusion
                    self.fusion.return_gates = True
                    result = self.fusion(streams_out, batch["gqs"])
                    self.fusion.return_gates = False
                    if isinstance(result, tuple):
                        fused, gate_weights_out = result
                    else:
                        fused = result
                else:
                    fused = self.fusion(streams_out, batch["gqs"])

            logits = self.head(fused).squeeze(-1)   # [B]
            if return_gates:
                return logits, gate_weights_out
            return logits

    return {"V9Model": V9Model}


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_split(split_name: str, split_df):
    """Return (npz_paths, labels) for the requested split, falling back to
    'val' if 'test' is absent (RWF-2000 has no separate test split)."""
    sub = split_df[split_df["split"] == split_name]
    if len(sub) == 0 and split_name == "test":
        print(f"  No '{split_name}' split found; falling back to 'val'.")
        sub = split_df[split_df["split"] == "val"]
    paths  = [CACHE_DIR / f"{v}.npz" for v in sub["video_id"]]
    labels = sub["label"].tolist()
    existing = [(p, l) for p, l in zip(paths, labels) if p.exists()]
    if not existing:
        raise RuntimeError(f"No existing NPZ files found for split '{split_name}'.")
    p_out, l_out = zip(*existing)
    return list(p_out), list(l_out)


def _build_model_for_exp(exp_id: str, V9Model, device):
    """Instantiate V9Model with the fixed stream/fusion config for exp_id."""
    EXP_CFG = {
        "E_full_qgf": (("skeleton", "interaction", "object", "vit"), "qgf"),
        "E_full_lw":  (("skeleton", "interaction", "object", "vit"), "lw"),
        "C_videomae_only": (("vit",), "qgf"),
    }
    if exp_id not in EXP_CFG:
        raise ValueError(f"Unknown experiment id: {exp_id}")
    active_streams, fusion_mode = EXP_CFG[exp_id]
    model = V9Model(
        active_streams = active_streams,
        fusion_mode    = fusion_mode,
        C=cfg.C, T=cfg.T, V=cfg.V, M=cfg.M, N=cfg.N,
        int_nd=7, int_ed=4,
        obj_nd=6, po_ed=5,
        vit_dim=cfg.vit_embed_dim,
        hidden=cfg.graph_hidden, embed_dim=cfg.embed_dim,
        dropout=cfg.dropout, stream_dropout=cfg.stream_dropout,
        n_heads_gat=cfg.n_heads_gat,
        n_heads_trf=cfg.n_heads_trf,
        n_layers_trf=cfg.n_layers_trf,
    ).to(device)
    return model


def _load_checkpoint(exp_id: str, model, device):
    """Load best_model.pt for exp_id; returns checkpoint dict."""
    ckpt_path = RUN_DIR / exp_id / "best_model.pt"
    if not ckpt_path.exists():
        # Fallback: some runs save as best.pt
        ckpt_path = RUN_DIR / exp_id / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {RUN_DIR / exp_id}/best_model.pt "
            f"or best.pt")
    ck = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ck["state_dict"])
    print(f"  Loaded {exp_id} checkpoint: "
          f"ep={ck.get('epoch','?')}  "
          f"val_f1={ck.get('val_macro_f1', float('nan')):.4f}  "
          f"thr={ck.get('threshold', float('nan')):.4f}")
    return ck


def _make_loader(ds, batch_size: int = 64):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=0, pin_memory=True)


def _compute_ece(y_true, y_prob, n_bins: int = 15) -> float:
    """Expected Calibration Error over n_bins equal-width bins."""
    import numpy as np
    bins    = np.linspace(0.0, 1.0, n_bins + 1)
    ece     = 0.0
    n_total = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n_total * abs(acc - conf)
    return float(ece)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 1: gate_weights
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def gate_weights() -> None:
    """
    Diagnose whether QGF gates collapse to VideoMAE dominance.
    Runs on E_full_qgf (per-sample gates) and E_full_lw (static weights).
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    vol_proc.reload()
    seed_everything(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(str(SPLIT_CSV))
    test_paths, test_labels = _load_split("test", split_df)

    RWFDataset = build_dataset_class()
    components = build_model_components()
    V9Model    = components["V9Model"]

    test_ds     = RWFDataset(test_paths, test_labels)
    test_loader = _make_loader(test_ds)

    # ── E_full_qgf: collect per-sample gate weights ───────────────────────────
    print("\n--- E_full_qgf gate weight analysis ---")
    model_qgf = _build_model_for_exp("E_full_qgf", V9Model, DEVICE)
    ck_qgf    = _load_checkpoint("E_full_qgf", model_qgf, DEVICE)
    model_qgf.eval()

    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="E_full_qgf inference", unit="batch"):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits, gates = model_qgf(batch, return_gates=True)
            probs  = torch.sigmoid(logits).cpu().numpy()
            labels = batch["label"].cpu().numpy().astype(int)
            # gates: [B, 4] — order matches active_streams tuple:
            # (skeleton, interaction, object, vit)
            g = gates.cpu().numpy()   # [B, 4]
            vids = batch["video_id"] if isinstance(batch["video_id"], list) \
                   else [batch["video_id"]]
            for i in range(len(probs)):
                rows.append({
                    "video_id":   vids[i],
                    "w_skel":     float(g[i, 0]),
                    "w_int":      float(g[i, 1]),
                    "w_obj":      float(g[i, 2]),
                    "w_vit":      float(g[i, 3]),
                    "true_label": int(labels[i]),
                    "pred_prob":  float(probs[i]),
                })

    gw_df = pd.DataFrame(rows)

    # Summary statistics
    stream_cols = ["w_skel", "w_int", "w_obj", "w_vit"]
    print("\n=== E_full_qgf — Gate weight summary (test set) ===")
    print(f"  N samples : {len(gw_df)}")
    print(f"\n  {'Stream':<8}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*30}")
    for col in stream_cols:
        print(f"  {col:<8}  {gw_df[col].mean():>8.4f}  {gw_df[col].std():>8.4f}")

    pct_dom  = (gw_df["w_vit"] > 0.5).mean() * 100
    pct_hdm  = (gw_df["w_vit"] > 0.7).mean() * 100
    print(f"\n  % samples w_vit > 0.5 (VideoMAE dominant)        : {pct_dom:.1f}%")
    print(f"  % samples w_vit > 0.7 (VideoMAE heavily dominant): {pct_hdm:.1f}%")

    print("\n  Mean gate weights stratified by true_label:")
    strat = gw_df.groupby("true_label")[stream_cols].mean()
    strat.index = strat.index.map({0: "non-violence", 1: "violence"})
    print(strat.round(4).to_string())

    out_path = DIAG_DIR / "gate_weights_E_full_qgf.csv"
    gw_df.to_csv(str(out_path), index=False)
    print(f"\n  Saved: {out_path}")

    # ── E_full_lw: static learned weights ────────────────────────────────────
    print("\n--- E_full_lw static weight analysis ---")
    model_lw = _build_model_for_exp("E_full_lw", V9Model, DEVICE)
    _load_checkpoint("E_full_lw", model_lw, DEVICE)
    model_lw.eval()

    with torch.no_grad():
        static_w = F.softmax(model_lw.fusion.weights, dim=0).cpu().numpy()

    print("\n=== E_full_lw — Learned static weights (softmax) ===")
    labels_str = ["w_skel", "w_int", "w_obj", "w_vit"]
    for name, val in zip(labels_str, static_w):
        print(f"  {name:<8}: {val:.4f}")

    vol_proc.commit()
    print("\ngate_weights complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 2: gqs_stratified
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def gqs_stratified() -> None:
    """
    Determine whether tri-graph outperforms VideoMAE-only on low-quality
    (low valid_ratio) test samples.
    """
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import f1_score, roc_auc_score
    from tqdm import tqdm

    vol_proc.reload()
    seed_everything(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(str(SPLIT_CSV))
    gqs_path = PROC_MOUNT / "gqs_summary_v9.csv"
    if not gqs_path.exists():
        raise FileNotFoundError(f"GQS summary not found: {gqs_path}")
    gqs_df = pd.read_csv(str(gqs_path))

    # Determine which split label maps to "test" in gqs_df
    test_gqs = gqs_df[gqs_df["split"] == "test"].copy()
    if len(test_gqs) == 0:
        print("  No 'test' split in gqs_summary; using 'val'.")
        test_gqs = gqs_df[gqs_df["split"] == "val"].copy()
    if len(test_gqs) == 0:
        raise RuntimeError("No test/val rows in gqs_summary_v9.csv")

    # Build valid_ratio bins
    try:
        test_gqs["q_bucket"], bin_edges = pd.qcut(
            test_gqs["valid_ratio"], q=4, duplicates="drop", retbins=True)
        n_bins = test_gqs["q_bucket"].nunique()
        print(f"  qcut bin edges (q=4, duplicates=drop): "
              f"{[round(e, 4) for e in bin_edges.tolist()]}")
        print(f"  Unique bins obtained: {n_bins}")
    except Exception as e:
        print(f"  WARN: qcut failed ({e}); falling back to median split.")
        n_bins = 0

    if n_bins < 3:
        print("  Fewer than 3 unique bins; falling back to median split.")
        med = test_gqs["valid_ratio"].median()
        test_gqs["q_bucket"] = test_gqs["valid_ratio"].apply(
            lambda v: "low" if v <= med else "high")
        print(f"  Median valid_ratio = {med:.4f}  "
              f"(low: ≤{med:.4f}, high: >{med:.4f})")

    # Load dataset for test split
    test_paths, test_labels = _load_split("test", split_df)
    RWFDataset = build_dataset_class()
    components = build_model_components()
    V9Model    = components["V9Model"]

    test_ds  = RWFDataset(test_paths, test_labels)
    vid_to_idx = {test_ds.samples[i]["video_id"]: i
                  for i in range(len(test_ds))}

    @torch.no_grad()
    def run_inference_subset(model, subset_indices):
        loader = DataLoader(
            Subset(test_ds, subset_indices),
            batch_size=64, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        y_true, y_prob = [], []
        for batch in tqdm(loader, desc="  inference", unit="batch", leave=False):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            probs  = torch.sigmoid(model(batch)).cpu().numpy()
            labels = batch["label"].cpu().numpy().astype(int)
            y_prob.extend(probs.tolist())
            y_true.extend(labels.tolist())
        return np.array(y_true), np.array(y_prob)

    def bucket_metrics(y_true, y_prob, thr):
        preds = (y_prob >= thr).astype(int)
        f1 = float(f1_score(y_true, preds,
                             average="macro", zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        return f1, auc

    # C_videomae_only has no checkpoint in runs_v9/ (it is a Phase-2 fine-tuned
    # VideoMAE model).  Pull its overall test F1 from experiment_comparison_v9.csv
    # and use it as a flat baseline; per-bucket inference is not possible.
    cmp_csv = RUN_DIR / "experiment_comparison_v9.csv"
    vit_overall_f1  = float("nan")
    vit_overall_auc = float("nan")
    if cmp_csv.exists():
        cmp_all = pd.read_csv(str(cmp_csv))
        vit_row = cmp_all[cmp_all["experiment"] == "C_videomae_only"]
        if len(vit_row):
            vit_overall_f1  = float(vit_row.iloc[0].get("test_macro_f1",  float("nan")))
            vit_overall_auc = float(vit_row.iloc[0].get("test_roc_auc",   float("nan")))
            print(f"  C_videomae_only overall test F1={vit_overall_f1:.4f}  "
                  f"AUC={vit_overall_auc:.4f}  (flat baseline, no per-bucket checkpoint)")
        else:
            print("  WARN: C_videomae_only row not found in experiment_comparison_v9.csv; "
                  "delta_f1 column will be NaN.")
    else:
        print("  WARN: experiment_comparison_v9.csv not found; "
              "delta_f1 column will be NaN.")

    # Load E_full_qgf once and run per-bucket inference
    print("\nLoading E_full_qgf ...")
    model_qgf = _build_model_for_exp("E_full_qgf", V9Model, DEVICE)
    ck_qgf    = _load_checkpoint("E_full_qgf", model_qgf, DEVICE)
    model_qgf.eval()
    thr_qgf   = float(ck_qgf.get("threshold", 0.5))

    table_rows = []
    for bucket_label, grp in test_gqs.groupby("q_bucket", observed=True):
        vids    = grp["video_id"].tolist()
        idxs    = [vid_to_idx[v] for v in vids if v in vid_to_idx]
        n_found = len(idxs)
        if n_found == 0:
            print(f"  WARN: bucket '{bucket_label}' has no matching NPZ files; skipping.")
            continue

        print(f"\nBucket '{bucket_label}'  n={n_found}")

        yt_q, yp_q      = run_inference_subset(model_qgf, idxs)
        f1_qgf, auc_qgf = bucket_metrics(yt_q, yp_q, thr_qgf)

        table_rows.append({
            "bucket":                str(bucket_label),
            "n":                     n_found,
            "E_full_qgf_f1":         round(f1_qgf,  4),
            "E_full_qgf_auc":        round(auc_qgf, 4),
            # Flat overall baseline — same value repeated per bucket so
            # delta_f1 shows how QGF fares on each quality stratum vs the
            # VideoMAE overall number (conservative comparison).
            "C_videomae_overall_f1": round(vit_overall_f1,  4),
            "delta_f1":              round(f1_qgf - vit_overall_f1, 4),
        })

    cmp_df = pd.DataFrame(table_rows)
    print("\n=== GQS Stratified Comparison (E_full_qgf per bucket vs C_videomae overall) ===")
    print(cmp_df[["bucket", "n",
                  "E_full_qgf_f1", "C_videomae_overall_f1", "delta_f1"]
                 ].to_string(index=False))

    out_path = DIAG_DIR / "gqs_stratified_comparison.csv"
    cmp_df.to_csv(str(out_path), index=False)
    print(f"\n  Saved: {out_path}")


    vol_proc.commit()
    print("gqs_stratified complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 3: calibration
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def calibration() -> None:
    """
    Temperature scaling on E_full_lw.
    Fits T on val-set NLL; evaluates pre/post on test set.
    Reports ECE (15 bins) before and after.
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    from tqdm import tqdm

    vol_proc.reload()
    seed_everything(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(str(SPLIT_CSV))

    RWFDataset = build_dataset_class()
    components = build_model_components()
    V9Model    = components["V9Model"]

    # Load model
    print("\nLoading E_full_lw ...")
    model = _build_model_for_exp("E_full_lw", V9Model, DEVICE)
    ck    = _load_checkpoint("E_full_lw", model, DEVICE)
    model.eval()

    # ── Collect raw logits on validation set ─────────────────────────────────
    val_paths, val_labels = _load_split("val", split_df)
    val_ds     = RWFDataset(val_paths, val_labels)
    val_loader = _make_loader(val_ds)

    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val logits", unit="batch"):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits = model(batch)
            val_logits_list.extend(logits.cpu().numpy().tolist())
            val_labels_list.extend(batch["label"].cpu().numpy().astype(int).tolist())

    val_logits = np.array(val_logits_list, dtype=np.float64)
    val_labels = np.array(val_labels_list, dtype=np.int32)

    # ── Fit temperature T via NLL minimisation on val set ────────────────────
    def nll(T):
        scaled = val_logits / T
        # Numerically stable sigmoid NLL
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return -np.mean(val_labels * np.log(probs) +
                        (1 - val_labels) * np.log(1 - probs))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_opt  = float(result.x)
    print(f"\n  Optimal temperature T = {T_opt:.4f}  "
          f"(val NLL before: {nll(1.0):.4f}, after: {nll(T_opt):.4f})")

    # ── Collect raw logits on test set ────────────────────────────────────────
    test_paths, test_labels_list = _load_split("test", split_df)
    test_ds     = RWFDataset(test_paths, test_labels_list)
    test_loader = _make_loader(test_ds)

    test_logits_list, test_labels_arr = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test logits", unit="batch"):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits = model(batch)
            test_logits_list.extend(logits.cpu().numpy().tolist())
            test_labels_arr.extend(batch["label"].cpu().numpy().astype(int).tolist())

    test_logits = np.array(test_logits_list, dtype=np.float64)
    test_labels = np.array(test_labels_arr, dtype=np.int32)

    # Pre-calibration probabilities (standard sigmoid)
    pre_probs = 1.0 / (1.0 + np.exp(-test_logits))
    # Post-calibration probabilities (temperature-scaled)
    post_probs = 1.0 / (1.0 + np.exp(-test_logits / T_opt))

    # ── Metrics ───────────────────────────────────────────────────────────────
    thr_pre  = 0.12
    thr_post = 0.50

    def report(y_true, y_prob, thr, tag):
        preds = (y_prob >= thr).astype(int)
        f1  = float(f1_score(y_true, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(y_true, preds))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        ece = _compute_ece(y_true, y_prob, n_bins=15)
        print(f"  {tag:<30}  F1={f1:.4f}  AUC={auc:.4f}  "
              f"Acc={acc:.4f}  ECE={ece:.4f}  thr={thr}")
        return {"macro_f1": f1, "roc_auc": auc, "accuracy": acc,
                "ece": ece, "threshold": thr}

    print("\n=== Calibration results (test set) ===")
    pre_m  = report(test_labels, pre_probs,  thr_pre,  "Pre-calibration  (thr=0.12)")
    post_m = report(test_labels, post_probs, thr_post, "Post-calibration (thr=0.50)")

    out = {
        "experiment":        "E_full_lw",
        "optimal_temperature": T_opt,
        "val_nll_before":    float(nll(1.0)),
        "val_nll_after":     float(nll(T_opt)),
        "pre_calibration":   pre_m,
        "post_calibration":  post_m,
    }

    out_path = DIAG_DIR / "calibration_results.json"
    with open(str(out_path), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")

    vol_proc.commit()
    print("calibration complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 4: error_analysis
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def error_analysis() -> None:
    """
    Profile FPs and FNs for E_full_lw at threshold = 0.12.
    Identifies systematic failure modes via GQS component scores.
    """
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    vol_proc.reload()
    seed_everything(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(str(SPLIT_CSV))

    gqs_path = PROC_MOUNT / "gqs_summary_v9.csv"
    if not gqs_path.exists():
        raise FileNotFoundError(f"GQS summary not found: {gqs_path}")
    gqs_df = pd.read_csv(str(gqs_path))
    # Build lookup: video_id → GQS row
    gqs_lookup = gqs_df.set_index("video_id")

    RWFDataset = build_dataset_class()
    components = build_model_components()
    V9Model    = components["V9Model"]

    # Load model
    print("\nLoading E_full_lw ...")
    model = _build_model_for_exp("E_full_lw", V9Model, DEVICE)
    _load_checkpoint("E_full_lw", model, DEVICE)
    model.eval()

    # Load test split
    test_paths, test_labels = _load_split("test", split_df)
    test_ds     = RWFDataset(test_paths, test_labels)
    test_loader = _make_loader(test_ds)

    THR = 0.12

    # ── Collect per-sample predictions ───────────────────────────────────────
    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="E_full_lw inference", unit="batch"):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            probs  = torch.sigmoid(model(batch)).cpu().numpy()
            labels = batch["label"].cpu().numpy().astype(int)
            vids   = batch["video_id"] if isinstance(batch["video_id"], list) \
                     else [batch["video_id"]]
            for i in range(len(probs)):
                vid = vids[i]
                p   = float(probs[i])
                lbl = int(labels[i])
                pred = int(p >= THR)

                # Pull GQS scores for this video
                if vid in gqs_lookup.index:
                    g = gqs_lookup.loc[vid]
                    valid_ratio = float(g.get("valid_ratio", float("nan")))
                    q_skel      = float(g.get("q_skel",      float("nan")))
                    q_int       = float(g.get("q_int",       float("nan")))
                    q_obj       = float(g.get("q_obj",       float("nan")))
                    q_po        = float(g.get("q_po",        float("nan")))
                else:
                    valid_ratio = q_skel = q_int = q_obj = q_po = float("nan")

                # Segment label
                if lbl == 1 and pred == 1:
                    segment = "TP"
                elif lbl == 0 and pred == 0:
                    segment = "TN"
                elif lbl == 0 and pred == 1:
                    segment = "FP"
                else:
                    segment = "FN"

                rows.append({
                    "video_id":    vid,
                    "true_label":  lbl,
                    "pred_prob":   round(p, 6),
                    "pred_label":  pred,
                    "segment":     segment,
                    "valid_ratio": valid_ratio,
                    "q_skel":      q_skel,
                    "q_int":       q_int,
                    "q_obj":       q_obj,
                    "q_po":        q_po,
                })

    df = pd.DataFrame(rows)

    # ── Per-segment summary ───────────────────────────────────────────────────
    gqs_cols = ["pred_prob", "valid_ratio", "q_skel", "q_int", "q_obj", "q_po"]
    print(f"\n=== Error analysis — E_full_lw at threshold={THR} ===")
    print(f"\n  Total test samples: {len(df)}")

    for seg in ["TP", "TN", "FP", "FN"]:
        sub = df[df["segment"] == seg]
        n   = len(sub)
        if n == 0:
            print(f"\n  {seg}: 0 samples")
            continue
        means = sub[gqs_cols].mean()
        print(f"\n  {seg} (n={n}):")
        for col in gqs_cols:
            print(f"    {col:<14}: {means[col]:.4f}")

    # ── Top-10 highest-confidence FPs ─────────────────────────────────────────
    fps = df[df["segment"] == "FP"].sort_values("pred_prob", ascending=False)
    print(f"\n=== Top-10 highest-confidence FPs ===")
    print(fps[["pred_prob", "video_id", "valid_ratio"]
              ].head(10).to_string(index=False))

    # ── Top-10 lowest-confidence FNs (violence missed, low pred_prob) ─────────
    fns = df[df["segment"] == "FN"].sort_values("pred_prob", ascending=True)
    print(f"\n=== Top-10 lowest-confidence FNs ===")
    print(fns[["pred_prob", "video_id", "valid_ratio"]
              ].head(10).to_string(index=False))

    # Save full table
    out_path = DIAG_DIR / "error_analysis.csv"
    df.to_csv(str(out_path), index=False)
    print(f"\n  Saved: {out_path}")

    vol_proc.commit()
    print("error_analysis complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 5: run_all  — run all four diagnostics in one container invocation
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def run_all() -> None:
    """
    Run gate_weights, gqs_stratified, calibration, and error_analysis
    sequentially inside a single Modal container.  Outputs accumulate in
    /data/proc/runs_v9/diagnostics/ on the volume.
    """
    print("=" * 70)
    print("STEP 1/4 — gate_weights")
    print("=" * 70)
    gate_weights.local()

    print("=" * 70)
    print("STEP 2/4 — gqs_stratified")
    print("=" * 70)
    gqs_stratified.local()

    print("=" * 70)
    print("STEP 3/4 — calibration")
    print("=" * 70)
    calibration.local()

    print("=" * 70)
    print("STEP 4/4 — error_analysis")
    print("=" * 70)
    error_analysis.local()

    print("=" * 70)
    print("All diagnostics complete. Results in /data/proc/runs_v9/diagnostics/")
    print("=" * 70)


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    """
    Single entry point: runs all four diagnostics on Modal (T4 GPU) then
    downloads every output file from the volume into ./diagnostics_output/
    on this machine.

    Usage:
        conda run -n dlcuda128 modal run trigraph_v9_rwf_diagnostics.py
    """
    import sys
    from pathlib import Path as LocalPath

    LOCAL_OUT = LocalPath("diagnostics_output")
    LOCAL_OUT.mkdir(exist_ok=True)

    # Output files that run_all will produce inside the Modal volume
    EXPECTED = [
        "runs_v9/diagnostics/gate_weights_E_full_qgf.csv",
        "runs_v9/diagnostics/gqs_stratified_comparison.csv",
        "runs_v9/diagnostics/calibration_results.json",
        "runs_v9/diagnostics/error_analysis.csv",
    ]

    print("Guardian Eye V9 — running all diagnostics on Modal (T4)...")
    print(f"Results will be saved locally to: {LOCAL_OUT.resolve()}")
    print()

    # Dispatch to cloud and stream logs back
    run_all.remote()

    # Pull output files from the Modal volume to local disk
    print()
    print("Downloading results from Modal volume...")
    vol_proc.reload()
    downloaded = 0
    failed     = []
    for rel_path in EXPECTED:
        try:
            data      = b"".join(vol_proc.read_file(rel_path))
            dest      = LOCAL_OUT / LocalPath(rel_path).name
            dest.write_bytes(data)
            print(f"  [ok] {dest}")
            downloaded += 1
        except Exception as e:
            print(f"  [WARN] could not download {rel_path}: {e}")
            failed.append(rel_path)

    print()
    print(f"Downloaded {downloaded}/{len(EXPECTED)} files to {LOCAL_OUT.resolve()}")
    if failed:
        print("Missing files (check Modal logs above for errors):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
