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
    thr_min:  float = 0.05
    thr_max:  float = 0.90
    thr_step: float = 0.01

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
            # Diagnostic flag: set to True before an eval forward pass to
            # also return the raw per-sample softmax gate weights [B, n].
            self.return_gates: bool = False

        def forward(self, streams: List[torch.Tensor],
                    gqs: torch.Tensor):
            stk        = torch.stack(streams, dim=1)             # [B, n, D]
            gate_logits = self.gate(gqs[:, :self.n])             # [B, n]
            gates       = F.softmax(
                gate_logits / self.current_temp, dim=-1)         # [B, n]
            if self.training and self.p > 0:
                B    = stk.shape[0]
                mask = torch.ones(B, self.n, 1, device=stk.device)
                drop = (torch.rand(B, self.n, device=stk.device) < self.p)
                all_dead = drop.all(dim=1, keepdim=True)
                drop = drop & ~all_dead
                mask[drop] = 0.0
                gates = gates * mask.squeeze(-1)
                gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
            if self.training:
                self.last_entropy = (
                    -(gates * (gates + 1e-8).log()).sum(-1).mean()
                )
            else:
                self.last_entropy = None
            fused = (stk * gates.unsqueeze(-1)).sum(1)           # [B, D]
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
                     n_layers_trf: int,
                     gate_temperature: float = 1.0,
                     entropy_weight: float = 0.0,
                     temp_anneal: bool = False):
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
                    self.fusion = QualityGatedFusion(
                        n_active, stream_dropout,
                        gate_temperature=gate_temperature,
                        entropy_weight=entropy_weight,
                        temp_anneal=temp_anneal,
                    )
                else:
                    self.fusion = LearnedWeightFusion(n_active, stream_dropout)

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
    # (streams, fusion_mode, extra_kwargs_for_qgf)
    EXP_CFG = {
        "E_full_qgf":       (("skeleton", "interaction", "object", "vit"), "qgf", {}),
        "E_full_lw":        (("skeleton", "interaction", "object", "vit"), "lw",  {}),
        "C_videomae_only":  (("vit",),                                     "qgf", {}),
        # E_full_qgf_fixed was trained with entropy_weight=0.10, temp_anneal=True;
        # at inference time only the architecture matters — temperature is 1.0
        # (buffer is non-persistent so the checkpoint stores no value for it).
        "E_full_qgf_fixed": (("skeleton", "interaction", "object", "vit"), "qgf",
                              {"entropy_weight": 0.10, "temp_anneal": True}),
    }
    if exp_id not in EXP_CFG:
        raise ValueError(f"Unknown experiment id: {exp_id}")
    active_streams, fusion_mode, extra = EXP_CFG[exp_id]
    model = V9Model(
        active_streams  = active_streams,
        fusion_mode     = fusion_mode,
        C=cfg.C, T=cfg.T, V=cfg.V, M=cfg.M, N=cfg.N,
        int_nd=7, int_ed=4,
        obj_nd=6, po_ed=5,
        vit_dim=cfg.vit_embed_dim,
        hidden=cfg.graph_hidden, embed_dim=cfg.embed_dim,
        dropout=cfg.dropout, stream_dropout=cfg.stream_dropout,
        n_heads_gat=cfg.n_heads_gat,
        n_heads_trf=cfg.n_heads_trf,
        n_layers_trf=cfg.n_layers_trf,
        **extra,
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
    test samples. Uses a composite GQS score instead of valid_ratio because
    valid_ratio saturates at 1.0 across the RWF-2000 dataset.
    Compares E_full_qgf and E_full_lw per bucket against the VideoMAE baseline.
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

    test_gqs = gqs_df[gqs_df["split"] == "test"].copy()
    if len(test_gqs) == 0:
        print("  No 'test' split in gqs_summary; using 'val'.")
        test_gqs = gqs_df[gqs_df["split"] == "val"].copy()
    if len(test_gqs) == 0:
        raise RuntimeError("No test/val rows in gqs_summary_v9.csv")

    # Composite quality score: weighted combination of per-stream quality
    # metrics. valid_ratio saturates at 1.0 on RWF-2000; this composite
    # preserves discriminative variance for bucketing.
    for col in ("q_skel", "q_int", "q_po"):
        if col not in test_gqs.columns:
            raise ValueError(f"gqs_summary_v9.csv missing column {col}")
    test_gqs["gqs_composite"] = (
        0.4 * test_gqs["q_skel"] +
        0.4 * test_gqs["q_int"]  +
        0.2 * test_gqs["q_po"]
    )

    # Primary bucketing: quartile cut; fall back to tertile cut if too few
    # unique composite values remain after deduplication.
    try:
        test_gqs["q_bucket"], _ = pd.qcut(
            test_gqs["gqs_composite"], q=4,
            duplicates="drop", retbins=True)
        n_bins = test_gqs["q_bucket"].nunique()
        print(f"  qcut (q=4, duplicates=drop): {n_bins} unique bins")
    except Exception as e:
        print(f"  WARN: qcut failed ({e}); falling back to tertile cut.")
        n_bins = 0

    if n_bins < 2:
        edges = np.quantile(test_gqs["gqs_composite"].values,
                            [0, 0.33, 0.66, 1.0])
        test_gqs["q_bucket"] = pd.cut(
            test_gqs["gqs_composite"], bins=edges,
            labels=["Q1_low", "Q2_mid", "Q3_high"],
            include_lowest=True)
        n_bins = test_gqs["q_bucket"].nunique()
        print(f"  Tertile cut: {n_bins} unique bins")

    if n_bins < 2:
        raise RuntimeError(
            "gqs_composite has no discriminative variance; "
            "cannot produce stratified comparison.")

    # Load dataset
    test_paths, test_labels = _load_split("test", split_df)
    RWFDataset = build_dataset_class()
    components = build_model_components()
    V9Model    = components["V9Model"]

    test_ds    = RWFDataset(test_paths, test_labels)
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

    # Pull C_videomae_only overall F1 as a flat per-bucket baseline.
    # Per-bucket VideoMAE inference is not possible (Phase-2 checkpoint is
    # not stored in runs_v9/).
    cmp_csv = RUN_DIR / "experiment_comparison_v9.csv"
    vit_overall_f1  = float("nan")
    vit_overall_auc = float("nan")
    if cmp_csv.exists():
        cmp_all = pd.read_csv(str(cmp_csv))
        vit_row = cmp_all[cmp_all["experiment"] == "C_videomae_only"]
        if len(vit_row):
            vit_overall_f1  = float(vit_row.iloc[0].get("test_macro_f1",  float("nan")))
            vit_overall_auc = float(vit_row.iloc[0].get("test_roc_auc",   float("nan")))
            print(f"  C_videomae_only overall F1={vit_overall_f1:.4f}  "
                  f"AUC={vit_overall_auc:.4f}  (flat baseline)")
        else:
            print("  WARN: C_videomae_only row not found; "
                  "delta columns will be NaN.")
    else:
        print("  WARN: experiment_comparison_v9.csv not found; "
              "delta columns will be NaN.")

    # Load both models once; iterate buckets for each
    print("\nLoading E_full_qgf ...")
    model_qgf = _build_model_for_exp("E_full_qgf", V9Model, DEVICE)
    ck_qgf    = _load_checkpoint("E_full_qgf", model_qgf, DEVICE)
    model_qgf.eval()
    thr_qgf   = float(ck_qgf.get("threshold", 0.5))

    print("\nLoading E_full_lw ...")
    model_lw = _build_model_for_exp("E_full_lw", V9Model, DEVICE)
    ck_lw    = _load_checkpoint("E_full_lw", model_lw, DEVICE)
    model_lw.eval()
    thr_lw   = float(ck_lw.get("threshold", 0.5))

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

        yt_l, yp_l      = run_inference_subset(model_lw, idxs)
        f1_lw,  auc_lw  = bucket_metrics(yt_l, yp_l, thr_lw)

        table_rows.append({
            "bucket":                 str(bucket_label),
            "n":                      n_found,
            "E_full_qgf_f1":          round(f1_qgf,  4),
            "E_full_qgf_auc":         round(auc_qgf, 4),
            "E_full_lw_f1":           round(f1_lw,   4),
            "E_full_lw_auc":          round(auc_lw,  4),
            # Same overall value repeated per bucket (conservative comparison)
            "C_videomae_overall_f1":  round(vit_overall_f1,  4),
            "delta_qgf_vs_lw":        round(f1_qgf - f1_lw,          4),
            "delta_qgf_vs_videomae":  round(f1_qgf - vit_overall_f1, 4),
        })

    cmp_df = pd.DataFrame(table_rows)
    print("\n=== GQS Stratified Comparison ===")
    print(cmp_df[["bucket", "n",
                  "E_full_qgf_f1", "E_full_qgf_auc",
                  "E_full_lw_f1",  "E_full_lw_auc",
                  "C_videomae_overall_f1",
                  "delta_qgf_vs_lw", "delta_qgf_vs_videomae",
                  ]].to_string(index=False))

    out_path = DIAG_DIR / "gqs_stratified_comparison.csv"
    cmp_df.to_csv(str(out_path), index=False)
    print(f"\n  Saved: {out_path}")

    vol_proc.commit()
    print("gqs_stratified complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 3: calibration
# Target: E_full_qgf (best model). Fits T* on val NLL, re-tunes threshold
# on calibrated val probs, then evaluates pre/post on test set with ECE.
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def calibration() -> None:
    import numpy as np
    import pandas as pd
    import torch
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

    EXP_ID = "E_full_qgf"
    print(f"\nLoading {EXP_ID} ...")
    model = _build_model_for_exp(EXP_ID, V9Model, DEVICE)
    ck    = _load_checkpoint(EXP_ID, model, DEVICE)
    model.eval()
    # Threshold stored in checkpoint by train.py
    thr_raw = float(ck.get("threshold", 0.5))

    def _collect_logits(split_name):
        paths, labels = _load_split(split_name, split_df)
        ds     = RWFDataset(paths, labels)
        loader = _make_loader(ds)
        logits_out, labels_out = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{split_name} logits", unit="batch"):
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                logits_out.extend(model(batch).cpu().numpy().tolist())
                labels_out.extend(batch["label"].cpu().numpy().astype(int).tolist())
        return (np.array(logits_out, dtype=np.float64),
                np.array(labels_out, dtype=np.int32))

    val_logits, val_labels   = _collect_logits("val")
    test_logits, test_labels = _collect_logits("test")

    # ── Fit T* on val NLL ────────────────────────────────────────────────────
    eps = 1e-7
    def nll(T):
        p = np.clip(1.0 / (1.0 + np.exp(-val_logits / T)), eps, 1 - eps)
        return -np.mean(val_labels * np.log(p) + (1 - val_labels) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_opt  = float(result.x)
    print(f"\n  T* = {T_opt:.4f}  "
          f"(val NLL before={nll(1.0):.4f}, after={nll(T_opt):.4f})")

    # ── Re-tune threshold on calibrated val probs ────────────────────────────
    val_probs_cal = np.clip(1.0 / (1.0 + np.exp(-val_logits / T_opt)), eps, 1 - eps)
    best_thr, best_f1 = thr_raw, 0.0
    for thr in np.arange(cfg.thr_min, cfg.thr_max, cfg.thr_step):
        f1 = float(f1_score(val_labels, (val_probs_cal >= thr).astype(int),
                             average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    print(f"  Re-tuned threshold on calibrated val probs: {best_thr:.2f}  "
          f"(val macro-F1={best_f1:.4f})")

    # ── Test-set metrics ─────────────────────────────────────────────────────
    pre_probs  = np.clip(1.0 / (1.0 + np.exp(-test_logits)),          eps, 1 - eps)
    post_probs = np.clip(1.0 / (1.0 + np.exp(-test_logits / T_opt)), eps, 1 - eps)

    def report(y_true, y_prob, thr, tag):
        preds = (y_prob >= thr).astype(int)
        f1  = float(f1_score(y_true, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(y_true, preds))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        ece = _compute_ece(y_true, y_prob, n_bins=15)
        print(f"  {tag:<38}  F1={f1:.4f}  AUC={auc:.4f}  "
              f"Acc={acc:.4f}  ECE={ece:.4f}  thr={thr:.2f}")
        return {"macro_f1": f1, "roc_auc": auc, "accuracy": acc,
                "ece": ece, "threshold": thr}

    print(f"\n=== Calibration results (test set) — {EXP_ID} ===")
    pre_m  = report(test_labels, pre_probs,  thr_raw,  f"Pre-calibration  (thr={thr_raw:.2f})")
    post_m = report(test_labels, post_probs, best_thr, f"Post-calibration (thr={best_thr:.2f})")

    out = {
        "experiment":          EXP_ID,
        "optimal_temperature": T_opt,
        "val_nll_before":      float(nll(1.0)),
        "val_nll_after":       float(nll(T_opt)),
        "cal_threshold":       best_thr,
        "pre_calibration":     pre_m,
        "post_calibration":    post_m,
    }
    out_path = DIAG_DIR / "calibration_results.json"
    with open(str(out_path), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")

    vol_proc.commit()
    print("calibration complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 4: error_analysis
# Target: E_full_qgf. Threshold loaded from checkpoint (not hardcoded).
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def error_analysis() -> None:
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
    gqs_df     = pd.read_csv(str(gqs_path))
    gqs_lookup = gqs_df.set_index("video_id")

    RWFDataset = build_dataset_class()
    components = build_model_components()
    V9Model    = components["V9Model"]

    EXP_ID = "E_full_qgf"
    print(f"\nLoading {EXP_ID} ...")
    model = _build_model_for_exp(EXP_ID, V9Model, DEVICE)
    ck    = _load_checkpoint(EXP_ID, model, DEVICE)
    model.eval()
    THR = float(ck.get("threshold", 0.5))
    print(f"  Using threshold from checkpoint: {THR:.4f}")

    test_paths, test_labels = _load_split("test", split_df)
    test_ds     = RWFDataset(test_paths, test_labels)
    test_loader = _make_loader(test_ds)

    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{EXP_ID} inference", unit="batch"):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            probs  = torch.sigmoid(model(batch)).cpu().numpy()
            labels = batch["label"].cpu().numpy().astype(int)
            vids   = batch["video_id"] if isinstance(batch["video_id"], list) \
                     else [batch["video_id"]]
            for i in range(len(probs)):
                vid  = vids[i]
                p    = float(probs[i])
                lbl  = int(labels[i])
                pred = int(p >= THR)

                if vid in gqs_lookup.index:
                    g = gqs_lookup.loc[vid]
                    valid_ratio = float(g.get("valid_ratio", float("nan")))
                    q_skel      = float(g.get("q_skel",      float("nan")))
                    q_int       = float(g.get("q_int",       float("nan")))
                    q_obj       = float(g.get("q_obj",       float("nan")))
                    q_po        = float(g.get("q_po",        float("nan")))
                else:
                    valid_ratio = q_skel = q_int = q_obj = q_po = float("nan")

                if   lbl == 1 and pred == 1: segment = "TP"
                elif lbl == 0 and pred == 0: segment = "TN"
                elif lbl == 0 and pred == 1: segment = "FP"
                else:                        segment = "FN"

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

    gqs_cols = ["pred_prob", "valid_ratio", "q_skel", "q_int", "q_obj", "q_po"]
    print(f"\n=== Error analysis — {EXP_ID} at threshold={THR:.4f} ===")
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

    fps = df[df["segment"] == "FP"].sort_values("pred_prob", ascending=False)
    print("\n=== Top-10 highest-confidence FPs ===")
    print(fps[["pred_prob", "video_id", "valid_ratio"]].head(10).to_string(index=False))

    fns = df[df["segment"] == "FN"].sort_values("pred_prob", ascending=True)
    print("\n=== Top-10 lowest-confidence FNs ===")
    print(fns[["pred_prob", "video_id", "valid_ratio"]].head(10).to_string(index=False))

    out_path = DIAG_DIR / "error_analysis.csv"
    df.to_csv(str(out_path), index=False)
    print(f"\n  Saved: {out_path}")

    vol_proc.commit()
    print("error_analysis complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 5: inference_speed
# Measures throughput and latency for every experiment on the test set.
# Reports: mean latency per sample (ms), std, and clips/sec on GPU.
# Also counts model parameters for each experiment.
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def inference_speed() -> None:
    import time
    import json as _json
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    vol_proc.reload()
    seed_everything(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    split_df   = pd.read_csv(str(SPLIT_CSV))
    RWFDataset = build_dataset_class()
    components = build_model_components()
    V9Model    = components["V9Model"]

    test_paths, test_labels = _load_split("test", split_df)
    test_ds = RWFDataset(test_paths, test_labels)

    # Use batch_size=1 for per-sample latency; batch_size=64 for throughput
    loader_single = DataLoader(test_ds, batch_size=1,
                               shuffle=False, num_workers=0, pin_memory=True)
    loader_batch  = DataLoader(test_ds, batch_size=64,
                               shuffle=False, num_workers=0, pin_memory=True)

    EXPERIMENTS = ["E_full_qgf", "E_full_lw", "E_full_qgf_fixed"]
    WARMUP_BATCHES = 5   # GPU warmup before timing

    rows = []
    for exp_id in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  {exp_id}")
        print(f"{'='*60}")

        model = _build_model_for_exp(exp_id, V9Model, DEVICE)
        _load_checkpoint(exp_id, model, DEVICE)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # ── Warmup ───────────────────────────────────────────────────────────
        with torch.no_grad():
            for i, batch in enumerate(loader_batch):
                if i >= WARMUP_BATCHES:
                    break
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                _ = model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        # ── Per-sample latency (batch_size=1) ────────────────────────────────
        latencies_ms = []
        with torch.no_grad():
            for batch in tqdm(loader_single, desc="  latency (bs=1)", unit="sample"):
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _  = model(batch)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        lat_arr = np.array(latencies_ms)
        mean_ms = float(lat_arr.mean())
        std_ms  = float(lat_arr.std())
        p50_ms  = float(np.percentile(lat_arr, 50))
        p95_ms  = float(np.percentile(lat_arr, 95))
        p99_ms  = float(np.percentile(lat_arr, 99))
        print(f"  Latency (bs=1)  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms  "
              f"p50={p50_ms:.2f}ms  p95={p95_ms:.2f}ms  p99={p99_ms:.2f}ms")

        # ── Throughput (batch_size=64) ────────────────────────────────────────
        n_samples  = 0
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            for batch in tqdm(loader_batch, desc="  throughput (bs=64)", unit="batch"):
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                _  = model(batch)
                n_samples += batch["label"].shape[0]
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed    = time.perf_counter() - t_start
        clips_per_sec = n_samples / elapsed
        print(f"  Throughput     {clips_per_sec:.1f} clips/sec  "
              f"({n_samples} samples in {elapsed:.2f}s)")

        rows.append({
            "experiment":    exp_id,
            "n_params":      n_params,
            "latency_mean_ms": round(mean_ms, 3),
            "latency_std_ms":  round(std_ms,  3),
            "latency_p50_ms":  round(p50_ms,  3),
            "latency_p95_ms":  round(p95_ms,  3),
            "latency_p99_ms":  round(p99_ms,  3),
            "throughput_clips_per_sec": round(clips_per_sec, 2),
            "device": str(DEVICE),
        })

    speed_df = pd.DataFrame(rows)
    print("\n=== Inference speed summary ===")
    print(speed_df[["experiment", "n_params", "latency_mean_ms",
                    "latency_p95_ms", "throughput_clips_per_sec"]].to_string(index=False))

    out_csv  = DIAG_DIR / "inference_speed.csv"
    out_json = DIAG_DIR / "inference_speed.json"
    speed_df.to_csv(str(out_csv), index=False)
    with open(str(out_json), "w") as f:
        _json.dump(rows, f, indent=2)
    print(f"\n  Saved: {out_csv}")
    print(f"  Saved: {out_json}")

    vol_proc.commit()
    print("inference_speed complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 6: run_all  — run all five diagnostics in one container invocation
# ══════════════════════════════════════════════════════════════════════════════
@app.function(**_FUNC_KWARGS)
def run_all() -> None:
    """
    Run gate_weights, gqs_stratified, calibration, error_analysis, and
    inference_speed sequentially inside a single Modal container.
    Outputs accumulate in /data/proc/runs_v9/diagnostics/ on the volume.
    """
    print("=" * 70)
    print("STEP 1/5 — gate_weights")
    print("=" * 70)
    gate_weights.local()

    print("=" * 70)
    print("STEP 2/5 — gqs_stratified")
    print("=" * 70)
    gqs_stratified.local()

    print("=" * 70)
    print("STEP 3/5 — calibration")
    print("=" * 70)
    calibration.local()

    print("=" * 70)
    print("STEP 4/5 — error_analysis")
    print("=" * 70)
    error_analysis.local()

    print("=" * 70)
    print("STEP 5/5 — inference_speed")
    print("=" * 70)
    inference_speed.local()

    print("=" * 70)
    print("All diagnostics complete. Results in /data/proc/runs_v9/diagnostics/")
    print("=" * 70)


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    """
    Single entry point: runs all five diagnostics on Modal (T4 GPU) then
    downloads every output file from the volume into ./diagnostics_output/
    on this machine.

    Usage:
        modal run trigraph_v9_rwf_diagnostics.py
    """
    import sys
    from pathlib import Path as LocalPath

    LOCAL_OUT = LocalPath("diagnostics_output")
    LOCAL_OUT.mkdir(exist_ok=True)

    EXPECTED = [
        "runs_v9/diagnostics/gate_weights_E_full_qgf.csv",
        "runs_v9/diagnostics/gqs_stratified_comparison.csv",
        "runs_v9/diagnostics/calibration_results.json",
        "runs_v9/diagnostics/error_analysis.csv",
        "runs_v9/diagnostics/inference_speed.csv",
        "runs_v9/diagnostics/inference_speed.json",
    ]

    print("Guardian Eye V9 — running all diagnostics on Modal (T4)...")
    print(f"Results will be saved locally to: {LOCAL_OUT.resolve()}")
    print()

    run_all.remote()

    print()
    print("Downloading results from Modal volume...")
    downloaded = 0
    failed     = []
    for rel_path in EXPECTED:
        try:
            data = b"".join(vol_proc.read_file(rel_path))
            dest = LOCAL_OUT / LocalPath(rel_path).name
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
