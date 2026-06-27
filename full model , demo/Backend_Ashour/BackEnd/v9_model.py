"""
Guardian Eye — V9 Model (inference-only copy)
Extracted from trigraph_v9_rlvs_train.py — training code stripped.

Added: QualityGatedFusion.forward() accepts return_gates=True to expose
       the softmax gate weights [B, 4] needed by the demo UI.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── COCO-17 bone computation (used for multi-input skeleton) ──────────────────

_COCO17_BONE_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


def compute_bones(skel: "np.ndarray") -> "np.ndarray":
    """
    skel: [T, M, V, 3] (x, y, conf) -> bone vectors [T, M, V, 3] (dx, dy, min_conf).
    Joints with no outgoing edge get zeros.
    """
    import numpy as np
    bones = np.zeros_like(skel)
    for p, c in _COCO17_BONE_EDGES:
        bones[:, :, p, 0] = skel[:, :, c, 0] - skel[:, :, p, 0]
        bones[:, :, p, 1] = skel[:, :, c, 1] - skel[:, :, p, 1]
        bones[:, :, p, 2] = np.minimum(skel[:, :, c, 2], skel[:, :, p, 2])
    return bones


# ── Stream 1: Enhanced STGCN ──────────────────────────────────────────────────

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
        emb       = xr.mean(dim=[2, 3]).view(B, M, -1)
        valid     = node_mask.any(dim=1)                       # [B, M]
        emb_m     = emb * valid.unsqueeze(-1).float()
        scores    = self.person_attn(emb_m).squeeze(-1)
        scores    = scores.masked_fill(~valid, -1e9)
        weights   = F.softmax(scores, dim=-1).unsqueeze(-1)
        return self.out_norm((emb_m * weights).sum(1))


# ── Stream 2: Improved Interaction ────────────────────────────────────────────

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


# ── Stream 3: Object / Person-Object ─────────────────────────────────────────

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


# ── Stream 4: VideoMAE Projection ─────────────────────────────────────────────

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


# ── Fusion ────────────────────────────────────────────────────────────────────

class QualityGatedFusion(nn.Module):
    def __init__(self, n_streams: int = 4, stream_dropout: float = 0.25,
                 gate_temperature: float = 1.0, entropy_weight: float = 0.0,
                 temp_anneal: bool = False):
        super().__init__()
        self.n = n_streams
        self.p = stream_dropout
        self.entropy_weight = entropy_weight
        self.temp_anneal    = temp_anneal
        self.gate = nn.Sequential(
            nn.LayerNorm(n_streams),
            nn.Linear(n_streams, 32), nn.ReLU(),
            nn.Linear(32, n_streams))
        self.register_buffer("current_temp",
                             torch.tensor(gate_temperature), persistent=False)

    def forward(self, streams: List[torch.Tensor],
                gqs: torch.Tensor,
                return_gates: bool = False):
        stk         = torch.stack(streams, dim=1)
        gate_logits = self.gate(gqs[:, :self.n])
        gates       = F.softmax(gate_logits / self.current_temp, dim=-1)  # [B, n]
        # stream dropout only during training
        if self.training and self.p > 0:
            B    = stk.shape[0]
            mask = torch.ones(B, self.n, 1, device=stk.device)
            drop = torch.rand(B, self.n, device=stk.device) < self.p
            drop = drop & ~drop.all(dim=1, keepdim=True)
            mask[drop] = 0.0
            gates = gates * mask.squeeze(-1)
            gates = gates / gates.sum(-1, keepdim=True).clamp(min=1e-6)
        fused = (stk * gates.unsqueeze(-1)).sum(1)
        if return_gates:
            return fused, gates
        return fused


class LearnedWeightFusion(nn.Module):
    def __init__(self, n_streams: int = 4, stream_dropout: float = 0.25):
        super().__init__()
        self.n = n_streams
        self.p = stream_dropout
        self.weights = nn.Parameter(torch.zeros(n_streams))

    def forward(self, streams: List[torch.Tensor],
                gqs: torch.Tensor,
                return_gates: bool = False):
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
        fused = (stk * gates.unsqueeze(-1)).sum(1)
        if return_gates:
            return fused, gates
        return fused


# ── Full V9 Model ─────────────────────────────────────────────────────────────

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

    def forward(self, batch: Dict,
                return_gates: bool = False):
        """
        Args:
            batch: dict of tensors (skeleton, int_nodes, …, gqs, vit_embedding)
            return_gates: if True, returns (logits [B], gates [B, n_streams])
        Returns:
            logits [B]  — or  (logits [B], gates [B, n_streams]) when return_gates=True
        """
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

        if len(streams_out) == 1:
            fused = streams_out[0]
            gates = torch.ones(fused.shape[0], 1, device=fused.device)
        else:
            fused, gates = self.fusion(streams_out, batch["gqs"],
                                       return_gates=True)

        logits = self.head(fused).squeeze(-1)
        if return_gates:
            return logits, gates
        return logits
