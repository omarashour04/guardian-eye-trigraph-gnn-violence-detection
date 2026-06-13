# Guardian Eye V9 — Complete Documentation

This README is the consolidated, exhaustive record of **everything about Guardian Eye V9**:
architecture, preprocessing, training, fine-tuning, parameters, results, diagnostics, every
bug fixed, and every trial across all five datasets (RWF-2000, RLVS, NTU CCTV-Fights,
UBI-Fights, Hockey Fights).

Every fact below is sourced from the project vault (`gp vault/gp/wiki/`). The originating
note is cited inline so any claim is traceable. No numbers are invented; where the vault has
a gap it is flagged as such.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture (V9Model) End-to-End](#2-architecture-v9model-end-to-end)
3. [The Four Streams in Detail](#3-the-four-streams-in-detail)
4. [Fusion — QGF and LW](#4-fusion--qgf-quality-gated-and-lw-learned-weight)
5. [Graph Quality Scores (GQS)](#5-graph-quality-scores-gqs)
6. [NPZ Schema and Data Invariants](#6-npz-schema-and-data-invariants)
7. [Three-Phase Pipeline (Staged Sequential Transfer)](#7-three-phase-pipeline-staged-sequential-transfer)
8. [Phase 1 — Preprocessing](#8-phase-1--preprocessing)
9. [Phase 2 — VideoMAE Fine-Tune + Embedding Extraction](#9-phase-2--videomae-fine-tune--embedding-extraction)
10. [Phase 3 — Graph Training + Ablation](#10-phase-3--graph-training--ablation)
11. [Diagnostics Methodology](#11-diagnostics-methodology)
12. [Per-Dataset Results](#12-per-dataset-results)
13. [The UBI-Fights Investigation (full negative-result story)](#13-the-ubi-fights-investigation)
14. [All Bugs Fixed](#14-all-bugs-fixed)
15. [Version History (V4.2 → V8.x → V9)](#15-version-history)
16. [Operational Notes & Costs](#16-operational-notes--costs)
17. [Source Map](#17-source-map-vault-files-this-document-draws-from)
18. [References](#18-references)

> **Citations.** Inline markers `[n]` refer to the numbered reference list in Section 18, which
> uses the **same numbering as the paper** (`guardian_eye_paper.bbl`, IEEE order of first
> citation). A claim with `[n]` is grounded in that external work; claims without a marker are
> grounded in the project vault (cited as `(Source: ...)`).

---

## 1. Project Overview

**Guardian Eye** is a binary violence-detection system for CCTV footage — a final-year CSIT
graduation project (AI & Data Science) targeting a Q1 research paper.
(Source: `entities/guardian-eye.md`)

**Core hypothesis.** Violence is a *relational* signal: it lives in person-person dynamics
(proximity, relative velocity, sustained close engagement) and person-object grounding
(weapon/body contact). Pure appearance models [1], [2] conflate *who is in the scene* with
*what people are doing*, especially on low-res oblique CCTV. Guardian Eye encodes relational
structure as explicit graphs and fuses them with a quality-aware gate.
(Source: `entities/guardian-eye.md`; surveys of the field: [11], [12])

**Three novelty claims** (Source: `concepts/tri-graph-architecture.md`, `concepts/quality-gated-fusion.md`, `concepts/gqs.md`):
1. **Tri-Graph Architecture** — decompose the violence signal along three semantic axes,
   each a separate graph stream (skeleton, interaction, person-object).
2. **Quality-Gated Fusion (QGF)** — per-sample fusion gates conditioned on GQS; a stream
   with poor detections on a clip is automatically down-weighted, no explicit
   missing-modality protocol needed.
3. **Graph Quality Scores (GQS)** — a 5-d quality vector computed at preprocessing time and
   stored per clip; the bridge between preprocessing quality and inference-time fusion.

**What V9 adds over V8.1** — a fourth **appearance** stream (VideoMAE-Base [1]), an enhanced
12-channel skeleton ST-GCN [3], [4] with adaptive adjacency, and a Transformer temporal encoder
for the interaction stream. (Source: `concepts/tri-graph-architecture.md`, `entities/guardian-eye.md`)

### Portfolio status (as of 2026-06-13)

| Dataset | Status | Best model | Macro-F1 | ROC-AUC | Targets |
|---|---|---|---|---|---|
| **RWF-2000** | COMPLETE | E_full_qgf | 0.9175 | 0.9617 | met |
| **RLVS** | COMPLETE | E_full_qgf | 0.975 | 0.997 | met |
| **NTU CCTV-Fights** | COMPLETE | E_full_qgf | 0.872 | 0.908 | met |
| **UBI-Fights** | PAUSED (honest negative) | E_full_qgf | 0.6725 | 0.7241 | missed |
| **Hockey Fights** | Saturated (V8.1 baseline) | E_full_trigraph | 0.930 | 0.9642 | n/a |

(Sources: `_hot.md`, `entities/rwf-2000.md`, `entities/rlvs.md`, `entities/ntu-cctv-fights.md`,
`entities/ubi-fights.md`, `entities/hockey-fights.md`)

Q1 portfolio target achieved: RWF + RLVS + NTU all complete. (Source: `_hot.md`)

---

## 2. Architecture (V9Model) End-to-End

**Class:** `V9Model` in `fresh/rwf/trigraph_v9_rwf_train.py`.
**Total fusion-head params:** ~798K (excludes the frozen ~86M VideoMAE backbone).
**embed_dim:** 128 for all four stream outputs (VideoMAE raw CLS/patch output is 768-d before projection).
(Source: `sources/v9-architecture.md`)

### High-level dataflow

```
Preprocessed NPZ
   ├─ skeleton arrays    →  EnhancedSTGCN        →  [B,128] ──┐
   ├─ interaction arrays →  ImprovedInteraction  →  [B,128] ──┤
   ├─ object/PO arrays   →  ObjectPOStream       →  [B,128] ──┤─→ QGF/LW → [B,128] → Head → logit
   └─ vit_embedding      →  VitProjection        →  [B,128] ──┘
        [B,768, frozen]
GQS vector [B,5] ───────────────────────────────────────────────→ QGF gate only
```
(Source: `sources/v9-architecture.md`)

### Graph constants (locked in V9)

| Constant | Value | Meaning |
|---|---|---|
| T | 32 | graph frames |
| T_vit | 16 | VideoMAE frames (new in V9) |
| M | 6 | max persons |
| N | 8 | max objects |
| V | 17 | COCO joints |
| C | 12 | skeleton input channels (joint + bone + joint-motion + bone-motion) |

(Sources: `sources/v9-architecture.md`, `sources/v9-rwf-preprocess.md`, `concepts/tri-graph-architecture.md`)

### Classification head

```
Linear(128 → 128) → ReLU → Dropout(0.35) → Linear(128 → 1)
```
Single logit. Decision threshold tuned per experiment on validation macro-F1 — **never fixed
at 0.50**. (Source: `sources/v9-architecture.md`)

### Stream dropout (training only)

`stream_dropout = 0.25` — each stream embedding is zeroed independently with p=0.25 during
training, forcing multi-stream robustness; disabled at inference. (Source: `sources/v9-architecture.md`)

---

## 3. The Four Streams in Detail

### Stream 1 — Skeleton (`EnhancedSTGCN`)

A spatial-temporal graph convolutional network over the COCO skeleton, in the lineage of
ST-GCN [3] with the adaptive-adjacency and multi-stream ideas of [4], [5].
Input: `skeleton [B, T=32, M=6, V=17, 3]`. Feature engineering builds **12 input channels**
from 4 channel types: joint (x,y,conf), bone (joint differences along COCO edges),
joint-motion (temporal diffs of joints), bone-motion (temporal diffs of bone vectors) [4].
(Sources: `sources/v9-architecture.md`, `concepts/stgcn.md`)

- 3 × `GraphTemporalConv` blocks, channels `[12 → 64 → 64 → 128]`.
- Each block: GCN over `A = A_fixed + softmax(A_learned)` → BatchNorm → ReLU → temporal conv.
- `A_fixed` = hand-crafted COCO-17 skeleton adjacency; `A_learned` = per-layer trainable
  additive perturbation (captures non-structural correlations, e.g. wrist-to-wrist for
  fights). This is **Decision D4**.
- **Learned person-attention pooling** over M=6 persons → `[B, 128]` (suppresses bystanders;
  replaces V8.1 mean-pool).
(Sources: `sources/v9-architecture.md`, `concepts/stgcn.md`)

Standalone params: ~71,715 (Ablation A). (Source: `analyses/v9-final-results-2026-05-23.md`)

### Stream 2 — Interaction (`ImprovedInteraction`)

Input: `int_nodes [B,T,M,7]`, `int_edges [B,T,M,M,4]`.
- Per-frame **2-layer GATv2 [24] with edge features**, 4 heads → `[B,T,M,128]`. GATv2 [24] is
  used over the original GAT [23] because its dynamic attention can rank neighbours conditioned
  on edge content (e.g. small distance + high closing speed).
- **Temporal Transformer encoder**: 2 layers, 4 heads, learned positional encoding, over
  T=32 frames (replaces the V8.1 BiGRU — Decision D5).
- Attention pooling over T → `[B,128]`.
(Sources: `sources/v9-architecture.md`, `concepts/gatv2.md`)

**Critical latent bug (NaN-safe softmax):** on frames where all M person slots are empty,
every attention row is `-inf` and a standard softmax produces NaN that collapses the run to
50% accuracy. Rows that are all `-inf` are replaced with `0.0` before softmax. This affected
39.5% of Hockey Fights videos. (Source: `concepts/gatv2.md`)

### Stream 3 — Object / Person-Object (`ObjectPOStream`)

Both branches use GINE [26] (the edge-feature-aware variant of GIN [25]), chosen so the message
function conditions on edge semantics while retaining GIN's expressive power.
Input: `obj_nodes [B,T,N,6]`, `po_edges [B,T,M,N,5]`.
- **Object branch:** `FrameGINE` per frame → BiGRU over T → `[B,64]`.
- **PO branch:** bipartite `FrameGINE` (person→object edges) per frame → BiGRU over T → `[B,64]`.
- Concat → `[B,128]`.

PO bipartite per-frame update for each target person *i*:
```
h_p[i] = LayerNorm( h_p[i] + MLP( masked_mean_j( ReLU( h_o[j] + We(po_edges[i,j]) ) ) ) )
```
The PO stream answers "is a person touching/proximate to an object that could be a weapon or
prop?" — the physical-grounding stream. (Source: `concepts/gine.md`)

**Object/PO unchanged between V8.1 and V9.** Known weakness: object GQS mean ~0.111 on Hockey
Fights (near-empty object signal); object stream hurts when used alone with skeleton and only
helps when PO bipartite binding is active. (Sources: `concepts/gine.md`, `concepts/tri-graph-architecture.md`)

### Stream 4 — VideoMAE Appearance (`VitProjection`)

Input: `vit_embedding [B,768]` — frozen, pre-extracted in Phase 2.
```
Linear(768 → 256) → LayerNorm → ReLU → Linear(256 → 128) → LayerNorm
```
Output `[B,128]`. This projection MLP is the **only trainable part** of the appearance stream
in Phase 3; the ~86M VideoMAE backbone is never loaded during graph training.
(Sources: `sources/v9-architecture.md`, `entities/videomae.md`)

**Backbone facts:** `MCG-NJU/videomae-base` [1], ViT-B/16, ~86M params. VideoMAE-Base has **no
CLS token** — mean-pool over patch tokens (`last_hidden_state`) → 768-d is mandatory. `vit_in_dim`
must be 768 (an earlier 384-d ViT-Small variant is not active). (Sources: `entities/videomae.md`,
`concepts/staged-sequential-transfer.md`; VideoMAE [1] follows the masked-autoencoder paradigm of [32])

---

## 4. Fusion — QGF (Quality-Gated) and LW (Learned-Weight)

### QualityGatedFusion (QGF) — primary

Input: 4 stream embeddings + GQS vector `[B,5]` (q_skel, q_int, q_obj, q_po, valid_ratio).
```
gate_input = GQS[:, :n_streams]                     # [B,4]
gate = LayerNorm(4) → Linear(4→32) → ReLU → Linear(32→4) → Softmax
fused = Σ gate_i · stream_i                          # weighted sum → [B,128]
```
(Source: `sources/v9-architecture.md`)

**The LayerNorm on the gate input is critical.** GQS values cluster near 1.0 (low variance);
without LayerNorm the Linear receives near-zero inputs and produces uniform gates. This is the
single most important QGF fix. (Sources: `sources/v9-architecture.md`, `sources/v9-bugs-fixed.md`)

The diagnostics build adds `return_gates: bool = False`; `forward(batch, return_gates=True)`
returns `(logits, gates [B,n_streams])`. The gate `nn.Sequential` must match the training
checkpoint exactly (LayerNorm → Linear(n→32) → ReLU → Linear(32→n)).
(Source: `sources/v9-rwf-diagnostics.md`)

### LearnedWeightFusion (LW) — ablation control

```
w = softmax(nn.Parameter([4]))     # fixed learned scalars, no GQS input
fused = Σ w_i · stream_i
```
Same weights for every sample; not GQS-conditioned. The control that isolates whether
per-sample quality conditioning helps. (Sources: `sources/v9-architecture.md`, `concepts/quality-gated-fusion.md`)

> Conditioning a combination on a routing signal is the basis of mixture-of-experts [30]; QGF
> differs in that the gate is driven by an externally defined, analytic graph-quality signal
> rather than by the features it weights, so the routing is both quality-aware and auditable.
> The closest conceptual ancestor in violence detection is the optical-flow-quality-aware fusion
> of [29], which folds quality into the fusion implicitly without exposing an explicit score.

### Gate behavior (RWF-2000 E_full_qgf, 400 test samples)

| Stream | Mean | Std | Min | Max |
|---|---|---|---|---|
| skeleton | 24.9% | 2.8% | 20.2% | 27.4% |
| interaction | 20.9% | 2.4% | 18.4% | 24.8% |
| object | 26.0% | 2.6% | 20.0% | 30.1% |
| **vit/VideoMAE** | **28.2%** | 2.3% | 24.6% | 38.4% |

No stream exceeds 50% on any single sample — the gate is well-distributed. The gate–GQS
Pearson correlation is **r = 0.81–0.96**: the gate reads quality and modulates weights
coherently (the early "gate collapse" framing was wrong — it was compressed dynamic range,
not degeneracy). (Sources: `analyses/v9-final-results-2026-05-23.md`, `analyses/v9-diagnostic-review-2026-05-20.md`, `concepts/quality-gated-fusion.md`)

> **Note on a documented discrepancy.** The architecture note (`v9-architecture.md`) lists the
> dominant stream as **object 30.2%**; the final-results note (`v9-final-results-2026-05-23.md`)
> lists **vit 28.2%** as dominant. The vault explicitly reconciles this: the 30.2%-object figure
> is from the *pre-GATv2-fix backup run*; the 28.2%-vit figure is from the *fixed* run. Both are
> within noise and are valid learned outcomes. (Source: `analyses/v9-final-results-2026-05-23.md`)

**Where QGF actually beats LW / VideoMAE.** On RWF-2000 it wins only in the mid/mid-high GQS
bucket (e.g. `(0.566, 0.789]` bucket: +0.030 F1 over VideoMAE; +0.011–0.020 over LW), and ties
or marginally loses elsewhere. Net aggregate over LW is ~+0.003–0.005 F1. QGF adds zero
measurable latency vs LW. (Sources: `analyses/v9-diagnostics-results-2026-05-20.md`,
`analyses/v9-final-results-2026-05-23.md`, `concepts/quality-gated-fusion.md`)

---

## 5. Graph Quality Scores (GQS)

A 5-dimensional float32 vector computed at preprocessing time for every clip, stored as
`gqs [5]` in its NPZ, never modified after. It is the input that makes fusion quality-aware.
(Source: `concepts/gqs.md`)

| Dim | Name | Meaning |
|---|---|---|
| 0 | `q_skel` | fraction of frames where ≥1 person has ≥5 confident joints |
| 1 | `q_int` | fraction of frames with ≥2 valid persons |
| 2 | `q_obj` | fraction of frames with ≥1 valid object |
| 3 | `q_po` | fraction of frames with ≥1 valid PO edge |
| 4 | `valid_ratio` | fraction of frames with ≥1 person detected |

### Exact formulas (corrected)

```python
q_skel      = (person-frame slots with ≥5 visible joints) / T   # CORRECTED (was /(T*M))
q_int       = (frames with ≥2 valid persons) / T
q_obj       = (frames with ≥1 valid object) / T
q_po        = (frames with ≥1 valid PO edge) / T
valid_ratio = (frames with ≥1 person) / T                       # saturates at 1.0 on RWF-2000
```
(Sources: `sources/v9-rwf-preprocess.md`, `concepts/gqs.md`, `sources/v9-bugs-fixed.md`)

### Composite GQS (used for stratification/bucketing)

```
q_composite = 0.4·q_skel + 0.4·q_int + 0.2·q_po
```
On RWF-2000, `valid_ratio` saturates at 1.0 on most clips, collapsing all clips into one
bucket; the composite restores meaningful quartile stratification. Note: RWF quality clusters
so tightly that only **3** non-empty buckets form (not 4) — a dataset property, not a bug.
(Sources: `concepts/gqs.md`, `sources/v9-rwf-preprocess.md`, `analyses/v9-final-results-2026-05-23.md`)

Reference q_obj means: Hockey Fights ~0.111 (near-empty object signal); RWF-2000 valid_ratio→1.0.
(Source: `concepts/gqs.md`)

---

## 6. NPZ Schema and Data Invariants

One NPZ per clip/video. Schema is **identical** across RWF, RLVS, NTU, and UBI so all are
model-compatible. (Sources: `sources/v9-rwf-preprocess.md`, `sources/v9-ubi-preprocess.md`)

```
skeleton      [T, M, V, 3]          joint (x,y,conf), normalised
int_nodes     [T, M, 7]             cx,cy,w,h,conf,speed,continuity
int_edges     [T, M, M, 4]          dist,iou,close,rel_speed
int_node_mask [T, M]                bool
int_edge_mask [T, M, M]             bool
obj_nodes     [T, N, 6]             cx,cy,w,h,conf,cls_norm
obj_node_mask [T, N]                bool
po_edges      [T, M, N, 5]          wrist_d,body_d,iou,near_wrist,near_body
po_edge_mask  [T, M, N]             bool
gqs           [5]                   q_skel,q_int,q_obj,q_po,valid_ratio
frames_vit    [T_vit, 224, 224, 3]  uint8 RGB  ← NEW in V9
vit_embedding [768]                 written back in Phase 2 (frozen)
label         scalar int64
split         scalar str
source        scalar str            ← NTU only (Mobile/CCTV/Car/Other)
```
(Sources: `sources/v9-rwf-preprocess.md`, `entities/ntu-cctv-fights.md`)

### Critical invariant: ID column names differ per dataset

| Item | RWF | UBI | RLVS | NTU |
|---|---|---|---|---|
| Split CSV | `split_v9.csv` | `split_ubi.csv` | `split_rlvs.csv` | (ntu split) |
| ID column | `video_id` | `clip_id` | `clip_id` | `clip_id` |
| GQS summary | `gqs_summary_v9.csv` | `gqs_summary_ubi.csv` | — | — |
| Run dir | `runs_v9/` | `runs_ubi/` | `runs_rlvs/` | `runs_ntu/` |

**⚠ Never mix:** RWF keys on `video_id`; UBI/RLVS/NTU key on `clip_id`. Five separate bugs (A2–A6)
came from copy-pasting `video_id` into UBI scripts. (Sources: `sources/v9-bugs-fixed.md`, `_hot.md`,
`analyses/problems-tackled-ubi-rwf.md`)

---

## 7. Three-Phase Pipeline (Staged Sequential Transfer)

**Decision D3.** VideoMAE is integrated as an appearance stream *without* joint end-to-end
training, because (1) jointly training the ~86M ViT with the ~740K graph head is compute-
prohibitive on Modal L4/T4, and (2) the large appearance model's gradients would dominate and
collapse the tri-graph contribution. (Source: `concepts/staged-sequential-transfer.md`)

```
Phase 1 — Preprocessing (L4, ~6h for 4000 videos)
    YOLO11x-pose + ByteTrack → skeleton/interaction/object/PO tensors
    RGB frame extraction → frames_vit [16,224,224,3] uint8
    GQS computation → one NPZ per video

Phase 2 — VideoMAE fine-tune + embedding extraction (A10G, ~2h)
    Stage A: fine-tune VideoMAE-Base as standalone binary classifier (AdamW, LLRD, EMA)
    Stage B: mean-pool last_hidden_state → 768-d; write vit_embedding back into every NPZ
    → VideoMAE weights NEVER loaded again

Phase 3 — Graph training (L4, ~4h for 6–7 ablations)
    Reads frozen vit_embedding [768] + graph arrays
    Trains EnhancedSTGCN + Transformer interaction + GINE/PO + projection MLP
    QGF conditioned on GQS → ablation table
```
(Sources: `concepts/staged-sequential-transfer.md`, `sources/v9-rwf-preprocess.md`,
`sources/v9-rwf-videomae.md`, `sources/v9-rwf-train.md`)

Key design choices: mean-pool not CLS (no CLS token); atomic NPZ write-back (tmp→rename) so an
interrupted Phase 2 can't corrupt the cache; Phase 3 never imports HuggingFace Transformers
(clean separation). (Source: `concepts/staged-sequential-transfer.md`)

---

## 8. Phase 1 — Preprocessing

**RWF file:** `fresh/rwf/trigraph_v9_rwf_preprocess.py` · `modal run ...::preprocess` · L4 · ~6h.
(Source: `sources/v9-rwf-preprocess.md`)

What it does: download from Kaggle → discover videos → use official split → per video run
YOLO11x-pose [33] + ByteTrack [34] (skeleton/interaction) and YOLO11x [33] (objects) → build
graph arrays + GQS → extract T_vit=16 frames @ 224×224 → save one NPZ + `split_v9.csv` +
`gqs_summary_v9.csv`.

### Constants (CFG)

| Param | Value |
|---|---|
| T / T_vit | 32 / 16 |
| M / N / V | 6 / 8 / 17 |
| pose_conf / obj_conf | 0.25 / 0.25 |
| min_joints (GQS) | 5 |
| pose_weights | `yolo11x-pose.pt` (falls back to l/m/s) |
| obj_weights | `yolo11x.pt` |
| COMMIT_EVERY | 100 videos |

Notes: ByteTrack reset per video via `predictor.trackers[i].reset()`; unreadable frames are
filled by copying the previous frame; `--force-reextract` flag forces re-extraction.
(Source: `sources/v9-rwf-preprocess.md`)

### Per-dataset preprocessing differences

| Aspect | RWF-2000 | UBI-Fights | NTU CCTV-Fights | RLVS |
|---|---|---|---|---|
| Clip unit | full video, uniform linspace | annotation-guided window per fight interval | 5s sliding window | full clip (folder=label) |
| Positives | all "Fight" videos | contiguous fight intervals (Approach B) | windows ≥50% inside a fight segment | Violence/ folder |
| Negatives | all "NonFight" videos | explicitly non-fight frames within source video | windows 0% in fight, **within-video** (+2s guard gap) | NonViolence/ folder |
| ByteTrack reset | per video | per clip | per clip | per clip |
| Annotation parsing | none | frame-level CSVs | `groundtruth.json` temporal segments | none |
| Compute | Modal L4 | Modal L4 (7h timeout) | local WS (RTX 3090) | local WS (RTX 3090) |

(Sources: `sources/v9-rwf-preprocess.md`, `sources/v9-ubi-preprocess.md`,
`entities/ntu-cctv-fights.md`, `entities/rlvs.md`)

**UBI Approach B specifics.** Positive clips = T=32 frames centred on each contiguous fight
interval; negatives = up to `neg_per_video` non-overlapping T=32 windows from explicitly
non-fight frames. CFG adds `neg_per_video` (3, later 4). Forward-seek-only frame reading to
avoid expensive backward seeks. `split_ubi.csv` columns: `clip_id, video_path, ann_path, label,
split, frame_start, frame_end, clip_type, n_ann_frames`. (Source: `sources/v9-ubi-preprocess.md`)

**NTU windowing (LOCKED 2026-06-07).** 5s window (scaled to per-video fps → T=32/T_vit=16),
positive stride 2.5s (50% overlap), negative stride 5s, ≥50% in fight = pos / 0% = neg /
partial discarded, 2s negative guard gap, per-video cap 6 pos + 6 neg, negatives downsampled to
positives per split. **Within-video hard negatives** are structurally forced (NTU has zero
normal videos) — this is exactly the fix UBI never got. (Source: `entities/ntu-cctv-fights.md`)

---

## 9. Phase 2 — VideoMAE Fine-Tune + Embedding Extraction

**RWF file:** `fresh/rwf/trigraph_v9_rwf_videomae.py` · `modal run ...::run_videomae` · A10G · ~2h.
`--skip-finetune` loads an existing checkpoint and jumps to Stage B. (Source: `sources/v9-rwf-videomae.md`)

**Stage A** — fine-tune VideoMAE-Base as a standalone binary classifier on the dataset; input
`frames_vit [16,224,224,3]`; outputs `videomae_best.pt` (EMA checkpoint) and
`videomae_test_results.json` (this becomes Ablation C). Val split is used for early stopping.

**Stage B** (always runs) — load best checkpoint, mean-pool final ViT hidden states → 768-d,
write `vit_embedding [768]` into every NPZ atomically (tmp→rename). VideoMAE is never loaded
again. (Source: `sources/v9-rwf-videomae.md`)

### Fine-tuning recipe (ViTCFG)

| Param | Value |
|---|---|
| model | `MCG-NJU/videomae-base` (Kinetics-400 pretrained) |
| batch_size | 16 (A10G 24GB) |
| epochs | 50 (+resume: adds 20 more if checkpoint found) |
| patience | 10 (lowered to 6 on UBI source-fix run) |
| min_ckpt_epoch | 3 |
| lr | 1e-4 with **LLRD** (layer_decay = 0.75) |
| lr_min | 1e-6 (cosine floor) |
| warmup_epochs | 5 |
| accum_steps | 4 → effective batch 64 |
| grad_clip | 1.0 |
| label_smoothing | 0.05 |
| mixup α / cutmix α | 0.8 / 1.0 (equal probability) |
| dropout | 0.1 (attention) |
| ema_decay | 0.999 |
| weight_decay | 0.05 |

(Sources: `sources/v9-rwf-videomae.md`, `concepts/training-protocol.md`, `entities/videomae.md`)

Implementation: LLRD (layer-wise lr decay, following [32]) assigns head/norm id=0 (highest lr),
`encoder.layer.i` → id=num_layers−i, patch embed → id=num_layers+1 (lowest). Augmentation (train only, consistent across all T
frames): pad 224→256 reflect, random 224×224 crop, HorizontalFlip p=0.5, ColorJitter(0.2,0.2,0.2),
no hue rotation; val/test centre-crop only. Checkpoint saves EMA `state_dict` only.
(Source: `sources/v9-ubi-videomae.md`, `sources/v9-rwf-videomae.md`)

UBI/RWF Phase-2 logic is **byte-identical**; only volume names, dataset classes, split CSV, and
app name change. (Source: `sources/v9-ubi-videomae.md`)

---

## 10. Phase 3 — Graph Training + Ablation

**RWF file:** `fresh/rwf/trigraph_v9_rwf_train.py` · `modal run ...::train` · L4 · ~4h. COMPLETE.
(Source: `sources/v9-rwf-train.md`)

### Training config (CFG)

| Param | Value |
|---|---|
| embed_dim | 128 |
| dropout (head) | 0.35 |
| stream_dropout | 0.25 |
| lr (single-stream A/B) | 3e-4 |
| lr (multi-stream D/E) | 1e-4 |
| weight_decay | 1e-3 |
| label_smoothing | 0.01 (overridable per experiment) |
| batch_size | 64 |
| epochs | 60 |
| early-stop patience | 12 (on val macro-F1) |
| scheduler | ReduceLROnPlateau(factor=0.5, patience=4) on val macro-F1 |
| optimizer | AdamW |
| pos_weight | 1.0 (RWF/RLVS balanced); **4.510 on UBI** |

(Sources: `sources/v9-rwf-train.md`, `sources/v9-training-strategy.md`, `concepts/training-protocol.md`)

### Loss

```
loss = BCEWithLogitsLoss(label_smoothing=0.01) + stream_dropout(0.25)
```
`E_full_qgf_fixed` override adds `entropy_weight=0.10` (gate-entropy regularisation) — not used
in the final best model. (Source: `sources/v9-training-strategy.md`)

### Threshold tuning + temperature calibration

After each experiment, threshold is searched on val: `thr in arange(0.05, 0.90, 0.01)`,
argmax macro-F1. `thr_min=0.05` and `thr_step=0.01` are required to find low thresholds like
0.17 (multi-stream logits skew negative). Then temperature scaling: `T* = minimize_scalar(NLL(val_probs/T),
bounds [0.1,10.0])`, and the threshold is **re-tuned on calibrated probs** (not reset to 0.50).
(Source: `sources/v9-training-strategy.md`)

### Ablation sequence (6–7 experiments)

| ID | Streams | Fusion | Purpose |
|---|---|---|---|
| A_skeleton_only | skeleton | QGF | lower bound |
| B_skel_interaction | skel + int | QGF | +interaction |
| C_videomae_only | vit | — | standalone VideoMAE (read from `videomae_test_results.json`, Phase 2) |
| D_skel_int_obj | skel + int + obj | QGF | +object/PO (graph-only full) |
| E_full_lw | all 4 | LW | full model, no quality gating |
| **E_full_qgf** | **all 4** | **QGF** | **full model + quality gating — BEST** |
| E_full_qgf_fixed | all 4 | QGF | entropy_weight=0.10, temp_anneal — not better |

Ablations are **sequential independent runs, no warm-starting** from previous checkpoints.
GQS bucketing in the loop uses the composite score, falling back from quartile to tertile if
variance is insufficient. (Sources: `sources/v9-rwf-train.md`, `sources/v9-training-strategy.md`)

### Checkpoint format

`runs_v9/{exp_id}/`: `best.pt` = `{model_state_dict, epoch, val_macro_f1, val_roc_auc,
threshold, cal_temp, cal_macro_f1}`; plus `train_history.csv`, `test_metrics.json`,
`gqs_quartile_f1.csv` (QGF/LW only). Global: `runs_v9/experiment_comparison_v9.csv`.
(Source: `sources/v9-training-strategy.md`)

### Per-dataset Phase-3 differences

- **UBI:** `pos_weight=4.510`; raises `RuntimeError` if test split empty (official 67-video
  split, no val fallback); `hard_negative_video_stems=()` placeholder; finer threshold step;
  `min_ckpt_epoch=5`; calibration NLL bounds [0.5,3.0]. (Source: `sources/v9-ubi-train.md`)
- **RLVS:** balanced 1:1 → dynamic `pos_weight ~1.0`; mixup/cutmix 0.8/1.0; 6 ablations; clip_id
  collision guard; fully resumable (atomic tmp→rename, finished experiments skipped via
  `test_metrics.json`). Local WS, no Modal. (Source: `entities/rlvs.md`)
- **NTU:** source-stratified test metrics; custom collate carrying `clip_id`/`source`; LOCAL
  diagnostics import `V9Model` from `train.py` so arch matches checkpoints. (Source: `entities/ntu-cctv-fights.md`)

---

## 11. Diagnostics Methodology

**RWF file:** `fresh/rwf/trigraph_v9_rwf_diagnostics.py`, T4, inference only, outputs to
`./diagnostics_output/`. Five steps; UBI has four (no `inference_speed`). All run on the
`E_full_qgf` checkpoint. (Sources: `sources/v9-rwf-diagnostics.md`, `sources/v9-ubi-diagnostics.md`)

1. **`gate_weights`** — per-sample gate `[B,4]` over the eval set; mean/std per stream, % of
   samples where any stream >50%, gates stratified by true label. → `gate_weights_E_full_qgf.csv`.
2. **`gqs_stratified`** — bucket the eval set by composite GQS; run E_full_qgf and E_full_lw per
   bucket; F1 + AUC per bucket. → `gqs_stratified_comparison.csv`.
3. **`calibration`** — fit T* on val NLL (`scipy.minimize_scalar`, bounds [0.1,10.0]); re-tune
   threshold on calibrated probs; report ECE (15 bins) pre/post. → `calibration_results.json`.
4. **`error_analysis`** — load threshold from checkpoint; segment into TP/TN/FP/FN; report mean
   GQS per segment + top offenders. → `error_analysis.csv`.
5. **`inference_speed`** (RWF only) — WARMUP_BATCHES=5; bs=1 latency (mean/std/p50/p95/p99 with
   CUDA sync); bs=64 throughput. → `inference_speed.csv/.json`.

(Source: `sources/v9-rwf-diagnostics.md`)

Additional UBI-only diagnostics added during the investigation: `shortcut_audit` (prior shift +
quality-alone AUC per split, no model), `embedding_shortcut` (linear probe on raw 768-d
embedding), and the standalone `exp_linear_decorrelation/linear_decorrelation.py` (INLP).
NTU adds `source_shortcut` (per-source pos_rate) and `inspect_errors.py` (FN/FP window
inspection + annotated mp4/contact-sheet export). (Sources: `analyses/v9-ubi-results-2026-05-29.md`,
`entities/ntu-cctv-fights.md`)

---

## 12. Per-Dataset Results

### RWF-2000 (COMPLETE — primary benchmark)

RWF-2000 [9]: official 2,000 videos (1,600 train + 400 val, no separate test); V9 carves 10%
stratified from train for internal val and uses the official val as test. Balanced classes.
(Source: `entities/rwf-2000.md`)

Final ablation (fixed-GATv2 run, 2026-05-23):

| Experiment | Streams | Fusion | Raw F1 | Cal F1 | AUC | Thr | Params |
|---|---|---|---|---|---|---|---|
| A_skeleton_only | skel | QGF | 0.7746 | 0.7770 | 0.8209 | 0.59 | 71,715 |
| B_skel_interaction | skel+int | QGF | 0.8200 | 0.8200 | 0.8874 | 0.48 | 477,522 |
| C_videomae_only | vit | — | 0.9075 | — | 0.9709 | 0.38 | 86M |
| D_skel_int_obj | skel+int+obj | QGF | 0.8425 | 0.8425 | 0.8953 | 0.56 | 568,023 |
| E_full_lw | all 4 | LW | 0.9150 | 0.9150 | 0.9641 | 0.12 | 798,322 |
| **E_full_qgf** | **all 4** | **QGF** | **0.9175** | **0.9175** | **0.9617** | **0.36** | **798,618** |
| E_full_qgf_fixed | all 4 | QGF | 0.9175 | 0.9175 | 0.9624 | 0.35 | 798,618 |

Incremental contribution: skeleton 0.775 → +interaction +0.045 → +object +0.023 →
**+VideoMAE +0.075** (largest jump) → QGF over LW +0.003.
(Source: `analyses/v9-final-results-2026-05-23.md`)

- **Confusion (400 test, thr=0.36):** TP=181, TN=186, FP=19, FN=14 (33 errors, confirmed
  irreducible vs pre-fix run).
- **Calibration:** T*=0.910 (<1, model slightly overconfident). ECE raw 0.041 → post-T 0.048
  (WORSE). **Decision: skip temperature scaling, report raw ECE=0.041.**
- **Inference (T4):** E_full_qgf 90.4ms mean / 93.3ms p95, ~492 clips/sec; QGF adds zero
  latency vs LW. (_hot.md cites ~90ms / ~500 clips/sec / ~517 clips/sec.)
- **Errors:** FNs (n=14) are low-q_skel fights (mean q_skel 0.316; repeat offender
  `Fight_XM0NI7ZmwWM`, OOD, excluded from training but still surfaces). FPs (n=19) are
  high-q_obj aggressive non-fights (sports/rough-play, mean q_obj 0.977). Repeat offenders:
  `1Kbw1bUw` (3 FN), `nuf-d5GugL0` / `2lrARl7utL4` (2 FP each).
(Sources: `analyses/v9-final-results-2026-05-23.md`, `analyses/v9-diagnostics-results-2026-05-20.md`, `_hot.md`)

SOTA bar: MSTFDet [13] 95.2% MCA, RTPNet 93.3%, IDG-ViolenceNet [28] 89.4% acc, CUE-Net [2]
(metrics not directly comparable — Guardian Eye reports AUC + binary F1). (Source: `entities/rwf-2000.md`)

### RLVS — Real Life Violence Situations (COMPLETE)

RLVS [10]: 2,000 clips (1,000/1,000, perfectly balanced); seeded 80/10/10 stratified (seed=42);
local WS (RTX 3090), no Modal. V1 baseline F1 ~0.819 / AUC ~0.884.
(Source: `entities/rlvs.md`)

| Exp | Streams | Acc | Macro-F1 | ROC-AUC | Params |
|---|---|---|---|---|---|
| A | skel | 84.5% | 0.845 | 0.938 | 71K |
| B | skel+int | 88.0% | 0.880 | 0.966 | 478K |
| D | skel+int+obj | 90.5% | 0.905 | 0.971 | 568K |
| **E_full_qgf** | all 4 | **97.5%** | **0.975** | **0.997** | 799K |
| E_full_lw | all 4 (lw) | 97.5% | 0.975 | 0.998 | 798K |
| E_full_qgf_fixed | all 4 (qgf v2) | 97.5% | 0.975 | 0.997 | 799K |
| C | VideoMAE only | 99.5% | 0.995 | 0.9998 | 86M |

Best checkpoint epoch 46 (val_f1=0.990, thr=0.44). Confusion (E_full_qgf test): Fight 99/1,
NonFight 4/96. V9 improvement over V1: +15.6pp F1, +11.3pp AUC. (Source: `entities/rlvs.md`)

**Leakage audit.** RLVS re-cuts the same real-life scene into consecutively-numbered clips. DCT
pHash found 130 scene clusters (483 duplicate pairs); largest cluster 18 clips (NV_683–NV_706);
**all clusters are NonViolence**. Under the random split, 36/200 test clips had a near-duplicate
twin in train/val. Measured impact: removing the 36 leaked clips moved Macro-F1 0.9950 → 0.9936
(**Δ = −0.0014** — not the driver). A group-aware clean split `split_rlvs_clean.csv` (0 straddling
pairs) was built; re-train deferred (Δ too small to justify GPU). **Open limitation:** a
motion-presence shortcut — 99/111 low-`valid_ratio` clips are NonViolence (static scenes:
eating, handshakes, walking), so "absence of motion" itself predicts NonViolence. Graph-only D
(90.5%) still sits between ESTS-GCN's [22] single pipeline (88.7%) and 3-model ensemble (~93%).
(Source: `entities/rlvs.md`)

### NTU CCTV-Fights (COMPLETE)

NTU CCTV-Fights [35]: 1,000 fight videos, ALL annotated, **zero normal videos**; temporal
segments `[start_s,end_s]`; official 500/250/250 split; sources Mobile 707 / CCTV 280 / Other 9 /
Car 4 (mostly Mobile, not CCTV). Converted to binary clip classification by sliding window →
4,940 balanced clips (train 1209/1209, val 624/624, test 637/637). (Source: `entities/ntu-cctv-fights.md`)

| Exp | Streams | Params | macro-F1 | ROC-AUC |
|---|---|---|---|---|
| A | skeleton | 72K | 0.613 | 0.647 |
| B | + interaction | 478K | 0.651 | 0.709 |
| D | + object | 568K | 0.655 | 0.716 |
| C | vit only | 86M | **0.876** | **0.940** |
| **E_full_qgf** | all 4 | 799K | **0.872** | 0.908 |
| E_full_lw | all 4 (lw) | 798K | 0.866 | 0.915 |

Story: pose streams top out ~0.65 (low-res CCTV/Mobile — pose is weak); VideoMAE carries the
signal; the fused E model matches VideoMAE at ~1/100th the params (efficiency claim).
(Source: `entities/ntu-cctv-fights.md`)

Diagnostics (all clean): **shortcut audit** quality-alone AUC 0.45–0.50 = chance (no source/
quality shortcut — strong paper claim); **gates** w_vit=0.32, w_obj=0.24, w_skel=0.24, w_int=0.20,
nearly static (std 0.037, so QGF≈LW); **calibration** temp=1.63, ECE 0.064→0.052 (report
calibrated); **source-stratified** Mobile (n=757) F1=0.879, CCTV (n=502) F1=0.857; **GQS
quartiles** highest-quality bucket has *lowest* F1 (0.838, counterintuitive — confirms quality
is not the shortcut). (Source: `entities/ntu-cctv-fights.md`)

**Error audit (paper-grade).** 91 FNs (median prob 0.055) + 72 FPs (median 0.907), both
confident, spread across 63/49 videos. FNs = label noise (lulls/aftermath inside broad
annotation intervals + compilation-video scene cuts mid-window). FPs = unannotated real violence
(GT recall gap — model correct, GT missed it). Residual 12% error is annotation-granularity
limited, not model-limited. **Do not retrain.** (Source: `entities/ntu-cctv-fights.md`)

Old pre-V9 result (deprecated): Macro-F1 0.654 with naive 87.5%-overlap windowing (~112
clips/source), no within-video hard negatives → scene memorization. Not comparable.
(Source: `entities/ntu-cctv-fights.md`)

### Hockey Fights (saturated, V8.1 baseline)

Hockey Fights [8]: 1,000 clips (500/500), 70/10/20 split (seed=42). Domain: ice hockey broadcast
(highly homogeneous, zero off-domain generalisation). **Label bug:** `nofights` contains substring
`fights` — substring tests must check `nofights` first. Best: V8.1 `E_full_trigraph` Acc 0.930 /
F1 0.9300 / AUC 0.9642. PO stream hurts here (object GQS ~0.111); retained with stream dropout.
Used in the paper ablation/cross-dataset table, not as a headline. (Source: `entities/hockey-fights.md`)

### Pre-V8 historical reference (Models 1–3)

Earlier tri-graph variants before the Modal pipeline. Best (Model 3): RLVS AUC 0.884 / F1 0.819;
AIRTLAB 0.904 / 0.784; SCVD 0.798 / 0.724; Hockey 0.947 / 0.887; **RWF-2000 AUC 0.820 / F1
0.747** (the V9 baseline-to-beat — V9 lifts RWF AUC 0.820 → 0.9617 via VideoMAE). FP/FN failure
signatures (high-interaction benign scenes; low-skeleton fights) are stable across all model
versions — intrinsic to the domain. (Source: `analyses/proposed-model-history.md`)

---

## 13. The UBI-Fights Investigation

UBI is the project's **honest negative result** — every fix was tried and documented; targets
(F1 ≥ 0.85, AUC ≥ 0.93) were missed and reported honestly. Best ever: E_full_qgf Phase 3 V2,
test AUC **0.7241**, F1 0.6725. (Sources: `entities/ubi-fights.md`, `analyses/v9-ubi-results-2026-05-29.md`)

**Dataset.** 5,889 clips after V9 tiled extraction. Per-split priors: train 1,295 pos / 2,957
neg (30.5% pos), val 203/744 (21.4% pos), **test 427/263 (61.9% pos — inverted vs train)**.
Official 67-video test split (40 fight + 27 non-fight), video-level. SOTA bar: CvlBiLT 98.56 AUC.
(Source: `entities/ubi-fights.md`)

**Pipeline ran fine; the results were misleading** — val AUC 0.91–0.92 every epoch while test
AUC ~0.71, a 21-point val→test collapse a plateau cannot explain. The investigation chain:

1. **Initial mis-diagnosis: "VideoMAE plateau."** Wrong — a plateau gives matching val/test; a
   *gap* is distribution shift.
2. **Bug 1 — inverted prior (prior shift).** Every threshold and `pos_weight` was calibrated to
   the wrong prior. Mitigated with a prior-matched + quality-decorrelated WeightedRandomSampler,
   `pos_weight ≈ 0.613` (eval prior), and prior-matched threshold search. Didn't fix transfer.
3. **Bug 2 — non-transferable quality shortcut (the AUC killer).** In train, fights have far
   cleaner skeletons than non-fights (q_skel 0.91 vs 0.67) → quality alone predicts the label at
   AUC 0.67 on train; on test the negatives are also clean (q_skel 0.78) so the shortcut breaks
   (AUC 0.58). The model learned "absolute clean skeleton → fight."
4. **Why QGF can't beat LW on UBI.** The QGF gate is fed raw GQS — it is literally handed the
   non-transferable shortcut feature. (Contrast: RWF is balanced and same-distribution, so the
   quality signal is valid there — the gate code is byte-identical between RWF and UBI.)
5. **Phase 3 V2 re-run (8 experiments, fixed sampler/prior/threshold + 2 new gate ablations
   E_full_cgf content-gate, E_full_qgf_adv adversarial GRL gate).** Best test AUC 0.7241 vs old
   0.7066 — **head-side fixes did not help**; the new shortcut-breaking gates scored identical.
6. **`embedding_shortcut` diagnostic.** A linear probe on the raw 768-d `vit_embedding`
   (no graph, no gate, no GQS) reproduces the collapse: train 0.984 / val 0.916 / **test 0.681**.
   The shortcut is frozen inside the VideoMAE embedding, upstream of the entire graph head.
7. **Option 2 — linear decorrelation (INLP).** Stripping k={0,1,2,4,8} linear quality directions
   moved test AUC by +0.001 and the val→test gap stayed pinned at +0.235; removing linear q_skel
   ≠ removing the label shortcut → **the shortcut is non-linear**, can't be projected out post-hoc.
8. **Option 3 — source fix (chosen + implemented).** Isolated folder
   `fresh/ubi/exp_source_fix_negquality/`: quality-stratified negative selection
   (`quality_match="fight_quantile"`, `neg_oversample=2.0`) so train negatives match the fight
   q_skel level, + quality×label balanced sampler, + fresh re-finetune.
   - Phase 1 (preprocess) result: 5,887 kept clips; train neg q_skel **0.670 → 0.786**; train
     pos−neg q_skel gap 0.240 → 0.123; train-vs-test separability mismatch 0.089 → 0.027. The
     GQS-level shortcut was broken. **GQS gate PASSED.**
   - Phase 2 (VideoMAE re-finetune) result: best epoch 18, val F1=0.8328, AUC=0.9205; standalone
     test acc 0.728 / F1 0.686 / AUC 0.703. **Embedding gate FAILED** — linear probe on the fresh
     embedding still 0.899 val / **0.676 test** (gap 0.235 → 0.223, barely moved).
9. **True root cause (post-mortem): scene/video-identity memorization, not quality.** 80.6% of
   train source videos are single-label (599 all-nonfight, 3 all-fight, 145 mixed) vs only 40.3%
   single-label in test. When 80% of train videos are single-label, VideoMAE reaches val AUC 0.90
   by memorizing "this scene/camera → label" rather than detecting fights; val shares the scene
   distribution (inflated), but the official test is 60% mixed-label and demands *within-video
   temporal discrimination* the scene-memorizing embedding can't do. q_skel was a proxy, not the
   shortcut itself — the real shortcut is visual scene identity, orthogonal to quality.

**What would actually fix it (not done):** mine **within-video hard negatives** — non-fight clips
from the SAME fight video — so the model must learn temporal discrimination, not scene identity.
This is the exact fix NTU structurally forced (and why NTU succeeded where UBI failed). Requires
a preprocessing redesign; user paused UBI as too costly for marginal progress.
(Source: `analyses/v9-ubi-results-2026-05-29.md`)

**UBI cost:** source-fix run Phase 1 ~$10 + Phase 2 ~$8–10 ≈ $20, for the same AUC as before.
(Source: `_hot.md`, `analyses/v9-ubi-results-2026-05-29.md`)

### UBI lessons (generalizable)

1. A val→test *gap* = distribution shift, not a plateau (a plateau gives matching val/test).
2. q_skel was a proxy, not the shortcut itself; fixing it didn't fix the gap.
3. Parallel sharding does NOT save credits — only wall time (same GPU-seconds).
4. The pre-run `embedding_shortcut` diagnostic is mandatory — run before Phase 3; it would have
   saved ~$15 of Phase 3 re-runs.
5. Within-video hard negatives are essential whenever train videos are predominantly
   single-label but test videos are mixed.
(Source: `_hot.md`)

---

## 14. All Bugs Fixed

Canonical record: `sources/v9-bugs-fixed.md`. Consolidated master list:
`analyses/problems-tackled-ubi-rwf.md`. **All FIXED — do not revert.**

### A. Code bugs (crash or silent corruption)

| # | Bug | Dataset(s) | Fix |
|---|---|---|---|
| A1 | **GATv2 multi-head collapsed to single-head** — `a.mean(-1)` averaged 4 heads into 1 scalar before multiplying V (effectively single-head). | all 4 train/diag scripts | einsum `bimh,bmhd->bihd` + reshape. Impact: RWF B +0.005 F1, D +0.009 F1 |
| A2 | `video_id` used on UBI CSV that has `clip_id` (KeyError on startup) | UBI train | `sub["video_id"]` → `sub["clip_id"]` |
| A3 | GQS stratification used `video_id` on UBI (KeyError swallowed → no quartile CSV) | UBI train | `grp["video_id"]` → `grp["clip_id"]` |
| A4 | VideoMAE `get_split()` used `video_id` on UBI | UBI videomae | column fix |
| A5 | Test split missing from Stage B embedding extraction (test clips had no `vit_embedding`) | UBI videomae | add `test_paths` to `all_paths` |
| A6 | `video_id` in diagnostics `_load_split` + `error_analysis` GQS lookup | UBI diagnostics | both → `clip_id` |
| A7 | `pw_val` used before defined (NameError) | UBI videomae | move block up (caught by audit) |
| A8 | `ann_map` referenced before assignment (UnboundLocalError) | UBI preprocess | reorder |

Root invariant behind A2–A6: RWF keys `video_id`, UBI keys `clip_id`. (Source: `sources/v9-bugs-fixed.md`)

### B. VideoMAE training failures (UBI)

| # | Bug | Fix |
|---|---|---|
| B1 | VideoMAE collapsed to all-negatives (val_f1=0.0 for 13 epochs) — no `pos_weight` under imbalance | dynamic `pos_weight = n_neg/n_pos` |
| B2 | No WeightedRandomSampler (imbalance-skewed batches) | added sampler (~50/50) |
| B3 | Mixup/CutMix too aggressive (α 0.8/1.0) on imbalanced data | lowered to 0.4/0.4 |
| B4 | Empty val split → checkpointed on val_f1=0.0, early-stopped at epoch 13 | carve 20% of non-test videos into val (seed=42) |
| B5 | Resume cap shrank epoch budget (`epochs = ck["epoch"]+20`) | `max(epochs, ck["epoch"]+20)` so cap only grows |
| B6 | First fine-tune timed out at 7h (~$10) — Stage B OOM loading all 5,889 clips into RAM | lazy per-batch disk reads |

(Source: `analyses/problems-tackled-ubi-rwf.md`)

### C. Preprocessing / data-quality

| # | Bug | Fix |
|---|---|---|
| C1 | Short positive clips (<16 frames) had `frames_vit` = one frame repeated 16× | drop intervals < T//2 = 16 frames |
| C2 | Long positive clips → one near-static centred window | tile up to `max_windows=3` per interval (linspace start-points) |
| C3 | Clip-count explosion (uncapped tiling → 21,059 clips, ~14h) | cap `max_windows=3` → 5,889 clips (~4h) |
| C4 | neg:pos ratio drift (4.5:1 post-tiling) | `neg_per_video` 3→4 to hold ~2.3:1 |
| C5 | **q_skel divided by T×M instead of T** — values compressed ~6× (M=6), never reached [0,1] | `valid_skel/T`; existing NPZs patched via `patch_gqs_q_skel.py` |
| C6 | `patch_gqs_q_skel` timeout (3,792 files at 1.5/s = 42min, limit was 30min) | raise to 60min (patch is idempotent) |

(Sources: `analyses/problems-tackled-ubi-rwf.md`, `sources/v9-bugs-fixed.md`)

### D. Methodology / interpretation (the expensive ones)

D1 UBI root cause mis-diagnosed as "VideoMAE plateau" (corrected to distribution shift) ·
D2 inverted prior · D3 scene-memorization shortcut (frozen non-linearly in embeddings; UNDERSTOOD,
paused) · D4 QGF gate fed raw GQS amplifies the shortcut · D5 why RWF worked but UBI didn't (RWF
balanced + same-distribution; UBI is a data-condition problem, not architecture) · D6 stale UBI
clip counts in vault corrected (3,960 → 5,889). (Source: `analyses/problems-tackled-ubi-rwf.md`)

### E. RWF findings (results, not bugs)

E1 object/PO stream dominates the gate (~30%) and fires on any dense scene · E2 QGF advantage is
narrow (one bucket) · E3 temperature scaling worsens calibration (skip it) · E4 33 irreducible
errors · E5 GQS clusters too tightly for 4 quartiles (only 3 on RWF) · E6 `Fight_XM0NI7ZmwWM`
excluded as hard negative still surfaces FNs. (Source: `analyses/problems-tackled-ubi-rwf.md`)

### F. Operational / infrastructure

F1 Modal account ran out of credits ($3.96 left) → switched to fresh $30 account, re-ran
preprocessing (volume not transferable) · F2 A10G spot preemption killed VideoMAE → `retries=2`
for checkpoint-resume · F3 SSL `UNSAFE_LEGACY_RENEGOTIATION_DISABLED` dropped a graph run mid-loop
→ relaunch with `modal run --detach` + `cfg.resume=True` per-experiment skip-and-reload.
(Source: `analyses/problems-tackled-ubi-rwf.md`)

**The mandatory 5-point pre-run audit** (catches A2–A7, B5, D1): CSV/NPZ/path column names;
tensor shapes; all splits handled; ratio denominators; Modal volume/mount/decorator.
(Source: `analyses/problems-tackled-ubi-rwf.md`)

---

## 15. Version History

| Version | Status | Key change | Primary dataset |
|---|---|---|---|
| V4.2–V7 | historical | flat aggregates → graph residuals | Hockey Fights |
| V8.1 | locked baseline | full 4-stream, QGF, ByteTrack, ~521K params | Hockey, SCVD |
| V8.2 | active | V8.1 on UBI, Approach B clip extraction | UBI-Fights |
| V8.3 | comparison | V8.1 on RWF, no VideoMAE | RWF-2000 |
| **V9** | primary | Enhanced STGCN + Transformer interaction + VideoMAE, ~798K fusion params | RWF, RLVS, NTU, UBI |

(Source: `entities/guardian-eye.md`)

### V8.1 → V9 stream-level changes

| Stream | V8.1 | V9 |
|---|---|---|
| Skeleton | CompactSTGCN, C=3 (x,y,conf), fixed COCO adjacency, mean-pool over M | EnhancedSTGCN, **C=12** (joint+bone+jm+bm), **adaptive adjacency** (fixed + softmax(learned)), **learned attention pooling** |
| Interaction | FrameGATv2 (1 layer) + BiGRU + attn pool | FrameGATv2 (**2 layers, 4 heads**) + **2-layer Transformer encoder** (learned pos enc) |
| Object / PO | FrameGINE + BiGRU | **unchanged** |
| Appearance | absent | **VideoMAE projection MLP (768→256→128)** |

(Sources: `concepts/tri-graph-architecture.md`, `concepts/stgcn.md`, `concepts/gatv2.md`, `concepts/gine.md`)

### V8.2 vs V9 (both on UBI, useful contrast)

V8.2 used CompactSTGCN(C=3), a *bigger* gate MLP that concatenates stream embeddings + GQS
(`Linear(n*embed+5 → embed → n)`) vs V9's GQS-only `Linear(n→32→n)`, stream_dropout 0.35 (V9:
0.25), label_smoothing 0.01, no LW ablation, and no VideoMAE. Skeleton GQS in V8.2 included a
temporal-stability term (`0.70·mean + 0.30·skel_stab`) vs V9's simpler `q_skel`. (Source: `sources/v8-2-ubi-fights.md`)

### Shared training protocol (V8.1 / V9 graph training)

AdamW · lr 3e-4 single / 1e-4 multi · ReduceLROnPlateau(factor 0.5, patience 4) on val
Macro-F1 · weight_decay 1e-3 · dropout 0.35 · stream_dropout 0.25 · grad_clip 1.0 ·
checkpoint on val Macro-F1 · `min_checkpoint_epoch=5` (guards lucky warm-up checkpoints) ·
NaN-batch guard (skip+count; interaction stream on empty-person frames was the trigger).
Dataset overrides: UBI `pos_weight=4.510`; SCVD WeightedRandomSampler + reduced config; Hockey
`head_dropout=0.35`. (Source: `concepts/training-protocol.md`)

---

## 16. Operational Notes & Costs

- **Sharding does NOT save credits** — cuts wall time only; total GPU-seconds identical. (`_hot.md`)
- **Modal GPU pricing:** T4 $0.59, L4 $0.80, A10G $1.10, A100-40 $2.10, H100 $3.95 /hr. (`_hot.md`)
- **UBI total (source-fix run):** Phase 1 ~$10 + Phase 2 ~$10 ≈ $20; result: same AUC as before.
- RWF/UBI run on **Modal cloud**; RLVS and NTU run **locally** on the workstation
  (Intel Xeon Gold 6230R, 64GB RAM, RTX 3090 24GB, Windows 11 Pro for Workstations). (`entities/rlvs.md`)
- Diagnostics run on T4 (inference only) and output to local `./diagnostics_output/`.
- Atomic NPZ write-back (tmp→rename) protects the cache against interrupted Phase 2.

---

## 17. Source Map (vault files this document draws from)

**Sources (`wiki/sources/`):** v9-architecture, v9-bugs-fixed, v9-training-strategy,
v9-rwf-preprocess, v9-rwf-videomae, v9-rwf-train, v9-rwf-diagnostics, v9-ubi-preprocess,
v9-ubi-videomae, v9-ubi-train, v9-ubi-diagnostics, v8-2-ubi-fights.

**Concepts (`wiki/concepts/`):** tri-graph-architecture, quality-gated-fusion, gqs,
staged-sequential-transfer, training-protocol, stgcn, gatv2, gine.

**Entities (`wiki/entities/`):** guardian-eye, videomae, rwf-2000, rlvs, ntu-cctv-fights,
ubi-fights, hockey-fights.

**Analyses (`wiki/analyses/`):** v9-final-results-2026-05-23, v9-ubi-results-2026-05-29,
v9-diagnostics-results-2026-05-20, v9-diagnostic-review-2026-05-20, proposed-model-history,
problems-tackled-ubi-rwf.

**Context:** `wiki/_hot.md`, `wiki/log.md`.

### Open gaps flagged during the sweep

- **Hockey Fights V9 results:** no full V9 ablation table exists in the vault for Hockey — only
  the V8.1 baseline (Acc 0.930). `_hot.md` mentions "HF dataset results" as *pending* paper work.
  Hockey is treated as a saturated V8.1 baseline, not a V9 headline.
- **SCVD / AIRTLAB:** appear only in the pre-V8 history (`proposed-model-history`); no V9 pipeline.
  SCVD/XD-Violence are noted as an optional future 4th dataset. (Source: `_hot.md`)
- Some SCVD baseline-variant AUC cells in `proposed-model-history.md` are blank in the source.

---

## 18. References

Numbering matches the paper (`guardian_eye_paper.bbl`, IEEE order of first citation). Inline
`[n]` markers throughout this document point here.

[1] Z. Tong, Y. Song, J. Wang, and L. Wang, "VideoMAE: Masked autoencoders are data-efficient
learners for self-supervised video pre-training," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*,
vol. 35, 2022, pp. 10078–10093.

[2] D. C. Senadeera, X. Yang, D. Kollias, and G. Slabaugh, "CUE-Net: Violence detection video
analytics with spatial cropping enhanced UniformerV2 and modified efficient additive attention,"
in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2024, pp. 4888–4897.

[3] S. Yan, Y. Xiong, and D. Lin, "Spatial temporal graph convolutional networks for
skeleton-based action recognition," in *Proc. AAAI Conf. Artif. Intell.*, vol. 32, no. 1, 2018.

[4] L. Shi, Y. Zhang, J. Cheng, and H. Lu, "Two-stream adaptive graph convolutional networks for
skeleton-based action recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR)*, 2019, pp. 12026–12035.

[5] Y. Chen, Z. Zhang, C. Yuan, B. Li, Y. Deng, and W. Hu, "Channel-wise topology refinement
graph convolution for skeleton-based action recognition," in *Proc. IEEE/CVF Int. Conf. Comput.
Vis. (ICCV)*, 2021, pp. 13359–13368.

[6] Z. Islam, M. Rukonuzzaman, R. Ahmed, M. H. Kabir, and M. Farazi, "Efficient two-stream
network for violence detection using separable convolutional LSTM," in *Proc. Int. Joint Conf.
Neural Netw. (IJCNN)*. IEEE, 2021, pp. 1–8.

[7] P. Nardelli and D. Comminiello, "JOSENet: A joint stream embedding network for violence
detection in surveillance videos," *arXiv preprint arXiv:2405.02961*, 2024.

[8] E. Bermejo Nievas, O. Deniz Suarez, G. Bueno García, and R. Sukthankar, "Violence detection
in video using computer vision techniques," in *Proc. 14th Int. Conf. Comput. Anal. Images
Patterns (CAIP)*. Springer, 2011, pp. 332–339.

[9] M. Cheng, K. Cai, and M. Li, "RWF-2000: An open large scale video database for violence
detection," in *Proc. 25th Int. Conf. Pattern Recognit. (ICPR)*. IEEE, 2021, pp. 4183–4190.

[10] M. M. Soliman, M. H. Kamal, M. A. E.-M. Nashed, Y. M. Mostafa, B. S. Chawky, and D. Khattab,
"Violence recognition from videos using deep learning techniques," in *Proc. 9th Int. Conf.
Intell. Comput. Inf. Syst. (ICICIS)*. IEEE, 2019, pp. 80–85.

[11] F. U. M. Ullah, M. S. Obaidat, A. Ullah, K. Muhammad, M. Hijji, and S. W. Baik, "A
comprehensive review on vision-based violence detection in surveillance videos," *ACM Comput.
Surv.*, vol. 55, no. 10, pp. 1–44, 2023.

[12] P. Negre, R. S. Alonso, A. González-Briones, J. Prieto, and S. Rodríguez-González,
"Literature review of deep-learning-based detection of violence in video," *Sensors*, vol. 24,
no. 12, p. 4016, 2024.

[13] B. Qi, B. Wu, and B. Sun, "Automated violence monitoring system for real-time fistfight
detection using deep learning-based temporal action localization," *Sci. Rep.*, vol. 15, no. 1,
p. 29497, 2025.

[14] F. J. Rendón-Segador, J. A. Álvarez-García, and L. M. Soria-Morillo, "Transformer and
adaptive threshold sliding window for improving violence detection in videos," *Sensors*, vol. 24,
no. 16, p. 5429, 2024.

[15] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in
*Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 6479–6488.

[16] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W. Verjans, and G. Carneiro, "Weakly-supervised
video anomaly detection with robust temporal feature magnitude learning," in *Proc. IEEE/CVF Int.
Conf. Comput. Vis. (ICCV)*, 2021, pp. 4975–4986.

[17] Y. Chen, Z. Liu, B. Zhang, W. Fok, X. Qi, and Y.-C. Wu, "MGFN: Magnitude-contrastive
glance-and-focus network for weakly-supervised video anomaly detection," in *Proc. AAAI Conf.
Artif. Intell.*, vol. 37, no. 1, 2023, pp. 387–395.

[18] Z. Liu, H. Zhang, Z. Chen, Z. Wang, and W. Ouyang, "Disentangling and unifying graph
convolutions for skeleton-based action recognition," in *Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR)*, 2020, pp. 143–152.

[19] H. Duan, J. Wang, K. Chen, and D. Lin, "DG-STGCN: Dynamic spatial-temporal modeling for
skeleton-based action recognition," *arXiv preprint arXiv:2210.05895*, 2022.

[20] H. Yang, Z. Ren, H. Yuan, W. Wei, Q. Zhang, and Z. Zhang, "Multi-scale and attention
enhanced graph convolution network for skeleton-based violence action recognition," *Front.
Neurorobot.*, vol. 16, p. 1091361, 2022.

[21] G. Garcia-Cobo and J. C. SanMiguel, "Human skeletons and change detection for efficient
violence detection in surveillance videos," *Comput. Vis. Image Underst.*, vol. 233, p. 103739,
2023.

[22] N. F. Janbi, M. A. Ghaseb, and A. A. Almazroi, "ESTS-GCN: An ensemble spatial–temporal
skeleton-based graph convolutional networks for violence detection," *Int. J. Intell. Syst.*,
vol. 2024, p. 2323337, 2024.

[23] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention
networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.

[24] S. Brody, U. Alon, and E. Yahav, "How attentive are graph attention networks?" in *Proc.
Int. Conf. Learn. Represent. (ICLR)*, 2022.

[25] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?" in
*Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[26] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec, "Strategies for
pre-training graph neural networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

[27] J. Ji, R. Krishna, L. Fei-Fei, and J. C. Niebles, "Action genome: Actions as compositions of
spatio-temporal scene graphs," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*,
2020, pp. 10236–10247.

[28] H. Huang and Q. Jiang, "IDG-ViolenceNet: A video violence detection model integrating
identity-aware graphs and 3D-CNN," *Sensors*, vol. 25, no. 20, p. 6272, 2025.

[29] Y. Xiao, G. Gao, L. Wang, and H. Lai, "Optical flow-aware-based multi-modal fusion network
for violence detection," *Entropy*, vol. 24, no. 7, p. 939, 2022.

[30] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local
experts," *Neural Comput.*, vol. 3, no. 1, pp. 79–87, 1991.

[31] K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, L. Wang, and Y. Qiao, "UniFormerV2: Unlocking the
potential of image ViTs for video understanding," in *Proc. IEEE/CVF Int. Conf. Comput. Vis.
(ICCV)*, 2023, pp. 1632–1643.

[32] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, "Masked autoencoders are scalable
vision learners," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp.
16000–16009.

[33] G. Jocher, J. Qiu, and A. Chaurasia, "Ultralytics YOLO11,"
https://github.com/ultralytics/ultralytics, 2024.

[34] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang,
"ByteTrack: Multi-object tracking by associating every detection box," in *Proc. Eur. Conf.
Comput. Vis. (ECCV)*. Springer, 2022, pp. 1–21.

[35] M. Perez, A. C. Kot, and A. Rocha, "Detection of real-world fights in surveillance videos,"
in *Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)*. IEEE, 2019, pp. 2662–2666.

> **Note.** References [6], [7], [14]–[21], [27], [31] are part of the paper's bibliography
> (related-work and baseline context) but are not yet cited inline in this documentation; they
> are listed here to keep the numbering identical to the paper. The UBI-Fights SOTA reference
> (CvlBiLT, 98.56 AUC, cited in Section 13) is a vault source (`sources/bilt-three-stage-2024.md`)
> that is not in the paper bibliography, so it carries no `[n]` number.

---

*Compiled 2026-06-13 from the Guardian Eye project vault. Every metric is traceable to the cited
note; no values were estimated. Where the vault is silent, the gap is flagged above rather than
filled. Reference numbering matches `guardian_eye_paper.bbl`.*
