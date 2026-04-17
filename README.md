# Guardian Eye — Tri-Graph Violence Detection Pipeline (v2)

> **Binary violence detection for CCTV surveillance footage using a tri-stream graph neural network architecture.**

Guardian Eye is a graduation research project in AI and Data Science. It processes raw surveillance video and classifies clips as violent or non-violent by jointly reasoning over three complementary graph representations: human skeleton motion, person–person interaction, and person–object context. The system spans a complete research pipeline from raw video ingestion to training and evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
  - [Preprocessing (v2)](#1-preprocessing-v2)
  - [Graph Construction](#2-graph-construction)
  - [Graph Quality Scoring (GQS)](#3-graph-quality-scoring-gqs)
  - [Data Preparation](#4-data-preparation)
  - [Tensorization (v2)](#5-tensorization-v2)
  - [Model Architecture](#6-model-architecture)
- [Datasets](#datasets)
- [Experimental Results](#experimental-results)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Key Design Decisions](#key-design-decisions)
- [Citation](#citation)

---

## Project Overview

Violence detection in CCTV footage is a safety-critical task where false negatives (missed violence) are more costly than false positives. Guardian Eye addresses this challenge through a multi-modal graph-based representation that encodes:

- **Who** is present and **how they are moving** (skeleton stream)
- **How people interact** with each other spatially (interaction stream)
- **What objects** are present and their spatial relationship to persons (object stream)

The three streams are fused by a quality-aware attention mechanism and processed by a Transformer temporal encoder before binary classification.

The pipeline is designed to degrade gracefully under real-world CCTV conditions — low resolution, partial occlusion, missing detections — through learned null stream embeddings and per-frame stream quality scores.

---

## Architecture

The v2 pipeline consists of six sequential stages:

### 1. Preprocessing (v2)

**Script:** `preprocessing_v2_fixed.py`

Raw video is processed at a target of ~15 fps (stride = 2) using YOLO-based inference:

- **Person detection** — YOLOv8 detector identifies person bounding boxes per frame.
- **Pose estimation** — YOLOv8-Pose extracts 17-joint COCO-format skeletons per person.
- **Object detection** — YOLOv8 detector identifies non-person objects with class labels and bounding boxes.
- **Identity tracking** — ByteTrack assigns stable person IDs across frames to enable temporal skeleton linkage and interaction graph construction.
- **Adaptive enhancement** — Per-frame brightness and contrast estimation selects one of three enhancement modes (None / Light / Adaptive CLAHE) to stabilize detection confidence in low-light or high-dynamic-range scenes.
- **Fallback retry logic** — Frames that fail detection are retried with relaxed confidence thresholds before being marked as empty.

**Outputs per video:**
```
skeleton_graph/     # Per-frame skeleton data with joint coordinates and confidence
interaction_graph/  # Per-frame person–person spatial interaction data
object_graph/       # Per-frame person–object spatial context data
meta.json           # Video metadata, frame count, detection statistics
```

---

### 2. Graph Construction

Three parallel graph representations are constructed per frame:

#### Skeleton Graph
Each detected person is represented as a graph following the COCO-17 joint topology. Node features encode joint coordinates (normalised) and detection confidence. Edges follow the standard COCO kinematic tree. Joints below a confidence threshold `τ = 0.25` are masked and their edges removed. Incomplete skeletons are retained rather than discarded — the model is trained to operate on partially observed graphs, which is essential for real CCTV footage.

#### Interaction Graph
Person–person spatial relationships are encoded as a graph where nodes represent persons and edges encode proximity features. Edges are constructed using both proximity thresholding and a KNN fallback (`K_nn = 1`) to ensure connectivity even in sparse scenes. Edge features include the normalised distance, an `is_real` flag, and an `is_knn` flag so the model can distinguish genuine interaction edges from fallback edges.

#### Object Graph
Person–object spatial context is encoded where nodes represent detected objects (with learned class embeddings) and edges connect objects to nearby persons. Node features include the object bounding box centre coordinates, size, class embedding, and detection confidence. Edge features encode distance and relative spatial position.

---

### 3. Graph Quality Scoring (GQS)

A per-frame, per-stream quality score in `[0, 1]` is computed for each of the three graphs:

- **Skeleton quality** — weighted function of the fraction of joints detected, mean confidence, and temporal stability of tracked persons.
- **Interaction quality** — fraction of real (non-KNN-fallback) interaction edges, weighted by proximity score.
- **Object quality** — fraction of frames with detected objects and mean detection confidence.

A composite **Graph Quality Score (GQS)** aggregates the three stream scores:

```
GQS = α · Q_skel + β · Q_int + γ · Q_obj
```

where `α = 0.40`, `β = 0.30`, `γ = 0.30`.

> **Important:** GQS is a diagnostic metric only. It is not used as a loss signal or training objective. It is used post-hoc to stratify clips by data quality and diagnose model behaviour on different quality tiers.

**Observed quality statistics across datasets (v2 pipeline):**

| Dataset | Q_skel (mean) | Q_int (mean) | Q_obj (mean) | Frames with any stream = 0 |
|---------|--------------|-------------|-------------|---------------------------|
| SCVD    | 0.380        | 0.122       | 0.460       | 71.4%                     |
| NTU CCTV-Fights | 0.493 | 0.188      | 0.414       | 54.7%                     |

SCVD's low skeleton and interaction quality reflects challenging low-resolution footage with frequent occlusion. NTU CCTV-Fights produces materially better stream completeness, confirming that GQS correctly identifies data quality as a dataset property rather than a fixed pipeline limitation.

---

### 4. Data Preparation

**Script:** `dataprep_v2.py`

Fixed-length clips are constructed from the preprocessed graph sequences:

- **Clip length:** 16 frames, stride = 8 (50% overlap between adjacent clips).
- **Splits:** Video-level 70 / 15 / 15 (train / val / test) for datasets without official splits. Official benchmark splits are used where provided (NTU CCTV-Fights uses the official 500 / 250 / 250 video split).
- **Labeling strategy:**
  - SCVD, RLVS, AIRTLab, RWF-2000: folder-based binary labels.
  - NTU CCTV-Fights: temporal overlap labeling — a clip receives label 1 if ≥ 50% of its frame duration falls within annotated fight segments (derived from `groundtruth.json`).
- **Output:** Per-dataset `index_with_splits.csv` with clip paths, labels, split assignments, and empty-frame ratios per stream.

---

### 5. Tensorization (v2)

**Script:** `tensorization_v2.py`  
**Output directory:** `tensorized_fixed/`

Clips are converted to fixed-size tensors for efficient batch loading:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `skeleton/joint` | `[5, T, V, M]` | Relative+absolute coords + confidence |
| `skeleton/bone` | `[3, T, V, M]` | Bone direction vectors + confidence |
| `skeleton/joint_motion` | `[5, T, V, M]` | Temporal joint displacement |
| `skeleton/bone_motion` | `[3, T, V, M]` | Temporal bone displacement |
| `skeleton/velocity` | `[3, T, V, M]` | First-order velocity + confidence |
| `skeleton/acceleration` | `[3, T, V, M]` | Second-order acceleration + confidence |
| `stream_quality` | `[T, 3]` | Per-frame quality scores (skel, int, obj) |

**Key capacity constants:**

```python
M_MAX      = 10   # maximum persons per frame
N_INT_MAX  =  8   # maximum interaction graph nodes
E_INT_MAX  = 10   # maximum interaction graph edges
N_OBJ_MAX  = 14   # maximum object graph nodes
E_OBJ_MAX  = 12   # maximum object graph edges
```

The tensorizer is dataset-agnostic; only the path configuration in Cell A requires updating for a new dataset.

---

### 6. Model Architecture

**Script:** `tri_graph_model_v2.py`

The model is a tri-stream GNN with Transformer temporal fusion:

#### Skeleton Branch — MSA-STGCN
Six skeleton stream encoders process the six stream tensors (joint, bone, joint_motion, bone_motion, velocity, acceleration) through independent MSA-STGCN backbones. Each backbone uses:
- Multi-scale spatial graph convolution (3 scales).
- Squeeze-and-Excitation channel attention.
- Multi-branch temporal convolution with parallel dilations (kernels 3 and 5, dilations 1 and 2).
- Residual connections throughout.
- Person attention pooling for variable-person aggregation.

#### Interaction Branch — GATv2
Batched GATv2 graph attention network processes the disconnected-graph batched interaction graphs. Edge features (weight, `is_real`, `is_knn`) are incorporated into the attention computation. Graph-level readout via attention-weighted pooling.

#### Object Branch — GINE
Batched GINE (Graph Isomorphism Network with Edge features) processes the object graphs. Object class indices are embedded via a learned `nn.Embedding` (replacing the hash-based approach used in v1). Edge features encode spatial relationships.

#### Fusion — Quality-Aware Stream Attention
The three branch embeddings are fused by a masked learnable stream attention mechanism:
- Learned null embeddings are substituted for absent streams rather than zeroing them, preserving gradient flow.
- Stream quality scores `[Q_skel, Q_int, Q_obj]` are concatenated to the fusion MLP input, enabling the model to learn to downweight unreliable streams.
- Cross-stream attention is available as a configuration option but was not found to improve performance in ablation experiments and is disabled by default.

#### Temporal Encoder
A 2-layer vanilla Transformer encoder with learnable positional embeddings processes the per-frame fused embeddings. Attention pooling produces a single clip-level representation.

#### Classification Head
`Linear(D, D) → ReLU → Dropout → Linear(D, 1) → Sigmoid`

**Recommended configuration (Step 2 — best results):**

```python
fusion_dim          = 192
transformer_dim     = 192
transformer_heads   = 8
transformer_layers  = 2
transformer_ff_dim  = 384
dropout             = 0.25
skel_hidden_dims    = (64, 96, 192)
# Total parameters: ~4.7M
```

---

## Datasets

| Dataset | Videos | Clips (v2) | Positive Rate | Split Strategy | Status |
|---------|--------|-----------|---------------|----------------|--------|
| SCVD | ~900 | ~5,251 | ~0.57 | Video-level 70/15/15 | Complete |
| RLVS | — | 4,388 | 0.58 | Video-level 70/15/15 | v1 prep (v2 pending) |
| AIRTLab | — | 991 | 0.68 | Video-level 70/15/15 | v1 prep (v2 pending) |
| RWF-2000 | 2,000 | — | 0.50 | Official | v2 pending |
| NTU CCTV-Fights | 1,000 | 111,804 | ~0.37 | Official 500/250/250 | Complete |
| UBI-Fights | 1,000 | — | — | — | Scheduled |
| Hockey-Fights | 1,000 | — | 0.50 | — | Deprioritized (saturated benchmark) |

NTU CCTV-Fights uses **temporal overlap labeling**: clip label = 1 if ≥ 50% of the clip's frame duration overlaps with annotated fight segments from `groundtruth.json`. Approximately 36.6% of total NTU video duration contains fighting content.

---

## Experimental Results

All experiments below use the v2 pipeline on SCVD unless otherwise noted. Primary metrics: Macro-F1 (balanced) and ROC-AUC. Threshold is optimised on the validation set.

### SCVD v2 Experiments

| Configuration | Macro-F1 | ROC-AUC | Params | Notes |
|---------------|----------|---------|--------|-------|
| Full v2 (baseline) | 0.697 | 0.754 | 14.2M | Severe overfitting |
| Step 1 — no aug, no cross-attn, no QC | 0.707 | 0.754 | 14.2M | Severe overfitting |
| **Step 2 — reduced capacity** | **0.737** | **0.768** | **4.7M** | **Best F1** |
| Step 2 + full augmentation | 0.692 | 0.739 | 4.7M | Aug consistently hurts SCVD |
| Step 2 + jitter + WD 5e-4 | 0.728 | 0.762 | 4.7M | — |
| Step 2 + thinning + low LR + LS | 0.707 | 0.764 | 4.7M | — |
| Step 2 + quality conditioning | 0.682 | **0.773** | 4.7M | Best AUC; calibration issue lowers F1 |

**Key findings:**
- Capacity reduction (14.2M → 4.7M parameters) produced the single largest performance gain.
- Augmentation consistently degrades performance on SCVD due to pre-existing stream sparsity (71.4% of frames have at least one empty stream).
- Mid-to-high 70s Macro-F1 is a realistic ceiling for SCVD given data quality constraints, not a model limitation.
- Quality conditioning improves AUC but introduces calibration bias; treat as a calibration problem to address separately.

### NTU CCTV-Fights v2 Baseline (Step 2 Architecture)

| Metric | Value |
|--------|-------|
| Macro-F1 | 0.624 |
| Binary-F1 (violence class) | 0.525 |
| ROC-AUC | 0.682 |
| PR-AUC | 0.533 |
| Brier Score | 0.224 |
| ECE (calibration) | 0.112 |

Post-hoc analysis with video-level aggregation and temporal smoothing (window = 15 frames) improved Macro-F1 to **0.659** (+0.035) without retraining.

Stratified analysis confirmed that clip-level labeling granularity is the primary binding constraint: clips from the interior of fight segments score 0.633 while boundary clips (buildup/aftermath) score 0.520, a gap of 0.113 Macro-F1 points. The bimodal true-positive score distribution further corroborates this — positive clips with no visible fighting (labeled positive solely due to temporal proximity to a fight segment) are the dominant error mode.

---

## Tech Stack

| Component | Library / Tool |
|-----------|----------------|
| Deep learning framework | PyTorch |
| Graph neural networks | PyTorch Geometric (PyG) |
| Detection and pose estimation | Ultralytics YOLOv8 |
| Person tracking | ByteTrack |
| Data processing | pandas, NumPy |
| Visualization | matplotlib |
| Progress tracking | tqdm |
| Environment | Python 3.x, Windows, CUDA |

---

## Repository Structure

```
guardian-eye/
│
├── preprocessing_v2_fixed.py     # Stage 1: YOLO inference, graph export, GQS
├── dataprep_v2.py                # Stage 2: Clip construction, split assignment
├── tensorization_v2.py           # Stage 3: Tensor serialization
├── tri_graph_model_v2.py         # Model definition (MSA-STGCN + GATv2 + GINE)
├── train_scvd_v2.py              # Training script (SCVD)
├── ablation_study.py             # Branch and configuration ablation runner
├── class_imbalance_check.py      # Dataset statistics and imbalance diagnostics
│
└── outputs/
    ├── preprocessing/            # skeleton_graph/, interaction_graph/, object_graph/, meta.json per video
    ├── dataprep/                 # index_with_splits.csv per dataset
    ├── tensorized_fixed/         # Tensorized clips + index.csv per dataset
    └── training/                 # Checkpoints, metrics, predictions, plots per run
```

---

## Key Design Decisions

**Why three graph streams?**
Skeleton alone captures kinematics but misses weapon context. Objects alone miss intentionality. Interaction alone misses individual motion patterns. The combination is specifically motivated by surveillance footage where any single stream may be absent or noisy; the model learns to fuse available evidence.

**Why GQS is diagnostic-only?**
Using GQS as a training signal would create a circular dependency: the model would implicitly learn to trust its own detection confidence rather than the underlying visual content. GQS is reserved for post-hoc quality stratification and stream reliability analysis.

**Why no augmentation on SCVD?**
Horizontal flip breaks the spatial context learned from consistent camera orientations in SCVD. Stream dropout compounds an already-sparse stream availability (71.4% of frames have at least one empty stream). Augmentation is expected to be reassessed on higher-quality datasets such as NTU CCTV-Fights and UBI-Fights.

**Why temporal overlap labeling for NTU CCTV-Fights?**
NTU videos average approximately 2 minutes each with fight annotations provided as time-stamped segments. A simple video-level label would conflate fight and non-fight content within the same clip. Temporal overlap at a ≥ 50% threshold provides a principled clip-level label that trades off false positives (boundary clips) against false negatives (interior non-fight clips).

**Full tri-graph is always the primary model.**
Per-dataset ablation findings (e.g., skeleton + object outperforming full tri-graph on SCVD) are reported for analysis but do not override the full tri-graph as the primary evaluated architecture. This is an explicit controlled experiment decision to preserve research integrity across datasets.

---

## Citation

If this work is used in research, please cite:

```
@article{guardianEye2025,
  title   = {Guardian Eye: Quality-Aware Tri-Graph Neural Network for
             Violence Detection in CCTV Surveillance Footage},
  author  = {[Author]},
  year    = {2025}
}
```
