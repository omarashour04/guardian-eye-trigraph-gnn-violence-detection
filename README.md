# Guardian Eye — Tri-Graph GNN Violence Detection (V9)

> **Binary violence detection for CCTV surveillance footage using a tri-stream graph neural network with VideoMAE-based clip embeddings and Quality-Guided Fusion.**

Guardian Eye is a graduation research project in AI and Data Science. It classifies surveillance video clips as violent or non-violent by jointly reasoning over three complementary graph streams — skeleton motion, person–person interaction, and person–object context — gated by a learned quality-aware fusion module (QGF). The V9 pipeline adds VideoMAE-Base clip embeddings as a fourth signal and runs entirely on Modal cloud GPU.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [V9 Pipeline: Stage by Stage](#v9-pipeline-stage-by-stage)
- [Experimental Results](#experimental-results)
  - [RWF-2000](#rwf-2000-results)
  - [UBI-Fights](#ubi-fights-results)
- [Key Findings & Lessons Learned](#key-findings--lessons-learned)
- [Repository Structure](#repository-structure)
- [Environment & Dependencies](#environment--dependencies)
- [Running the Pipeline](#running-the-pipeline)
- [Targets vs SOTA](#targets-vs-sota)
- [Citation](#citation)

---

## Project Overview

Violence detection in CCTV footage is a safety-critical task. Guardian Eye addresses it through a multi-modal, quality-aware representation that encodes:

- **Who is present and how they are moving** — skeleton stream (17-joint COCO poses via YOLO11x-Pose)
- **How people interact spatially** — interaction stream (person-to-person proximity graph)
- **What objects are present** — object stream (person-to-object context graph)
- **Global clip semantics** — 768-d VideoMAE-Base CLS embedding fine-tuned on the target dataset

The three graph streams are fused by a Quality-Guided Fusion (QGF) gate that weights each stream by its per-frame quality score (GQS). The fused representation is concatenated with the VideoMAE embedding and passed to a binary classifier.

The pipeline degrades gracefully under real-world CCTV conditions (low resolution, occlusion, missing detections) through learned null stream embeddings and per-frame quality scoring.

---

## Architecture

```
Raw video clip (16 frames, 224×224)
         │
         ├─── YOLO11x-Pose → Skeleton graph (17 joints × M persons)
         │                      │
         ├─── YOLO11x      → Interaction graph (M persons, K-NN edges)
         │                      │
         └─── YOLO11x      → Object graph (N objects, person–object edges)
                                │
                    GQS quality scores [Q_skel, Q_int, Q_obj]
                                │
              ┌─────────────────▼──────────────────┐
              │   Tri-Graph GNN (GATv2 + GINE)      │
              │   Quality-Guided Fusion (QGF gate)  │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   VideoMAE-Base (fine-tuned)         │
              │   768-d CLS embedding               │
              └─────────────────┬──────────────────┘
                                │
                    Concat → MLP → Binary output
```

### Graph Quality Score (GQS)

A per-frame composite quality score in `[0, 1]` computed from five sub-scores:

| Component | Description |
|-----------|-------------|
| `q_skel`  | Fraction of COCO joints detected with confidence ≥ threshold, divided by T |
| `q_int`   | Fraction of real (non-KNN-fallback) interaction edges |
| `q_obj`   | Fraction of frames with detected objects × mean confidence |
| `q_flow`  | Temporal skeleton stability (cosine similarity of consecutive skeletons) |
| `q_scene` | Scene-level quality from object detection confidence |

GQS feeds the QGF gate directly — the gate learns to upweight streams that are reliable for a given clip.

> **Key invariant:** `q_skel = valid_joints / T` (divided by T, *not* T×M). This formula is locked; changing it breaks the GQS scale.

### Quality-Guided Fusion (QGF)

Learned attention gate that takes the three stream embeddings and their quality scores as input and outputs a quality-weighted fused embedding. Contrast with `lw` (fixed learned weights, quality-blind) and `cgf` (content-only gate, no quality signal).

---

## Datasets

### RWF-2000

| Property | Value |
|----------|-------|
| Total clips | 2,000 videos (10-second each) |
| Split | Official 70/15/15 (train/val/test) — balanced 50/50 |
| Prior | Uniform — train, val, test all 50% positive |
| Status | **COMPLETE** — all targets met |

### UBI-Fights

| Property | Value |
|----------|-------|
| Source | University of Beira Interior |
| Total clips (V9 extraction) | 5,889 |
| Train | 1,295 positive / 2,957 negative (~2.3:1) |
| Val | 203 positive / 744 negative (21.4% positive) |
| Test | 427 positive / 263 negative (**61.9% positive — inverted vs train**) |
| Test split | Official 67-video split, video-level |
| Extraction method | Approach B — annotation-guided sliding window |
| Status | **Re-run in progress** (quality-shortcut source fix) |

UBI-Fights uses **Approach B**: positive clips are carved around annotated fight intervals (max 3 windows per interval, min duration T//2=16 frames). Negative clips are sampled from explicitly labelled non-fight intervals (`neg_per_video=4`).

---

## V9 Pipeline: Stage by Stage

### Stage 1 — Preprocessing (`trigraph_v9_*_preprocess.py`)

Runs on Modal **L4 GPU** (~3-4 hours for UBI-Fights).

1. Downloads dataset from Kaggle (via `KAGGLE_TOKEN` secret) if not already on volume.
2. Runs YOLO11x and YOLO11x-Pose on every frame to extract skeleton, interaction, and object graphs.
3. Computes GQS features (`q_skel`, `q_int`, `q_obj`, `q_flow`, `q_scene`) per frame.
4. Saves clips as NPZ files: `frames_vit [16, 224, 224, 3] uint8` + `gqs [5] float32`.
5. Writes `split_*.csv` (train/val/test assignments) and `gqs_summary_*.csv`.

**V9-specific (UBI source-fix):** negative candidates are oversampled (`neg_oversample=2.0`), each candidate is scored by `q_skel` after YOLO, and only the `neg_per_video` candidates per video whose `q_skel` is closest to the global fight median are kept. This kills the "low-quality negative" correlation that caused the quality shortcut.

### Stage 2A — VideoMAE Fine-Tune (`trigraph_v9_*_videomae.py`)

Runs on Modal **A10G GPU**.

1. Fine-tunes VideoMAE-Base (`MCG-NJU/videomae-base`) on the extracted clips.
2. Uses `pos_weight = n_neg/n_pos` in BCE loss + `WeightedRandomSampler` for class balance.
3. Mixup (`alpha=0.4`) + CutMix (`alpha=0.4`) augmentation.
4. Quality × label balanced sampler (UBI source-fix): 3-tier `q_skel` bins × label, inverse cell frequency weights — breaks any residual quality-label correlation at the batch level.
5. Fresh re-finetune guard: deletes stale checkpoint before re-run so shortcut-trained weights are never resumed.
6. Best checkpoint saved by val AUC; `videomae_test_results.json` written after Stage B.

**embed_dim = 768** (VideoMAE-Base CLS mean-pool). Do not change.

### Stage 2B — Embedding Extraction

Runs immediately after Stage A in the same script. Iterates all train/val/test NPZs, encodes each clip through the fine-tuned VideoMAE encoder, writes `vit_embedding [768] float32` back into each NPZ. NPZ schema is locked.

### Stage 3 — Graph Training (`trigraph_v9_*_train.py`)

Runs on Modal **L4 GPU** (~1-2 hours).

Trains the tri-graph GNN + QGF gate on top of the frozen VideoMAE embeddings. Eight experiment variants per dataset:

| Experiment | Fusion | Description |
|-----------|--------|-------------|
| `A_skeleton_only` | skeleton | Skeleton stream only, no QGF |
| `B_skel_interaction` | skel+int | Two streams |
| `D_skel_int_obj` | all graphs, lw | Full tri-graph, fixed-weight fusion |
| `E_full_lw` | lw | Full + VideoMAE, fixed weights |
| `E_full_qgf` | qgf | Full + VideoMAE + quality gate |
| `E_full_qgf_fixed` | qgf | QGF with fixed quality input |
| `E_full_cgf` | cgf | Content gate (no quality signal) |
| `E_full_qgf_adv` | qgf_adv | QGF + adversarial quality decorrelation |

`WeightedRandomSampler` + prior-matched threshold + dynamic `pos_weight` are applied in all experiments.

### Stage 4 — Diagnostics (`trigraph_v9_*_diagnostics.py`)

Offline analysis run after training:

- **`embedding_shortcut`** — linear probe on raw `vit_embedding` → label; confirms whether the quality shortcut has been fixed (val→test gap should close after source fix).
- **`shortcut_audit`** — quality-alone AUC per split; confirms `q_skel` no longer predicts the label.
- **`gqs_stratified`** — per-GQS-quartile F1 and AUC; confirms the high-quality bucket no longer collapses.
- **`error_analysis`** — per-clip FP/FN breakdown.
- **`calibration`** — temperature scaling and ECE.

---

## Experimental Results

### RWF-2000 Results

RWF-2000 is balanced (50/50 prior, same distribution across all splits) — no prior shift, no quality shortcut. All targets met.

| Experiment | Val F1 | Test Macro-F1 | Test AUC | Notes |
|-----------|--------|--------------|---------|-------|
| A_skeleton_only | — | 0.8912 | 0.9543 | Skeleton alone strong |
| B_skel_interaction | — | 0.9023 | 0.9601 | +0.005 F1 from interaction |
| D_skel_int_obj | — | 0.9087 | 0.9654 | +0.009 F1 from object stream |
| E_full_lw | — | 0.9134 | 0.9701 | Full model, fixed weights |
| **E_full_qgf** | — | **0.9211** | **0.9743** | **Best — QGF adds +0.008 F1** |
| E_full_qgf_fixed | — | 0.9187 | 0.9722 | — |

**Targets: Macro-F1 ≥ 0.85 ✅ · ROC-AUC ≥ 0.93 ✅**

Key diagnostics:
- GQS stratification: QGF advantage concentrated in mid-to-high quality bucket. Net aggregate ≈ VideoMAE standalone on low-quality clips.
- Temperature scaling: skipped — T*=0.910 sharpened an already-calibrated model (ECE would increase from 0.041 to 0.048).
- 33 irreducible errors: FNs are occluded/crowd fights; FPs are aggressive non-fights (sports, rough play).

### UBI-Fights Results

UBI-Fights has two compounding distribution problems:

**Problem 1 — Inverted prior:** train 30.5% positive vs test 61.9% positive. Every threshold and `pos_weight` calibrated to the wrong distribution.

**Problem 2 — Non-transferable quality shortcut:** train negatives have `q_skel ≈ 0.67` while train positives have `q_skel ≈ 0.91`. The model learns "clean skeleton → fight". On the test set, negatives are also clean (`q_skel ≈ 0.78`) and the shortcut breaks.

Both problems were confirmed by a linear probe on the raw 768-d VideoMAE embedding (no graph, no gate):

| Probe | Train AUC | Val AUC | Test AUC |
|-------|-----------|---------|---------|
| embedding → label | 0.984 | 0.916 | **0.681** |
| embedding → q_skel | 0.986 | 0.897 | 0.871 |

The embedding alone reproduces the full-model val→test collapse, proving the shortcut is frozen **upstream** of the entire graph head.

#### V2 Re-run (head-side fixes — did not help)

Fixed `WeightedRandomSampler`, prior-matched `pos_weight`, threshold search, 2 new gate ablations:

| Experiment | Val F1 | Test Macro-F1 | Test AUC |
|-----------|--------|--------------|---------|
| E_full_qgf | 0.9176 | 0.6725 | **0.7241** |
| E_full_cgf | 0.9176 | 0.6743 | 0.7106 |
| E_full_lw | 0.9207 | 0.6697 | 0.7036 |
| E_full_qgf_adv | 0.9204 | 0.6730 | 0.7025 |

Best V2 AUC 0.7241 — val still ~0.92, test ~0.70. **Head-side fixes did not help because the shortcut is upstream.**

Option 2 (INLP linear decorrelation — strip up to 8 linear quality directions from embeddings): test AUC moved +0.001. Shortcut is non-linear; post-hoc linear removal is insufficient.

#### Source Fix (current, in progress)

**`exp_source_fix_negquality/`** — quality-stratified negative resampling:
- Oversample negative candidates (`neg_oversample=2.0`), score each by `q_skel` after YOLO
- Keep the `neg_per_video` candidates per video whose `q_skel` is closest to the global fight `q_skel` median
- Delete dropped candidate NPZs so only quality-matched negatives survive
- Re-fine-tune VideoMAE from scratch on balanced data (fresh-finetune guard prevents resuming shortcut-trained checkpoint)
- Quality × label balanced sampler in VideoMAE as insurance

**Status: code complete, Phase 1 run pending.**

---

## Key Findings & Lessons Learned

### 1. val→test gap = distribution shift, not plateau
A 21-point val→test AUC gap (0.92 vs 0.71) cannot be a training plateau — a plateau gives matching val/test. The gap means the train/val distribution differs from test. Always compute val AND test AUC during training; a plateau gives low values everywhere, not a gap.

### 2. Head-side fixes cannot cure an embedding-level shortcut
Sampler, `pos_weight`, threshold, content gate, adversarial gate — all are downstream of the frozen VideoMAE embeddings. If a linear probe on the raw embedding reproduces the model's failure, the fix must be upstream (data) not downstream (head).

### 3. GRL adversarial decorrelation is insufficient for non-linear shortcuts
Stripping 8 linear quality directions from the embedding (INLP) moved test AUC +0.001. The shortcut is encoded non-linearly; removing the linear projection of `q_skel` does not remove it.

### 4. Run `embedding_shortcut` diagnostic before paying for graph training
A single linear probe on the raw `vit_embedding` confirms or rules out a quality shortcut in under 5 minutes. This should be standard after every VideoMAE fine-tune, before running the 3-hour graph training.

### 5. Negative clip quality must match positive clip quality
In any dataset with annotation-guided positive extraction, annotated fight clips tend to be well-framed (high skeleton quality) while random non-fight intervals are not. If negatives are sampled randomly, the quality gap becomes a free shortcut. Oversample candidates, score them, and select to match the positive quality distribution.

### 6. RWF succeeds because it is balanced; UBI fails because it is not
The GNN architecture and QGF gate are byte-identical across RWF and UBI. RWF reaches AUC 0.97 because train/val/test share the same prior and quality distribution. UBI's failure is entirely a data-condition problem, not an architecture problem.

---

## Repository Structure

```
fresh/
├── README.md
├── .gitignore
├── patch_gqs_q_skel.py          # One-off NPZ patch (q_skel formula fix)
│
├── rwf/                         # RWF-2000 pipeline (COMPLETE)
│   ├── trigraph_v9_rwf_preprocess.py
│   ├── trigraph_v9_rwf_videomae.py
│   ├── trigraph_v9_rwf_train.py
│   ├── trigraph_v9_rwf_diagnostics.py
│   ├── training_output/         # Per-experiment metrics, train history CSVs
│   ├── diagnostics_output/      # Calibration, error analysis, gate weights, GQS
│   ├── perfecto/                # Final clean-run scripts + RESULTS.md
│   └── old/                     # Archived prior-version scripts
│
└── ubi/                         # UBI-Fights pipeline (source-fix in progress)
    ├── exp_source_fix_negquality/   # ACTIVE — quality-stratified neg resampling fix
    │   ├── trigraph_v9_ubi_preprocess.py
    │   ├── trigraph_v9_ubi_videomae.py
    │   ├── trigraph_v9_ubi_train.py
    │   ├── trigraph_v9_ubi_diagnostics.py
    │   └── README.md
    └── old/                     # Archived scripts + vol_results (V1/V2 outputs)
        └── vol_results/
            ├── preprocess/      # split_ubi.csv, gqs_summary_ubi.csv (V1 run)
            ├── training_output/ # V1 and V2 experiment results
            └── exp_linear_decorrelation/  # Ruled-out Option 2 (INLP)
```

---

## Environment & Dependencies

```
Python      3.10
PyTorch     2.2.2
CUDA        12.1
transformers 4.40.2
timm        0.9.16
ultralytics (YOLO11x, YOLO11x-Pose)
modal       (cloud GPU orchestration)
```

### Modal Volumes

| Volume | Contents |
|--------|----------|
| `ubi-fights-raw` | Raw UBI-Fights Kaggle zip |
| `ubi-fights-processed` | NPZs, split CSVs, GQS CSVs, checkpoints |
| `rwf-processed` | RWF NPZs, split CSVs, checkpoints |

### Modal Secrets

| Secret | Keys |
|--------|------|
| `KAGGLE_TOKEN` | `KAGGLE_USERNAME`, `KAGGLE_KEY` |

---

## Running the Pipeline

### New Modal account setup

```bash
# 1. Authenticate Modal
modal setup

# 2. Create volumes
modal volume create ubi-fights-raw
modal volume create ubi-fights-processed

# 3. Set Kaggle secret (enter your username and API key when prompted)
modal secret create KAGGLE_TOKEN KAGGLE_USERNAME=<your_username> KAGGLE_KEY=<your_key>
```

### UBI-Fights source-fix run (3 phases)

```bash
# Phase 1 — Preprocessing with quality-stratified negative selection (~3-4h, L4)
# Downloads UBI-Fights from Kaggle automatically if volume is empty
conda run -n dlcuda128 modal run exp_source_fix_negquality/trigraph_v9_ubi_preprocess.py::preprocess

# Validation gate: inspect printed GQS table
# train neg q_skel should rise from ~0.67 toward ~0.80

# Phase 2 — VideoMAE fine-tune + embedding extraction (~3-4h, A10G)
conda run -n dlcuda128 modal run exp_source_fix_negquality/trigraph_v9_ubi_videomae.py::run_videomae

# Validation gate: run embedding_shortcut diagnostic
# val->test linear-probe gap should close from 0.916/0.681
conda run -n dlcuda128 modal run exp_source_fix_negquality/trigraph_v9_ubi_diagnostics.py::download_embedding_shortcut

# Phase 3 — Graph training (~1-2h, L4)
conda run -n dlcuda128 modal run exp_source_fix_negquality/trigraph_v9_ubi_train.py::train
```

### RWF-2000 (reference — already complete)

```bash
conda run -n dlcuda128 modal run rwf/trigraph_v9_rwf_preprocess.py::preprocess
conda run -n dlcuda128 modal run rwf/trigraph_v9_rwf_videomae.py::run_videomae
conda run -n dlcuda128 modal run rwf/trigraph_v9_rwf_train.py::train
```

### Detached runs (recommended to avoid local client drops)

```bash
modal run --detach exp_source_fix_negquality/trigraph_v9_ubi_train.py::train
```

---

## Targets vs SOTA

| Dataset | Metric | Target | Our Best (current) | SOTA |
|---------|--------|--------|--------------------|------|
| RWF-2000 | Macro-F1 | ≥ 0.85 | **0.9211 ✅** | ~0.99 (survey) |
| RWF-2000 | ROC-AUC | ≥ 0.93 | **0.9743 ✅** | ~0.99 |
| UBI-Fights | Macro-F1 | ≥ 0.85 | 0.6746 ⚠️ (V2, pre-fix) | — |
| UBI-Fights | ROC-AUC | ≥ 0.93 | 0.7241 ⚠️ (V2, pre-fix) | **98.56** (CvlBiLT 2024) |

SOTA for UBI-Fights: CvlBiLT (ConvNeXt-Large + BiLSTM + Transformer, 2024). Uses the same official 67-video test split.

---

## Citation

```bibtex
@article{guardianEye2025,
  title   = {Guardian Eye: Quality-Aware Tri-Graph GNN for Violence Detection
             in CCTV Surveillance Footage},
  author  = {Ashour, Omar},
  year    = {2025}
}
```
