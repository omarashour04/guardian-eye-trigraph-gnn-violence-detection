# Guardian Eye V9 — RLVS (Real Life Violence Situations)

Tri-Graph GNN + VideoMAE violence detection pipeline adapted to the **RLVS** dataset.

## Dataset

| Property | Value |
|----------|-------|
| Name | Real Life Violence Situations (RLVS) |
| Total clips | 2,000 (1,000 violence / 1,000 non-violence — 1:1 balanced) |
| Domain | Street / real-life; bare-hand + weapon fights vs. handshakes, conversations, walking, etc. |
| Split | 80/10/10 seeded stratified (seed=42) — RLVS has **no** fixed official split |
| Kaggle | `mohamedmustafa/real-life-violence-situations-dataset` |
| Citation | Soliman et al., IJCI 2019 |
| Targets | Macro-F1 ≥ 0.85 · ROC-AUC ≥ 0.92 |

> **Note:** CUE-Net (2024) flagged mislabelled samples in the RLVS test split. State the
> evaluation protocol and any detected anomalies when reporting.

## Pipeline

| Stage | Script |
|-------|--------|
| 1. Preprocess (YOLO11x graphs + GQS, NPZ export, seeded split) | `trigraph_v9_rlvs_preprocess.py` |
| 2. VideoMAE fine-tune + embedding extraction | `trigraph_v9_rlvs_videomae.py` |
| 3. Tri-graph GNN + QGF training | `trigraph_v9_rlvs_train.py` |
| 4. Diagnostics | `trigraph_v9_rlvs_diagnostics.py` |

`scripts/` contains the local leakage-analysis tooling that imports the V9 modules
directly (rather than redefining the network), so the architecture matches the trained
checkpoints by construction:

- `build_clean_split.py` — rebuild a leakage-free split
- `eval_clean_split.py` — evaluate on the clean split
- `reeval_leakage.py` — re-score for train/test leakage
- `inspect_outputs.py` — inspect model outputs per clip

`run_pipeline.py` is the local orchestrator. See `docs/` for the explanation-RAG system,
demo app, and leakage-analysis write-ups.

## Project invariants (shared across all V9 datasets)

- `embed_dim = 768` (VideoMAE-Base CLS mean-pool). Locked.
- NPZ schema locked: `frames_vit` shape `[16, 224, 224, 3]` uint8.
- Graph constants: `T=32, T_vit=16, M=6, N=8, V=17`.
- `pos_weight` computed dynamically from label counts (≈1.0, balanced).
- GATv2: mean-over-heads (not per-head einsum).

## Environment

Python 3.10 · PyTorch 2.2.2 · transformers 4.40.2 · timm 0.9.16 · ultralytics (YOLO11x / YOLO11x-Pose).
