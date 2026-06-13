# Guardian Eye V9 — Hockey Fights

Tri-Graph GNN + VideoMAE violence detection adapted to the **Hockey Fights** dataset.
Local-workstation pipeline (no Modal), clip-level labels — structured like RLVS/RWF.

## Dataset

| Property | Value |
|----------|-------|
| Name | Hockey Fights (Bermejo Nievas et al., CAIP 2011) |
| Total clips | 1,000 (500 fight / 500 non-fight — balanced) |
| Layout | Two folders `fights/` and `nofights/` of short `.avi` clips (typically 2–4s) |
| Split | 70/10/20 stratified by label, seed=42 → 700 train / 150 val / 150 test |
| Domain | Ice-hockey broadcast — homogeneous, single-scene; classic benchmark / ablation only |
| Targets | Macro-F1 ≥ 0.93 · ROC-AUC ≥ 0.96 |

> **⚠️ Label bug:** `nofights` contains the substring `fights`. Any string/folder test
> must check `nofights` **first**, or labels silently flip.

## Pipeline

| Stage | Script |
|-------|--------|
| 1. Preprocess (folder scan + seeded stratified split + YOLO graphs/GQS) | `trigraph_v9_hf_preprocess.py` |
| 2. VideoMAE fine-tune + embedding extraction | `trigraph_v9_hf_videomae.py` |
| 3. Tri-graph GNN + QGF training | `trigraph_v9_hf_train.py` |
| 4. Diagnostics | `trigraph_v9_hf_diagnostics.py` |

```
python trigraph_v9_hf_preprocess.py
python trigraph_v9_hf_videomae.py
python trigraph_v9_hf_train.py
python trigraph_v9_hf_diagnostics.py
```

`run_pipeline.py` orchestrates the run. `v8/trigraph_v8_1_hockey_fight.py` is the prior
V8.1 reference implementation (different architecture — kept for comparison only).

## V8.1 reference results (different architecture — not V9-comparable)

| Variant | Acc / Macro-F1 | AUC |
|---------|----------------|-----|
| E_full_trigraph (best) | 0.930 | 0.9642 |
| D_skel_obj_po | 0.930 | 0.9658 |
| A_skeleton_only | 0.868 | 0.9495 |

V8.1 had no VideoMAE, no QGF LayerNorm fix, and a different GATv2 — cite V9 numbers only.
Known V8.1 finding: the PO (person-object) stream **increased** false positives on Hockey
(object GQS mean ≈ 0.111, near-empty). Watch this in V9 diagnostics.

## Project invariants (shared across all V9 datasets)

- `embed_dim = 768` · NPZ `frames_vit [16, 224, 224, 3]` uint8 · `T=32, T_vit=16, M=6, N=8, V=17`.
- `id` column in split CSV: `clip_id`.
- `pos_weight` computed dynamically (≈1.0, balanced).
- GATv2: mean-over-heads. VideoMAE: `WeightedRandomSampler`, `mixup_alpha=0.4`, `cutmix_alpha=0.4`.

## Environment

Python 3.10 · PyTorch 2.2.2 · transformers 4.40.2 · timm 0.9.16 · ultralytics (YOLO11x / YOLO11x-Pose).
