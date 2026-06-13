# Guardian Eye V9 — NTU CCTV-Fights

Tri-Graph GNN + VideoMAE violence detection adapted to **NTU CCTV-Fights**, a
**temporal-localization** dataset (long untrimmed videos, no separate non-fight videos).

## Dataset

| Property | Value |
|----------|-------|
| Total videos | 1,000 — all contain ≥1 annotated fight segment (no "normal" videos) |
| Official split | 500 train / 250 val / 250 test (fixed — do **not** reseed) |
| Sources | CCTV (280), Mobile (707), Other (9), Car (4) — mixed domains |
| Duration | min 3.2s / mean 63.7s / max 703.8s — long untrimmed |
| Annotations | `[start_sec, end_sec]` fight intervals; 661/1000 have multiple segments |
| Label field | `"Fight"` (single class in groundtruth) |
| Groundtruth | `groundtruth.json` — `database[video_id] = {duration, subset, nb_frames, frame_rate, source, annotations}` |
| Targets | Macro-F1 ≥ 0.85 · ROC-AUC ≥ 0.92 |

## The core design problem — windowing

NTU has no non-fight videos, so balanced clips are **synthesized** by sliding a window
over each video and labelling by overlap with annotated fight segments:

- **Window:** 5s = 150 frames @ 30fps → downsampled to `T=32` graph / `T_vit=16` VideoMAE.
  For non-30fps videos, window length is computed from per-video `frame_rate` so the span stays 5s.
- **Positive:** window overlaps ≥50% with a fight segment → label 1. Positive stride 2.5s (50% overlap).
- **Negative:** 0% overlap with any fight segment **and** ≥2s from any boundary → label 0. Negative stride 5s.
- **Ambiguous** (0 < overlap < 50%, or within the 2s guard gap) → discarded.
- **Per-video cap:** max 6 positives + 6 negatives (prevents long videos dominating / scene memorization).
- Negatives downsampled to match positive count per split; `pos_weight ≈ 1.0` computed dynamically.
- Each clip tagged with source `video_id`, `start_frame`, and source category for source-stratified reporting.

**Within-video hard negatives** (negatives from the same video as positives) are
intentional — a harder decision boundary than RLVS and a paper contribution.

## Pipeline

| Stage | Script | Notes |
|-------|--------|-------|
| 1. Preprocess | `trigraph_v9_ntu_preprocess.py` | Windowing front-end; `--plan-only` inspects the clip plan first |
| 2. VideoMAE | `trigraph_v9_ntu_videomae.py` | Adapted from RLVS Phase 2 |
| 3. Train | `trigraph_v9_ntu_train.py` | + source-stratified test metrics |
| 4. Diagnostics | `trigraph_v9_ntu_diagnostics.py` | Local; imports `V9Model` from train.py (mean-over-heads GATv2) |

```
python trigraph_v9_ntu_preprocess.py --plan-only
python trigraph_v9_ntu_preprocess.py
python trigraph_v9_ntu_videomae.py
python trigraph_v9_ntu_train.py
python trigraph_v9_ntu_diagnostics.py
```

`inspect_errors.py` and `smoke_videomae.py` are local helpers; `run_pipeline.py` orchestrates the run.

## Project invariants (shared across all V9 datasets)

- `embed_dim = 768` · NPZ `frames_vit [16, 224, 224, 3]` uint8 · `T=32, T_vit=16, M=6, N=8, V=17`.
- VideoMAE: `WeightedRandomSampler`, `mixup_alpha=0.4`, `cutmix_alpha=0.4`.

## Environment

Python 3.10 · PyTorch 2.2.2 · transformers 4.40.2 · timm 0.9.16 · ultralytics (YOLO11x / YOLO11x-Pose).
