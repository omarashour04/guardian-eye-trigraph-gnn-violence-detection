# Guardian Eye — Tri-Graph GNN Violence Detection

PyTorch + HuggingFace Transformers + Modal cloud pipeline for violence detection using a tri-graph GNN over skeleton, interaction, and object streams, gated by a Quality-Guided Fusion (QGF) module.

## Datasets

| Dataset | Status | Best Test AUC |
|---------|--------|--------------|
| RWF-2000 | Complete | ~0.97 |
| UBI-Fights | Re-run in progress (source-fix) | 0.7241 (V2, shortcut fix pending) |

## Structure

```
rwf/          — RWF-2000 pipeline (preprocessing, VideoMAE, training, diagnostics)
ubi/          — UBI-Fights pipeline
  exp_source_fix_negquality/   — quality-stratified negative resampling fix (active)
  exp_linear_decorrelation/    — ruled-out INLP decorrelation experiment
```

## Pipeline

1. **Preprocess** — extract clips (YOLO11x skeleton/object detection, Approach B annotation-guided), compute GQS features, build split CSVs.
2. **VideoMAE fine-tune** — fine-tune VideoMAE-Base on extracted clips (Stage A), extract 768-d CLS embeddings (Stage B).
3. **Graph training** — tri-graph GNN + QGF gate, WeightedRandomSampler + prior-matched threshold.

## Environment

- Python 3.10, PyTorch 2.2.2, transformers 4.40.2, timm 0.9.16
- Modal cloud GPU: L4 (preprocess/train), A10G (VideoMAE fine-tune)
- Volumes: `ubi-fights-raw`, `ubi-fights-processed`, `rwf-processed`
