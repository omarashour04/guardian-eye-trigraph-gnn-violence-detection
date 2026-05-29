# Guardian Eye — Claude Code Context

## Wiki
Project wiki: `c:/Users/lenovo/Desktop/school/graduation project/gp vault/gp/`
To log results or look up context: open vault, read `wiki/_hot.md` first.
- Update `_hot.md` and `log.md` only at checkpoints: end of INGEST, end of LINT, or when user says "checkpoint" / "save" / "wrap up"
- Do NOT update `_hot.md` or `log.md` for QUERY-only sessions

## Active task
UBI-Fights V9 preprocessing is RUNNING on L4 (started 2026-05-23).
Expected ~4h. Next: VideoMAE fine-tune, then graph training.
Working scripts: `trigraph_v9_ubi_preprocess.py`, `trigraph_v9_ubi_videomae.py`, `trigraph_v9_ubi_train.py`.

## Project invariants
- Framework: PyTorch + HuggingFace Transformers + Modal cloud compute
- Dataset: UBI-Fights — 5,889 clips total (V9 tiled extraction).
  - Train: 1,295 pos / 2,957 neg (~2.3:1). Val: 203 pos / 744 neg. Test: 427 pos / 263 neg.
  - Official 67-video test split preserved video-level via test_videos.csv.
- Clip extraction: Approach B with sliding window — max_windows=3 per fight interval, min_dur=T//2=16 frames. neg_per_video=4.
- Val split: 20% of non-test videos, seeded (cfg.seed=42), carved in preprocessing.
- Class imbalance train: ~2.3:1 → pos_weight computed dynamically from actual label counts in videomae script.
- VideoMAE: pos_weight added to BCE, WeightedRandomSampler on train loader, mixup_alpha=0.4, cutmix_alpha=0.4.
- Output volume: ubi-fights-processed, mount /data/proc, cache at /data/proc/cache_v9.
- embed_dim is 768 (VideoMAE-Base CLS mean-pool over patch tokens). Do not change.
- NPZ schema locked. NPZ key frames_vit shape: [16, 224, 224, 3] uint8.
- Split CSV column: clip_id (NOT video_id — that is RWF). Never mix.

## Targets
- Macro-F1 ≥ 0.85
- ROC-AUC ≥ 0.93  (SOTA: CvlBiLT 98.56 — BiLT three-stage 2024)

## Environment
- Modal cloud GPU: L4 (preprocess/train), A10G (VideoMAE fine-tune)
- Python 3.10, PyTorch 2.2.2, transformers 4.40.2, timm 0.9.16

## Pre-run audit (MANDATORY)
Before telling the user any script is ready to run, you MUST perform a full audit:
1. Read every function that touches a CSV, NPZ, or volume path — verify column names, file names, and path constants match the actual files for THIS dataset (RWF uses `video_id`; UBI uses `clip_id`).
2. Check every tensor operation for shape correctness — especially multi-head attention, einsum dimensions, and reshape/view calls.
3. Verify all splits ("train"/"val"/"test") are handled — confirm no split is silently dropped (e.g. test embeddings missing from extraction loop).
4. Cross-check every formula that computes a ratio or normalisation — confirm the denominator is correct (e.g. divide by T not T×M).
5. Confirm Modal volume names, mount paths, and function decorators match this dataset's volume.
Only after completing all 5 checks may you tell the user the script is ready.
See [[sources/v9-bugs-fixed]] for the history of what these checks are designed to catch.

## Code style
- tqdm on every loop with unit label
- Inline comments on every non-obvious decision
- No emojis in print statements
- Return unified diff only unless told otherwise
