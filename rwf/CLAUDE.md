# Guardian Eye — Claude Code Context

## Wiki
Project wiki: `c:/Users/lenovo/Desktop/school/graduation project/gp vault/gp/`
To log results or look up context: open vault, read `wiki/_hot.md` first.
- Update `_hot.md` and `log.md` only at checkpoints: end of INGEST, end of LINT, or when user says "checkpoint" / "save" / "wrap up"
- Do NOT update `_hot.md` or `log.md` for QUERY-only sessions

## Active task
RWF-2000 V9 is COMPLETE. Next: UBI-Fights training.
Working files: trigraph_v9_rwf_train.py, trigraph_v9_rwf_diagnostics.py.
Do not touch old/ or preprocess/videomae scripts unless instructed.

## Project invariants
- Framework: PyTorch + HuggingFace Transformers + Modal cloud compute
- Target file: trigraph_v9_rwf_videomae.py
- Output volume path: /data/proc (Modal Volume rwf-2000-processed)
- Cache directory: /data/proc/cache_v9/
- Split CSV: /data/proc/split_v9.csv
- Dataset: RWF-2000 — 2,000 videos total (1,600 train, 400 val). No separate test split.
- embed_dim is 768 (VideoMAE-Base CLS output after mean-pool). Do not change to 384.
- Never rewrite preprocessing (trigraph_v9_rwf_preprocess.py). NPZ schema is locked.
- NPZ key frames_vit shape: [16, 224, 224, 3] uint8

## Environment
- Modal cloud GPU: A10G
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