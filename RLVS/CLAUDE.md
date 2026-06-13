# Guardian Eye — Claude Code Context (RLVS)

## Wiki
Project wiki: `c:/Users/lenovo/Desktop/school/graduation project/gp vault/gp/`
To log results or look up context: open vault, read `wiki/_hot.md` first.
- Update `_hot.md` and `log.md` only at checkpoints: end of INGEST, end of LINT, or when user says "checkpoint" / "save" / "wrap up"
- Do NOT update `_hot.md` or `log.md` for QUERY-only sessions

## Active task
RLVS V9 re-run — SETUP IN PROGRESS (started 2026-06-05).
Code was seeded from the clean `ubi/old/*.py` V9 scripts and renamed ubi -> rlvs.
The scripts are NOT yet adapted to RLVS — they still contain UBI ingestion logic
(frame-level annotation parsing, official 67-video test split, kaggle slug, volume
names). DO NOT RUN until the preprocessing rework below is done.
Working scripts (renamed copies, pending rework): `trigraph_v9_rlvs_preprocess.py`,
`trigraph_v9_rlvs_videomae.py`, `trigraph_v9_rlvs_train.py`, `trigraph_v9_rlvs_diagnostics.py`.

## TODO before first run (pending user data details)
- Rework `preprocess` ingestion for RLVS: clip-level folder structure
  (1,000 fight / 1,000 non-fight), NOT frame-level interval parsing.
- Define seeded 80/10/10 split (seed=42) — RLVS has NO fixed official split.
- Update kaggle_slug, Modal volume names, mount paths, CSV/cache names.
- Decide split CSV id column name for RLVS (UBI=clip_id, RWF=video_id).
- Note: a prior V1 RLVS ablation exists but on the OLD (pre-V9) architecture —
  its NPZs are almost certainly NOT V9 schema. Re-preprocess to V9 schema.
- Run the MANDATORY pre-run audit (below) before declaring ready.

## Project invariants
- Framework: PyTorch + HuggingFace Transformers + Modal cloud compute
- Dataset: RLVS — Real Life Violence Situations.
  - 2,000 clips total: 1,000 fight (violence) / 1,000 non-fight (1:1 balanced).
  - Domain: street / real-life. Violence = bare-hand + weapon fights.
    Non-violence = handshakes, conversations, eating, walking, horse riding.
  - No fixed official split; 80/10/10 seeded (seed=42) is the plan.
  - Kaggle: mohamedmustafa/real-life-violence-situations-dataset
  - Citation: Soliman et al., IJCI 2019.
  - Known issue: CUE-Net (2024) flagged mislabelled samples in the RLVS test
    split — state evaluation protocol + any detected anomalies in the paper.
- embed_dim is 768 (VideoMAE-Base CLS mean-pool over patch tokens). Do not change.
- NPZ schema locked (V9, identical to RWF/UBI). frames_vit shape: [16, 224, 224, 3] uint8.
- Graph constants: T=32, T_vit=16, M=6, N=8, V=17.
- Class balance is ~1:1 — pos_weight should be ~1.0, but still compute it
  dynamically from actual label counts (do not hardcode).
- VideoMAE: pos_weight in BCE, WeightedRandomSampler on train loader,
  mixup_alpha=0.4, cutmix_alpha=0.4.

## Prior result (context, OLD architecture)
- Guardian Eye V1 (pre-V9): Macro-F1 ~0.819, ROC-AUC ~0.884.
- RLVS is NOT saturated. Graph/skeleton SOTA ~82–88% F1; appearance models 92–95% acc.

## Targets
- Macro-F1 ≥ 0.85
- ROC-AUC ≥ 0.92

## Environment
- Modal cloud GPU: L4 (preprocess/train), A10G (VideoMAE fine-tune)
- Python 3.10, PyTorch 2.2.2, transformers 4.40.2, timm 0.9.16

## Pre-run audit (MANDATORY)
Before telling the user any script is ready to run, you MUST perform a full audit:
1. Read every function that touches a CSV, NPZ, or volume path — verify column names, file names, and path constants match the actual files for THIS dataset (RWF uses `video_id`; UBI uses `clip_id`; RLVS column TBD).
2. Check every tensor operation for shape correctness — especially multi-head attention, einsum dimensions, and reshape/view calls.
3. Verify all splits ("train"/"val"/"test") are handled — confirm no split is silently dropped (e.g. test embeddings missing from extraction loop).
4. Cross-check every formula that computes a ratio or normalisation — confirm the denominator is correct (e.g. divide by T not T×M).
5. Confirm Modal volume names, mount paths, and function decorators match this dataset's volume (RLVS volume, NOT ubi-fights-processed).
Only after completing all 5 checks may you tell the user the script is ready.
See [[sources/v9-bugs-fixed]] for the history of what these checks are designed to catch.

## Code style
- tqdm on every loop with unit label
- Inline comments on every non-obvious decision
- No emojis in print statements
- Return unified diff only unless told otherwise