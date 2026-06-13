# Guardian Eye — Claude Code Context (NTU CCTV-Fights)

## Wiki
Project wiki: `c:/Users/lenovo/Desktop/school/graduation project/gp vault/gp/`
To log results or look up context: open vault, read `wiki/_hot.md` first.
- Update `_hot.md` and `log.md` only at checkpoints: end of INGEST, end of LINT, or when user says "checkpoint" / "save" / "wrap up"
- Do NOT update `_hot.md` or `log.md` for QUERY-only sessions

## Active task
NTU CCTV-Fights V9 — SCRIPTS WRITTEN + AUDITED (2026-06-07). READY TO RUN on WS.
All four scripts written and passed the mandatory 5-point pre-run audit + py_compile:
  - trigraph_v9_ntu_preprocess.py  (NEW windowing front-end; per-clip core identical to RLVS)
  - trigraph_v9_ntu_videomae.py    (adapted from RLVS Phase 2)
  - trigraph_v9_ntu_train.py       (adapted from RLVS Phase 3; + source-stratified test metrics)
  - trigraph_v9_ntu_diagnostics.py (LOCAL, no Modal; imports V9Model from train.py)
Run order on WS (paths already set to C:\Violence detection\...\fresh\ntu\):
  1. python trigraph_v9_ntu_preprocess.py --plan-only   # inspect clip plan first
  2. python trigraph_v9_ntu_preprocess.py               # full YOLO extraction
  3. python trigraph_v9_ntu_videomae.py
  4. python trigraph_v9_ntu_train.py
  5. python trigraph_v9_ntu_diagnostics.py

## Confirmed data layout (WS, 2026-06-07)
- Path: C:\Violence detection\datasets\NTU-CCTV fights\
- Flat folder: fight_0001.mpeg ... fight_1000.mpeg (+ groundtruth.json). No subfolders.
- Filenames match groundtruth keys exactly (stem = fight_XXXX). Format .mpeg.

## Diagnostics file caveat (IMPORTANT)
The RLVS *pipeline* diagnostics file (RLVS/trigraph_v9_rlvs_diagnostics.py) is the OLD
UBI+Modal version (import modal, clip_id, runs_ubi, and a DIFFERENT GATv2 = per-head
einsum). It does NOT match the RLVS-trained checkpoints. The REAL RLVS diagnostics the
user ran are the LOCAL leakage scripts in RLVS/scripts/ (inspect_outputs, build_clean_split,
eval_clean_split, reeval_leakage), which IMPORT the V9 modules rather than redefining the
net. NTU diagnostics follows that pattern: it imports V9Model from trigraph_v9_ntu_train,
so the architecture matches by construction (mean-over-heads GATv2). Do NOT seed NTU
diagnostics from the stale root UBI file.

## Dataset facts (from groundtruth.json)
- **1,000 fight videos total** — all keys are `fight_XXXX`, no non-fight videos.
- **Official split (fixed — do NOT reseed):**
  - training: 500 videos
  - validation: 250 videos
  - testing: 250 videos
- **Sources:** CCTV (280), Mobile (707), Other (9), Car (4) — mixed domains.
- **Duration:** min 3.2s / max 703.8s / mean 63.7s per video — long untrimmed videos.
- **Annotations:** Each video has ≥1 fight segment as `[start_sec, end_sec]` intervals.
  - 661 / 1000 videos have multiple fight segments per video.
  - ALL 1000 videos contain at least one fight annotation (no "normal" videos in GT).
- **Label field:** `"Fight"` (capital F) — only one class label in annotations.
- **Frame rate:** mostly 30.0 fps (check per-video, stored in groundtruth.json).
- **Groundtruth file:** `ntu/groundtruth.json` — schema: `{version, database}` where
  `database[video_id] = {duration, subset, nb_frames, frame_rate, source, annotations}`.

## The core design problem
NTU has NO non-fight videos. This is a **temporal localization** dataset:
- Fight = frames within annotated `[start_sec, end_sec]` segments.
- Non-fight (hard negatives) = frames from the SAME videos, OUTSIDE those segments.
- You must synthesize balanced positive/negative clips by windowing each video.

### Locked conversion plan (ALL params locked 2026-06-07)
Extract fixed-length clips using a sliding window. Window = 5s = 150 frames @ 30fps,
downsampled to T=32 graph / T_vit=16 VideoMAE (identical to RLVS). For videos not at
30fps, compute window length in frames from the per-video frame_rate so the temporal
span stays 5s.
- **Positive clips:** window overlaps ≥50% with a fight segment → label = 1.
  Positive stride = 2.5s (50% overlap) — fights are scarce/fragmented, harvest densely.
- **Negative clips:** window has 0% overlap with any fight segment AND is ≥2s (guard
  gap) from any fight boundary → label = 0. Negative stride = 5s (no overlap) —
  negatives are abundant, avoid flooding the downsampler.
- Ambiguous windows (partial overlap, 0 < overlap < 50%, OR within the 2s guard gap)
  → discard.
- **Per-video cap:** max 6 positives + 6 negatives per video. Flattens the 703s video
  so a handful of long videos can't dominate (kills scene-memorization confound).
- Balance: downsample negatives to match positive count PER SPLIT. pos_weight ≈ 1.0,
  computed dynamically from actual counts.
- Tag each clip with source video_id, start_frame, AND source category (CCTV/Mobile/
  Other/Car) for traceability and source-stratified reporting.
- Respect the official train/val/test split from groundtruth.json — do NOT mix splits.

### Within-video hard negatives
Negatives come from the SAME video as positives → high visual similarity, harder
decision boundary. This is intentional and a paper contribution (harder evaluation
than RLVS which has fully separate normal videos).

## Project invariants (inherited from V9, identical to RLVS/RWF/UBI)
- Framework: PyTorch + HuggingFace Transformers + Modal cloud compute
- embed_dim is 768 (VideoMAE-Base CLS mean-pool over patch tokens). Do not change.
- NPZ schema locked (V9). frames_vit shape: [16, 224, 224, 3] uint8.
- Graph constants: T=32, T_vit=16, M=6, N=8, V=17.
- pos_weight: compute dynamically from actual label counts after windowing.
- VideoMAE: pos_weight in BCE, WeightedRandomSampler on train loader,
  mixup_alpha=0.4, cutmix_alpha=0.4.

## Environment
- Modal cloud GPU: L4 (preprocess/train), A10G (VideoMAE fine-tune)
- Python 3.10, PyTorch 2.2.2, transformers 4.40.2, timm 0.9.16
- Kaggle slug: TBD (confirm dataset download source before writing preprocess script)

## Targets
- Macro-F1 ≥ 0.85
- ROC-AUC ≥ 0.92

## TODO before first run
- Write `trigraph_v9_ntu_preprocess.py` from scratch using windowing plan above.
- Window params LOCKED (see conversion plan): 5s window, pos stride 2.5s, neg stride 5s,
  ≥50% overlap threshold, 2s guard gap, 6+6 clips/video cap.
- Write `trigraph_v9_ntu_videomae.py`, `trigraph_v9_ntu_train.py`, `trigraph_v9_ntu_diagnostics.py`.
- Define Modal volume name for NTU (e.g. `ntu-fights-processed`).
- Run the MANDATORY pre-run audit (below) before declaring ready.

## Pre-run audit (MANDATORY)
Before telling the user any script is ready to run, you MUST perform a full audit:
1. Read every function that touches groundtruth.json, NPZ, or volume path — verify
   segment overlap logic, clip window boundaries, and label assignment are correct.
2. Check every tensor operation for shape correctness — especially multi-head
   attention, einsum dimensions, and reshape/view calls.
3. Verify all splits ("train"/"val"/"test") are handled — confirm no split is
   silently dropped from either windowing or embedding extraction.
4. Cross-check every formula that computes a ratio or normalisation — confirm
   the denominator is correct (e.g. divide by T not T×M).
5. Confirm Modal volume names, mount paths, and function decorators reference
   the NTU volume, NOT rlvs or ubi volumes.
Only after completing all 5 checks may you tell the user the script is ready.

## Code style
- tqdm on every loop with unit label
- Inline comments on every non-obvious decision
- No emojis in print statements
- Return unified diff only unless told otherwise

## Citation
NTU CCTV-Fights dataset: Require proper citation in paper (TBD — confirm paper ref).
