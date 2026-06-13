# Guardian Eye — Claude Code Context (Hockey Fights)

## Wiki
Project wiki: `c:/Users/lenovo/Desktop/school/graduation project/gp vault/gp/`
To log results or look up context: open vault, read `wiki/_hot.md` first.
- Update `_hot.md` and `log.md` only at checkpoints: end of INGEST, end of LINT, or when user says "checkpoint" / "save" / "wrap up"
- Do NOT update `_hot.md` or `log.md` for QUERY-only sessions

## Active task
Hockey Fights V9 — SCRIPTS TO BE WRITTEN. Adapt the NTU/RLVS/RWF V9 stack to Hockey
Fights. The preprocess/videomae/train scripts should be adapted from BOTH the RLVS and
RWF V9 scripts (RLVS for the local-WS, no-Modal, clip-level pipeline structure; RWF for
any dataset-specific preprocessing patterns — Hockey is clip-level like both).
This is the 4th dataset in the Q1 portfolio (RWF ✅, RLVS ✅, NTU ✅, HF = active).
V8.1 canonical result: F1=0.930, AUC=0.9642. V9 target: match or exceed.
Run order on WS (paths set to C:\Violence detection\...\fresh\hf\):
  1. python trigraph_v9_hf_preprocess.py
  2. python trigraph_v9_hf_videomae.py
  3. python trigraph_v9_hf_train.py
  4. python trigraph_v9_hf_diagnostics.py

## Confirmed data layout (WS — confirm before writing preprocess)
- Dataset: Hockey Fights (Bermejo Nievas et al., CAIP 2011)
- Expected layout: two folders `fights/` and `nofights/` containing short .avi clips
- Total: 1,000 clips (500 fight / 500 non-fight) — balanced, clip-level labels
- Split: 70/10/20 stratified by label, seed=42 → 700 train / 150 val / 150 test
- **CRITICAL label bug:** `nofights` contains the substring `fights` — always check
  `nofights` FIRST in any string/folder test or you will silently flip labels.
- Confirm exact WS path before writing scripts (likely
  `C:\Violence detection\datasets\HockeyFights\` or similar).

## V8.1 results (for comparison — architecture was NOT V9)
- Best: E_full_trigraph — Acc=0.930, Macro-F1=0.930, AUC=0.9642
- D_skel_obj_po: F1=0.930, AUC=0.9658
- C_skel_int: F1=0.905, AUC=0.9587
- A_skeleton_only: F1=0.868, AUC=0.9495
- Note: V8.1 used a different architecture (no VideoMAE, no QGF LayerNorm fix,
  no mean-over-heads GATv2). V9 results will NOT be directly comparable — cite V9 only.
- Known: PO stream increased FPs on Hockey in V8.1 (PO stream hurts here); object
  GQS mean ~0.111 (near-empty signal on most clips). Watch in V9 diagnostics.

## Dataset facts
- Domain: ice hockey broadcast footage — highly homogeneous, single-scene, two-team
  uniforms. Zero generalisation off-domain. Use as ablation/classic benchmark only.
- Clip length: short (typically 2–4s). All clips pre-trimmed, clip-level labels.
- No temporal localisation needed (unlike NTU) — same pipeline as RLVS/RWF.
- No within-video hard negatives needed — separate fight / nofight clips.
- Balanced dataset (500/500) — pos_weight expected ~1.0, no class-imbalance handling needed.

## Project invariants (inherited from V9, identical to RLVS/NTU/RWF)
- Framework: PyTorch + HuggingFace Transformers (local WS, no Modal)
- embed_dim = 768 (VideoMAE-Base CLS mean-pool). Do not change.
- NPZ schema locked (V9). frames_vit shape: [16, 224, 224, 3] uint8.
- Graph constants: T=32, T_vit=16, M=6, N=8, V=17.
- pos_weight: compute dynamically from actual label counts (expect ~1.0, balanced).
- VideoMAE: WeightedRandomSampler on train loader, mixup_alpha=0.4, cutmix_alpha=0.4.
- id column in split CSV: `clip_id` (same as NTU/RLVS, NOT video_id like RWF).
- GATv2: mean-over-heads (NOT per-head einsum — that was the stale UBI bug).

## Adaptation notes (RLVS + RWF → HF)
- Folder-scan preprocessing: adapt from RLVS (local WS pattern, seeded stratified split,
  clip_id column). RWF also uses folder scan with Violence/NonViolence folders — reference
  both for the label-reading pattern, but Hockey folder names are `fights/` and `nofights/`
  so the label logic must be rewritten (see label bug above).
- VideoMAE and train scripts: near-identical to RLVS — change dataset name, paths, and
  split CSV reference. No windowing, no source tag, no temporal localisation.
- Diagnostics: adapt from NTU local diagnostics (no Modal, imports V9Model from train.py).
  Keep gate_weights, gqs_stratified, calibration, error_analysis. Drop source_shortcut
  (no source field in HF). Add note about PO stream FP pattern from V8.1.

## Targets
- Macro-F1 ≥ 0.93 (match V8.1), ROC-AUC ≥ 0.96
- Dataset is saturated so matching or small gain is fine — consistency matters.

## Pre-run audit (MANDATORY)
Before telling the user any script is ready to run, you MUST perform a full audit:
1. Read every function that touches the folder scan, CSV, or NPZ — verify label
   assignment logic checks `nofights` BEFORE `fights` (the label bug).
2. Check every tensor operation for shape correctness — multi-head attention,
   einsum dimensions, reshape/view calls.
3. Verify all splits ("train"/"val"/"test") are handled — confirm no split is
   silently dropped from preprocessing or Stage B embedding extraction.
4. Cross-check every formula that computes a ratio or normalisation — confirm
   the denominator is correct (e.g. q_skel divides by T not T×M).
5. Confirm output paths reference HF, NOT rlvs/ntu/ubi/rwf.
Only after completing all 5 checks may you tell the user the script is ready.

## Code style
- tqdm on every loop with unit label
- Inline comments on every non-obvious decision
- No emojis in print statements
- Return unified diff only unless told otherwise

## Citation
Hockey Fights dataset: Bermejo Nievas et al., "Violence Detection in Video using
Computer Vision Techniques", CAIP 2011.
