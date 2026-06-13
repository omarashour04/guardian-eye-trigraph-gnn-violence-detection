# RLVS V9 — Test-Set Leakage & Data-Quality Analysis

**Date:** 2026-06-06
**Run:** VideoMAE fine-tune, best checkpoint epoch 46 (`videomae_best.pt`),
val-tuned threshold 0.44.
**Decision:** Report the result with this analysis as a documented caveat
(no re-fine-tune performed). Rationale: leakage was shown to inflate the score
by < 0.002 macro-F1, below the noise floor of a 200-clip test set.

---

## 1. Headline result (random seed=42 split)

| Metric    | Test value |
|-----------|-----------|
| Accuracy  | 0.9950 |
| Macro-F1  | 0.9950 |
| ROC-AUC   | 0.9998 |

Targets (Macro-F1 ≥ 0.85, ROC-AUC ≥ 0.92) are met. The score sits **above the
~0.92–0.95 appearance-model ceiling** commonly reported on RLVS, which prompted
the integrity investigation below.

## 2. Split integrity (exact-ID)

No `clip_id` is shared across train/val/test; every ID is unique; class balance
is exact (train 800/800, val 100/100, test 100/100). No exact-duplicate leakage.

## 3. Near-duplicate leakage (DCT perceptual-hash)

RLVS re-cuts the same real-life scene into multiple consecutively-numbered clips.
A random split scatters these cuts across train/val/test, leaking scene identity.

**Method.** 8 evenly-spaced frames per clip → 64-bit DCT perceptual hash (DC term
dropped for brightness invariance). Two clips are near-duplicates if ≥ 3 frame
pairs match within a Hamming distance of 8 bits. Opposite-label pairs are
suppressed (a violence clip cannot be a re-cut of a non-violence clip). An earlier
8×8 average-hash attempt over-flagged massively (dark/static clips collapsed to
near-identical hashes, matching fights with handshakes) and was discarded.

**Findings.**
- 483 same-label near-duplicate clip pairs; 130 multi-clip scene clusters
  (largest: 18 clips, NV_683–NV_706 — one scene cut into 18 "clips").
- All duplicate clusters are NonViolence. The Violence class shows no scene re-cuts.
- Under the random split, **36 of 200 test clips (18%) had a near-duplicate twin
  in train/val** (8 of them near-identical re-encodes). All 36 are NonViolence.

## 4. Did leakage inflate the score?

The model was re-evaluated on the test set with the 36 leaked clips separated out
(same checkpoint, same threshold, per-clip probabilities recorded):

| Subset                         | n   | Accuracy | Macro-F1 |
|--------------------------------|-----|----------|----------|
| FULL test                      | 200 | 0.9950   | 0.9950   |
| CLEAN (no train/val twin)      | 164 | 0.9939   | 0.9936   |
| LEAKED only                    | 36  | 1.0000   | 1.0000   |

**Removing all leaked clips lowers Macro-F1 by only 0.0014.** Leakage is *not* the
driver of the high score; the model separates the clean test clips essentially as
well as the full set. (The CLEAN subset is class-imbalanced at 99/65 because every
removed clip was NonViolence, so its absolute value is indicative, not definitive.)

## 5. Group-aware clean split (built, not yet trained on)

A leakage-free split was generated (`split_rlvs_clean.csv`): scene clusters were
formed by union-find over the duplicate pairs and assigned **whole** to
train/val/test (largest-first greedy, seed=42). Result: train 1600 (800/800),
val 200 (100/100), test 200 (100/100), and **0 cross-split duplicate pairs** —
verified. A quick eval of the current model on this split was inconclusive (the
contamination-free subset degenerated to 11 single-class clips), confirming that a
definitive clean number would require a full re-fine-tune on the clean train set.
That re-train was deferred given the < 0.002 leakage effect measured in §4.

## 6. Open concern: motion-presence shortcut (NOT leakage)

Independent of leakage, a graph-quality asymmetry exists: of 111 clips with
graph `valid_ratio < 0.5`, **99 are NonViolence** (and the worst have zero detected
skeletons/flow — static scenes: handshakes, eating, walking). "Absence of
skeleton/motion activity" is therefore itself a strong predictor of the
NonViolence class. Part of the model's near-perfect negative recall may reflect a
motion-presence shortcut rather than violence semantics. This is a property of the
RLVS dataset, not a bug, but it should be stated as a limitation: the reported
score may overstate genuine violence-vs-non-violence understanding.

## 7. Recommended statement for the paper

> We audited the RLVS test split for scene-level leakage using DCT perceptual
> hashing and found that a naïve random split placed 36/200 test clips with a
> near-duplicate scene twin in train/val (all NonViolence). Re-evaluating with
> these clips removed changed Macro-F1 by < 0.002, indicating the result is not
> driven by leakage. We additionally provide a group-aware split that eliminates
> all cross-split scene duplicates. We note that the RLVS NonViolence class is
> dominated by low-motion scenes, so appearance/graph models may partly exploit
> motion presence; reported scores should be read with this in mind.

## 8. Artifacts produced

| File | Contents |
|------|----------|
| `inspect_outputs.py` | 5-audit integrity inspector (split, adjacency, quality, results, pHash) |
| `inspection_report_v2.txt` | full audit log (hardened pHash) |
| `preproc_output/phash_cross_split_pairs.csv` | all cross-split duplicate pairs |
| `preproc_output/phash_test_leakage_pairs.csv` | test↔train/val duplicate pairs |
| `preproc_output/low_quality_clips.csv` | 111 clips with valid_ratio < 0.5 |
| `reeval_leakage.py` + `reeval_leakage_report.txt` | FULL/CLEAN/LEAKED re-eval |
| `finetune_output/test_predictions.csv` | per-clip test probabilities |
| `build_clean_split.py` + `split_rlvs_clean.csv` | group-aware leakage-free split |
| `eval_clean_split.py` + `eval_clean_split_report.txt` | quick eval on clean split |
