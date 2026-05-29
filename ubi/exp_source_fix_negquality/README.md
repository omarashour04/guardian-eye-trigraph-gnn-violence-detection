# Source Fix — Quality-Stratified Negative Selection (UBI-Fights V9)

Isolated experiment that fixes the **non-transferable quality shortcut** at its
true origin: the negative-clip sampler. The original pipeline picked random
non-fight windows with no quality consideration, so train negatives ended up
systematically lower skeleton-quality (q_skel 0.67) than the well-framed fight
clips (0.91). VideoMAE learned "clean skeleton -> fight"; on the official test
split the negatives are also clean (0.78), the shortcut breaks, and test AUC
collapses to ~0.68 while val stays ~0.92.

We proved (linear probe + INLP) the shortcut is frozen non-linearly into the
VideoMAE embeddings — head-side and post-hoc fixes all failed. This fix removes
the correlation in the data before VideoMAE ever sees it.

Full background: `gp vault/gp/wiki/analyses/v9-ubi-results-2026-05-29.md` and
`analyses/problems-tackled-ubi-rwf.md`.

## What changed vs the parent scripts

The four scripts here are copies of `fresh/ubi/*.py`. Only the data-generation
and fine-tune entry behavior changed:

### `trigraph_v9_ubi_preprocess.py`
- **CFG knobs added:** `neg_oversample=2.0` (propose 2x candidates per kept
  negative), `quality_match="fight_quantile"` (keep candidates whose q_skel is
  closest to the fight median, destroying the clean->fight gap; alt `"high"`).
- **`negative_clip_indices()`** now proposes an oversampled candidate pool
  (`n_candidates`) using the SAME random-window logic; selection happens later.
- **Per-clip loop** runs YOLO on every candidate and records its q_skel
  (already computed) into `gqs_rows`, tagged with `src_stem` + `is_neg_cand`.
- **Step 6 (post-loop, CPU, free):** per source video, keep `neg_per_video`
  candidates matched to the fight q_skel distribution; **delete the dropped
  candidates' NPZs** (guarded so a kept clip can never be deleted).
- **Step 6b:** rewrite the authoritative `split_ubi.csv` and
  `gqs_summary_ubi.csv` from kept clips only (helper columns dropped — schema
  stays compatible with downstream scripts).
- **Cost knob:** GPU `A10G -> L4` (YOLO11x fits in 24 GB; cheaper/hr).
- **Clean slate:** `force_reextract` defaults **True**; purges old NPZs first so
  no stale window from the buggy run survives under a colliding clip_id.

### `trigraph_v9_ubi_videomae.py`
- **`run_videomae` defaults:** `skip_finetune=False` (re-train from scratch on
  the new clips), `fresh_finetune=True` (delete the prior shortcut-trained
  `videomae_best.pt` before Stage A; a preemption-retry within THIS run still
  resumes via a run-marker file).
- **Quality x label balanced sampler** (insurance): the WeightedRandomSampler now
  equalizes (label x q_skel-tier) cells in addition to label balance, reading
  q_skel from `gqs_summary_ubi.csv`. Stacks on the data fix; no extra cost.
- Stage B (embedding extraction) unchanged — re-extracts `vit_embedding` into
  every NPZ with the new encoder.

### `trigraph_v9_ubi_train.py`, `trigraph_v9_ubi_diagnostics.py`
- Unchanged copies, kept here so the whole experiment runs from one folder.

## ⚠ Same-volume warning

These scripts write the **same Modal volume** (`ubi-fights-processed`,
`/data/proc/cache_v9`) as the original pipeline. Running the preprocess here
**overwrites the existing NPZs, `split_ubi.csv`, and `gqs_summary_ubi.csv`** —
i.e. it regenerates the dataset from scratch (intended). The original *code* is
preserved in the parent `fresh/ubi/` folder for reference; the original *data*
on the volume is replaced.

## Run order (from this folder)

```sh
# Phase 1 — re-preprocess with quality-matched negatives (L4, ~$3-4)
conda run -n dlcuda128 modal run exp_source_fix_negquality/trigraph_v9_ubi_preprocess.py::preprocess

# Phase 2 — re-fine-tune VideoMAE + re-extract embeddings (A10G, ~$3-4)
conda run -n dlcuda128 modal run exp_source_fix_negquality/trigraph_v9_ubi_videomae.py::run_videomae

# Phase 3 — re-run graph training (L4, cheap)
conda run -n dlcuda128 modal run exp_source_fix_negquality/trigraph_v9_ubi_train.py::train
```

## Verification gates (cheapest-first)

1. **After Phase 1** — inspect the new `gqs_summary_ubi.csv`: train negative mean
   q_skel should rise from ~0.67 toward the fight level (~0.78-0.91); the
   train pos-vs-neg q_skel contrast should shrink. The preprocess prints a
   `split x label` GQS-means table at the end for exactly this check.
2. **After Phase 2** — run the embedding-shortcut diagnostic BEFORE spending
   Phase 3 credits:
   ```sh
   conda run -n dlcuda128 modal run exp_source_fix_negquality/trigraph_v9_ubi_diagnostics.py::download_embedding_shortcut
   ```
   The linear label-probe val->test AUC gap should shrink from 0.916/0.681
   toward parity. If it does, the shortcut is gone; proceed to Phase 3.
3. **After Phase 3** — best test AUC materially above 0.72; high-GQS test bucket
   no longer collapsing.

## Cost notes

- Phase 1 on L4 + bounded `neg_oversample` (the only added cost is YOLO on
  dropped candidates) dominates (~$3-4).
- Phase 2 VideoMAE fine-tune + Stage B (~$3-4) is unavoidable in any fix.
- Phase 3 is cheap (L4). Run the Phase-2 gate before paying for Phase 3.
- If the q_skel gap doesn't shrink enough after Phase 1, raise `neg_oversample`
  (more candidates to match from) and re-run Phase 1.
