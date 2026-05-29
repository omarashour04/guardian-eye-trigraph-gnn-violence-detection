# Guardian Eye V9 — RWF-2000

## Quick reference

| What you want | Where to look |
|---|---|
| Run training | `modal run trigraph_v9_rwf_train.py --detach` |
| Run diagnostics | `modal run trigraph_v9_rwf_diagnostics.py` |
| Latest results summary | `training_output/experiment_comparison_v9.csv` |
| Best model checkpoint | `training_output/E_full_qgf/best.pt` |
| Calibration / gate weights / error analysis | `diagnostics_output/` |
| Full analysis in wiki | `gp vault/gp/wiki/analyses/v9-final-results-2026-05-23.md` |

---

## Folder layout

```
rwf/
│
├── trigraph_v9_rwf_train.py        ACTIVE — GATv2 fixed (2026-05-23)
├── trigraph_v9_rwf_diagnostics.py  ACTIVE — GATv2 fixed (2026-05-23)
├── trigraph_v9_rwf_preprocess.py   ACTIVE — do not touch (preprocessing done)
├── trigraph_v9_rwf_videomae.py     ACTIVE — do not touch (Phase 2 done)
│
├── training_output/                CURRENT RUN (GATv2-fixed, 2026-05-23)
│   ├── experiment_comparison_v9.csv
│   ├── A_skeleton_only/
│   ├── B_skel_interaction/
│   ├── D_skel_int_obj/
│   ├── E_full_lw/
│   ├── E_full_qgf/                 ← BEST MODEL (F1=0.9175, AUC=0.9617)
│   └── E_full_qgf_fixed/
│
├── diagnostics_output/             CURRENT DIAGNOSTICS (2026-05-23)
│   ├── calibration_results.json    T*=0.910, ECE worsens → skip T scaling
│   ├── gate_weights_E_full_qgf.csv vit=28.2% dominant
│   ├── gqs_stratified_comparison.csv
│   ├── error_analysis.csv
│   └── inference_speed.json        ~90ms / 500 clips/sec on T4
│
├── perfecto/                       BACKUP — pre-GATv2-fix run (keep; ablation baseline)
│   ├── RESULTS.md                  old results (B: 0.815, D: 0.834, E: 0.920 cal)
│   ├── trigraph_v9_rwf_train.py    old script (buggy GATv2 single-head)
│   └── training_output/
│
└── old/                            ARCHIVED older scripts (pre-V9, ignore)
```

---

## Key numbers (current run)

| Metric | Value |
|---|---|
| Best model | E_full_qgf |
| Macro-F1 | **0.9175** |
| ROC-AUC | **0.9617** |
| Decision threshold | **0.36** |
| ECE (raw) | 0.041 |
| T* | 0.910 (skip — worsens ECE) |
| Params | 798,618 |
| Latency | ~90ms / ~500 clips/sec (T4) |

---

## What changed vs perfecto/ backup

The `perfecto/` run used a buggy GATv2 where `a.mean(-1)` collapsed 4 attention heads to 1 before weighting values. Fixed with `torch.einsum('bimh,bmhd->bihd', a, V_heads)`. Impact:
- B_skel_interaction: 0.815 → **0.820** (+0.005 F1)
- D_skel_int_obj: 0.834 → **0.843** (+0.009 F1)
- E_full_qgf: unchanged (VideoMAE dominates; graph fix doesn't reach the full model)

Gate weight ordering also shifted: dominant stream is now **vit (28.2%)** vs perfecto's obj (30.2%). Both are valid learned outcomes — within noise given the dataset.

See `wiki/sources/v9-bugs-fixed.md` for the full bug record.
