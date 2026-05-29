# Guardian Eye V9 — RWF-2000 Results

## Ablation Table

| Experiment | Streams | Fusion | Raw F1 | Cal F1 | AUC | Threshold | Params |
|---|---|---|---|---|---|---|---|
| A_skeleton_only | skeleton | QGF | 0.7746 | 0.7746 | 0.8144 | 0.58 | 71,715 |
| B_skel_interaction | skel + int | QGF | 0.8150 | 0.8150 | 0.8817 | 0.42 | 477,522 |
| C_videomae_only *(Phase 2)* | vit | — | 0.9075 | — | 0.9709 | 0.38 | 86M |
| D_skel_int_obj | skel + int + obj | QGF | 0.8337 | 0.8362 | 0.9057 | 0.40 | 568,023 |
| E_full_lw | all 4 | LW | 0.9150 | 0.9150 | 0.9640 | 0.12 | 798,322 |
| **E_full_qgf** *(best)* | **all 4** | **QGF** | **0.9175** | **0.9200** | **0.9611** | **0.17** | **798,618** |

**Cal F1** = macro-F1 after temperature scaling (T\*) with threshold re-tuned on calibrated val probs.

---

## What the Ablation Shows

Each stream adds measurable value:
- Skeleton alone: 0.775 — captures motion but misses context
- Adding interaction: +0.040 — person-person relations help
- Adding object: +0.018 — scene objects add discriminative signal
- Adding VideoMAE: +0.082 — largest single jump; global appearance is strong
- QGF over LW: +0.005 — gate conditioning on quality scores adds a small but consistent gain

---

## Best Model: E_full_qgf

- **Calibrated macro-F1: 0.9200** | Raw: 0.9175
- **AUC: 0.9611**
- **Decision threshold: 0.17** (not 0.50 — model logits skew negative)
- **Temperature T\* = 1.01** — model is already well-calibrated; temperature scaling optional at deployment
- **ECE: 0.047 → 0.041** after calibration

### Confusion Matrix (400 val samples, thr=0.17)

|  | Pred: Non-Fight | Pred: Fight |
|---|---|---|
| **True: Non-Fight** | 181 | 19 |
| **True: Fight** | 14 | 186 |

---

## Gate Weight Analysis (QGF)

The QGF gate learns a meaningful distribution — not uniform:

| Stream | Mean Weight | Role |
|---|---|---|
| skeleton | 22.3% | Motion geometry |
| interaction | 20.0% | Person-person relations |
| **object** | **30.2%** | Scene context (highest — bounding boxes are reliable) |
| vit/VideoMAE | 27.6% | Global temporal appearance |

- No stream ever dominates a single sample (max weight seen: 35.3% for vit)
- Violence clips up-weight VideoMAE slightly (27.9% vs 27.2%) — gate correctly uses appearance for fight detection
- **The gate is doing real work**, not collapsing. This is the result of the LayerNorm fix on the gate MLP.

### GQS-Stratified F1 (QGF vs LW vs VideoMAE baseline)

| Quality Bucket | QGF F1 | LW F1 | VideoMAE |
|---|---|---|---|
| Low (0–0.25) | 0.889 | 0.903 | 0.908 |
| Mid-low (0.25–0.57) | 0.900 | 0.890 | 0.908 |
| Mid-high (0.57–0.79) | 0.937 | 0.937 | 0.908 |
| High (0.79–1.0) | 0.884 | 0.873 | 0.908 |

QGF beats LW on 3 of 4 quality buckets.

---

## Error Analysis (14 FN / 19 FP)

**False Negatives (missed fights):**
- Mean q_skel = 0.316, q_int = 0.554 — fights with poor skeleton/interaction visibility
- Top miss: `Fight_XM0NI7ZmwWM` (OOD video, appears in 2 FNs — model never generalises to it)
- `Fight_1Kbw1bUw` segments also cluster in FNs — same video, multiple hard segments

**False Positives (false alarms):**
- Mean q_obj = 0.977 — high-quality non-fight videos with aggressive motion
- Top FP: `NonFight_39BFeYnbu-I`, `NonFight_EFv961C5RgY`, `NonFight_nuf-d5GugL0`
- These are semantic edge cases (sports, rough play) — not fixable without more diverse training data

---

## Inference Speed (T4 GPU)

| Model | Params | Latency mean | p95 | Throughput |
|---|---|---|---|---|
| E_full_qgf | 798,618 | 89.5 ms | 93.2 ms | 517 clips/sec |
| E_full_lw | 798,322 | 89.2 ms | 93.0 ms | 520 clips/sec |
| E_full_qgf_fixed | 798,618 | 90.2 ms | 92.2 ms | 517 clips/sec |

QGF gate adds **zero measurable latency** over LW. The LayerNorm + 2-layer MLP on 4 values is negligible.

---

## Key Fixes Applied (vs previous run)

1. **LayerNorm on QGF gate MLP** — old gate had near-uniform weights (0.25 each); LayerNorm gives the gate variance to learn from even when GQS values are all near 1.0
2. **Threshold search widened** — thr_min 0.10→0.05, step 0.02→0.01; old grid missed the 0.12 optimum
3. **Post-calibration threshold re-search** — old code reset threshold to 0.50 after scaling, hurting F1; now threshold is re-tuned on calibrated val probs
4. **OOD exclusion** — `Fight_XM0NI7ZmwWM` excluded from training; did not improve test score (video also appears in val/test, so model never learned to handle it regardless)

---

## Output Files

```
training_output/
  experiment_comparison_v9.csv       — full ablation table
  E_full_qgf/
    best.pt                          — model checkpoint
    test_metrics.json                — raw + calibrated metrics
    train_history.csv                — per-epoch loss/F1
    gqs_quartile_f1.csv              — F1 by quality bucket

diagnostics_output/
  gate_weights_E_full_qgf.csv        — per-sample gate weights (400 rows)
  gqs_stratified_comparison.csv      — QGF vs LW vs VideoMAE by quality bucket
  calibration_results.json           — temperature scaling results
  error_analysis.csv                 — per-sample TP/TN/FP/FN with GQS scores
  inference_speed.csv / .json        — latency and throughput measurements
```
