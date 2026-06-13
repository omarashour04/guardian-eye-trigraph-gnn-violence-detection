"""
trigraph_v9_hf_diagnostics.py
Guardian Eye — V9  |  Post-hoc Diagnostics (LOCAL, inference only)
Dataset : Hockey Fights (Bermejo Nievas et al., CAIP 2011)

Design note
-----------
LOCAL script (no Modal). Imports model + dataset + cfg directly from
trigraph_v9_hf_train so the diagnostic V9Model is guaranteed identical to
the checkpoints train.py produced (mean-over-heads GATv2).

Four diagnostics written to OUTPUT_DIR/runs_hf/diagnostics/:

  1. gate_weights   — per-sample QGF gate weights on E_full_qgf + static
                      weights from E_full_lw. Watch w_obj on Hockey Fights —
                      V8.1 showed PO stream increased FPs here (object GQS
                      mean ~0.111; near-empty signal on most ice-rink clips).
  2. gqs_stratified — E_full_qgf vs E_full_lw per GQS-composite quartile.
  3. calibration    — temperature scaling on E_full_qgf; pre/post ECE.
  4. error_analysis — TP/TN/FP/FN breakdown with per-clip GQS.

NOTE: source_shortcut diagnostic is NOT included for Hockey Fights.
      HF has no source field (all clips are ice-hockey broadcast footage,
      single-scene, two-team). Source-shortcut analysis is not applicable.

Usage
-----
  python trigraph_v9_hf_diagnostics.py            # run all 4
  python trigraph_v9_hf_diagnostics.py --only gate_weights
  python trigraph_v9_hf_diagnostics.py --only calibration error_analysis
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score)
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Import the trained-model definition + dataset + cfg from the training module.
# Guarantees the diagnostic V9Model is identical to the checkpoints.
TRAIN = importlib.import_module("trigraph_v9_hf_train")

cfg        = TRAIN.cfg
CACHE_DIR  = TRAIN.CACHE_DIR
SPLIT_CSV  = TRAIN.SPLIT_CSV
GQS_CSV    = TRAIN.GQS_CSV
RUN_DIR    = TRAIN.RUN_DIR
DIAG_DIR   = RUN_DIR / "diagnostics"

HFDataset       = TRAIN.HFDataset
V9Model         = TRAIN.V9Model
collate_v9      = TRAIN.collate_v9
seed_everything = TRAIN.seed_everything

# (active_streams, fusion_mode, extra_kwargs) per experiment — mirrors train.py
EXP_CFG = {
    "E_full_qgf":       (("skeleton", "interaction", "object", "vit"), "qgf", {}),
    "E_full_lw":        (("skeleton", "interaction", "object", "vit"), "lw",  {}),
    "E_full_qgf_fixed": (("skeleton", "interaction", "object", "vit"), "qgf",
                         {"entropy_weight": 0.10, "temp_anneal": True}),
}


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_split(split_name: str, split_df):
    """Return (npz_paths, labels) for the named split. Uses clip_id column."""
    sub = split_df[split_df["split"] == split_name]
    paths  = [CACHE_DIR / f"{c}.npz" for c in sub["clip_id"]]
    labels = sub["label"].tolist()
    existing = [(p, l) for p, l in zip(paths, labels) if p.exists()]
    if not existing:
        raise RuntimeError(f"No NPZ files found for split '{split_name}'. "
                           "Run Phase 1 (preprocess) first.")
    p_out, l_out = zip(*existing)
    return list(p_out), list(l_out)


def _build_model(exp_id: str, device):
    if exp_id not in EXP_CFG:
        raise ValueError(f"Unknown experiment: {exp_id}")
    active_streams, fmode, extra = EXP_CFG[exp_id]
    model = V9Model(
        active_streams=active_streams, fusion_mode=fmode,
        C=cfg.C, T=cfg.T, V=cfg.V, M=cfg.M, N=cfg.N,
        int_nd=7, int_ed=4, obj_nd=6, po_ed=5,
        vit_dim=cfg.vit_embed_dim,
        hidden=cfg.graph_hidden, embed_dim=cfg.embed_dim,
        dropout=cfg.dropout, stream_dropout=cfg.stream_dropout,
        n_heads_gat=cfg.n_heads_gat, n_heads_trf=cfg.n_heads_trf,
        n_layers_trf=cfg.n_layers_trf, **extra,
    ).to(device)
    return model


def _load_checkpoint(exp_id: str, model, device):
    ckpt = RUN_DIR / exp_id / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")
    ck = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(ck["state_dict"])
    print(f"  Loaded {exp_id}: ep={ck.get('epoch','?')}  "
          f"val_f1={ck.get('val_macro_f1', float('nan')):.4f}  "
          f"thr={ck.get('threshold', 0.5):.4f}")
    return ck


def _make_loader(ds, batch_size: int = 64):
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True, collate_fn=collate_v9)


def _ece(y_true, y_prob, n_bins: int = 15) -> float:
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    n_tot = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0:
            continue
        ece += m.sum() / n_tot * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


# ══════════════════════════════════════════════════════════════════════════════
# 1. gate_weights
# ══════════════════════════════════════════════════════════════════════════════
def diag_gate_weights(split_df, device):
    print("\n" + "=" * 70)
    print("1. GATE WEIGHTS  (E_full_qgf per-sample, E_full_lw static)")
    print("=" * 70)

    # V8.1 context: PO stream increased FPs on Hockey Fights. If w_obj is
    # high here, investigate whether disabling the PO gate improves FP rate.
    print("  Note: V8.1 PO stream had near-zero signal on HF "
          "(object GQS mean ~0.111). Expect w_obj to be low.")

    test_paths, test_labels = _load_split("test", split_df)
    test_ds     = HFDataset(test_paths, test_labels)
    test_loader = _make_loader(test_ds)

    model = _build_model("E_full_qgf", device)
    ck    = _load_checkpoint("E_full_qgf", model, device)
    model.eval()
    THR   = float(ck.get("threshold", 0.5))

    # Capture gate weights by hooking into QGF.gate output
    gate_hook_outputs = []

    def _gate_hook(module, inp, out):
        # out is raw gate_logits before softmax — compute softmax here
        n_active = len(EXP_CFG["E_full_qgf"][0])  # 4
        gates = F.softmax(out[:, :n_active] / model.fusion.current_temp,
                          dim=-1)
        gate_hook_outputs.append(gates.detach().cpu())

    handle = model.fusion.gate.register_forward_hook(_gate_hook)

    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="E_full_qgf", unit="batch"):
            gate_hook_outputs.clear()
            bdev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            logits = model(bdev)
            probs  = torch.sigmoid(logits).cpu().numpy()
            labels = bdev["label"].cpu().numpy().astype(int)
            g      = gate_hook_outputs[0].numpy() if gate_hook_outputs else \
                     np.full((len(probs), 4), float("nan"))
            for i in range(len(probs)):
                rows.append({
                    "clip_id": batch["clip_id"][i],
                    "w_skel": float(g[i, 0]), "w_int": float(g[i, 1]),
                    "w_obj":  float(g[i, 2]), "w_vit": float(g[i, 3]),
                    "true_label": int(labels[i]),
                    "pred_prob":  float(probs[i]),
                    "pred_label": int(float(probs[i]) >= THR),
                })
    handle.remove()

    gw   = pd.DataFrame(rows)
    cols = ["w_skel", "w_int", "w_obj", "w_vit"]
    print(f"\n  N={len(gw)}   mean gate weights:")
    for c in cols:
        print(f"    {c:<8}: mean={gw[c].mean():.4f}  std={gw[c].std():.4f}")
    print(f"  %% w_vit>0.5: {(gw['w_vit']>0.5).mean()*100:.1f}%%   "
          f"%% w_vit>0.7: {(gw['w_vit']>0.7).mean()*100:.1f}%%")
    print(f"  %% w_obj>0.3: {(gw['w_obj']>0.3).mean()*100:.1f}%%  "
          "(flag if high — PO stream known FP risk on Hockey Fights)")

    print("\n  Mean gate weights by true label:")
    print(gw.groupby("true_label")[cols].mean().round(4).to_string())

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    gw.to_csv(str(DIAG_DIR / "gate_weights_E_full_qgf.csv"), index=False)

    # E_full_lw static weights
    try:
        model_lw = _build_model("E_full_lw", device)
        _load_checkpoint("E_full_lw", model_lw, device)
        model_lw.eval()
        with torch.no_grad():
            sw = F.softmax(model_lw.fusion.weights, dim=0).cpu().numpy()
        print("\n  E_full_lw static weights (learned, not per-sample):")
        for name, val in zip(cols, sw):
            print(f"    {name:<8}: {val:.4f}")
    except FileNotFoundError as e:
        print(f"  (skipping E_full_lw static weights: {e})")

    print(f"\n  Saved -> {DIAG_DIR / 'gate_weights_E_full_qgf.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. gqs_stratified
# ══════════════════════════════════════════════════════════════════════════════
def diag_gqs_stratified(split_df, device):
    print("\n" + "=" * 70)
    print("2. GQS-STRATIFIED  (E_full_qgf vs E_full_lw per quartile)")
    print("=" * 70)
    if not GQS_CSV.exists():
        print(f"  GQS summary not found: {GQS_CSV}; skipping.")
        return

    gqs_df   = pd.read_csv(str(GQS_CSV))
    test_gqs = gqs_df[gqs_df["split"] == "test"].copy()
    if len(test_gqs) == 0:
        print("  No test rows in GQS summary; skipping.")
        return

    test_gqs["gqs_composite"] = (0.4 * test_gqs["q_skel"] +
                                  0.4 * test_gqs["q_int"]  +
                                  0.2 * test_gqs["q_po"])
    try:
        test_gqs["q_bucket"], _ = pd.qcut(test_gqs["gqs_composite"], q=4,
                                           duplicates="drop", retbins=True)
        n_bins = test_gqs["q_bucket"].nunique()
    except Exception:
        n_bins = 0
    if n_bins < 2:
        # HF clips are very homogeneous — low GQS variance expected
        edges = np.quantile(test_gqs["gqs_composite"].values, [0, 0.33, 0.66, 1.0])
        test_gqs["q_bucket"] = pd.cut(test_gqs["gqs_composite"], bins=edges,
                                       labels=["Q1_low", "Q2_mid", "Q3_high"],
                                       include_lowest=True)
        n_bins = test_gqs["q_bucket"].nunique()
    if n_bins < 2:
        print("  GQS composite has no variance (expected for homogeneous HF clips); "
              "stratification not informative.")
        return

    test_paths, test_labels = _load_split("test", split_df)
    test_ds    = HFDataset(test_paths, test_labels)
    cid_to_idx = {test_ds.samples[i]["clip_id"]: i for i in range(len(test_ds))}

    @torch.no_grad()
    def run_subset(model, idxs):
        loader = DataLoader(Subset(test_ds, idxs), batch_size=64,
                            shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=collate_v9)
        yt, yp = [], []
        for batch in loader:
            bdev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            yp.extend(torch.sigmoid(model(bdev)).cpu().numpy().tolist())
            yt.extend(bdev["label"].cpu().numpy().astype(int).tolist())
        return np.array(yt), np.array(yp)

    def bmetrics(yt, yp, thr):
        f1 = float(f1_score(yt, (yp >= thr).astype(int), average="macro",
                             zero_division=0))
        try:
            auc = float(roc_auc_score(yt, yp))
        except ValueError:
            auc = float("nan")
        return f1, auc

    models = {}
    for exp in ("E_full_qgf", "E_full_lw"):
        try:
            m  = _build_model(exp, device)
            ck = _load_checkpoint(exp, m, device)
            m.eval()
            models[exp] = (m, float(ck.get("threshold", 0.5)))
        except FileNotFoundError as e:
            print(f"  (skipping {exp}: {e})")

    rows = []
    for bucket, grp in test_gqs.groupby("q_bucket", observed=True):
        idxs = [cid_to_idx[c] for c in grp["clip_id"].tolist()
                if c in cid_to_idx]
        if not idxs:
            continue
        row = {"bucket": str(bucket), "n": len(idxs)}
        for exp, (m, thr) in models.items():
            yt, yp = run_subset(m, idxs)
            f1, auc = bmetrics(yt, yp, thr)
            row[f"{exp}_f1"]  = round(f1,  4)
            row[f"{exp}_auc"] = round(auc, 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(str(DIAG_DIR / "gqs_stratified_comparison.csv"), index=False)
    print(f"  Saved -> {DIAG_DIR / 'gqs_stratified_comparison.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. calibration
# ══════════════════════════════════════════════════════════════════════════════
def diag_calibration(split_df, device):
    from scipy.optimize import minimize_scalar
    print("\n" + "=" * 70)
    print("3. CALIBRATION  (E_full_qgf temperature scaling, pre/post ECE)")
    print("=" * 70)
    EXP_ID = "E_full_qgf"
    model = _build_model(EXP_ID, device)
    ck    = _load_checkpoint(EXP_ID, model, device)
    model.eval()
    thr_raw = float(ck.get("threshold", 0.5))

    def collect_logits(split_name):
        paths, labels = _load_split(split_name, split_df)
        loader        = _make_loader(HFDataset(paths, labels))
        lg, lb = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{split_name} logits", unit="batch"):
                bdev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                lg.extend(model(bdev).cpu().numpy().tolist())
                lb.extend(bdev["label"].cpu().numpy().astype(int).tolist())
        return np.array(lg, dtype=np.float64), np.array(lb, dtype=np.int32)

    val_logits,  val_labels  = collect_logits("val")
    test_logits, test_labels = collect_logits("test")

    eps = 1e-7

    def nll(T):
        p = np.clip(1.0 / (1.0 + np.exp(-val_logits / T)), eps, 1 - eps)
        return -np.mean(val_labels * np.log(p) + (1 - val_labels) * np.log(1 - p))

    T_opt = float(minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x)
    print(f"  T*={T_opt:.4f}  (val NLL {nll(1.0):.4f} -> {nll(T_opt):.4f})")

    val_cal    = np.clip(1.0 / (1.0 + np.exp(-val_logits / T_opt)), eps, 1 - eps)
    best_thr   = thr_raw
    best_f1    = 0.0
    for thr in np.arange(cfg.thr_min, cfg.thr_max + 1e-6, cfg.thr_step):
        f1 = float(f1_score(val_labels, (val_cal >= thr).astype(int),
                            average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    pre  = np.clip(1.0 / (1.0 + np.exp(-test_logits)),         eps, 1 - eps)
    post = np.clip(1.0 / (1.0 + np.exp(-test_logits / T_opt)), eps, 1 - eps)

    def report(y, p, thr, tag):
        preds = (p >= thr).astype(int)
        f1    = float(f1_score(y, preds, average="macro", zero_division=0))
        acc   = float(accuracy_score(y, preds))
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = float("nan")
        ece = _ece(y, p, 15)
        print(f"  {tag:<30} F1={f1:.4f} AUC={auc:.4f} "
              f"Acc={acc:.4f} ECE={ece:.4f} thr={thr:.2f}")
        return {"macro_f1": f1, "roc_auc": auc, "accuracy": acc,
                "ece": ece, "threshold": thr}

    pre_m  = report(test_labels, pre,  thr_raw,  "Pre-calibration")
    post_m = report(test_labels, post, best_thr, "Post-calibration")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(DIAG_DIR / "calibration_results.json"), "w") as f:
        json.dump({
            "experiment": EXP_ID,
            "optimal_temperature": T_opt,
            "val_nll_before": float(nll(1.0)),
            "val_nll_after":  float(nll(T_opt)),
            "cal_threshold":  best_thr,
            "pre_calibration":  pre_m,
            "post_calibration": post_m,
        }, f, indent=2)
    print(f"  Saved -> {DIAG_DIR / 'calibration_results.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. error_analysis
# ══════════════════════════════════════════════════════════════════════════════
def diag_error_analysis(split_df, device):
    print("\n" + "=" * 70)
    print("4. ERROR ANALYSIS  (E_full_qgf TP/TN/FP/FN with per-clip GQS)")
    print("=" * 70)
    # PO-stream context: V8.1 FPs were linked to high w_obj on low-GQS clips.
    # Flag FP clips with q_po > 0.3 — may indicate spurious PO activations.

    gqs_lookup = None
    if GQS_CSV.exists():
        gqs_lookup = pd.read_csv(str(GQS_CSV)).set_index("clip_id")

    EXP_ID = "E_full_qgf"
    model  = _build_model(EXP_ID, device)
    ck     = _load_checkpoint(EXP_ID, model, device)
    model.eval()
    THR    = float(ck.get("threshold", 0.5))

    test_paths, test_labels = _load_split("test", split_df)
    test_loader = _make_loader(HFDataset(test_paths, test_labels))

    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=EXP_ID, unit="batch"):
            bdev   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            probs  = torch.sigmoid(model(bdev)).cpu().numpy()
            labels = bdev["label"].cpu().numpy().astype(int)
            for i in range(len(probs)):
                cid  = batch["clip_id"][i]
                p    = float(probs[i]); lbl = int(labels[i])
                pred = int(p >= THR)
                seg  = ("TP" if lbl == 1 and pred == 1 else
                        "TN" if lbl == 0 and pred == 0 else
                        "FP" if lbl == 0 and pred == 1 else "FN")
                g = (gqs_lookup.loc[cid]
                     if gqs_lookup is not None and cid in gqs_lookup.index
                     else None)
                rows.append({
                    "clip_id":    cid,
                    "true_label": lbl,
                    "pred_prob":  round(p, 6),
                    "pred_label": pred,
                    "segment":    seg,
                    "valid_ratio": float(g["valid_ratio"]) if g is not None else float("nan"),
                    "q_skel":  float(g["q_skel"]) if g is not None else float("nan"),
                    "q_int":   float(g["q_int"])  if g is not None else float("nan"),
                    "q_obj":   float(g["q_obj"])  if g is not None else float("nan"),
                    "q_po":    float(g["q_po"])   if g is not None else float("nan"),
                })

    df = pd.DataFrame(rows)
    print(f"\n  Total test={len(df)}  thr={THR:.4f}")
    qcols = ["pred_prob", "valid_ratio", "q_skel", "q_int", "q_obj", "q_po"]
    for seg in ["TP", "TN", "FP", "FN"]:
        sub = df[df["segment"] == seg]
        mean_str = ("" if len(sub) == 0 else
                    "  " + "  ".join(f"{c}={sub[c].mean():.3f}"
                                     for c in qcols))
        print(f"  {seg} (n={len(sub)}):{mean_str}")

    # Flag FPs with potentially spurious PO activation
    fp_sub = df[(df["segment"] == "FP") & (df["q_po"] > 0.3)]
    if len(fp_sub) > 0:
        print(f"\n  FPs with q_po>0.3 (PO-driven FP risk): {len(fp_sub)} clips")
        print("  -> Consider reporting D_skel_int_obj as primary if "
              "E_full_qgf FPs cluster here.")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(DIAG_DIR / "error_analysis.csv"), index=False)
    print(f"  Saved -> {DIAG_DIR / 'error_analysis.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
ALL_DIAGS = {
    "gate_weights":   diag_gate_weights,
    "gqs_stratified": diag_gqs_stratified,
    "calibration":    diag_calibration,
    "error_analysis": diag_error_analysis,
}


def main(args):
    seed_everything(cfg.seed)
    device = _device()
    print(f"Device : {device}")
    if not SPLIT_CSV.exists():
        raise RuntimeError(f"split_hf.csv not found at {SPLIT_CSV}. "
                           "Run Phase 1 (preprocess) + Phase 3 (train) first.")
    split_df = pd.read_csv(str(SPLIT_CSV))
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    to_run = args.only if args.only else list(ALL_DIAGS.keys())
    for name in to_run:
        if name not in ALL_DIAGS:
            print(f"  WARN: unknown diagnostic '{name}'. "
                  f"Available: {list(ALL_DIAGS.keys())}")
            continue
        try:
            ALL_DIAGS[name](split_df, device)
        except FileNotFoundError as e:
            print(f"  [skip {name}] missing checkpoint/file: {e}")
        except Exception as e:
            print(f"  [error {name}] {e}")
            raise

    print(f"\nAll requested diagnostics complete. Output -> {DIAG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hockey Fights V9 Diagnostics (local)")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Subset to run (gate_weights gqs_stratified "
             "calibration error_analysis)")
    args = parser.parse_args()
    main(args)
