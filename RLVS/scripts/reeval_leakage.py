"""
reeval_leakage.py — Re-evaluate the fine-tuned VideoMAE on the TEST set with the
leaked (near-duplicate-of-train/val) clips separated out, to measure how much the
0.995 headline is inflated by data leakage.

Why this exists
---------------
inspect_outputs.py --phash (v2) found that 36 of 200 test clips are near-duplicates
of a train/val clip (all NonViolence). The saved run only stored AGGREGATE test
metrics, not per-clip probabilities, so we cannot recompute from disk — the model
must do one more forward pass over the test set. This script does exactly that and
NOTHING else (no training, no checkpoint writes).

It reuses the real training module so preprocessing / model / sigmoid path are
byte-identical to how the reported 0.995 was produced:
    - build_videomae_model(), VideoMAEDataset, collate_fn, evaluate()  (imported)
    - same best checkpoint (videomae_best.pt), same EMA weights, same threshold

What it reports
---------------
Metrics computed THREE ways, all using the checkpoint's own val-tuned threshold:
    FULL    : all 200 test clips                  (= reproduces the 0.995)
    CLEAN   : the 164 clips with NO train/val twin (= honest generalization)
    LEAKED  : the 36 clips that DO have a twin      (= what leakage was buying)

Usage
-----
    python reeval_leakage.py
    python reeval_leakage.py --leak-csv preproc_output/phash_test_leakage_pairs.csv \
                             --out finetune_output/reeval_leakage_report.txt
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# force UTF-8 console (Windows cp1252 chokes on em-dashes / symbols)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Import the real training module so we reuse its exact model + data pipeline.
# The module name has no .py and contains no top-level side effects beyond defs
# (main() is guarded by __main__), so importing is safe.
import importlib

VMAE = importlib.import_module("trigraph_v9_rlvs_videomae")


def load_leaked_test_ids(leak_csv):
    """Return the set of TEST-side clip_ids that have a train/val near-dup twin."""
    leaked = set()
    if not os.path.exists(leak_csv):
        raise FileNotFoundError(
            f"leak CSV not found: {leak_csv}. Run inspect_outputs.py --phash first."
        )
    with open(leak_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # a pair can have the test clip on either side
            if row["split_a"] == "test":
                leaked.add(row["clip_a"])
            if row["split_b"] == "test":
                leaked.add(row["clip_b"])
    return leaked


def metrics_block(name, y_true, y_prob, thr):
    """Compute and print acc / macro-F1 / ROC-AUC at a FIXED threshold."""
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = int(y_true.sum())
    neg = n - pos
    print(f"\n  [{name}]  n={n}  (pos={pos} / neg={neg})")
    if n == 0:
        print("    (empty subset — skipped)")
        return None
    y_pred = (y_prob >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if pos and neg else float("nan")
    except Exception:
        auc = float("nan")
    print(f"    accuracy : {acc:.4f}")
    print(f"    macro_f1 : {f1:.4f}")
    print(f"    roc_auc  : {auc:.4f}" + ("   (single-class subset)" if not (pos and neg) else ""))
    return {"n": n, "pos": pos, "neg": neg, "accuracy": acc,
            "macro_f1": f1, "roc_auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak-csv",
                    default=os.path.join("preproc_output", "phash_test_leakage_pairs.csv"))
    ap.add_argument("--out",
                    default=os.path.join("finetune_output", "reeval_leakage_report.txt"))
    args = ap.parse_args()

    import torch

    here = os.path.dirname(os.path.abspath(__file__))
    leak_csv = args.leak_csv if os.path.isabs(args.leak_csv) else os.path.join(here, args.leak_csv)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    report = open(out_path, "w", encoding="utf-8")

    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, d):
            for x in self.s: x.write(d)
        def flush(self):
            for x in self.s: x.flush()

    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, report)

    try:
        print("RLVS V9 — TEST RE-EVALUATION WITH LEAKAGE SEPARATED")
        print("=" * 72)

        # ---- 1. which test clips are leaked --------------------------------
        leaked_ids = load_leaked_test_ids(leak_csv)
        print(f"leaked test clip_ids (have a train/val near-dup twin): {len(leaked_ids)}")
        print(f"  {sorted(leaked_ids)}")

        # ---- 2. rebuild the TEST set exactly as training did ----------------
        import pandas as pd
        split_df = pd.read_csv(str(VMAE.SPLIT_CSV))
        test_rows = split_df[split_df["split"] == "test"]
        # NPZ path per clip = CACHE_DIR/{clip_id}.npz  (same rule as get_split)
        paths, labels, clip_ids = [], [], []
        missing = 0
        for _, r in test_rows.iterrows():
            cid = r["clip_id"]
            p = VMAE.CACHE_DIR / f"{cid}.npz"
            if not p.exists():
                missing += 1
                continue
            paths.append(p)
            labels.append(int(r["label"]))
            clip_ids.append(cid)
        print(f"\ntest clips with NPZ present: {len(paths)} (missing {missing})")

        # ---- 3. load model + best checkpoint (EMA weights, val threshold) ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        model = VMAE.build_videomae_model(VMAE.vcfg.model_name, VMAE.vcfg.embed_dim,
                                          VMAE.vcfg.dropout).to(device)
        if not VMAE.VIT_CKPT.exists():
            raise FileNotFoundError(f"checkpoint not found: {VMAE.VIT_CKPT}")
        ck = torch.load(str(VMAE.VIT_CKPT), map_location=device)
        # the reported metrics used the EMA weights — match that exactly
        model.load_state_dict(ck["ema_state_dict"])
        thr = float(ck["threshold"])
        print(f"loaded best checkpoint: epoch={ck.get('epoch','?')}  "
              f"val_f1={ck.get('val_macro_f1', float('nan')):.4f}  "
              f"threshold={thr:.4f}")

        # ---- 4. forward pass over the test set, collect per-clip prob -------
        ds = VMAE.RLVSVideoDataset(paths, labels, augment=False)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=VMAE.vcfg.batch_size, shuffle=False,
            num_workers=0, collate_fn=VMAE.collate_fn)

        # evaluate() returns y_true,y_prob in dataset order; dataset preserves the
        # order of `paths`, so we can zip back to clip_ids. But evaluate() drops
        # any sample that failed to load — guard by re-deriving from ds.samples.
        y_true, y_prob, _, _, _ = VMAE.evaluate(model, loader, device)
        loaded_ids = [os.path.splitext(os.path.basename(s["path"]))[0]
                      for s in ds.samples]
        assert len(loaded_ids) == len(y_prob) == len(y_true), \
            f"alignment mismatch: ids={len(loaded_ids)} prob={len(y_prob)} true={len(y_true)}"

        # persist per-clip predictions so we never have to re-run the model again
        pred_csv = os.path.join(os.path.dirname(out_path), "test_predictions.csv")
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "label", "prob", "leaked"])
            for cid, yt, yp in zip(loaded_ids, y_true, y_prob):
                w.writerow([cid, int(yt), float(yp), int(cid in leaked_ids)])
        print(f"wrote per-clip predictions -> {pred_csv}")

        # ---- 5. split into FULL / CLEAN / LEAKED and score -----------------
        idx_clean = [i for i, c in enumerate(loaded_ids) if c not in leaked_ids]
        idx_leak = [i for i, c in enumerate(loaded_ids) if c in leaked_ids]

        print("\n" + "=" * 72)
        print(f"METRICS  (fixed val-tuned threshold = {thr:.4f})")
        print("=" * 72)
        m_full = metrics_block("FULL  test (reproduces the reported number)",
                               y_true, y_prob, thr)
        m_clean = metrics_block("CLEAN test (no train/val twin — honest score)",
                                [y_true[i] for i in idx_clean],
                                [y_prob[i] for i in idx_clean], thr)
        m_leak = metrics_block("LEAKED clips only (what leakage was buying)",
                               [y_true[i] for i in idx_leak],
                               [y_prob[i] for i in idx_leak], thr)

        # ---- 6. verdict ----------------------------------------------------
        print("\n" + "=" * 72)
        print("VERDICT")
        print("=" * 72)
        if m_full and m_clean:
            d_f1 = m_full["macro_f1"] - m_clean["macro_f1"]
            d_acc = m_full["accuracy"] - m_clean["accuracy"]
            print(f"  FULL  macro_f1 {m_full['macro_f1']:.4f}  ->  "
                  f"CLEAN macro_f1 {m_clean['macro_f1']:.4f}   "
                  f"(drop {d_f1:+.4f})")
            print(f"  FULL  accuracy {m_full['accuracy']:.4f}  ->  "
                  f"CLEAN accuracy {m_clean['accuracy']:.4f}   "
                  f"(drop {d_acc:+.4f})")
            print()
            if abs(d_f1) < 0.01:
                print("  The CLEAN score barely moves: leakage was NOT the main driver of")
                print("  the high number. The model genuinely separates this test set.")
                print("  (Still rebuild a group-aware split for a defensible protocol.)")
            else:
                print("  The CLEAN score drops materially: part of the 0.995 was memorized")
                print("  duplicates. The honest generalization figure is the CLEAN row above.")
                print("  Rebuild a group-aware split and re-report on it.")
            if m_clean["accuracy"] <= VMAE_ceiling():
                print(f"\n  Note: CLEAN accuracy now sits at/under the ~0.95 RLVS appearance")
                print(f"  ceiling — consistent with a believable, non-leaked result.")
    finally:
        sys.stdout = real_stdout
        report.close()
        print(f"\n[report written to: {out_path}]")


def VMAE_ceiling():
    return 0.95


if __name__ == "__main__":
    main()
