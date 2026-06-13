"""
eval_clean_split.py — Quick directional eval of the CURRENT (already-trained)
VideoMAE on the new group-aware clean TEST set, WITHOUT re-training.

The honesty caveat this script handles
--------------------------------------
The current checkpoint was trained on the OLD train set (split_rlvs.csv). The new
clean test set (split_rlvs_clean.csv) re-assigns clips, so some clean-test clips
were in the OLD training data — directly, or via a scene-twin. Scoring on those is
still optimistic. So we report TWO numbers:

    CLEAN-ALL    : every clip now labelled test in split_rlvs_clean.csv
    TRULY-UNSEEN : clean-test clips whose ENTIRE scene cluster was absent from the
                   old TRAIN split — i.e. the old model never saw the scene at all.
                   This is the trustworthy directional figure.

Scene clusters come from re-running the same cached-hash clustering used by
build_clean_split.py (instant, no decode), so "twin in old train" is exact.

This does NO training and writes NO checkpoint. One forward pass over clean-test.

Usage
-----
    python eval_clean_split.py
"""

import argparse
import csv
import importlib
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VMAE = importlib.import_module("trigraph_v9_rlvs_videomae")
BCS = importlib.import_module("build_clean_split")   # reuse clustering helpers


def metrics_block(name, y_true, y_prob, thr):
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    n = len(y_true)
    yt = np.asarray(y_true); yp = np.asarray(y_prob)
    pos = int(yt.sum()); neg = n - pos
    print(f"\n  [{name}]  n={n}  (pos={pos} / neg={neg})")
    if n == 0:
        print("    (empty subset — skipped)"); return None
    pred = (yp >= thr).astype(int)
    acc = float(accuracy_score(yt, pred))
    f1 = float(f1_score(yt, pred, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(yt, yp)) if pos and neg else float("nan")
    except Exception:
        auc = float("nan")
    print(f"    accuracy : {acc:.4f}")
    print(f"    macro_f1 : {f1:.4f}")
    print(f"    roc_auc  : {auc:.4f}" + ("" if pos and neg else "   (single-class)"))
    return {"n": n, "pos": pos, "neg": neg, "accuracy": acc, "macro_f1": f1, "roc_auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=os.path.join("preproc_output", "split_rlvs_clean.csv"))
    ap.add_argument("--old", default=os.path.join("preproc_output", "split_rlvs.csv"))
    ap.add_argument("--cache", default=os.path.join("preproc_output", "phash_cache_v2.json"))
    ap.add_argument("--out", default=os.path.join("finetune_output", "eval_clean_split_report.txt"))
    args = ap.parse_args()

    import torch
    import pandas as pd

    here = os.path.dirname(os.path.abspath(__file__))
    def _abs(p): return p if os.path.isabs(p) else os.path.join(here, p)
    clean_p, old_p, cache_p, out_p = _abs(args.clean), _abs(args.old), _abs(args.cache), _abs(args.out)

    report = open(out_p, "w", encoding="utf-8")
    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, d):
            for x in self.s: x.write(d)
        def flush(self):
            for x in self.s: x.flush()
    real = sys.stdout
    sys.stdout = _Tee(real, report)

    try:
        print("RLVS V9 — CURRENT MODEL on CLEAN TEST SPLIT (quick directional eval)")
        print("=" * 72)

        # ---- 1. clusters (to know which clean-test scenes were in OLD train) -
        old_df = pd.read_csv(old_p)
        clean_df = pd.read_csv(clean_p)
        labels = {r["clip_id"]: int(r["label"]) for _, r in clean_df.iterrows()}
        hashes = BCS.load_cache(cache_p)
        pairs = BCS.build_pairs(hashes, labels)
        clusters = BCS.make_clusters(list(labels.keys()), pairs)
        cluster_of = {}
        for ci, c in enumerate(clusters):
            for cid in c:
                cluster_of[cid] = ci

        old_train_ids = set(old_df[old_df["split"] == "train"]["clip_id"])
        old_train_clusters = {cluster_of[c] for c in old_train_ids if c in cluster_of}
        print(f"old train clips: {len(old_train_ids)}  "
              f"spanning {len(old_train_clusters)} scene clusters")

        # ---- 2. build the CLEAN test set -----------------------------------
        clean_test = clean_df[clean_df["split"] == "test"]
        paths, lbls, ids = [], [], []
        missing = 0
        for _, r in clean_test.iterrows():
            cid = r["clip_id"]
            p = VMAE.CACHE_DIR / f"{cid}.npz"
            if not p.exists():
                missing += 1; continue
            paths.append(p); lbls.append(int(r["label"])); ids.append(cid)
        print(f"clean test clips with NPZ: {len(paths)} (missing {missing})")

        # which clean-test clips are 'truly unseen' by the old model?
        truly_unseen = {c for c in ids
                        if cluster_of.get(c) not in old_train_clusters}
        print(f"  of these, truly-unseen by old model (scene not in old train): "
              f"{len(truly_unseen)}")

        # ---- 3. load current best checkpoint (EMA weights + val threshold) --
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        model = VMAE.build_videomae_model(VMAE.vcfg.model_name, VMAE.vcfg.embed_dim,
                                          VMAE.vcfg.dropout).to(device)
        ck = torch.load(str(VMAE.VIT_CKPT), map_location=device)
        model.load_state_dict(ck["ema_state_dict"])
        thr = float(ck["threshold"])
        print(f"checkpoint epoch={ck.get('epoch','?')} thr={thr:.4f} "
              f"(val_f1={ck.get('val_macro_f1', float('nan')):.4f})")

        # ---- 4. forward pass ------------------------------------------------
        ds = VMAE.RLVSVideoDataset(paths, lbls, augment=False)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=VMAE.vcfg.batch_size, shuffle=False,
            num_workers=0, collate_fn=VMAE.collate_fn)
        y_true, y_prob, _, _, _ = VMAE.evaluate(model, loader, device)
        loaded_ids = [os.path.splitext(os.path.basename(s["path"]))[0] for s in ds.samples]
        assert len(loaded_ids) == len(y_prob) == len(y_true)

        # persist predictions
        pred_csv = os.path.join(os.path.dirname(out_p), "clean_test_predictions.csv")
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "label", "prob", "truly_unseen"])
            for cid, yt, yp in zip(loaded_ids, y_true, y_prob):
                w.writerow([cid, int(yt), float(yp), int(cid in truly_unseen)])
        print(f"wrote per-clip predictions -> {pred_csv}")

        # ---- 5. metrics -----------------------------------------------------
        print("\n" + "=" * 72)
        print(f"METRICS  (fixed val-tuned threshold = {thr:.4f})")
        print("=" * 72)
        m_all = metrics_block("CLEAN-ALL  (every clean-test clip)", y_true, y_prob, thr)
        idx_u = [i for i, c in enumerate(loaded_ids) if c in truly_unseen]
        m_uns = metrics_block("TRULY-UNSEEN (scene absent from old train — trustworthy)",
                              [y_true[i] for i in idx_u], [y_prob[i] for i in idx_u], thr)

        print("\n" + "=" * 72)
        print("VERDICT")
        print("=" * 72)
        print(f"  reported (old random split) : macro_f1 0.9950")
        if m_all:
            print(f"  CLEAN-ALL                   : macro_f1 {m_all['macro_f1']:.4f}")
        if m_uns:
            print(f"  TRULY-UNSEEN (trust this)   : macro_f1 {m_uns['macro_f1']:.4f}")
            print()
            print("  TRULY-UNSEEN is the directional read on genuine generalization for")
            print("  the CURRENT model. If it stays high, a full re-train on the clean")
            print("  split will likely confirm a strong, defensible number. If it drops")
            print("  a lot, the old score leaned on memorized scenes and the re-train is")
            print("  essential before reporting anything.")
            print()
            print("  NOTE: even TRULY-UNSEEN is only directional — the model still TRAINED")
            print("  on the old data distribution. The publishable number requires a full")
            print("  re-fine-tune on split_rlvs_clean.csv. This just tells you what to expect.")
    finally:
        sys.stdout = real
        report.close()
        print(f"\n[report written to: {out_p}]")


if __name__ == "__main__":
    main()
