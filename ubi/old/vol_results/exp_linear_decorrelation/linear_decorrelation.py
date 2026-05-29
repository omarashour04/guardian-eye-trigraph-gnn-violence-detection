"""
linear_decorrelation.py
Guardian Eye — V9  |  Option 2: post-hoc LINEAR quality decorrelation of the
                                 frozen VideoMAE embeddings (NO re-fine-tune).

Question this answers
---------------------
embedding_shortcut.csv proved the val->test collapse is frozen INTO the 768-d
VideoMAE embedding: a linear probe (embedding -> label) scores AUC 0.916 on val
but 0.681 on test, mirroring the full graph head. The quality direction q_skel
is linearly decodable from the embedding at AUC 0.87-0.99.

If the shortcut is mostly LINEAR, we can remove it for free: fit the quality
direction(s) on TRAIN, project them OUT of every embedding, then re-fit the
label probe. If test AUC jumps toward val, a cheap projection layer in the graph
head fixes everything — no A10G re-fine-tune. If test AUC barely moves, the
shortcut is non-linear and we have earned the case for re-fine-tuning Stage A.

This is read-only w.r.t. the volume: it reads NPZ vit_embedding + the GQS
summary, fits tiny sklearn models in-memory, and writes ONE results CSV.

Run
---
    conda run -n dlcuda128 modal run \
        exp_linear_decorrelation/linear_decorrelation.py::run

Output (downloaded to ./exp_linear_decorrelation/results/):
    linear_decorrelation.csv
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from pathlib import Path
from dataclasses import dataclass
import modal

# ── Modal primitives (same volume/mounts as the diagnostics script) ────────────
APP_NAME      = "guardian-eye-v9-ubi-lindecorr"
VOL_NAME_PROC = "ubi-fights-processed"

app      = modal.App(APP_NAME)
vol_proc = modal.Volume.from_name(VOL_NAME_PROC, create_if_missing=True)

PROC_MOUNT = Path("/data/proc")
CACHE_DIR  = PROC_MOUNT / "cache_v9"
SPLIT_CSV  = PROC_MOUNT / "split_ubi.csv"
GQS_CSV    = PROC_MOUNT / "gqs_summary_ubi.csv"
RUN_DIR    = PROC_MOUNT / "runs_ubi"
DIAG_DIR   = RUN_DIR / "diagnostics"

# No GPU needed — sklearn on 768-d vectors. CPU container is enough.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "tqdm==4.66.4",
    )
)

_FUNC_KWARGS = dict(
    image=image,
    cpu=4,
    memory=16384,
    timeout=1200,
    volumes={str(PROC_MOUNT): vol_proc},
)


@dataclass
class CFG:
    vit_embed_dim: int = 768
    seed:          int = 42
    # How many leading quality directions to strip. We sweep [0, 1, 2, 4, 8]
    # so we can see how AUC moves as we remove more of the linear shortcut.
    n_strip_sweep: tuple = (0, 1, 2, 4, 8)


cfg = CFG()


# ══════════════════════════════════════════════════════════════════════════════
# Embedding loading (mirrors embedding_shortcut in the diagnostics script;
# UBI CSVs key on clip_id, NOT video_id)
# ══════════════════════════════════════════════════════════════════════════════
def _load_split_embeddings():
    """Return dict[split] -> (X[N,768], y[N], q_skel[N]) loaded from the volume."""
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    split_df = pd.read_csv(str(SPLIT_CSV))                 # clip_id, split, label
    g = pd.read_csv(str(GQS_CSV)).set_index("clip_id")     # per-clip quality

    def load(sp: str):
        sub = split_df[split_df["split"] == sp]
        X, y, qsk = [], [], []
        for cid, lab in tqdm(zip(sub["clip_id"], sub["label"]),
                             total=len(sub), desc=f"emb[{sp}]", unit="clip"):
            p = CACHE_DIR / f"{cid}.npz"
            if not p.exists():
                continue
            d = np.load(str(p), allow_pickle=True)
            if "vit_embedding" not in d:
                continue
            X.append(d["vit_embedding"].astype(np.float32).ravel())
            y.append(int(lab))
            qsk.append(float(g["q_skel"].get(cid, np.nan))
                       if "q_skel" in g.columns else np.nan)
        X = np.stack(X) if X else np.zeros((0, cfg.vit_embed_dim), np.float32)
        return X, np.array(y), np.array(qsk, dtype=np.float32)

    return {sp: load(sp) for sp in ("train", "val", "test")}


# ══════════════════════════════════════════════════════════════════════════════
# Linear concept removal (INLP-style): iteratively fit a linear quality probe on
# TRAIN, then project its direction out of the embedding space. Repeating k times
# strips the top-k linear quality directions. All projections are derived from
# TRAIN only, then applied to val/test (no leakage).
# ══════════════════════════════════════════════════════════════════════════════
def _build_nullspace_projection(Xtr_std, q_target, k: int, seed: int):
    """Return a [D,D] projection matrix P that removes k linear directions most
    predictive of q_target (high-quality skeleton) on standardized train embs."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    D = Xtr_std.shape[1]
    P = np.eye(D, dtype=np.float64)
    X = Xtr_std.astype(np.float64).copy()
    for _ in range(k):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        # Fit on the CURRENT (already-partially-projected) train embeddings.
        clf.fit(X @ P.T, q_target)
        w = clf.coef_.ravel().astype(np.float64)
        # Map the direction back through the current projection so it lives in
        # the original standardized space, then build its rank-1 nullspace proj.
        w = P.T @ w
        nrm = np.linalg.norm(w)
        if nrm < 1e-12:
            break
        w = w / nrm
        P = (np.eye(D) - np.outer(w, w)) @ P
    return P


def _label_probe_auc(Xtr, ytr, splits, scaler, P):
    """Fit a balanced logistic label-probe on projected TRAIN, score each split."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def proj(X):
        return scaler.transform(X) @ P.T

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(proj(Xtr), ytr)
    out = {}
    for sp, (X, y, _q) in splits.items():
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            out[sp] = float("nan")
        else:
            out[sp] = float(roc_auc_score(
                y, clf.predict_proba(proj(X))[:, 1]))
    return out


def _quality_probe_auc(Xtr, q_tr_hi, splits, scaler, P, thr_q):
    """How well can q_skel-high still be decoded AFTER projection? (Should drop.)"""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def proj(X):
        return scaler.transform(X) @ P.T

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(proj(Xtr), q_tr_hi)
    out = {}
    for sp, (X, _y, q) in splits.items():
        q_hi = (q >= thr_q).astype(int)
        if X.shape[0] == 0 or len(np.unique(q_hi)) < 2:
            out[sp] = float("nan")
        else:
            out[sp] = float(roc_auc_score(
                q_hi, clf.predict_proba(proj(X))[:, 1]))
    return out


@app.function(**_FUNC_KWARGS)
def run() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    np.random.seed(cfg.seed)
    vol_proc.reload()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading frozen VideoMAE embeddings from volume ===")
    splits = _load_split_embeddings()
    Xtr, ytr, qtr = splits["train"]
    print(f"  train {Xtr.shape}  val {splits['val'][0].shape}  "
          f"test {splits['test'][0].shape}")
    if Xtr.shape[0] == 0:
        raise RuntimeError("No train embeddings loaded — check cache_v9 on volume.")

    # Standardizer fit on TRAIN only; reused for every projection level.
    scaler = StandardScaler().fit(Xtr)

    # q_skel-high target, binarized at the TRAIN median (global, leakage-free).
    thr_q = float(np.nanmedian(qtr))
    qtr_hi = (qtr >= thr_q).astype(int)
    print(f"  q_skel high/low split at train median {thr_q:.3f} "
          f"(train high={qtr_hi.sum()} / low={len(qtr_hi) - qtr_hi.sum()})")

    Xtr_std = scaler.transform(Xtr)

    rows = []
    print("\n=== Sweep: strip top-k linear quality directions, re-probe ===")
    print(f"{'k':>3} | {'label_tr':>8} {'label_va':>8} {'label_te':>8} "
          f"{'va-te gap':>9} | {'qskl_tr':>8} {'qskl_te':>8}")
    print("-" * 72)
    for k in cfg.n_strip_sweep:
        if k == 0:
            D = Xtr_std.shape[1]
            P = np.eye(D, dtype=np.float64)
        else:
            P = _build_nullspace_projection(Xtr_std, qtr_hi, k, cfg.seed)

        lab = _label_probe_auc(Xtr, ytr, splits, scaler, P)
        qa  = _quality_probe_auc(Xtr, qtr_hi, splits, scaler, P, thr_q)
        gap = lab["val"] - lab["test"]
        print(f"{k:>3} | {lab['train']:>8.4f} {lab['val']:>8.4f} "
              f"{lab['test']:>8.4f} {gap:>+9.4f} | "
              f"{qa['train']:>8.4f} {qa['test']:>8.4f}")
        rows.append({
            "k_directions_stripped": k,
            "label_auc_train": round(lab["train"], 4),
            "label_auc_val":   round(lab["val"], 4),
            "label_auc_test":  round(lab["test"], 4),
            "val_test_gap":    round(gap, 4),
            "qskel_auc_train": round(qa["train"], 4),
            "qskel_auc_test":  round(qa["test"], 4),
        })

    out_df = pd.DataFrame(rows)
    out_path = DIAG_DIR / "linear_decorrelation.csv"
    out_df.to_csv(str(out_path), index=False)
    print(f"\n  Saved: {out_path}")

    # Verdict helper: did stripping quality help test without nuking it?
    base = out_df.iloc[0]
    best = out_df.loc[out_df["label_auc_test"].idxmax()]
    print("\n=== Verdict ===")
    print(f"  baseline (k=0) test AUC : {base['label_auc_test']:.4f}  "
          f"(val {base['label_auc_val']:.4f}, gap {base['val_test_gap']:+.4f})")
    print(f"  best test AUC at k={int(best['k_directions_stripped'])} : "
          f"{best['label_auc_test']:.4f}  (gap {best['val_test_gap']:+.4f})")
    delta = best["label_auc_test"] - base["label_auc_test"]
    print(f"  test AUC change from stripping linear quality: {delta:+.4f}")
    if delta >= 0.05:
        print("  > Linear decorrelation HELPS materially. A projection layer in "
              "the graph head may fix this WITHOUT re-fine-tuning VideoMAE.")
    else:
        print("  > Linear decorrelation does NOT recover test AUC. The shortcut "
              "is non-linear / baked deep — re-fine-tuning Stage A is justified.")

    vol_proc.commit()
    print("linear_decorrelation complete.")


@app.local_entrypoint()
def main() -> None:
    """Run on Modal, then download linear_decorrelation.csv locally."""
    import sys
    from pathlib import Path as LocalPath

    LOCAL_OUT = LocalPath("exp_linear_decorrelation/results")
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    print("Running linear quality-decorrelation sweep on Modal (CPU)...")
    run.remote()

    rel_path = "runs_ubi/diagnostics/linear_decorrelation.csv"
    try:
        data = b"".join(vol_proc.read_file(rel_path))
        dest = LOCAL_OUT / "linear_decorrelation.csv"
        dest.write_bytes(data)
        print(f"  [ok] {dest.resolve()}")
    except Exception as e:
        print(f"  [FAIL] could not download {rel_path}: {e}")
        sys.exit(1)
