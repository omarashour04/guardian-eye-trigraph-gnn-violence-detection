"""
plot_figures.py  —  Build ALL paper inference figures from per-clip dump CSVs.

Runs on the PAPER machine (no GPU needed). Consumes (new per-dataset layout):
    paper_data/ntu/perclip.csv
    paper_data/rlvs/perclip.csv
    paper_data/rwf/perclip.csv
    paper_data/hf/perclip.csv

Each CSV must have:
    clip_id, true_label,
    q_skel, q_int, q_obj, q_po, q_valid, gqs_composite,
    w_skel, w_int, w_obj, w_vit,
    prob__<variant>      (one or more columns)
    [source]             (optional — present in NTU/RLVS)

Generates per dataset present (NNN = ntu/rlvs/rwf/hf):
    NNN_roc.pdf                 ROC curves, one per variant
    NNN_pr.pdf                  Precision-recall curves, one per variant
    NNN_confusion.pdf           Confusion matrix (deployed variant)
    NNN_calibration.pdf         Calibration curve (reliability diagram) + ECE
    NNN_score_dist.pdf          Score distribution (Violence vs. NonViolence)
    NNN_gate_dist.pdf           Gate-weight violin/boxplot, 4 streams
    NNN_gate_by_quality.pdf     Mean gate weight vs GQS-composite tercile (stacked bar)
    NNN_gate_vs_gqs_comp.pdf    Per-gate scatter / line vs gqs_composite
    NNN_gate_vs_q_skel.pdf      Per-gate vs q_skel
    NNN_gate_vs_q_int.pdf       Per-gate vs q_int
    NNN_gate_vs_q_obj.pdf       Per-gate vs q_obj
    NNN_gate_vs_q_po.pdf        Per-gate vs q_po
    NNN_gqs_dist.pdf            GQS-component distribution (violin)
    NNN_gqs_stratified.pdf      Macro-F1 per GQS tercile (deployed variant)
    NNN_gate_entropy.pdf        Per-clip gate entropy histogram (QGF diversity)
    NNN_ablation_bar.pdf        Macro-F1 ablation bar chart across variants

Cross-dataset figures (when >=2 datasets present):
    all_roc_overlay.pdf         All datasets + variants on one ROC canvas

Also prints a NUMBERS block (ground-truth values to reconcile paper text/tables).

Usage:
    python plot_figures.py --dumps ./dumps --outdir ./out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix,
)

# --------------------------------------------------------------------------- #
# Style constants — match paper TikZ palette
# --------------------------------------------------------------------------- #
STREAM_COLORS = {
    "w_skel": "#3B6FB6",
    "w_int":  "#E08A1E",
    "w_obj":  "#3E9B57",
    "w_vit":  "#7E5AA8",
}
STREAM_LABELS = {
    "w_skel": "Skeleton", "w_int": "Interaction",
    "w_obj": "Object", "w_vit": "Appearance",
}
GATE_COLS = ["w_skel", "w_int", "w_obj", "w_vit"]
GQS_COLS  = ["q_skel", "q_int", "q_obj", "q_po", "q_valid"]
GQS_LABELS = {
    "q_skel": "$q_{\\mathrm{skel}}$",
    "q_int":  "$q_{\\mathrm{int}}$",
    "q_obj":  "$q_{\\mathrm{obj}}$",
    "q_po":   "$q_{\\mathrm{po}}$",
    "q_valid":"$q_{\\mathrm{valid}}$",
}

VARIANT_LABELS = {
    "E_full_qgf":       "QGF (full)",
    "E_full_lw":        "Learned weights",
    "E_full_qgf_fixed": "QGF (fixed)",
    "E_full_qgf_adv":   "QGF-adv",
    "E_full_cgf":       "Content-gated",
    "C_videomae_only":  "VideoMAE only",
}
# Ablation order (A→B→D→E_lw→E_qgf in paper; use subset of what exists)
ABLATION_ORDER = [
    "C_videomae_only",
    "E_full_lw",
    "E_full_qgf_fixed",
    "E_full_qgf",
    "E_full_cgf",
    "E_full_qgf_adv",
]

DEPLOYED = "E_full_qgf"

DS_COLORS = {"ntu": "#C0392B", "rlvs": "#2980B9", "rwf": "#27AE60", "hf": "#8E44AD"}

RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _variant_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("prob__")]


def _macro_f1(y_true, prob, thr=0.5) -> float:
    return float(f1_score(y_true, (prob >= thr).astype(int),
                          average="macro", zero_division=0))


def _best_thr(y_true, prob) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = _macro_f1(y_true, prob, t)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def _terciles(gqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(gqs, [0.0, 1/3, 2/3, 1.0])
    edges = np.unique(edges)
    if len(edges) < 4:
        return np.array(["All"] * len(gqs)), edges
    idx = np.clip(np.digitize(gqs, edges[1:-1]), 0, 2)
    return np.array(["Low", "Mid", "High"])[idx], edges


def _gate_entropy(gates: np.ndarray) -> np.ndarray:
    g = np.clip(gates, 1e-9, 1.0)
    g = g / g.sum(axis=1, keepdims=True)
    return -np.sum(g * np.log(g), axis=1) / np.log(g.shape[1])  # normalised [0,1]


def _ece(y_true, prob, n_bins=10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = prob[m].mean()
        ece += m.sum() / n * abs(acc - conf)
    return float(ece)


def _deployed_col(df: pd.DataFrame) -> str:
    if f"prob__{DEPLOYED}" in df.columns:
        return f"prob__{DEPLOYED}"
    return _variant_cols(df)[0]


def _fig(w=3.4, h=3.0):
    plt.rcParams.update(RC)
    return plt.subplots(figsize=(w, h))


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


# --------------------------------------------------------------------------- #
# Figure generators
# --------------------------------------------------------------------------- #

def fig_roc(df: pd.DataFrame, ds: str, outdir: Path) -> dict[str, float]:
    y = df["true_label"].values.astype(int)
    vcols = _variant_cols(df)
    fig, ax = _fig(3.5, 3.2)
    aucs: dict[str, float] = {}
    for c in sorted(vcols):
        exp = c[len("prob__"):]
        p = df[c].values
        m = ~np.isnan(p)
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue
        auc = roc_auc_score(y[m], p[m])
        aucs[exp] = auc
        fpr, tpr, _ = roc_curve(y[m], p[m])
        lbl = VARIANT_LABELS.get(exp, exp)
        lw = 2.0 if exp == DEPLOYED else 1.2
        ax.plot(fpr, tpr, lw=lw, label=f"{lbl} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, outdir / f"{ds}_roc.pdf")
    return aucs


def fig_pr(df: pd.DataFrame, ds: str, outdir: Path) -> dict[str, float]:
    y = df["true_label"].values.astype(int)
    vcols = _variant_cols(df)
    fig, ax = _fig(3.5, 3.2)
    aps: dict[str, float] = {}
    for c in sorted(vcols):
        exp = c[len("prob__"):]
        p = df[c].values
        m = ~np.isnan(p)
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue
        ap = average_precision_score(y[m], p[m])
        aps[exp] = ap
        prec, rec, _ = precision_recall_curve(y[m], p[m])
        lbl = VARIANT_LABELS.get(exp, exp)
        lw = 2.0 if exp == DEPLOYED else 1.2
        ax.plot(rec, prec, lw=lw, label=f"{lbl} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=False)
    _save(fig, outdir / f"{ds}_pr.pdf")
    return aps


def fig_confusion(df: pd.DataFrame, ds: str, outdir: Path) -> None:
    col = _deployed_col(df)
    y = df["true_label"].values.astype(int)
    p = df[col].values
    thr, _ = _best_thr(y, p)
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    fig, ax = _fig(2.8, 2.5)
    im = ax.imshow(cm, cmap="Blues")
    classes = ["NonViolence", "Violence"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=10, fontweight="bold")
    ax.set_title(f"thr={thr:.2f}  |  {col[len('prob__'):]}", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, outdir / f"{ds}_confusion.pdf")


def fig_calibration(df: pd.DataFrame, ds: str, outdir: Path) -> float:
    col = _deployed_col(df)
    y = df["true_label"].values.astype(int)
    p = df[col].values
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means, acc_means, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        bin_means.append(p[m].mean())
        acc_means.append(y[m].mean())
        counts.append(m.sum())
    ece = _ece(y, p, n_bins)
    fig, ax = _fig(3.2, 3.0)
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=0.8, label="Perfect")
    ax.scatter(bin_means, acc_means, s=[c / 2 for c in counts],
               color="#3B6FB6", alpha=0.7, zorder=3)
    ax.plot(bin_means, acc_means, color="#3B6FB6", lw=1.5, label=f"ECE={ece:.4f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("Reliability diagram")
    _save(fig, outdir / f"{ds}_calibration.pdf")
    return ece


def fig_score_dist(df: pd.DataFrame, ds: str, outdir: Path) -> None:
    col = _deployed_col(df)
    y = df["true_label"].values.astype(int)
    p = df[col].values
    fig, ax = _fig(3.5, 3.0)
    ax.hist(p[y == 0], bins=40, alpha=0.6, color="#2980B9", density=True,
            label="NonViolence")
    ax.hist(p[y == 1], bins=40, alpha=0.6, color="#C0392B", density=True,
            label="Violence")
    thr, _ = _best_thr(y, p)
    ax.axvline(thr, color="k", linestyle="--", lw=1.0, label=f"thr={thr:.2f}")
    ax.set_xlabel("Predicted score"); ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _save(fig, outdir / f"{ds}_score_dist.pdf")


def fig_gate_dist(df: pd.DataFrame, ds: str, outdir: Path) -> None:
    fig, ax = _fig(3.8, 3.2)
    data = [df[c].dropna().values for c in GATE_COLS]
    vp = ax.violinplot(data, positions=range(4), showmedians=True,
                       showextrema=True)
    for i, (body, col) in enumerate(zip(vp["bodies"], GATE_COLS)):
        body.set_facecolor(STREAM_COLORS[col])
        body.set_alpha(0.7)
    vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(1.5)
    vp["cmaxes"].set_color("0.5"); vp["cmins"].set_color("0.5")
    vp["cbars"].set_color("0.5")
    ax.set_xticks(range(4))
    ax.set_xticklabels([STREAM_LABELS[c] for c in GATE_COLS], rotation=15, ha="right")
    ax.set_ylabel("Gate weight"); ax.set_ylim(0, 1)
    _save(fig, outdir / f"{ds}_gate_dist.pdf")


def fig_gate_by_quality(df: pd.DataFrame, ds: str, outdir: Path) -> pd.DataFrame:
    g, _ = _terciles(df["gqs_composite"].values)
    order = [b for b in ["Low", "Mid", "High", "All"] if b in set(g)]
    means = {b: df.loc[g == b, GATE_COLS].mean() for b in order}
    fig, ax = _fig(3.4, 3.0)
    x = np.arange(len(order)); bottom = np.zeros(len(order))
    for col in GATE_COLS:
        vals = np.array([means[b][col] for b in order])
        ax.bar(x, vals, bottom=bottom, width=0.6,
               color=STREAM_COLORS[col], label=STREAM_LABELS[col])
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_xlabel("GQS-composite tercile"); ax.set_ylabel("Mean gate weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    _save(fig, outdir / f"{ds}_gate_by_quality.pdf")
    return pd.DataFrame(means).T


def fig_gate_vs_gqs(df: pd.DataFrame, ds: str, outdir: Path,
                    gqs_col: str, gqs_label: str | None = None) -> None:
    if gqs_col not in df.columns:
        return
    x = df[gqs_col].values
    fig, ax = _fig(4.0, 3.0)
    # scatter one point per clip, colour by stream
    for gc in GATE_COLS:
        ax.scatter(x, df[gc].values, s=3, alpha=0.25, color=STREAM_COLORS[gc],
                   rasterized=True, label=None)
    # overlay lowess-style binned means
    bins = np.linspace(x.min(), x.max(), 12)
    for gc in GATE_COLS:
        means = []
        mids = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum() >= 5:
                means.append(df.loc[m, gc].mean())
                mids.append((lo + hi) / 2)
        if means:
            ax.plot(mids, means, color=STREAM_COLORS[gc], lw=2.0,
                    label=STREAM_LABELS[gc])
    lbl = gqs_label or gqs_col
    ax.set_xlabel(lbl); ax.set_ylabel("Gate weight"); ax.set_ylim(0, 1)
    ax.legend(loc="center right", frameon=False, ncol=1)
    name = gqs_col.replace("gqs_", "").replace("q_", "")
    _save(fig, outdir / f"{ds}_gate_vs_{name}.pdf")


def fig_gqs_dist(df: pd.DataFrame, ds: str, outdir: Path) -> None:
    present = [c for c in GQS_COLS if c in df.columns]
    if not present:
        return
    fig, ax = _fig(4.5, 3.2)
    data = [df[c].dropna().values for c in present]
    colors_gqs = ["#3B6FB6", "#E08A1E", "#3E9B57", "#7E5AA8", "#999999"]
    vp = ax.violinplot(data, positions=range(len(present)), showmedians=True,
                       showextrema=True)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors_gqs[i % len(colors_gqs)])
        body.set_alpha(0.7)
    vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(1.5)
    vp["cmaxes"].set_color("0.5"); vp["cmins"].set_color("0.5")
    vp["cbars"].set_color("0.5")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([GQS_LABELS.get(c, c) for c in present],
                       rotation=0, ha="center", usetex=False)
    ax.set_ylabel("GQS component value"); ax.set_ylim(0, 1.05)
    _save(fig, outdir / f"{ds}_gqs_dist.pdf")


def fig_gqs_stratified(df: pd.DataFrame, ds: str, outdir: Path) -> tuple:
    col = _deployed_col(df)
    y = df["true_label"].values.astype(int)
    p = df[col].values
    thr, _ = _best_thr(y, p)
    g, _ = _terciles(df["gqs_composite"].values)
    order = [b for b in ["Low", "Mid", "High", "All"] if b in set(g)]
    f1s, ns = [], []
    for b in order:
        m = (g == b)
        f1s.append(_macro_f1(y[m], p[m], thr))
        ns.append(int(m.sum()))
    fig, ax = _fig(3.4, 3.0)
    x = np.arange(len(order))
    ax.bar(x, f1s, width=0.6, color="#C0392B")
    for xi, (f, n) in enumerate(zip(f1s, ns)):
        ax.text(xi, f + 0.01, f"{f:.2f}\n(n={n})", ha="center",
                va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_xlabel("GQS-composite tercile"); ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    _save(fig, outdir / f"{ds}_gqs_stratified.pdf")
    return list(zip(order, f1s, ns)), thr


def fig_gate_entropy(df: pd.DataFrame, ds: str, outdir: Path) -> None:
    gates = df[GATE_COLS].values
    ent = _gate_entropy(gates)
    fig, ax = _fig(3.4, 3.0)
    ax.hist(ent, bins=30, color="#3B6FB6", edgecolor="white", linewidth=0.3)
    ax.axvline(ent.mean(), color="k", linestyle="--", lw=1.2,
               label=f"mean={ent.mean():.3f}")
    ax.set_xlabel("Normalised gate entropy"); ax.set_ylabel("Count")
    ax.set_xlim(0, 1); ax.legend(frameon=False)
    _save(fig, outdir / f"{ds}_gate_entropy.pdf")


def fig_ablation_bar(df: pd.DataFrame, ds: str, outdir: Path) -> None:
    y = df["true_label"].values.astype(int)
    vcols = _variant_cols(df)
    rows = []
    for c in vcols:
        exp = c[len("prob__"):]
        p = df[c].values
        m = ~np.isnan(p)
        if m.sum() < 2:
            continue
        thr, f1 = _best_thr(y[m], p[m])
        rows.append({"exp": exp, "f1": f1, "thr": thr})
    if not rows:
        return
    rows_df = pd.DataFrame(rows)
    # Sort by ablation order when possible
    def _order(exp):
        try:
            return ABLATION_ORDER.index(exp)
        except ValueError:
            return 99
    rows_df["ord"] = rows_df["exp"].map(_order)
    rows_df = rows_df.sort_values("ord").reset_index(drop=True)

    fig, ax = _fig(max(3.5, 1.0 + 0.7 * len(rows_df)), 3.0)
    x = np.arange(len(rows_df))
    colors = ["#E08A1E" if r["exp"] == DEPLOYED else "#3B6FB6"
              for _, r in rows_df.iterrows()]
    ax.bar(x, rows_df["f1"], width=0.6, color=colors)
    for xi, (_, r) in enumerate(rows_df.iterrows()):
        ax.text(xi, r["f1"] + 0.005, f"{r['f1']:.3f}", ha="center",
                va="bottom", fontsize=6.5)
    labels = [VARIANT_LABELS.get(e, e) for e in rows_df["exp"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6.5)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.05)
    legend_els = [Patch(color="#E08A1E", label="Deployed (QGF)"),
                  Patch(color="#3B6FB6", label="Other variant")]
    ax.legend(handles=legend_els, frameon=False, fontsize=6)
    _save(fig, outdir / f"{ds}_ablation_bar.pdf")


# --------------------------------------------------------------------------- #
# Cross-dataset ROC overlay
# --------------------------------------------------------------------------- #

def fig_all_roc_overlay(dfs: dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, ax = _fig(4.5, 3.5)
    for ds, df in dfs.items():
        y = df["true_label"].values.astype(int)
        col = _deployed_col(df)
        p = df[col].values
        m = ~np.isnan(p)
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue
        auc = roc_auc_score(y[m], p[m])
        fpr, tpr, _ = roc_curve(y[m], p[m])
        ax.plot(fpr, tpr, lw=2.0, color=DS_COLORS.get(ds, "k"),
                label=f"{ds.upper()} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, outdir / "all_roc_overlay.pdf")


# --------------------------------------------------------------------------- #
# Main per-dataset processor
# --------------------------------------------------------------------------- #

def process(ds: str, csv_path: Path, outdir: Path) -> None:
    df = pd.read_csv(str(csv_path))
    print(f"\n{'='*64}")
    print(f"DATASET: {ds.upper()}   N_test={len(df)}")
    print(f"{'='*64}")
    print(f"  Columns: {list(df.columns)}")

    y = df["true_label"].values.astype(int)

    # -- ROC
    aucs = fig_roc(df, ds, outdir)
    for exp, a in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print(f"  AUC  {VARIANT_LABELS.get(exp, exp):<20} = {a:.4f}")

    # -- PR
    aps = fig_pr(df, ds, outdir)
    for exp, a in sorted(aps.items(), key=lambda kv: -kv[1]):
        print(f"  AP   {VARIANT_LABELS.get(exp, exp):<20} = {a:.4f}")

    # -- Confusion matrix
    fig_confusion(df, ds, outdir)

    # -- Calibration + ECE
    ece = fig_calibration(df, ds, outdir)
    print(f"  ECE  = {ece:.4f}")

    # -- Score distributions
    fig_score_dist(df, ds, outdir)

    # -- Gate distributions
    if all(c in df.columns for c in GATE_COLS):
        fig_gate_dist(df, ds, outdir)
        fig_gate_by_quality(df, ds, outdir)
        fig_gate_vs_gqs(df, ds, outdir, "gqs_composite",
                        "GQS composite")
        for qc in GQS_COLS:
            if qc in df.columns:
                fig_gate_vs_gqs(df, ds, outdir, qc,
                                GQS_LABELS.get(qc, qc))
        fig_gate_entropy(df, ds, outdir)

    # -- GQS distribution
    if any(c in df.columns for c in GQS_COLS):
        fig_gqs_dist(df, ds, outdir)

    # -- GQS-stratified F1
    if "gqs_composite" in df.columns:
        strat, thr = fig_gqs_stratified(df, ds, outdir)
        print(f"  GQS-stratified F1 (thr={thr:.2f}):")
        for b, f, n in strat:
            print(f"    [{b:<4}] F1={f:.4f}  n={n}")

    # -- Ablation bar
    fig_ablation_bar(df, ds, outdir)

    # -- Overall deployed metrics (for paper table reconciliation)
    col = _deployed_col(df)
    p = df[col].values
    best_thr, best_f1 = _best_thr(y, p)
    try:
        best_auc = roc_auc_score(y, p)
    except ValueError:
        best_auc = float("nan")
    print(f"  DEPLOYED ({col[len('prob__'):]}):"
          f"  macro-F1={best_f1:.4f}  @thr={best_thr:.2f}  AUC={best_auc:.4f}")


def main(args: argparse.Namespace) -> None:
    dumps  = Path(args.dumps)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    found: dict[str, pd.DataFrame] = {}
    for ds in ("ntu", "rlvs", "rwf", "hf"):
        # New per-dataset layout: paper_data/{ds}/perclip.csv
        csv = dumps / ds / "perclip.csv"
        if not csv.exists():
            # Fallback: old flat layout: dumps/perclip_{ds}.csv
            csv = dumps / f"perclip_{ds}.csv"
        if csv.exists():
            found[ds] = csv
        else:
            print(f"[skip] {ds} perclip.csv not found")

    if not found:
        print(f"No perclip_*.csv under {dumps}. Run the dump scripts first.")
        return

    dfs: dict[str, pd.DataFrame] = {}
    for ds, csv in found.items():
        dfs[ds] = pd.read_csv(str(csv))
        process(ds, csv, outdir)

    if len(dfs) >= 2:
        print("\n[cross-dataset ROC overlay]")
        fig_all_roc_overlay(dfs, outdir)

    print(f"\nAll figures -> {outdir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot all paper figures from per-clip dumps.")
    ap.add_argument("--dumps",  default="../paper_data", help="paper_data/ root (contains {ds}/perclip.csv)")
    ap.add_argument("--outdir", default="../paper_data/out", help="folder for output PDFs")
    main(ap.parse_args())
