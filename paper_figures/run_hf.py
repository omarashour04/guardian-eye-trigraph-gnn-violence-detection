"""
run_hf.py  —  WS-ONLY script. Handles Hockey Fights (HF).

Run on the WORKSTATION:
    python run_hf.py

Does:
  1. HF inference dump  -> paper_data/hf/perclip.csv
  2. HF logs + checkpoints -> paper_data/hf/ + paper_data/checkpoints/hf/

Transfer fresh/paper_data/ to the laptop when done.
Then run plot_figures.py on the laptop to generate figures.
"""

from __future__ import annotations

import importlib
import inspect as _inspect
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Repo root
# --------------------------------------------------------------------------- #

def _find_repo_root() -> Path:
    for start in [Path(__file__).resolve().parent, Path.cwd()]:
        for d in [start, *start.parents]:
            if (d / "hf").is_dir() and (d / "ntu").is_dir():
                return d
    raise FileNotFoundError("Cannot find repo root. Run from inside the fresh/ tree.")

_REPO = _find_repo_root()

# --------------------------------------------------------------------------- #
# Dataset wiring
# --------------------------------------------------------------------------- #

SPEC = {
    "dir":          _REPO / "hf",
    "train_mod":    "trigraph_v9_hf_train",
    "dataset_cls":  "HFDataset",
    "id_col":       "clip_id",
    "ckpt_name":    "best.pt",
    "output_dir":   Path(r"C:\Violence detection\ashour\fresh\HF\train_output"),
    "run_subdir":   "runs_hf",
    "finetune_dir": Path(r"C:\Violence detection\ashour\fresh\HF\finetune_output"),
    "cmp_csv":      "experiment_comparison_hf.csv",
}

EXP_CFG = {
    "A_skeleton_only":  (("skeleton",),                                    "qgf", {}),
    "B_skel_interaction": (("skeleton", "interaction"),                    "qgf", {}),
    "D_skel_int_obj":   (("skeleton", "interaction", "object"),            "qgf", {}),
    "E_full_lw":        (("skeleton", "interaction", "object", "vit"),     "lw",  {}),
    "E_full_qgf":       (("skeleton", "interaction", "object", "vit"),     "qgf", {}),
    "E_full_qgf_fixed": (("skeleton", "interaction", "object", "vit"),     "qgf",
                         {"entropy_weight": 0.10, "temp_anneal": True}),
}

GATE_COLS = ["w_skel", "w_int", "w_obj", "w_vit"]
GQS_COLS  = ["q_skel", "q_int", "q_obj", "q_po", "q_valid"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cp(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        print(f"  cp  {src.name}")
        return True
    print(f"  --  (missing) {src.name}")
    return False


def _import_train(spec: dict):
    sys.path.insert(0, str(spec["dir"]))
    return importlib.import_module(spec["train_mod"])


def _build_model(TRAIN, exp_id: str, device):
    active, fusion, extra = EXP_CFG[exp_id]
    cfg = TRAIN.cfg
    # Read graph dims from a sample NPZ (same as train script does at line 750-753)
    sample = next(TRAIN.CACHE_DIR.glob("*.npz"), None)
    if sample is None:
        raise RuntimeError(f"No NPZs in {TRAIN.CACHE_DIR}")
    s0 = np.load(str(sample))
    int_nd = int(s0["int_nodes"].shape[-1])   # 7
    int_ed = int(s0["int_edges"].shape[-1])   # 4
    obj_nd = int(s0["obj_nodes"].shape[-1])   # 6
    po_ed  = int(s0["po_edges"].shape[-1])    # 5
    return TRAIN.V9Model(
        active_streams=active, fusion_mode=fusion,
        C=cfg.C, T=cfg.T, V=cfg.V, M=cfg.M, N=cfg.N,
        int_nd=int_nd, int_ed=int_ed,
        obj_nd=obj_nd, po_ed=po_ed,
        vit_dim=cfg.vit_embed_dim,
        hidden=cfg.graph_hidden, embed_dim=cfg.embed_dim,
        dropout=cfg.dropout, stream_dropout=cfg.stream_dropout,
        n_heads_gat=cfg.n_heads_gat, n_heads_trf=cfg.n_heads_trf,
        n_layers_trf=cfg.n_layers_trf, **extra,
    ).to(device)


def _model_supports_return_gates(model) -> bool:
    try:
        return "return_gates" in _inspect.signature(model.forward).parameters
    except Exception:
        return False


def _make_gate_hook():
    captured = []
    def hook(module, inputs, output):
        streams, gqs = inputs[0], inputs[1]
        with torch.no_grad():
            if hasattr(module, "gate"):
                gl = module.gate(gqs[:, :module.n])
                temp = getattr(module, "current_temp", torch.tensor(1.0))
                gates = F.softmax(gl / temp, dim=-1)
            elif hasattr(module, "weights"):
                gates = F.softmax(module.weights, dim=0
                                  ).unsqueeze(0).expand(streams[0].shape[0], -1)
            else:
                return
            captured.append(gates.detach().cpu())
    return hook, captured


@torch.no_grad()
def _run_variant(model, loader, device, want_gates: bool):
    supports = _model_supports_return_gates(model)
    hook_handle, hook_storage = None, []
    if want_gates and not supports and hasattr(model, "fusion"):
        hook_fn, hook_storage = _make_gate_hook()
        hook_handle = model.fusion.register_forward_hook(hook_fn)

    out = {}
    hook_idx = 0
    try:
        for batch in loader:
            bdev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
            if want_gates and supports:
                logits, gates_t = model(bdev, return_gates=True)
                g = gates_t.cpu().numpy() if gates_t is not None else None
            else:
                logits = model(bdev)
                g = hook_storage[hook_idx].numpy() \
                    if (want_gates and hook_idx < len(hook_storage)) else None
                if want_gates:
                    hook_idx += 1

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            gqs   = bdev["gqs"].cpu().numpy()
            labs  = bdev["label"].cpu().numpy().astype(int).reshape(-1)
            cids  = batch["clip_id"]
            srcs  = batch.get("source", ["?"] * len(cids))
            for i, cid in enumerate(cids):
                rec = out.setdefault(cid, {})
                rec["prob"] = float(probs[i])
                if want_gates:
                    rec["true_label"] = int(labs[i])
                    rec["source"] = str(srcs[i]) if not isinstance(srcs, str) else srcs
                    for j, qc in enumerate(GQS_COLS):
                        rec[qc] = float(gqs[i, j])
                    for j, wc in enumerate(GATE_COLS):
                        rec[wc] = float(g[i, j]) if (g is not None and j < g.shape[1]) \
                                  else float("nan")
    finally:
        if hook_handle:
            hook_handle.remove()
    return out


# --------------------------------------------------------------------------- #
# Inference dump
# --------------------------------------------------------------------------- #

def run_dump(spec: dict, out_csv: Path) -> None:
    if out_csv.exists():
        print(f"  [skip] {out_csv.name} already exists")
        return

    device = _device()
    TRAIN  = _import_train(spec)
    TRAIN.seed_everything(getattr(TRAIN.cfg, "seed", 42))
    DatasetCls = getattr(TRAIN, spec["dataset_cls"])
    collate    = getattr(TRAIN, "collate_v9", None)

    split_df = pd.read_csv(str(TRAIN.SPLIT_CSV))
    sub      = split_df[split_df["split"] == "test"]
    ids      = [str(c) for c in sub[spec["id_col"]]]
    paths    = [TRAIN.CACHE_DIR / f"{c}.npz" for c in ids]
    labels   = [int(l) for l in sub["label"]]
    keep     = [(i, p, l) for i, p, l in zip(ids, paths, labels) if p.exists()]
    if not keep:
        raise RuntimeError(f"No test NPZs under {TRAIN.CACHE_DIR}")
    ids, paths, labels = map(list, zip(*keep))
    print(f"  test clips: {len(ids)}")

    loader = DataLoader(DatasetCls(paths, labels), batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn=collate)

    run_dir = spec["output_dir"] / spec["run_subdir"]
    variants = {eid: run_dir / eid / spec["ckpt_name"]
                for eid in EXP_CFG
                if (run_dir / eid / spec["ckpt_name"]).exists()}
    if not variants:
        raise RuntimeError(f"No checkpoints under {run_dir}")
    print(f"  variants: {list(variants)}")

    base_var  = "E_full_qgf" if "E_full_qgf" in variants else next(iter(variants))
    per_prob  = {}
    base_rows = None

    for exp_id, ckpt in variants.items():
        model = _build_model(TRAIN, exp_id, device)
        ck    = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        print(f"  {exp_id}: ep={ck.get('epoch','?')}")
        res = _run_variant(model, loader, device, want_gates=(exp_id == base_var))
        per_prob[exp_id] = {c: r["prob"] for c, r in res.items()}
        if exp_id == base_var:
            base_rows = res
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for cid, rec in base_rows.items():
        row = {"clip_id": cid, "source": rec.get("source", "?"),
               "true_label": rec["true_label"]}
        for qc in GQS_COLS:
            row[qc] = rec[qc]
        row["gqs_composite"] = 0.4*rec["q_skel"] + 0.4*rec["q_int"] + 0.2*rec["q_po"]
        for wc in GATE_COLS:
            row[wc] = rec[wc]
        for eid, pp in per_prob.items():
            row[f"prob__{eid}"] = pp.get(cid, float("nan"))
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("clip_id").reset_index(drop=True).to_csv(
        str(out_csv), index=False)
    print(f"  -> {out_csv.name}  ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# Log + checkpoint collector
# --------------------------------------------------------------------------- #

def collect_hf(spec: dict, out_dir: Path, ckpt_dir: Path) -> None:
    ft      = spec["finetune_dir"]
    run_dir = spec["output_dir"] / spec["run_subdir"]

    _cp(ft / "videomae_train_log.csv",     out_dir / "videomae_train_log.csv")
    _cp(ft / "videomae_test_results.json", out_dir / "videomae_test_results.json")
    _cp(spec["output_dir"] / spec["cmp_csv"], out_dir / "experiment_comparison.csv")

    if not run_dir.is_dir():
        print(f"  -- run dir missing: {run_dir}")
        return

    for exp_dir in sorted(run_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        _cp(exp_dir / "train_history.csv",   out_dir / f"{name}_train_history.csv")
        _cp(exp_dir / "test_metrics.json",   out_dir / f"{name}_test_metrics.json")
        _cp(exp_dir / "gqs_quartile_f1.csv", out_dir / f"{name}_gqs_quartile_f1.csv")
        dst = ckpt_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        _cp(exp_dir / spec["ckpt_name"], dst / "best.pt")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    root     = _REPO / "paper_data"
    hf_dir   = root / "hf"
    ckpt_dir = root / "checkpoints" / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Output: {hf_dir}")
    print("=" * 60)

    print(f"\n{'─'*40}\n[HF] dump\n{'─'*40}")
    run_dump(SPEC, hf_dir / "perclip.csv")

    print(f"\n[HF] logs + checkpoints")
    collect_hf(SPEC, hf_dir, ckpt_dir)

    print("\n" + "=" * 60)
    print("DONE (Hockey Fights)")
    print("Transfer fresh/paper_data/hf/ to the laptop.")
    print("Then on the laptop run:  python plot_figures.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
