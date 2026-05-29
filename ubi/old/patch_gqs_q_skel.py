"""
patch_gqs_q_skel.py
Guardian Eye — one-off patch to fix q_skel in all existing NPZ files
and rebuild the gqs_summary CSVs from the patched NPZs.

Bug: q_skel was computed as valid_skel / (T * M) instead of valid_skel / T.
     Values were compressed roughly 6× and never reached the full [0, 1] range.

Fix: For each NPZ, reload skeleton + int_node_mask, recompute q_skel as the
     fraction of *frames* (not person-slots) that have ≥ min_joints confident
     keypoints in at least one person, overwrite gqs[0] in-place, then rebuild
     the dataset's gqs_summary CSV from the patched NPZs so train.py reads the
     corrected values too.

Affects: ubi-fights-processed  and/or  rwf-2000-processed

Usage:
    modal run patch_gqs_q_skel.py::patch_ubi
    modal run patch_gqs_q_skel.py::patch_rwf
"""

from pathlib import Path
import modal

app = modal.App("guardian-eye-patch-gqs-qskel")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "pandas==2.2.2", "tqdm==4.66.4")
)

# Threshold used for "confident keypoint" — must match preprocess scripts.
KP_CONF_THR = 0.25
MIN_JOINTS  = 5


def _patch_volume(cache_dir: Path,
                  summary_csv: Path,
                  split_csv: Path,
                  vol) -> None:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    vol.reload()

    if not cache_dir.exists():
        raise RuntimeError(f"Cache dir not found: {cache_dir}")

    # Clean up any orphan .tmp.npz from prior aborted runs
    orphans = list(cache_dir.glob("*.tmp.npz"))
    if orphans:
        print(f"Cleaning up {len(orphans)} orphan .tmp.npz files from prior runs")
        for o in orphans:
            try:
                o.unlink()
            except Exception:
                pass

    # Snapshot file list before we start writing (glob is materialized eagerly)
    npz_files = sorted(p for p in cache_dir.glob("*.npz")
                       if not p.name.endswith(".tmp.npz"))
    print(f"Found {len(npz_files)} NPZ files in {cache_dir}")

    # ── Load split CSV (clip_id/video_id → split, label) for the summary rebuild
    split_df = None
    id_col   = None
    if split_csv.exists():
        split_df = pd.read_csv(str(split_csv))
        # UBI uses clip_id; RWF uses video_id. Auto-detect.
        for cand in ("clip_id", "video_id"):
            if cand in split_df.columns:
                id_col = cand
                break
        if id_col is None:
            print(f"  WARN: {split_csv.name} has no clip_id/video_id column; "
                  f"summary CSV split/label will be left blank.")
            split_df = None
        else:
            split_df = split_df.set_index(id_col)
    else:
        print(f"  WARN: {split_csv} not found; summary CSV split/label will be blank.")

    patched         = 0
    already_correct = 0
    errors          = 0
    summary_rows    = []

    for npz_path in tqdm(npz_files, desc="Patching q_skel", unit="file"):
        try:
            with np.load(str(npz_path), allow_pickle=True) as _src:
                d = {k: _src[k] for k in _src.files}

            skeleton      = d["skeleton"]       # [T, M, V, 3]
            int_node_mask = d["int_node_mask"]  # [T, M] bool
            gqs           = d["gqs"].copy()     # [5] float32

            # Read T and M from the actual array (don't assume constants)
            T_actual = skeleton.shape[0]
            M_actual = skeleton.shape[1]

            old_q_skel = float(gqs[0])

            # Correct formula: fraction of frames where ≥1 person has
            # ≥ MIN_JOINTS confident keypoints.
            #
            # Vectorised over the [T, M] grid:
            #   confident_kp_count[ti, pi] = #joints with conf > KP_CONF_THR
            confident_kp_count = (skeleton[..., 2] > KP_CONF_THR).sum(axis=-1)  # [T, M]
            person_ok          = (int_node_mask                                  # [T, M]
                                  & (confident_kp_count >= MIN_JOINTS))
            valid_frames       = int(person_ok.any(axis=1).sum())                # scalar
            new_q_skel         = valid_frames / T_actual

            # Look up clip metadata from the split CSV
            clip_id = npz_path.stem
            if split_df is not None and clip_id in split_df.index:
                row = split_df.loc[clip_id]
                # Handle duplicate index entries defensively
                if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                    row = row.iloc[0]
                split_val = str(row["split"]) if "split" in row else ""
                label_val = int(row["label"]) if "label" in row else -1
            else:
                split_val = ""
                label_val = -1

            summary_rows.append({
                "clip_id":     clip_id,
                "split":       split_val,
                "label":       label_val,
                "q_skel":      float(new_q_skel),
                "q_int":       float(gqs[1]),
                "q_obj":       float(gqs[2]),
                "q_po":        float(gqs[3]),
                "valid_ratio": float(gqs[4]),
            })

            # Skip the write if already correct (re-runs are idempotent)
            if abs(new_q_skel - old_q_skel) < 1e-5:
                already_correct += 1
                continue

            gqs[0]   = np.float32(new_q_skel)
            d["gqs"] = gqs

            # Atomic write: save to .tmp.npz then rename
            tmp = npz_path.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp), **d)
            tmp.replace(npz_path)   # atomic on POSIX; replaces destination on Windows too
            patched += 1

        except Exception as e:
            print(f"  ERROR {npz_path.name}: {e}")
            errors += 1

    # ── Rebuild gqs_summary CSV from the patched NPZs ─────────────────────
    if summary_rows:
        # Match the column name "video_id" used by RWF if its split CSV used that
        col_name = id_col if id_col == "video_id" else "clip_id"
        df = pd.DataFrame(summary_rows)
        if col_name == "video_id":
            df = df.rename(columns={"clip_id": "video_id"})
        df.to_csv(str(summary_csv), index=False)
        print(f"Rewrote {summary_csv} with {len(df)} rows")

    vol.commit()
    print(f"\nDone. Patched={patched}  Already-correct={already_correct}  Errors={errors}")


# ── UBI-Fights ────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,   # 60 min — 3,792 files at ~1.5 file/s; idempotent so re-runs skip already-patched files
    volumes={"/data/proc": modal.Volume.from_name(
        "ubi-fights-processed", create_if_missing=False)},
)
def patch_ubi() -> None:
    proc = Path("/data/proc")
    _patch_volume(
        cache_dir   = proc / "cache_v9",
        summary_csv = proc / "gqs_summary_ubi.csv",
        split_csv   = proc / "split_ubi.csv",
        vol         = modal.Volume.from_name("ubi-fights-processed"),
    )


# ── RWF-2000 ──────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=1800,
    volumes={"/data/proc": modal.Volume.from_name(
        "rwf-2000-processed", create_if_missing=False)},
)
def patch_rwf() -> None:
    proc = Path("/data/proc")
    _patch_volume(
        cache_dir   = proc / "cache_v9",
        summary_csv = proc / "gqs_summary_v9.csv",
        split_csv   = proc / "split_v9.csv",
        vol         = modal.Volume.from_name("rwf-2000-processed"),
    )


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    print("Run one of:")
    print("  modal run patch_gqs_q_skel.py::patch_ubi")
    print("  modal run patch_gqs_q_skel.py::patch_rwf")
