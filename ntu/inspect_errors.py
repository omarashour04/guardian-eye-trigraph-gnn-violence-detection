"""
inspect_errors.py
Guardian Eye - V9  |  FN/FP window inspection (LOCAL, no model needed)
Dataset : NTU CCTV-Fights

Pulls the most-confident false negatives and false positives from the
diagnostics error_analysis.csv, lines each clip window up against the
groundtruth fight segments of its source video, and:

  1. writes diagnostics/error_inspection.csv - one row per error clip with
     window times, GT segments, overlap fraction and gap to nearest fight
     boundary (so you can spot annotation-boundary artifacts numerically)
  2. with --export-clips, dumps each window as an .mp4 snippet (with a
     text overlay: segment / pred_prob / overlap) to
     diagnostics/error_clips/ for direct eyeballing

Usage (on WS)
-------------
  python inspect_errors.py                     # CSV report only, top 10 FN + 10 FP
  python inspect_errors.py --top 20            # top 20 of each
  python inspect_errors.py --export-clips      # also write mp4 snippets
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT      = Path(__file__).parent
DIAG_DIR  = ROOT / "train_output" / "runs_ntu" / "diagnostics"
SPLIT_CSV = ROOT / "preproc_output" / "split_ntu.csv"
GT_JSON   = ROOT / "groundtruth.json"
OUT_CSV   = DIAG_DIR / "error_inspection.csv"
CLIP_DIR  = DIAG_DIR / "error_clips"


def load_inputs():
    err = pd.read_csv(DIAG_DIR / "error_analysis.csv")
    split = pd.read_csv(SPLIT_CSV)
    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt = json.load(f)["database"]
    return err, split, gt


def window_vs_segments(start_sec, end_sec, segments):
    """Overlap fraction of the window covered by fight segments, plus the
    signed gap (sec) from the window to the nearest segment boundary.
    gap < 0 means the window intersects a segment; gap = 0 touches one."""
    win_len = end_sec - start_sec
    covered = 0.0
    min_gap = float("inf")
    for seg_start, seg_end in segments:
        inter = max(0.0, min(end_sec, seg_end) - max(start_sec, seg_start))
        covered += inter
        if inter > 0:
            gap = -inter  # negative gap encodes depth of intersection
        else:
            # distance between disjoint intervals
            gap = max(seg_start - end_sec, start_sec - seg_end)
        min_gap = min(min_gap, gap)
    # segments can overlap each other in GT; cap coverage at window length
    overlap_frac = min(covered, win_len) / win_len if win_len > 0 else 0.0
    return overlap_frac, min_gap


def build_report(err, split, gt, top_n):
    # most-confident errors first: lowest prob FNs, highest prob FPs
    fn = err[err.segment == "FN"].sort_values("pred_prob").head(top_n)
    fp = err[err.segment == "FP"].sort_values("pred_prob", ascending=False).head(top_n)
    sel = pd.concat([fn, fp]).merge(
        split[["clip_id", "video_id", "start_frame", "end_frame", "video_path"]],
        on="clip_id", how="left")

    rows = []
    for _, r in tqdm(sel.iterrows(), total=len(sel), unit="clip", desc="report"):
        vid = gt[r.video_id]
        fps = float(vid["frame_rate"])
        start_sec = r.start_frame / fps
        end_sec = r.end_frame / fps
        segments = [a["segment"] for a in vid["annotations"]]
        overlap_frac, gap = window_vs_segments(start_sec, end_sec, segments)
        rows.append({
            "clip_id": r.clip_id,
            "segment": r.segment,
            "source": r.source,
            "pred_prob": round(r.pred_prob, 4),
            "win_start_sec": round(start_sec, 2),
            "win_end_sec": round(end_sec, 2),
            "video_duration": vid["duration"],
            "n_gt_segments": len(segments),
            "gt_segments": "; ".join(f"[{s:.1f}-{e:.1f}]" for s, e in segments),
            "overlap_frac": round(overlap_frac, 3),
            "gap_to_nearest_seg": round(gap, 2),
            "video_path": r.video_path,
            "start_frame": int(r.start_frame),
            "end_frame": int(r.end_frame),
        })
    return pd.DataFrame(rows)


def export_clips(report, fps_fallback=30.0):
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    for _, r in tqdm(report.iterrows(), total=len(report), unit="clip", desc="export"):
        cap = cv2.VideoCapture(r.video_path)
        if not cap.isOpened():
            print(f"WARN cannot open {r.video_path}, skipping {r.clip_id}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = CLIP_DIR / f"{r.segment}_{r.clip_id}_p{r.pred_prob:.2f}.mp4"
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        overlay = (f"{r.segment} prob={r.pred_prob:.2f} "
                   f"ovl={r.overlap_frac:.2f} gap={r.gap_to_nearest_seg:.1f}s")
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.start_frame)
        for _ in range(int(r.end_frame - r.start_frame)):
            ok, frame = cap.read()
            if not ok:
                break
            # banner so you know what you are looking at while scrubbing
            cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(frame, overlay, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            writer.write(frame)
        writer.release()
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10,
                    help="number of most-confident FNs and FPs to inspect (each)")
    ap.add_argument("--export-clips", action="store_true",
                    help="also dump each window as an annotated mp4 snippet")
    args = ap.parse_args()

    err, split, gt = load_inputs()
    report = build_report(err, split, gt, args.top)
    report.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(report)} rows)")

    # quick console digest: are errors hugging GT boundaries?
    for seg in ("FN", "FP"):
        d = report[report.segment == seg]
        if len(d) == 0:
            continue
        near = (d.gap_to_nearest_seg.abs() <= 3.0).sum()
        print(f"{seg}: {len(d)} clips | median overlap={d.overlap_frac.median():.2f} "
              f"| within 3s of a GT boundary: {near}/{len(d)}")

    if args.export_clips:
        export_clips(report)
        print(f"clips written to {CLIP_DIR}")


if __name__ == "__main__":
    main()
