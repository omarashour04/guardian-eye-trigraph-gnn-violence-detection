from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
POSITIVE_LABELS = {"violence", "violent", "fight", "fighting", "positive", "pos"}
NEGATIVE_LABELS = {
    "non-violence",
    "nonviolence",
    "nonviolent",
    "normal",
    "negative",
    "neg",
    "calm",
}


def collect_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    )


def infer_label(path: Path) -> str | None:
    parts = [part.casefold().replace("_", "-") for part in path.parts]
    stem = path.stem.casefold().replace("_", "-")
    candidates = [stem, *parts]
    for candidate in candidates:
        tokens = set(candidate.replace("-", " ").split())
        if candidate in NEGATIVE_LABELS:
            return "non-violence"
        if candidate in POSITIVE_LABELS:
            return "violence"
        if candidate.startswith("non-") or candidate.startswith("no-viol"):
            return "non-violence"
        if tokens.intersection(NEGATIVE_LABELS):
            return "non-violence"
        if tokens.intersection(POSITIVE_LABELS):
            return "violence"
    return None


def prediction_row(video_path: Path, pred: Any, label: str | None) -> dict[str, Any]:
    telemetry = pred.telemetry
    return {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "label": label or "",
        "verdict": pred.verdict,
        "confidence": float(pred.confidence),
        "threshold": float(pred.threshold),
        "is_correct": _is_correct(pred.verdict, label),
        "peak_window_start": telemetry.peak_window[0] if telemetry.peak_window else "",
        "peak_window_end": telemetry.peak_window[1] if len(telemetry.peak_window) > 1 else "",
        "people": telemetry.people,
        "weapon_flag": telemetry.weapon.flag,
        "weapon_class": telemetry.weapon.cls or "",
        "gate_skeleton": pred.gate.skeleton,
        "gate_interaction": pred.gate.interaction,
        "gate_object": pred.gate.object,
        "gate_vit": pred.gate.vit,
        "gqs_q_skel": pred.gqs.q_skel,
        "gqs_q_int": pred.gqs.q_int,
        "gqs_q_obj": pred.gqs.q_obj,
        "gqs_q_po": pred.gqs.q_po,
        "gqs_valid_ratio": pred.gqs.valid_ratio,
    }


def build_report(rows: list[dict[str, Any]], *, input_dir: Path) -> dict[str, Any]:
    confidences = [float(row["confidence"]) for row in rows]
    by_verdict = {
        verdict: _confidence_stats(
            [float(row["confidence"]) for row in rows if row["verdict"] == verdict]
        )
        for verdict in ("violence", "non-violence")
    }
    labeled_rows = [row for row in rows if row["label"]]
    false_positives = [
        row for row in labeled_rows
        if row["label"] == "non-violence" and row["verdict"] == "violence"
    ]
    false_negatives = [
        row for row in labeled_rows
        if row["label"] == "violence" and row["verdict"] == "non-violence"
    ]
    suggestion = suggest_threshold(labeled_rows)
    return {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_dir": str(input_dir),
        "mock_mode": os.getenv("GUARDIAN_MOCK", "1") == "1",
        "total_videos": len(rows),
        "labeled_videos": len(labeled_rows),
        "current_thresholds": sorted({float(row["threshold"]) for row in rows}),
        "confidence_distribution": _confidence_stats(confidences),
        "confidence_by_verdict": by_verdict,
        "confusion_matrix": _confusion_matrix(labeled_rows),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives": _compact_error_rows(false_positives),
        "false_negatives": _compact_error_rows(false_negatives),
        "suggested_threshold": suggestion,
        "notes": [
            "This report does not change model weights or prediction behavior.",
            "Suggested thresholds are analysis output only; apply only after validation on a representative labeled set.",
            "Labels are optional and inferred from folder/file names unless a label CSV is provided in a future workflow.",
        ],
    }


def suggest_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row["label"] in {"violence", "non-violence"}]
    if not labeled:
        return {
            "status": "unavailable",
            "reason": "No labels were inferred; threshold quality cannot be evaluated.",
        }
    candidates = sorted({round(i / 100, 2) for i in range(5, 96)})
    scored = []
    for threshold in candidates:
        tp = fp = tn = fn = 0
        for row in labeled:
            predicted = "violence" if float(row["confidence"]) >= threshold else "non-violence"
            actual = row["label"]
            if predicted == "violence" and actual == "violence":
                tp += 1
            elif predicted == "violence" and actual == "non-violence":
                fp += 1
            elif predicted == "non-violence" and actual == "non-violence":
                tn += 1
            else:
                fn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-9)) if tp else 0.0
        balanced_accuracy = ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))) / 2
        scored.append(
            {
                "threshold": threshold,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "balanced_accuracy": round(balanced_accuracy, 6),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    best = max(scored, key=lambda item: (item["f1"], item["balanced_accuracy"], -item["fp"]))
    return {
        "status": "analysis_only",
        "recommended_candidate": best,
        "top_candidates": sorted(
            scored,
            key=lambda item: (-item["f1"], -item["balanced_accuracy"], item["fp"], item["threshold"]),
        )[:10],
    }


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "video_name",
        "video_path",
        "label",
        "verdict",
        "confidence",
        "threshold",
        "is_correct",
        "peak_window_start",
        "peak_window_end",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(report: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def run_calibration(input_dir: Path, *, output_dir: Path, limit: int | None = None) -> dict[str, Any]:
    videos = collect_videos(input_dir)
    if limit is not None:
        videos = videos[: max(limit, 0)]
    if os.getenv("GUARDIAN_MOCK", "1") != "1":
        from inference_classifier import load_v9_model

        load_v9_model()

    from model_service import run_predict

    rows: list[dict[str, Any]] = []
    for index, video_path in enumerate(videos, start=1):
        clip_id = video_path.name
        label = infer_label(video_path)
        print(f"[calibration] {index}/{len(videos)} {video_path} label={label or 'unknown'}")
        pred = run_predict(
            str(video_path),
            clip_id,
            force_reprocess=os.getenv("GUARDIAN_MOCK", "1") != "1",
            source=video_path.name,
        )
        row = prediction_row(video_path, pred, label)
        rows.append(row)
        print(
            "[calibration] result "
            f"name={row['video_name']} verdict={row['verdict']} "
            f"confidence={row['confidence']:.4f} threshold={row['threshold']:.4f} "
            f"peak_window=[{row['peak_window_start']}, {row['peak_window_end']}]"
        )

    report = build_report(rows, input_dir=input_dir)
    write_csv(rows, output_dir / "calibration_predictions.csv")
    write_json(report, output_dir / "calibration_report.json")
    return report


def _confidence_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "p10": round(_percentile(ordered, 0.10), 6),
        "p90": round(_percentile(ordered, 0.90), 6),
    }


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, int] | None:
    if not rows:
        return None
    matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        if row["label"] == "violence" and row["verdict"] == "violence":
            matrix["tp"] += 1
        elif row["label"] == "non-violence" and row["verdict"] == "violence":
            matrix["fp"] += 1
        elif row["label"] == "non-violence" and row["verdict"] == "non-violence":
            matrix["tn"] += 1
        elif row["label"] == "violence" and row["verdict"] == "non-violence":
            matrix["fn"] += 1
    return matrix


def _compact_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "video_name": row["video_name"],
            "confidence": row["confidence"],
            "threshold": row["threshold"],
            "peak_window": [row["peak_window_start"], row["peak_window_end"]],
            "people": row["people"],
            "weapon_flag": row["weapon_flag"],
            "weapon_class": row["weapon_class"],
        }
        for row in rows[:50]
    ]


def _is_correct(verdict: str, label: str | None) -> bool | str:
    if not label:
        return ""
    return verdict == label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Guardian Eye prediction calibration analysis on a folder of videos."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing test videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration_reports"),
        help="Folder for calibration_predictions.csv and calibration_report.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of videos to process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[calibration] input folder does not exist: {input_dir}", file=sys.stderr)
        return 2
    report = run_calibration(input_dir, output_dir=output_dir, limit=args.limit)
    print(
        "[calibration] wrote "
        f"{output_dir / 'calibration_predictions.csv'} and "
        f"{output_dir / 'calibration_report.json'}"
    )
    print(json.dumps(report["suggested_threshold"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
