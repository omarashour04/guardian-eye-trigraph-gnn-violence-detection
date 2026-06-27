from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.calibration_eval import infer_label, run_calibration, suggest_threshold


class CalibrationEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_mock = os.environ.get("GUARDIAN_MOCK")
        os.environ["GUARDIAN_MOCK"] = "1"

    def tearDown(self) -> None:
        if self.old_mock is None:
            os.environ.pop("GUARDIAN_MOCK", None)
        else:
            os.environ["GUARDIAN_MOCK"] = self.old_mock

    def test_infer_label_from_parent_folder(self) -> None:
        self.assertEqual(infer_label(Path("dataset/violence/sample.mp4")), "violence")
        self.assertEqual(infer_label(Path("dataset/non-violence/sample.mp4")), "non-violence")
        self.assertIsNone(infer_label(Path("dataset/unknown/sample.mp4")))

    def test_suggest_threshold_uses_labeled_rows_without_applying_change(self) -> None:
        rows = [
            {"label": "violence", "confidence": 0.91, "verdict": "violence"},
            {"label": "violence", "confidence": 0.78, "verdict": "violence"},
            {"label": "non-violence", "confidence": 0.42, "verdict": "violence"},
            {"label": "non-violence", "confidence": 0.12, "verdict": "non-violence"},
        ]

        suggestion = suggest_threshold(rows)

        self.assertEqual(suggestion["status"], "analysis_only")
        self.assertIn("recommended_candidate", suggestion)
        self.assertGreaterEqual(suggestion["recommended_candidate"]["threshold"], 0.0)
        self.assertLessEqual(suggestion["recommended_candidate"]["threshold"], 1.0)

    def test_run_calibration_outputs_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = root / "report"
            (input_dir / "violence").mkdir(parents=True)
            (input_dir / "non-violence").mkdir(parents=True)
            (input_dir / "violence" / "fight_a.mp4").write_bytes(b"demo")
            (input_dir / "non-violence" / "normal_a.mp4").write_bytes(b"demo")

            report = run_calibration(input_dir, output_dir=output_dir)

            csv_path = output_dir / "calibration_predictions.csv"
            json_path = output_dir / "calibration_report.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["total_videos"], 2)
            self.assertEqual(saved["labeled_videos"], 2)
            self.assertEqual(report["total_videos"], 2)
            self.assertIn("suggested_threshold", saved)


if __name__ == "__main__":
    unittest.main()
