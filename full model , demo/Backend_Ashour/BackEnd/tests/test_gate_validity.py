from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_classifier import _modality_activity
from schemas import GateValidity, PredictResponse


class GateValidityTests(unittest.TestCase):
    def test_zero_filled_vit_is_marked_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            npz_path = Path(temp) / "sample.npz"
            np.savez(
                npz_path,
                skeleton=np.ones((32, 1, 17, 3), dtype=np.float32),
                int_nodes=np.ones((32, 1, 7), dtype=np.float32),
                int_edges=np.ones((32, 1, 1, 4), dtype=np.float32),
                obj_nodes=np.ones((32, 1, 6), dtype=np.float32),
                po_edges=np.ones((32, 1, 1, 5), dtype=np.float32),
                vit_embedding=np.zeros((768,), dtype=np.float32),
            )

            with np.load(npz_path, allow_pickle=False) as data:
                metadata = _modality_activity(
                    data,
                    {"skeleton": 0.3, "interaction": 0.4, "object": 0.1, "vit": 0.2},
                )

        self.assertIn("vit", metadata["inactive_modalities"])
        self.assertNotIn("vit", metadata["active_modalities"])
        self.assertEqual(metadata["gate_validity"]["status"], "partial")
        self.assertAlmostEqual(metadata["gate_validity"]["inactive_gate_sum"], 0.2)

    def test_missing_vit_embedding_is_marked_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            npz_path = Path(temp) / "sample.npz"
            np.savez(
                npz_path,
                skeleton=np.ones((32, 1, 17, 3), dtype=np.float32),
                int_nodes=np.ones((32, 1, 7), dtype=np.float32),
                int_edges=np.ones((32, 1, 1, 4), dtype=np.float32),
                obj_nodes=np.ones((32, 1, 6), dtype=np.float32),
                po_edges=np.ones((32, 1, 1, 5), dtype=np.float32),
            )

            with np.load(npz_path, allow_pickle=False) as data:
                metadata = _modality_activity(
                    data,
                    {"skeleton": 0.3, "interaction": 0.4, "object": 0.1, "vit": 0.2},
                )

        self.assertIn("vit", metadata["inactive_modalities"])
        self.assertNotIn("vit", metadata["active_modalities"])
        self.assertEqual(metadata["gate_validity"]["status"], "partial")

    def test_predict_response_defaults_keep_existing_constructors_compatible(self) -> None:
        payload = {
            "verdict": "violence",
            "confidence": 0.8,
            "threshold": 0.28,
            "gate": {"skeleton": 0.3, "interaction": 0.4, "object": 0.1, "vit": 0.2},
            "gqs": {"q_skel": 0.8, "q_int": 0.7, "q_obj": 0.6, "q_po": 0.5, "valid_ratio": 0.9},
            "telemetry": {
                "people": 2,
                "peak_window": [4, 7],
                "weapon": {"flag": False, "cls": None},
            },
            "clip_id": "compat.mp4",
        }

        pred = PredictResponse(**payload)

        self.assertEqual(pred.active_modalities, ["skeleton", "interaction", "object", "vit"])
        self.assertEqual(pred.inactive_modalities, [])
        self.assertIsInstance(pred.gate_validity, GateValidity)
        self.assertEqual(pred.gate_validity.status, "unknown")


if __name__ == "__main__":
    unittest.main()
