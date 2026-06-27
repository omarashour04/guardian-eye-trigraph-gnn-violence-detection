from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_router import route_video
from model_service import run_predict


class DatasetRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_env = {
            name: os.environ.get(name)
            for name in (
                "GUARDIAN_MODEL_ROUTING_ENABLED",
                "GUARDIAN_FORCE_DATASET_ROUTE",
                "GUARDIAN_MODEL_ROOT",
                "GUARDIAN_V9_CKPT",
                "GUARDIAN_VIDEOMAE_CKPT",
            )
        }

    def tearDown(self) -> None:
        for name, value in self.old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_routing_disabled_returns_no_route(self) -> None:
        os.environ["GUARDIAN_MODEL_ROUTING_ENABLED"] = "0"
        os.environ["GUARDIAN_FORCE_DATASET_ROUTE"] = "RLVS"

        pred = run_predict("missing.mp4", "route-disabled.mp4")

        self.assertIsNone(pred.model_route)

    def test_forced_rlvs_route(self) -> None:
        route = self._forced_route("RLVS")

        self.assertEqual(route["selected_dataset"], "RLVS")
        self.assertEqual(route["route_source"], "forced")
        self.assertEqual(route["routing_confidence"], 1.0)

    def test_forced_hockey_route(self) -> None:
        route = self._forced_route("HF")

        self.assertEqual(route["selected_dataset"], "HF")
        self.assertEqual(route["route_source"], "forced")
        self.assertEqual(route["dataset_similarity"]["HF"], 1.0)

    def test_forced_ntu_route(self) -> None:
        route = self._forced_route("NTU")

        self.assertEqual(route["selected_dataset"], "NTU")
        self.assertEqual(route["route_source"], "forced")
        self.assertEqual(route["dataset_similarity"]["NTU"], 1.0)

    def test_missing_selected_checkpoint_falls_back_to_rlvs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models" / "v9").mkdir(parents=True)
            (root / "models" / "videomae").mkdir(parents=True)
            (root / "models" / "v9" / "RLVS_best.pt").write_bytes(b"stub")
            (root / "models" / "videomae" / "RLVS_videomae_best.pt").write_bytes(b"stub")
            os.environ["GUARDIAN_MODEL_ROOT"] = str(root)
            os.environ["GUARDIAN_MODEL_ROUTING_ENABLED"] = "1"
            os.environ["GUARDIAN_FORCE_DATASET_ROUTE"] = "HF"

            route = route_video("unused.mp4")

        self.assertIsNotNone(route)
        payload = route.model_dump()
        self.assertEqual(payload["selected_dataset"], "HF")
        self.assertTrue(payload["fallback_used"])
        self.assertEqual(payload["selected_v9_checkpoint"], "RLVS_best.pt")
        self.assertEqual(payload["selected_videomae_checkpoint"], "RLVS_videomae_best.pt")

    def test_predict_includes_routing_metadata_when_enabled(self) -> None:
        os.environ["GUARDIAN_MODEL_ROUTING_ENABLED"] = "1"
        os.environ["GUARDIAN_FORCE_DATASET_ROUTE"] = "NTU"

        pred = run_predict("missing.mp4", "forced-ntu.mp4")

        self.assertIsNotNone(pred.model_route)
        assert pred.model_route is not None
        self.assertEqual(pred.model_route.selected_dataset, "NTU")
        self.assertEqual(pred.model_route.route_source, "forced")

    def test_rule_router_scores_ice_like_video_as_hockey(self) -> None:
        os.environ["GUARDIAN_MODEL_ROUTING_ENABLED"] = "1"
        os.environ.pop("GUARDIAN_FORCE_DATASET_ROUTE", None)
        with tempfile.TemporaryDirectory() as temp:
            video_path = Path(temp) / "ice.avi"
            self._write_color_video(video_path, (235, 235, 235))
            route = route_video(str(video_path))

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.selected_dataset, "HF")

    def _forced_route(self, dataset: str) -> dict:
        os.environ["GUARDIAN_MODEL_ROUTING_ENABLED"] = "1"
        os.environ["GUARDIAN_FORCE_DATASET_ROUTE"] = dataset
        pred = run_predict("missing.mp4", f"forced-{dataset.lower()}.mp4")
        self.assertIsNotNone(pred.model_route)
        assert pred.model_route is not None
        return pred.model_route.model_dump()

    def _write_color_video(self, path: Path, bgr: tuple[int, int, int]) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"XVID"),
            8.0,
            (224, 224),
        )
        for _ in range(8):
            frame = np.full((224, 224, 3), bgr, dtype=np.uint8)
            writer.write(frame)
        writer.release()


if __name__ == "__main__":
    unittest.main()
