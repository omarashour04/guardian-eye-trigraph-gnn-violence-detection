from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.inspect_npz_vit import inspect_npz_vit


class InspectNpzVitTests(unittest.TestCase):
    def test_reports_zero_filled_vit_as_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "zero.npz"
            np.savez_compressed(path, vit_embedding=np.zeros(768, dtype=np.float32))

            result = inspect_npz_vit(path)

            self.assertTrue(result["vit_embedding_exists"])
            self.assertEqual(result["shape"], [768])
            self.assertTrue(result["all_zero"])
            self.assertFalse(result["active"])

    def test_reports_nonzero_vit_as_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "active.npz"
            embedding = np.zeros(768, dtype=np.float32)
            embedding[10] = 0.25
            np.savez_compressed(path, vit_embedding=embedding)

            result = inspect_npz_vit(path)

            self.assertFalse(result["all_zero"])
            self.assertTrue(result["active"])
            self.assertGreater(result["std"], 0)


if __name__ == "__main__":
    unittest.main()
