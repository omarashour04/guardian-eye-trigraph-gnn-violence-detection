from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from overlay_service import overlay_status_from_paths


class OverlayStatusTests(unittest.TestCase):
    def test_reports_available_missing_and_placeholder_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / "stream.mp4"
            existing.write_bytes(b"demo")

            status = overlay_status_from_paths(
                {
                    "skeleton": str(existing),
                    "interaction": None,
                    "object": str(Path(temp) / "missing.mp4"),
                    "vit": str(existing),
                }
            )

            self.assertEqual(status["skeleton"], "available")
            self.assertEqual(status["interaction"], "missing")
            self.assertEqual(status["object"], "missing")
            self.assertEqual(status["vit"], "available")

            placeholder = overlay_status_from_paths(
                {"skeleton": str(existing)},
                fallback_placeholder=True,
            )
            self.assertEqual(placeholder["skeleton"], "fallback_placeholder")


if __name__ == "__main__":
    unittest.main()
