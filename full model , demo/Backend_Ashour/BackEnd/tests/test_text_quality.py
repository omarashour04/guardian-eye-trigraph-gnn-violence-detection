from __future__ import annotations

import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_quality import (
    cleanup_repetition,
    contains_significant_chinese,
    polish_generated_text,
    validate_generated_language,
)


class TextQualityTests(unittest.TestCase):
    def test_arabic_validation_rejects_chinese_characters(self) -> None:
        with self.assertRaises(ValueError):
            validate_generated_language("هذا ملخص عربي 你好", "ar", label="test")

    def test_arabic_validation_rejects_a_single_chinese_character(self) -> None:
        with self.assertRaises(ValueError):
            validate_generated_language("هذا ملخص عربي 中", "ar", label="test")

    def test_arabic_validation_can_temporarily_allow_chinese_characters(self) -> None:
        text = "\u0647\u0630\u0627 \u0646\u0635 \u0639\u0631\u0628\u064a \u6d4b\u8bd5"
        self.assertEqual(
            validate_generated_language(text, "ar", allow_chinese=True),
            text,
        )

    def test_arabic_validation_accepts_modern_standard_arabic(self) -> None:
        text = validate_generated_language(
            "إذا ثبتت المسؤولية من الجهات المختصة فهذا ملخص حذر.",
            "ar",
            label="test",
        )

        self.assertIn("المسؤولية", text)
        self.assertFalse(contains_significant_chinese(text))

    def test_strict_arabic_validation_rejects_english_sentences(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-Arabic words"):
            validate_generated_language(
                "الأشخاص المرصودون: Person 1 appears to be on the ground.",
                "ar",
                label="test",
                require_arabic_only=True,
            )

    def test_strict_arabic_validation_allows_guardian_eye_brand(self) -> None:
        text = validate_generated_language(
            "صنف Guardian Eye المقطع على أنه عنيف بثقة مرتفعة.",
            "ar",
            label="test",
            require_arabic_only=True,
        )

        self.assertIn("Guardian Eye", text)

    def test_strict_arabic_validation_allows_explicit_filename_and_model_id(self) -> None:
        text = validate_generated_language(
            "الملف fight_01.avi عولج بواسطة google/gemma-3-4b-it.",
            "ar",
            label="test",
            require_arabic_only=True,
            allowed_latin_terms=("fight_01.avi", "google/gemma-3-4b-it"),
        )

        self.assertIn("fight_01.avi", text)
        self.assertIn("google/gemma-3-4b-it", text)

    def test_repetition_cleanup_removes_adjacent_duplicates(self) -> None:
        self.assertEqual(
            cleanup_repetition("violent violent conduct conduct may may require review"),
            "violent conduct may require review",
        )

    def test_polish_generated_text_removes_markdown_artifacts(self) -> None:
        self.assertEqual(
            polish_generated_text("### Summary\n**Non-violent** clip\n- Calm movement.", "en"),
            "Summary Non-violent clip Calm movement.",
        )

    def test_polish_generated_text_removes_raw_narration_telemetry(self) -> None:
        clean = polish_generated_text(
            "Verdict: violence. Peak Window: Frames 11 to 14. "
            "Strongest Available Model Evidence: vit index 3. V9 classifier flagged it.",
            "en",
        )

        self.assertNotIn("Verdict:", clean)
        self.assertNotIn("Peak Window", clean)
        self.assertNotIn("Frames 11", clean)
        self.assertNotIn("vit index", clean.lower())
        self.assertNotIn("V9", clean)
        self.assertIn("Guardian Eye classifier", clean)

    def test_polish_generated_text_replaces_internal_vlm_wording(self) -> None:
        clean = polish_generated_text(
            "According to the VLM summary, the current_vlm_summary has summary_source data.",
            "en",
        )

        self.assertIn("Guardian Eye", clean)
        self.assertNotIn("VLM summary", clean)
        self.assertNotIn("current_vlm_summary", clean)
        self.assertNotIn("summary_source", clean)


if __name__ == "__main__":
    unittest.main()
