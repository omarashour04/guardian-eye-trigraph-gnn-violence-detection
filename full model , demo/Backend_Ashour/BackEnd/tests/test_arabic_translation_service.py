from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_translation_service import (
    DEFAULT_QWEN_MODEL,
    TranslationResult,
    get_translation_config,
    get_translation_health,
    translate_texts_to_arabic,
)
from narration_service import _translate_narration_to_arabic
from text_quality import contains_significant_chinese


def accept(text: str) -> str:
    return text


class ArabicTranslationServiceTests(unittest.TestCase):
    def test_qwen_is_default_and_translategemma_is_disabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GUARDIAN_TRANSLATION_MODEL": "translategemma",
                "GUARDIAN_TRANSLATION_MODEL_PATH": "google/translategemma-4b-it",
            },
        ):
            config = get_translation_config()
            health = get_translation_health()

        self.assertEqual(config.provider, "qwen")
        self.assertEqual(config.model_path, DEFAULT_QWEN_MODEL)
        self.assertTrue(config.local_files_only)
        self.assertEqual(health["translategemma_enabled"], "no")
        self.assertEqual(health["translation_isolation"], "in_process_ephemeral")

    @patch("arabic_translation_service.Path.is_dir", return_value=True)
    @patch("arabic_translation_service._generate_batch_with_ephemeral_qwen")
    def test_qwen_translates_batch_without_worker(self, generate, _is_dir) -> None:
        generate.return_value = ["المشهد.", "النشاط."]

        with patch("arabic_translation_service.subprocess", create=True) as subprocess_module:
            results = translate_texts_to_arabic(["Scene.", "Activity."], validator=accept)

        self.assertEqual([result.text for result in results], ["المشهد.", "النشاط."])
        self.assertTrue(all(result.provider == "qwen" for result in results))
        self.assertTrue(generate.call_args.kwargs["local_files_only"])
        self.assertFalse(subprocess_module.Popen.called)

    @patch("arabic_translation_service.Path.is_dir", return_value=True)
    @patch("arabic_translation_service._generate_batch_with_ephemeral_qwen")
    def test_chinese_guard_retries_then_accepts_arabic(self, generate, _is_dir) -> None:
        generate.side_effect = [["نص 中 غير صالح."], ["نص عربي صالح."]]

        result = translate_texts_to_arabic(["Valid text."], validator=accept)[0]

        self.assertEqual(result.attempts, 2)
        self.assertFalse(contains_significant_chinese(result.text))
        self.assertEqual(generate.call_count, 2)

    def test_narration_markers_stay_out_of_qwen_input_and_are_restored(self) -> None:
        seen: list[str] = []

        def fake_batch(texts, *, validator):
            seen.extend(texts)
            translations = ["غرفة.", "شخص واحد.", "وقوف قريب.", "المقطع غير عنيف."]
            return [
                TranslationResult(validator(text), "qwen", DEFAULT_QWEN_MODEL, 1)
                for text in translations
            ]

        with patch("arabic_translation_service.translate_texts_to_arabic", side_effect=fake_batch):
            translated = _translate_narration_to_arabic(
                "Scene: A room. People observed: One person. "
                "Observed activity: Standing nearby. Guardian Eye assessment: Non-violent."
            )

        self.assertTrue(all("[[GE_" not in text for text in seen))
        self.assertIn("المشهد:", translated)
        self.assertIn("الأشخاص المرصودون:", translated)
        self.assertIn("تقييم Guardian Eye:", translated)


if __name__ == "__main__":
    unittest.main()
