from __future__ import annotations

import gc
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


DEFAULT_MODEL_ID = str(
    Path(__file__).resolve().parents[1] / "models" / "translategemma-4b-it"
)
ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
LANGUAGE_CODES = {
    "Arabic": "ar",
    "English": "en",
}


def contains_arabic(text: str) -> bool:
    """Return whether text contains at least one Arabic-script character."""

    return bool(ARABIC_PATTERN.search(text))


class TranslateGemmaTranslator:
    """Request-scoped Arabic/English translator backed by TranslateGemma 4B."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load TranslateGemma only when an Arabic request needs it."""

        if self.is_loaded:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            local_files_only=True,
            trust_remote_code=True,
            device_map=os.getenv("TRANSLATEGEMMA_DEVICE_MAP", "auto"),
            dtype=torch.bfloat16,
        )

    def unload(self) -> None:
        """Release TranslateGemma after the Arabic request is complete."""

        self.model = None
        self.processor = None
        gc.collect()

        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        self.torch = None

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate one text value while preserving its meaning."""

        if not text.strip():
            return text

        self.load()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": LANGUAGE_CODES[source_language],
                        "target_lang_code": LANGUAGE_CODES[target_language],
                        "text": text,
                    }
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(
                    os.getenv("TRANSLATEGEMMA_MAX_NEW_TOKENS", "512")
                ),
                do_sample=False,
            )

        translated = self.processor.decode(
            generated[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        return translated


@contextmanager
def translation_session(
    translator: TranslateGemmaTranslator | None = None,
) -> Iterator[TranslateGemmaTranslator]:
    """Create one lazy translator and always release it after the request."""

    active_translator = translator or TranslateGemmaTranslator()
    try:
        yield active_translator
    finally:
        active_translator.unload()
