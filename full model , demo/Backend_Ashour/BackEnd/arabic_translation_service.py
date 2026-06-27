"""Local Qwen Arabic translation for Guardian Eye narration."""

from __future__ import annotations

import gc
import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from text_quality import contains_significant_chinese


DEFAULT_QWEN_MODEL = str(
    Path(__file__).resolve().parents[2] / "models" / "qwen2.5-1.5b-instruct"
)
_QWEN_TRANSLATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class TranslationConfig:
    provider: str
    model_path: str
    use_4bit: bool
    local_files_only: bool
    device: str
    dtype: str
    max_new_tokens: int


@dataclass(frozen=True)
class TranslationResult:
    text: str
    provider: str
    model_path: str
    attempts: int


def get_translation_config() -> TranslationConfig:
    model_path = os.getenv(
        "GUARDIAN_TRANSLATION_QWEN_MODEL_PATH",
        os.getenv("GUARDIAN_LLM_MODEL_ID", DEFAULT_QWEN_MODEL),
    ).strip()
    device = os.getenv("GUARDIAN_TRANSLATION_DEVICE", "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        device = "auto"
    dtype = os.getenv("GUARDIAN_TRANSLATION_DTYPE", "bfloat16").strip().lower()
    if dtype not in {"float16", "bfloat16", "float32"}:
        dtype = "float16"
    default_4bit = "0" if platform.system().lower() == "windows" else "1"
    try:
        max_new_tokens = int(os.getenv("GUARDIAN_TRANSLATION_MAX_NEW_TOKENS", "256"))
    except ValueError:
        max_new_tokens = 256
    return TranslationConfig(
        provider="qwen",
        model_path=model_path,
        use_4bit=os.getenv("GUARDIAN_TRANSLATION_4BIT", default_4bit) == "1",
        local_files_only=True,
        device=device,
        dtype=dtype,
        max_new_tokens=max(64, min(max_new_tokens, 384)),
    )


def log_translation_startup() -> None:
    config = get_translation_config()
    print(
        "[translation] "
        f"provider=qwen model_path={config.model_path} mode=ephemeral "
        f"device={config.device} dtype={config.dtype} 4bit={config.use_4bit} "
        f"max_new_tokens={config.max_new_tokens} local_files_only=True "
        f"path_exists={Path(config.model_path).is_dir()!r} translategemma=disabled"
    )
    log_translation_health()


def get_translation_health() -> dict[str, str]:
    config = get_translation_config()
    return {
        "translation_provider": "qwen",
        "translation_loaded": "no",
        "translation_load_failed": "no",
        "translation_device": config.device,
        "translation_dtype": config.dtype,
        "translation_4bit": "yes" if config.use_4bit else "no",
        "translation_isolation": "in_process_ephemeral",
        "translategemma_enabled": "no",
    }


def log_translation_health() -> None:
    health = get_translation_health()
    print(
        "[translation-health] "
        f"translation_provider={health['translation_provider']} "
        f"translation_loaded={health['translation_loaded']} "
        "translategemma_enabled=no"
    )


def unload_translation_model() -> None:
    """Compatibility no-op; Qwen translation is released after each request."""


def translate_messages_to_arabic(
    messages: list[dict[str, str]],
    *,
    validator: Callable[[str], str],
) -> TranslationResult:
    source_text = _translation_source_from_messages(messages)
    return translate_texts_to_arabic([source_text], validator=validator)[0]


def translate_texts_to_arabic(
    source_texts: list[str],
    *,
    validator: Callable[[str], str],
) -> list[TranslationResult]:
    """Translate a narration batch with local Qwen and release it afterward."""
    config = get_translation_config()
    if not source_texts or any(not str(text).strip() for text in source_texts):
        raise ValueError("translation batch contains empty source text")
    if not Path(config.model_path).is_dir():
        raise RuntimeError(f"local Qwen translation checkpoint is missing: {config.model_path}")

    errors: list[str] = []
    with _QWEN_TRANSLATION_LOCK:
        for attempt in (1, 2):
            try:
                print(
                    "[translation] provider=qwen "
                    f"attempt={attempt} items={len(source_texts)} "
                    f"model_path={config.model_path!r} local_files_only=True"
                )
                raw_texts = _generate_batch_with_ephemeral_qwen(
                    config.model_path,
                    source_texts,
                    use_4bit=config.use_4bit,
                    local_files_only=True,
                    device=config.device,
                    dtype=config.dtype,
                    max_new_tokens=config.max_new_tokens,
                )
                results: list[TranslationResult] = []
                for raw in raw_texts:
                    if contains_significant_chinese(raw):
                        raise ValueError("translation contains Chinese characters")
                    results.append(
                        TranslationResult(
                            validator(raw), "qwen", config.model_path, attempt
                        )
                    )
                print(f"[translation] completed provider=qwen attempt={attempt}")
                return results
            except Exception as exc:
                errors.append(f"attempt {attempt}: {exc!r}")
                print(f"[translation] rejected provider=qwen attempt={attempt} reason={exc!r}")
    raise RuntimeError("Local Qwen Arabic translation failed: " + " | ".join(errors))


def _generate_batch_with_ephemeral_qwen(
    model_path: str,
    source_texts: list[str],
    *,
    use_4bit: bool,
    local_files_only: bool,
    device: str,
    dtype: str,
    max_new_tokens: int,
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
            **_model_load_kwargs(use_4bit, device=device, dtype=dtype),
        )
        model.eval()
        return [
            _generate_with_loaded_model(
                model,
                tokenizer,
                _qwen_translation_messages(source),
                max_new_tokens=max_new_tokens,
            )
            for source in source_texts
        ]
    finally:
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _translation_source_from_messages(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            source = message["content"].strip()
            if source:
                return source
    raise ValueError("translation request has no user source text")


def _qwen_translation_messages(source_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Translate the supplied English text into fluent Modern Standard Arabic only. "
                "Preserve facts, numbers, filenames, and model IDs. Never output Chinese "
                "characters. Return only the translation."
            ),
        },
        {"role": "user", "content": source_text},
    ]


def _model_load_kwargs(use_4bit: bool, *, device: str, dtype: str) -> dict[str, Any]:
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GUARDIAN_TRANSLATION_DEVICE=cuda but CUDA is unavailable")
    device_map: Any = {"": 0} if device == "cuda" else ({"": "cpu"} if device == "cpu" else "auto")
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "dtype": dtype_map[dtype],
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype_map[dtype],
        )
    return kwargs


def _generate_with_loaded_model(
    model: Any,
    processor: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
) -> str:
    import torch

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _reset_translation_state_for_tests() -> None:
    """Retained for compatibility with older test helpers."""
