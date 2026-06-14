from __future__ import annotations

from typing import Any, Callable, Collection

from translation_service import (
    TranslateGemmaTranslator,
    contains_arabic,
    translation_session,
)


USER_FACING_TEXT_KEYS = frozenset(
    {
        "answer",
        "message",
        "narrative",
        "snippet",
        "summary",
        "text",
    }
)


def _translate_user_facing_text(
    value: Any,
    translator: TranslateGemmaTranslator,
    text_keys: Collection[str],
    current_key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _translate_user_facing_text(
                item,
                translator,
                text_keys,
                current_key=key,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _translate_user_facing_text(item, translator, text_keys)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _translate_user_facing_text(item, translator, text_keys)
            for item in value
        )

    if isinstance(value, str) and (
        current_key is None or current_key in text_keys
    ):
        return translator.translate(value, "English", "Arabic")

    return value


def run_bilingual_rag(
    query: str,
    rag_callable: Callable[..., Any],
    *args: Any,
    translator: TranslateGemmaTranslator | None = None,
    text_keys: Collection[str] = USER_FACING_TEXT_KEYS,
    **kwargs: Any,
) -> Any:
    """
    Run an English RAG callable with optional Arabic input/output translation.

    English requests call the RAG directly, without constructing or loading
    TranslateGemma. Arabic requests use one request-scoped translator.
    """

    if not contains_arabic(query):
        return rag_callable(query, *args, **kwargs)

    with translation_session(translator) as active_translator:
        english_query = active_translator.translate(
            query,
            "Arabic",
            "English",
        )
        english_result = rag_callable(english_query, *args, **kwargs)
        return _translate_user_facing_text(
            english_result,
            active_translator,
            text_keys,
        )
