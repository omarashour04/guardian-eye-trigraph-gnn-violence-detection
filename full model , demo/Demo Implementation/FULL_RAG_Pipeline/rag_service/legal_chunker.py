"""Article-level chunking for cleaned legal sections.

This module converts cleaned violence-related legal sections into stable chunks
for a future retrieval store. It does not index, retrieve, summarize,
or call an LLM.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from rag_service.legal_cleaner import CleanedLegalSection, SUCCESS


DEFAULT_MAX_CHARS = 2400

_ENGLISH_MARKER = r"(?:Article|Section)\s+\d+[A-Za-z\-]*|s\.\s*\d+[A-Za-z\-]*"
_ARABIC_INDIC_DIGITS = "\u0660-\u0669"
_ARABIC_MARKER = (
    rf"(?:\u0627\u0644\u0645\u0627\u062f\u0629|\u0645\u0627\u062f\u0629)\s+"
    rf"[\d{_ARABIC_INDIC_DIGITS}]+"
    rf"|(?:\u0627\u0644\u0641\u0635\u0644|\u0627\u0644\u0628\u0627\u0628)"
    rf"(?:\s+[^\n:：]{{1,40}})?"
)
_ARTICLE_MARKER_RE = re.compile(
    rf"(?=(?:^|\n)\s*(?:{_ENGLISH_MARKER}|{_ARABIC_MARKER}))",
    re.IGNORECASE,
)
_ARTICLE_LINE_RE = re.compile(
    rf"^\s*(?P<number>{_ENGLISH_MARKER}|{_ARABIC_MARKER})"
    rf"(?:\s*[:：\-.]?\s*(?P<title>[^\n]{{0,160}}))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    country: str
    source_language: str
    law_title: str
    article_number: str | None
    section_title: str | None
    violence_category: str | None
    source_url: str
    official_source: bool
    text: str
    matched_keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_cleaned_section(
    section: CleanedLegalSection | dict[str, Any],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[LegalChunk]:
    """Create stable article/section chunks from one cleaned legal section."""

    source = _coerce_section(section)
    if source.get("cleaning_status") != SUCCESS:
        return []

    text = _normalize_text(source.get("cleaned_text") or "")
    if not text:
        return []

    raw_chunks = _split_by_article_markers(text)
    if not raw_chunks:
        raw_chunks = _fallback_paragraph_chunks(text, max_chars=max_chars)

    chunks: list[LegalChunk] = []
    for index, raw_chunk in enumerate(raw_chunks):
        normalized_chunk = _normalize_text(raw_chunk)
        if not normalized_chunk:
            continue

        article_number, section_title = _extract_article_metadata(normalized_chunk)
        chunk_id = _build_chunk_id(source, normalized_chunk, article_number, index)
        chunks.append(
            LegalChunk(
                chunk_id=chunk_id,
                country=source["country"],
                source_language=source["source_language"],
                law_title=source["law_title"],
                article_number=article_number,
                section_title=section_title,
                violence_category=source.get("violence_category"),
                source_url=source["source_url"],
                official_source=source["official_source"],
                text=normalized_chunk,
                matched_keywords=list(source.get("matched_keywords") or []),
            )
        )

    return chunks


def chunk_cleaned_sections(
    sections: list[CleanedLegalSection | dict[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[LegalChunk]:
    """Create stable chunks from multiple cleaned legal sections."""

    chunks: list[LegalChunk] = []
    for section in sections:
        chunks.extend(chunk_cleaned_section(section, max_chars=max_chars))
    return chunks


def _coerce_section(section: CleanedLegalSection | dict[str, Any]) -> dict[str, Any]:
    if isinstance(section, dict):
        return section
    if is_dataclass(section):
        return asdict(section)
    raise TypeError("chunk_cleaned_section expects a CleanedLegalSection or dict.")


def _split_by_article_markers(text: str) -> list[str]:
    starts = [match.start() for match in _ARTICLE_MARKER_RE.finditer(text)]
    if not starts:
        return []

    starts.append(len(text))
    chunks = []
    for index in range(len(starts) - 1):
        chunk = text[starts[index] : starts[index + 1]].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _fallback_paragraph_chunks(text: str, *, max_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        would_exceed = current and current_len + paragraph_len + 2 > max_chars
        if would_exceed:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = paragraph_len
        else:
            current.append(paragraph)
            current_len += paragraph_len + (2 if current_len else 0)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _extract_article_metadata(text: str) -> tuple[str | None, str | None]:
    first_line = text.split("\n", 1)[0].strip()
    match = _ARTICLE_LINE_RE.match(first_line)
    if not match:
        return None, None

    article_number = _normalize_text(match.group("number"))
    title = _normalize_text(match.group("title") or "")
    return article_number or None, title or None


def _build_chunk_id(
    source: dict[str, Any],
    text: str,
    article_number: str | None,
    index: int,
) -> str:
    stable_basis = "|".join(
        [
            source["country"],
            source["law_title"],
            source["source_url"],
            article_number or f"fallback-{index}",
            text,
        ]
    )
    digest = hashlib.sha256(stable_basis.encode("utf-8")).hexdigest()[:16]
    return f"legal-{digest}"


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
