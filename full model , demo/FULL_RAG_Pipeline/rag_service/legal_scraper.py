"""Raw legal text extraction for the Legal Consequences RAG store.

This module fetches source bytes and extracts raw text from HTML or PDF inputs.
It does not clean, chunk, index, retrieve, summarize, orchestrate, or call an
LLM.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from io import BytesIO
from typing import Any

import requests

from rag_service.legal_sources import LEGAL_SOURCE_REGISTRY


DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRIES = 2


@dataclass(frozen=True)
class RawLegalDocument:
    country: str
    source_url: str
    law_title: str
    source_language: str
    source_type: str
    official_source: bool
    raw_text: str
    extraction_method: str
    extraction_status: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _TextOnlyHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return _normalize_whitespace(" ".join(self._parts))


def fetch_source_content(
    source_url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    session: Any | None = None,
) -> bytes:
    """Fetch source bytes with timeout and minimal retry handling."""

    client = session or requests.Session()
    attempts = max(1, retries + 1)
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            response = client.get(source_url, timeout=timeout)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            return response.content
        except Exception as exc:
            last_error = exc

    if last_error is None:
        raise RuntimeError("Failed to fetch source content.")
    raise last_error


def extract_html_text(content: bytes | str) -> tuple[str, str]:
    """Extract raw text from HTML using optional libraries, then stdlib fallback."""

    html = _decode_content(content)

    trafilatura_text = _extract_html_with_trafilatura(html)
    if trafilatura_text:
        return trafilatura_text, "html_trafilatura"

    bs4_text = _extract_html_with_bs4(html)
    if bs4_text:
        return bs4_text, "html_beautifulsoup"

    parser = _TextOnlyHTMLParser()
    parser.feed(html)
    text = parser.get_text()
    if text:
        return text, "html_stdlib"

    raise ValueError("HTML extraction produced no text.")


def extract_pdf_text(content: bytes) -> tuple[str, str]:
    """Extract raw text from PDF bytes using optional libraries and a fixture fallback."""

    pymupdf_text = _extract_pdf_with_pymupdf(content)
    if pymupdf_text:
        return pymupdf_text, "pdf_pymupdf"

    pypdf_text = _extract_pdf_with_pypdf(content)
    if pypdf_text:
        return pypdf_text, "pdf_pypdf"

    literal_text = _extract_pdf_literal_strings(content)
    if literal_text:
        return literal_text, "pdf_literal_fallback"

    raise ValueError("PDF extraction produced no text.")


def scrape_legal_source(
    source: dict[str, Any],
    *,
    content: bytes | str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    session: Any | None = None,
) -> RawLegalDocument:
    """Fetch and extract one source, returning a failed document on errors."""

    metadata = _source_metadata(source)

    try:
        payload = (
            content
            if content is not None
            else fetch_source_content(
                metadata["source_url"],
                timeout=timeout,
                retries=retries,
                session=session,
            )
        )

        source_type = metadata["source_type"].lower()
        if source_type == "html":
            raw_text, method = extract_html_text(payload)
        elif source_type == "pdf":
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            raw_text, method = extract_pdf_text(payload)
        else:
            raise ValueError(f"Unsupported legal source_type: {source_type}")

        return RawLegalDocument(
            **metadata,
            raw_text=raw_text,
            extraction_method=method,
            extraction_status="success",
            error_message=None,
        )
    except Exception as exc:
        return RawLegalDocument(
            **metadata,
            raw_text="",
            extraction_method="none",
            extraction_status="failed",
            error_message=str(exc),
        )


def scrape_legal_sources(
    sources: list[dict[str, Any]] | None = None,
    *,
    content: bytes | str | dict[str, bytes | str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    session: Any | None = None,
) -> list[RawLegalDocument]:
    """Scrape a list of legal sources, preserving failures per source."""

    selected_sources = sources if sources is not None else _flatten_registry()
    return [
        scrape_legal_source(
            source,
            content=_content_for_source(content, source),
            timeout=timeout,
            retries=retries,
            session=session,
        )
        for source in selected_sources
    ]


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "country": source["country"],
        "source_url": source["source_url"],
        "law_title": source["law_title"],
        "source_language": source["source_language"],
        "source_type": source["source_type"],
        "official_source": source["official_source"],
    }


def _flatten_registry() -> list[dict[str, Any]]:
    return [
        source
        for country_sources in LEGAL_SOURCE_REGISTRY.values()
        for source in country_sources
    ]


def _content_for_source(
    content: bytes | str | dict[str, bytes | str] | None,
    source: dict[str, Any],
) -> bytes | str | None:
    if not isinstance(content, dict):
        return content
    return content.get(source["source_url"])


def _decode_content(content: bytes | str) -> str:
    if isinstance(content, str):
        return content
    return content.decode("utf-8", errors="replace")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_html_with_trafilatura(html: str) -> str:
    try:
        import trafilatura
    except ImportError:
        return ""

    extracted = trafilatura.extract(html)
    return _normalize_whitespace(extracted or "")


def _extract_html_with_bs4(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return _normalize_whitespace(soup.get_text(" "))
    except Exception:
        return ""


def _extract_pdf_with_pymupdf(content: bytes) -> str:
    try:
        import fitz
    except ImportError:
        return ""

    try:
        parts: list[str] = []
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                parts.append(page.get_text())
        return _normalize_whitespace(" ".join(parts))
    except Exception:
        return ""


def _extract_pdf_with_pypdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(BytesIO(content))
        parts = [page.extract_text() or "" for page in reader.pages]
        return _normalize_whitespace(" ".join(parts))
    except Exception:
        return ""


def _extract_pdf_literal_strings(content: bytes) -> str:
    decoded = content.decode("latin-1", errors="ignore")
    candidates = re.findall(r"\(([^()]*)\)\s*Tj", decoded)
    candidates.extend(re.findall(r"\(([^()]*)\)", decoded))
    text = " ".join(_unescape_pdf_literal(value) for value in candidates)
    return _normalize_whitespace(text)


def _unescape_pdf_literal(value: str) -> str:
    return (
        value.replace(r"\(", "(")
        .replace(r"\)", ")")
        .replace(r"\\", "\\")
        .replace(r"\n", " ")
        .replace(r"\r", " ")
        .replace(r"\t", " ")
    )
