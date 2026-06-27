"""Deterministic cleaner and violence filter for raw legal documents.

This module normalizes raw legal text and keeps only violence-related legal
content using keyword/category matching. It does not chunk for retrieval,
embed, index, summarize, orchestrate, or call an LLM.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from rag_service.legal_scraper import RawLegalDocument


NO_RELEVANT_CONTENT = "no_relevant_content"
SUCCESS = "success"
FAILED = "failed"


VIOLENCE_KEYWORDS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "assault": (
        "assault",
        "attack",
        "attacks",
        "attacked",
        "اعتداء",
        "يعتدي",
        "اعتدى",
        "هجوم",
    ),
    "battery_or_physical_harm": (
        "battery",
        "bodily harm",
        "physical harm",
        "actual bodily harm",
        "grievous bodily harm",
        "wound",
        "wounding",
        "injury",
        "injure",
        "force",
        "ضرب",
        "إيذاء",
        "ايذاء",
        "أذى",
        "اذى",
        "ضرر جسدي",
        "أذى جسدي",
        "إصابة",
        "اصابة",
        "جرح",
    ),
    "weapon_or_dangerous_object": (
        "weapon",
        "dangerous weapon",
        "dangerous object",
        "offensive weapon",
        "knife",
        "bottle",
        "firearm",
        "سلاح",
        "أداة خطرة",
        "اداة خطرة",
        "أداة حادة",
        "اداة حادة",
        "سكين",
        "زجاجة",
    ),
    "threat_or_intimidation": (
        "threat",
        "threaten",
        "threatens",
        "threatening",
        "intimidation",
        "intimidate",
        "menace",
        "تهديد",
        "يهدد",
        "هدد",
        "تخويف",
        "ترويع",
        "إرهاب",
        "ارهاب",
    ),
    "public_order_or_affray": (
        "affray",
        "riot",
        "violent disorder",
        "public order",
        "breach of the peace",
        "شغب",
        "مشاجرة",
        "النظام العام",
        "إخلال بالنظام",
        "اخلال بالنظام",
        "تجمهر",
    ),
    "child_or_family_protection": (
        "child abuse",
        "domestic violence",
        "family protection",
        "abuse",
        "cruelty",
        "طفل",
        "الأطفال",
        "الاطفال",
        "أسرة",
        "اسرة",
        "عنف أسري",
        "عنف اسري",
        "حماية من الإيذاء",
        "حماية من الايذاء",
    ),
    "general_violence": (
        "violence",
        "violent",
        "harm",
        "violent conduct",
        "عنف",
        "العنف",
        "إيذاء",
        "ايذاء",
    ),
}


@dataclass(frozen=True)
class CleanedLegalSection:
    country: str
    source_url: str
    law_title: str
    source_language: str
    source_type: str
    official_source: bool
    cleaned_text: str
    matched_keywords: list[str]
    violence_category: str | None
    cleaning_status: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_legal_document(
    document: RawLegalDocument | dict[str, Any],
) -> list[CleanedLegalSection]:
    """Normalize and retain violence-related content from one raw document."""

    source = _coerce_document(document)
    metadata = _metadata(source)

    if source.get("extraction_status") == FAILED:
        return [
            CleanedLegalSection(
                **metadata,
                cleaned_text="",
                matched_keywords=[],
                violence_category=None,
                cleaning_status=FAILED,
                error_message=source.get("error_message") or "Raw extraction failed.",
            )
        ]

    try:
        normalized_text = normalize_legal_text(source.get("raw_text") or "")
        if not normalized_text:
            return [_empty_result(metadata)]

        retained_sections: list[CleanedLegalSection] = []
        for section in _split_candidate_sections(normalized_text):
            matches_by_category = _match_violence_keywords(section)
            if not matches_by_category:
                continue

            category = _choose_category(matches_by_category)
            matched_keywords = sorted(matches_by_category[category])
            retained_sections.append(
                CleanedLegalSection(
                    **metadata,
                    cleaned_text=section,
                    matched_keywords=matched_keywords,
                    violence_category=category,
                    cleaning_status=SUCCESS,
                    error_message=None,
                )
            )

        if not retained_sections:
            return [_empty_result(metadata)]
        return retained_sections
    except Exception as exc:
        return [
            CleanedLegalSection(
                **metadata,
                cleaned_text="",
                matched_keywords=[],
                violence_category=None,
                cleaning_status=FAILED,
                error_message=str(exc),
            )
        ]


def clean_legal_documents(
    documents: list[RawLegalDocument | dict[str, Any]],
) -> list[CleanedLegalSection]:
    """Normalize and filter multiple raw legal documents."""

    cleaned: list[CleanedLegalSection] = []
    for document in documents:
        cleaned.extend(clean_legal_document(document))
    return cleaned


def normalize_legal_text(text: str) -> str:
    """Normalize whitespace and obvious repeated headers/footers."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line(line) for line in text.split("\n")]
    lines = [line for line in lines if line]
    lines = _remove_obvious_repeated_headers_footers(lines)

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _coerce_document(document: RawLegalDocument | dict[str, Any]) -> dict[str, Any]:
    if isinstance(document, dict):
        return document
    if is_dataclass(document):
        return asdict(document)
    raise TypeError("clean_legal_document expects a RawLegalDocument or dict.")


def _metadata(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "country": source["country"],
        "source_url": source["source_url"],
        "law_title": source["law_title"],
        "source_language": source["source_language"],
        "source_type": source["source_type"],
        "official_source": source["official_source"],
    }


def _empty_result(metadata: dict[str, Any]) -> CleanedLegalSection:
    return CleanedLegalSection(
        **metadata,
        cleaned_text="",
        matched_keywords=[],
        violence_category=None,
        cleaning_status=NO_RELEVANT_CONTENT,
        error_message=None,
    )


def _normalize_line(line: str) -> str:
    return re.sub(r"[ \t\f\v]+", " ", line).strip()


def _remove_obvious_repeated_headers_footers(lines: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for line in lines:
        if _can_be_repeated_header_or_footer(line):
            counts[line] = counts.get(line, 0) + 1

    repeated = {line for line, count in counts.items() if count >= 3}
    if not repeated:
        return lines
    return [line for line in lines if line not in repeated]


def _can_be_repeated_header_or_footer(line: str) -> bool:
    if len(line) > 120:
        return False
    if _starts_with_article_marker(line):
        return False
    return True


def _split_candidate_sections(text: str) -> list[str]:
    article_pattern = re.compile(
        r"(?=(?:^|\n)(?:"
        r"(?:Article|Section|Clause)\s+\d+[A-Za-z\-]*"
        r"|(?:المادة|مادة)\s+[^\n:：]{1,30}"
        r"))",
        re.IGNORECASE,
    )
    starts = [match.start() for match in article_pattern.finditer(text)]
    if starts:
        starts.append(len(text))
        sections = [
            text[starts[index] : starts[index + 1]].strip()
            for index in range(len(starts) - 1)
        ]
    else:
        sections = re.split(r"\n{2,}", text)

    return [_normalize_section(section) for section in sections if section.strip()]


def _normalize_section(section: str) -> str:
    lines = [_normalize_line(line) for line in section.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _starts_with_article_marker(line: str) -> bool:
    return bool(
        re.match(
            r"^(?:Article|Section|Clause)\s+\d+[A-Za-z\-]*|^(?:المادة|مادة)\s+",
            line,
            re.IGNORECASE,
        )
    )


def _match_violence_keywords(section: str) -> dict[str, set[str]]:
    normalized_section = section.casefold()
    matches: dict[str, set[str]] = {}

    for category, keywords in VIOLENCE_KEYWORDS_BY_CATEGORY.items():
        for keyword in keywords:
            if keyword.casefold() in normalized_section:
                matches.setdefault(category, set()).add(keyword)

    return matches


def _choose_category(matches_by_category: dict[str, set[str]]) -> str:
    priority = (
        "weapon_or_dangerous_object",
        "battery_or_physical_harm",
        "assault",
        "threat_or_intimidation",
        "public_order_or_affray",
        "child_or_family_protection",
        "general_violence",
    )
    for category in priority:
        if category in matches_by_category:
            return category
    return next(iter(matches_by_category))
