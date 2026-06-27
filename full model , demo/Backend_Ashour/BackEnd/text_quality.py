from __future__ import annotations

import re


_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")

_ARABIC_ALLOWED_LATIN_TERMS = (
    "Guardian Eye",
    "VLM",
    "LLM",
    "RGB",
    "ViT",
)


ARABIC_ONLY_INSTRUCTION = (
    "For Arabic output, every heading, label, and sentence must be written in natural "
    "Modern Standard Arabic. Translate all English descriptive words, including scene, "
    "people, clothing, position, action, and assessment labels. Do not copy any English "
    "sentence and do not use Chinese characters. The only Latin-script text permitted is "
    "the proper name Guardian Eye and the technical names VLM, LLM, RGB, or ViT. Before "
    "answering, silently verify that no other Latin-script word remains. Output the Arabic "
    "narration only, with no translation notes or preface."
)

_ARABIC_LEFTOVER_REPLACEMENTS = {
    "evidenced": "مدعوم",
    "evidence": "الدليل",
    "peak window": "فترة النشاط الأبرز",
    "peak": "الذروة",
    "classifier": "المصنف",
    "classification": "التصنيف",
    "classified": "صنف",
    "detected": "رصد",
    "stream": "مسار",
    "gate": "بوابة",
    "frames": "المشاهد",
    "frame": "المشهد",
}

_USER_FACING_REPLACEMENTS = {
    "V9 classifier": "Guardian Eye classifier",
    "model v9": "Guardian Eye classifier",
    "V9": "Guardian Eye",
    "VLM summary": "Guardian Eye visual analysis",
    "current_vlm_summary": "current incident analysis",
    "summary_source": "analysis source",
    "retrieved VLM context": "Guardian Eye visual analysis",
    "according to the VLM summary": "based on Guardian Eye's visual analysis",
    "based on the VLM summary": "based on Guardian Eye's visual analysis",
}

_RAW_TELEMETRY_PATTERNS = [
    re.compile(r"\bframes?\s+\d+\s*[-–]\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bframes?\s+\d+\s+to\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bpeak[_ -]?frame\w*\b", re.IGNORECASE),
    re.compile(r"\bpeak[_ -]?window\b", re.IGNORECASE),
    re.compile(r"\bselected[_ -]?frames?\b", re.IGNORECASE),
    re.compile(r"\bevidence[_ -]?packet\b", re.IGNORECASE),
    re.compile(r"\bprompt instructions?\b", re.IGNORECASE),
    re.compile(r"\bframe indices?\b", re.IGNORECASE),
    re.compile(r"\bvit\s+index\b", re.IGNORECASE),
    re.compile(r"\bpeak activity\b", re.IGNORECASE),
    re.compile(r"الإطارات?\s+\d+\s*[-–]\s*\d+", re.IGNORECASE),
    re.compile(r"إطار\s+\d+\s*[-–]\s*\d+", re.IGNORECASE),
]

_RAW_HEADING_RE = re.compile(
    r"\b(?:verdict|confidence|peak window|weapon flagged|gate contribution language|"
    r"strongest available model evidence|peaked activity|vit index|frame indices)\s*:\s*",
    re.IGNORECASE,
)


def cleanup_repetition(text: str) -> str:
    """Remove obvious adjacent repeated tokens without changing sentence meaning."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return clean

    tokens = clean.split(" ")
    output: list[str] = []
    previous_key = ""
    repeat_count = 0
    for token in tokens:
        key = re.sub(r"^[^\w\u0600-\u06ff]+|[^\w\u0600-\u06ff]+$", "", token).casefold()
        if key and key == previous_key:
            repeat_count += 1
        else:
            previous_key = key
            repeat_count = 1
        if repeat_count <= 1:
            output.append(token)
    return " ".join(output).strip()


def remove_markdown_artifacts(text: str) -> str:
    clean_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*•]\s+", "", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        if line:
            clean_lines.append(line)
    return " ".join(clean_lines).strip()


def scrub_user_facing_artifacts(text: str, language: str) -> str:
    clean = str(text or "")
    for old, new in _USER_FACING_REPLACEMENTS.items():
        clean = re.sub(re.escape(old), new, clean, flags=re.IGNORECASE)
    clean = _RAW_HEADING_RE.sub("", clean)
    for pattern in _RAW_TELEMETRY_PATTERNS:
        clean = pattern.sub("", clean)
    if language == "ar":
        for old, new in _ARABIC_LEFTOVER_REPLACEMENTS.items():
            clean = re.sub(rf"\b{re.escape(old)}\b", new, clean, flags=re.IGNORECASE)
    return clean


def polish_generated_text(
    text: str,
    language: str,
    *,
    label: str = "generation",
    require_arabic_only: bool = False,
    allow_chinese: bool = False,
    allowed_latin_terms: tuple[str, ...] = (),
) -> str:
    clean = remove_markdown_artifacts(text)
    clean = scrub_user_facing_artifacts(clean, language)
    clean = cleanup_repetition(clean)
    clean = re.sub(r"\s+([.,;:!?؟،؛])", r"\1", clean)
    clean = re.sub(r"([.!?؟])\s*([.!?؟])+", r"\1", clean)
    return validate_generated_language(
        clean,
        language,
        label=label,
        require_arabic_only=require_arabic_only,
        allow_chinese=allow_chinese,
        allowed_latin_terms=allowed_latin_terms,
    )


def contains_significant_chinese(text: str) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def validate_generated_language(
    text: str,
    language: str,
    *,
    label: str = "generation",
    require_arabic_only: bool = False,
    allow_chinese: bool = False,
    allowed_latin_terms: tuple[str, ...] = (),
) -> str:
    clean = cleanup_repetition(text)
    if language == "ar":
        if not allow_chinese and contains_significant_chinese(clean):
            raise ValueError(f"{label} contains Chinese characters in Arabic mode")
        if not _ARABIC_RE.search(clean):
            raise ValueError(f"{label} does not contain Arabic text in Arabic mode")
        if require_arabic_only:
            latin_check = clean
            for allowed_term in _ARABIC_ALLOWED_LATIN_TERMS:
                latin_check = re.sub(
                    re.escape(allowed_term),
                    "",
                    latin_check,
                    flags=re.IGNORECASE,
                )
            for allowed_term in allowed_latin_terms:
                latin_check = re.sub(
                    re.escape(allowed_term),
                    "",
                    latin_check,
                    flags=re.IGNORECASE,
                )
            unexpected_latin = _LATIN_WORD_RE.findall(latin_check)
            if unexpected_latin:
                preview = ", ".join(unexpected_latin[:5])
                raise ValueError(
                    f"{label} contains non-Arabic words in Arabic mode: {preview}"
                )
    return clean
