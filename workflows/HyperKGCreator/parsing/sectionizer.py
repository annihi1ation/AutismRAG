"""Heading heuristics → section-level spans.

The main (and only) span unit is a SECTION. No child spans are produced.
Heuristics are intentionally format-agnostic: a line is a heading if it is
short and visually/semantically heading-like, or matches a known section
keyword. Each SectionSpan.text holds everything until the next heading.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import List, Optional

from ..schemas.article import SectionSpan, SectionType

# Ordered keyword → section_type map. First match wins.
_SECTION_KEYWORDS: list[tuple[re.Pattern[str], SectionType]] = [
    (re.compile(r"^abstract\b|^summary\b"), "abstract"),
    (re.compile(r"^introduction\b|^background\b"), "introduction"),
    (re.compile(r"^participants?\b|^subjects?\b|^sample\b|^study population\b"), "participants"),
    (re.compile(r"^interventions?\b|^treatments?\b|^procedure\b|^study procedure"), "intervention"),
    (re.compile(r"^outcome|^measures?\b|^assessment(s)?\b|^instruments?\b"), "outcomes"),
    (re.compile(r"^results?\b|^findings?\b"), "results"),
    (re.compile(r"^methods?\b|^materials and methods\b|^methodolog"), "methods"),
    (re.compile(r"^discussion\b"), "discussion"),
    (re.compile(r"^conclusions?\b|^concluding"), "conclusion"),
    (re.compile(r"^supplement|^appendix|^supporting information"), "supplementary"),
    (re.compile(r"^title\b"), "title"),
]

_KNOWN_HEADING_RE = re.compile(
    r"^(abstract|summary|introduction|background|methods?|materials and methods|"
    r"methodolog\w*|participants?|subjects?|sample|study population|interventions?|"
    r"treatments?|procedure|outcomes?|measures?|assessments?|instruments?|results?|"
    r"findings?|discussion|conclusions?|references|acknowledg\w*|appendix|"
    r"supplement\w*|supporting information)\b",
    re.IGNORECASE,
)

_NUM_PREFIX_RE = re.compile(r"^\s*\d+(\.\d+)*[\.\)]?\s+")
_SENTENCE_END_RE = re.compile(r"[\.;:,]\s*$")


@dataclass
class LineItem:
    page: int
    text: str
    font_size: float
    bold: bool
    order: int


def _normalize_heading(raw: str) -> str:
    text = _NUM_PREFIX_RE.sub("", raw or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip(" :.-")


def _section_type(normalized: str, is_first: bool) -> SectionType:
    for pattern, stype in _SECTION_KEYWORDS:
        if pattern.search(normalized):
            return stype
    return "other"


def _is_heading(line: LineItem, body_size: float) -> bool:
    text = line.text.strip()
    if not text:
        return False
    word_count = len(text.split())
    if word_count == 0 or word_count > 12 or len(text) > 120:
        # A long line is body text even if styled.
        return bool(_KNOWN_HEADING_RE.match(_normalize_heading(text))) and word_count <= 14
    norm = _normalize_heading(text)
    if _KNOWN_HEADING_RE.match(norm):
        return True
    bigger = line.font_size >= body_size * 1.12
    styled = line.bold and line.font_size >= body_size * 0.98
    if (bigger or styled) and not _SENTENCE_END_RE.search(text):
        # Mostly title/upper-case short lines look like headings.
        letters = [c for c in text if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.4 or text.istitle():
                return True
    return False


def build_sections(
    lines: List[LineItem], article_id: str, doc_title: Optional[str] = None
) -> List[SectionSpan]:
    """Group line items into section spans (no child spans)."""
    sizes = [ln.font_size for ln in lines if ln.text.strip()]
    body_size = statistics.median(sizes) if sizes else 10.0

    sections: List[SectionSpan] = []
    order = 0

    def _new_section(heading: str, page: int) -> SectionSpan:
        nonlocal order
        norm = _normalize_heading(heading)
        stype = _section_type(norm, is_first=not sections)
        sec = SectionSpan(
            section_id=f"{article_id}::sec{order:03d}",
            article_id=article_id,
            heading=heading.strip() or "(untitled)",
            normalized_heading=norm or None,
            section_order=order,
            page_start=page,
            page_end=page,
            text="",
            section_type=stype,
        )
        order += 1
        return sec

    current: Optional[SectionSpan] = None
    buffer: List[str] = []

    def _flush() -> None:
        if current is not None:
            current.text = "\n".join(buffer).strip()
            current.token_count_estimate = len(current.text.split())
            sections.append(current)

    for ln in lines:
        if _is_heading(ln, body_size):
            _flush()
            buffer = []
            current = _new_section(ln.text, ln.page)
        else:
            if current is None:
                # Preamble before any heading → synthetic title/front section.
                current = _new_section(doc_title or "Front Matter", ln.page)
                if current.section_type == "other":
                    current.section_type = "title"
                buffer = []
            buffer.append(ln.text)
            current.page_end = ln.page
    _flush()

    if not sections:
        # Robust fallback: a single "other" section holding all text.
        all_text = "\n".join(ln.text for ln in lines).strip()
        sections.append(
            SectionSpan(
                section_id=f"{article_id}::sec000",
                article_id=article_id,
                heading=doc_title or "Body",
                normalized_heading="body",
                section_order=0,
                page_start=lines[0].page if lines else None,
                page_end=lines[-1].page if lines else None,
                text=all_text,
                section_type="other",
                token_count_estimate=len(all_text.split()),
            )
        )
    return sections
