"""Table extraction + section attachment.

Rules (plan critical span rule):
- attach a table to the section it belongs to;
- if it can't be confidently assigned, attach to the nearest *previous*
  section (in reading order);
- if there is no previous section, attach to a synthetic
  ``__UNASSIGNED_TABLES__`` article-level section.
Tables never become evidence spans themselves — they hang off a SectionSpan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..schemas.article import (
    UNASSIGNED_TABLES_SECTION_ID,
    SectionSpan,
    TableBlock,
)


@dataclass
class RawTable:
    page: int
    order: int  # global reading-order position
    markdown: str
    raw_text: str
    caption: Optional[str]
    extraction_confidence: float


def _owning_section_index(
    sections: List[SectionSpan], table_page: int, section_order_by_page: List[int]
) -> Optional[int]:
    """Nearest *previous* section by page (reading order)."""
    candidate: Optional[int] = None
    for i, sec in enumerate(sections):
        start = sec.page_start if sec.page_start is not None else 0
        if start <= table_page:
            candidate = i
        else:
            break
    return candidate


def attach_tables(
    sections: List[SectionSpan], raw_tables: List[RawTable], article_id: str
) -> List[SectionSpan]:
    if not raw_tables:
        return sections

    synthetic: Optional[SectionSpan] = None
    next_order = (max((s.section_order for s in sections), default=-1)) + 1

    for t_idx, rt in enumerate(sorted(raw_tables, key=lambda r: (r.page, r.order))):
        sec_idx = _owning_section_index(sections, rt.page, [])
        if sec_idx is None:
            if synthetic is None:
                synthetic = SectionSpan(
                    section_id=f"{article_id}::{UNASSIGNED_TABLES_SECTION_ID}",
                    article_id=article_id,
                    heading=UNASSIGNED_TABLES_SECTION_ID,
                    normalized_heading="unassigned tables",
                    section_order=next_order,
                    text="",
                    section_type="unassigned_tables",
                )
                sections.append(synthetic)
            target = synthetic
        else:
            target = sections[sec_idx]

        block = TableBlock(
            table_id=f"{target.section_id}::tbl{t_idx:02d}",
            article_id=article_id,
            parent_section_id=target.section_id,
            caption=rt.caption,
            page_start=rt.page,
            page_end=rt.page,
            markdown=rt.markdown,
            raw_text=rt.raw_text,
            extraction_confidence=rt.extraction_confidence,
        )
        target.tables.append(block)
    return sections
