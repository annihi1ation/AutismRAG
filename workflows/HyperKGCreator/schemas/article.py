"""Article + section-level span schemas.

The evidence span unit is a SECTION. ``ArticleSpan`` == ``SectionSpan``.
Pydantic models here enforce only LOCAL (intra-object) invariants; cross-object
reference checks live in ``validation.schema_validator`` (plan R3).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

UNASSIGNED_TABLES_SECTION_ID = "__UNASSIGNED_TABLES__"

SectionType = Literal[
    "title",
    "abstract",
    "introduction",
    "methods",
    "participants",
    "intervention",
    "outcomes",
    "results",
    "discussion",
    "conclusion",
    "supplementary",
    "unassigned_tables",
    "other",
]


class ArticleMetadata(BaseModel):
    article_id: str
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    source_pdf_path: Optional[str] = None

    @field_validator("article_id")
    @classmethod
    def _non_empty_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("article_id must be non-empty")
        return v


class TableBlock(BaseModel):
    table_id: str
    article_id: str
    parent_section_id: str
    caption: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    markdown: str = ""
    raw_text: Optional[str] = None
    extraction_confidence: float = 0.0

    @field_validator("parent_section_id")
    @classmethod
    def _section_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("TableBlock.parent_section_id must be non-empty")
        return v


class SectionSpan(BaseModel):
    section_id: str
    article_id: str
    heading: str
    normalized_heading: Optional[str] = None
    section_order: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str = ""
    tables: List[TableBlock] = Field(default_factory=list)
    section_type: SectionType = "other"
    embedding: Optional[List[float]] = None
    token_count_estimate: Optional[int] = None

    @field_validator("section_id")
    @classmethod
    def _non_empty_section_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("section_id must be non-empty")
        return v


class ArticleDocument(BaseModel):
    metadata: ArticleMetadata
    sections: List[SectionSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def _local_integrity(self) -> "ArticleDocument":
        seen: set[str] = set()
        last_order: Optional[int] = None
        for sec in self.sections:
            if sec.section_id in seen:
                raise ValueError(f"duplicate section_id within article: {sec.section_id}")
            seen.add(sec.section_id)
            if last_order is not None and sec.section_order < last_order:
                raise ValueError(
                    "section_order must be monotonic non-decreasing "
                    f"(saw {sec.section_order} after {last_order})"
                )
            last_order = sec.section_order
        return self

    def section_ids(self) -> set[str]:
        return {s.section_id for s in self.sections}

    def get_section(self, section_id: str) -> Optional[SectionSpan]:
        for s in self.sections:
            if s.section_id == section_id:
                return s
        return None
