"""PDF → ArticleDocument (section-level spans only).

Uses PyMuPDF (``fitz``), matching the repo convention (``src/parse_pdf.py``).
Page numbers are preserved. Heuristics avoid overfitting to one paper format.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..schemas.article import ArticleDocument, ArticleMetadata
from .sectionizer import LineItem, build_sections
from .table_extractor import RawTable, attach_tables

logger = logging.getLogger(__name__)

_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold
_TABLE_CAPTION_RE = re.compile(r"^(table\s+\d+|table\s+[ivxlcdm]+)\b", re.IGNORECASE)
_NONALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def _article_id_from_path(pdf_path: Path) -> str:
    stem = pdf_path.stem.strip()
    sanitized = _NONALNUM_RE.sub("_", stem).strip("_")
    return sanitized or "article"


def _extract_lines_and_tables(
    pdf_path: Path,
) -> Tuple[List[LineItem], List[RawTable], Optional[str]]:
    import fitz  # lazy import (PyMuPDF)

    lines: List[LineItem] = []
    raw_tables: List[RawTable] = []
    order = 0
    doc_title: Optional[str] = None

    with fitz.open(pdf_path) as doc:
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()
        doc_title = title or None

        for pno in range(doc.page_count):
            page = doc[pno]
            try:
                text_dict = page.get_text("dict")
            except Exception as err:  # noqa: BLE001
                logger.warning("text extraction failed on page %d: %s", pno, err)
                continue

            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    font_size = max((float(s.get("size", 0.0)) for s in spans), default=0.0)
                    bold = any(
                        (int(s.get("flags", 0)) & _BOLD_FLAG)
                        or ("bold" in str(s.get("font", "")).lower())
                        for s in spans
                    )
                    lines.append(
                        LineItem(
                            page=pno,
                            text=text,
                            font_size=font_size,
                            bold=bold,
                            order=order,
                        )
                    )
                    order += 1

            # Table extraction (best-effort; text remains in section text as a
            # fallback even if structured extraction is unavailable).
            try:
                finder = page.find_tables()
                page_tables = list(getattr(finder, "tables", []) or [])
            except Exception as err:  # noqa: BLE001
                logger.debug("find_tables unavailable on page %d: %s", pno, err)
                page_tables = []

            for tbl in page_tables:
                try:
                    rows = tbl.extract()
                except Exception:  # noqa: BLE001
                    rows = []
                try:
                    md = tbl.to_markdown()
                except Exception:  # noqa: BLE001
                    md = ""
                n_rows = len(rows)
                n_cols = max((len(r) for r in rows), default=0)
                confidence = 0.8 if (n_rows >= 2 and n_cols >= 2) else 0.4
                raw_text = "\n".join(
                    "\t".join("" if c is None else str(c) for c in r) for r in rows
                )
                if not md and not raw_text:
                    continue
                raw_tables.append(
                    RawTable(
                        page=pno,
                        order=order,
                        markdown=md or raw_text,
                        raw_text=raw_text,
                        caption=None,
                        extraction_confidence=confidence,
                    )
                )
                order += 1

    # Light caption association: a nearby "Table N" line on the same page.
    by_page_captions: dict[int, List[str]] = {}
    for ln in lines:
        if _TABLE_CAPTION_RE.match(ln.text):
            by_page_captions.setdefault(ln.page, []).append(ln.text)
    for rt in raw_tables:
        caps = by_page_captions.get(rt.page)
        if caps:
            rt.caption = caps.pop(0)

    return lines, raw_tables, doc_title


def parse_pdf_to_article(pdf_path: Path) -> ArticleDocument:
    """Parse a PDF into an ArticleDocument with section-level spans."""
    pdf_path = Path(pdf_path)
    article_id = _article_id_from_path(pdf_path)
    lines, raw_tables, doc_title = _extract_lines_and_tables(pdf_path)

    sections = build_sections(lines, article_id, doc_title=doc_title)
    sections = attach_tables(sections, raw_tables, article_id)

    title = doc_title
    if not title and sections:
        first = sections[0]
        if first.section_type in {"title", "other"} and first.heading:
            title = first.heading

    metadata = ArticleMetadata(
        article_id=article_id,
        title=title,
        source_pdf_path=str(pdf_path),
    )
    return ArticleDocument(metadata=metadata, sections=sections)


def save_article_document(article: ArticleDocument, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(article.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def load_article_document(path: Path) -> ArticleDocument:
    path = Path(path)
    return ArticleDocument.model_validate_json(path.read_text(encoding="utf-8"))
