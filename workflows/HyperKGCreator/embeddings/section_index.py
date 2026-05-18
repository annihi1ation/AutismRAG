"""Section embedding index: one vector per SectionSpan.

Long sections are internally split for model limits and pooled by the backend;
the final vector belongs to the SectionSpan and is the only thing indexed.
Internal pieces never become evidence spans.
"""

from __future__ import annotations

from typing import List

from ..schemas.article import ArticleDocument
from .base import EmbeddingModel
from .vector_index import SectionVectorIndex


def section_embed_text(article: ArticleDocument, section_id: str) -> str:
    sec = article.get_section(section_id)
    if sec is None:
        return ""
    parts: List[str] = [sec.heading or "", sec.text or ""]
    for tbl in sec.tables:
        if tbl.caption:
            parts.append(tbl.caption)
        if tbl.markdown:
            parts.append(tbl.markdown)
    return "\n".join(p for p in parts if p)


def build_section_index(
    article: ArticleDocument, model: EmbeddingModel
) -> SectionVectorIndex:
    ids = [s.section_id for s in article.sections]
    texts = [section_embed_text(article, sid) for sid in ids]
    vectors = model.embed_texts(texts)
    index = SectionVectorIndex()
    if ids:
        index.add(ids, vectors)
    # Cache vectors back onto the sections (final per-section vector only).
    for sec, vec in zip(article.sections, vectors):
        sec.embedding = [float(x) for x in vec]
    return index
