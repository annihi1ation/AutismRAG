"""Section-level span guarantees."""

from __future__ import annotations

import pytest

from workflows.HyperKGCreator.embeddings.base import HashingEmbeddingModel
from workflows.HyperKGCreator.embeddings.section_index import build_section_index
from workflows.HyperKGCreator.embeddings.sentence_transformer_backend import (
    _split_for_embedding,
)
from workflows.HyperKGCreator.parsing.sectionizer import LineItem, build_sections
from workflows.HyperKGCreator.parsing.table_extractor import RawTable, attach_tables


def _lines() -> list[LineItem]:
    return [
        LineItem(0, "A Study of Intellectual Disability", 18.0, True, 0),
        LineItem(0, "Abstract", 14.0, True, 1),
        LineItem(0, "We treated a child and behavior improved.", 10.0, False, 2),
        LineItem(1, "Methods", 14.0, True, 3),
        LineItem(1, "Applied behavior analysis was used over twelve weeks.", 10.0, False, 4),
        LineItem(2, "Results", 14.0, True, 5),
        LineItem(2, "Adaptive behavior improved significantly.", 10.0, False, 6),
    ]


def test_build_sections_produces_section_spans():
    sections = build_sections(_lines(), "ART", doc_title="A Study")
    assert sections, "expected at least one section"
    headings = {s.normalized_heading for s in sections}
    assert "abstract" in headings
    assert "methods" in headings
    assert "results" in headings
    # Section ids are the only span ids — no chunk/paragraph/sentence ids.
    for s in sections:
        assert s.section_id.startswith("ART::")
        assert "chunk" not in s.section_id
        assert "para" not in s.section_id
        assert "sent" not in s.section_id


def test_tables_get_parent_section_id():
    sections = build_sections(_lines(), "ART")
    sections = attach_tables(
        sections,
        [RawTable(page=2, order=99, markdown="| a |", raw_text="a", caption=None,
                  extraction_confidence=0.8)],
        "ART",
    )
    seen_tables = [t for s in sections for t in s.tables]
    assert len(seen_tables) == 1
    section_ids = {s.section_id for s in sections}
    for t in seen_tables:
        assert t.parent_section_id in section_ids


def test_unassigned_table_goes_to_synthetic_section():
    # Table on page 0 with the only section starting on page 5 → no previous
    # section → synthetic __UNASSIGNED_TABLES__.
    sections = build_sections(
        [LineItem(5, "Results", 14.0, True, 0), LineItem(5, "text", 10.0, False, 1)],
        "ART",
    )
    sections = attach_tables(
        sections,
        [RawTable(page=0, order=0, markdown="| x |", raw_text="x", caption=None,
                  extraction_confidence=0.3)],
        "ART",
    )
    synth = [s for s in sections if s.section_type == "unassigned_tables"]
    assert synth and synth[0].tables


def test_internal_embedding_pieces_are_not_spans():
    # A long section is internally split for embedding, but build_section_index
    # only ever indexes section ids.
    long_text = "word " * 800
    sections = build_sections(
        [LineItem(0, "Results", 14.0, True, 0), LineItem(0, long_text, 10.0, False, 1)],
        "ART",
    )
    pieces = _split_for_embedding(long_text, max_tokens=128)
    assert len(pieces) > 1  # internal split happened

    from workflows.HyperKGCreator.schemas.article import ArticleDocument, ArticleMetadata

    doc = ArticleDocument(
        metadata=ArticleMetadata(article_id="ART"), sections=sections
    )
    index = build_section_index(doc, HashingEmbeddingModel(dim=32))
    assert set(index.ids) == {s.section_id for s in doc.sections}
    assert len(index) == len(doc.sections)  # one vector per section, not per piece
