"""Deterministic CROSS-OBJECT validation (plan R3).

Local/intra-object invariants are enforced by the Pydantic models. This module
only checks references *between* objects and the section-level evidence rule:
every evidence id must be a SectionSpan.section_id (never a table id, internal
embedding piece, or quote).
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from ..config import ALLOWED_ENTITY_TYPES
from ..schemas.article import ArticleDocument
from ..schemas.hyperedge import HyperEdge
from ..schemas.pto_graph import ArticlePTOGraphLite
from ..schemas.storage import ValidationReport


def _json_ok(model) -> bool:
    try:
        json.loads(model.model_dump_json())
        return True
    except Exception:
        return False


def validate_article_document(article: ArticleDocument) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    section_ids = article.section_ids()

    table_ids: set[str] = set()
    for sec in article.sections:
        for tbl in sec.tables:
            if tbl.parent_section_id not in section_ids:
                errors.append(
                    f"TableBlock {tbl.table_id} parent_section_id "
                    f"'{tbl.parent_section_id}' is not an existing section"
                )
            if tbl.article_id != article.metadata.article_id:
                errors.append(
                    f"TableBlock {tbl.table_id} article_id mismatch"
                )
            if tbl.table_id in table_ids:
                errors.append(f"duplicate table_id {tbl.table_id}")
            table_ids.add(tbl.table_id)
        if sec.article_id != article.metadata.article_id:
            errors.append(f"Section {sec.section_id} article_id mismatch")

    if not _json_ok(article):
        errors.append("ArticleDocument is not JSON-serializable")

    return ValidationReport.from_errors(
        errors, warnings, subject=f"article:{article.metadata.article_id}"
    )


def _check_evidence_ids(
    label: str, ids: Sequence[str], section_ids: set[str], errors: List[str]
) -> None:
    for sid in ids:
        if sid not in section_ids:
            errors.append(
                f"{label}: evidence_span_id '{sid}' is not a SectionSpan id "
                "(evidence must be section-level only)"
            )


def validate_pto_graph(
    graph: ArticlePTOGraphLite, article: ArticleDocument
) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    section_ids = article.section_ids()

    if graph.article_id != article.metadata.article_id:
        errors.append(
            f"PTO graph article_id '{graph.article_id}' != article "
            f"'{article.metadata.article_id}'"
        )

    p_ids = {n.id for n in graph.patient_nodes}
    t_ids = {n.id for n in graph.treatment_nodes}
    o_ids = {n.id for n in graph.outcome_nodes}

    for n in graph.patient_nodes + graph.treatment_nodes + graph.outcome_nodes:
        _check_evidence_ids(f"node {n.id}", n.evidence_span_ids, section_ids, errors)

    for ev in graph.pto_events:
        _check_evidence_ids(f"event {ev.id}", ev.evidence_span_ids, section_ids, errors)
        for pid in ev.patient_node_ids:
            if pid not in p_ids:
                errors.append(f"event {ev.id} references missing patient node {pid}")
        for tid in ev.treatment_node_ids:
            if tid not in t_ids:
                errors.append(f"event {ev.id} references missing treatment node {tid}")
        for oid in ev.outcome_node_ids:
            if oid not in o_ids:
                errors.append(f"event {ev.id} references missing outcome node {oid}")

    if not _json_ok(graph):
        errors.append("PTO graph is not JSON-serializable")

    return ValidationReport.from_errors(
        errors, warnings, subject=f"pto_graph:{graph.article_id}"
    )


def validate_hyperedge(
    he: HyperEdge,
    article: ArticleDocument,
    graph: ArticlePTOGraphLite,
    allowed_entity_types: Optional[Sequence[str]] = None,
) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    allowed = set(allowed_entity_types or ALLOWED_ENTITY_TYPES)
    section_ids = article.section_ids()
    event_ids = {e.id for e in graph.pto_events}

    if he.type != "ID_TREATMENT_RESPONSE":
        errors.append(f"HyperEdge {he.id} type must be ID_TREATMENT_RESPONSE")

    if he.source.article_id != article.metadata.article_id:
        errors.append(f"HyperEdge {he.id} source.article_id mismatch")
    if he.source.pto_event_id not in event_ids:
        errors.append(
            f"HyperEdge {he.id} source.pto_event_id "
            f"'{he.source.pto_event_id}' not in PTO graph"
        )
    _check_evidence_ids(
        f"HyperEdge {he.id}", he.source.evidence_span_ids, section_ids, errors
    )
    if not he.source.evidence_span_ids:
        warnings.append(f"HyperEdge {he.id} has no evidence_span_ids")

    roles = {e.role for e in he.entities}
    if "TREATMENT" not in roles:
        errors.append(f"HyperEdge {he.id} missing a TREATMENT entity")
    if "OUTCOME" not in roles:
        errors.append(f"HyperEdge {he.id} missing an OUTCOME entity")

    for ent in he.entities:
        if ent.entity_type not in allowed:
            errors.append(
                f"HyperEdge {he.id} entity '{ent.entity_id}' has unsupported "
                f"entity_type '{ent.entity_type}'"
            )

    if he.verifier_status == "rejected":
        warnings.append(f"HyperEdge {he.id} carries verifier_status=rejected")

    if not _json_ok(he):
        errors.append(f"HyperEdge {he.id} is not JSON-serializable")

    return ValidationReport.from_errors(errors, warnings, subject=f"hyperedge:{he.id}")


def validate_hyperedges(
    hyperedges: Sequence[HyperEdge],
    article: ArticleDocument,
    graph: ArticlePTOGraphLite,
    allowed_entity_types: Optional[Sequence[str]] = None,
) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    seen: set[str] = set()
    for he in hyperedges:
        if he.id in seen:
            errors.append(f"duplicate HyperEdge id {he.id}")
        seen.add(he.id)
        rep = validate_hyperedge(he, article, graph, allowed_entity_types)
        errors.extend(rep.errors)
        warnings.extend(rep.warnings)
    return ValidationReport.from_errors(
        errors, warnings, subject=f"hyperedges:{article.metadata.article_id}"
    )
