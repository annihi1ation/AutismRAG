"""Risk gate: decide whether the conditional verifier must run.

The verifier is triggered if ANY spec §11 condition holds, plus the
table-extraction-confidence trigger (plan R6).
"""

from __future__ import annotations

from typing import List, Optional

from ..config import HKGConfig
from ..schemas.article import ArticleDocument
from ..schemas.hyperedge import HyperEdge
from ..schemas.pto_graph import ArticlePTOGraphLite
from ..schemas.storage import ValidationReport
from ..schemas.verification import RiskGateResult, SupportScore

_RISKY_DIRECTIONS = {"no_effect", "worsened", "adverse_event", "mixed", "unclear"}


def risk_gate(
    he: HyperEdge,
    graph: ArticlePTOGraphLite,
    article: ArticleDocument,
    support_score: SupportScore,
    validation_report: ValidationReport,
    config: HKGConfig,
    candidate_margin: Optional[float] = None,
) -> RiskGateResult:
    reasons: List[str] = []

    if validation_report.status == "fail":
        reasons.append("schema_validation_failed")

    if he.confidence < config.confidence_threshold:
        reasons.append("low_hyperedge_confidence")

    if support_score.min_support() < config.support_score_threshold:
        reasons.append("low_support_score")

    roles = {e.role for e in he.entities}
    if "TREATMENT" not in roles:
        reasons.append("no_treatment_entity")
    if "OUTCOME" not in roles:
        reasons.append("no_outcome_entity")

    event = next(
        (e for e in graph.pto_events if e.id == he.source.pto_event_id), None
    )
    if event is not None:
        outcomes = {o.id: o for o in graph.outcome_nodes}
        for oid in event.outcome_node_ids:
            o = outcomes.get(oid)
            if o is not None and o.direction in _RISKY_DIRECTIONS:
                reasons.append(f"risky_outcome_direction:{o.direction}")
                break
        if event.comparator:
            reasons.append("comparator_present")
        if len(event.treatment_node_ids) > 1:
            reasons.append("multiple_treatment_nodes")
        if event.confidence < config.confidence_threshold:
            reasons.append("low_event_confidence")
    else:
        reasons.append("pto_event_not_found")

    n_groups = sum(
        1
        for p in graph.patient_nodes
        if p.granularity in {"group", "subgroup", "study_arm"}
    )
    if n_groups > 1 or len(graph.patient_nodes) > 1:
        reasons.append("multiple_patient_groups")

    if candidate_margin is not None and candidate_margin < config.candidate_margin_threshold:
        reasons.append("small_entity_candidate_margin")

    if len(set(he.source.evidence_span_ids)) > config.section_dispersion_threshold:
        reasons.append("dispersed_evidence_sections")

    # R6: low table extraction confidence on an evidence section.
    ev_sections = set(he.source.evidence_span_ids)
    for sec in article.sections:
        if sec.section_id not in ev_sections:
            continue
        for tbl in sec.tables:
            if tbl.extraction_confidence < config.table_extraction_confidence_threshold:
                reasons.append("low_table_extraction_confidence")
                break
        else:
            continue
        break

    risk_score = round(min(1.0, len(reasons) * 0.2), 3)
    needs = len(reasons) > 0 or risk_score >= config.risk_score_threshold
    return RiskGateResult(
        he_id=he.id,
        needs_verification=needs,
        risk_score=risk_score,
        risk_reasons=reasons,
    )
