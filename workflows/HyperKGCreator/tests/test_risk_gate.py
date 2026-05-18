"""Risk gate triggering behaviour."""

from __future__ import annotations

from workflows.HyperKGCreator.config import load_default_config
from workflows.HyperKGCreator.schemas.article import (
    ArticleDocument,
    ArticleMetadata,
    SectionSpan,
)
from workflows.HyperKGCreator.schemas.hyperedge import HyperEdge, HyperEdgeSource
from workflows.HyperKGCreator.schemas.pto_graph import (
    ArticlePTOGraphLite,
    OutcomeNode,
    PatientNode,
    PTOEvent,
    TreatmentNode,
)
from workflows.HyperKGCreator.schemas.storage import ValidationReport
from workflows.HyperKGCreator.schemas.verification import SupportScore
from workflows.HyperKGCreator.validation.risk_gate import risk_gate

SID = "A::sec000"
CFG = load_default_config()
OK_REPORT = ValidationReport(status="pass")


def _article():
    return ArticleDocument(
        metadata=ArticleMetadata(article_id="A"),
        sections=[
            SectionSpan(section_id=SID, article_id="A", heading="Results",
                        section_order=0, text="improved", section_type="results")
        ],
    )


def _graph(direction="improved", comparator=None):
    return ArticlePTOGraphLite(
        article_id="A",
        patient_nodes=[PatientNode(id="P1", text="single child", granularity="single_case")],
        treatment_nodes=[TreatmentNode(id="T1", text="ABA")],
        outcome_nodes=[OutcomeNode(id="O1", text="x", direction=direction)],
        pto_events=[
            PTOEvent(
                id="E1",
                patient_node_ids=["P1"],
                treatment_node_ids=["T1"],
                outcome_node_ids=["O1"],
                comparator=comparator,
                confidence=0.9,
            )
        ],
    )


def _he():
    return HyperEdge(
        id="HE1",
        patient_text="single child with ID",
        treatment_text="ABA",
        outcome_text="x",
        edge_text="ABA -> x",
        entities=[
            {"entity_id": "t", "name": "ABA", "entity_type": "THERAPEUTIC_APPROACH",
             "mention": "ABA", "role": "TREATMENT"},
            {"entity_id": "o", "name": "behavior", "entity_type": "FUNCTIONAL_ABILITY",
             "mention": "behavior", "role": "OUTCOME"},
        ],
        source=HyperEdgeSource(article_id="A", pto_event_id="E1", evidence_span_ids=[SID]),
        confidence=0.9,
    )


def _support(score: float) -> SupportScore:
    return SupportScore(
        he_id="HE1",
        patient_support_score=score,
        treatment_support_score=score,
        outcome_support_score=score,
        edge_support_score=score,
        nearest_patient_section_ids=[SID],
        nearest_treatment_section_ids=[SID],
        nearest_outcome_section_ids=[SID],
        nearest_edge_section_ids=[SID],
    )


def test_clean_edge_does_not_trigger():
    res = risk_gate(_he(), _graph(), _article(), _support(0.9), OK_REPORT, CFG)
    assert res.needs_verification is False, res.risk_reasons
    assert res.risk_reasons == []


def test_low_support_triggers():
    res = risk_gate(_he(), _graph(), _article(), _support(0.01), OK_REPORT, CFG)
    assert res.needs_verification is True
    assert "low_support_score" in res.risk_reasons


def test_adverse_event_triggers():
    res = risk_gate(
        _he(), _graph(direction="adverse_event"), _article(), _support(0.9),
        OK_REPORT, CFG,
    )
    assert res.needs_verification is True
    assert any(r.startswith("risky_outcome_direction") for r in res.risk_reasons)


def test_comparator_triggers():
    res = risk_gate(
        _he(), _graph(comparator="placebo"), _article(), _support(0.9),
        OK_REPORT, CFG,
    )
    assert res.needs_verification is True
    assert "comparator_present" in res.risk_reasons
