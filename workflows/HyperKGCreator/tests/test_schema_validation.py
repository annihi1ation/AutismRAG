"""HyperEdge local + cross-object validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
from workflows.HyperKGCreator.validation.schema_validator import validate_hyperedge

SID = "A::sec000"


def _entities(roles=("TREATMENT", "OUTCOME")):
    base = {
        "TREATMENT": {
            "entity_id": "seed:therapy:applied_behavior_analysis",
            "name": "ABA",
            "entity_type": "THERAPEUTIC_APPROACH",
            "mention": "ABA",
            "role": "TREATMENT",
        },
        "OUTCOME": {
            "entity_id": "seed:functional:adaptive_behavior",
            "name": "adaptive behavior",
            "entity_type": "FUNCTIONAL_ABILITY",
            "mention": "behavior",
            "role": "OUTCOME",
        },
    }
    return [base[r] for r in roles]


def _he(**overrides):
    data = {
        "id": "HE1",
        "type": "ID_TREATMENT_RESPONSE",
        "patient_text": "a child with ID",
        "treatment_text": "ABA",
        "outcome_text": "behavior improved",
        "edge_text": "ABA improved behavior",
        "entities": _entities(),
        "source": {"article_id": "A", "pto_event_id": "E1", "evidence_span_ids": [SID]},
        "confidence": 0.9,
    }
    data.update(overrides)
    return data


def _article_and_graph():
    article = ArticleDocument(
        metadata=ArticleMetadata(article_id="A"),
        sections=[
            SectionSpan(
                section_id=SID, article_id="A", heading="Results",
                section_order=0, text="adaptive behavior improved", section_type="results",
            )
        ],
    )
    graph = ArticlePTOGraphLite(
        article_id="A",
        patient_nodes=[PatientNode(id="P1", text="child")],
        treatment_nodes=[TreatmentNode(id="T1", text="ABA")],
        outcome_nodes=[OutcomeNode(id="O1", text="improved", direction="improved")],
        pto_events=[
            PTOEvent(
                id="E1",
                patient_node_ids=["P1"],
                treatment_node_ids=["T1"],
                outcome_node_ids=["O1"],
            )
        ],
    )
    return article, graph


def test_missing_patient_text_fails():
    with pytest.raises(ValidationError):
        HyperEdge.model_validate(_he(patient_text=""))


def test_missing_treatment_entity_fails():
    with pytest.raises(ValidationError):
        HyperEdge.model_validate(_he(entities=_entities(("OUTCOME",))))


def test_missing_outcome_entity_fails():
    with pytest.raises(ValidationError):
        HyperEdge.model_validate(_he(entities=_entities(("TREATMENT",))))


def test_evidence_span_not_section_id_fails():
    article, graph = _article_and_graph()
    he = HyperEdge.model_validate(
        _he(source=HyperEdgeSource(
            article_id="A", pto_event_id="E1", evidence_span_ids=["A::chunk-7"]
        ))
    )
    report = validate_hyperedge(he, article, graph)
    assert report.status == "fail"
    assert any("not a SectionSpan id" in e for e in report.errors)


def test_valid_hyperedge_passes():
    article, graph = _article_and_graph()
    he = HyperEdge.model_validate(_he())
    report = validate_hyperedge(he, article, graph)
    assert report.status == "pass", report.errors
