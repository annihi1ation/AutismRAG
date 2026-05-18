"""Offline entity candidate retrieval."""

from __future__ import annotations

from workflows.HyperKGCreator.routing.entity_retrieval import (
    NodeText,
    build_entity_index,
    load_entity_catalog,
    pto_node_texts,
    retrieve_entity_candidates,
)
from workflows.HyperKGCreator.schemas.pto_graph import (
    ArticlePTOGraphLite,
    OutcomeNode,
    PatientNode,
    TreatmentNode,
)


def test_catalog_loads(catalog_path):
    records = load_entity_catalog(catalog_path, for_build=False)
    assert len(records) >= 30
    ids = {r.entity_id for r in records}
    assert "seed:therapy:applied_behavior_analysis" in ids


def test_index_retrieves_likely_candidate(catalog_path, hashing_model):
    records = load_entity_catalog(catalog_path, for_build=False)
    index = build_entity_index(records, hashing_model)
    pack = retrieve_entity_candidates(
        [NodeText("treatment_text", "T1", "applied behavior analysis")],
        index,
        hashing_model,
        records,
        top_k=5,
    )
    assert pack, "expected at least one candidate pack"
    all_ids = {c.entity_id for p in pack for c in p.candidates}
    assert "seed:therapy:applied_behavior_analysis" in all_ids


def test_role_hint_comes_from_source_field_not_entity_type(catalog_path, hashing_model):
    records = load_entity_catalog(catalog_path, for_build=False)
    index = build_entity_index(records, hashing_model)
    # A treatment node whose text mentions a DISORDER term: role_hint must
    # still be TREATMENT (from the source field, not the candidate's type).
    packs = retrieve_entity_candidates(
        [NodeText("treatment_text", "T1", "therapy for intellectual disability")],
        index,
        hashing_model,
        records,
        top_k=5,
    )
    assert packs
    assert all(p.role_hint == "TREATMENT" for p in packs)
    assert all(p.source_node_id == "T1" for p in packs)


def test_pto_node_texts_mapping():
    graph = ArticlePTOGraphLite(
        article_id="A",
        patient_nodes=[PatientNode(id="P1", text="child")],
        treatment_nodes=[TreatmentNode(id="T1", text="ABA")],
        outcome_nodes=[OutcomeNode(id="O1", text="improved", direction="improved")],
    )
    nts = {nt.source_node_id: nt for nt in pto_node_texts(graph)}
    assert nts["P1"].role_hint == "PATIENT"
    assert nts["T1"].role_hint == "TREATMENT"
    assert nts["O1"].role_hint == "OUTCOME"
