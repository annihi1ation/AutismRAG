"""Shared test fixtures. Offline + deterministic (no LLM, no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable when pytest is invoked from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflows.HyperKGCreator.embeddings.base import HashingEmbeddingModel
from workflows.HyperKGCreator.llm.base import FakeLLMClient
from workflows.HyperKGCreator.schemas.article import (
    ArticleDocument,
    ArticleMetadata,
    SectionSpan,
    TableBlock,
)

ARTICLE_ID = "ART1"
SEC_ABSTRACT = f"{ARTICLE_ID}::sec000"
SEC_INTERVENTION = f"{ARTICLE_ID}::sec001"
SEC_RESULTS = f"{ARTICLE_ID}::sec002"

CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "entity_catalog.jsonl"
)


@pytest.fixture
def hashing_model() -> HashingEmbeddingModel:
    return HashingEmbeddingModel(dim=64)


@pytest.fixture
def mini_article() -> ArticleDocument:
    sections = [
        SectionSpan(
            section_id=SEC_ABSTRACT,
            article_id=ARTICLE_ID,
            heading="Abstract",
            normalized_heading="abstract",
            section_order=0,
            page_start=0,
            page_end=0,
            text=(
                "A single child with moderate intellectual disability received "
                "a behavioral intervention; adaptive behavior outcomes improved."
            ),
            section_type="abstract",
        ),
        SectionSpan(
            section_id=SEC_INTERVENTION,
            article_id=ARTICLE_ID,
            heading="Intervention",
            normalized_heading="intervention",
            section_order=1,
            page_start=1,
            page_end=1,
            text=(
                "The intervention was applied behavior analysis delivered over "
                "twelve weeks by a trained therapist."
            ),
            section_type="intervention",
        ),
        SectionSpan(
            section_id=SEC_RESULTS,
            article_id=ARTICLE_ID,
            heading="Results",
            normalized_heading="results",
            section_order=2,
            page_start=2,
            page_end=2,
            text=(
                "Adaptive behavior improved after applied behavior analysis; "
                "challenging behavior decreased on the Aberrant Behavior "
                "Checklist."
            ),
            section_type="results",
            tables=[
                TableBlock(
                    table_id=f"{SEC_RESULTS}::tbl00",
                    article_id=ARTICLE_ID,
                    parent_section_id=SEC_RESULTS,
                    caption="Table 1. Outcomes",
                    page_start=2,
                    page_end=2,
                    markdown="| metric | pre | post |\n|---|---|---|\n| ABC | 30 | 12 |",
                    raw_text="metric pre post ABC 30 12",
                    extraction_confidence=0.8,
                )
            ],
        ),
    ]
    return ArticleDocument(
        metadata=ArticleMetadata(article_id=ARTICLE_ID, title="Mini ID study"),
        sections=sections,
    )


def _pto_payload(_vars) -> dict:
    return {
        "article_id": ARTICLE_ID,
        "patient_nodes": [
            {
                "id": "P1",
                "text": "single child with moderate intellectual disability",
                "granularity": "single_case",
                "evidence_span_ids": [SEC_ABSTRACT],
                "confidence": 0.9,
            }
        ],
        "treatment_nodes": [
            {
                "id": "T1",
                "text": "applied behavior analysis",
                "treatment_type_hint": "THERAPEUTIC_APPROACH",
                "evidence_span_ids": [SEC_INTERVENTION],
                "confidence": 0.9,
            }
        ],
        "outcome_nodes": [
            {
                "id": "O1",
                "text": "adaptive behavior improved",
                "direction": "improved",
                "evidence_span_ids": [SEC_RESULTS],
                "confidence": 0.9,
            }
        ],
        "pto_events": [
            {
                "id": "E1",
                "patient_node_ids": ["P1"],
                "treatment_node_ids": ["T1"],
                "outcome_node_ids": ["O1"],
                "evidence_span_ids": [SEC_INTERVENTION, SEC_RESULTS],
                "confidence": 0.9,
            }
        ],
    }


def _hyperedge_payload(_vars) -> dict:
    return {
        "hyperedges": [
            {
                "id": "HE1",
                "type": "ID_TREATMENT_RESPONSE",
                "patient_text": "single child with moderate intellectual disability",
                "treatment_text": "applied behavior analysis",
                "outcome_text": "adaptive behavior improved",
                "edge_text": (
                    "Applied behavior analysis improved adaptive behavior in a "
                    "child with moderate intellectual disability."
                ),
                "entities": [
                    {
                        "entity_id": "seed:therapy:applied_behavior_analysis",
                        "name": "Applied Behavior Analysis",
                        "entity_type": "THERAPEUTIC_APPROACH",
                        "mention": "applied behavior analysis",
                        "role": "TREATMENT",
                        "confidence": 0.9,
                    },
                    {
                        "entity_id": "seed:functional:adaptive_behavior",
                        "name": "adaptive behavior",
                        "entity_type": "FUNCTIONAL_ABILITY",
                        "mention": "adaptive behavior",
                        "role": "OUTCOME",
                        "confidence": 0.85,
                    },
                    {
                        "entity_id": "seed:disorder:intellectual_disability",
                        "name": "intellectual disability",
                        "entity_type": "DISORDER",
                        "mention": "intellectual disability",
                        "role": "PATIENT",
                        "confidence": 0.8,
                    },
                ],
                "source": {
                    "article_id": ARTICLE_ID,
                    "pto_event_id": "E1",
                    "evidence_span_ids": [SEC_INTERVENTION, SEC_RESULTS],
                },
                "confidence": 0.88,
            }
        ]
    }


def _verifier_payload(variables) -> dict:
    inputs = variables.get("verifier_inputs", [])
    results = []
    for vi in inputs:
        he_id = vi.get("hyperedge", {}).get("id", "")
        results.append({"he_id": he_id, "status": "accepted", "reason": "fake-ok"})
    return {"results": results}


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(
        handlers={
            "pto_builder": _pto_payload,
            "hyperedge_builder_entity_selector": _hyperedge_payload,
            "conditional_verifier": _verifier_payload,
        }
    )


@pytest.fixture
def catalog_path() -> Path:
    return CATALOG_PATH
