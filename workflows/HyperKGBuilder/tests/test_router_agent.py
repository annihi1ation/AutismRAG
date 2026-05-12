"""Tests for CandidatePackRouterAgent."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List

from workflows.HyperKGBuilder.agents.embedding_index_agent import ClaimKGEmbeddingIndexAgent
from workflows.HyperKGBuilder.agents.router_agent import CandidatePackRouterAgent
from workflows.HyperKGBuilder.config import RoutingConfig
from workflows.HyperKGBuilder.schemas import ClaimRecord
from workflows.HyperKGBuilder.tests.conftest import (
    FakeEmbeddingService,
    tiny_unified_kg_payload,
)
from workflows.HyperKGBuilder.unified_kg_index import UnifiedKGIndex


def _build_setup(claims: List[ClaimRecord], routing: RoutingConfig | None = None):
    routing = routing or RoutingConfig()
    emb = FakeEmbeddingService()
    payload = tiny_unified_kg_payload()
    # Inflate the KG to thousands of entities/triples for boundedness checks.
    base_entities = list(payload["entities"])
    base_triples = list(payload["triples"])
    big_entities: List[Dict[str, Any]] = []
    big_triples: List[Dict[str, Any]] = []
    for i in range(500):
        for e in base_entities:
            ee = dict(e)
            ee["canonical_name"] = f"{e['canonical_name']}_{i}"
            ee["normalized_id"] = f"{e['normalized_id']}_{i}"
            big_entities.append(ee)
        for t in base_triples:
            tt = dict(t)
            tt["head"] = f"{t['head']}_{i}"
            tt["tail"] = f"{t['tail']}_{i}"
            big_triples.append(tt)
    payload["entities"].extend(big_entities)
    payload["triples"].extend(big_triples)
    kg = UnifiedKGIndex(payload, embedding_service=emb)

    summaries: Dict[str, Dict[str, Any]] = {}
    for c in claims:
        if c.summary_id:
            summaries[c.summary_id] = {
                "summary_id": c.summary_id,
                "article_id": c.article_id,
                "segment_id": c.segment_id,
                "summary_text": c.summary_snippet or "",
            }
    index_agent = ClaimKGEmbeddingIndexAgent(embedding_service=emb)
    adapter, _ = index_agent.build(claims, kg, summaries)

    router = CandidatePackRouterAgent(routing=routing, embedding_service=emb)
    return router, adapter, kg, summaries


def _ambiguous_claim() -> ClaimRecord:
    return ClaimRecord(
        claim_id="C:amb:0001",
        claim_text="Risperidone might possibly reduce hyperactivity in some children.",
        claim_type="INTERVENTION_OUTCOME",
        polarity="positive",
        modality="speculative",
        candidate_entities=[
            {"mention": "Risperidone", "entity_id": "E:medication:risperidone", "canonical_name": "Risperidone"},
        ],
        candidate_triples=[],
        scope={},
        article_id="other_article",
        segment_id="seg0001",
        summary_id="SUM:art:seg0001",
        summary_snippet="risperidone may reduce hyperactivity",
    )


def _high_conf_claim() -> ClaimRecord:
    return ClaimRecord(
        claim_id="C:auto:0001",
        claim_text="Children with Down's syndrome showed elevated ABC scores.",
        claim_type="SYMPTOM_EXHIBITION",
        polarity="positive",
        modality="comparative",
        candidate_entities=[
            {"mention": "Down's syndrome", "entity_id": "E:disorder:down_syndrome", "canonical_name": "Down's syndrome"},
            {"mention": "Aberrant Behavior Checklist", "entity_id": "E:assessment:abc", "canonical_name": "Aberrant Behavior Checklist"},
        ],
        candidate_triples=[],
        scope={
            "population": "children",
            "severity": "elevated",
            "developmental_stage": "childhood",
            "dose_or_intensity": "n/a",
            "timepoint": "study",
            "comparator": "controls",
        },
        article_id="10.1046_j.1365-2788.1998.00123.x",
        segment_id="seg0001",
        summary_id="SUM:art:seg0001",
        summary_snippet="children with DS scored higher on the ABC",
    )


class RouterAgentTest(unittest.TestCase):
    def test_candidate_pack_is_bounded(self) -> None:
        claim = _high_conf_claim()
        routing = RoutingConfig(
            max_candidate_entities_per_mention=3,
            max_candidate_triples_per_work_unit=4,
            max_summary_snippets_per_work_unit=2,
            entity_top_k=8,
            triple_top_k=6,
            summary_top_k=2,
        )
        router, adapter, kg, summaries = _build_setup([claim], routing=routing)
        packs, _ = router.build_packs([claim], adapter, kg, summaries)

        self.assertEqual(len(packs), 1)
        pack = packs[0]
        # Entity cap is per-mention; the claim has 2 mentions, so total ≤ 6.
        self.assertLessEqual(
            len(pack.entity_candidates),
            routing.max_candidate_entities_per_mention * 2 + routing.entity_top_k,
        )
        self.assertLessEqual(
            len(pack.triple_candidates),
            routing.max_candidate_triples_per_work_unit,
        )
        self.assertLessEqual(
            len(pack.summary_snippets),
            routing.max_summary_snippets_per_work_unit,
        )

    def test_ambiguous_case_routes_to_llm(self) -> None:
        ambiguous = _ambiguous_claim()
        router, adapter, kg, summaries = _build_setup([ambiguous])
        packs, _ = router.build_packs([ambiguous], adapter, kg, summaries)
        self.assertEqual(len(packs), 1)
        pack = packs[0]
        self.assertEqual(pack.auto_decision, "online_llm")
        # At minimum, the hedging trigger should fire.
        self.assertIn("negation_or_hedging", pack.routing_reasons)

    def test_high_impact_entity_type_triggers_routing(self) -> None:
        # Risperidone is a Medication → high impact.
        claim = ClaimRecord(
            claim_id="C:hi:0001",
            claim_text="Risperidone reduces hyperactivity.",
            claim_type="INTERVENTION_OUTCOME",
            polarity="positive",
            modality="declarative",
            candidate_entities=[
                {"mention": "Risperidone", "entity_id": "E:medication:risperidone", "canonical_name": "Risperidone"},
            ],
            candidate_triples=[],
            scope={"population": "children", "severity": "moderate", "developmental_stage": "school-age",
                   "dose_or_intensity": "varies", "timepoint": "weekly", "comparator": "placebo"},
            article_id="other_article",
            segment_id="seg0001",
            summary_id="SUM:art:seg0001",
            summary_snippet="risperidone reduces hyperactivity",
        )
        router, adapter, kg, summaries = _build_setup([claim])
        packs, _ = router.build_packs([claim], adapter, kg, summaries)
        self.assertEqual(packs[0].auto_decision, "online_llm")
        self.assertIn("high_impact_claim", packs[0].routing_reasons)


if __name__ == "__main__":
    unittest.main()
