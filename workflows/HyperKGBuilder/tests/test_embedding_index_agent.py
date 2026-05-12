"""Tests for ClaimKGEmbeddingIndexAgent."""

from __future__ import annotations

import unittest

from workflows.HyperKGBuilder.agents.embedding_index_agent import (
    CLAIMS_NS,
    ENTITIES_NS,
    SUMMARIES_NS,
    TRIPLES_NS,
    ClaimKGEmbeddingIndexAgent,
)
from workflows.HyperKGBuilder.schemas import ClaimRecord
from workflows.HyperKGBuilder.tests.conftest import (
    FakeEmbeddingService,
    tiny_unified_kg_payload,
)
from workflows.HyperKGBuilder.unified_kg_index import UnifiedKGIndex


class EmbeddingIndexAgentTest(unittest.TestCase):
    def test_embedding_index_builds_for_entities_triples_claims_summaries(self) -> None:
        emb = FakeEmbeddingService()
        kg = UnifiedKGIndex(tiny_unified_kg_payload(), embedding_service=emb)

        claim = ClaimRecord(
            claim_id="C:test:0001",
            claim_text="Children with Down's syndrome had elevated ABC scores.",
            claim_type="SYMPTOM_EXHIBITION",
            polarity="positive",
            modality="comparative",
            candidate_entities=[{"mention": "Down's syndrome", "entity_id": "E:disorder:down_syndrome"}],
            candidate_triples=[],
            scope={"population": "children"},
            article_id="art1",
            segment_id="seg0001",
            summary_id="SUM:art1:seg0001",
            summary_snippet="children showed higher ABC scores",
        )
        summaries = {
            "SUM:art1:seg0001": {
                "summary_id": "SUM:art1:seg0001",
                "article_id": "art1",
                "segment_id": "seg0001",
                "summary_text": "children with Down's syndrome ...",
            },
        }

        agent = ClaimKGEmbeddingIndexAgent(embedding_service=emb)
        adapter, manifest = agent.build([claim], kg, summaries)

        self.assertIn(CLAIMS_NS, adapter.records)
        self.assertIn(ENTITIES_NS, adapter.records)
        self.assertIn(TRIPLES_NS, adapter.records)
        self.assertIn(SUMMARIES_NS, adapter.records)

        self.assertEqual(manifest.namespaces[CLAIMS_NS], 1)
        self.assertEqual(manifest.namespaces[ENTITIES_NS], 3)
        self.assertEqual(manifest.namespaces[TRIPLES_NS], 2)
        self.assertEqual(manifest.namespaces[SUMMARIES_NS], 1)


if __name__ == "__main__":
    unittest.main()
