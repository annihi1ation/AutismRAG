"""Smoke tests for UnifiedKGIndex."""

from __future__ import annotations

import unittest

from workflows.HyperKGBuilder.tests.conftest import (
    FakeEmbeddingService,
    tiny_unified_kg_payload,
)
from workflows.HyperKGBuilder.unified_kg_index import UnifiedKGIndex


class UnifiedKGIndexTest(unittest.TestCase):
    def test_lookup_by_name(self) -> None:
        index = UnifiedKGIndex(tiny_unified_kg_payload(), embedding_service=FakeEmbeddingService())
        hits = index.lookup_entity_by_name("Down's syndrome")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["entity_type"], "Disorder")

    def test_relation_distribution_is_bounded(self) -> None:
        index = UnifiedKGIndex(tiny_unified_kg_payload())
        dist = index.relation_distribution(top_n=5)
        self.assertLessEqual(len(dist), 5)
        self.assertIn("has_diagnosis", dist)

    def test_top_k_uses_embedding(self) -> None:
        emb = FakeEmbeddingService()
        index = UnifiedKGIndex(tiny_unified_kg_payload(), embedding_service=emb)
        index.build_vectors()
        query = emb.encode_one("Down syndrome aberrant behavior")
        results = index.entity_top_k(query, k=2)
        self.assertEqual(len(results), 2)
        for entity, score in results:
            self.assertIn("entity_id", entity)
            self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
