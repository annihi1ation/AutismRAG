"""Tests for HyperKGWriterIndexerAgent."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from workflows.HyperKGBuilder.agents.writer_indexer_agent import (
    HyperKGWriterIndexerAgent,
)
from workflows.HyperKGBuilder.config import WriterConfig
from workflows.HyperKGBuilder.schemas import (
    EvidenceHyperedge,
    HyperKGRunReport,
)
from workflows.HyperKGBuilder.tests.conftest import (
    FakeEmbeddingService,
    tiny_unified_kg_payload,
)
from workflows.HyperKGBuilder.unified_kg_index import UnifiedKGIndex


def _make_edge(suffix: str, claim_text: str) -> EvidenceHyperedge:
    return EvidenceHyperedge(
        evidence_hyperedge_id=f"EH:{suffix}",
        claim_text=claim_text,
        claim_type="SYMPTOM_EXHIBITION",
        entities=[
            {"entity_id": "E:disorder:down_syndrome", "mention": "Down's syndrome", "role": "subject", "entity_type": "Disorder", "linking_confidence": 0.9},
            {"entity_id": "E:assessment:abc", "mention": "ABC", "role": "instrument", "entity_type": "Assessment_Tool", "linking_confidence": 0.85},
        ],
        qualifiers={},
        scope={"population": "children"},
        polarity="positive",
        negation=False,
        speculative=False,
        confidence=0.8,
        primary_triples=[{"head": "Down's syndrome", "relation": "elevates", "tail": "ABC", "confidence": 0.9}],
        supporting_triples=[],
        warnings=[],
        method="online_llm",
        source={"pack_id": f"PACK:{suffix}", "pack_hash": f"H:{suffix}", "member_claim_ids": [f"C:{suffix}"], "article_ids": ["a1"], "summary_ids": [f"SUM:{suffix}"]},
    )


class WriterIndexerAgentTest(unittest.TestCase):
    def test_writer_creates_evidence_and_canonical_hyperedges(self) -> None:
        tmp = tempfile.mkdtemp(prefix="hkg_writer_")
        emb = FakeEmbeddingService()
        kg = UnifiedKGIndex(tiny_unified_kg_payload(), embedding_service=emb)

        edges = [
            _make_edge("0001", "Children with Down's syndrome scored higher on ABC."),
            # Near-duplicate (same claim type + same entity ids).
            _make_edge("0002", "Children with Down's syndrome had elevated ABC scores."),
        ]

        agent = HyperKGWriterIndexerAgent(
            embedding_service=emb,
            writer=WriterConfig(canonical_similarity_threshold=0.0, canonical_jaccard_threshold=0.0),
        )
        report = HyperKGRunReport(run_id="test", started_at=0.0)
        paths = agent.write(
            output_dir=tmp,
            packs=[],
            evidence_hyperedges=edges,
            unified_kg_index=kg,
            summaries={},
            run_report=report,
            llm_usage=[],
            review_items=[],
        )

        # Read the written canonical hyperedges and confirm collapse.
        canonical_path = paths["canonical_hyperedges"]
        with open(canonical_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["support_count"], 2)
        self.assertEqual(set(lines[0]["member_evidence_hyperedges"]), {"EH:0001", "EH:0002"})


if __name__ == "__main__":
    unittest.main()
