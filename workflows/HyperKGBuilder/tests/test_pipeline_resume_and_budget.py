"""Resume + budget guard + cross-agent invariants."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import List

from workflows.HyperKGBuilder.agents.online_hyperedge_agent import (
    OnlineLLMHyperedgeAgent,
)
from workflows.HyperKGBuilder.config import (
    BudgetConfig,
    HyperKGRunConfig,
    RoutingConfig,
)
from workflows.HyperKGBuilder.pack_hash import compute_pack_hash
from workflows.HyperKGBuilder.pipeline import HyperKGBuilderPipeline
from workflows.HyperKGBuilder.schemas import (
    CandidatePack,
    EntityCandidate,
    SummarySnippet,
    TripleCandidate,
)
from workflows.HyperKGBuilder.tests.conftest import (
    FakeChatLLM,
    FakeEmbeddingService,
    _default_llm_payload,
    tiny_claim_row,
    tiny_unified_kg_payload,
    write_json,
    write_jsonl,
)
from workflows.HyperKGBuilder.unified_kg_index import UnifiedKGIndex
from workflows.HyperKGConstruction.review_queue import ReviewQueue


PROMPTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts.toml"
)


def _ambiguous_pack(idx: int) -> CandidatePack:
    pack = CandidatePack(
        pack_id=f"PACK:test-{idx}",
        representative_claim_id=f"C:test-{idx}",
        representative_claim_text=f"Risperidone might possibly reduce X (case {idx}).",
        claim_type="INTERVENTION_OUTCOME",
        member_claim_ids=[f"C:test-{idx}"],
        entity_candidates=[
            EntityCandidate(
                entity_id="E:medication:risperidone",
                canonical_name="Risperidone",
                entity_type="Medication",
                aliases=[],
                normalized_id="RxNorm:35636",
                mention_count=7,
                source_articles=["x"],
                score=0.7,
                source="unified_kg",
            ),
        ],
        triple_candidates=[],
        summary_snippets=[
            SummarySnippet(
                summary_id=f"SUM:art:seg-{idx}",
                article_id="art",
                segment_id=f"seg-{idx}",
                summary_text="possible effect",
                score=0.5,
            )
        ],
        local_kg_snippet={"entities": [], "triples": []},
        routing_reasons=["negation_or_hedging", "high_impact_claim"],
        auto_decision="online_llm",
    )
    pack.pack_hash = compute_pack_hash(pack)
    return pack


class BudgetGuardTest(unittest.TestCase):
    def test_budget_guard_limits_online_calls(self) -> None:
        tmp = tempfile.mkdtemp(prefix="hkg_budget_")
        kg = UnifiedKGIndex(tiny_unified_kg_payload())
        chat = FakeChatLLM(responses=[{"json": _default_llm_payload()}])
        cfg = HyperKGRunConfig(claims_path="", unified_kg_path="", output_dir=tmp, prompts_file=PROMPTS_FILE)
        cfg.budget = BudgetConfig(
            max_online_work_units_per_batch=2,
            max_online_work_unit_ratio=1.0,
            max_online_tokens_per_work_unit=4000,
            max_retries_per_work_unit=0,
        )
        queue = ReviewQueue()
        agent = OnlineLLMHyperedgeAgent(
            chat_llm=chat,
            config=cfg,
            prompt_file=PROMPTS_FILE,
            cache_dir=os.path.join(tmp, "cache"),
            review_queue=queue,
        )
        packs = [_ambiguous_pack(i) for i in range(5)]
        edges, _ = agent.process(packs, kg)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(len(edges), 5)  # 2 online, 3 fallback
        # Reasoning queue must contain budget_exhausted entries for the 3 leftovers.
        self.assertEqual(
            sum(1 for item in queue.to_list() if item.get("reason") == "budget_exhausted"),
            3,
        )


class ResumeAndInvariantsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hkg_resume_")
        self.claims_path = os.path.join(self.tmp, "claims.jsonl")
        self.kg_path = os.path.join(self.tmp, "unified_kg.json")

        rows = [
            tiny_claim_row(
                "C:hi:0001",
                "Children with Down's syndrome had elevated ABC scores.",
                claim_type="SYMPTOM_EXHIBITION",
                scope={
                    "population": "children",
                    "severity": "elevated",
                    "developmental_stage": "childhood",
                    "dose_or_intensity": "n/a",
                    "timepoint": "study",
                    "comparator": "controls",
                },
            ),
            tiny_claim_row(
                "C:hi:0002",
                "Risperidone might possibly reduce hyperactivity.",
                claim_type="INTERVENTION_OUTCOME",
                candidate_entities=[
                    "{'mention': 'Risperidone', 'entity_id': 'E:medication:risperidone',"
                    " 'canonical_name': 'Risperidone'}",
                ],
            ),
        ]
        write_jsonl(self.claims_path, rows)
        write_json(self.kg_path, tiny_unified_kg_payload())

    def _build_pipeline(self, chat: FakeChatLLM) -> HyperKGBuilderPipeline:
        cfg = HyperKGRunConfig(
            claims_path=self.claims_path,
            unified_kg_path=self.kg_path,
            output_dir=self.tmp,
            prompts_file=PROMPTS_FILE,
        )
        cfg.budget = BudgetConfig(
            max_online_work_units_per_batch=10,
            max_online_work_unit_ratio=1.0,
            max_retries_per_work_unit=0,
        )
        return HyperKGBuilderPipeline(
            config=cfg,
            embedding_service=FakeEmbeddingService(),
            chat_llm=chat,
        )

    def test_unified_kg_not_passed_to_llm(self) -> None:
        # Inflate the KG with thousands of sentinel entities/triples. The
        # bounded retrieval must keep the prompt small and entity-count
        # capped regardless of how big the KG grows.
        kg_payload = tiny_unified_kg_payload()
        for i in range(2000):
            kg_payload["entities"].append({
                "canonical_name": f"Unrelated_Entity_{i}",
                "entity_type": "Unrelated_Type",
                "normalized_id": f"sentinel-{i}",
                "aliases": [f"alt-{i}"],
                "source_articles": [f"unrelated_article_{i}"],
                "mention_count": 1,
            })
            kg_payload["triples"].append({
                "head": f"Unrelated_Entity_{i}",
                "relation": "associated_with",
                "tail": f"Unrelated_Entity_{(i + 1) % 2000}",
                "confidence": 0.5,
                "relevance": 0.5,
                "clarity": 0.5,
                "source_articles": [f"unrelated_article_{i}"],
                "evidence_count": 1,
                "conflict_status": "consistent",
            })
        write_json(self.kg_path, kg_payload)

        chat = FakeChatLLM(responses=[{"json": _default_llm_payload()}])
        pipeline = self._build_pipeline(chat)
        pipeline.run()
        self.assertGreaterEqual(chat.call_count, 1)

        with open(self.kg_path, "r", encoding="utf-8") as f:
            kg_blob = f.read()

        routing = pipeline.config.routing
        # Loose upper bound on candidate-entity count surfaced in the prompt.
        # (Two mentions per claim × per-mention cap, plus top-k expansion.)
        max_entities_in_prompt = (
            routing.max_candidate_entities_per_mention * 4 + routing.entity_top_k + 2
        )
        for record in chat.captured_prompts:
            blob = record.get("user", "")
            # Prompt must be many times smaller than the raw KG.
            self.assertLess(len(blob), len(kg_blob) // 50)
            # Bounded entity count: count occurrences of the entity_id key,
            # which appears once per candidate entity in the JSON block.
            self.assertLess(blob.count("\"entity_id\":"), max_entities_in_prompt)
            # No verbatim sentinel article id list.
            self.assertNotIn("unrelated_article_500", blob)

    def test_resume_skips_completed_work_units(self) -> None:
        # First run: process all packs.
        chat1 = FakeChatLLM(responses=[{"json": _default_llm_payload()}])
        pipeline1 = self._build_pipeline(chat1)
        report1 = pipeline1.run()
        first_calls = chat1.call_count
        self.assertGreaterEqual(first_calls, 1)

        # Second run with the same fingerprint should resume — no LLM calls.
        chat2 = FakeChatLLM(responses=[{"json": _default_llm_payload()}])
        pipeline2 = self._build_pipeline(chat2)
        report2 = pipeline2.run()
        self.assertTrue(report2.resumed)
        self.assertEqual(chat2.call_count, 0)


if __name__ == "__main__":
    unittest.main()
