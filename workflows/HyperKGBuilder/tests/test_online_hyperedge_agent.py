"""Tests for OnlineLLMHyperedgeAgent."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

from workflows.HyperKGBuilder.agents.online_hyperedge_agent import (
    OnlineLLMHyperedgeAgent,
)
from workflows.HyperKGBuilder.config import (
    BudgetConfig,
    HyperKGRunConfig,
    RoutingConfig,
)
from workflows.HyperKGBuilder.pack_hash import compute_pack_hash
from workflows.HyperKGBuilder.schemas import (
    CandidatePack,
    EntityCandidate,
    SummarySnippet,
    TripleCandidate,
)
from workflows.HyperKGBuilder.tests.conftest import (
    FakeChatLLM,
    _default_llm_payload,
    tiny_unified_kg_payload,
)
from workflows.HyperKGBuilder.unified_kg_index import UnifiedKGIndex
from workflows.HyperKGConstruction.review_queue import ReviewQueue


PROMPTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts.toml"
)


def _make_pack(decision: str = "online_llm", routing_reasons: List[str] | None = None) -> CandidatePack:
    pack = CandidatePack(
        pack_id="PACK:test",
        representative_claim_id="C:test",
        representative_claim_text="Risperidone may reduce hyperactivity in children.",
        claim_type="INTERVENTION_OUTCOME",
        member_claim_ids=["C:test"],
        entity_candidates=[
            EntityCandidate(
                entity_id="E:medication:risperidone",
                canonical_name="Risperidone",
                entity_type="Medication",
                aliases=[],
                normalized_id="RxNorm:35636",
                mention_count=7,
                source_articles=["other_article"],
                score=0.95,
                source="unified_kg",
            ),
        ],
        triple_candidates=[
            TripleCandidate(
                head="Risperidone",
                relation="treats",
                tail="Hyperactivity",
                confidence=0.9,
                relevance=0.5,
                clarity=0.5,
                evidence_count=2,
                conflict_status="consistent",
                source_articles=["other_article"],
                score=0.91,
            ),
        ],
        summary_snippets=[
            SummarySnippet(
                summary_id="SUM:test",
                article_id="other_article",
                segment_id="seg0001",
                summary_text="Risperidone may reduce hyperactivity",
                score=0.9,
            ),
        ],
        local_kg_snippet={"entities": [], "triples": []},
        routing_reasons=routing_reasons or ["negation_or_hedging"],
        auto_decision=decision,
    )
    pack.pack_hash = compute_pack_hash(pack)
    return pack


def _make_config(tmpdir: str, max_units: int = 100, max_retries: int = 1) -> HyperKGRunConfig:
    config = HyperKGRunConfig(
        claims_path="",
        unified_kg_path="",
        output_dir=tmpdir,
        prompts_file=PROMPTS_FILE,
    )
    config.budget = BudgetConfig(
        max_online_work_units_per_batch=max_units,
        max_online_work_unit_ratio=1.0,
        max_online_tokens_per_work_unit=4000,
        max_retries_per_work_unit=max_retries,
    )
    config.routing = RoutingConfig()
    return config


class OnlineLLMHyperedgeAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="online_hyperedge_")
        self.kg = UnifiedKGIndex(tiny_unified_kg_payload())
        self.queue = ReviewQueue()

    def test_auto_case_does_not_call_llm(self) -> None:
        pack = _make_pack(decision="auto", routing_reasons=["auto_high_confidence_anchor"])
        chat = FakeChatLLM()
        agent = OnlineLLMHyperedgeAgent(
            chat_llm=chat,
            config=_make_config(self.tmpdir),
            prompt_file=PROMPTS_FILE,
            cache_dir=os.path.join(self.tmpdir, "cache"),
            review_queue=self.queue,
        )
        edges, _ = agent.process([pack], self.kg)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].method, "auto_embedding_kg_anchor")
        self.assertEqual(chat.call_count, 0)

    def test_routed_pack_calls_llm(self) -> None:
        pack = _make_pack()
        chat = FakeChatLLM(responses=[{"json": _default_llm_payload()}])
        agent = OnlineLLMHyperedgeAgent(
            chat_llm=chat,
            config=_make_config(self.tmpdir),
            prompt_file=PROMPTS_FILE,
            cache_dir=os.path.join(self.tmpdir, "cache"),
            review_queue=self.queue,
        )
        edges, usage = agent.process([pack], self.kg)
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(edges[0].method, "online_llm")
        self.assertEqual(usage[0].parse_status, "ok")

    def test_retry_on_invalid_json(self) -> None:
        bad_payload = {"oops": True}  # missing all required keys
        good_payload = _default_llm_payload()
        chat = FakeChatLLM(
            responses=[
                {"content": "not json at all"},
                {"json": bad_payload},
                {"json": good_payload},
            ]
        )
        config = _make_config(self.tmpdir, max_retries=3)
        agent = OnlineLLMHyperedgeAgent(
            chat_llm=chat,
            config=config,
            prompt_file=PROMPTS_FILE,
            cache_dir=os.path.join(self.tmpdir, "cache"),
            review_queue=self.queue,
        )
        edges, usage = agent.process([_make_pack()], self.kg)
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(edges[0].method, "online_llm")
        self.assertEqual(usage[0].parse_status, "retry_succeeded")
        self.assertEqual(usage[0].attempt_count, 3)

    def test_concurrent_llm_calls_dispatch_in_parallel(self) -> None:
        """Concurrency >1 must let multiple LLM calls overlap in time."""
        import threading
        import time

        barrier = threading.Barrier(4)

        class _BarrierClient:
            """Stand-in for an OpenAI client that blocks on a barrier so that
            we can verify N calls are actually in flight at the same time."""

            def __init__(self) -> None:
                self.calls = 0
                self.peak_concurrent = 0
                self._inflight = 0
                self._lock = threading.Lock()
                self.chat = self
                self.completions = self

            def create(self, model, messages, temperature, **_):
                with self._lock:
                    self._inflight += 1
                    self.peak_concurrent = max(self.peak_concurrent, self._inflight)
                    self.calls += 1
                # Wait until 4 callers are simultaneously inside this method.
                barrier.wait(timeout=5.0)
                with self._lock:
                    self._inflight -= 1
                content = json.dumps(_default_llm_payload())
                resp = type("R", (), {})()
                resp.choices = [type("C", (), {"message": type("M", (), {"content": content})})()]
                resp.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})()
                return resp

        fake_client = _BarrierClient()
        chat = FakeChatLLM()  # only used so .model_name / .temperature exist
        chat.captured_prompts = []  # bypass _create-on-_FakeCompletions

        # Inject our barrier client into ChatLLM-like surface
        from workflows.HyperKGConstruction.llm import ChatLLM
        real_chat = ChatLLM(client=fake_client, model_name="fake-llm", temperature=0.0)

        config = _make_config(self.tmpdir, max_units=10, max_retries=0)
        config.budget.online_concurrency = 4

        agent = OnlineLLMHyperedgeAgent(
            chat_llm=real_chat,
            config=config,
            prompt_file=PROMPTS_FILE,
            cache_dir=os.path.join(self.tmpdir, "cache_concurrent"),
            review_queue=self.queue,
        )

        packs = [_make_pack() for _ in range(4)]
        # Each pack must have a unique hash so cache doesn't collapse them.
        for i, p in enumerate(packs):
            p.pack_id = f"PACK:test-{i}"
            p.member_claim_ids = [f"C:test-{i}"]
            p.pack_hash = compute_pack_hash(p)

        start = time.time()
        edges, usage = agent.process(packs, UnifiedKGIndex(tiny_unified_kg_payload()))
        duration = time.time() - start

        self.assertEqual(len(edges), 4)
        self.assertEqual(fake_client.calls, 4)
        self.assertEqual(fake_client.peak_concurrent, 4)
        # If calls had been serialized, the barrier would have deadlocked and
        # we'd have hit the 5s timeout per call → ~20s total. With concurrency=4
        # all 4 release together and total wall clock stays well under that.
        self.assertLess(duration, 4.5)

    def test_llm_json_schema_validation(self) -> None:
        # Missing 'confidence' must be rejected and retried; if all retries fail,
        # parse_status becomes 'failed'.
        broken = dict(_default_llm_payload())
        broken.pop("confidence")
        chat = FakeChatLLM(
            responses=[
                {"json": broken},
                {"json": broken},
            ]
        )
        config = _make_config(self.tmpdir, max_retries=1)
        agent = OnlineLLMHyperedgeAgent(
            chat_llm=chat,
            config=config,
            prompt_file=PROMPTS_FILE,
            cache_dir=os.path.join(self.tmpdir, "cache"),
            review_queue=self.queue,
        )
        edges, usage = agent.process([_make_pack()], self.kg)
        self.assertEqual(usage[0].parse_status, "failed")
        # The agent falls back to a deterministic edge with a warning.
        self.assertEqual(edges[0].method, "auto_embedding_kg_anchor")
        self.assertIn("llm_parse_failed", edges[0].warnings)


if __name__ == "__main__":
    unittest.main()
