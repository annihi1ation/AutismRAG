from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

from workflows.HyperKGConstruction.config import HyperKGConfig
from workflows.HyperKGConstruction.pipeline import HyperKGPipeline


class FakeChatCompletions:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float) -> Any:
        del model, temperature
        self.calls += 1
        if self.fail:
            raise AssertionError("LLM should not be called during full checkpoint resume")
        system = messages[0]["content"]
        if "decompose a segment-level summary" in system:
            payload = {
                "claims": [
                    {
                        "claim_text": "Alpha improves Beta.",
                        "claim_type": "INTERVENTION_OUTCOME",
                        "candidate_entities": ["Alpha", "Beta"],
                    }
                ]
            }
        elif "Convert one atomic claim" in system:
            payload = {
                "claim_text": "Alpha improves Beta.",
                "claim_type": "INTERVENTION_OUTCOME",
                "entities": [
                    {
                        "entity_id": "E:intervention:alpha",
                        "mention": "Alpha",
                        "role": "intervention",
                        "entity_type": "intervention",
                        "linking_confidence": 1.0,
                    },
                    {
                        "entity_id": "E:outcome:beta",
                        "mention": "Beta",
                        "role": "outcome",
                        "entity_type": "outcome",
                        "linking_confidence": 1.0,
                    },
                ],
                "triple_projections": [
                    {
                        "head_entity_id": "E:intervention:alpha",
                        "relation": "improves",
                        "tail_entity_id": "E:outcome:beta",
                        "support": "Alpha improves Beta.",
                        "confidence": 0.9,
                    }
                ],
                "scores": {
                    "entity_linking": 1.0,
                    "local_kg_agreement": 1.0,
                    "nary_completeness": 1.0,
                },
            }
        elif "strict quality critic" in system:
            payload = {
                "decision": "ACCEPT",
                "scores": {
                    "faithfulness": 1.0,
                    "entity_linking": 1.0,
                    "local_kg_agreement": 1.0,
                    "scope_completeness": 1.0,
                    "relation_correctness": 1.0,
                },
                "warnings": [],
                "violations": [],
            }
        else:
            payload = {"decision": "NEW"}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )


class FakeClient:
    def __init__(self, fail: bool = False):
        completions = FakeChatCompletions(fail=fail)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def sample_inputs() -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    summaries = [
        {
            "article_id": "A",
            "segment_id": "seg0001",
            "summary_id": "SUM:A:seg0001",
            "summary_text": "Alpha improves Beta.",
        },
        {
            "article_id": "A",
            "segment_id": "seg0002",
            "summary_id": "SUM:A:seg0002",
            "summary_text": "Alpha improves Beta again.",
        },
    ]
    local_kgs = {
        "A": {
            "article_id": "A",
            "entities": [
                {"name": "Alpha", "entity_type": "intervention"},
                {"name": "Beta", "entity_type": "outcome"},
            ],
            "triples": [{"head": "Alpha", "relation": "improves", "tail": "Beta"}],
        }
    }
    unified_kg = {
        "canonical_entities": [
            {"canonical_name": "Alpha", "entity_type": "intervention", "aliases": ["Alpha"]},
            {"canonical_name": "Beta", "entity_type": "outcome", "aliases": ["Beta"]},
        ]
    }
    return summaries, local_kgs, unified_kg


def build_config(workers: int) -> HyperKGConfig:
    return HyperKGConfig(
        embedding_enabled=False,
        enable_vector_index=False,
        progress_enabled=False,
        max_workers=workers,
    )


class PipelineRuntimeTest(unittest.TestCase):
    def test_parallel_and_serial_outputs_keep_order(self) -> None:
        summaries, local_kgs, unified_kg = sample_inputs()
        with tempfile.TemporaryDirectory() as serial_dir, tempfile.TemporaryDirectory() as parallel_dir:
            serial = HyperKGPipeline(
                FakeClient(),
                model_name="fake/model",
                output_dir=serial_dir,
                config=build_config(workers=1),
            ).run(summaries, local_kgs, unified_kg)
            parallel = HyperKGPipeline(
                FakeClient(),
                model_name="fake/model",
                output_dir=parallel_dir,
                config=build_config(workers=4),
            ).run(summaries, local_kgs, unified_kg)

        self.assertEqual(
            [row["claim_id"] for row in serial["claims"]],
            [row["claim_id"] for row in parallel["claims"]],
        )
        self.assertEqual(
            [row["evidence_hyperedge_id"] for row in serial["evidence_hyperedges"]],
            [row["evidence_hyperedge_id"] for row in parallel["evidence_hyperedges"]],
        )
        self.assertEqual(serial["run_stats"]["claim_count"], parallel["run_stats"]["claim_count"])

    def test_resume_skips_completed_llm_stages(self) -> None:
        summaries, local_kgs, unified_kg = sample_inputs()
        with tempfile.TemporaryDirectory() as output_dir:
            first = HyperKGPipeline(
                FakeClient(),
                model_name="fake/model",
                output_dir=output_dir,
                config=build_config(workers=2),
            ).run(summaries, local_kgs, unified_kg)
            second = HyperKGPipeline(
                FakeClient(fail=True),
                model_name="fake/model",
                output_dir=output_dir,
                config=build_config(workers=4),
            ).run(summaries, local_kgs, unified_kg)

        self.assertEqual(first["run_stats"]["claim_count"], second["run_stats"]["claim_count"])
        self.assertEqual(second["run_stats"]["checkpoint"]["resumed"], True)
        self.assertEqual(second["run_stats"]["phase_timings"]["claim_split"]["skipped"], 2)
        self.assertEqual(second["run_stats"]["phase_timings"]["compose"]["skipped"], 2)
        self.assertEqual(second["run_stats"]["phase_timings"]["critic"]["skipped"], 2)


if __name__ == "__main__":
    unittest.main()
