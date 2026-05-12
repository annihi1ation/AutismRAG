"""Test fixtures for HyperKGBuilder tests.

This module is imported via plain unittest test modules — there is no pytest
fixture machinery; helpers below are plain factories.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from workflows.HyperKGConstruction.embedding_service import EmbeddingService

DEFAULT_DIM = 16


@dataclass
class FakeEmbeddingService:
    """Drop-in replacement for ``EmbeddingService`` that does not need torch.

    Vectors are deterministic per text (md5-derived), so semantically identical
    inputs cluster together.
    """

    enabled: bool = True
    model_name: str = "fake-embedding-model"
    dim: int = DEFAULT_DIM

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.md5((t or "").encode("utf-8")).digest()
            for j in range(self.dim):
                out[i, j] = (digest[j % len(digest)] / 255.0) * 2 - 1
        # Normalize.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norms, 1e-12)
        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]


@dataclass
class FakeChatLLM:
    """Records calls and returns scripted responses."""

    responses: List[Dict[str, Any]] = field(default_factory=list)
    captured_prompts: List[Dict[str, str]] = field(default_factory=list)
    model_name: str = "fake-llm"
    temperature: float = 0.1
    max_retries: int = 0
    cursor: int = 0

    @property
    def call_count(self) -> int:
        return len(self.captured_prompts)

    @property
    def client(self):
        return _FakeClient(self)


class _FakeClient:
    def __init__(self, parent: "FakeChatLLM"):
        self.parent = parent
        self.chat = _FakeCompletions(parent)


class _FakeCompletions:
    def __init__(self, parent: "FakeChatLLM"):
        self.parent = parent
        self.completions = self


class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeResponse:
    def __init__(self, content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


def _create(self, model: str, messages: List[Dict[str, str]], temperature: float, **_: Any):
    parent: FakeChatLLM = self.parent
    parent.captured_prompts.append({
        "system": messages[0]["content"] if messages else "",
        "user": messages[1]["content"] if len(messages) > 1 else "",
    })
    if not parent.responses:
        return _FakeResponse(json.dumps(_default_llm_payload()))
    if parent.cursor >= len(parent.responses):
        scripted = parent.responses[-1]
    else:
        scripted = parent.responses[parent.cursor]
        parent.cursor += 1
    content = scripted.get("content")
    if content is None:
        payload = scripted.get("json")
        content = json.dumps(payload) if payload is not None else ""
    return _FakeResponse(
        content,
        prompt_tokens=int(scripted.get("prompt_tokens", 120)),
        completion_tokens=int(scripted.get("completion_tokens", 80)),
    )


_FakeCompletions.create = _create


def _default_llm_payload() -> Dict[str, Any]:
    return {
        "evidence_hyperedges": [{"id": "EH:1"}],
        "canonical_hyperedge": {"canonical_claim": "test", "claim_type": "TEST"},
        "selected_canonical_entities": [{"entity_id": "E:test", "canonical_name": "Test"}],
        "primary_triples": [{"head": "A", "relation": "rel", "tail": "B", "confidence": 0.9}],
        "supporting_triples": [],
        "entity_roles": [{"entity_id": "E:test", "role": "subject"}],
        "qualifiers": {},
        "polarity": "positive",
        "negation": False,
        "speculative": False,
        "confidence": 0.85,
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Tiny fixtures
# ---------------------------------------------------------------------------


def tiny_unified_kg_payload() -> Dict[str, Any]:
    return {
        "entities": [
            {
                "canonical_name": "Down's syndrome",
                "entity_type": "Disorder",
                "normalized_id": "Q170990",
                "aliases": ["DS", "trisomy 21"],
                "source_articles": ["10.1046_j.1365-2788.1998.00123.x"],
                "mention_count": 5,
            },
            {
                "canonical_name": "Aberrant Behavior Checklist",
                "entity_type": "Assessment_Tool",
                "normalized_id": "N/A",
                "aliases": ["ABC"],
                "source_articles": ["10.1046_j.1365-2788.1998.00123.x"],
                "mention_count": 3,
            },
            {
                "canonical_name": "Risperidone",
                "entity_type": "Medication",
                "normalized_id": "RxNorm:35636",
                "aliases": [],
                "source_articles": ["other_article"],
                "mention_count": 7,
            },
        ],
        "triples": [
            {
                "head": "children",
                "relation": "has_diagnosis",
                "tail": "Down's syndrome",
                "confidence": 1.0,
                "relevance": 0.6,
                "clarity": 0.7,
                "source_articles": ["10.1046_j.1365-2788.1998.00123.x"],
                "evidence_count": 6,
                "conflict_status": "consistent",
            },
            {
                "head": "Risperidone",
                "relation": "treats",
                "tail": "Hyperactivity",
                "confidence": 0.55,
                "relevance": 0.4,
                "clarity": 0.5,
                "source_articles": ["other_article"],
                "evidence_count": 1,
                "conflict_status": "unresolved",
            },
        ],
        "metadata": {
            "domain": "test",
            "pipeline_version": "test-0.0",
            "total_source_articles": 2,
        },
        "statistics": {
            "relation_distribution": {"has_diagnosis": 1, "treats": 1},
            "entity_type_distribution": {"Disorder": 1, "Medication": 1, "Assessment_Tool": 1},
        },
    }


def tiny_claim_row(claim_id: str, claim_text: str, claim_type: str = "SYMPTOM_EXHIBITION", **kwargs) -> Dict[str, Any]:
    candidate_entities = kwargs.get("candidate_entities") or [
        "{'mention': \"Down's syndrome\", 'entity_id': 'E:disorder:down_syndrome',"
        " 'canonical_name': \"Down's syndrome\"}",
    ]
    metadata = kwargs.get("metadata") or {
        "article_id": "10.1046_j.1365-2788.1998.00123.x",
        "segment_id": "seg0001",
        "summary_id": "SUM:test:seg0001",
    }
    raw_payload = json.dumps({
        "claims": [{
            "claim_text": claim_text,
            "claim_type": claim_type,
            "polarity": kwargs.get("polarity", "positive"),
            "modality": kwargs.get("modality", "comparative"),
            "scope": kwargs.get("scope", {"population": "children"}),
            "supporting_summary_span": kwargs.get("summary_span", "supporting span text"),
        }],
    })
    return {
        "claims": [{
            "candidate_entities": candidate_entities,
            "candidate_triples": kwargs.get("candidate_triples") or [],
            "claim_id": claim_id,
            "claim_text": claim_text,
            "claim_type": claim_type,
            "metadata": metadata,
            "packet_id": "P:test:seg0001",
            "raw_llm_output": raw_payload,
            "source_segment_id": "seg0001",
            "source_summary_id": "SUM:test:seg0001",
        }],
        "index": 0,
        "object_id": "P:test:seg0001",
        "review_items": [],
        "selected_claim_id": claim_id,
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def make_temp_dir() -> str:
    return tempfile.mkdtemp(prefix="hyperkg_builder_test_")
