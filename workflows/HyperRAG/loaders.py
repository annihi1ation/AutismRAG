"""Load existing HyperKGBuilder artefacts into in-memory structures."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np

from workflows.HyperKGConstruction.config import EmbeddingConfig
from workflows.HyperKGConstruction.embedding_service import EmbeddingService
from workflows.HyperKGConstruction.llm import (
    ChatLLM,
    OpenRouterConfigurationError,
    build_openrouter_client,
)
from workflows.HyperKGConstruction.prompt_store import load_prompts
from workflows.HyperKGConstruction.vector_index import VectorIndexAdapter

from .config import HyperRAGConfig

logger = logging.getLogger(__name__)


@dataclass
class IncidenceIndex:
    entity_to_hyperedges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    hyperedge_to_entities: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    hyperedge_to_chunks: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    chunk_to_hyperedges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    entity_role_in_hyperedge: Dict[tuple, str] = field(default_factory=dict)


@dataclass
class HyperRAGContext:
    config: HyperRAGConfig
    hyperedge_store: Dict[str, Dict[str, Any]]
    entity_index: VectorIndexAdapter
    hyperedge_index: VectorIndexAdapter
    chunk_index: VectorIndexAdapter
    incidence: IncidenceIndex
    embedding_service: EmbeddingService
    llm: Optional[ChatLLM]
    prompts: Dict[str, Dict[str, str]]
    rerank_llm: Optional[ChatLLM] = None


def load_vector_index_namespace(
    indexes_dir: Path,
    namespace: str,
    embedding_service: EmbeddingService,
) -> VectorIndexAdapter:
    """Materialise a VectorIndexAdapter from the on-disk save() format."""

    indexes_dir = Path(indexes_dir)
    metadata_path = indexes_dir / f"{namespace}.metadata.jsonl"
    vectors_path = indexes_dir / f"{namespace}.vectors.npy"
    if not metadata_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(
            f"Index files for namespace '{namespace}' not found in {indexes_dir}"
        )

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    vectors = np.load(vectors_path).astype(np.float32, copy=False)
    if vectors.shape[0] != len(records):
        raise ValueError(
            f"Namespace '{namespace}': vectors ({vectors.shape[0]}) and "
            f"records ({len(records)}) length mismatch."
        )

    adapter = VectorIndexAdapter(embedding_service=embedding_service)
    adapter.records[namespace] = records
    adapter.vectors[namespace] = vectors
    logger.info("Loaded namespace '%s': %d records", namespace, len(records))
    return adapter


def load_hyperedge_store(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    store: Dict[str, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            edge = record.get("edge")
            if not edge:
                continue
            edge_id = edge.get("evidence_hyperedge_id")
            if not edge_id:
                continue
            store[edge_id] = edge
    logger.info("Loaded %d hyperedges from %s", len(store), jsonl_path)
    return store


def load_incidence(
    incidence_path: Path,
    summary_links_path: Path,
    hyperedge_store: Dict[str, Dict[str, Any]],
) -> IncidenceIndex:
    incidence = IncidenceIndex()

    with open(incidence_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") != "PARTICIPATES_IN":
                continue
            entity_id = row.get("from")
            edge_id = row.get("to")
            if not entity_id or not edge_id:
                continue
            incidence.entity_to_hyperedges[entity_id].add(edge_id)
            incidence.hyperedge_to_entities[edge_id].add(entity_id)
            role = row.get("role") or ""
            if role:
                incidence.entity_role_in_hyperedge[(entity_id, edge_id)] = role

    if Path(summary_links_path).exists():
        with open(summary_links_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                edge_id = row.get("evidence_hyperedge_id")
                chunk_id = row.get("summary_id")
                if not edge_id or not chunk_id:
                    continue
                incidence.hyperedge_to_chunks[edge_id].add(chunk_id)
                incidence.chunk_to_hyperedges[chunk_id].add(edge_id)

    # Backstop from edge.source.summary_ids — fills in any gaps.
    for edge_id, edge in hyperedge_store.items():
        for sid in (edge.get("source") or {}).get("summary_ids") or []:
            incidence.hyperedge_to_chunks[edge_id].add(sid)
            incidence.chunk_to_hyperedges[sid].add(edge_id)

    logger.info(
        "Incidence: %d entity->edges, %d edge->entities, %d edge->chunks",
        len(incidence.entity_to_hyperedges),
        len(incidence.hyperedge_to_entities),
        len(incidence.hyperedge_to_chunks),
    )
    return incidence


def _try_build_llm(
    config: HyperRAGConfig,
    model_name: Optional[str] = None,
) -> Optional[ChatLLM]:
    try:
        client = build_openrouter_client()
    except OpenRouterConfigurationError as err:
        logger.warning("LLM disabled (%s). Falling back to stub mode.", err)
        return None
    except Exception as err:  # noqa: BLE001
        logger.warning("LLM disabled (unexpected error: %s). Falling back to stub mode.", err)
        return None
    return ChatLLM(
        client=client,
        model_name=model_name or config.llm_model_name,
        temperature=config.llm_temperature,
        max_retries=1,
    )


def build_context(config: HyperRAGConfig) -> HyperRAGContext:
    emb_cfg = EmbeddingConfig(
        enabled=True,
        model_name=config.embedding_model_name,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
    )
    embedding_service = EmbeddingService(emb_cfg)

    entity_index = load_vector_index_namespace(
        config.vector_indexes_dir, config.entity_namespace, embedding_service
    )
    hyperedge_index = load_vector_index_namespace(
        config.vector_indexes_dir, config.hyperedge_namespace, embedding_service
    )
    chunk_index = load_vector_index_namespace(
        config.vector_indexes_dir, config.chunk_namespace, embedding_service
    )

    hyperedge_store = load_hyperedge_store(config.hyperedge_jsonl)
    incidence = load_incidence(
        config.incidence_jsonl, config.summary_links_jsonl, hyperedge_store
    )

    prompts = load_prompts(str(config.prompt_file))
    llm = _try_build_llm(config)
    rerank_llm: Optional[ChatLLM] = None
    if config.enable_llm_rerank and llm is not None:
        rerank_llm = _try_build_llm(config, model_name=config.rerank_llm_model_name)

    return HyperRAGContext(
        config=config,
        hyperedge_store=hyperedge_store,
        entity_index=entity_index,
        hyperedge_index=hyperedge_index,
        chunk_index=chunk_index,
        incidence=incidence,
        embedding_service=embedding_service,
        llm=llm,
        prompts=prompts,
        rerank_llm=rerank_llm,
    )
