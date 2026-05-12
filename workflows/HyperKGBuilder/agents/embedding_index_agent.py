"""ClaimKGEmbeddingIndexAgent — builds dense vector indexes for claims,
unified-KG entities, unified-KG triples, and segment summaries.

Does NOT call the Online LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from workflows.HyperKGConstruction.embedding_service import (
    EmbeddingService,
    entity_embedding_text,
)
from workflows.HyperKGConstruction.vector_index import VectorIndexAdapter

from ..schemas import ClaimRecord
from ..unified_kg_index import UnifiedKGIndex, triple_embedding_text

logger = logging.getLogger(__name__)


CLAIMS_NS = "claims"
ENTITIES_NS = "entities"
TRIPLES_NS = "triples"
SUMMARIES_NS = "summaries"


def claim_embedding_text(claim: ClaimRecord) -> str:
    entity_names = ", ".join(
        str(e.get("canonical_name") or e.get("mention") or e.get("entity_id") or "")
        for e in claim.candidate_entities
    )
    triple_strs = "; ".join(
        f"{t.get('head', '')}--{t.get('relation', '')}->{t.get('tail', '')}"
        for t in claim.candidate_triples
    )
    snippet = claim.summary_snippet or ""
    return (
        f"Claim: {claim.claim_text} | Type: {claim.claim_type} | "
        f"Polarity: {claim.polarity} | Modality: {claim.modality} | "
        f"Summary: {snippet} | Entities: {entity_names} | Triples: {triple_strs}"
    )


def summary_index_text(record: Dict[str, Any]) -> str:
    return (
        f"Summary: {record.get('summary_text', '')} | "
        f"article={record.get('article_id', '')} | "
        f"segment={record.get('segment_id', '')}"
    )


@dataclass
class EmbeddingIndexManifest:
    namespaces: Dict[str, int] = field(default_factory=dict)
    embedding_model: str = ""
    embedding_dim: int = 0


@dataclass
class ClaimKGEmbeddingIndexAgent:
    """Build a single VectorIndexAdapter populated with four namespaces."""

    embedding_service: Optional[EmbeddingService]

    def build(
        self,
        claims: Iterable[ClaimRecord],
        unified_kg_index: UnifiedKGIndex,
        summaries: Dict[str, Dict[str, Any]],
    ) -> tuple[VectorIndexAdapter, EmbeddingIndexManifest]:
        adapter = VectorIndexAdapter(embedding_service=self.embedding_service)
        manifest = EmbeddingIndexManifest()

        if self.embedding_service is None or not self.embedding_service.enabled:
            logger.warning(
                "Embedding service is disabled; vector indexes will be empty. "
                "Provide --embedding-model-name to enable retrieval."
            )
            return adapter, manifest

        manifest.embedding_model = self.embedding_service.model_name

        # ---- Claims -------------------------------------------------------
        claim_list = list(claims)
        if claim_list:
            ids = [c.claim_id for c in claim_list]
            texts = [claim_embedding_text(c) for c in claim_list]
            metadata = [
                {
                    "claim_type": c.claim_type,
                    "article_id": c.article_id,
                    "summary_id": c.summary_id,
                    "polarity": c.polarity,
                    "modality": c.modality,
                }
                for c in claim_list
            ]
            vectors = self.embedding_service.encode_texts(texts)
            adapter.add(CLAIMS_NS, ids, texts, vectors, metadata)
            manifest.namespaces[CLAIMS_NS] = len(ids)
            manifest.embedding_dim = int(vectors.shape[1]) if vectors.size else 0

        # ---- Entities -----------------------------------------------------
        entities = unified_kg_index.entities
        if entities:
            ids = [e["entity_id"] for e in entities]
            texts = [entity_embedding_text(e) for e in entities]
            metadata = [
                {
                    "entity_type": e.get("entity_type"),
                    "normalized_id": e.get("normalized_id"),
                    "mention_count": e.get("mention_count"),
                }
                for e in entities
            ]
            vectors = self.embedding_service.encode_texts(texts)
            adapter.add(ENTITIES_NS, ids, texts, vectors, metadata)
            manifest.namespaces[ENTITIES_NS] = len(ids)

        # ---- Triples ------------------------------------------------------
        triples = unified_kg_index.triples
        if triples:
            ids: List[str] = []
            texts: List[str] = []
            metadata: List[Dict[str, Any]] = []
            for t in triples:
                ids.append(t["triple_id"])
                texts.append(triple_embedding_text(t))
                metadata.append(
                    {
                        "relation": t.get("relation"),
                        "confidence": t.get("confidence"),
                        "relevance": t.get("relevance"),
                        "evidence_count": t.get("evidence_count"),
                        "conflict_status": t.get("conflict_status"),
                    }
                )
            vectors = self.embedding_service.encode_texts(texts)
            adapter.add(TRIPLES_NS, ids, texts, vectors, metadata)
            manifest.namespaces[TRIPLES_NS] = len(ids)

        # ---- Summaries ----------------------------------------------------
        if summaries:
            items = [v for v in summaries.values() if v.get("summary_text")]
            if items:
                ids = [str(v["summary_id"]) for v in items]
                texts = [summary_index_text(v) for v in items]
                metadata = [
                    {
                        "article_id": v.get("article_id", ""),
                        "segment_id": v.get("segment_id", ""),
                    }
                    for v in items
                ]
                vectors = self.embedding_service.encode_texts(texts)
                adapter.add(SUMMARIES_NS, ids, texts, vectors, metadata)
                manifest.namespaces[SUMMARIES_NS] = len(ids)

        # Pre-build internal KG vectors so router-time triple/entity top-k
        # is fast and uses the same encoder.
        unified_kg_index.build_vectors()

        return adapter, manifest
