"""Bounded retrieval API over the unified KG.

The full unified KG must NEVER be passed to the Online LLM. All access goes
through this read-only index, which exposes only bounded lookups + small
distribution summaries.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from workflows.HyperKGConstruction.embedding_service import (
    EmbeddingService,
    entity_embedding_text,
)
from workflows.HyperKGConstruction.utils import normalize_text, slugify

logger = logging.getLogger(__name__)

_ENTITY_NS = "kg_entities"
_TRIPLE_NS = "kg_triples"


def triple_embedding_text(triple: Dict[str, Any]) -> str:
    head = triple.get("head") or ""
    relation = triple.get("relation") or ""
    tail = triple.get("tail") or ""
    head_type = triple.get("head_type") or ""
    tail_type = triple.get("tail_type") or ""
    confidence = triple.get("confidence")
    relevance = triple.get("relevance")
    clarity = triple.get("clarity")
    evidence_count = triple.get("evidence_count")
    conflict = triple.get("conflict_status")
    return (
        f"{head}({head_type}) --{relation}--> {tail}({tail_type}) | "
        f"conf={confidence} rel={relevance} clarity={clarity} "
        f"evid={evidence_count} conflict={conflict}"
    )


def _kg_entity_id(entity: Dict[str, Any]) -> str:
    nid = entity.get("normalized_id") or ""
    if nid and nid != "N/A":
        return f"E:{slugify(nid)}"
    name = entity.get("canonical_name") or ""
    etype = entity.get("entity_type") or "unknown"
    return f"E:{slugify(etype)}:{slugify(name)}"


def _triple_id(triple: Dict[str, Any]) -> str:
    return (
        f"T:{slugify(triple.get('head', ''))}:"
        f"{slugify(triple.get('relation', ''))}:"
        f"{slugify(triple.get('tail', ''))}"
    )


class UnifiedKGIndex:
    """Read-only access over the unified KG, with optional vector retrieval."""

    def __init__(self, payload: Dict[str, Any], embedding_service: Optional[EmbeddingService] = None):
        self._raw_entities: List[Dict[str, Any]] = list(payload.get("entities") or [])
        self._raw_triples: List[Dict[str, Any]] = list(payload.get("triples") or [])
        self._metadata: Dict[str, Any] = dict(payload.get("metadata") or {})
        self._statistics: Dict[str, Any] = dict(payload.get("statistics") or {})
        self.embedding_service = embedding_service

        self._entity_ids: List[str] = []
        self._triple_ids: List[str] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._by_name: Dict[str, List[str]] = {}
        self._triples_by_head: Dict[str, List[str]] = {}
        self._triples_by_tail: Dict[str, List[str]] = {}
        self._triple_by_id: Dict[str, Dict[str, Any]] = {}

        self._entity_vectors: Optional[np.ndarray] = None
        self._triple_vectors: Optional[np.ndarray] = None

        self._build_lookups()

    # ----------- Construction --------------------------------------------
    def _build_lookups(self) -> None:
        for entity in self._raw_entities:
            eid = _kg_entity_id(entity)
            entity = dict(entity)
            entity["entity_id"] = eid
            self._by_id[eid] = entity
            self._entity_ids.append(eid)
            keys = [normalize_text(entity.get("canonical_name") or "")]
            for alias in entity.get("aliases") or []:
                keys.append(normalize_text(alias))
            for key in {k for k in keys if k}:
                self._by_name.setdefault(key, []).append(eid)

        for triple in self._raw_triples:
            tid = _triple_id(triple)
            triple = dict(triple)
            triple["triple_id"] = tid
            self._triple_by_id[tid] = triple
            self._triple_ids.append(tid)
            head_key = normalize_text(triple.get("head") or "")
            tail_key = normalize_text(triple.get("tail") or "")
            if head_key:
                self._triples_by_head.setdefault(head_key, []).append(tid)
            if tail_key:
                self._triples_by_tail.setdefault(tail_key, []).append(tid)

    @classmethod
    def from_path(cls, path: str, embedding_service: Optional[EmbeddingService] = None) -> "UnifiedKGIndex":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload, embedding_service=embedding_service)

    # ----------- Vector index --------------------------------------------
    def build_vectors(self) -> None:
        if self.embedding_service is None or not self.embedding_service.enabled:
            return
        if self._entity_vectors is None:
            texts = [entity_embedding_text(self._by_id[eid]) for eid in self._entity_ids]
            self._entity_vectors = (
                self.embedding_service.encode_texts(texts) if texts else np.zeros((0, 1), dtype=np.float32)
            )
        if self._triple_vectors is None:
            texts = [triple_embedding_text(self._triple_by_id[tid]) for tid in self._triple_ids]
            self._triple_vectors = (
                self.embedding_service.encode_texts(texts) if texts else np.zeros((0, 1), dtype=np.float32)
            )

    # ----------- Public accessors ----------------------------------------
    @property
    def entities(self) -> List[Dict[str, Any]]:
        # Defensive copy to discourage callers from passing the full list to
        # an LLM; the agents use the bounded retrieval API instead.
        return [dict(self._by_id[eid]) for eid in self._entity_ids]

    @property
    def triples(self) -> List[Dict[str, Any]]:
        return [dict(self._triple_by_id[tid]) for tid in self._triple_ids]

    def lookup_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        if not entity_id:
            return None
        if entity_id in self._by_id:
            return dict(self._by_id[entity_id])
        # Fallback: try to interpret an upstream "E:type:name" id by matching
        # canonical_name slug.
        for eid, entity in self._by_id.items():
            if eid == entity_id:
                return dict(entity)
        return None

    def lookup_entity_by_name(self, name: str, limit: int = 5) -> List[Dict[str, Any]]:
        key = normalize_text(name)
        if not key:
            return []
        eids = self._by_name.get(key, [])
        return [dict(self._by_id[eid]) for eid in eids[:limit]]

    def lookup_triples_by_entity(self, entity_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        key = normalize_text(entity_name)
        if not key:
            return []
        seen: List[str] = []
        for tid in self._triples_by_head.get(key, []):
            if tid not in seen:
                seen.append(tid)
        for tid in self._triples_by_tail.get(key, []):
            if tid not in seen:
                seen.append(tid)
        return [dict(self._triple_by_id[tid]) for tid in seen[:limit]]

    # ----------- Top-k retrieval -----------------------------------------
    def _topk(self, vectors: np.ndarray, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if vectors is None or vectors.size == 0 or query.size == 0 or k <= 0:
            return []
        if query.ndim == 1:
            query = query[None, :]
        c = vectors.astype(np.float32, copy=False)
        q = query.astype(np.float32, copy=False)
        c_norm = np.linalg.norm(c, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        c = c / np.maximum(c_norm, 1e-12)
        q = q / np.maximum(q_norm, 1e-12)
        sims = (c @ q.T).squeeze(-1)
        kk = min(k, sims.shape[0])
        idx = np.argpartition(-sims, kk - 1)[:kk]
        ordered = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in ordered]

    def entity_top_k(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        if self._entity_vectors is None:
            self.build_vectors()
        if self._entity_vectors is None:
            return []
        hits = self._topk(self._entity_vectors, query_vector, k)
        return [(dict(self._by_id[self._entity_ids[i]]), score) for i, score in hits]

    def triple_top_k(self, query_vector: np.ndarray, k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        if self._triple_vectors is None:
            self.build_vectors()
        if self._triple_vectors is None:
            return []
        hits = self._topk(self._triple_vectors, query_vector, k)
        return [(dict(self._triple_by_id[self._triple_ids[i]]), score) for i, score in hits]

    # ----------- Helpers --------------------------------------------------
    @staticmethod
    def source_article_overlap(article_ids: Iterable[str], record: Dict[str, Any]) -> float:
        target = {a for a in (record.get("source_articles") or []) if a}
        own = {a for a in article_ids if a}
        if not target or not own:
            return 0.0
        return len(target & own) / float(len(target | own))

    def relation_distribution(self, top_n: int = 20) -> Dict[str, int]:
        dist = self._statistics.get("relation_distribution")
        if isinstance(dist, dict) and dist:
            ordered = sorted(dist.items(), key=lambda kv: -int(kv[1]))[:top_n]
            return {str(k): int(v) for k, v in ordered}
        counter: Counter = Counter()
        for triple in self._raw_triples:
            counter[str(triple.get("relation", ""))] += 1
        return dict(counter.most_common(top_n))

    def entity_type_distribution(self, top_n: int = 20) -> Dict[str, int]:
        dist = self._statistics.get("entity_type_distribution")
        if isinstance(dist, dict) and dist:
            ordered = sorted(dist.items(), key=lambda kv: -int(kv[1]))[:top_n]
            return {str(k): int(v) for k, v in ordered}
        counter: Counter = Counter()
        for entity in self._raw_entities:
            counter[str(entity.get("entity_type", ""))] += 1
        return dict(counter.most_common(top_n))

    def pipeline_version(self) -> str:
        return str(self._metadata.get("pipeline_version", ""))

    def metadata_summary(self) -> Dict[str, Any]:
        return {
            "total_source_articles": self._metadata.get("total_source_articles"),
            "domain": self._metadata.get("domain"),
            "pipeline_version": self.pipeline_version(),
            "entity_count": len(self._entity_ids),
            "triple_count": len(self._triple_ids),
        }
