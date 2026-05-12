"""Module 3: DenseRetriever — raw cosine top-k, no reranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from workflows.HyperKGConstruction.vector_index import VectorIndexAdapter

from .loaders import HyperRAGContext


def _search_namespace(
    adapter: VectorIndexAdapter,
    namespace: str,
    query_text: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not query_text or top_k <= 0:
        return []
    return adapter.search(namespace=namespace, query_text=query_text, top_k=top_k)


def _annotate(hits: List[Dict[str, Any]], source_query: str) -> List[Dict[str, Any]]:
    out = []
    for rank, hit in enumerate(hits, start=1):
        item = dict(hit)
        item["source_queries"] = [source_query]
        item["rank"] = rank
        out.append(item)
    return out


def _merge_dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in items:
        item_id = item.get("id")
        if not item_id:
            continue
        existing = by_id.get(item_id)
        if existing is None:
            by_id[item_id] = dict(item)
            continue
        if item.get("score", 0.0) > existing.get("score", 0.0):
            existing["score"] = item["score"]
            existing["rank"] = item.get("rank", existing.get("rank"))
        merged_sources = list(dict.fromkeys(existing.get("source_queries", []) + item.get("source_queries", [])))
        existing["source_queries"] = merged_sources
    return sorted(by_id.values(), key=lambda r: r.get("score", 0.0), reverse=True)


class DenseRetriever:
    def __init__(self, ctx: HyperRAGContext):
        self.ctx = ctx
        self.cfg = ctx.config

    def run(self, retrieval_plan: Dict[str, Any]) -> Dict[str, Any]:
        queries: Dict[str, Optional[str]] = retrieval_plan.get("retrieval_queries") or {}
        seed_entities = retrieval_plan.get("seed_entities") or []

        all_entities: List[Dict[str, Any]] = []
        all_hyperedges: List[Dict[str, Any]] = []
        all_chunks: List[Dict[str, Any]] = []

        per_query_counts: Dict[str, Dict[str, int]] = {}

        for query_name, query_text in queries.items():
            if not query_text:
                continue
            entity_hits = _annotate(
                _search_namespace(
                    self.ctx.entity_index,
                    self.cfg.entity_namespace,
                    query_text,
                    self.cfg.top_k_entities,
                ),
                query_name,
            )
            hyperedge_hits = _annotate(
                _search_namespace(
                    self.ctx.hyperedge_index,
                    self.cfg.hyperedge_namespace,
                    query_text,
                    self.cfg.top_k_hyperedges,
                ),
                query_name,
            )
            chunk_hits = _annotate(
                _search_namespace(
                    self.ctx.chunk_index,
                    self.cfg.chunk_namespace,
                    query_text,
                    self.cfg.top_k_chunks,
                ),
                query_name,
            )
            all_entities.extend(entity_hits)
            all_hyperedges.extend(hyperedge_hits)
            all_chunks.extend(chunk_hits)
            per_query_counts[query_name] = {
                "entities": len(entity_hits),
                "hyperedges": len(hyperedge_hits),
                "chunks": len(chunk_hits),
            }

        for seed in seed_entities:
            mention = (seed.get("mention") or "").strip()
            if not mention:
                continue
            seed_hits = _annotate(
                _search_namespace(
                    self.ctx.entity_index,
                    self.cfg.entity_namespace,
                    mention,
                    self.cfg.top_k_seed_entity,
                ),
                f"seed::{mention}",
            )
            all_entities.extend(seed_hits)

        return {
            "entities": _merge_dedupe(all_entities),
            "hyperedges": _merge_dedupe(all_hyperedges),
            "chunks": _merge_dedupe(all_chunks),
            "per_query_counts": per_query_counts,
        }
