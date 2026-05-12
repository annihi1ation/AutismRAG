"""HyperKGWriterIndexerAgent — aggregates evidence hyperedges into canonical
hyperedges, writes JSONL outputs, and saves vector indexes.

Does NOT call the Online LLM.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from workflows.HyperKGConstruction.embedding_service import (
    EmbeddingService,
    entity_embedding_text,
    summary_embedding_text,
)


def canonical_hyperedge_embedding_text(canonical) -> str:
    """Local override — the upstream helper assumes a ``qualifier_summary``
    attribute that HyperKGBuilder's ``CanonicalHyperedge`` does not carry."""
    return "\n".join([
        f"Canonical claim: {canonical.canonical_claim}",
        f"Type: {canonical.claim_type}",
        f"Core entities: {', '.join(canonical.core_entity_ids)}",
        f"Scope: {compact_json(canonical.scope_summary)}",
        f"Conflicts: {compact_json(canonical.conflict_status_distribution)}",
        f"Roles: {compact_json(canonical.entity_role_distribution)}",
    ])
from workflows.HyperKGConstruction.utils import compact_json, slugify, to_plain
from workflows.HyperKGConstruction.vector_index import VectorIndexAdapter

from ..config import WriterConfig
from ..schemas import (
    CandidatePack,
    CanonicalHyperedge,
    EvidenceHyperedge,
    HyperKGRunReport,
    SummarySnippet,
    TripleProjection,
)
from ..unified_kg_index import UnifiedKGIndex, triple_embedding_text

logger = logging.getLogger(__name__)


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_plain(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_plain(payload), f, ensure_ascii=False, sort_keys=True, indent=2)


@dataclass
class HyperKGWriterIndexerAgent:
    embedding_service: Optional[EmbeddingService]
    writer: WriterConfig

    def write(
        self,
        output_dir: str,
        packs: List[CandidatePack],
        evidence_hyperedges: List[EvidenceHyperedge],
        unified_kg_index: UnifiedKGIndex,
        summaries: Dict[str, Dict[str, Any]],
        run_report: HyperKGRunReport,
        llm_usage: List[Dict[str, Any]],
        review_items: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        # 1. Cluster into canonical hyperedges.
        canonical_hyperedges = self._build_canonical_hyperedges(evidence_hyperedges)

        # 2. Pairwise relations between canonical hyperedges.
        canonical_hyperedges = self._infer_relations(canonical_hyperedges)

        # 3. Triple projections (one per primary triple in each evidence edge).
        triple_projections = self._build_triple_projections(evidence_hyperedges)

        # 4. Vector indexes.
        index_dir = os.path.join(output_dir, "vector_indexes")
        self._build_and_save_indexes(
            index_dir, packs, evidence_hyperedges, canonical_hyperedges, summaries, unified_kg_index
        )

        # 5. JSONL outputs.
        paths = {
            "candidate_packs": os.path.join(output_dir, "candidate_packs.jsonl"),
            "evidence_hyperedges": os.path.join(output_dir, "evidence_hyperedges.jsonl"),
            "canonical_hyperedges": os.path.join(output_dir, "canonical_hyperedges.jsonl"),
            "triple_projections": os.path.join(output_dir, "triple_projections.jsonl"),
            "incidence_edges": os.path.join(output_dir, "incidence_edges.jsonl"),
            "summary_links": os.path.join(output_dir, "summary_links.jsonl"),
            "hyperkg_run_report": os.path.join(output_dir, "hyperkg_run_report.json"),
            "llm_usage_report": os.path.join(output_dir, "llm_usage_report.json"),
            "review_queue": os.path.join(output_dir, "review_queue.json"),
        }

        _write_jsonl(paths["candidate_packs"], (p.to_dict() for p in packs))
        _write_jsonl(paths["evidence_hyperedges"], (e.to_dict() for e in evidence_hyperedges))
        _write_jsonl(paths["canonical_hyperedges"], (c.to_dict() for c in canonical_hyperedges))
        _write_jsonl(paths["triple_projections"], (t.to_dict() for t in triple_projections))

        incidence = self._build_incidence_edges(evidence_hyperedges, canonical_hyperedges, triple_projections)
        _write_jsonl(paths["incidence_edges"], incidence)

        summary_links = self._build_summary_links(evidence_hyperedges, summaries)
        _write_jsonl(paths["summary_links"], summary_links)

        # 6. Reports.
        run_report.output_paths = dict(paths)
        run_report.output_paths["vector_indexes_manifest"] = os.path.join(index_dir, "manifest.json")
        run_report.phase_counts.setdefault("write", {})
        run_report.phase_counts["write"].update({
            "evidence_hyperedges": len(evidence_hyperedges),
            "canonical_hyperedges": len(canonical_hyperedges),
            "triple_projections": len(triple_projections),
            "incidence_edges": len(incidence),
            "summary_links": len(summary_links),
        })

        _write_json(paths["hyperkg_run_report"], run_report.to_dict())
        _write_json(paths["llm_usage_report"], {
            "model": run_report.fingerprint.get("model_name", ""),
            "calls": llm_usage,
            "totals": _llm_usage_totals(llm_usage),
        })
        _write_json(paths["review_queue"], {"items": review_items})

        return run_report.output_paths

    # ------------------------------------------------------------------
    def _build_canonical_hyperedges(
        self,
        edges: List[EvidenceHyperedge],
    ) -> List[CanonicalHyperedge]:
        if not edges:
            return []

        # Embed evidence-hyperedge texts for similarity-based clustering.
        if self.embedding_service is not None and self.embedding_service.enabled:
            texts = [evidence_hyperedge_embedding_text_or_text(e) for e in edges]
            vectors = self.embedding_service.encode_texts(texts)
        else:
            vectors = np.zeros((len(edges), 1), dtype=np.float32)

        thresh = self.writer.canonical_similarity_threshold
        jaccard = self.writer.canonical_jaccard_threshold
        clusters: List[List[int]] = []
        cluster_centers: List[np.ndarray] = []
        cluster_types: List[str] = []
        cluster_entity_sets: List[set] = []

        for idx, edge in enumerate(edges):
            vec = vectors[idx]
            entity_ids = {str(e.get("entity_id")) for e in edge.entities if e.get("entity_id")}
            assigned = False
            for ci in range(len(clusters)):
                if cluster_types[ci] != edge.claim_type:
                    continue
                if vectors.shape[1] > 1:
                    a = cluster_centers[ci]
                    b = vec
                    a_n = a / max(float(np.linalg.norm(a)), 1e-12)
                    b_n = b / max(float(np.linalg.norm(b)), 1e-12)
                    sim = float(a_n @ b_n)
                else:
                    sim = 1.0 if cluster_types[ci] == edge.claim_type else 0.0
                if sim < thresh:
                    continue
                set_other = cluster_entity_sets[ci]
                if set_other or entity_ids:
                    inter = len(set_other & entity_ids)
                    union = max(1, len(set_other | entity_ids))
                    if inter / union < jaccard:
                        continue
                clusters[ci].append(idx)
                # incremental mean
                n = len(clusters[ci])
                cluster_centers[ci] = cluster_centers[ci] * ((n - 1) / n) + vec / n
                cluster_entity_sets[ci] |= entity_ids
                assigned = True
                break
            if not assigned:
                clusters.append([idx])
                cluster_centers.append(vec.copy())
                cluster_types.append(edge.claim_type)
                cluster_entity_sets.append(entity_ids)

        canonical: List[CanonicalHyperedge] = []
        for ci, member_indices in enumerate(clusters):
            members = [edges[i] for i in member_indices]
            canon = self._aggregate_canonical(members)
            canonical.append(canon)
        return canonical

    def _aggregate_canonical(self, members: List[EvidenceHyperedge]) -> CanonicalHyperedge:
        canon_id = f"CH:{uuid.uuid4().hex[:12]}"
        # Pick the longest claim text as canonical claim by default.
        canonical_claim = max((m.claim_text for m in members), key=len) if members else ""
        claim_type = members[0].claim_type if members else ""

        source_articles: List[str] = []
        seen_articles: set = set()
        confidences: List[float] = []
        relevances: List[float] = []
        clarities: List[float] = []
        conflict_counter: Counter = Counter()
        role_counter: Counter = Counter()
        entity_id_counter: Counter = Counter()
        scope_keys: Dict[str, Counter] = {}
        evidence_count = 0

        for m in members:
            evidence_count += 1
            for aid in (m.source.get("article_ids") or []):
                if aid not in seen_articles:
                    seen_articles.add(aid)
                    source_articles.append(aid)
            confidences.append(float(m.confidence or 0.0))
            for tr in (m.primary_triples or []) + (m.supporting_triples or []):
                if isinstance(tr, dict):
                    if tr.get("relevance") is not None:
                        relevances.append(float(tr.get("relevance") or 0.0))
                    if tr.get("clarity") is not None:
                        clarities.append(float(tr.get("clarity") or 0.0))
                    cs = str(tr.get("conflict_status") or "")
                    if cs:
                        conflict_counter[cs] += 1
            for entity in m.entities:
                role_counter[str(entity.get("role") or "participant")] += 1
                eid = str(entity.get("entity_id") or "")
                if eid:
                    entity_id_counter[eid] += 1
            for k, v in (m.scope or {}).items():
                if v is None:
                    continue
                scope_keys.setdefault(str(k), Counter())[str(v)] += 1

        scope_summary = {
            k: counter.most_common(1)[0][0] for k, counter in scope_keys.items() if counter
        }

        # Core entities = ids that appear in >= half of members (inclusive).
        threshold = max(1, len(members) // 2)
        core_entity_ids = [eid for eid, c in entity_id_counter.items() if c >= threshold]
        if not core_entity_ids and entity_id_counter:
            core_entity_ids = [eid for eid, _ in entity_id_counter.most_common(2)]

        return CanonicalHyperedge(
            canonical_hyperedge_id=canon_id,
            canonical_claim=canonical_claim,
            claim_type=claim_type,
            member_evidence_hyperedges=[m.evidence_hyperedge_id for m in members],
            support_count=len(members),
            evidence_count=evidence_count,
            source_articles=source_articles,
            mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
            mean_relevance=float(np.mean(relevances)) if relevances else 0.0,
            mean_clarity=float(np.mean(clarities)) if clarities else 0.0,
            conflict_status_distribution=dict(conflict_counter),
            entity_role_distribution=dict(role_counter),
            core_entity_ids=core_entity_ids,
            scope_summary=scope_summary,
        )

    def _infer_relations(self, canonical: List[CanonicalHyperedge]) -> List[CanonicalHyperedge]:
        if len(canonical) < 2 or self.embedding_service is None or not self.embedding_service.enabled:
            return canonical
        texts = [canonical_hyperedge_embedding_text(c) for c in canonical]
        vectors = self.embedding_service.encode_texts(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normed = vectors / np.maximum(norms, 1e-12)
        sims = normed @ normed.T

        thresh = self.writer.relation_supports_threshold
        for i, c_i in enumerate(canonical):
            entities_i = set(c_i.core_entity_ids)
            for j in range(i + 1, len(canonical)):
                c_j = canonical[j]
                if c_i.claim_type != c_j.claim_type:
                    continue
                entities_j = set(c_j.core_entity_ids)
                if not entities_i & entities_j:
                    continue
                sim = float(sims[i, j])
                # Decide kind via scope agreement (we don't track polarity at
                # canonical level explicitly here, so we use scope diffs +
                # similarity as proxies).
                scope_diff = any(
                    c_i.scope_summary.get(k) and c_j.scope_summary.get(k)
                    and c_i.scope_summary[k] != c_j.scope_summary[k]
                    for k in set(c_i.scope_summary) & set(c_j.scope_summary)
                )
                if scope_diff:
                    c_i.relations.append({"target": c_j.canonical_hyperedge_id, "type": "DIFFERENT_SCOPE_FROM", "score": sim})
                    c_j.relations.append({"target": c_i.canonical_hyperedge_id, "type": "DIFFERENT_SCOPE_FROM", "score": sim})
                elif sim >= thresh:
                    c_i.relations.append({"target": c_j.canonical_hyperedge_id, "type": "SUPPORTS", "score": sim})
                    c_j.relations.append({"target": c_i.canonical_hyperedge_id, "type": "SUPPORTS", "score": sim})
        return canonical

    def _build_triple_projections(self, edges: List[EvidenceHyperedge]) -> List[TripleProjection]:
        out: List[TripleProjection] = []
        for edge in edges:
            for tr in edge.primary_triples or []:
                if not isinstance(tr, dict):
                    continue
                head_id = _entity_id_from_edge(edge, tr.get("head"))
                tail_id = _entity_id_from_edge(edge, tr.get("tail"))
                out.append(TripleProjection(
                    head_entity_id=head_id,
                    relation=str(tr.get("relation") or ""),
                    tail_entity_id=tail_id,
                    support=str(edge.claim_text)[:300],
                    confidence=float(tr.get("confidence") or edge.confidence or 0.0),
                    evidence_hyperedge_id=edge.evidence_hyperedge_id,
                ))
        return out

    def _build_incidence_edges(
        self,
        edges: List[EvidenceHyperedge],
        canonical: List[CanonicalHyperedge],
        triple_projections: List[TripleProjection],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for edge in edges:
            for entity in edge.entities:
                eid = entity.get("entity_id")
                if not eid:
                    continue
                rows.append({
                    "type": "PARTICIPATES_IN",
                    "from": eid,
                    "to": edge.evidence_hyperedge_id,
                    "role": entity.get("role"),
                })
            for sid in edge.source.get("summary_ids") or []:
                rows.append({
                    "type": "SUPPORTED_BY",
                    "from": edge.evidence_hyperedge_id,
                    "to": sid,
                })
            for aid in edge.source.get("article_ids") or []:
                rows.append({
                    "type": "FROM_ARTICLE",
                    "from": edge.source.get("summary_ids") or "",
                    "to": aid,
                })
        for proj in triple_projections:
            rows.append({
                "type": "PROJECTS_TO",
                "from": proj.evidence_hyperedge_id,
                "to": f"TP:{slugify(proj.head_entity_id)}:{slugify(proj.relation)}:{slugify(proj.tail_entity_id)}",
            })
        for canon in canonical:
            for ehid in canon.member_evidence_hyperedges:
                rows.append({
                    "type": "MEMBER_OF",
                    "from": ehid,
                    "to": canon.canonical_hyperedge_id,
                })
            for rel in canon.relations:
                rows.append({
                    "type": rel.get("type") or "SUPPORTS",
                    "from": canon.canonical_hyperedge_id,
                    "to": rel.get("target"),
                    "score": rel.get("score"),
                })
        return rows

    def _build_summary_links(
        self,
        edges: List[EvidenceHyperedge],
        summaries: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for edge in edges:
            for sid in edge.source.get("summary_ids") or []:
                body = summaries.get(sid) or {}
                rows.append({
                    "evidence_hyperedge_id": edge.evidence_hyperedge_id,
                    "summary_id": sid,
                    "article_id": body.get("article_id", ""),
                    "segment_id": body.get("segment_id", ""),
                })
        return rows

    # ------------------------------------------------------------------
    def _build_and_save_indexes(
        self,
        index_dir: str,
        packs: List[CandidatePack],
        edges: List[EvidenceHyperedge],
        canonical: List[CanonicalHyperedge],
        summaries: Dict[str, Dict[str, Any]],
        unified_kg_index: UnifiedKGIndex,
    ) -> None:
        if self.embedding_service is None or not self.embedding_service.enabled:
            os.makedirs(index_dir, exist_ok=True)
            with open(os.path.join(index_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"namespaces": {}, "note": "embedding service disabled"}, f, indent=2)
            return

        adapter = VectorIndexAdapter(embedding_service=self.embedding_service)
        # claim_vector_index — represented by candidate_pack texts for simplicity
        if packs:
            ids = [p.pack_id for p in packs]
            texts = [p.representative_claim_text for p in packs]
            metadata = [{"claim_type": p.claim_type, "auto_decision": p.auto_decision} for p in packs]
            vectors = self.embedding_service.encode_texts(texts)
            adapter.add("claim_vector_index", ids, texts, vectors, metadata)
            # update vector_id back-refs on edges that originated from this pack
        if edges:
            ids = [e.evidence_hyperedge_id for e in edges]
            texts = [evidence_hyperedge_embedding_text_or_text(e) for e in edges]
            metadata = [{"claim_type": e.claim_type, "method": e.method} for e in edges]
            vectors = self.embedding_service.encode_texts(texts)
            adapter.add("evidence_hyperedge_vector_index", ids, texts, vectors, metadata)
            for edge, text in zip(edges, texts):
                edge.embedding_text = text
                edge.vector_id = edge.evidence_hyperedge_id
        if canonical:
            ids = [c.canonical_hyperedge_id for c in canonical]
            texts = [canonical_hyperedge_embedding_text(c) for c in canonical]
            metadata = [{"claim_type": c.claim_type, "support_count": c.support_count} for c in canonical]
            vectors = self.embedding_service.encode_texts(texts)
            adapter.add("canonical_hyperedge_vector_index", ids, texts, vectors, metadata)
            for canon, text in zip(canonical, texts):
                canon.embedding_text = text
                canon.vector_id = canon.canonical_hyperedge_id
        # Canonical entity index (subset participating in any edge).
        participating_ids: List[str] = []
        seen: set = set()
        for edge in edges:
            for entity in edge.entities:
                eid = str(entity.get("entity_id") or "")
                if not eid or eid in seen:
                    continue
                seen.add(eid)
                participating_ids.append(eid)
        if participating_ids:
            ids = []
            texts = []
            metadata = []
            for eid in participating_ids:
                entity = unified_kg_index.lookup_entity_by_id(eid) or {"canonical_name": eid, "entity_type": ""}
                ids.append(eid)
                texts.append(entity_embedding_text(entity))
                metadata.append({
                    "entity_type": entity.get("entity_type"),
                    "normalized_id": entity.get("normalized_id"),
                })
            vectors = self.embedding_service.encode_texts(texts)
            adapter.add("canonical_entity_vector_index", ids, texts, vectors, metadata)
        if summaries:
            items = [v for v in summaries.values() if v.get("summary_text")]
            if items:
                ids = [str(v["summary_id"]) for v in items]
                texts = [_summary_text_for_index(v) for v in items]
                metadata = [
                    {
                        "article_id": v.get("article_id", ""),
                        "segment_id": v.get("segment_id", ""),
                    }
                    for v in items
                ]
                vectors = self.embedding_service.encode_texts(texts)
                adapter.add("summary_vector_index", ids, texts, vectors, metadata)
        # Triple projection index (deduplicated by head|relation|tail per edge).
        projection_ids: List[str] = []
        projection_texts: List[str] = []
        projection_meta: List[Dict[str, Any]] = []
        seen_proj: set = set()
        for edge in edges:
            for tr in (edge.primary_triples or []) + (edge.supporting_triples or []):
                if not isinstance(tr, dict):
                    continue
                key = f"{tr.get('head','')}|{tr.get('relation','')}|{tr.get('tail','')}"
                if not key or key in seen_proj:
                    continue
                seen_proj.add(key)
                projection_ids.append(f"TP:{slugify(key)}")
                projection_texts.append(triple_embedding_text(tr))
                projection_meta.append({
                    "relation": tr.get("relation"),
                    "evidence_hyperedge_id": edge.evidence_hyperedge_id,
                })
        if projection_ids:
            vectors = self.embedding_service.encode_texts(projection_texts)
            adapter.add("triple_vector_index", projection_ids, projection_texts, vectors, projection_meta)

        adapter.save(index_dir)


def _summary_text_for_index(record: Dict[str, Any]) -> str:
    fake = type("S", (), {})()
    fake.summary_text = record.get("summary_text", "")
    fake.metadata = {
        "article_id": record.get("article_id", ""),
        "segment_id": record.get("segment_id", ""),
        "section": record.get("section", ""),
    }
    return summary_embedding_text(fake)


def evidence_hyperedge_embedding_text_or_text(edge: EvidenceHyperedge) -> str:
    """Adapter so callers don't depend on HyperedgeEntity dataclass shape."""
    entities_str = "; ".join(
        f"{e.get('mention') or e.get('entity_id') or ''} ({e.get('role') or 'participant'},"
        f" {e.get('entity_type') or 'unknown'})"
        for e in edge.entities
    )
    return "\n".join([
        f"Claim: {edge.claim_text}",
        f"Type: {edge.claim_type}",
        f"Entities: {entities_str}",
        f"Qualifiers: {compact_json(edge.qualifiers)}",
        f"Polarity: {edge.polarity}",
    ])


def _entity_id_from_edge(edge: EvidenceHyperedge, mention: Any) -> str:
    target = str(mention or "").strip()
    if not target:
        return ""
    for entity in edge.entities:
        name = str(entity.get("mention") or entity.get("canonical_name") or "")
        if name and name.lower() == target.lower():
            return str(entity.get("entity_id") or "")
    return f"E:{slugify(target)}"


def _llm_usage_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_calls = sum(1 for r in rows if r.get("attempt_count", 0) > 0)
    total_prompt = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    total_completion = sum(int(r.get("completion_tokens") or 0) for r in rows)
    parse_dist: Counter = Counter()
    for r in rows:
        parse_dist[str(r.get("parse_status") or "unknown")] += 1
    return {
        "total_calls": total_calls,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "parse_status_distribution": dict(parse_dist),
    }
