"""CandidatePackRouterAgent — clusters claims, retrieves bounded candidates,
and decides which packs need the Online LLM.

Does NOT call the Online LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from workflows.HyperKGConstruction.embedding_service import EmbeddingService
from workflows.HyperKGConstruction.utils import normalize_text, slugify
from workflows.HyperKGConstruction.vector_index import VectorIndexAdapter

from ..config import RoutingConfig
from ..pack_hash import compute_pack_hash
from ..schemas import (
    CandidatePack,
    ClaimRecord,
    EntityCandidate,
    SummarySnippet,
    TripleCandidate,
)
from ..unified_kg_index import UnifiedKGIndex
from .embedding_index_agent import (
    CLAIMS_NS,
    ENTITIES_NS,
    SUMMARIES_NS,
    TRIPLES_NS,
    claim_embedding_text,
)

logger = logging.getLogger(__name__)

_HEDGING_RE = re.compile(
    r"\b(?:not|no|without|denied|unable|fail|fails|failed|doubt|uncertain|"
    r"may|might|possibly|likely|unclear|inconclusive|appear|seem|seems)\b",
    re.IGNORECASE,
)

_ACCEPTABLE_CONFLICTS = {"consistent", "single_source", "", None}


@dataclass
class RouterReport:
    total_claims: int = 0
    total_packs: int = 0
    auto_packs: int = 0
    online_packs: int = 0
    routing_reason_counts: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.routing_reason_counts is None:
            self.routing_reason_counts = {}


@dataclass
class CandidatePackRouterAgent:
    """Build CandidatePack records and decide auto vs online_llm."""

    routing: RoutingConfig
    embedding_service: Optional[EmbeddingService]

    # ---------- public entry ------------------------------------------------
    def build_packs(
        self,
        claims: List[ClaimRecord],
        index: VectorIndexAdapter,
        unified_kg_index: UnifiedKGIndex,
        summaries: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[CandidatePack], RouterReport]:
        report = RouterReport(total_claims=len(claims))
        if not claims:
            return [], report

        clusters = self._cluster_claims(claims, index)
        packs: List[CandidatePack] = []

        for cluster in clusters:
            pack = self._build_pack(cluster, claims, index, unified_kg_index, summaries)
            if pack is None:
                continue
            packs.append(pack)
            report.total_packs += 1
            if pack.auto_decision == "online_llm":
                report.online_packs += 1
            else:
                report.auto_packs += 1
            for reason in pack.routing_reasons:
                report.routing_reason_counts[reason] = (
                    report.routing_reason_counts.get(reason, 0) + 1
                )

        return packs, report

    # ---------- clustering --------------------------------------------------
    def _cluster_claims(
        self,
        claims: List[ClaimRecord],
        index: VectorIndexAdapter,
    ) -> List[List[int]]:
        threshold = self.routing.cluster_similarity_threshold
        cap = max(1, self.routing.max_claims_per_work_unit)

        vectors = index.vectors.get(CLAIMS_NS) if index else None
        if vectors is None or vectors.size == 0:
            return [[i] for i in range(len(claims))]

        records = index.records.get(CLAIMS_NS) or []
        id_to_pos = {r["id"]: i for i, r in enumerate(records)}

        clusters: List[List[int]] = []
        cluster_centers: List[np.ndarray] = []
        for i, claim in enumerate(claims):
            pos = id_to_pos.get(claim.claim_id)
            if pos is None:
                clusters.append([i])
                cluster_centers.append(np.zeros((vectors.shape[1],), dtype=np.float32))
                continue
            v = vectors[pos]
            v_norm = v / max(float(np.linalg.norm(v)), 1e-12)
            assigned = False
            for ci, center in enumerate(cluster_centers):
                c_norm = center / max(float(np.linalg.norm(center)), 1e-12)
                sim = float(c_norm @ v_norm)
                if sim >= threshold and len(clusters[ci]) < cap and (
                    claims[clusters[ci][0]].claim_type == claim.claim_type
                ):
                    clusters[ci].append(i)
                    # incremental running mean
                    n = len(clusters[ci])
                    cluster_centers[ci] = center * ((n - 1) / n) + v / n
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
                cluster_centers.append(v.copy())
        return clusters

    # ---------- pack assembly ----------------------------------------------
    def _build_pack(
        self,
        cluster: List[int],
        claims: List[ClaimRecord],
        index: VectorIndexAdapter,
        unified_kg_index: UnifiedKGIndex,
        summaries: Dict[str, Dict[str, Any]],
    ) -> Optional[CandidatePack]:
        if not cluster:
            return None
        rep_idx = cluster[0]
        rep_claim = claims[rep_idx]
        members = [claims[i] for i in cluster]

        article_ids = {c.article_id for c in members if c.article_id}

        entity_candidates = self._build_entity_candidates(members, index, unified_kg_index, article_ids)
        triple_candidates = self._build_triple_candidates(
            members, index, unified_kg_index, article_ids, entity_candidates
        )
        summary_snippets = self._build_summary_snippets(
            members, index, summaries, article_ids
        )

        local_kg_snippet = self._build_local_kg_snippet(members)
        intra_sim = self._intra_cluster_similarity(cluster, claims, index)
        routing_reasons = self._decide_routing(
            rep_claim, members, entity_candidates, triple_candidates, intra_sim
        )
        decision = "online_llm" if routing_reasons else "auto"
        if not routing_reasons:
            routing_reasons = ["auto_high_confidence_anchor"]

        pack_id = f"PACK:{slugify(rep_claim.claim_id)}"
        pack = CandidatePack(
            pack_id=pack_id,
            representative_claim_id=rep_claim.claim_id,
            representative_claim_text=rep_claim.claim_text,
            claim_type=rep_claim.claim_type,
            member_claim_ids=[c.claim_id for c in members],
            entity_candidates=entity_candidates,
            triple_candidates=triple_candidates,
            summary_snippets=summary_snippets,
            local_kg_snippet=local_kg_snippet,
            routing_reasons=routing_reasons,
            auto_decision=decision,
        )
        pack.pack_hash = compute_pack_hash(pack)
        return pack

    # ---------- candidate retrieval ----------------------------------------
    def _query_vector(self, claim: ClaimRecord, index: VectorIndexAdapter) -> Optional[np.ndarray]:
        records = index.records.get(CLAIMS_NS) or []
        for i, record in enumerate(records):
            if record["id"] == claim.claim_id:
                return index.vectors.get(CLAIMS_NS)[i]
        if self.embedding_service is None or not self.embedding_service.enabled:
            return None
        return self.embedding_service.encode_one(claim_embedding_text(claim))

    def _build_entity_candidates(
        self,
        members: List[ClaimRecord],
        index: VectorIndexAdapter,
        unified_kg_index: UnifiedKGIndex,
        article_ids: set,
    ) -> List[EntityCandidate]:
        seen: Dict[str, EntityCandidate] = {}
        per_mention_count: Dict[str, int] = {}
        max_per_mention = self.routing.max_candidate_entities_per_mention

        # 1. Resolve claim-side mentions against the unified KG.
        for claim in members:
            for raw in claim.candidate_entities:
                mention = str(raw.get("mention") or raw.get("canonical_name") or "").strip()
                key = normalize_text(mention) or normalize_text(raw.get("entity_id", ""))
                resolved = unified_kg_index.lookup_entity_by_name(mention, limit=2)
                if not resolved:
                    # Fall back to a synthetic candidate from the upstream id.
                    eid = str(raw.get("entity_id") or "")
                    if not eid:
                        continue
                    if eid in seen:
                        continue
                    seen[eid] = EntityCandidate(
                        entity_id=eid,
                        canonical_name=str(raw.get("canonical_name") or mention),
                        entity_type=str(raw.get("entity_type") or ""),
                        aliases=[mention] if mention else [],
                        normalized_id="",
                        mention_count=1,
                        source_articles=[claim.article_id] if claim.article_id else [],
                        score=0.55,
                        source="claim",
                    )
                    per_mention_count[key] = per_mention_count.get(key, 0) + 1
                    continue
                for entity in resolved:
                    if per_mention_count.get(key, 0) >= max_per_mention:
                        break
                    eid = entity["entity_id"]
                    if eid in seen:
                        continue
                    overlap = unified_kg_index.source_article_overlap(article_ids, entity)
                    mc_norm = min(1.0, float(entity.get("mention_count", 0) or 0) / 50.0)
                    score = 0.6 * 0.95 + 0.2 * overlap + 0.2 * mc_norm
                    seen[eid] = EntityCandidate(
                        entity_id=eid,
                        canonical_name=str(entity.get("canonical_name") or ""),
                        entity_type=str(entity.get("entity_type") or ""),
                        aliases=list(entity.get("aliases") or []),
                        normalized_id=str(entity.get("normalized_id") or ""),
                        mention_count=int(entity.get("mention_count") or 0),
                        source_articles=list(entity.get("source_articles") or []),
                        score=float(score),
                        source="unified_kg",
                    )
                    per_mention_count[key] = per_mention_count.get(key, 0) + 1

        # 2. Embedding-based top-k expansion (only if budget allows).
        rep = members[0]
        qv = self._query_vector(rep, index)
        if qv is not None:
            results = unified_kg_index.entity_top_k(qv, k=self.routing.entity_top_k)
            for entity, sim in results:
                eid = entity["entity_id"]
                if eid in seen:
                    continue
                key = normalize_text(entity.get("canonical_name") or "")
                if per_mention_count.get(key, 0) >= max_per_mention:
                    continue
                overlap = unified_kg_index.source_article_overlap(article_ids, entity)
                mc_norm = min(1.0, float(entity.get("mention_count", 0) or 0) / 50.0)
                score = 0.6 * float(sim) + 0.2 * overlap + 0.2 * mc_norm
                seen[eid] = EntityCandidate(
                    entity_id=eid,
                    canonical_name=str(entity.get("canonical_name") or ""),
                    entity_type=str(entity.get("entity_type") or ""),
                    aliases=list(entity.get("aliases") or []),
                    normalized_id=str(entity.get("normalized_id") or ""),
                    mention_count=int(entity.get("mention_count") or 0),
                    source_articles=list(entity.get("source_articles") or []),
                    score=float(score),
                    source="unified_kg",
                )
                per_mention_count[key] = per_mention_count.get(key, 0) + 1

        return sorted(seen.values(), key=lambda c: -c.score)

    def _build_triple_candidates(
        self,
        members: List[ClaimRecord],
        index: VectorIndexAdapter,
        unified_kg_index: UnifiedKGIndex,
        article_ids: set,
        entity_candidates: List[EntityCandidate],
    ) -> List[TripleCandidate]:
        cap = self.routing.max_candidate_triples_per_work_unit
        entity_names = {normalize_text(e.canonical_name) for e in entity_candidates}
        entity_names |= {
            normalize_text(alias) for e in entity_candidates for alias in (e.aliases or [])
        }
        seen: Dict[str, TripleCandidate] = {}

        rep = members[0]
        qv = self._query_vector(rep, index)
        results: List[Tuple[Dict[str, Any], float]] = []
        if qv is not None:
            results = unified_kg_index.triple_top_k(qv, k=self.routing.triple_top_k)

        for triple, sim in results:
            head = normalize_text(triple.get("head") or "")
            tail = normalize_text(triple.get("tail") or "")
            anchor_bonus = 0.0
            if head in entity_names or tail in entity_names:
                anchor_bonus = 0.15
            score = (
                0.5 * float(sim)
                + 0.2 * float(triple.get("confidence") or 0.0)
                + 0.15 * unified_kg_index.source_article_overlap(article_ids, triple)
                + 0.15 * float(triple.get("relevance") or 0.0)
                + anchor_bonus
            )
            tid = triple.get("triple_id") or f"{head}|{triple.get('relation','')}|{tail}"
            if tid in seen:
                continue
            seen[tid] = TripleCandidate(
                head=str(triple.get("head", "")),
                relation=str(triple.get("relation", "")),
                tail=str(triple.get("tail", "")),
                head_entity_id="",
                tail_entity_id="",
                confidence=float(triple.get("confidence") or 0.0),
                relevance=float(triple.get("relevance") or 0.0),
                clarity=float(triple.get("clarity") or 0.0),
                evidence_count=int(triple.get("evidence_count") or 0),
                conflict_status=str(triple.get("conflict_status") or ""),
                source_articles=list(triple.get("source_articles") or []),
                score=float(score),
            )

        ordered = sorted(seen.values(), key=lambda t: -t.score)[:cap]
        return ordered

    def _build_summary_snippets(
        self,
        members: List[ClaimRecord],
        index: VectorIndexAdapter,
        summaries: Dict[str, Dict[str, Any]],
        article_ids: set,
    ) -> List[SummarySnippet]:
        cap = self.routing.max_summary_snippets_per_work_unit
        seen: Dict[str, SummarySnippet] = {}

        # Always include the claims' own summaries first.
        for claim in members:
            sid = claim.summary_id
            if not sid or sid in seen:
                continue
            body = summaries.get(sid)
            if not body and claim.summary_snippet:
                body = {
                    "summary_id": sid,
                    "article_id": claim.article_id,
                    "segment_id": claim.segment_id,
                    "summary_text": claim.summary_snippet,
                }
            if not body:
                continue
            seen[sid] = SummarySnippet(
                summary_id=sid,
                article_id=str(body.get("article_id", "")),
                segment_id=str(body.get("segment_id", "")),
                summary_text=str(body.get("summary_text", "")),
                score=1.0,
            )

        # Embedding top-k expansion.
        rep = members[0]
        qv = self._query_vector(rep, index)
        if qv is not None and SUMMARIES_NS in index.records:
            hits = index.search(SUMMARIES_NS, query_vector=qv, top_k=self.routing.summary_top_k)
            for hit in hits:
                sid = hit["id"]
                if sid in seen:
                    continue
                meta = hit.get("metadata") or {}
                if article_ids and meta.get("article_id") and meta.get("article_id") not in article_ids:
                    # Allow but with discount.
                    score = float(hit.get("score", 0.0)) * 0.7
                else:
                    score = float(hit.get("score", 0.0))
                body = summaries.get(sid)
                summary_text = str(body.get("summary_text") if body else hit.get("text", ""))
                seen[sid] = SummarySnippet(
                    summary_id=sid,
                    article_id=str(meta.get("article_id", "")),
                    segment_id=str(meta.get("segment_id", "")),
                    summary_text=summary_text,
                    score=score,
                )

        ordered = sorted(seen.values(), key=lambda s: -s.score)[:cap]
        return ordered

    def _build_local_kg_snippet(self, members: List[ClaimRecord]) -> Dict[str, Any]:
        local_entities: List[Dict[str, Any]] = []
        local_triples: List[Dict[str, Any]] = []
        seen_e: set = set()
        seen_t: set = set()
        for claim in members:
            for e in claim.local_entities:
                eid = str(e.get("entity_id") or e.get("canonical_name") or "")
                if not eid or eid in seen_e:
                    continue
                seen_e.add(eid)
                local_entities.append({
                    "entity_id": eid,
                    "mention": e.get("mention"),
                    "canonical_name": e.get("canonical_name"),
                    "entity_type": e.get("entity_type"),
                })
            for t in claim.local_triples:
                tid = (
                    f"{t.get('head', '')}|{t.get('relation', '')}|{t.get('tail', '')}"
                )
                if tid in seen_t:
                    continue
                seen_t.add(tid)
                local_triples.append({
                    "head": t.get("head"),
                    "relation": t.get("relation"),
                    "tail": t.get("tail"),
                })
        return {"entities": local_entities, "triples": local_triples}

    # ---------- routing decision -------------------------------------------
    def _intra_cluster_similarity(
        self,
        cluster: List[int],
        claims: List[ClaimRecord],
        index: VectorIndexAdapter,
    ) -> float:
        if len(cluster) <= 1:
            return 1.0
        vectors = index.vectors.get(CLAIMS_NS) if index else None
        if vectors is None:
            return 1.0
        records = index.records.get(CLAIMS_NS) or []
        id_to_pos = {r["id"]: i for i, r in enumerate(records)}
        positions = [id_to_pos.get(claims[i].claim_id) for i in cluster]
        positions = [p for p in positions if p is not None]
        if len(positions) <= 1:
            return 1.0
        sub = vectors[positions]
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        sub = sub / np.maximum(norms, 1e-12)
        sims = sub @ sub.T
        # Min off-diagonal similarity is the conservative measure.
        np.fill_diagonal(sims, 1.0)
        return float(sims.min())

    def _decide_routing(
        self,
        rep_claim: ClaimRecord,
        members: List[ClaimRecord],
        entities: List[EntityCandidate],
        triples: List[TripleCandidate],
        intra_sim: float,
    ) -> List[str]:
        reasons: List[str] = []

        if not entities or max((e.score for e in entities), default=0.0) < self.routing.entity_link_confidence_auto:
            reasons.append("entity_linking_ambiguous")

        if triples:
            best_triple = max(t.score for t in triples)
            if best_triple < self.routing.triple_anchor_confidence_auto:
                reasons.append("triple_anchor_ambiguous")
        else:
            reasons.append("no_stable_kg_anchor")

        # Anchor entity-triple intersection.
        entity_names = {normalize_text(e.canonical_name) for e in entities}
        anchored = any(
            normalize_text(t.head) in entity_names or normalize_text(t.tail) in entity_names
            for t in triples
        )
        if triples and not anchored:
            reasons.append("no_stable_kg_anchor")

        # Scope ambiguity for high-impact claim types.
        if rep_claim.claim_type in self.routing.scope_required_claim_types:
            scope = rep_claim.scope or {}
            keys_missing = [
                k for k in (
                    "population",
                    "severity",
                    "developmental_stage",
                    "dose_or_intensity",
                    "timepoint",
                    "comparator",
                ) if not scope.get(k)
            ]
            if keys_missing:
                reasons.append("complex_scope")

        # Negation / hedging.
        if any(_HEDGING_RE.search(c.claim_text or "") for c in members):
            reasons.append("negation_or_hedging")

        # Conflict status.
        if any(t.conflict_status not in _ACCEPTABLE_CONFLICTS for t in triples):
            reasons.append("conflict_unresolved")

        # High-impact entity types.
        high_impact = set(self.routing.high_impact_entity_types)
        if any(e.entity_type in high_impact for e in entities):
            reasons.append("high_impact_claim")

        # Cluster certainty.
        if len(members) > 1 and intra_sim < self.routing.intra_cluster_certainty_threshold:
            reasons.append("cluster_canonicalization_uncertain")

        # Deduplicate while preserving order.
        seen: set = set()
        ordered: List[str] = []
        for r in reasons:
            if r in seen:
                continue
            seen.add(r)
            ordered.append(r)
        return ordered
