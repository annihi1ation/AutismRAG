"""HyperedgeMergerAgent: merge evidence hyperedges into canonical hyperedges."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from ..embedding_service import (
    EmbeddingService,
    canonical_hyperedge_embedding_text,
    evidence_hyperedge_embedding_text,
)
from ..review_queue import ReviewQueue
from ..schemas import CanonicalHyperedge, EvidenceHyperedge
from ..utils import as_list, clip01, pretty_json, stable_hash, unique_preserve_order
from ..vector_index import VectorIndexAdapter
from .base import PromptBackedAgent


CORE_ROLES = {
    "intervention",
    "risk_factor",
    "mechanism",
    "outcome",
    "adverse_outcome",
    "condition",
    "population",
    "assessment_tool",
}


def _canonical_from_raw(raw: Any) -> CanonicalHyperedge:
    if isinstance(raw, CanonicalHyperedge):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("canonical hyperedge must be a dict or CanonicalHyperedge")
    allowed = {f.name for f in fields(CanonicalHyperedge)}
    data = {k: v for k, v in raw.items() if k in allowed}
    return CanonicalHyperedge(
        canonical_hyperedge_id=str(data.get("canonical_hyperedge_id") or ""),
        canonical_claim=str(data.get("canonical_claim") or ""),
        claim_type=str(data.get("claim_type") or "OTHER"),
        member_evidence_hyperedges=[str(x) for x in as_list(data.get("member_evidence_hyperedges"))],
        support_count=int(data.get("support_count") or 0),
        core_entity_ids=[str(x) for x in as_list(data.get("core_entity_ids"))],
        qualifier_summary=dict(data.get("qualifier_summary") or {}),
        scope_summary=dict(data.get("scope_summary") or {}),
        confidence_summary=dict(data.get("confidence_summary") or {}),
        related_hyperedges=[dict(x) for x in as_list(data.get("related_hyperedges")) if isinstance(x, dict)],
        conflicts=[dict(x) for x in as_list(data.get("conflicts")) if isinstance(x, dict)],
        embedding_text=data.get("embedding_text"),
        vector_id=data.get("vector_id"),
    )


@dataclass
class HyperedgeMergerAgent(PromptBackedAgent):
    embedding_service: Optional[EmbeddingService] = None
    vector_index: Optional[VectorIndexAdapter] = None
    auto_merge_similarity: float = 0.90
    llm_merge_similarity: float = 0.78
    max_merge_candidates: int = 10

    def __init__(
        self,
        client: Any,
        model_name: str,
        embedding_service: Optional[EmbeddingService] = None,
        vector_index: Optional[VectorIndexAdapter] = None,
        prompt_file: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 1,
        auto_merge_similarity: float = 0.90,
        llm_merge_similarity: float = 0.78,
        max_merge_candidates: int = 10,
    ):
        self.embedding_service = embedding_service
        self.vector_index = vector_index
        self.auto_merge_similarity = auto_merge_similarity
        self.llm_merge_similarity = llm_merge_similarity
        self.max_merge_candidates = max_merge_candidates
        super().__init__(
            client=client,
            model_name=model_name,
            prompt_section="hyperkg_hyperedge_merger",
            prompt_file=prompt_file,
            temperature=temperature,
            max_retries=max_retries,
        )
        self.review_queue = ReviewQueue()

    def process(
        self,
        evidence_hyperedges: List[EvidenceHyperedge],
        existing_canonical_hyperedges: Optional[List[CanonicalHyperedge]] = None,
    ) -> Tuple[List[CanonicalHyperedge], List[Dict[str, Any]]]:
        canonical = [_canonical_from_raw(h) for h in (existing_canonical_hyperedges or [])]
        for hyperedge in evidence_hyperedges:
            if hyperedge.decision != "ACCEPT":
                continue
            if not hyperedge.embedding_text:
                hyperedge.embedding_text = evidence_hyperedge_embedding_text(hyperedge)
            target = self._find_merge_target(hyperedge, canonical)
            if target is None:
                new_hyperedge = self._new_canonical(hyperedge)
                canonical.append(new_hyperedge)
                self._index_canonical(new_hyperedge)
            else:
                self._merge_into(target, hyperedge)
                self._index_canonical(target)
        return canonical, self.review_items

    def _find_merge_target(
        self,
        hyperedge: EvidenceHyperedge,
        canonical: List[CanonicalHyperedge],
    ) -> Optional[CanonicalHyperedge]:
        if not canonical:
            return None

        h_key = self._blocking_key(hyperedge)
        for candidate in canonical:
            if self._canonical_blocking_key(candidate) == h_key:
                return candidate

        dense_candidates = self._dense_candidates(hyperedge, canonical)
        for candidate, score in dense_candidates:
            if score >= self.auto_merge_similarity and self._compatible(hyperedge, candidate):
                return candidate
            if score >= self.llm_merge_similarity:
                decision = self._llm_merge_decision(hyperedge, [candidate])
                if decision == "MERGE":
                    return candidate
                if decision in {"RELATED_TO", "DIFFERENT_SCOPE", "CONTRADICTS"}:
                    self._record_relation(candidate, hyperedge, decision, score)
                    return None
                self.review_queue.add(
                    "UNCERTAIN_MERGE",
                    hyperedge.evidence_hyperedge_id,
                    f"Dense similarity {score:.3f} reached LLM threshold but was not merged.",
                    "Review whether this evidence hyperedge should join an existing canonical hyperedge.",
                    object_json={
                        "evidence_hyperedge": hyperedge.to_dict(),
                        "candidate": candidate.to_dict(),
                        "score": score,
                    },
                )
        return None

    def _dense_candidates(
        self,
        hyperedge: EvidenceHyperedge,
        canonical: List[CanonicalHyperedge],
    ) -> List[Tuple[CanonicalHyperedge, float]]:
        if not (self.embedding_service and self.embedding_service.enabled and self.vector_index):
            return []
        if not self.vector_index.records.get("canonical_hyperedge"):
            for candidate in canonical:
                self._index_canonical(candidate)
        try:
            query = self.embedding_service.encode_one(hyperedge.embedding_text or "")
            hits = self.vector_index.search(
                "canonical_hyperedge",
                query_vector=query,
                top_k=self.max_merge_candidates,
                filters={"claim_type": hyperedge.claim_type},
            )
        except Exception:
            return []
        by_id = {c.canonical_hyperedge_id: c for c in canonical}
        out = []
        for hit in hits:
            candidate = by_id.get(hit["id"])
            if candidate is not None:
                out.append((candidate, float(hit.get("score", 0.0))))
        return out

    def _llm_merge_decision(
        self,
        hyperedge: EvidenceHyperedge,
        candidates: List[CanonicalHyperedge],
    ) -> str:
        user_prompt = self.render_prompt(
            evidence_hyperedge_json=pretty_json(hyperedge.to_dict()),
            candidate_canonical_hyperedges_json=pretty_json([c.to_dict() for c in candidates]),
            merge_criteria="claim_type, core entity roles, triple projection overlap, qualifier compatibility, direction/negation, and scope.",
        )
        result = self.llm.chat(self.system_prompt, user_prompt, want_json=True)
        parsed = result["json"] if isinstance(result["json"], dict) else {}
        return str(parsed.get("decision") or parsed.get("merge_decision") or "NEW").upper()

    def _new_canonical(self, hyperedge: EvidenceHyperedge) -> CanonicalHyperedge:
        core_ids = self._core_entity_ids(hyperedge)
        relations = sorted({p.relation for p in hyperedge.triple_projections})
        canonical_id = f"CH:{hyperedge.claim_type}:{stable_hash([core_ids, relations, hyperedge.claim_text])}"
        canonical = CanonicalHyperedge(
            canonical_hyperedge_id=canonical_id,
            canonical_claim=hyperedge.claim_text,
            claim_type=hyperedge.claim_type,
            member_evidence_hyperedges=[hyperedge.evidence_hyperedge_id],
            support_count=1,
            core_entity_ids=core_ids,
            qualifier_summary=self._summarize_qualifiers([hyperedge]),
            scope_summary=self._summarize_scope([hyperedge]),
            confidence_summary=self._summarize_confidence([hyperedge]),
        )
        canonical.embedding_text = canonical_hyperedge_embedding_text(canonical)
        return canonical

    def _merge_into(self, canonical: CanonicalHyperedge, hyperedge: EvidenceHyperedge) -> None:
        canonical.member_evidence_hyperedges = unique_preserve_order(
            canonical.member_evidence_hyperedges + [hyperedge.evidence_hyperedge_id]
        )
        canonical.support_count = len(canonical.member_evidence_hyperedges)
        canonical.core_entity_ids = unique_preserve_order(
            canonical.core_entity_ids + self._core_entity_ids(hyperedge)
        )
        samples = canonical.scope_summary.setdefault("claim_samples", [])
        samples.append(hyperedge.claim_text)
        canonical.qualifier_summary = self._merge_qualifier_summary(
            canonical.qualifier_summary,
            hyperedge.qualifiers,
        )
        canonical.scope_summary = self._merge_scope_summary(canonical.scope_summary, hyperedge)
        canonical.confidence_summary = self._merge_confidence_summary(
            canonical.confidence_summary,
            hyperedge,
        )
        canonical.embedding_text = canonical_hyperedge_embedding_text(canonical)

    @staticmethod
    def _core_entity_ids(hyperedge: EvidenceHyperedge) -> List[str]:
        core = [e.entity_id for e in hyperedge.entities if e.role in CORE_ROLES]
        return sorted(unique_preserve_order(core or [e.entity_id for e in hyperedge.entities]))

    @staticmethod
    def _blocking_key(hyperedge: EvidenceHyperedge) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        return (
            hyperedge.claim_type,
            tuple(HyperedgeMergerAgent._core_entity_ids(hyperedge)),
            tuple(sorted({p.relation for p in hyperedge.triple_projections})),
        )

    @staticmethod
    def _canonical_blocking_key(
        hyperedge: CanonicalHyperedge,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        relations = tuple(sorted(as_list(hyperedge.scope_summary.get("projection_relations"))))
        return (hyperedge.claim_type, tuple(sorted(hyperedge.core_entity_ids)), relations)

    @staticmethod
    def _compatible(hyperedge: EvidenceHyperedge, candidate: CanonicalHyperedge) -> bool:
        if hyperedge.claim_type != candidate.claim_type:
            return False
        h_core = set(HyperedgeMergerAgent._core_entity_ids(hyperedge))
        c_core = set(candidate.core_entity_ids)
        if h_core and c_core and not h_core.intersection(c_core):
            return False
        adverse = {"ADVERSE_EFFECT"}
        intervention = {"INTERVENTION_OUTCOME"}
        if hyperedge.claim_type in adverse and candidate.claim_type in intervention:
            return False
        return True

    def _index_canonical(self, canonical: CanonicalHyperedge) -> None:
        if not (self.embedding_service and self.embedding_service.enabled and self.vector_index):
            return
        if not canonical.embedding_text:
            canonical.embedding_text = canonical_hyperedge_embedding_text(canonical)
        try:
            vector = self.embedding_service.encode_texts([canonical.embedding_text])
            self.vector_index.add(
                "canonical_hyperedge",
                [canonical.canonical_hyperedge_id],
                [canonical.embedding_text],
                vector,
                [{"claim_type": canonical.claim_type}],
            )
        except Exception:
            return

    @staticmethod
    def _summarize_qualifiers(hyperedges: List[EvidenceHyperedge]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for hyperedge in hyperedges:
            for key, value in hyperedge.qualifiers.items():
                summary.setdefault(key, [])
                summary[key].extend(as_list(value))
        return {k: unique_preserve_order(v) for k, v in summary.items()}

    @staticmethod
    def _summarize_scope(hyperedges: List[EvidenceHyperedge]) -> Dict[str, Any]:
        return {
            "source_articles": unique_preserve_order(
                h.source.get("article_id") for h in hyperedges if h.source.get("article_id")
            ),
            "source_segments": unique_preserve_order(
                h.source.get("segment_id") for h in hyperedges if h.source.get("segment_id")
            ),
            "projection_relations": sorted(
                {p.relation for h in hyperedges for p in h.triple_projections}
            ),
            "claim_samples": [h.claim_text for h in hyperedges[:5]],
        }

    @staticmethod
    def _summarize_confidence(hyperedges: List[EvidenceHyperedge]) -> Dict[str, Any]:
        values = [clip01(h.scores.get("integration_score"), 0.0) for h in hyperedges]
        if not values:
            return {"avg_integration_score": 0.0}
        return {
            "avg_integration_score": round(sum(values) / len(values), 4),
            "min_integration_score": round(min(values), 4),
            "max_integration_score": round(max(values), 4),
            "_integration_values": values,
        }

    @staticmethod
    def _merge_qualifier_summary(summary: Dict[str, Any], qualifiers: Dict[str, Any]) -> Dict[str, Any]:
        merged = {k: as_list(v) for k, v in (summary or {}).items()}
        for key, value in qualifiers.items():
            merged.setdefault(key, [])
            merged[key].extend(as_list(value))
        return {k: unique_preserve_order(v) for k, v in merged.items()}

    @staticmethod
    def _merge_scope_summary(summary: Dict[str, Any], hyperedge: EvidenceHyperedge) -> Dict[str, Any]:
        summary = dict(summary or {})
        summary["source_articles"] = unique_preserve_order(
            as_list(summary.get("source_articles")) + [hyperedge.source.get("article_id")]
        )
        summary["source_segments"] = unique_preserve_order(
            as_list(summary.get("source_segments")) + [hyperedge.source.get("segment_id")]
        )
        summary["projection_relations"] = sorted(
            set(as_list(summary.get("projection_relations")))
            | {p.relation for p in hyperedge.triple_projections}
        )
        return summary

    @staticmethod
    def _merge_confidence_summary(summary: Dict[str, Any], hyperedge: EvidenceHyperedge) -> Dict[str, Any]:
        values = as_list((summary or {}).get("_integration_values"))
        values.append(clip01(hyperedge.scores.get("integration_score"), 0.0))
        public = {
            "avg_integration_score": round(sum(values) / len(values), 4),
            "min_integration_score": round(min(values), 4),
            "max_integration_score": round(max(values), 4),
            "_integration_values": values,
        }
        return public

    @staticmethod
    def _record_relation(
        candidate: CanonicalHyperedge,
        hyperedge: EvidenceHyperedge,
        relation: str,
        score: float,
    ) -> None:
        relation_item = {
            "relation": relation,
            "evidence_hyperedge_id": hyperedge.evidence_hyperedge_id,
            "score": round(score, 4),
        }
        if relation == "CONTRADICTS":
            candidate.conflicts.append(relation_item)
        else:
            candidate.related_hyperedges.append(relation_item)
