"""HyperedgeComposerAgent: turn atomic claims into evidence hyperedges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..embedding_service import evidence_hyperedge_embedding_text
from ..schemas import (
    AtomicClaim,
    EvidenceHyperedge,
    HyperedgeEntity,
    SegmentEvidencePacket,
    TripleProjection,
)
from ..utils import as_list, clip01, pretty_json
from .base import PromptBackedAgent


def _claim_index(claim_id: str, fallback: int = 1) -> str:
    suffix = str(claim_id).split(":")[-1]
    return f"{int(suffix):02d}" if suffix.isdigit() else f"{fallback:02d}"


def _entity_from_raw(raw: Any) -> HyperedgeEntity | None:
    if not isinstance(raw, dict):
        return None
    entity_id = str(raw.get("entity_id") or raw.get("canonical_entity_id") or "").strip()
    mention = str(raw.get("mention") or raw.get("name") or raw.get("canonical_name") or "").strip()
    role = str(raw.get("role") or "unknown").strip() or "unknown"
    if not entity_id or not mention:
        return None
    return HyperedgeEntity(
        entity_id=entity_id,
        mention=mention,
        role=role,
        entity_type=raw.get("entity_type") or raw.get("type"),
        linking_confidence=clip01(raw.get("linking_confidence"), 1.0),
    )


def _projection_from_raw(raw: Any) -> TripleProjection | None:
    if not isinstance(raw, dict):
        return None
    head = str(raw.get("head_entity_id") or raw.get("head") or "").strip()
    relation = str(raw.get("relation") or "").strip()
    tail = str(raw.get("tail_entity_id") or raw.get("tail") or "").strip()
    if not (head and relation and tail):
        return None
    return TripleProjection(
        head_entity_id=head,
        relation=relation,
        tail_entity_id=tail,
        support=str(raw.get("support") or raw.get("evidence") or ""),
        confidence=clip01(raw.get("confidence"), 0.5),
    )


@dataclass
class HyperedgeComposerAgent(PromptBackedAgent):
    min_entities_per_hyperedge: int = 2

    def __init__(
        self,
        client: Any,
        model_name: str,
        prompt_file: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 1,
        min_entities_per_hyperedge: int = 2,
    ):
        self.min_entities_per_hyperedge = min_entities_per_hyperedge
        super().__init__(
            client=client,
            model_name=model_name,
            prompt_section="hyperkg_hyperedge_composer",
            prompt_file=prompt_file,
            temperature=temperature,
            max_retries=max_retries,
        )

    def process(
        self,
        claims: List[AtomicClaim],
        packet_map: Dict[str, SegmentEvidencePacket],
    ) -> List[EvidenceHyperedge]:
        hyperedges: List[EvidenceHyperedge] = []
        for idx, claim in enumerate(claims, start=1):
            packet = packet_map.get(claim.packet_id)
            if packet is None:
                self.review_queue.add(
                    "MISSING_PACKET",
                    claim.claim_id,
                    "Claim references a packet that is not available.",
                    "Check pipeline packet map construction.",
                    object_json=claim.to_dict(),
                )
                continue
            hyperedges.append(self.compose_one(claim, packet, idx))
        return hyperedges

    def compose_one(
        self,
        claim: AtomicClaim,
        packet: SegmentEvidencePacket,
        index: int = 1,
    ) -> EvidenceHyperedge:
        user_prompt = self.render_prompt(
            claim_json=pretty_json(claim.to_dict()),
            packet_json=pretty_json(packet.to_dict()),
            summary_text=packet.summary_text,
            local_entities=pretty_json(packet.local_entities),
            local_triples=pretty_json(packet.local_triples),
            canonical_entity_map=pretty_json(packet.canonical_entity_map),
        )
        result = self.llm.chat(self.system_prompt, user_prompt, want_json=True)
        parsed = result["json"]
        if not isinstance(parsed, dict):
            hyperedge = self._fallback_hyperedge(claim, packet, index)
            hyperedge.warnings.append("composer returned non-object JSON")
            hyperedge.raw_llm_output = result["text"]
            self._queue_validation_reviews(hyperedge, packet)
            return hyperedge

        hyperedge = self._parse_hyperedge(parsed, claim, packet, index)
        hyperedge.raw_llm_output = result["text"]
        self._validate(hyperedge, packet)
        hyperedge.embedding_text = evidence_hyperedge_embedding_text(hyperedge)
        self._queue_validation_reviews(hyperedge, packet)
        return hyperedge

    def _parse_hyperedge(
        self,
        data: Dict[str, Any],
        claim: AtomicClaim,
        packet: SegmentEvidencePacket,
        index: int,
    ) -> EvidenceHyperedge:
        entities = [
            entity
            for entity in (_entity_from_raw(raw) for raw in as_list(data.get("entities")))
            if entity is not None
        ]
        projections = [
            projection
            for projection in (
                _projection_from_raw(raw) for raw in as_list(data.get("triple_projections"))
            )
            if projection is not None
        ]
        source = dict(data.get("source") or {})
        source.setdefault("article_id", packet.article_id)
        source.setdefault("segment_id", packet.segment_id)
        source.setdefault("summary_id", packet.summary_id)
        if packet.metadata.get("section"):
            source.setdefault("section", packet.metadata.get("section"))

        raw_scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        scores = {str(k): clip01(v, 0.0) for k, v in raw_scores.items()}
        scores.setdefault("entity_linking", 0.5)
        scores.setdefault("local_kg_agreement", 0.5)
        scores.setdefault("nary_completeness", 0.5)

        return EvidenceHyperedge(
            evidence_hyperedge_id=str(
                data.get("evidence_hyperedge_id")
                or f"EH:{packet.article_id}:{packet.segment_id}:{_claim_index(claim.claim_id, index)}"
            ),
            claim_text=str(data.get("claim_text") or claim.claim_text),
            claim_type=str(data.get("claim_type") or claim.claim_type or "OTHER"),
            entities=entities,
            qualifiers=dict(data.get("qualifiers") or {}),
            triple_projections=projections,
            source=source,
            scores=scores,
            decision=str(data.get("decision") or "CANDIDATE"),
            warnings=[str(w) for w in as_list(data.get("warnings"))],
            embedding_text=data.get("embedding_text"),
            vector_id=data.get("vector_id"),
        )

    def _fallback_hyperedge(
        self,
        claim: AtomicClaim,
        packet: SegmentEvidencePacket,
        index: int,
    ) -> EvidenceHyperedge:
        entities: List[HyperedgeEntity] = []
        for mention, entity_id in packet.canonical_entity_map.items():
            if claim.candidate_entities and (
                mention not in claim.candidate_entities and entity_id not in claim.candidate_entities
            ):
                continue
            entities.append(
                HyperedgeEntity(
                    entity_id=entity_id,
                    mention=mention,
                    role="unknown",
                    entity_type=None,
                    linking_confidence=0.5,
                )
            )
        hyperedge = EvidenceHyperedge(
            evidence_hyperedge_id=f"EH:{packet.article_id}:{packet.segment_id}:{_claim_index(claim.claim_id, index)}",
            claim_text=claim.claim_text,
            claim_type=claim.claim_type or "OTHER",
            entities=entities,
            qualifiers={},
            triple_projections=[],
            source={
                "article_id": packet.article_id,
                "segment_id": packet.segment_id,
                "summary_id": packet.summary_id,
                "section": packet.metadata.get("section", ""),
            },
            scores={
                "entity_linking": 0.5,
                "local_kg_agreement": 0.0,
                "nary_completeness": 0.0,
            },
            decision="REVIEW",
        )
        hyperedge.embedding_text = evidence_hyperedge_embedding_text(hyperedge)
        self._validate(hyperedge, packet)
        return hyperedge

    def _validate(self, hyperedge: EvidenceHyperedge, packet: SegmentEvidencePacket) -> None:
        known_ids = set(packet.canonical_entity_map.values())
        entity_ids = {e.entity_id for e in hyperedge.entities}
        if len(hyperedge.entities) < self.min_entities_per_hyperedge:
            hyperedge.warnings.append("hyperedge has fewer than two entities")
            hyperedge.decision = "REVIEW"
        for entity in hyperedge.entities:
            if not (entity.entity_id and entity.mention and entity.role):
                hyperedge.warnings.append("hyperedge entity missing entity_id, mention, or role")
                hyperedge.decision = "REVIEW"
            if known_ids and entity.entity_id not in known_ids:
                hyperedge.warnings.append(f"entity_id not found in packet canonical map: {entity.entity_id}")
                hyperedge.decision = "REVIEW"
        if not hyperedge.claim_type:
            hyperedge.claim_type = "OTHER"
        for key in ("article_id", "segment_id", "summary_id"):
            if not hyperedge.source.get(key):
                hyperedge.warnings.append(f"source.{key} is missing")
                hyperedge.decision = "REVIEW"
        for projection in hyperedge.triple_projections:
            if projection.head_entity_id not in entity_ids or projection.tail_entity_id not in entity_ids:
                hyperedge.warnings.append("triple projection references unknown hyperedge entity")
                hyperedge.decision = "REVIEW"

    def _queue_validation_reviews(
        self,
        hyperedge: EvidenceHyperedge,
        packet: SegmentEvidencePacket,
    ) -> None:
        if hyperedge.decision != "REVIEW" and not hyperedge.warnings:
            return
        self.review_queue.add(
            "HYPEREDGE_VALIDATION",
            hyperedge.evidence_hyperedge_id,
            "; ".join(hyperedge.warnings) or "Hyperedge marked for review.",
            "Inspect entity roles, projections, and source provenance.",
            article_id=packet.article_id,
            segment_id=packet.segment_id,
            summary_text=packet.summary_text,
            object_json=hyperedge.to_dict(),
        )
