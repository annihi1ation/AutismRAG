"""Typed artifacts exchanged by the HyperKG construction agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SegmentEvidencePacket:
    packet_id: str
    article_id: str
    segment_id: str
    summary_id: str
    summary_text: str
    local_entities: List[Dict[str, Any]]
    local_triples: List[Dict[str, Any]]
    canonical_entity_map: Dict[str, str]
    unresolved_entities: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AtomicClaim:
    claim_id: str
    packet_id: str
    claim_text: str
    claim_type: str
    candidate_entities: List[str]
    candidate_triples: List[Dict[str, Any]]
    source_summary_id: str
    source_segment_id: str
    metadata: Dict[str, Any]
    raw_llm_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HyperedgeEntity:
    entity_id: str
    mention: str
    role: str
    entity_type: Optional[str] = None
    linking_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TripleProjection:
    head_entity_id: str
    relation: str
    tail_entity_id: str
    support: str
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceHyperedge:
    evidence_hyperedge_id: str
    claim_text: str
    claim_type: str
    entities: List[HyperedgeEntity]
    qualifiers: Dict[str, Any]
    triple_projections: List[TripleProjection]
    source: Dict[str, Any]
    scores: Dict[str, float]
    decision: str = "CANDIDATE"
    warnings: List[str] = field(default_factory=list)
    embedding_text: Optional[str] = None
    vector_id: Optional[str] = None
    raw_llm_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalHyperedge:
    canonical_hyperedge_id: str
    canonical_claim: str
    claim_type: str
    member_evidence_hyperedges: List[str]
    support_count: int
    core_entity_ids: List[str]
    qualifier_summary: Dict[str, Any]
    scope_summary: Dict[str, Any]
    confidence_summary: Dict[str, Any]
    related_hyperedges: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    embedding_text: Optional[str] = None
    vector_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
