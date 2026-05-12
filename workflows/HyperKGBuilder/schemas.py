"""Typed dataclasses for the HyperKGBuilder workflow."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClaimRecord:
    claim_id: str
    claim_text: str
    claim_type: str
    polarity: str = ""
    modality: str = ""
    candidate_entities: List[Dict[str, Any]] = field(default_factory=list)
    candidate_triples: List[Dict[str, Any]] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)
    article_id: str = ""
    segment_id: str = ""
    summary_id: str = ""
    summary_snippet: Optional[str] = None
    local_entities: List[Dict[str, Any]] = field(default_factory=list)
    local_triples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityCandidate:
    entity_id: str
    canonical_name: str
    entity_type: str = ""
    aliases: List[str] = field(default_factory=list)
    normalized_id: str = ""
    mention_count: int = 0
    source_articles: List[str] = field(default_factory=list)
    score: float = 0.0
    source: str = "unified_kg"  # one of: claim, unified_kg, local

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TripleCandidate:
    head: str
    relation: str
    tail: str
    head_entity_id: str = ""
    tail_entity_id: str = ""
    confidence: float = 0.0
    relevance: float = 0.0
    clarity: float = 0.0
    evidence_count: int = 0
    conflict_status: str = ""
    source_articles: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SummarySnippet:
    summary_id: str
    article_id: str
    segment_id: str
    summary_text: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePack:
    pack_id: str
    representative_claim_id: str
    representative_claim_text: str
    claim_type: str
    member_claim_ids: List[str]
    entity_candidates: List[EntityCandidate]
    triple_candidates: List[TripleCandidate]
    summary_snippets: List[SummarySnippet]
    local_kg_snippet: Dict[str, Any]
    routing_reasons: List[str]
    auto_decision: str = "auto"  # "auto" or "online_llm"
    pack_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# WorkUnit is just a CandidatePack with auto_decision == "online_llm".
WorkUnit = CandidatePack


@dataclass
class TripleProjection:
    head_entity_id: str
    relation: str
    tail_entity_id: str
    support: str
    confidence: float = 0.5
    evidence_hyperedge_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceHyperedge:
    evidence_hyperedge_id: str
    claim_text: str
    claim_type: str
    entities: List[Dict[str, Any]]
    qualifiers: Dict[str, Any]
    scope: Dict[str, Any]
    polarity: str
    negation: bool
    speculative: bool
    confidence: float
    primary_triples: List[Dict[str, Any]]
    supporting_triples: List[Dict[str, Any]]
    warnings: List[str]
    method: str  # "online_llm" or "auto_embedding_kg_anchor"
    source: Dict[str, Any]
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
    evidence_count: int
    source_articles: List[str]
    mean_confidence: float
    mean_relevance: float
    mean_clarity: float
    conflict_status_distribution: Dict[str, int]
    entity_role_distribution: Dict[str, int]
    core_entity_ids: List[str]
    scope_summary: Dict[str, Any]
    relations: List[Dict[str, Any]] = field(default_factory=list)
    embedding_text: Optional[str] = None
    vector_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMUsageRecord:
    work_unit_id: str
    pack_hash: str
    model: str
    routing_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    parse_status: str = "ok"  # "ok", "retry_succeeded", "failed"
    attempt_count: int = 1
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HyperKGRunReport:
    run_id: str
    started_at: float
    finished_at: float = 0.0
    resumed: bool = False
    dry_run: bool = False
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    phase_counts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    routing_distribution: Dict[str, int] = field(default_factory=dict)
    parse_status_distribution: Dict[str, int] = field(default_factory=dict)
    budget_usage: Dict[str, Any] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
