"""HyperKGBuilder — embedding-routed HyperKG construction over collected claims.

This workflow is intentionally separated from ``workflows.HyperKGConstruction``.
It does NOT re-run ingestion, reader, summarizer, entity / relationship
extraction, or unified-KG construction. Inputs are:

* collected claims (JSONL)
* the existing unified KG (entities + triples + metadata)

Outputs evidence + canonical hyperedges plus vector indexes. Online LLM is
routed only to bounded candidate packs that pass the router.
"""

from .config import (
    BudgetConfig,
    EmbeddingConfig,
    HyperKGRunConfig,
    RoutingConfig,
)
from .schemas import (
    CanonicalHyperedge,
    CandidatePack,
    ClaimRecord,
    EntityCandidate,
    EvidenceHyperedge,
    HyperKGRunReport,
    LLMUsageRecord,
    SummarySnippet,
    TripleCandidate,
    TripleProjection,
    WorkUnit,
)

__all__ = [
    "BudgetConfig",
    "CanonicalHyperedge",
    "CandidatePack",
    "ClaimRecord",
    "EmbeddingConfig",
    "EntityCandidate",
    "EvidenceHyperedge",
    "HyperKGRunConfig",
    "HyperKGRunReport",
    "LLMUsageRecord",
    "RoutingConfig",
    "SummarySnippet",
    "TripleCandidate",
    "TripleProjection",
    "WorkUnit",
]
