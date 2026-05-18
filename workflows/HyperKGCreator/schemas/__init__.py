"""Pydantic v2 schemas for HyperKGCreator.

Models enforce LOCAL invariants only. Cross-object reference integrity is
checked by ``validation.schema_validator`` (plan R3).
"""

from __future__ import annotations

from .article import (
    UNASSIGNED_TABLES_SECTION_ID,
    ArticleDocument,
    ArticleMetadata,
    SectionSpan,
    SectionType,
    TableBlock,
)
from .entity import (
    NODE_ROLE_HINT,
    EntityCandidate,
    EntityCandidatePack,
    EntityLink,
    EntityRecord,
    EntityRole,
)
from .hyperedge import HyperEdge, HyperEdgeSource, HyperEdgeType, VerifierStatus
from .pto_graph import (
    ArticlePTOGraphLite,
    OutcomeNode,
    PatientNode,
    PTOEvent,
    TreatmentNode,
)
from .storage import (
    EvidenceBundle,
    IncidenceRecord,
    PTOBuilderInput,
    ValidationReport,
)
from .verification import (
    EvidenceQuote,
    RiskGateResult,
    SupportScore,
    VerificationResult,
    VerifierInput,
)

__all__ = [
    "ArticleDocument",
    "ArticleMetadata",
    "SectionSpan",
    "SectionType",
    "TableBlock",
    "UNASSIGNED_TABLES_SECTION_ID",
    "ArticlePTOGraphLite",
    "PatientNode",
    "TreatmentNode",
    "OutcomeNode",
    "PTOEvent",
    "EntityRecord",
    "EntityCandidate",
    "EntityCandidatePack",
    "EntityLink",
    "EntityRole",
    "NODE_ROLE_HINT",
    "HyperEdge",
    "HyperEdgeSource",
    "HyperEdgeType",
    "VerifierStatus",
    "VerificationResult",
    "RiskGateResult",
    "SupportScore",
    "EvidenceQuote",
    "VerifierInput",
    "IncidenceRecord",
    "ValidationReport",
    "EvidenceBundle",
    "PTOBuilderInput",
]
