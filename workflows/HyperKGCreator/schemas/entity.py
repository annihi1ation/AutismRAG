"""Entity ontology + candidate retrieval schemas.

``entity_type`` (ontology class) and ``role`` (PATIENT/TREATMENT/OUTCOME in a
specific HyperEdge) are kept strictly separate. The same ``entity_id`` may
appear with different roles in different HyperEdges.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..config import ALLOWED_ENTITY_TYPES

EntityRole = Literal["PATIENT", "TREATMENT", "OUTCOME"]
CandidateSource = Literal["embedding", "string_match", "external"]
SourceField = Literal["patient_text", "treatment_text", "outcome_text"]

# Map PTO node type -> role hint (R4). Role hint never comes from entity_type.
NODE_ROLE_HINT = {
    "patient_text": "PATIENT",
    "treatment_text": "TREATMENT",
    "outcome_text": "OUTCOME",
}


class EntityRecord(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    synonyms: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    embedding: Optional[List[float]] = None

    @field_validator("entity_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        # Local check against the static ontology. Config-driven extension is
        # enforced at load time (load_entity_catalog), not here.
        if v not in ALLOWED_ENTITY_TYPES:
            raise ValueError(
                f"entity_type '{v}' not in allowed ontology; "
                "use config.allow_entity_type_extension to opt in"
            )
        return v

    def index_text(self) -> str:
        parts = [self.canonical_name, *self.synonyms]
        if self.description:
            parts.append(self.description)
        return " | ".join(p for p in parts if p)


class EntityCandidate(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    similarity: float
    source: CandidateSource = "embedding"


class EntityCandidatePack(BaseModel):
    mention: str
    role_hint: EntityRole
    source_field: SourceField
    source_node_id: Optional[str] = None  # R4: PTO node provenance
    candidates: List[EntityCandidate] = Field(default_factory=list)


class EntityLink(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    mention: str
    role: EntityRole
    confidence: Optional[float] = None
