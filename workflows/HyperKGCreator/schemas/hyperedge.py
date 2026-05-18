"""HyperEdge schema: one intellectual-disability treatment-response observation.

``type`` is fixed to ``ID_TREATMENT_RESPONSE``. Local validation enforces
non-empty texts and the presence of at least one TREATMENT and one OUTCOME
entity. Evidence-id ⊆ section-id checks are cross-object (schema_validator).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .entity import EntityLink

HyperEdgeType = Literal["ID_TREATMENT_RESPONSE"]
VerifierStatus = Literal["accepted", "needs_review", "rejected"]


class HyperEdgeSource(BaseModel):
    article_id: str
    pto_event_id: str
    evidence_span_ids: List[str] = Field(default_factory=list)


class HyperEdge(BaseModel):
    id: str
    type: HyperEdgeType = "ID_TREATMENT_RESPONSE"
    patient_text: str
    treatment_text: str
    outcome_text: str
    edge_text: str
    entities: List[EntityLink] = Field(default_factory=list)
    source: HyperEdgeSource
    confidence: float = 0.0
    verifier_status: Optional[VerifierStatus] = None
    weight: Optional[float] = None

    @model_validator(mode="after")
    def _local_invariants(self) -> "HyperEdge":
        for field_name in ("patient_text", "treatment_text", "outcome_text", "edge_text"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"HyperEdge.{field_name} must be non-empty")
        roles = {e.role for e in self.entities}
        if "TREATMENT" not in roles:
            raise ValueError("HyperEdge must include at least one role==TREATMENT entity")
        if "OUTCOME" not in roles:
            raise ValueError("HyperEdge must include at least one role==OUTCOME entity")
        return self
