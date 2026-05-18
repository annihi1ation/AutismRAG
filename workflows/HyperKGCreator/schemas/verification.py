"""Verification, risk-gate and support-score schemas.

``EvidenceQuote`` (R2): a verbatim slice tied to a SECTION id, used only as
advisory context for the conditional verifier. Quotes are NEVER evidence ids,
never persisted as spans.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .hyperedge import HyperEdge

VerificationStatus = Literal["accepted", "needs_review", "rejected"]


class EvidenceQuote(BaseModel):
    section_id: str
    quote: str


class VerificationResult(BaseModel):
    he_id: str
    status: VerificationStatus
    reason: Optional[str] = None
    repaired_hyperedge: Optional[HyperEdge] = None


class RiskGateResult(BaseModel):
    he_id: str
    needs_verification: bool
    risk_score: float
    risk_reasons: List[str] = Field(default_factory=list)


class SupportScore(BaseModel):
    he_id: str
    patient_support_score: float
    treatment_support_score: float
    outcome_support_score: float
    edge_support_score: float
    nearest_patient_section_ids: List[str] = Field(default_factory=list)
    nearest_treatment_section_ids: List[str] = Field(default_factory=list)
    nearest_outcome_section_ids: List[str] = Field(default_factory=list)
    nearest_edge_section_ids: List[str] = Field(default_factory=list)

    def min_support(self) -> float:
        return min(
            self.patient_support_score,
            self.treatment_support_score,
            self.outcome_support_score,
            self.edge_support_score,
        )


class VerifierInput(BaseModel):
    """Bundle handed to the conditional verifier (advisory quotes allowed)."""

    hyperedge: HyperEdge
    risk_reasons: List[str] = Field(default_factory=list)
    nearest_section_ids: List[str] = Field(default_factory=list)
    evidence_quotes: List[EvidenceQuote] = Field(default_factory=list)
