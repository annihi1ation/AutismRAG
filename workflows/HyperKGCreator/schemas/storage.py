"""Storage / routing-bundle / report schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .article import SectionSpan

IncidenceRole = Literal["PATIENT", "TREATMENT", "OUTCOME"]
ReportStatus = Literal["pass", "fail"]


class IncidenceRecord(BaseModel):
    """entity_id --role--> hyperedge_id"""

    he_id: str
    entity_id: str
    role: IncidenceRole
    mention: str
    confidence: Optional[float] = None


class ValidationReport(BaseModel):
    status: ReportStatus
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    subject: Optional[str] = None

    @classmethod
    def from_errors(
        cls,
        errors: List[str],
        warnings: Optional[List[str]] = None,
        subject: Optional[str] = None,
    ) -> "ValidationReport":
        return cls(
            status="fail" if errors else "pass",
            errors=errors,
            warnings=warnings or [],
            subject=subject,
        )


class EvidenceBundle(BaseModel):
    """Router output. Section ids only."""

    article_id: str
    patient_section_ids: List[str] = Field(default_factory=list)
    treatment_section_ids: List[str] = Field(default_factory=list)
    outcome_section_ids: List[str] = Field(default_factory=list)
    comparator_section_ids: List[str] = Field(default_factory=list)
    study_design_section_ids: List[str] = Field(default_factory=list)
    selected_section_ids: List[str] = Field(default_factory=list)


class PTOBuilderInput(BaseModel):
    """Compact, non-LLM-prepared input for Agent 1. Section ids preserved."""

    article_id: str
    selected_sections: List[SectionSpan] = Field(default_factory=list)
    section_role_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    pairwise_section_similarities: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
