"""Article-level Patient-Treatment-Outcome graph schema (lite).

Every ``evidence_span_id`` must be a SECTION id. Cross-object reference checks
live in ``validation.schema_validator``; here only local invariants apply.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

PatientGranularity = Literal[
    "single_case", "case_series", "group", "subgroup", "study_arm", "unclear"
]
OutcomeDirection = Literal[
    "improved", "no_effect", "worsened", "adverse_event", "mixed", "unclear"
]


class PatientNode(BaseModel):
    id: str
    text: str
    granularity: PatientGranularity = "unclear"
    evidence_span_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class TreatmentNode(BaseModel):
    id: str
    text: str
    treatment_type_hint: Optional[str] = None
    evidence_span_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class OutcomeNode(BaseModel):
    id: str
    text: str
    direction: OutcomeDirection = "unclear"
    timepoint: Optional[str] = None
    evidence_span_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class PTOEvent(BaseModel):
    id: str
    patient_node_ids: List[str] = Field(default_factory=list)
    treatment_node_ids: List[str] = Field(default_factory=list)
    outcome_node_ids: List[str] = Field(default_factory=list)
    arm_label: Optional[str] = None
    comparator: Optional[str] = None
    timepoint: Optional[str] = None
    evidence_span_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    rejected: bool = False

    @model_validator(mode="after")
    def _completeness(self) -> "PTOEvent":
        # A non-rejected event must reference at least one node of each role.
        if not self.rejected:
            if not self.patient_node_ids:
                raise ValueError(f"PTOEvent {self.id} missing patient_node_ids")
            if not self.treatment_node_ids:
                raise ValueError(f"PTOEvent {self.id} missing treatment_node_ids")
            if not self.outcome_node_ids:
                raise ValueError(f"PTOEvent {self.id} missing outcome_node_ids")
        return self


class ArticlePTOGraphLite(BaseModel):
    article_id: str
    patient_nodes: List[PatientNode] = Field(default_factory=list)
    treatment_nodes: List[TreatmentNode] = Field(default_factory=list)
    outcome_nodes: List[OutcomeNode] = Field(default_factory=list)
    pto_events: List[PTOEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids_per_type(self) -> "ArticlePTOGraphLite":
        for name, nodes in (
            ("patient_nodes", self.patient_nodes),
            ("treatment_nodes", self.treatment_nodes),
            ("outcome_nodes", self.outcome_nodes),
            ("pto_events", self.pto_events),
        ):
            ids = [n.id for n in nodes]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate id within {name}")
        return self
