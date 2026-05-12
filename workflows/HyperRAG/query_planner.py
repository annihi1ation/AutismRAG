"""Module 2: SimpleQueryPlanner — deterministic, no LLM, no ranking.

Turns case_state into a small set of retrieval queries plus the spec-fixed
claim_type filters and diffusion policy. possible_function is included in
queries but always framed as 'possible / hypothesized', never confirmed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


CLAIM_TYPE_FILTERS = {
    "assessment_evidence": [
        "SYMPTOM_EXHIBITION",
        "FUNCTIONAL_PROFILE",
        "SEVERITY_CLASSIFICATION",
        "DIAGNOSTIC_ASSESSMENT",
        "DEVELOPMENTAL_POSITION",
        "CARE_GIVING_CONTEXT",
        "PREFERENCE",
    ],
    "intervention_evidence": [
        "BEHAVIORAL_INTERVENTION",
        "EDUCATIONAL_INTERVENTION",
        "INTERVENTION_OUTCOME",
        "COMPARATIVE_OUTCOME",
        "COMPARATIVE",
        "COMPARISON",
    ],
    "side_effect_evidence": [
        "ADVERSE_EFFECT",
        "RISK_FACTOR",
        "SERVICE_ACCESS",
        "PHARMACOLOGICAL_INTERVENTION",
        "PRESCRIPTION",
        "PRESCRIPTION_RATE",
    ],
    "counterfactual_evidence": [
        "COMPARATIVE",
        "COMPARISON",
        "COMPARATIVE_OUTCOME",
        "INTERVENTION_OUTCOME",
        "ADVERSE_EFFECT",
        "RISK_FACTOR",
    ],
}

DIFFUSION_POLICY = {
    "default_hops": 1,
    "allow_second_hop_if": "no intervention_evidence found",
}


_WS_RE = re.compile(r"\s+")


def _join(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        return " ".join(str(v).strip() for v in values if str(v).strip())
    return str(values).strip()


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _target_text(case_state: Dict[str, Any]) -> str:
    """Return target_behavior, falling back to all candidates if ambiguous."""
    target = (case_state.get("target_behavior") or "").strip()
    if target:
        return target
    if case_state.get("ambiguity"):
        behaviors = [
            (c.get("behavior") or "").strip()
            for c in case_state.get("candidate_challenging_behaviors") or []
            if (c.get("behavior") or "").strip()
        ]
        return " ".join(behaviors)
    return ""


def _seed_entities(
    case_state: Dict[str, Any],
    subject_only: bool = True,
) -> List[Dict[str, str]]:
    seeds: List[Dict[str, str]] = []
    seen: set = set()

    def _add(mention: str, entity_type: str, role: str) -> None:
        m = (mention or "").strip()
        if not m:
            return
        key = (m.lower(), entity_type, role)
        if key in seen:
            return
        seen.add(key)
        seeds.append({"mention": m, "entity_type": entity_type, "role": role})

    target = (case_state.get("target_behavior") or "").strip()
    if target:
        _add(target, "Symptom_Behavior", "condition")
    for c in case_state.get("candidate_challenging_behaviors") or []:
        _add((c.get("behavior") or "").strip(), "Symptom_Behavior", "condition")

    pf = case_state.get("patient_features") or {}
    for symptom in pf.get("symptom_behavior") or []:
        _add(symptom, "Symptom_Behavior", "condition")
    for emotion in pf.get("symptom_emotion") or []:
        _add(emotion, "Symptom_Emotion", "condition")
    for ability in pf.get("functional_ability") or []:
        _add(ability, "Functional_Ability", "profile")
    if not subject_only:
        for env in pf.get("social_environment") or []:
            _add(env, "Social_Environment", "environment")
        for med in pf.get("medication") or []:
            _add(med, "Medication", "current_treatment")
    for therapy in pf.get("therapeutic_approach") or []:
        _add(therapy, "Therapeutic_Approach", "current_treatment")
    for edu in pf.get("educational_approach") or []:
        _add(edu, "Educational_Approach", "current_treatment")
    return seeds


class SimpleQueryPlanner:
    """Deterministic, rule-based. Never calls the LLM."""

    def run(
        self,
        case_state: Dict[str, Any],
        subject_only_seeding: bool = True,
    ) -> Dict[str, Any]:
        pf = case_state.get("patient_features") or {}
        target = _target_text(case_state)
        functional_ability = _join(pf.get("functional_ability"))
        developmental_stage = pf.get("developmental_stage") or ""
        social_environment = _join(pf.get("social_environment"))
        clinical_severity = pf.get("clinical_severity") or ""
        symptom_emotion = _join(pf.get("symptom_emotion"))
        current_therapeutic = _join(pf.get("therapeutic_approach"))
        current_educational = _join(pf.get("educational_approach"))
        possible_function = (case_state.get("possible_function") or "").strip()
        user_strategy_x = (case_state.get("user_strategy_x") or "") if case_state.get(
            "user_strategy_x"
        ) else ""

        assessment_query = _clean(
            f"{target} {functional_ability} {developmental_stage} {clinical_severity} "
            f"{social_environment} {symptom_emotion} symptom behavior functional profile "
            f"frequency intensity antecedent consequence possible function"
        )
        intervention_query = _clean(
            f"{target} autism {functional_ability} {social_environment} {clinical_severity} "
            f"{current_therapeutic} {current_educational} possible function {possible_function} "
            f"behavioral intervention educational intervention intervention outcome"
        )
        side_effect_query = None

        retrieval_task = "intervention_planning"
        counterfactual_query = None
        if user_strategy_x:
            retrieval_task = "counterfactual_strategy_check"
            counterfactual_query = _clean(
                f"{user_strategy_x} {target} autism {functional_ability} {clinical_severity} "
                f"possible function {possible_function} intervention outcome comparison "
                f"adverse effect risk factor"
            )

        return {
            "retrieval_task": retrieval_task,
            "seed_entities": _seed_entities(case_state, subject_only=subject_only_seeding),
            "retrieval_queries": {
                "assessment_query": assessment_query,
                "intervention_query": intervention_query,
                "side_effect_query": side_effect_query,
                "counterfactual_query": counterfactual_query,
            },
            "claim_type_filters": {k: list(v) for k, v in CLAIM_TYPE_FILTERS.items()},
            "diffusion_policy": dict(DIFFUSION_POLICY),
            "user_strategy_x": user_strategy_x or None,
        }

    def plan_side_effect_followup(
        self,
        case_state: Dict[str, Any],
        intervention_mentions: List[str],
    ) -> Dict[str, Any]:
        """Stage-2 plan: focused side-effect lookup for the interventions we
        actually plan to recommend (plus any current medication). All other
        retrieval queries are None so DenseRetriever skips them."""
        pf = case_state.get("patient_features") or {}
        target = _target_text(case_state)
        current_medication = _join(pf.get("medication"))
        intervention_blob = " ".join(
            (m or "").strip() for m in intervention_mentions if (m or "").strip()
        )
        side_effect_query = _clean(
            f"{intervention_blob} {target} autism {current_medication} "
            f"adverse effect risk factor service access limitation "
            f"medication prescription"
        ) or None

        user_strategy_x = (case_state.get("user_strategy_x") or "").strip() or None

        return {
            "retrieval_task": "side_effect_lookup",
            "seed_entities": [],
            "retrieval_queries": {
                "assessment_query": None,
                "intervention_query": None,
                "side_effect_query": side_effect_query,
                "counterfactual_query": None,
            },
            "claim_type_filters": {k: list(v) for k, v in CLAIM_TYPE_FILTERS.items()},
            "diffusion_policy": dict(DIFFUSION_POLICY),
            "user_strategy_x": user_strategy_x,
        }
