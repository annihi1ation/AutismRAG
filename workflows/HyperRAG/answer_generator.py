"""Module 6: AnswerGeneratorVerifier — LLM (or stub) generation + rule-based verifier.

Verifier never rewrites the answer wholesale; it performs minimal repairs
(remove hallucinated IDs, gate medication, enforce ambiguity wording, derive
counterfactual verdict from hyperedge schema fields).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from .evidence_organizer import canonical_claim_type
from .loaders import HyperRAGContext

logger = logging.getLogger(__name__)


_MEDICATION_RE = re.compile(
    r"\bmedicat\w*\b|\bdrug\w*\b|\bprescription\w*\b|\bpharmac\w*\b|\banti[- ]?psychotic\w*\b",
    re.IGNORECASE,
)
_VERDICT_LABELS = {
    "supported",
    "partially_supported",
    "not_supported_as_primary",
    "insufficient_evidence",
}
_INTERVENTION_TYPES = {
    "BEHAVIORAL_INTERVENTION",
    "EDUCATIONAL_INTERVENTION",
    "INTERVENTION_OUTCOME",
    "COMPARATIVE_OUTCOME",
    "COMPARATIVE",
    "COMPARISON",
}
_SIDE_EFFECT_TYPES = {
    "ADVERSE_EFFECT",
    "RISK_FACTOR",
    "SERVICE_ACCESS",
    "PHARMACOLOGICAL_INTERVENTION",
    "PRESCRIPTION",
    "PRESCRIPTION_RATE",
}
_PHARM_TYPES = {"PHARMACOLOGICAL_INTERVENTION", "INTERVENTION_OUTCOME"}


def _user_asked_about_medication(question: str) -> bool:
    return bool(_MEDICATION_RE.search(question or ""))


def _slim_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_hyperedge_id": edge.get("evidence_hyperedge_id"),
        "claim_text": edge.get("claim_text"),
        "claim_type": canonical_claim_type(edge.get("claim_type", "")),
        "polarity": edge.get("polarity"),
        "negation": edge.get("negation"),
        "speculative": edge.get("speculative"),
        "entities": [
            {
                "entity_id": e.get("entity_id"),
                "entity_type": e.get("entity_type"),
                "mention": e.get("mention"),
                "role": e.get("role"),
            }
            for e in edge.get("entities") or []
        ],
        "qualifiers": edge.get("qualifiers") or {},
        "source_article_ids": (edge.get("source") or {}).get("article_ids") or [],
    }


def _slim_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": chunk.get("id"),
        "text": (chunk.get("text") or "")[:600],
    }


class AnswerGeneratorVerifier:
    def __init__(self, ctx: HyperRAGContext):
        self.ctx = ctx
        self.cfg = ctx.config

    def run(
        self,
        case_state: Dict[str, Any],
        evidence_buckets: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        retrieval_plan: Dict[str, Any],
        allowed_hyperedge_ids: Set[str],
        user_question: str,
        expanded_hyperedges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        asked_about_med = _user_asked_about_medication(user_question)

        answer = None
        if self.ctx.llm is not None:
            answer = self._llm_generate(
                case_state=case_state,
                evidence_buckets=evidence_buckets,
                chunks=chunks,
                retrieval_plan=retrieval_plan,
                asked_about_medication=asked_about_med,
            )
        if answer is None:
            answer = self._stub_generate(
                case_state=case_state,
                evidence_buckets=evidence_buckets,
                chunks=chunks,
                retrieval_plan=retrieval_plan,
                asked_about_medication=asked_about_med,
            )

        verified = self._verify(
            answer=answer,
            case_state=case_state,
            evidence_buckets=evidence_buckets,
            retrieval_plan=retrieval_plan,
            allowed_hyperedge_ids=allowed_hyperedge_ids,
            asked_about_medication=asked_about_med,
            expanded_hyperedges=expanded_hyperedges,
        )
        return verified

    # ---------------- LLM generation ----------------

    def _llm_generate(
        self,
        case_state: Dict[str, Any],
        evidence_buckets: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        retrieval_plan: Dict[str, Any],
        asked_about_medication: bool,
    ) -> Optional[Dict[str, Any]]:
        section = self.ctx.prompts.get("answer_generator") or {}
        system = section.get("system", "")
        user_template = section.get("user", "")
        if not system or not user_template:
            return None

        max_n = self.cfg.max_hyperedges_per_bucket_in_prompt
        max_chunks = self.cfg.max_chunks_in_prompt

        def _bucket_payload(name: str) -> str:
            edges = (evidence_buckets.get(name) or [])[:max_n]
            return json.dumps([_slim_edge(e) for e in edges], ensure_ascii=False, indent=2)

        chunks_payload = json.dumps(
            [_slim_chunk(c) for c in (chunks or [])[:max_chunks]],
            ensure_ascii=False,
            indent=2,
        )
        case_state_payload = json.dumps(case_state, ensure_ascii=False, indent=2)

        user_prompt = user_template.format(
            case_state=case_state_payload,
            retrieval_task=retrieval_plan.get("retrieval_task") or "intervention_planning",
            user_strategy_x=case_state.get("user_strategy_x") or "",
            asked_about_medication="yes" if asked_about_medication else "no",
            assessment_evidence=_bucket_payload("assessment_evidence"),
            intervention_evidence=_bucket_payload("intervention_evidence"),
            side_effect_evidence=_bucket_payload("side_effect_evidence"),
            counterfactual_evidence=_bucket_payload("counterfactual_evidence"),
            chunks=chunks_payload,
        )

        try:
            result = self.ctx.llm.chat(system_prompt=system, user_prompt=user_prompt, want_json=True)
        except Exception as err:  # noqa: BLE001
            logger.warning("AnswerGenerator LLM call failed (%s) — using stub.", err)
            return None
        parsed = result.get("json")
        if not isinstance(parsed, dict):
            return None
        return parsed

    # ---------------- Stub generation ----------------

    def _stub_generate(
        self,
        case_state: Dict[str, Any],
        evidence_buckets: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        retrieval_plan: Dict[str, Any],
        asked_about_medication: bool,
    ) -> Dict[str, Any]:
        intervention_edges = evidence_buckets.get("intervention_evidence") or []
        intervention_edge_ids = {
            e.get("evidence_hyperedge_id") for e in intervention_edges if e.get("evidence_hyperedge_id")
        }

        recommended: List[Dict[str, Any]] = []
        used_ids: List[str] = []
        for strat in evidence_buckets.get("candidate_strategies") or []:
            if not strat.get("has_intervention_support"):
                continue
            if strat["entity_type"] == "Medication" and not asked_about_medication:
                continue
            if strat["entity_type"] == "Medication":
                pharm_supports = [
                    eid for eid in strat.get("supporting_intervention_edge_ids") or []
                    if any(
                        canonical_claim_type(e.get("claim_type", "")) in _PHARM_TYPES
                        for e in intervention_edges
                        if e.get("evidence_hyperedge_id") == eid
                    )
                ]
                if not pharm_supports:
                    continue
            support_ids = list(strat.get("supporting_intervention_edge_ids") or [])
            recommended.append(
                {
                    "strategy": strat.get("mention") or strat.get("entity_id"),
                    "supporting_hyperedge_ids": support_ids,
                    "notes": f"entity_type={strat['entity_type']}",
                }
            )
            used_ids.extend(support_ids)

        side_effects: List[str] = []
        for edge in (evidence_buckets.get("side_effect_evidence") or [])[:3]:
            side_effects.append(
                f"{canonical_claim_type(edge.get('claim_type', ''))}: {edge.get('claim_text')}"
            )

        target_section = self._format_target_section(case_state)

        possible_function = case_state.get("possible_function") or ""
        if possible_function and "confirmed" in possible_function.lower():
            possible_function = possible_function.replace("confirmed", "possible")

        patient_summary = self._format_patient_summary(case_state)

        final_text_lines = [
            "Patient summary: " + patient_summary,
            "Target challenging behavior: " + target_section,
            "Possible function or context: " + (possible_function or "no clear hypothesis from profile"),
            "Recommended strategies: "
            + (
                "; ".join(s["strategy"] for s in recommended)
                if recommended
                else "insufficient evidence — no intervention_evidence supports a specific strategy"
            ),
            "Side effects / limitations / missing evidence: "
            + ("; ".join(side_effects) if side_effects else ""),
        ]

        structured = {
            "patient_summary": patient_summary,
            "target_challenging_behavior": target_section,
            "possible_function_or_context": possible_function,
            "recommended_strategies": recommended,
            "side_effects_limitations_missing_evidence": side_effects,
        }

        out: Dict[str, Any] = {
            "final_answer": "\n".join(final_text_lines),
            "structured_answer": structured,
            "used_hyperedge_ids": list(dict.fromkeys(used_ids)),
        }
        if retrieval_plan.get("retrieval_task") == "counterfactual_strategy_check":
            out["counterfactual_verdict"] = None  # let verifier derive
        return out

    def _format_target_section(self, case_state: Dict[str, Any]) -> str:
        if case_state.get("ambiguity"):
            cands = [
                (c.get("behavior") or "").strip()
                for c in case_state.get("candidate_challenging_behaviors") or []
            ]
            cands = [c for c in cands if c]
            return f"ambiguous — candidates: {', '.join(cands)}"
        target = (case_state.get("target_behavior") or "").strip()
        return target or "(none specified)"

    def _format_patient_summary(self, case_state: Dict[str, Any]) -> str:
        pf = case_state.get("patient_features") or {}
        # Render the full 18-field taxonomy; only show populated fields.
        ordered_fields = [
            ("disorder", "disorder"),
            ("clinical_severity", "severity"),
            ("developmental_stage", "stage"),
            ("functional_ability", "functional"),
            ("symptom_behavior", "behaviors"),
            ("symptom_emotion", "emotion"),
            ("trauma", "trauma"),
            ("medication", "medication"),
            ("therapeutic_approach", "therapy"),
            ("educational_approach", "education"),
            ("assessment_tool", "assessment"),
            ("diagnostic_framework", "framework"),
            ("person_role", "people"),
            ("social_environment", "setting"),
            ("gene", "gene"),
            ("neurobiology", "neuro"),
            ("intersectional_identity", "identity"),
            ("healthcare_access", "access"),
        ]
        parts: List[str] = []
        for key, label in ordered_fields:
            value = pf.get(key)
            if isinstance(value, list) and value:
                parts.append(f"{label}: " + ", ".join(value))
            elif isinstance(value, str) and value:
                parts.append(f"{label}: {value}")
        return "; ".join(parts) or "(profile not characterised)"

    # ---------------- Verifier ----------------

    def _verify(
        self,
        answer: Dict[str, Any],
        case_state: Dict[str, Any],
        evidence_buckets: Dict[str, Any],
        retrieval_plan: Dict[str, Any],
        allowed_hyperedge_ids: Set[str],
        asked_about_medication: bool,
        expanded_hyperedges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        intervention_edges = evidence_buckets.get("intervention_evidence") or []
        intervention_edge_ids = {
            e.get("evidence_hyperedge_id") for e in intervention_edges if e.get("evidence_hyperedge_id")
        }
        side_effect_edges = evidence_buckets.get("side_effect_evidence") or []

        structured = answer.get("structured_answer") or {}
        if not isinstance(structured, dict):
            structured = {}

        # 1. Ambiguity-honesty rewrite
        target_text = str(structured.get("target_challenging_behavior") or "").strip()
        if case_state.get("ambiguity"):
            cands = [
                (c.get("behavior") or "").strip()
                for c in case_state.get("candidate_challenging_behaviors") or []
            ]
            cands = [c for c in cands if c]
            if "ambiguous" not in target_text.lower() or not all(c.lower() in target_text.lower() for c in cands):
                structured["target_challenging_behavior"] = (
                    f"ambiguous — candidates: {', '.join(cands)}"
                )
        else:
            if not target_text:
                structured["target_challenging_behavior"] = (
                    case_state.get("target_behavior") or "(not specified)"
                )

        # 2 & 3. Strategy support + medication gate
        recs = structured.get("recommended_strategies") or []
        kept_recs: List[Dict[str, Any]] = []
        moved_to_limitations: List[str] = []
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            strategy_name = str(rec.get("strategy") or "").strip()
            support_ids = [s for s in (rec.get("supporting_hyperedge_ids") or []) if s in intervention_edge_ids]
            if not support_ids:
                if strategy_name:
                    moved_to_limitations.append(
                        f"{strategy_name}: no supporting intervention hyperedge"
                    )
                continue
            if self._is_medication_strategy(strategy_name, evidence_buckets):
                if not asked_about_medication:
                    moved_to_limitations.append(
                        f"{strategy_name}: medication not recommended (user did not ask about medication)"
                    )
                    continue
                pharm_ok = any(
                    canonical_claim_type(
                        next(
                            (e.get("claim_type", "") for e in intervention_edges
                             if e.get("evidence_hyperedge_id") == sid),
                            "",
                        )
                    )
                    in _PHARM_TYPES
                    for sid in support_ids
                )
                if not pharm_ok:
                    moved_to_limitations.append(
                        f"{strategy_name}: medication lacks PHARMACOLOGICAL_INTERVENTION/INTERVENTION_OUTCOME support"
                    )
                    continue
            rec["supporting_hyperedge_ids"] = support_ids
            kept_recs.append(rec)
        structured["recommended_strategies"] = kept_recs

        # 4. Side-effects section
        side_section = structured.get("side_effects_limitations_missing_evidence") or []
        if not isinstance(side_section, list):
            side_section = [str(side_section)] if side_section else []
        side_section.extend(moved_to_limitations)
        if not side_effect_edges and not any(
            "no side-effect" in s.lower() or "no adverse-effect" in s.lower()
            for s in side_section
        ):
            side_section.append("no side-effect / limitation evidence was retrieved")
        # If a Medication strategy IS being recommended but no side-effect evidence exists, flag it.
        med_recommended = any(
            self._is_medication_strategy(str(r.get("strategy") or ""), evidence_buckets)
            for r in kept_recs
        )
        if med_recommended and not side_effect_edges and not any(
            "adverse-effect" in s.lower() for s in side_section
        ):
            side_section.append("no adverse-effect evidence retrieved for the recommended medication")
        structured["side_effects_limitations_missing_evidence"] = side_section

        # 5. Counterfactual verdict
        verdict = answer.get("counterfactual_verdict")
        if retrieval_plan.get("retrieval_task") == "counterfactual_strategy_check":
            user_strategy_x = (case_state.get("user_strategy_x") or "").strip()
            verdict = self._derive_counterfactual_verdict(
                verdict=verdict,
                user_strategy_x=user_strategy_x,
                expanded_hyperedges=expanded_hyperedges,
            )
            answer["counterfactual_verdict"] = verdict
        else:
            answer.pop("counterfactual_verdict", None)

        # 6. used_hyperedge_ids filtering
        used = answer.get("used_hyperedge_ids") or []
        if not isinstance(used, list):
            used = []
        # also include IDs cited in recommended_strategies (in case the LLM forgot)
        for rec in kept_recs:
            for sid in rec.get("supporting_hyperedge_ids") or []:
                if sid not in used:
                    used.append(sid)
        used_filtered = [u for u in used if isinstance(u, str) and u in allowed_hyperedge_ids]
        answer["used_hyperedge_ids"] = list(dict.fromkeys(used_filtered))

        # 7. Forbidden phrase
        possible_function = str(structured.get("possible_function_or_context") or "")
        possible_function = re.sub(r"confirmed function", "possible function", possible_function, flags=re.I)
        structured["possible_function_or_context"] = possible_function

        # Patient summary fallback
        if not structured.get("patient_summary"):
            structured["patient_summary"] = self._format_patient_summary(case_state)

        # final_answer text guard
        final_answer = str(answer.get("final_answer") or "")
        final_answer = re.sub(r"confirmed function", "possible function", final_answer, flags=re.I)
        if not final_answer:
            final_answer = self._compose_final_answer(structured)
        else:
            # Make sure ambiguity wording shows up in the final_answer text too.
            if case_state.get("ambiguity") and "ambiguous" not in final_answer.lower():
                final_answer += "\n\nNote: target challenging behavior is ambiguous — multiple candidates remain."
            # If side_effect_edges is empty, ensure phrase present.
            if not side_effect_edges and "no side-effect" not in final_answer.lower():
                final_answer += "\nNo side-effect / limitation evidence was retrieved."
            if not kept_recs and "insufficient evidence" not in final_answer.lower():
                final_answer += "\nInsufficient evidence to recommend a specific intervention strategy."
        answer["final_answer"] = final_answer
        answer["structured_answer"] = structured
        return answer

    def _is_medication_strategy(
        self, strategy_name: str, evidence_buckets: Dict[str, Any]
    ) -> bool:
        if not strategy_name:
            return False
        for strat in evidence_buckets.get("candidate_strategies") or []:
            mention = (strat.get("mention") or "").strip()
            if mention and mention.lower() in strategy_name.lower():
                return strat.get("entity_type") == "Medication"
        # fallback: explicit medication-ish names
        return bool(_MEDICATION_RE.search(strategy_name))

    def _derive_counterfactual_verdict(
        self,
        verdict: Optional[str],
        user_strategy_x: str,
        expanded_hyperedges: List[Dict[str, Any]],
    ) -> str:
        if isinstance(verdict, str) and verdict.lower().strip() in _VERDICT_LABELS:
            return verdict.lower().strip()

        needle = (user_strategy_x or "").lower().strip()
        if not needle:
            return "insufficient_evidence"

        related = []
        for edge in expanded_hyperedges:
            for ent in edge.get("entities") or []:
                mention = (ent.get("mention") or "").lower()
                if mention and needle in mention:
                    related.append(edge)
                    break
        if not related:
            return "insufficient_evidence"

        # supported: any intervention edge with positive, non-negated, non-speculative
        for edge in related:
            ct = canonical_claim_type(edge.get("claim_type", ""))
            if (
                ct in _INTERVENTION_TYPES
                and edge.get("polarity") == "positive"
                and not edge.get("negation")
                and not edge.get("speculative")
            ):
                return "supported"
        # partially_supported: intervention exists but hedged/non-positive
        for edge in related:
            ct = canonical_claim_type(edge.get("claim_type", ""))
            if ct in _INTERVENTION_TYPES:
                return "partially_supported"
        # not_supported_as_primary: side_effect or negative/negation evidence
        for edge in related:
            ct = canonical_claim_type(edge.get("claim_type", ""))
            if ct in _SIDE_EFFECT_TYPES:
                return "not_supported_as_primary"
            if edge.get("polarity") == "negative" or edge.get("negation"):
                return "not_supported_as_primary"
        return "insufficient_evidence"

    def _compose_final_answer(self, structured: Dict[str, Any]) -> str:
        recs = structured.get("recommended_strategies") or []
        rec_text = (
            "; ".join(r.get("strategy", "") for r in recs if r.get("strategy"))
            if recs
            else "insufficient evidence — no intervention_evidence supports a specific strategy"
        )
        side = structured.get("side_effects_limitations_missing_evidence") or []
        side_text = "; ".join(side) if side else ""
        return (
            f"Patient summary: {structured.get('patient_summary','')}\n"
            f"Target challenging behavior: {structured.get('target_challenging_behavior','')}\n"
            f"Possible function or context: {structured.get('possible_function_or_context','')}\n"
            f"Recommended strategies: {rec_text}\n"
            f"Side effects / limitations / missing evidence: {side_text}"
        )
