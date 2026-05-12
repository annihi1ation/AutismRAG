"""Module 5: EvidenceOrganizer — canonicalize claim types and bucket hyperedges."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set


CARE_CONTEXT_ALIASES = {
    "CAREGIVING_CONTEXT": "CARE_GIVING_CONTEXT",
    "CAREGING_CONTEXT": "CARE_GIVING_CONTEXT",
    "CAREgiving_CONTEXT": "CARE_GIVING_CONTEXT",
    "CARING_CONTEXT": "CARE_GIVING_CONTEXT",
    "CARE_GIVING_CONTEXT": "CARE_GIVING_CONTEXT",
}


def canonical_claim_type(claim_type: str) -> str:
    if not claim_type:
        return "OTHER"
    return CARE_CONTEXT_ALIASES.get(claim_type, claim_type)


STRATEGY_ENTITY_TYPES = {"Therapeutic_Approach", "Educational_Approach", "Medication"}


class EvidenceOrganizer:
    """Bucket hyperedges by canonical claim type and identify candidate strategies."""

    def run(
        self,
        expanded_evidence: Dict[str, Any],
        retrieval_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        filters: Dict[str, List[str]] = retrieval_plan.get("claim_type_filters") or {}
        intervention_filter = set(filters.get("intervention_evidence", []))
        assessment_filter = set(filters.get("assessment_evidence", []))
        side_effect_filter = set(filters.get("side_effect_evidence", []))
        counterfactual_filter = set(filters.get("counterfactual_evidence", []))

        buckets: Dict[str, List[Dict[str, Any]]] = {
            "assessment_evidence": [],
            "intervention_evidence": [],
            "side_effect_evidence": [],
            "counterfactual_evidence": [],
            "insufficient_evidence": [],
        }

        edges = expanded_evidence.get("hyperedges") or []

        for edge in edges:
            ct = canonical_claim_type(edge.get("claim_type", ""))
            if ct in assessment_filter:
                buckets["assessment_evidence"].append(edge)
            if ct in intervention_filter:
                buckets["intervention_evidence"].append(edge)
            if ct in side_effect_filter:
                buckets["side_effect_evidence"].append(edge)
            if ct in counterfactual_filter:
                buckets["counterfactual_evidence"].append(edge)
            # Hyperedges with non-canonical claim_types (e.g. "association") are
            # simply not bucketed; they can still be cited via diffusion provenance
            # but do not count toward any of the four evidence categories.

        intervention_edge_ids = {
            e.get("evidence_hyperedge_id") for e in buckets["intervention_evidence"]
        }
        candidate_strategies = self._extract_candidate_strategies(
            edges, intervention_edge_ids=intervention_edge_ids
        )

        # insufficient_evidence carries strategies (entity-level) that have no
        # intervention-bucket support — i.e. ones that cannot be recommended.
        buckets["insufficient_evidence"] = [
            {
                "strategy": strat.get("mention") or strat.get("entity_id"),
                "entity_id": strat.get("entity_id"),
                "entity_type": strat.get("entity_type"),
                "all_edge_ids": strat.get("all_edge_ids") or [],
                "reason": "no intervention_evidence hyperedge supports this strategy",
            }
            for strat in candidate_strategies
            if not strat.get("has_intervention_support")
        ]

        return {
            **buckets,
            "candidate_strategies": candidate_strategies,
        }

    def _extract_candidate_strategies(
        self,
        edges: List[Dict[str, Any]],
        intervention_edge_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        # entity_id -> {entity_type, mention, supporting_intervention_edge_ids, all_edge_ids}
        by_entity: Dict[str, Dict[str, Any]] = {}
        for edge in edges:
            edge_id = edge.get("evidence_hyperedge_id")
            for ent in edge.get("entities") or []:
                etype = ent.get("entity_type")
                if etype not in STRATEGY_ENTITY_TYPES:
                    continue
                ent_id = ent.get("entity_id")
                if not ent_id:
                    continue
                slot = by_entity.setdefault(
                    ent_id,
                    {
                        "entity_id": ent_id,
                        "entity_type": etype,
                        "mention": ent.get("mention") or "",
                        "supporting_intervention_edge_ids": [],
                        "all_edge_ids": [],
                    },
                )
                if edge_id and edge_id not in slot["all_edge_ids"]:
                    slot["all_edge_ids"].append(edge_id)
                if edge_id and edge_id in intervention_edge_ids and edge_id not in slot[
                    "supporting_intervention_edge_ids"
                ]:
                    slot["supporting_intervention_edge_ids"].append(edge_id)

        strategies: List[Dict[str, Any]] = []
        for slot in by_entity.values():
            slot["requires_explicit_user_or_evidence"] = slot["entity_type"] == "Medication"
            slot["has_intervention_support"] = bool(slot["supporting_intervention_edge_ids"])
            strategies.append(slot)
        return strategies
