"""
TripleMerger – remaps all triples to canonical entity names,
deduplicates identical (head, relation, tail) triples across articles,
and aggregates confidence / provenance metadata.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MergedTriple:
    """A triple after cross-article merging with provenance."""
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    relevance: float = 0.0
    clarity: float = 0.0
    source_articles: List[str] = field(default_factory=list)
    evidence_count: int = 1
    conflict_status: str = "consistent"  # consistent | resolved_kept | resolved_rejected | unresolved

    def integration_score(self) -> float:
        """Weighted integration score (same formula as KARMA EvaluatorAgent)."""
        return 0.5 * self.confidence + 0.25 * self.clarity + 0.25 * self.relevance

    def to_dict(self) -> Dict:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "confidence": round(self.confidence, 4),
            "relevance": round(self.relevance, 4),
            "clarity": round(self.clarity, 4),
            "source_articles": self.source_articles,
            "evidence_count": self.evidence_count,
            "conflict_status": self.conflict_status,
        }


# ── Relation normalization ─────────────────────────────────────────────
# Maps variant relation strings to a canonical form.  This is applied
# BEFORE deduplication so that e.g. "reduces" and "Reduces" and
# "reduce" all hash to the same key.

RELATION_SYNONYMS: Dict[str, str] = {
    # causal
    "causes": "causal_of",
    "cause": "causal_of",
    "caused_by": "causal_of",
    "causal_of": "causal_of",
    "leads_to": "causal_of",
    "results_in": "causal_of",
    "triggers": "causal_of",
    # risk
    "risk_factor_for": "risk_factor_for",
    "predisposes_to": "risk_factor_for",
    "increases_risk_of": "risk_factor_for",
    # association
    "associated_with": "associated_with",
    "correlated_with": "associated_with",
    "linked_to": "associated_with",
    "related_to": "associated_with",
    "co_occurs_with": "associated_with",
    # exacerbation
    "exacerbates": "exacerbates",
    "worsens": "exacerbates",
    "aggravates": "exacerbates",
    # reduction
    "reduces": "reduces",
    "decreases": "reduces",
    "lowers": "reduces",
    "diminishes": "reduces",
    "alleviates": "reduces",
    # increase
    "increases": "increases",
    "elevates": "increases",
    "heightens": "increases",
    "raises": "increases",
    # improvement
    "improves_function": "improves_function",
    "improves": "improves_function",
    "enhances": "improves_function",
    "facilitates": "improves_function",
    # prevention
    "prevents": "prevents",
    "protects_against": "prevents",
    # adverse
    "adverse_effect": "adverse_effect",
    "side_effect": "adverse_effect",
    "causes_harm": "adverse_effect",
    # diagnosis
    "has_diagnosis": "has_diagnosis",
    "diagnosed_with": "has_diagnosis",
    # symptom
    "exhibits_symptom": "exhibits_symptom",
    "presents_with": "exhibits_symptom",
    "displays": "exhibits_symptom",
    # severity
    "has_severity_level": "has_severity_level",
    # functional profile
    "has_functional_profile": "has_functional_profile",
    # genetic
    "has_genetic_marker": "has_genetic_marker",
    # neurobiological
    "has_neurobiological_feature": "has_neurobiological_feature",
    # developmental
    "at_developmental_stage": "at_developmental_stage",
    # treatment
    "treats": "reduces",
    "treated_with": "reduces",
    "effective_for": "reduces",
    # inhibition
    "inhibits": "reduces",
    "blocks": "reduces",
}


def normalize_relation(relation: str) -> str:
    """Normalize a relation label to its canonical form."""
    key = relation.lower().strip().replace(" ", "_")
    return RELATION_SYNONYMS.get(key, key)


class TripleMerger:
    """
    Remaps triples to canonical entity names and deduplicates.

    Parameters
    ----------
    name_mapping : dict[str, str]
        Mapping raw entity name → canonical name (from EntityResolver).
    integration_threshold : float
        Minimum integration score to keep a triple (default 0.6).
    multi_source_bonus : float
        Confidence bonus per additional source article (default 0.03, capped).
    max_bonus : float
        Maximum cumulative multi-source bonus (default 0.10).
    """

    def __init__(
        self,
        name_mapping: Dict[str, str],
        integration_threshold: float = 0.6,
        multi_source_bonus: float = 0.03,
        max_bonus: float = 0.10,
    ):
        self.name_mapping = name_mapping
        self.integration_threshold = integration_threshold
        self.multi_source_bonus = multi_source_bonus
        self.max_bonus = max_bonus

    def merge(self, source_triples: list) -> List[MergedTriple]:
        """
        Remap, deduplicate, and aggregate all source triples.

        Parameters
        ----------
        source_triples : list[SourceTriple]
            All triples from all articles (via ``loader.collect_all_triples``).

        Returns
        -------
        list[MergedTriple]
            Deduplicated, provenance-tagged triples sorted by confidence desc.
        """
        # ── Step 1: remap & normalize ───────────────────────────────────
        groups: Dict[Tuple[str, str, str], List] = defaultdict(list)

        for st in source_triples:
            head = self.name_mapping.get(st.head, st.head)
            tail = self.name_mapping.get(st.tail, st.tail)
            relation = normalize_relation(st.relation)

            # Skip self-loops
            if head.lower().strip() == tail.lower().strip():
                continue

            key = (head, relation, tail)
            groups[key].append(st)

        logger.info(
            "Triple merger: %d raw triples → %d unique (head, relation, tail) groups",
            len(source_triples),
            len(groups),
        )

        # ── Step 2: aggregate within each group ─────────────────────────
        merged: List[MergedTriple] = []
        for (head, relation, tail), members in groups.items():
            sources: Set[str] = set()
            conf_sum = 0.0
            rel_sum = 0.0
            clar_sum = 0.0
            for m in members:
                sources.add(m.source_article)
                conf_sum += m.confidence
                rel_sum += m.relevance
                clar_sum += m.clarity

            n = len(members)
            avg_conf = conf_sum / n
            avg_rel = rel_sum / n
            avg_clar = clar_sum / n

            # Multi-source bonus
            bonus = min(
                (len(sources) - 1) * self.multi_source_bonus,
                self.max_bonus,
            )
            boosted_conf = min(avg_conf + bonus, 1.0)

            mt = MergedTriple(
                head=head,
                relation=relation,
                tail=tail,
                confidence=boosted_conf,
                relevance=avg_rel,
                clarity=avg_clar,
                source_articles=sorted(sources),
                evidence_count=len(sources),
            )
            merged.append(mt)

        # ── Step 3: quality filter ──────────────────────────────────────
        before = len(merged)
        merged = [
            m for m in merged if m.integration_score() >= self.integration_threshold
        ]
        logger.info(
            "Quality filter: %d → %d triples (threshold=%.2f)",
            before,
            len(merged),
            self.integration_threshold,
        )

        # Sort by confidence descending
        merged.sort(key=lambda m: m.confidence, reverse=True)
        return merged
