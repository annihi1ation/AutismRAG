"""
ConflictResolver – LLM-based cross-article contradiction detection and
arbitration using Gemini Flash.

Detects contradictory triples that share the same (head, tail) but have
opposing relations, then calls the LLM with evidence (summaries from
source articles) to decide which triple to keep.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

from .loader import load_summaries
from .triple_merger import MergedTriple

logger = logging.getLogger(__name__)

# ── Contradiction pairs (relation1, relation2) ─────────────────────────
# If two merged triples share the SAME (head, tail) but have relations
# that form one of these pairs, they are considered conflicting.

CONTRADICTION_PAIRS: Set[Tuple[str, str]] = {
    ("reduces", "increases"),
    ("reduces", "exacerbates"),
    ("improves_function", "adverse_effect"),
    ("prevents", "causal_of"),
    ("prevents", "risk_factor_for"),
    ("causal_of", "reduces"),
    ("associated_with", "prevents"),
    ("increases", "prevents"),
}


CONFLICT_RESOLUTION_SYSTEM_PROMPT = """You are a specialized Conflict Resolution Agent for an Intellectual Disability (ID) Knowledge Graph.

You are presented with two CONTRADICTORY knowledge triples extracted from different research articles about intellectual disability. Your task is to:

1. Analyze both triples and their supporting evidence from the source articles
2. Determine which triple is more reliable based on:
   - Quality and specificity of evidence (quantitative data > qualitative claims)
   - Confidence scores
   - Number of supporting articles
   - Clinical/scientific plausibility
   - Whether the relationship is direct vs. indirect
3. Make a clear decision: KEEP_FIRST, KEEP_SECOND, or KEEP_BOTH (if they can coexist under different conditions)

DECISION RULES:
- Higher evidence count (more articles supporting) is a strong signal
- Specific statistical evidence (p-values, effect sizes) overrides general claims
- If both are equally supported, KEEP_BOTH with a note
- Explicit negations should be carefully weighed against weak positive assertions

OUTPUT FORMAT (JSON only):
{
  "decision": "KEEP_FIRST" | "KEEP_SECOND" | "KEEP_BOTH",
  "reasoning": "Brief explanation of your decision",
  "confidence": 0.0-1.0
}
"""


class ConflictResolver:
    """
    LLM-based cross-article conflict resolution.

    Parameters
    ----------
    client : OpenAI
        OpenAI-compatible API client.
    model_name : str
        LLM model name (e.g. ``google/gemini-3-flash-preview``).
    output_dir : str
        Path to KARMA output directory (for loading _summaries.json).
    temperature : float
        LLM temperature for conflict resolution calls.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str = "google/gemini-3-flash-preview",
        output_dir: str = "",
        temperature: float = 0.1,
    ):
        self.client = client
        self.model_name = model_name
        self.output_dir = output_dir
        self.temperature = temperature

    def resolve(self, triples: List[MergedTriple]) -> List[MergedTriple]:
        """
        Detect and resolve contradictory triples.

        Parameters
        ----------
        triples : list[MergedTriple]
            Merged triples from TripleMerger.

        Returns
        -------
        list[MergedTriple]
            Triples with ``conflict_status`` updated.
            Rejected triples have status ``resolved_rejected`` but are
            still returned (caller can filter as needed).
        """
        # ── Step 1: index by (head, tail) ───────────────────────────────
        pair_index: Dict[Tuple[str, str], List[int]] = {}
        for idx, t in enumerate(triples):
            key = (t.head.lower(), t.tail.lower())
            pair_index.setdefault(key, []).append(idx)

        # ── Step 2: find contradictions ─────────────────────────────────
        conflicts: List[Tuple[int, int]] = []
        for key, indices in pair_index.items():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    ti, tj = triples[indices[i]], triples[indices[j]]
                    pair = (ti.relation, tj.relation)
                    pair_rev = (tj.relation, ti.relation)
                    if pair in CONTRADICTION_PAIRS or pair_rev in CONTRADICTION_PAIRS:
                        conflicts.append((indices[i], indices[j]))

        if not conflicts:
            logger.info("No contradictions detected among %d triples", len(triples))
            return triples

        logger.info("Detected %d contradiction pairs to resolve", len(conflicts))

        # ── Step 3: resolve each conflict via LLM ───────────────────────
        for idx_a, idx_b in conflicts:
            ta, tb = triples[idx_a], triples[idx_b]

            # Skip if already resolved
            if ta.conflict_status == "resolved_rejected" or tb.conflict_status == "resolved_rejected":
                continue

            decision = self._arbitrate(ta, tb)

            if decision == "KEEP_FIRST":
                ta.conflict_status = "resolved_kept"
                tb.conflict_status = "resolved_rejected"
                logger.info(
                    "Conflict: kept [%s --%s--> %s], rejected [%s --%s--> %s]",
                    ta.head, ta.relation, ta.tail,
                    tb.head, tb.relation, tb.tail,
                )
            elif decision == "KEEP_SECOND":
                ta.conflict_status = "resolved_rejected"
                tb.conflict_status = "resolved_kept"
                logger.info(
                    "Conflict: rejected [%s --%s--> %s], kept [%s --%s--> %s]",
                    ta.head, ta.relation, ta.tail,
                    tb.head, tb.relation, tb.tail,
                )
            else:  # KEEP_BOTH or parse error
                ta.conflict_status = "unresolved"
                tb.conflict_status = "unresolved"
                logger.info(
                    "Conflict unresolved (keeping both): [%s --%s--> %s] vs [%s --%s--> %s]",
                    ta.head, ta.relation, ta.tail,
                    tb.head, tb.relation, tb.tail,
                )

        resolved_count = sum(
            1 for t in triples if t.conflict_status in ("resolved_kept", "resolved_rejected")
        )
        logger.info(
            "Conflict resolution done: %d triples resolved, %d unresolved",
            resolved_count,
            sum(1 for t in triples if t.conflict_status == "unresolved"),
        )
        return triples

    # ── internals ───────────────────────────────────────────────────────
    def _arbitrate(self, ta: MergedTriple, tb: MergedTriple) -> str:
        """Call LLM to decide between two conflicting triples."""
        # Gather evidence summaries from source articles
        evidence_a = self._gather_evidence(ta.source_articles)
        evidence_b = self._gather_evidence(tb.source_articles)

        prompt = (
            "Two contradictory knowledge triples were extracted from different research articles:\n\n"
            f"TRIPLE 1:\n"
            f"  ({ta.head}) --[{ta.relation}]--> ({ta.tail})\n"
            f"  Confidence: {ta.confidence:.3f}\n"
            f"  Source articles ({ta.evidence_count}): {', '.join(ta.source_articles)}\n"
        )
        if evidence_a:
            prompt += f"  Evidence summaries:\n"
            for i, s in enumerate(evidence_a[:3], 1):
                prompt += f"    [{i}] {s[:300]}\n"

        prompt += (
            f"\nTRIPLE 2:\n"
            f"  ({tb.head}) --[{tb.relation}]--> ({tb.tail})\n"
            f"  Confidence: {tb.confidence:.3f}\n"
            f"  Source articles ({tb.evidence_count}): {', '.join(tb.source_articles)}\n"
        )
        if evidence_b:
            prompt += f"  Evidence summaries:\n"
            for i, s in enumerate(evidence_b[:3], 1):
                prompt += f"    [{i}] {s[:300]}\n"

        prompt += (
            "\nBased on the evidence quality, confidence scores, and number of supporting articles, "
            "decide which triple to keep. Return JSON: "
            '{"decision": "KEEP_FIRST"|"KEEP_SECOND"|"KEEP_BOTH", "reasoning": "...", "confidence": 0.0-1.0}'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": CONFLICT_RESOLUTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            content = response.choices[0].message.content.strip()
            return self._parse_decision(content)
        except Exception as e:
            logger.error("LLM conflict resolution failed: %s", e)
            return "KEEP_BOTH"

    def _gather_evidence(self, article_ids: List[str]) -> List[str]:
        """Load summaries from source articles as evidence."""
        summaries: List[str] = []
        if not self.output_dir:
            return summaries
        for aid in article_ids:
            article_summaries = load_summaries(self.output_dir, aid)
            summaries.extend(article_summaries)
        return summaries

    @staticmethod
    def _parse_decision(text: str) -> str:
        """Parse the LLM's decision from JSON output."""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        try:
            if "{" in text and "}" in text:
                json_str = text[text.find("{") : text.rfind("}") + 1]
                data = json.loads(json_str)
                decision = data.get("decision", "KEEP_BOTH").upper()
                if decision in ("KEEP_FIRST", "KEEP_SECOND", "KEEP_BOTH"):
                    return decision
        except json.JSONDecodeError:
            pass

        # Fallback: simple keyword match
        upper = text.upper()
        if "KEEP_FIRST" in upper:
            return "KEEP_FIRST"
        if "KEEP_SECOND" in upper:
            return "KEEP_SECOND"
        return "KEEP_BOTH"
