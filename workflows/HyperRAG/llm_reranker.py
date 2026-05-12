"""Module 3.5: LLMReranker — bucket hyperedge candidates and LLM-rerank each bucket.

Sits between DenseRetriever and MinimalHyperKGDiffuser. Input/output dicts are
shape-compatible with the DenseRetriever output, so swapping is a one-liner in
pipeline.py.

Failure semantics: any LLM failure, schema violation, or unknown id falls back
to a per-bucket heuristic ranker. The module never raises.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .evidence_organizer import canonical_claim_type
from .loaders import HyperRAGContext

logger = logging.getLogger(__name__)


_MIN_ENTITY_COUNT = 2     # calibrated against p10 of online_results.jsonl
_MIN_CLAIM_CHARS = 60     # calibrated against p10 of online_results.jsonl

_HIGH_VALUE_CLAIM_TYPES = {
    "INTERVENTION_OUTCOME",
    "COMPARATIVE_OUTCOME",
    "BEHAVIORAL_INTERVENTION",
}

_BUCKET_TO_SUBQUERY_KEY = {
    "assessment_evidence": "assessment_query",
    "intervention_evidence": "intervention_query",
    "side_effect_evidence": "side_effect_query",
    "counterfactual_evidence": "counterfactual_query",
}

_BUCKET_SPECIFIC_RULE = {
    "intervention_evidence": (
        "The edge must reference an actual intervention entity "
        "(Therapeutic_Approach / Educational_Approach / Medication). "
        "If it only mentions the target behavior without an intervention, set specificity = 0."
    ),
    "side_effect_evidence": (
        "The edge must reference an adverse effect, risk, or service-access limitation. "
        "General behavioral consequences do not count."
    ),
    "assessment_evidence": (
        "The edge should characterize the patient (symptoms, function, severity, context). "
        "Caregiver-stress edges score subject_match = 0."
    ),
    "counterfactual_evidence": (
        "The edge must inform whether the user-proposed strategy is appropriate, "
        "either supporting or contradicting it."
    ),
}


def _case_summary(case_state: Dict[str, Any]) -> str:
    pf = case_state.get("patient_features") or {}
    target = (case_state.get("target_behavior") or "").strip() or "unspecified"
    dev = (pf.get("developmental_stage") or "").strip() or "unspecified"
    sev = (pf.get("clinical_severity") or "").strip() or "unspecified"
    fa = pf.get("functional_ability") or []
    fa_str = ", ".join(s.strip() for s in fa if s and s.strip()) or "unspecified"
    return (
        f"target_behavior={target}; developmental_stage={dev}; "
        f"clinical_severity={sev}; functional_ability=[{fa_str}]"
    )


def _candidate_line(edge: Dict[str, Any]) -> str:
    eid = edge.get("evidence_hyperedge_id") or ""
    ct = canonical_claim_type(edge.get("claim_type", "") or "")
    ents = edge.get("entities") or []
    ent_pairs = []
    for e in ents:
        mention = (e.get("mention") or "").strip()
        etype = (e.get("entity_type") or "").strip()
        if mention:
            ent_pairs.append(f"{mention}:{etype}")
    ent_str = ", ".join(ent_pairs[:8])
    claim = (edge.get("claim_text") or "").strip().replace("\n", " ")
    if len(claim) > 200:
        claim = claim[:200] + "..."
    return f'[{eid}] claim_type={ct} entities=[{ent_str}] claim="{claim}"'


def _bucketize(
    hits: List[Dict[str, Any]],
    edge_store: Dict[str, Dict[str, Any]],
    filters: Dict[str, List[str]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Set[str]]:
    """Return (bucket_name -> [hit, ...]) plus the set of hit ids that landed in
    at least one bucket. A hit can land in multiple buckets (same rule as
    EvidenceOrganizer)."""
    filter_sets = {bucket: set(types) for bucket, types in filters.items()}
    buckets: Dict[str, List[Dict[str, Any]]] = {
        bk: [] for bk in _BUCKET_TO_SUBQUERY_KEY
    }
    placed: Set[str] = set()
    for hit in hits:
        edge_id = hit.get("id")
        if not edge_id:
            continue
        edge = edge_store.get(edge_id)
        if edge is None:
            continue
        ct = canonical_claim_type(edge.get("claim_type", "") or "")
        for bucket in _BUCKET_TO_SUBQUERY_KEY:
            if ct in filter_sets.get(bucket, set()):
                buckets[bucket].append(hit)
                placed.add(edge_id)
    return buckets, placed


def _prefilter(
    hits: List[Dict[str, Any]],
    edge_store: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for h in hits:
        edge = edge_store.get(h.get("id"))
        if edge is None:
            dropped += 1
            continue
        n_ent = len(edge.get("entities") or [])
        claim_len = len((edge.get("claim_text") or "").strip())
        if n_ent < _MIN_ENTITY_COUNT or claim_len < _MIN_CLAIM_CHARS:
            dropped += 1
            continue
        kept.append(h)
    return kept, dropped


def _fallback_rank(
    hits: List[Dict[str, Any]],
    edge_store: Dict[str, Dict[str, Any]],
    top_k: int,
) -> List[Tuple[Dict[str, Any], float]]:
    def score(h: Dict[str, Any]) -> Tuple[int, int, float]:
        edge = edge_store.get(h.get("id")) or {}
        n_ent = len(edge.get("entities") or [])
        ct = canonical_claim_type(edge.get("claim_type", "") or "")
        ct_bonus = 1 if ct in _HIGH_VALUE_CLAIM_TYPES else 0
        return (ct_bonus, n_ent, float(h.get("score") or 0.0))

    ranked = sorted(hits, key=score, reverse=True)[:top_k]
    # Map to (hit, synthetic_score) so the merge step has a numeric to compare.
    out: List[Tuple[Dict[str, Any], float]] = []
    for h in ranked:
        out.append((h, float(h.get("score") or 0.0)))
    return out


def _parse_llm_ranked(
    parsed: Any,
    candidate_ids: Set[str],
    top_k: int,
) -> Optional[List[Tuple[str, float]]]:
    """Validate the LLM response. Return [(id, score), ...] or None on any error."""
    if not isinstance(parsed, dict):
        return None
    ranked = parsed.get("ranked")
    if not isinstance(ranked, list):
        return None
    out: List[Tuple[str, float]] = []
    seen: Set[str] = set()
    for item in ranked:
        if not isinstance(item, dict):
            return None
        eid = item.get("id")
        if not isinstance(eid, str) or eid not in candidate_ids or eid in seen:
            return None
        try:
            relevance = int(item.get("relevance"))
            specificity = int(item.get("specificity"))
            subject = int(item.get("subject_match"))
        except (TypeError, ValueError):
            return None
        if subject not in (0, 1):
            return None
        if relevance < 0 or relevance > 3 or specificity < 0 or specificity > 3:
            return None
        # Hard drop rules — defensive; LLM should have dropped these.
        if subject == 0 or (relevance + specificity) < 3:
            continue
        # Score = (subject_match, relevance + specificity); flatten to float.
        score = subject * 10.0 + (relevance + specificity)
        out.append((eid, score))
        seen.add(eid)
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]


class LLMReranker:
    """Bucket retrieved hyperedges and LLM-rerank each bucket independently."""

    def __init__(self, ctx: HyperRAGContext) -> None:
        self.ctx = ctx
        self.cfg = ctx.config

    def run(
        self,
        retrieved: Dict[str, Any],
        retrieval_plan: Dict[str, Any],
        case_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        hits: List[Dict[str, Any]] = list(retrieved.get("hyperedges") or [])
        edge_store = self.ctx.hyperedge_store
        filters: Dict[str, List[str]] = retrieval_plan.get("claim_type_filters") or {}
        sub_queries: Dict[str, Optional[str]] = retrieval_plan.get("retrieval_queries") or {}

        buckets, placed = _bucketize(hits, edge_store, filters)

        case_summary = _case_summary(case_state)
        rerank_stats: Dict[str, Dict[str, Any]] = {}
        # edge_id -> (rerank_score, [bucket, ...])
        winners: Dict[str, Tuple[float, List[str]]] = {}

        for bucket, candidates in buckets.items():
            sub_query_key = _BUCKET_TO_SUBQUERY_KEY[bucket]
            sub_query = sub_queries.get(sub_query_key)
            before = len(candidates)
            if not candidates or not sub_query:
                rerank_stats[bucket] = {
                    "before": before,
                    "after": 0,
                    "dropped_low_info": 0,
                    "mode": "skipped" if not sub_query else "empty",
                }
                continue

            filtered, dropped = _prefilter(candidates, edge_store)
            filtered = filtered[: self.cfg.rerank_input_top_n]
            top_k = self.cfg.rerank_top_k_per_bucket

            mode = "fallback"
            survivors: List[Tuple[str, float]] = []

            if self.ctx.rerank_llm is not None and filtered:
                survivors_or_none = self._call_llm(
                    bucket=bucket,
                    sub_query=sub_query,
                    case_summary=case_summary,
                    candidates=filtered,
                    top_k=top_k,
                )
                if survivors_or_none is not None:
                    survivors = survivors_or_none
                    mode = "llm"

            if mode == "fallback":
                survivors = [
                    (h.get("id"), s) for h, s in _fallback_rank(filtered, edge_store, top_k)
                ]

            for eid, score in survivors:
                if not eid:
                    continue
                prev = winners.get(eid)
                if prev is None or score > prev[0]:
                    bucket_list = list(prev[1]) if prev else []
                    if bucket not in bucket_list:
                        bucket_list.append(bucket)
                    winners[eid] = (score, bucket_list)
                else:
                    if bucket not in prev[1]:
                        prev[1].append(bucket)

            rerank_stats[bucket] = {
                "before": before,
                "after": len(survivors),
                "dropped_low_info": dropped,
                "mode": mode,
            }

        # Build output hyperedges:
        #  - bucketed survivors come first (annotated with rerank metadata)
        #  - hits that never bucketed pass through unchanged (so the diffuser's
        #    seed set and downstream allowed_hyperedge_ids don't lose evidence
        #    that EvidenceOrganizer wouldn't have used anyway).
        out_hits: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        hits_by_id = {h.get("id"): h for h in hits if h.get("id")}

        for eid, (score, bks) in sorted(winners.items(), key=lambda kv: kv[1][0], reverse=True):
            hit = hits_by_id.get(eid)
            if hit is None:
                continue
            new_hit = dict(hit)
            new_hit["_rerank_score"] = score
            new_hit["_rerank_buckets"] = bks
            out_hits.append(new_hit)
            seen_ids.add(eid)

        for hit in hits:
            eid = hit.get("id")
            if not eid or eid in seen_ids:
                continue
            if eid in placed:
                # Was bucketed but didn't survive rerank → drop.
                continue
            out_hits.append(hit)
            seen_ids.add(eid)

        result = dict(retrieved)
        result["hyperedges"] = out_hits
        result["rerank_stats"] = rerank_stats
        return result

    def _call_llm(
        self,
        bucket: str,
        sub_query: str,
        case_summary: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> Optional[List[Tuple[str, float]]]:
        prompts = self.ctx.prompts.get("reranker") or {}
        system = prompts.get("system") or ""
        user_template = prompts.get("user") or ""
        if not system or not user_template:
            logger.warning("Reranker prompts missing — bucket %s falling back.", bucket)
            return None

        candidate_lines = []
        candidate_ids: Set[str] = set()
        for hit in candidates:
            edge = self.ctx.hyperedge_store.get(hit.get("id"))
            if edge is None:
                continue
            candidate_lines.append(_candidate_line(edge))
            eid = edge.get("evidence_hyperedge_id")
            if eid:
                candidate_ids.add(eid)
        if not candidate_lines or not candidate_ids:
            return None

        try:
            user_prompt = user_template.format(
                bucket_name=bucket,
                bucket_specific_rule=_BUCKET_SPECIFIC_RULE.get(bucket, ""),
                sub_query=sub_query,
                case_summary=case_summary,
                top_k=top_k,
                candidates="\n".join(candidate_lines),
            )
        except (KeyError, IndexError) as err:
            logger.warning("Reranker user template format failed (%s) for %s.", err, bucket)
            return None

        try:
            result = self.ctx.rerank_llm.chat(
                system_prompt=system,
                user_prompt=user_prompt,
                want_json=True,
            )
        except Exception as err:  # noqa: BLE001
            logger.warning("Reranker LLM call failed for %s (%s) — fallback.", bucket, err)
            return None

        parsed = result.get("json") if isinstance(result, dict) else None
        survivors = _parse_llm_ranked(parsed, candidate_ids, top_k)
        if survivors is None:
            logger.warning("Reranker LLM returned invalid schema for %s — fallback.", bucket)
            return None
        return survivors
