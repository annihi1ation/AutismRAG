"""Main entry point: autism_hyperkg_rag(patient_profile, user_question)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .answer_generator import AnswerGeneratorVerifier
from .case_parser import CaseParser
from .config import HyperRAGConfig, load_default_config
from .dense_retriever import DenseRetriever
from .diffuser import MinimalHyperKGDiffuser
from .evidence_organizer import EvidenceOrganizer
from .llm_reranker import LLMReranker
from .loaders import HyperRAGContext, build_context
from .query_planner import SimpleQueryPlanner

logger = logging.getLogger(__name__)


def build_default_pipeline(config: Optional[HyperRAGConfig] = None) -> HyperRAGContext:
    return build_context(config or load_default_config())


def _intervention_mentions_for_side_effects(
    case_state: Dict[str, Any],
    buckets: Dict[str, Any],
) -> list[str]:
    """Collect intervention names worth a side-effect lookup, in this order:
      1. candidate_strategies with has_intervention_support == True
      2. case_state["user_strategy_x"]
      3. case_state["patient_features"]["medication"]
    De-dupe case-insensitively, preserve first-seen order.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(m: Any) -> None:
        if not m:
            return
        s = str(m).strip()
        if not s:
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    for strat in buckets.get("candidate_strategies") or []:
        if not strat.get("has_intervention_support"):
            continue
        _add(strat.get("mention") or strat.get("entity_id"))

    _add(case_state.get("user_strategy_x"))

    pf = case_state.get("patient_features") or {}
    for med in pf.get("medication") or []:
        _add(med)

    return out


def _merge_side_effects(
    buckets: Dict[str, Any],
    expanded: Dict[str, Any],
    buckets2: Dict[str, Any],
    expanded2: Dict[str, Any],
) -> None:
    """Merge stage-2 side-effect results back into stage-1 structures.

    Only edges that landed in stage-2's side_effect_evidence bucket are
    eligible; other stage-2 edges are dropped so they never enter
    allowed_hyperedge_ids. Dedupe by evidence_hyperedge_id (stage-1 wins
    on collisions). NEW stage-2 edges get _provenance prefixed with
    "side_effect_pass:". Mutates buckets and expanded in place.
    """
    eligible_ids = {
        e.get("evidence_hyperedge_id")
        for e in buckets2.get("side_effect_evidence") or []
        if e.get("evidence_hyperedge_id")
    }
    if not eligible_ids:
        return

    existing_edge_ids = {
        e.get("evidence_hyperedge_id")
        for e in expanded.get("hyperedges") or []
        if e.get("evidence_hyperedge_id")
    }
    for edge in expanded2.get("hyperedges") or []:
        eid = edge.get("evidence_hyperedge_id")
        if not eid or eid not in eligible_ids or eid in existing_edge_ids:
            continue
        new_edge = dict(edge)
        prov = new_edge.get("_provenance") or "from_retrieval"
        new_edge["_provenance"] = f"side_effect_pass:{prov}"
        expanded.setdefault("hyperedges", []).append(new_edge)
        existing_edge_ids.add(eid)

    expanded_by_id = {
        e.get("evidence_hyperedge_id"): e
        for e in expanded.get("hyperedges") or []
        if e.get("evidence_hyperedge_id")
    }
    existing_se_ids = {
        e.get("evidence_hyperedge_id")
        for e in buckets.get("side_effect_evidence") or []
        if e.get("evidence_hyperedge_id")
    }
    for edge in buckets2.get("side_effect_evidence") or []:
        eid = edge.get("evidence_hyperedge_id")
        if not eid or eid in existing_se_ids:
            continue
        merged_edge = expanded_by_id.get(eid)
        if merged_edge is None:
            merged_edge = dict(edge)
            prov = merged_edge.get("_provenance") or "from_retrieval"
            merged_edge["_provenance"] = f"side_effect_pass:{prov}"
        buckets.setdefault("side_effect_evidence", []).append(merged_edge)
        existing_se_ids.add(eid)


def autism_hyperkg_rag(
    patient_profile: str,
    user_question: str,
    context: Optional[HyperRAGContext] = None,
    return_trace: bool = False,
) -> Dict[str, Any]:
    """Run the minimal Autism HyperKG RAG workflow end-to-end.

    If ``return_trace`` is True the returned dict includes intermediate
    artefacts (case_state, retrieval_plan, retrieved, expanded, buckets) under
    ``_trace`` for debugging — without changing the answer schema.
    """
    ctx = context or build_default_pipeline()

    case_state = CaseParser(ctx).run(patient_profile, user_question)
    planner = SimpleQueryPlanner()
    retrieval_plan = planner.run(
        case_state,
        subject_only_seeding=ctx.config.subject_only_seeding,
    )
    retrieved = DenseRetriever(ctx).run(retrieval_plan)
    if ctx.config.enable_llm_rerank:
        retrieved = LLMReranker(ctx).run(retrieved, retrieval_plan, case_state)
    expanded = MinimalHyperKGDiffuser(ctx).run(retrieved, retrieval_plan)
    buckets = EvidenceOrganizer().run(expanded, retrieval_plan)

    # ---- Stage 2: focused side-effect retrieval ----
    intervention_mentions = _intervention_mentions_for_side_effects(case_state, buckets)
    if intervention_mentions:
        followup_plan = planner.plan_side_effect_followup(case_state, intervention_mentions)
        if (followup_plan.get("retrieval_queries") or {}).get("side_effect_query"):
            retrieved2 = DenseRetriever(ctx).run(followup_plan)
            if ctx.config.enable_llm_rerank:
                retrieved2 = LLMReranker(ctx).run(retrieved2, followup_plan, case_state)
            expanded2 = MinimalHyperKGDiffuser(ctx).run(retrieved2, followup_plan)
            buckets2 = EvidenceOrganizer().run(expanded2, followup_plan)
            _merge_side_effects(buckets, expanded, buckets2, expanded2)
    # ---- end Stage 2 ----

    allowed_hyperedge_ids = {
        edge["evidence_hyperedge_id"]
        for edge in expanded.get("hyperedges") or []
        if edge.get("evidence_hyperedge_id")
    }

    answer = AnswerGeneratorVerifier(ctx).run(
        case_state=case_state,
        evidence_buckets=buckets,
        chunks=retrieved.get("chunks") or [],
        retrieval_plan=retrieval_plan,
        allowed_hyperedge_ids=allowed_hyperedge_ids,
        user_question=user_question,
        expanded_hyperedges=expanded.get("hyperedges") or [],
    )

    if return_trace:
        answer = dict(answer)
        answer["_trace"] = {
            "case_state": case_state,
            "retrieval_plan": retrieval_plan,
            "retrieved_counts": {
                "entities": len(retrieved.get("entities") or []),
                "hyperedges": len(retrieved.get("hyperedges") or []),
                "chunks": len(retrieved.get("chunks") or []),
                "per_query": retrieved.get("per_query_counts") or {},
            },
            "expanded_counts": {
                "hyperedges": len(expanded.get("hyperedges") or []),
                "entities": len(expanded.get("entities") or []),
                "chunks": len(expanded.get("chunks") or []),
            },
            "bucket_counts": {
                k: len(v if isinstance(v, list) else [])
                for k, v in buckets.items()
                if k != "candidate_strategies"
            },
            "candidate_strategies": buckets.get("candidate_strategies") or [],
            "rerank_stats": retrieved.get("rerank_stats") or {},
        }
    return answer
