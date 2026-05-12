"""CLI for ad-hoc smoke testing the Autism HyperKG RAG workflow.

Usage:
    python -m workflows.HyperRAG.run --profile "..." --question "..." [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .pipeline import autism_hyperkg_rag, build_default_pipeline


_DEFAULT_PROFILE = (
    "8-year-old boy with autism, moderate verbal ability, in self-contained classroom; "
    "mother is primary caregiver."
)
_DEFAULT_QUESTION = (
    "He has daily severe self-injurious behavior (head-banging) when transitions are introduced. "
    "What intervention should we try? How about sensory integration therapy?"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autism HyperKG RAG smoke test")
    parser.add_argument("--profile", default=_DEFAULT_PROFILE, help="patient profile")
    parser.add_argument("--question", default=_DEFAULT_QUESTION, help="user question")
    parser.add_argument("--debug", action="store_true", help="emit trace + soft checks")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ctx = build_default_pipeline()
    print(
        f"Context built: hyperedges={len(ctx.hyperedge_store)}, "
        f"entity_index={len(ctx.entity_index.records.get(ctx.config.entity_namespace, []))}, "
        f"hyperedge_index={len(ctx.hyperedge_index.records.get(ctx.config.hyperedge_namespace, []))}, "
        f"chunk_index={len(ctx.chunk_index.records.get(ctx.config.chunk_namespace, []))}, "
        f"llm={'on' if ctx.llm else 'off (stub mode)'}",
        file=sys.stderr,
    )

    answer = autism_hyperkg_rag(
        patient_profile=args.profile,
        user_question=args.question,
        context=ctx,
        return_trace=args.debug,
    )

    if args.debug:
        _soft_checks(answer, args.question)

    print(json.dumps(answer, ensure_ascii=False, indent=2))
    return 0


def _soft_checks(answer: dict, question: str) -> None:
    trace = answer.get("_trace") or {}
    bucket_counts = trace.get("bucket_counts") or {}
    print(
        "[debug] retrieved_counts=" + json.dumps(trace.get("retrieved_counts") or {}),
        file=sys.stderr,
    )
    print(
        "[debug] expanded_counts=" + json.dumps(trace.get("expanded_counts") or {}),
        file=sys.stderr,
    )
    print(
        "[debug] bucket_counts=" + json.dumps(bucket_counts),
        file=sys.stderr,
    )
    rerank_stats = trace.get("rerank_stats") or {}
    print(
        "[debug] rerank_stats=" + json.dumps(rerank_stats),
        file=sys.stderr,
    )
    for bk, stats in rerank_stats.items():
        stats = stats or {}
        if stats.get("after", 0) == 0 and stats.get("before", 0) > 0:
            print(
                f"[warn] rerank emptied bucket {bk} (was {stats.get('before')}, mode={stats.get('mode')})",
                file=sys.stderr,
            )

    structured = answer.get("structured_answer") or {}
    final_text = answer.get("final_answer") or ""

    # 8. Forbidden phrase
    assert "confirmed function" not in final_text.lower(), \
        "FAIL: 'confirmed function' must not appear in final_answer"
    # 7. Ambiguity honesty
    case_state = trace.get("case_state") or {}
    if case_state.get("ambiguity"):
        assert "ambiguous" in str(structured.get("target_challenging_behavior", "")).lower(), \
            "FAIL: ambiguity flag set but answer does not say 'ambiguous'"
    # 4. No-side-effect explicit message
    if bucket_counts.get("side_effect_evidence", 0) == 0:
        side = " ".join(structured.get("side_effects_limitations_missing_evidence") or [])
        assert "no side-effect" in side.lower() or "no adverse-effect" in side.lower(), \
            "FAIL: side-effect bucket empty but answer does not say so"
    # 3. No-intervention explicit message
    if bucket_counts.get("intervention_evidence", 0) == 0:
        assert "insufficient" in final_text.lower(), \
            "FAIL: intervention bucket empty but answer does not say 'insufficient'"
    # 5. Strategy support
    intervention_ids = {
        e.get("evidence_hyperedge_id")
        for e in (trace.get("candidate_strategies") or [])
    }
    for rec in structured.get("recommended_strategies") or []:
        assert rec.get("supporting_hyperedge_ids"), \
            f"FAIL: recommended_strategy {rec.get('strategy')!r} has no supporting hyperedge"
    # 6. ID containment is enforced inside the verifier; we sanity-check non-emptiness only
    used_ids = set(answer.get("used_hyperedge_ids") or [])
    for rec in structured.get("recommended_strategies") or []:
        for sid in rec.get("supporting_hyperedge_ids") or []:
            assert sid in used_ids, \
                f"FAIL: cited supporting id {sid!r} not in used_hyperedge_ids"
    # 9. Counterfactual verdict label
    if (case_state and case_state.get("user_strategy_x")):
        verdict = answer.get("counterfactual_verdict")
        assert verdict in {"supported", "partially_supported", "not_supported_as_primary", "insufficient_evidence"}, \
            f"FAIL: bad counterfactual_verdict {verdict!r}"
    print("[debug] soft checks passed", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
