"""OnlineLLMHyperedgeAgent — the only agent that calls the Online LLM.

Inputs are bounded ``CandidatePack`` records. The full unified KG is never
passed to the LLM. Auto packs produce deterministic evidence hyperedges.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None

from workflows.HyperKGConstruction.llm import ChatLLM, extract_json
from workflows.HyperKGConstruction.prompt_store import get_agent_config
from workflows.HyperKGConstruction.review_queue import ReviewQueue
from workflows.HyperKGConstruction.utils import compact_json, slugify, to_plain

from ..config import BudgetConfig, HyperKGRunConfig
from ..schemas import (
    CandidatePack,
    EntityCandidate,
    EvidenceHyperedge,
    LLMUsageRecord,
    SummarySnippet,
    TripleCandidate,
)
from ..unified_kg_index import UnifiedKGIndex

logger = logging.getLogger(__name__)


_REQUIRED_KEYS = (
    "evidence_hyperedges",
    "canonical_hyperedge",
    "selected_canonical_entities",
    "primary_triples",
    "supporting_triples",
    "entity_roles",
    "qualifiers",
    "polarity",
    "negation",
    "speculative",
    "confidence",
    "warnings",
)


class LLMValidationError(ValueError):
    pass


@dataclass
class OnlineLLMHyperedgeAgent:
    """Routes packs to either deterministic or LLM-driven hyperedge construction."""

    chat_llm: Optional[ChatLLM]
    config: HyperKGRunConfig
    prompt_file: Optional[str] = None
    cache_dir: Optional[str] = None
    review_queue: Optional[ReviewQueue] = None
    completed_pack_hashes: set = field(default_factory=set)
    show_progress: bool = True

    # Token/budget tracking
    online_calls_used: int = 0
    total_tokens_used: int = 0

    # ------------------------------------------------------------------
    def process(
        self,
        packs: List[CandidatePack],
        unified_kg_index: UnifiedKGIndex,
        on_complete: Optional[Callable[[EvidenceHyperedge, LLMUsageRecord], None]] = None,
    ) -> Tuple[List[EvidenceHyperedge], List[LLMUsageRecord]]:
        edges: List[EvidenceHyperedge] = []
        usage: List[LLMUsageRecord] = []
        coord_lock = threading.Lock()

        def emit(edge: EvidenceHyperedge, record: LLMUsageRecord, count_call: bool) -> None:
            with coord_lock:
                edges.append(edge)
                usage.append(record)
                if count_call:
                    self.online_calls_used += 1
                    self.total_tokens_used += record.total_tokens
                if on_complete is not None:
                    on_complete(edge, record)

        budget = self.config.budget
        total_packs = max(1, len(packs))
        ratio_cap = max(1, int(budget.max_online_work_unit_ratio * total_packs))
        cap = min(ratio_cap, budget.max_online_work_units_per_batch)

        # Pass 1: pre-partition packs into auto / cache / budget_exhausted /
        # work_to_run. This lets the LLM step run with thread-level concurrency
        # while keeping budget accounting deterministic.
        work_to_run: List[CandidatePack] = []
        budget_remaining = cap

        for pack in packs:
            if pack.pack_hash in self.completed_pack_hashes:
                continue

            if pack.auto_decision != "online_llm":
                edge = self._auto_hyperedge(pack)
                record = _make_usage(pack, self.config.model_name, "auto", 0, 0, 0.0, "ok", 0)
                emit(edge, record, count_call=False)
                continue

            cached = self._load_cache(pack.pack_hash)
            if cached is not None:
                edge = _hyperedge_from_dict(cached, pack)
                record = _make_usage(pack, self.config.model_name, "cache_hit", 0, 0, 0.0, "ok", 0)
                emit(edge, record, count_call=False)
                continue

            if budget_remaining <= 0:
                if self.review_queue is not None:
                    self.review_queue.add(
                        item_type="pack",
                        object_id=pack.pack_id,
                        reason="budget_exhausted",
                        suggested_action="re-run with a higher budget or process the remainder later",
                        extra={"pack_hash": pack.pack_hash},
                    )
                edge = self._auto_hyperedge(pack, extra_warnings=["budget_exhausted"])
                record = _make_usage(
                    pack, self.config.model_name, "budget_exhausted", 0, 0, 0.0, "ok", 0
                )
                emit(edge, record, count_call=False)
                continue

            work_to_run.append(pack)
            budget_remaining -= 1

        # Pass 2: run the LLM step. OpenRouter calls are IO-bound, so a thread
        # pool gives near-linear speedup. The OpenAI SDK's httpx client is
        # already thread-safe for concurrent requests.
        concurrency = max(1, int(getattr(budget, "online_concurrency", 1)))

        def _run_one(pack: CandidatePack) -> Tuple[EvidenceHyperedge, LLMUsageRecord]:
            try:
                return self._llm_hyperedge(pack, unified_kg_index)
            except Exception as err:  # noqa: BLE001
                logger.exception("Online LLM call failed for pack %s: %s", pack.pack_id, err)
                if self.review_queue is not None:
                    with coord_lock:
                        self.review_queue.add(
                            item_type="pack",
                            object_id=pack.pack_id,
                            reason="llm_call_failed",
                            suggested_action="retry once budget is available or inspect the prompt",
                            extra={"error": str(err)},
                        )
                fallback = self._auto_hyperedge(pack, extra_warnings=["llm_call_failed"])
                fallback_usage = _make_usage(
                    pack,
                    self.config.model_name,
                    ",".join(pack.routing_reasons),
                    0,
                    0,
                    0.0,
                    "failed",
                    1,
                )
                return fallback, fallback_usage

        if not work_to_run:
            return edges, usage

        progress = self._make_progress_bar(len(work_to_run))

        try:
            if concurrency == 1 or len(work_to_run) == 1:
                for pack in work_to_run:
                    edge, record = _run_one(pack)
                    emit(edge, record, count_call=True)
                    self._tick_progress(progress, record)
                return edges, usage

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_run_one, pack): pack for pack in work_to_run}
                for fut in as_completed(futures):
                    edge, record = fut.result()
                    emit(edge, record, count_call=True)
                    self._tick_progress(progress, record)
        finally:
            if progress is not None:
                progress.close()

        return edges, usage

    # ------------------------------------------------------------------
    def _make_progress_bar(self, total: int):
        if not self.show_progress or _tqdm is None or total <= 0:
            return None
        return _tqdm(
            total=total,
            desc="online_llm",
            unit="pack",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
        )

    def _tick_progress(self, progress, record: LLMUsageRecord) -> None:
        if progress is None:
            return
        progress.set_postfix(
            ok=self.online_calls_used,
            tok=self.total_tokens_used,
            last=record.parse_status,
            refresh=False,
        )
        progress.update(1)

    # ------------------------------------------------------------------
    # Deterministic path
    # ------------------------------------------------------------------
    def _auto_hyperedge(
        self,
        pack: CandidatePack,
        extra_warnings: Optional[List[str]] = None,
    ) -> EvidenceHyperedge:
        entities_payload = []
        for e in pack.entity_candidates[: max(2, self.config.routing.max_candidate_entities_per_mention)]:
            entities_payload.append({
                "entity_id": e.entity_id,
                "mention": e.canonical_name,
                "role": "participant",
                "entity_type": e.entity_type,
                "linking_confidence": float(round(e.score, 4)),
            })
        primary = []
        if pack.triple_candidates:
            t = pack.triple_candidates[0]
            primary.append({
                "head": t.head,
                "relation": t.relation,
                "tail": t.tail,
                "confidence": t.confidence,
                "score": t.score,
            })

        confidence = 0.0
        if pack.entity_candidates:
            confidence = float(sum(e.score for e in pack.entity_candidates) / len(pack.entity_candidates))
        warnings = list(extra_warnings or [])
        edge_id = f"EH:{slugify(pack.representative_claim_id)}:{uuid.uuid4().hex[:8]}"
        return EvidenceHyperedge(
            evidence_hyperedge_id=edge_id,
            claim_text=pack.representative_claim_text,
            claim_type=pack.claim_type,
            entities=entities_payload,
            qualifiers={},
            scope={},
            polarity="neutral",
            negation=False,
            speculative=False,
            confidence=round(confidence, 4),
            primary_triples=primary,
            supporting_triples=[
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "confidence": t.confidence,
                    "score": t.score,
                }
                for t in pack.triple_candidates[1:6]
            ],
            warnings=warnings,
            method="auto_embedding_kg_anchor",
            source={
                "pack_id": pack.pack_id,
                "pack_hash": pack.pack_hash,
                "member_claim_ids": list(pack.member_claim_ids),
                "article_ids": sorted({s.article_id for s in pack.summary_snippets if s.article_id}),
                "summary_ids": [s.summary_id for s in pack.summary_snippets],
                "routing_reasons": list(pack.routing_reasons),
            },
        )

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------
    def _llm_hyperedge(
        self,
        pack: CandidatePack,
        unified_kg_index: UnifiedKGIndex,
    ) -> Tuple[EvidenceHyperedge, LLMUsageRecord]:
        if self.chat_llm is None:
            raise RuntimeError("ChatLLM is not configured but pack was routed to online_llm")

        section = self._select_prompt_section(pack)
        agent_cfg = get_agent_config(section, prompt_file=self.prompt_file)
        system_prompt = agent_cfg.get("system_prompt", "")
        template = agent_cfg.get("prompt_template", "")
        if not template:
            raise RuntimeError(f"prompt_template missing for [{section}]")

        prompt_payload = self._build_prompt_payload(pack, unified_kg_index)
        user_prompt = template.format(**prompt_payload)

        # Trim if it would exceed prompt budget.
        user_prompt = self._enforce_prompt_budget(user_prompt, pack, prompt_payload, template)

        max_attempts = self.config.budget.max_retries_per_work_unit + 1
        last_err: Optional[Exception] = None
        attempt = 0
        latency_total = 0.0
        prompt_tokens = completion_tokens = 0
        parsed: Optional[Dict[str, Any]] = None
        raw_text = ""

        while attempt < max_attempts:
            attempt += 1
            stricter_suffix = "" if attempt == 1 else "\n\nReturn STRICT JSON ONLY. No prose, no Markdown."
            full_prompt = user_prompt + stricter_suffix

            t0 = time.time()
            try:
                resp = self._chat(system_prompt, full_prompt)
            except Exception as err:  # noqa: BLE001
                latency_total += (time.time() - t0) * 1000.0
                last_err = err
                continue
            latency_total += (time.time() - t0) * 1000.0

            raw_text = resp.get("text") or ""
            usage_meta = resp.get("usage") or {}
            prompt_tokens += int(usage_meta.get("prompt_tokens") or 0)
            completion_tokens += int(usage_meta.get("completion_tokens") or 0)

            parsed_candidate = resp.get("json") or extract_json(raw_text)
            try:
                _validate_llm_json(parsed_candidate)
                parsed = parsed_candidate
                break
            except LLMValidationError as err:
                last_err = err
                continue

        parse_status = "ok" if attempt == 1 and parsed is not None else (
            "retry_succeeded" if parsed is not None else "failed"
        )

        if parsed is None:
            if self.review_queue is not None:
                self.review_queue.add(
                    item_type="pack",
                    object_id=pack.pack_id,
                    reason="llm_parse_failed",
                    suggested_action="inspect raw output and refine prompt",
                    extra={"error": str(last_err) if last_err else "unknown"},
                )
            edge = self._auto_hyperedge(pack, extra_warnings=["llm_parse_failed"])
            usage = _make_usage(
                pack,
                self.config.model_name,
                ",".join(pack.routing_reasons),
                prompt_tokens,
                completion_tokens,
                latency_total,
                "failed",
                attempt,
            )
            return edge, usage

        edge = _hyperedge_from_llm(pack, parsed, raw_text)
        self._save_cache(pack.pack_hash, edge)
        usage = _make_usage(
            pack,
            self.config.model_name,
            ",".join(pack.routing_reasons),
            prompt_tokens,
            completion_tokens,
            latency_total,
            parse_status,
            attempt,
        )
        self.total_tokens_used += prompt_tokens + completion_tokens
        return edge, usage

    def _chat(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        # Wrap the underlying ChatLLM to also surface token usage from the
        # raw OpenRouter response (the shared ChatLLM only returns text+json).
        if self.chat_llm is None:
            raise RuntimeError("ChatLLM unset")
        client = getattr(self.chat_llm, "client", None)
        if client is None:
            raise RuntimeError("ChatLLM client is None")
        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat.completions.create(
            model=self.chat_llm.model_name,
            messages=messages,
            temperature=self.chat_llm.temperature,
        )
        text = resp.choices[0].message.content or ""
        usage_obj = getattr(resp, "usage", None)
        usage_dict: Dict[str, Any] = {}
        if usage_obj is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
        return {"text": text, "json": extract_json(text), "usage": usage_dict}

    # ------------------------------------------------------------------
    def _select_prompt_section(self, pack: CandidatePack) -> str:
        reasons = set(pack.routing_reasons)
        if reasons & {"complex_scope", "conflict_unresolved", "negation_or_hedging"}:
            return "hyperkg_online_scope_conflict_judge"
        return "hyperkg_online_hyperedge_builder"

    def _build_prompt_payload(
        self,
        pack: CandidatePack,
        unified_kg_index: UnifiedKGIndex,
    ) -> Dict[str, str]:
        member_block = "\n".join(
            f"- {cid}: {pack.representative_claim_text if cid == pack.representative_claim_id else cid}"
            for cid in pack.member_claim_ids
        )
        entities_block = compact_json([_entity_for_prompt(e) for e in pack.entity_candidates])
        triples_block = compact_json([_triple_for_prompt(t) for t in pack.triple_candidates])
        summary_block = compact_json([_summary_for_prompt(s) for s in pack.summary_snippets])
        relation_dist = compact_json(unified_kg_index.relation_distribution(top_n=15))
        type_dist = compact_json(unified_kg_index.entity_type_distribution(top_n=15))
        conflict_summary = compact_json({
            "statuses": sorted({t.conflict_status for t in pack.triple_candidates if t.conflict_status}),
            "max_evidence_count": max((t.evidence_count for t in pack.triple_candidates), default=0),
        })
        return {
            "representative_claim_text": pack.representative_claim_text,
            "member_claims_block": member_block,
            "candidate_entities_block": entities_block,
            "candidate_triples_block": triples_block,
            "summary_snippets_block": summary_block,
            "relation_distribution_summary": relation_dist,
            "entity_type_distribution_summary": type_dist,
            "routing_reasons": ", ".join(pack.routing_reasons),
            "conflict_summary": conflict_summary,
        }

    def _enforce_prompt_budget(
        self,
        rendered: str,
        pack: CandidatePack,
        payload: Dict[str, str],
        template: str,
    ) -> str:
        cap = self.config.budget.max_prompt_chars
        if len(rendered) <= cap:
            return rendered

        # Trim summary snippets first.
        for trim_keys in (
            ("summary_snippets_block",),
            ("candidate_triples_block",),
            ("candidate_entities_block",),
        ):
            payload = dict(payload)
            for key in trim_keys:
                payload[key] = "[]"
            rendered = template.format(**payload)
            if len(rendered) <= cap:
                logger.warning(
                    "Prompt for pack %s trimmed by dropping %s",
                    pack.pack_id,
                    ",".join(trim_keys),
                )
                return rendered
        # If still too long, return a truncated header — caller may still
        # send it, but we add a warning.
        return rendered[:cap]

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_path(self, pack_hash: str) -> Optional[str]:
        if not self.cache_dir or not pack_hash:
            return None
        return os.path.join(self.cache_dir, f"{pack_hash}.json")

    def _load_cache(self, pack_hash: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(pack_hash)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, pack_hash: str, edge: EvidenceHyperedge) -> None:
        path = self._cache_path(pack_hash)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(to_plain(edge), f, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _entity_for_prompt(e: EntityCandidate) -> Dict[str, Any]:
    return {
        "entity_id": e.entity_id,
        "canonical_name": e.canonical_name,
        "entity_type": e.entity_type,
        "aliases": list(e.aliases or [])[:5],
        "score": round(e.score, 4),
        "source": e.source,
    }


def _triple_for_prompt(t: TripleCandidate) -> Dict[str, Any]:
    return {
        "head": t.head,
        "relation": t.relation,
        "tail": t.tail,
        "confidence": t.confidence,
        "relevance": t.relevance,
        "evidence_count": t.evidence_count,
        "conflict_status": t.conflict_status,
        "score": round(t.score, 4),
    }


def _summary_for_prompt(s: SummarySnippet) -> Dict[str, Any]:
    return {
        "summary_id": s.summary_id,
        "article_id": s.article_id,
        "segment_id": s.segment_id,
        "summary_text": s.summary_text[:500],
    }


def _validate_llm_json(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise LLMValidationError("LLM payload is not a JSON object")
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise LLMValidationError(f"LLM payload missing keys: {missing}")
    # Type checks
    if not isinstance(payload.get("evidence_hyperedges"), list):
        raise LLMValidationError("evidence_hyperedges must be a list")
    if not isinstance(payload.get("primary_triples"), list):
        raise LLMValidationError("primary_triples must be a list")
    if not isinstance(payload.get("supporting_triples"), list):
        raise LLMValidationError("supporting_triples must be a list")
    if not isinstance(payload.get("entity_roles"), (list, dict)):
        raise LLMValidationError("entity_roles must be a list or dict")
    if not isinstance(payload.get("qualifiers"), dict):
        raise LLMValidationError("qualifiers must be a dict")
    try:
        float(payload.get("confidence"))
    except (TypeError, ValueError) as err:
        raise LLMValidationError(f"confidence must be numeric: {err}") from err
    if not isinstance(payload.get("warnings"), list):
        raise LLMValidationError("warnings must be a list")


def _hyperedge_from_llm(
    pack: CandidatePack,
    payload: Dict[str, Any],
    raw_text: str,
) -> EvidenceHyperedge:
    canonical = payload.get("canonical_hyperedge") or {}
    selected = payload.get("selected_canonical_entities") or []
    entity_roles = payload.get("entity_roles") or []
    if isinstance(entity_roles, dict):
        entity_roles_iter = [{"entity_id": k, "role": v} for k, v in entity_roles.items()]
    else:
        entity_roles_iter = list(entity_roles)

    entities_payload = []
    by_id = {str(s.get("entity_id")): s for s in selected if isinstance(s, dict)}
    for role_entry in entity_roles_iter:
        if not isinstance(role_entry, dict):
            continue
        eid = str(role_entry.get("entity_id") or role_entry.get("id") or "")
        sel = by_id.get(eid, {})
        entities_payload.append({
            "entity_id": eid or sel.get("entity_id", ""),
            "mention": role_entry.get("mention") or sel.get("canonical_name") or sel.get("mention", ""),
            "role": str(role_entry.get("role") or role_entry.get("entity_role") or "participant"),
            "entity_type": role_entry.get("entity_type") or sel.get("entity_type", ""),
            "linking_confidence": float(role_entry.get("linking_confidence") or sel.get("linking_confidence") or 0.0),
        })

    edge_id = f"EH:{slugify(pack.representative_claim_id)}:{uuid.uuid4().hex[:8]}"
    return EvidenceHyperedge(
        evidence_hyperedge_id=edge_id,
        claim_text=str(canonical.get("canonical_claim") or pack.representative_claim_text),
        claim_type=str(canonical.get("claim_type") or pack.claim_type),
        entities=entities_payload,
        qualifiers=dict(payload.get("qualifiers") or {}),
        scope=dict(payload.get("scope") or {}),
        polarity=str(payload.get("polarity") or "neutral"),
        negation=bool(payload.get("negation") or False),
        speculative=bool(payload.get("speculative") or False),
        confidence=float(payload.get("confidence") or 0.0),
        primary_triples=list(payload.get("primary_triples") or []),
        supporting_triples=list(payload.get("supporting_triples") or []),
        warnings=list(payload.get("warnings") or []),
        method="online_llm",
        source={
            "pack_id": pack.pack_id,
            "pack_hash": pack.pack_hash,
            "member_claim_ids": list(pack.member_claim_ids),
            "article_ids": sorted({s.article_id for s in pack.summary_snippets if s.article_id}),
            "summary_ids": [s.summary_id for s in pack.summary_snippets],
            "routing_reasons": list(pack.routing_reasons),
        },
        raw_llm_output=raw_text,
    )


def _hyperedge_from_dict(payload: Dict[str, Any], pack: CandidatePack) -> EvidenceHyperedge:
    return EvidenceHyperedge(
        evidence_hyperedge_id=str(payload.get("evidence_hyperedge_id") or f"EH:{pack.pack_id}"),
        claim_text=str(payload.get("claim_text") or pack.representative_claim_text),
        claim_type=str(payload.get("claim_type") or pack.claim_type),
        entities=list(payload.get("entities") or []),
        qualifiers=dict(payload.get("qualifiers") or {}),
        scope=dict(payload.get("scope") or {}),
        polarity=str(payload.get("polarity") or "neutral"),
        negation=bool(payload.get("negation") or False),
        speculative=bool(payload.get("speculative") or False),
        confidence=float(payload.get("confidence") or 0.0),
        primary_triples=list(payload.get("primary_triples") or []),
        supporting_triples=list(payload.get("supporting_triples") or []),
        warnings=list(payload.get("warnings") or []),
        method=str(payload.get("method") or "online_llm"),
        source=dict(payload.get("source") or {}),
        embedding_text=payload.get("embedding_text"),
        vector_id=payload.get("vector_id"),
        raw_llm_output=str(payload.get("raw_llm_output") or ""),
    )


def _make_usage(
    pack: CandidatePack,
    model: str,
    routing_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    parse_status: str,
    attempt_count: int,
) -> LLMUsageRecord:
    return LLMUsageRecord(
        work_unit_id=pack.pack_id,
        pack_hash=pack.pack_hash,
        model=model,
        routing_reason=routing_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
        parse_status=parse_status,
        attempt_count=attempt_count,
    )
