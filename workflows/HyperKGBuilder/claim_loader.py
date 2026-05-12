"""Claim loader for the HyperKGBuilder workflow.

The input JSONL emitted by the upstream claim-splitter pipeline has rows of
the form::

    {
      "claims": [ { ... claim ... }, ... ],
      "selected_claim_id": "C:...",
      "review_items": [...],
      "object_id": "P:...",
      "index": 0
    }

Each claim has stringified-Python-dict ``candidate_entities`` /
``candidate_triples`` that we parse defensively. ``raw_llm_output`` is a JSON
string with per-claim ``supporting_summary_span`` and ``scope`` blocks; we
join that data in to populate ``summary_snippet`` and ``scope`` for the
selected claim.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .schemas import ClaimRecord

logger = logging.getLogger(__name__)


def _safe_parse_dict(value: Any) -> Dict[str, Any]:
    """Best-effort parse of a stringified dict (single-quoted Python literal)."""
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    # Try strict JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _parse_candidate_entities(values: Any) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    out: List[Dict[str, Any]] = []
    for v in values:
        parsed = _safe_parse_dict(v)
        if parsed:
            out.append(parsed)
    return out


def _parse_candidate_triples(values: Any) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    out: List[Dict[str, Any]] = []
    for v in values:
        if isinstance(v, dict):
            out.append(dict(v))
            continue
        parsed = _safe_parse_dict(v)
        if parsed:
            out.append(parsed)
    return out


def _parse_raw_llm_output(raw: Any) -> Dict[str, Any]:
    if not raw or not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _matching_split_claim(raw_payload: Dict[str, Any], claim_text: str) -> Dict[str, Any]:
    claims = raw_payload.get("claims") or []
    for c in claims:
        if not isinstance(c, dict):
            continue
        if str(c.get("claim_text", "")).strip() == claim_text.strip():
            return c
    return {}


def _select_claim(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    claims = row.get("claims") or []
    if not claims:
        return None
    selected_id = row.get("selected_claim_id")
    if selected_id:
        for c in claims:
            if isinstance(c, dict) and c.get("claim_id") == selected_id:
                return c
    for c in claims:
        if isinstance(c, dict):
            return c
    return None


def _build_record(row: Dict[str, Any]) -> Optional[ClaimRecord]:
    selected = _select_claim(row)
    if not selected:
        return None
    claim_id = str(selected.get("claim_id", "")).strip()
    claim_text = str(selected.get("claim_text", "")).strip()
    if not claim_id or not claim_text:
        return None

    metadata = dict(selected.get("metadata") or {})
    raw_payload = _parse_raw_llm_output(selected.get("raw_llm_output"))
    matching = _matching_split_claim(raw_payload, claim_text)

    polarity = str(matching.get("polarity") or selected.get("polarity") or "")
    modality = str(matching.get("modality") or selected.get("modality") or "")
    scope = matching.get("scope") if isinstance(matching.get("scope"), dict) else {}
    summary_snippet = matching.get("supporting_summary_span") or matching.get("summary_span")
    if summary_snippet is not None:
        summary_snippet = str(summary_snippet).strip() or None

    candidate_entities = _parse_candidate_entities(selected.get("candidate_entities"))
    candidate_triples = _parse_candidate_triples(selected.get("candidate_triples"))

    return ClaimRecord(
        claim_id=claim_id,
        claim_text=claim_text,
        claim_type=str(selected.get("claim_type", "")).strip(),
        polarity=polarity,
        modality=modality,
        candidate_entities=candidate_entities,
        candidate_triples=candidate_triples,
        scope=dict(scope or {}),
        article_id=str(metadata.get("article_id", "")).strip(),
        segment_id=str(
            metadata.get("segment_id") or selected.get("source_segment_id") or ""
        ).strip(),
        summary_id=str(
            metadata.get("summary_id") or selected.get("source_summary_id") or ""
        ).strip(),
        summary_snippet=summary_snippet,
        local_entities=list(candidate_entities),
        local_triples=list(candidate_triples),
        metadata=metadata,
    )


def iter_claim_rows(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                logger.warning("Skipping malformed JSONL row: %s", err)


def load_claims_jsonl(path: str) -> List[ClaimRecord]:
    """Load and normalize claim rows. Empty rows are filtered."""
    records: List[ClaimRecord] = []
    skipped = 0
    for row in iter_claim_rows(path):
        if not row.get("claims"):
            skipped += 1
            continue
        record = _build_record(row)
        if record is None:
            skipped += 1
            continue
        records.append(record)
    logger.info("Loaded %d claim records from %s (skipped %d)", len(records), path, skipped)
    return records


def load_summaries_index(
    claims: Iterable[ClaimRecord],
    summaries_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return ``{summary_id: {summary_text, article_id, segment_id}}``.

    If ``summaries_path`` is provided and exists, it is loaded as either a
    JSON object keyed by ``summary_id`` or a JSON array of summary objects.
    Otherwise we fall back to ``claim.summary_snippet`` recovered from the
    upstream LLM output.
    """
    index: Dict[str, Dict[str, Any]] = {}

    if summaries_path:
        try:
            with open(summaries_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            logger.warning("summaries_path not found: %s — falling back to claim metadata", summaries_path)
            payload = None
        except json.JSONDecodeError as err:
            logger.warning("summaries_path malformed (%s) — falling back to claim metadata", err)
            payload = None
        if isinstance(payload, dict):
            for sid, body in payload.items():
                if not isinstance(body, dict):
                    continue
                index[str(sid)] = {
                    "summary_id": str(sid),
                    "article_id": str(body.get("article_id", "")),
                    "segment_id": str(body.get("segment_id", "")),
                    "summary_text": str(body.get("summary_text") or body.get("text") or ""),
                }
        elif isinstance(payload, list):
            for body in payload:
                if not isinstance(body, dict):
                    continue
                sid = str(body.get("summary_id") or body.get("id") or "")
                if not sid:
                    continue
                index[sid] = {
                    "summary_id": sid,
                    "article_id": str(body.get("article_id", "")),
                    "segment_id": str(body.get("segment_id", "")),
                    "summary_text": str(body.get("summary_text") or body.get("text") or ""),
                }

    for claim in claims:
        sid = claim.summary_id
        if not sid:
            continue
        if sid in index and index[sid].get("summary_text"):
            continue
        index[sid] = {
            "summary_id": sid,
            "article_id": claim.article_id,
            "segment_id": claim.segment_id,
            "summary_text": claim.summary_snippet or "",
        }

    return index
