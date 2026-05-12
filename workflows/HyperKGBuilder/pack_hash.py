"""Canonical hashing for CandidatePack cache + resume keys."""

from __future__ import annotations

from typing import Iterable, List

from workflows.HyperKGConstruction.utils import stable_hash

from .schemas import CandidatePack


def _sorted_ids(values: Iterable[str]) -> List[str]:
    return sorted({str(v) for v in values if v})


def compute_pack_hash(pack: CandidatePack) -> str:
    payload = {
        "pack_id": pack.pack_id,
        "claim_type": pack.claim_type,
        "members": _sorted_ids(pack.member_claim_ids),
        "entities": _sorted_ids(e.entity_id for e in pack.entity_candidates),
        "triples": _sorted_ids(
            f"{t.head}|{t.relation}|{t.tail}" for t in pack.triple_candidates
        ),
        "summaries": _sorted_ids(s.summary_id for s in pack.summary_snippets),
    }
    return stable_hash(payload)
