"""Offline entity candidate retrieval (non-LLM).

Retrieval runs over PTO NODE texts (plan R4), never over ambiguous draft
HyperEdge fields. Each EntityCandidatePack carries ``source_node_id`` and a
``role_hint`` derived from the node type — never from ``entity_type``.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..embeddings.base import EmbeddingModel
from ..embeddings.entity_index import build_entity_index  # re-exported
from ..schemas.entity import (
    NODE_ROLE_HINT,
    EntityCandidate,
    EntityCandidatePack,
    EntityRecord,
)
from ..schemas.pto_graph import ArticlePTOGraphLite

logger = logging.getLogger(__name__)

__all__ = [
    "NodeText",
    "pto_node_texts",
    "load_entity_catalog",
    "build_entity_index",
    "extract_mentions",
    "retrieve_entity_candidates",
]

_SEED_SENTINEL_KEY = "__seed_demo__"
_SEED_ENTITY_ID = "__SEED_DEMO__"
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")
_STOP = {
    "the", "and", "for", "with", "was", "were", "that", "this", "from", "have",
    "has", "had", "are", "not", "but", "all", "who", "any", "may", "can",
    "study", "group", "groups", "patient", "patients", "participant",
    "participants", "result", "results", "outcome", "outcomes",
}


@dataclass
class NodeText:
    source_field: str  # patient_text | treatment_text | outcome_text
    source_node_id: str
    text: str

    @property
    def role_hint(self) -> str:
        return NODE_ROLE_HINT[self.source_field]


def pto_node_texts(graph: ArticlePTOGraphLite) -> List[NodeText]:
    out: List[NodeText] = []
    for n in graph.patient_nodes:
        out.append(NodeText("patient_text", n.id, n.text))
    for n in graph.treatment_nodes:
        out.append(NodeText("treatment_text", n.id, n.text))
    for n in graph.outcome_nodes:
        out.append(NodeText("outcome_text", n.id, n.text))
    return out


# --------------------------------------------------------------------------- #
# Catalog loading                                                             #
# --------------------------------------------------------------------------- #
def _coerce_synonyms(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not value:
        return []
    return [s.strip() for s in re.split(r"[;|]", str(value)) if s.strip()]


def load_entity_catalog(
    path: str | Path,
    allowed_types: Optional[Sequence[str]] = None,
    allow_extension: bool = False,
    for_build: bool = True,
) -> List[EntityRecord]:
    """Load entity records from JSONL or CSV.

    Emits a clear warning when a starter/demo seed catalog is used for a build
    (plan R5). Records with unknown ``entity_type`` are skipped unless
    ``allow_extension``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Entity catalog not found: {path}")

    is_seed = "seed" in path.name.lower() or "demo" in path.name.lower()
    rows: List[dict] = []

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "seed" in line.lower() or "demo" in line.lower():
                    is_seed = True
                continue
            obj = json.loads(line)
            if obj.get(_SEED_SENTINEL_KEY) or obj.get("entity_id") == _SEED_ENTITY_ID:
                is_seed = True
                continue
            rows.append(obj)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            for obj in csv.DictReader(fh):
                if obj.get("entity_id") == _SEED_ENTITY_ID:
                    is_seed = True
                    continue
                rows.append(obj)
    else:
        raise ValueError(f"Unsupported catalog format: {path.suffix}")

    if is_seed and for_build:
        logger.warning(
            "Entity catalog %s is a STARTER/DEMO seed catalog. Replace it with "
            "a curated catalog before relying on production builds.",
            path,
        )

    allowed = set(allowed_types) if allowed_types is not None else None
    records: List[EntityRecord] = []
    for obj in rows:
        etype = str(obj.get("entity_type", "")).strip()
        if allowed is not None and not allow_extension and etype not in allowed:
            logger.warning("Skipping entity '%s' with unknown type '%s'",
                           obj.get("entity_id"), etype)
            continue
        try:
            rec = EntityRecord(
                entity_id=str(obj["entity_id"]),
                canonical_name=str(obj["canonical_name"]),
                entity_type=etype,
                synonyms=_coerce_synonyms(obj.get("synonyms")),
                description=(obj.get("description") or None),
            )
        except Exception as err:  # noqa: BLE001
            logger.warning("Skipping invalid entity record %s: %s",
                           obj.get("entity_id"), err)
            continue
        records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# Mention extraction                                                          #
# --------------------------------------------------------------------------- #
def _spacy_noun_chunks(text: str) -> Optional[List[str]]:
    try:
        import spacy  # type: ignore
    except ImportError:
        return None
    try:
        nlp = _spacy_noun_chunks._nlp  # type: ignore[attr-defined]
    except AttributeError:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception:  # noqa: BLE001
            return None
        _spacy_noun_chunks._nlp = nlp  # type: ignore[attr-defined]
    doc = nlp(text)
    return [c.text.strip() for c in doc.noun_chunks if c.text.strip()]


def extract_mentions(text: str, max_window: int = 4, max_mentions: int = 24) -> List[str]:
    """Lightweight non-LLM mention candidates."""
    text = (text or "").strip()
    if not text:
        return []
    chunks = _spacy_noun_chunks(text)
    mentions: List[str] = []
    if chunks:
        mentions.extend(chunks)
    tokens = _WORD_RE.findall(text)
    content = [t for t in tokens if t.lower() not in _STOP and len(t) > 2]
    for size in range(1, max_window + 1):
        for i in range(0, max(0, len(content) - size + 1)):
            mentions.append(" ".join(content[i : i + size]))
    if text not in mentions:
        mentions.append(text)
    # De-dup, keep order, cap.
    seen: set[str] = set()
    out: List[str] = []
    for m in mentions:
        key = m.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(m.strip())
        if len(out) >= max_mentions:
            break
    return out


# --------------------------------------------------------------------------- #
# Retrieval                                                                    #
# --------------------------------------------------------------------------- #
def retrieve_entity_candidates(
    node_texts: Sequence[NodeText],
    entity_index,  # EntityVectorIndex
    model: EmbeddingModel,
    records: Sequence[EntityRecord],
    top_k: int = 5,
) -> List[EntityCandidatePack]:
    by_id: Dict[str, EntityRecord] = {r.entity_id: r for r in records}
    packs: List[EntityCandidatePack] = []
    for nt in node_texts:
        for mention in extract_mentions(nt.text):
            qvec = model.embed_text(mention)
            hits = entity_index.search(qvec, top_k)
            candidates: List[EntityCandidate] = []
            for eid, score in hits:
                rec = by_id.get(eid)
                if rec is None:
                    continue
                candidates.append(
                    EntityCandidate(
                        entity_id=rec.entity_id,
                        canonical_name=rec.canonical_name,
                        entity_type=rec.entity_type,
                        similarity=float(score),
                        source="embedding",
                    )
                )
            if not candidates:
                continue
            packs.append(
                EntityCandidatePack(
                    mention=mention,
                    role_hint=nt.role_hint,  # from node type only
                    source_field=nt.source_field,  # type: ignore[arg-type]
                    source_node_id=nt.source_node_id,
                    candidates=candidates,
                )
            )
    return packs
