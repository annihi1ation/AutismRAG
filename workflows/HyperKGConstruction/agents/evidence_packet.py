"""EvidencePacketAgent: align summaries, local KGs, and unified KG entities."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..embedding_service import EmbeddingService, entity_embedding_text
from ..schemas import SegmentEvidencePacket
from ..utils import as_list, normalize_text, slugify, unique_preserve_order
from ..vector_index import VectorIndexAdapter

logger = logging.getLogger(__name__)


def _canonical_entity_id(entity: Dict[str, Any]) -> str:
    existing = entity.get("entity_id") or entity.get("id") or entity.get("canonical_entity_id")
    if existing:
        return str(existing)
    name = entity.get("canonical_name") or entity.get("name") or "unknown"
    entity_type = entity.get("entity_type") or entity.get("type") or "entity"
    return f"E:{slugify(entity_type, 40)}:{slugify(name, 120)}"


def _entity_name(entity: Dict[str, Any]) -> str:
    return str(entity.get("canonical_name") or entity.get("name") or entity.get("mention") or "")


def _local_entity_mention(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("mention")
            or value.get("name")
            or value.get("text")
            or value.get("canonical_name")
            or ""
        )
    return str(value or "")


def _local_entity_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        mention = value.get("mention") or value.get("name") or value.get("text") or value.get("canonical_name")
        out = dict(value)
        out.setdefault("mention", str(mention or ""))
        return out
    return {"mention": str(value), "name": str(value)}


def _summary_dict(value: Any, index: int, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = metadata or {}
    if isinstance(value, dict):
        out = dict(value)
    else:
        out = {"summary_text": str(value)}
    article_id = (
        out.get("article_id")
        or out.get("paper_id")
        or out.get("document_id")
        or metadata.get("article_id")
        or metadata.get("paper_id")
        or "article"
    )
    segment_id = (
        out.get("segment_id")
        or out.get("section_id")
        or out.get("chunk_id")
        or f"seg{index + 1:04d}"
    )
    summary_text = out.get("summary_text") or out.get("summary") or out.get("text") or ""
    out["article_id"] = str(article_id)
    out["segment_id"] = str(segment_id)
    out["summary_id"] = str(out.get("summary_id") or f"SUM:{article_id}:{segment_id}")
    out["summary_text"] = str(summary_text)
    return out


@dataclass
class EvidencePacketAgent:
    embedding_service: Optional[EmbeddingService] = None
    vector_index: Optional[VectorIndexAdapter] = None
    entity_linking_threshold: float = 0.82
    entity_linking_top_k: int = 10
    max_local_entities_per_packet: int = 80
    max_local_triples_per_packet: int = 60
    canonical_entities: List[Dict[str, Any]] = field(default_factory=list)
    canonical_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lookup: Dict[str, str] = field(default_factory=dict)

    def process(
        self,
        summaries: List[Dict[str, Any]] | List[str],
        local_kgs: List[Dict[str, Any]] | Dict[str, Dict[str, Any]],
        unified_kg: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SegmentEvidencePacket]:
        self._prepare_unified_kg(unified_kg)
        kg_index = self._index_local_kgs(local_kgs)

        packets: List[SegmentEvidencePacket] = []
        seen_ids: Dict[str, int] = {}
        for idx, raw_summary in enumerate(summaries):
            summary = _summary_dict(raw_summary, idx, metadata)
            article_id = summary["article_id"]
            segment_id = summary["segment_id"]
            packet_id = f"P:{article_id}:{segment_id}"
            if packet_id in seen_ids:
                seen_ids[packet_id] += 1
                packet_id = f"{packet_id}:dup{seen_ids[packet_id]}"
            else:
                seen_ids[packet_id] = 0

            local_kg = self._find_local_kg(kg_index, article_id, segment_id)
            warnings: List[str] = []
            if local_kg is None:
                local_kg = {}
                warnings.append("local KG missing for summary")
            else:
                local_kg = self._local_kg_for_summary(local_kg, summary["summary_text"])
                if (local_kg.get("metadata") or {}).get("hyperkg_filter_mode") == "article_fallback":
                    warnings.append("local KG filtering found no summary matches; using capped article KG fallback")
            if not summary["summary_text"].strip():
                warnings.append("summary_text is empty")

            local_entities = [_local_entity_dict(e) for e in as_list(local_kg.get("entities"))]
            local_triples = [dict(t) for t in as_list(local_kg.get("triples")) if isinstance(t, dict)]

            # Include triple endpoints as local entities if the old KG did not list them.
            endpoint_entities = []
            for triple in local_triples:
                endpoint_entities.extend([triple.get("head"), triple.get("tail")])
            local_entities = [
                _local_entity_dict(e)
                for e in unique_preserve_order(
                    local_entities + [e for e in endpoint_entities if e]
                )
            ][: self.max_local_entities_per_packet]

            canonical_map, unresolved = self._link_local_entities(local_entities)
            packet_metadata = dict(metadata or {})
            packet_metadata.update(local_kg.get("metadata") or {})
            packet_metadata.update(
                {k: v for k, v in summary.items() if k not in {"summary_text"}}
            )
            packets.append(
                SegmentEvidencePacket(
                    packet_id=packet_id,
                    article_id=article_id,
                    segment_id=segment_id,
                    summary_id=summary["summary_id"],
                    summary_text=summary["summary_text"],
                    local_entities=local_entities,
                    local_triples=local_triples,
                    canonical_entity_map=canonical_map,
                    unresolved_entities=unresolved,
                    metadata=packet_metadata,
                    warnings=warnings,
                )
            )
        return packets

    def _prepare_unified_kg(self, unified_kg: Dict[str, Any]) -> None:
        entities = unified_kg.get("canonical_entities") or unified_kg.get("entities") or []
        self.canonical_entities = []
        self.canonical_by_id = {}
        self.lookup = {}

        for raw in entities:
            if not isinstance(raw, dict):
                raw = {"canonical_name": str(raw), "aliases": []}
            entity = dict(raw)
            entity_id = _canonical_entity_id(entity)
            entity["entity_id"] = entity_id
            self.canonical_entities.append(entity)
            self.canonical_by_id[entity_id] = entity

            keys = [
                entity_id,
                entity.get("normalized_id"),
                entity.get("ontology_id"),
                entity.get("canonical_name"),
                entity.get("name"),
            ]
            keys.extend(entity.get("ontology_ids") or [])
            keys.extend(entity.get("aliases") or [])
            for key in keys:
                key_norm = normalize_text(key)
                if key_norm and key_norm != "n/a":
                    self.lookup.setdefault(key_norm, entity_id)

        self._build_canonical_entity_vectors()

    def _build_canonical_entity_vectors(self) -> None:
        if not (self.embedding_service and self.embedding_service.enabled and self.vector_index):
            return
        if self.vector_index.records.get("canonical_entity"):
            return
        try:
            texts = [entity_embedding_text(e) for e in self.canonical_entities]
            ids = [e["entity_id"] for e in self.canonical_entities]
            vectors = self.embedding_service.encode_texts(texts)
            metadata = [
                {
                    "canonical_name": _entity_name(e),
                    "entity_type": e.get("entity_type") or e.get("type") or "",
                }
                for e in self.canonical_entities
            ]
            self.vector_index.add("canonical_entity", ids, texts, vectors, metadata)
        except Exception as err:  # noqa: BLE001
            logger.warning("Failed to build canonical entity vector index: %s", err)

    def _index_local_kgs(
        self,
        local_kgs: List[Dict[str, Any]] | Dict[str, Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if isinstance(local_kgs, dict):
            items = list(local_kgs.items())
            for key, kg in items:
                if not isinstance(kg, dict):
                    continue
                article_id = str(
                    kg.get("article_id")
                    or (kg.get("metadata") or {}).get("article_id")
                    or key
                )
                segment_id = str(
                    kg.get("segment_id")
                    or (kg.get("metadata") or {}).get("segment_id")
                    or ""
                )
                index[(article_id, segment_id)] = kg
                index[(article_id, "")] = kg
                index[(str(key), "")] = kg
            return index

        for kg in local_kgs or []:
            if not isinstance(kg, dict):
                continue
            article_id = str(kg.get("article_id") or (kg.get("metadata") or {}).get("article_id") or "")
            segment_id = str(kg.get("segment_id") or (kg.get("metadata") or {}).get("segment_id") or "")
            if article_id:
                index[(article_id, segment_id)] = kg
                index[(article_id, "")] = kg
        return index

    def _find_local_kg(
        self,
        kg_index: Dict[Tuple[str, str], Dict[str, Any]],
        article_id: str,
        segment_id: str,
    ) -> Optional[Dict[str, Any]]:
        return (
            kg_index.get((article_id, segment_id))
            or kg_index.get((article_id, ""))
            or kg_index.get((f"{article_id}:{segment_id}", ""))
        )

    def _local_kg_for_summary(
        self,
        local_kg: Dict[str, Any],
        summary_text: str,
    ) -> Dict[str, Any]:
        """Adapt article-level KARMA KGs into compact summary-level packets."""
        entities = as_list(local_kg.get("entities"))
        triples = [dict(t) for t in as_list(local_kg.get("triples")) if isinstance(t, dict)]
        metadata = dict(local_kg.get("metadata") or {})
        original_entity_count = len(entities)
        original_triple_count = len(triples)

        has_segment_id = bool(local_kg.get("segment_id") or metadata.get("segment_id"))
        if has_segment_id:
            metadata.update(
                {
                    "hyperkg_filter_mode": "segment_exact",
                    "hyperkg_original_local_entity_count": original_entity_count,
                    "hyperkg_original_local_triple_count": original_triple_count,
                }
            )
            return {
                **local_kg,
                "entities": entities[: self.max_local_entities_per_packet],
                "triples": triples[: self.max_local_triples_per_packet],
                "metadata": metadata,
            }

        summary_norm = normalize_text(summary_text)
        entity_scores: Dict[str, float] = {}
        for entity in entities:
            mention = _local_entity_mention(entity)
            score = self._mention_score(mention, summary_norm)
            if score > 0:
                entity_scores[mention] = max(entity_scores.get(mention, 0.0), score)

        triple_scores: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, triple in enumerate(triples):
            score = self._triple_score(triple, summary_norm, entity_scores)
            if score > 0:
                triple_scores.append((score, idx, triple))

        triple_scores.sort(
            key=lambda item: (
                item[0],
                float(item[2].get("confidence", 0.0) or 0.0),
                float(item[2].get("relevance", 0.0) or 0.0),
                float(item[2].get("clarity", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected_triples = [t for _, _, t in triple_scores[: self.max_local_triples_per_packet]]

        endpoint_scores: Dict[str, float] = {}
        for triple in selected_triples:
            for endpoint in (triple.get("head"), triple.get("tail")):
                if endpoint:
                    endpoint_scores[str(endpoint)] = max(endpoint_scores.get(str(endpoint), 0.0), 2.0)

        selected_entity_mentions = set(entity_scores) | set(endpoint_scores)
        entity_rows = []
        for entity in entities:
            mention = _local_entity_mention(entity)
            if mention in selected_entity_mentions:
                entity_rows.append((entity_scores.get(mention, 0.0) + endpoint_scores.get(mention, 0.0), entity))
        entity_rows.sort(key=lambda item: item[0], reverse=True)
        selected_entities = [
            entity for _, entity in entity_rows[: self.max_local_entities_per_packet]
        ]

        filter_mode = "article_filtered"
        if not selected_entities and not selected_triples:
            filter_mode = "article_fallback"
            selected_entities = entities[: self.max_local_entities_per_packet]
            selected_triples = triples[: self.max_local_triples_per_packet]

        metadata.update(
            {
                "hyperkg_filter_mode": filter_mode,
                "hyperkg_original_local_entity_count": original_entity_count,
                "hyperkg_original_local_triple_count": original_triple_count,
                "hyperkg_packet_entity_count": len(selected_entities),
                "hyperkg_packet_triple_count": len(selected_triples),
            }
        )
        return {
            **local_kg,
            "entities": selected_entities,
            "triples": selected_triples,
            "metadata": metadata,
        }

    @staticmethod
    def _mention_score(mention: str, summary_norm: str) -> float:
        mention_norm = normalize_text(mention)
        if not mention_norm or not summary_norm:
            return 0.0
        if len(mention_norm) < 3:
            return 0.0
        if mention_norm in summary_norm:
            return 3.0 + min(len(mention_norm) / 80.0, 1.0)
        if " " not in mention_norm and re.search(rf"\b{re.escape(mention_norm)}\b", summary_norm):
            return 3.0
        return 0.0

    def _triple_score(
        self,
        triple: Dict[str, Any],
        summary_norm: str,
        entity_scores: Dict[str, float],
    ) -> float:
        head = str(triple.get("head") or "")
        tail = str(triple.get("tail") or "")
        head_score = entity_scores.get(head) or self._mention_score(head, summary_norm)
        tail_score = entity_scores.get(tail) or self._mention_score(tail, summary_norm)
        if not head_score and not tail_score:
            return 0.0
        relation = normalize_text(str(triple.get("relation") or "").replace("_", " "))
        relation_bonus = 0.25 if relation and relation in summary_norm else 0.0
        quality = (
            float(triple.get("confidence", 0.0) or 0.0)
            + float(triple.get("relevance", 0.0) or 0.0)
            + float(triple.get("clarity", 0.0) or 0.0)
        ) / 3.0
        endpoint_score = 4.0 if head_score and tail_score else 2.0
        return endpoint_score + relation_bonus + 0.25 * quality

    def _link_local_entities(
        self,
        local_entities: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        canonical_map: Dict[str, str] = {}
        unresolved: List[Dict[str, Any]] = []

        for entity in local_entities:
            mention = str(entity.get("mention") or entity.get("name") or "").strip()
            if not mention:
                continue
            linked_id = self._deterministic_link(entity)
            if linked_id:
                canonical_map[mention] = linked_id
                continue

            dense_result = self._dense_link(entity)
            if dense_result.get("linked_id"):
                canonical_map[mention] = dense_result["linked_id"]
            else:
                unresolved.append(
                    {
                        "mention": mention,
                        "local_entity_type": entity.get("entity_type") or entity.get("type"),
                        "candidates": dense_result.get("candidates", []),
                        "reason": dense_result.get("reason", "no deterministic match"),
                    }
                )

        return canonical_map, unresolved

    def _deterministic_link(self, entity: Dict[str, Any]) -> Optional[str]:
        keys = [
            entity.get("normalized_id"),
            entity.get("ontology_id"),
            entity.get("canonical_name"),
            entity.get("name"),
            entity.get("mention"),
            entity.get("text"),
        ]
        keys.extend(entity.get("ontology_ids") or [])
        keys.extend(entity.get("aliases") or [])
        for key in keys:
            key_norm = normalize_text(key)
            if key_norm in self.lookup:
                return self.lookup[key_norm]

        mention_norm = normalize_text(entity.get("mention") or entity.get("name"))
        if not mention_norm or not self.lookup:
            return None
        matches = difflib.get_close_matches(mention_norm, self.lookup.keys(), n=1, cutoff=0.92)
        return self.lookup.get(matches[0]) if matches else None

    def _dense_link(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        if not (self.embedding_service and self.embedding_service.enabled and self.vector_index):
            return {"reason": "no dense embedding service"}
        mention = entity.get("mention") or entity.get("name") or ""
        text = entity_embedding_text(
            {
                "canonical_name": mention,
                "entity_type": entity.get("entity_type") or entity.get("type") or "",
                "aliases": entity.get("aliases") or [],
            }
        )
        try:
            query = self.embedding_service.encode_one(text)
            hits = self.vector_index.search(
                "canonical_entity",
                query_vector=query,
                top_k=self.entity_linking_top_k,
            )
        except Exception as err:  # noqa: BLE001
            return {"reason": f"dense retrieval failed: {err}"}

        candidates = []
        for hit in hits:
            meta = hit.get("metadata", {})
            candidates.append(
                {
                    "canonical_entity_id": hit["id"],
                    "score": round(float(hit.get("score", 0.0)), 4),
                    "canonical_name": meta.get("canonical_name", ""),
                    "entity_type": meta.get("entity_type", ""),
                }
            )
        if candidates and candidates[0]["score"] >= self.entity_linking_threshold:
            return {"linked_id": candidates[0]["canonical_entity_id"], "candidates": candidates}
        if candidates:
            return {
                "candidates": candidates,
                "reason": "dense candidate below auto-link threshold",
            }
        return {"reason": "dense retrieval returned no candidates"}
