"""HyperKGWriterAgent: persist HyperKG artifacts as JSONL plus vector indexes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..embedding_service import (
    EmbeddingService,
    canonical_hyperedge_embedding_text,
    evidence_hyperedge_embedding_text,
    summary_embedding_text,
)
from ..schemas import AtomicClaim, CanonicalHyperedge, EvidenceHyperedge, SegmentEvidencePacket
from ..utils import stable_hash, to_plain
from ..vector_index import VectorIndexAdapter


def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(to_plain(item), ensure_ascii=False, sort_keys=True) + "\n")


@dataclass
class HyperKGWriterAgent:
    output_dir: str
    embedding_service: Optional[EmbeddingService] = None
    vector_index: Optional[VectorIndexAdapter] = None
    graph_adapter: Optional[object] = None

    def process(
        self,
        evidence_hyperedges: List[EvidenceHyperedge],
        canonical_hyperedges: List[CanonicalHyperedge],
        packets: List[SegmentEvidencePacket],
        review_items: List[Dict[str, Any]],
        claims: Optional[List[AtomicClaim]] = None,
    ) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)

        self._maybe_index_outputs(evidence_hyperedges, canonical_hyperedges, packets, review_items)

        packet_rows = [p.to_dict() for p in packets]
        claim_rows = [c.to_dict() for c in (claims or [])]
        evidence_rows = [h.to_dict() for h in evidence_hyperedges]
        canonical_rows = [self._public_canonical_dict(h) for h in canonical_hyperedges]
        incidence_rows = self._incidence_edges(evidence_hyperedges)
        projection_rows = self._triple_projection_rows(evidence_hyperedges)
        summary_rows = self._summary_links(evidence_hyperedges)

        _write_jsonl(os.path.join(self.output_dir, "packets.jsonl"), packet_rows)
        _write_jsonl(os.path.join(self.output_dir, "claims.jsonl"), claim_rows)
        _write_jsonl(os.path.join(self.output_dir, "evidence_hyperedges.jsonl"), evidence_rows)
        _write_jsonl(os.path.join(self.output_dir, "canonical_hyperedges.jsonl"), canonical_rows)
        _write_jsonl(os.path.join(self.output_dir, "incidence_edges.jsonl"), incidence_rows)
        _write_jsonl(os.path.join(self.output_dir, "triple_projections.jsonl"), projection_rows)
        _write_jsonl(os.path.join(self.output_dir, "summary_links.jsonl"), summary_rows)
        _write_jsonl(os.path.join(self.output_dir, "review_queue.jsonl"), review_items)

        if self.vector_index is not None and self.vector_index.records:
            self.vector_index.save(os.path.join(self.output_dir, "vector_indexes"))

        stats = {
            "packet_count": len(packets),
            "claim_count": len(claims or []),
            "evidence_hyperedge_count": len(evidence_hyperedges),
            "accepted_evidence_hyperedge_count": sum(
                1 for h in evidence_hyperedges if h.decision == "ACCEPT"
            ),
            "review_evidence_hyperedge_count": sum(
                1 for h in evidence_hyperedges if h.decision == "REVIEW"
            ),
            "rejected_evidence_hyperedge_count": sum(
                1 for h in evidence_hyperedges if h.decision == "REJECT"
            ),
            "canonical_hyperedge_count": len(canonical_hyperedges),
            "incidence_edge_count": len(incidence_rows),
            "triple_projection_count": len(projection_rows),
            "review_item_count": len(review_items),
            "output_dir": self.output_dir,
        }
        with open(os.path.join(self.output_dir, "run_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
        return stats

    def _maybe_index_outputs(
        self,
        evidence_hyperedges: List[EvidenceHyperedge],
        canonical_hyperedges: List[CanonicalHyperedge],
        packets: List[SegmentEvidencePacket],
        review_items: List[Dict[str, Any]],
    ) -> None:
        if not (self.embedding_service and self.embedding_service.enabled and self.vector_index):
            return
        try:
            packet_texts = [summary_embedding_text(packet) for packet in packets]
            packet_ids = [packet.summary_id for packet in packets]
            if packet_ids:
                vectors = self.embedding_service.encode_texts(packet_texts)
                self.vector_index.add(
                    "summary",
                    packet_ids,
                    packet_texts,
                    vectors,
                    [
                        {
                            "article_id": packet.article_id,
                            "segment_id": packet.segment_id,
                            "summary_id": packet.summary_id,
                        }
                        for packet in packets
                    ],
                )

            evidence_texts = []
            evidence_ids = []
            evidence_metadata = []
            for hyperedge in evidence_hyperedges:
                if not hyperedge.embedding_text:
                    hyperedge.embedding_text = evidence_hyperedge_embedding_text(hyperedge)
                hyperedge.vector_id = self._vector_id("evidence_hyperedge", hyperedge.evidence_hyperedge_id)
                evidence_ids.append(hyperedge.evidence_hyperedge_id)
                evidence_texts.append(hyperedge.embedding_text)
                evidence_metadata.append(
                    {
                        "claim_type": hyperedge.claim_type,
                        "decision": hyperedge.decision,
                        "vector_id": hyperedge.vector_id,
                    }
                )
            if evidence_ids:
                vectors = self.embedding_service.encode_texts(evidence_texts)
                self.vector_index.add(
                    "evidence_hyperedge",
                    evidence_ids,
                    evidence_texts,
                    vectors,
                    evidence_metadata,
                )

            canonical_texts = []
            canonical_ids = []
            canonical_metadata = []
            for hyperedge in canonical_hyperedges:
                if not hyperedge.embedding_text:
                    hyperedge.embedding_text = canonical_hyperedge_embedding_text(hyperedge)
                hyperedge.vector_id = self._vector_id("canonical_hyperedge", hyperedge.canonical_hyperedge_id)
                canonical_ids.append(hyperedge.canonical_hyperedge_id)
                canonical_texts.append(hyperedge.embedding_text)
                canonical_metadata.append(
                    {
                        "claim_type": hyperedge.claim_type,
                        "support_count": hyperedge.support_count,
                        "vector_id": hyperedge.vector_id,
                    }
                )
            if canonical_ids:
                vectors = self.embedding_service.encode_texts(canonical_texts)
                self.vector_index.add(
                    "canonical_hyperedge",
                    canonical_ids,
                    canonical_texts,
                    vectors,
                    canonical_metadata,
                )
        except Exception as err:  # noqa: BLE001
            review_items.append(
                {
                    "review_item_id": f"REV:{len(review_items) + 1:06d}",
                    "type": "VECTOR_INDEX_FAILED",
                    "object_id": "vector_indexes",
                    "article_id": "",
                    "segment_id": "",
                    "summary_text": "",
                    "object_json": {},
                    "reason": f"Vector indexing failed: {err}",
                    "suggested_action": "Check embedding model configuration and dependencies.",
                }
            )

    def _vector_id(self, namespace: str, object_id: str) -> str:
        model = self.embedding_service.model_name if self.embedding_service else "none"
        return f"VEC:{namespace}:{object_id}:{stable_hash(model, 8)}"

    @staticmethod
    def _public_canonical_dict(hyperedge: CanonicalHyperedge) -> Dict[str, Any]:
        data = hyperedge.to_dict()
        confidence = dict(data.get("confidence_summary") or {})
        confidence.pop("_integration_values", None)
        data["confidence_summary"] = confidence
        return data

    @staticmethod
    def _incidence_edges(hyperedges: List[EvidenceHyperedge]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for hyperedge in hyperedges:
            for entity in hyperedge.entities:
                rows.append(
                    {
                        "entity_id": entity.entity_id,
                        "evidence_hyperedge_id": hyperedge.evidence_hyperedge_id,
                        "role": entity.role,
                        "mention": entity.mention,
                        "linking_confidence": entity.linking_confidence,
                    }
                )
        return rows

    @staticmethod
    def _triple_projection_rows(hyperedges: List[EvidenceHyperedge]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for hyperedge in hyperedges:
            for projection in hyperedge.triple_projections:
                rows.append(
                    {
                        "triple_projection_id": "TP:"
                        + stable_hash(
                            [
                                hyperedge.evidence_hyperedge_id,
                                projection.head_entity_id,
                                projection.relation,
                                projection.tail_entity_id,
                            ]
                        ),
                        "evidence_hyperedge_id": hyperedge.evidence_hyperedge_id,
                        "head_entity_id": projection.head_entity_id,
                        "relation": projection.relation,
                        "tail_entity_id": projection.tail_entity_id,
                        "support": projection.support,
                        "confidence": projection.confidence,
                    }
                )
        return rows

    @staticmethod
    def _summary_links(hyperedges: List[EvidenceHyperedge]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for hyperedge in hyperedges:
            rows.append(
                {
                    "evidence_hyperedge_id": hyperedge.evidence_hyperedge_id,
                    "article_id": hyperedge.source.get("article_id", ""),
                    "segment_id": hyperedge.source.get("segment_id", ""),
                    "summary_id": hyperedge.source.get("summary_id", ""),
                }
            )
        return rows
