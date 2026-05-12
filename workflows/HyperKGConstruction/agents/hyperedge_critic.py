"""HyperedgeCriticAgent: score and accept/review/reject evidence hyperedges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..schemas import EvidenceHyperedge, SegmentEvidencePacket
from ..utils import as_list, clip01, pretty_json
from .base import PromptBackedAgent


SCORE_WEIGHTS = {
    "faithfulness": 0.30,
    "entity_linking": 0.20,
    "local_kg_agreement": 0.20,
    "scope_completeness": 0.15,
    "relation_correctness": 0.15,
}


@dataclass
class HyperedgeCriticAgent(PromptBackedAgent):
    accept_threshold: float = 0.75
    review_threshold: float = 0.55

    def __init__(
        self,
        client: Any,
        model_name: str,
        prompt_file: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 1,
        accept_threshold: float = 0.75,
        review_threshold: float = 0.55,
    ):
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold
        super().__init__(
            client=client,
            model_name=model_name,
            prompt_section="hyperkg_hyperedge_critic",
            prompt_file=prompt_file,
            temperature=temperature,
            max_retries=max_retries,
        )

    def process(
        self,
        hyperedges: List[EvidenceHyperedge],
        packet_map: Dict[str, SegmentEvidencePacket],
    ) -> Tuple[List[EvidenceHyperedge], List[Dict[str, Any]]]:
        reviewed: List[EvidenceHyperedge] = []
        for hyperedge in hyperedges:
            packet = self._packet_for_hyperedge(hyperedge, packet_map)
            if packet is None:
                hyperedge.decision = "REVIEW"
                hyperedge.warnings.append("critic could not find source packet")
                self.review_queue.add(
                    "MISSING_PACKET",
                    hyperedge.evidence_hyperedge_id,
                    "Critic could not find the source packet for this hyperedge.",
                    "Check source ids and packet map.",
                    object_json=hyperedge.to_dict(),
                )
                reviewed.append(hyperedge)
                continue
            reviewed.append(self.critique_one(hyperedge, packet))
        return reviewed, self.review_items

    def critique_one(
        self,
        hyperedge: EvidenceHyperedge,
        packet: SegmentEvidencePacket,
    ) -> EvidenceHyperedge:
        user_prompt = self.render_prompt(
            hyperedge_json=pretty_json(hyperedge.to_dict()),
            summary_text=packet.summary_text,
            local_entities=pretty_json(packet.local_entities),
            local_triples=pretty_json(packet.local_triples),
            canonical_entity_map=pretty_json(packet.canonical_entity_map),
        )
        result = self.llm.chat(self.system_prompt, user_prompt, want_json=True)
        parsed = result["json"] if isinstance(result["json"], dict) else {}

        scores = dict(hyperedge.scores or {})
        raw_scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else parsed
        for key in SCORE_WEIGHTS:
            scores[key] = clip01(raw_scores.get(key), scores.get(key, 0.5))
        scores["integration_score"] = self.aggregate_scores(scores)

        hyperedge.scores = scores
        warnings = [str(w) for w in as_list(parsed.get("warnings"))]
        violations = [str(v) for v in as_list(parsed.get("violations"))]
        hyperedge.warnings.extend(warnings + violations)
        proposed = str(parsed.get("decision") or "").upper()
        hyperedge.decision = proposed if proposed in {"ACCEPT", "REVIEW", "REJECT"} else self._decision(scores)

        if hyperedge.decision == "REVIEW" or warnings or violations:
            self.review_queue.add(
                "CRITIC_REVIEW",
                hyperedge.evidence_hyperedge_id,
                "; ".join(warnings + violations) or "Integration score is in review range.",
                "Inspect faithfulness, relation correctness, and entity linking.",
                article_id=packet.article_id,
                segment_id=packet.segment_id,
                summary_text=packet.summary_text,
                object_json=hyperedge.to_dict(),
            )
        return hyperedge

    @staticmethod
    def aggregate_scores(scores: Dict[str, Any]) -> float:
        return round(sum(SCORE_WEIGHTS[k] * clip01(scores.get(k), 0.5) for k in SCORE_WEIGHTS), 4)

    def _decision(self, scores: Dict[str, Any]) -> str:
        integration = clip01(scores.get("integration_score"), 0.0)
        if integration >= self.accept_threshold:
            return "ACCEPT"
        if integration >= self.review_threshold:
            return "REVIEW"
        return "REJECT"

    @staticmethod
    def _packet_for_hyperedge(
        hyperedge: EvidenceHyperedge,
        packet_map: Dict[str, SegmentEvidencePacket],
    ) -> SegmentEvidencePacket | None:
        article_id = hyperedge.source.get("article_id")
        segment_id = hyperedge.source.get("segment_id")
        for packet in packet_map.values():
            if packet.article_id == article_id and packet.segment_id == segment_id:
                return packet
        return None
