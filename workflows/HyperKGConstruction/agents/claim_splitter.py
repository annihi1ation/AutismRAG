"""ClaimSplitterAgent: split segment summaries into atomic claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..schemas import AtomicClaim, SegmentEvidencePacket
from ..utils import as_list, pretty_json
from .base import PromptBackedAgent


def _claim_index_from_id(claim_id: str, default: int) -> str:
    suffix = str(claim_id).split(":")[-1]
    return suffix if suffix.isdigit() else f"{default:02d}"


@dataclass
class ClaimSplitterAgent(PromptBackedAgent):
    max_claims_per_summary: int = 8

    def __init__(
        self,
        client: Any,
        model_name: str,
        prompt_file: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 1,
        max_claims_per_summary: int = 8,
    ):
        self.max_claims_per_summary = max_claims_per_summary
        super().__init__(
            client=client,
            model_name=model_name,
            prompt_section="hyperkg_claim_splitter",
            prompt_file=prompt_file,
            temperature=temperature,
            max_retries=max_retries,
        )

    def process(self, packets: List[SegmentEvidencePacket]) -> List[AtomicClaim]:
        claims: List[AtomicClaim] = []
        for packet in packets:
            claims.extend(self.split_packet(packet))
        return claims

    def split_packet(self, packet: SegmentEvidencePacket) -> List[AtomicClaim]:
        if not packet.summary_text.strip():
            self.review_queue.add(
                "EMPTY_SUMMARY",
                packet.packet_id,
                "Cannot split claims from an empty summary.",
                "Check upstream summary artifact.",
                article_id=packet.article_id,
                segment_id=packet.segment_id,
                summary_text=packet.summary_text,
                object_json=packet.to_dict(),
            )
            return []

        user_prompt = self.render_prompt(
            summary_text=packet.summary_text,
            local_entities=pretty_json(packet.local_entities),
            local_triples=pretty_json(packet.local_triples),
            canonical_entity_map=pretty_json(packet.canonical_entity_map),
            metadata=pretty_json(packet.metadata),
        )
        result = self.llm.chat(self.system_prompt, user_prompt, want_json=True)
        parsed = result["json"]
        raw_items = self._extract_claim_items(parsed)
        if not raw_items:
            self.review_queue.add(
                "LLM_JSON_PARSE_FAILED",
                packet.packet_id,
                "Claim splitter returned no parseable JSON claims.",
                "Review the splitter prompt and output schema.",
                article_id=packet.article_id,
                segment_id=packet.segment_id,
                summary_text=packet.summary_text,
                object_json={"raw_llm_output": result["text"]},
            )
            return []

        claims: List[AtomicClaim] = []
        for idx, raw in enumerate(raw_items[: self.max_claims_per_summary], start=1):
            if not isinstance(raw, dict):
                continue
            claim_id = str(raw.get("claim_id") or f"C:{packet.article_id}:{packet.segment_id}:{idx:02d}")
            claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    packet_id=packet.packet_id,
                    claim_text=str(raw.get("claim_text") or raw.get("text") or "").strip(),
                    claim_type=str(raw.get("claim_type") or "OTHER").strip() or "OTHER",
                    candidate_entities=[str(x) for x in as_list(raw.get("candidate_entities"))],
                    candidate_triples=[
                        dict(x) for x in as_list(raw.get("candidate_triples")) if isinstance(x, dict)
                    ],
                    source_summary_id=packet.summary_id,
                    source_segment_id=packet.segment_id,
                    metadata=dict(raw.get("metadata") or packet.metadata or {}),
                    raw_llm_output=result["text"],
                )
            )
        return [claim for claim in claims if claim.claim_text]

    @staticmethod
    def _extract_claim_items(parsed: Any) -> List[Any]:
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("claims", "atomic_claims", "items"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
        return []
