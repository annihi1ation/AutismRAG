"""Pipeline controller for the HyperKG construction workflow."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agents import (
    ClaimSplitterAgent,
    EvidencePacketAgent,
    HyperedgeComposerAgent,
    HyperedgeCriticAgent,
    HyperedgeMergerAgent,
    HyperKGWriterAgent,
)
from .checkpoint import HyperKGCheckpointStore
from .config import HyperKGConfig
from .embedding_service import build_embedding_service
from .llm import build_openrouter_client
from .progress import ProgressReporter
from .prompt_store import load_prompts
from .schemas import (
    AtomicClaim,
    CanonicalHyperedge,
    EvidenceHyperedge,
    HyperedgeEntity,
    SegmentEvidencePacket,
    TripleProjection,
)
from .utils import stable_hash
from .vector_index import build_vector_index

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]


@dataclass
class HyperKGPipeline:
    client: Any
    model_name: str
    output_dir: str
    config: HyperKGConfig
    prompt_file: Optional[str] = None
    temperature: float = 0.1

    def __init__(
        self,
        client: Any,
        model_name: str,
        output_dir: str,
        config: Optional[HyperKGConfig] = None,
        embedding_service: Any = None,
        vector_index: Any = None,
        prompt_file: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.client = client
        self.model_name = model_name
        self.output_dir = output_dir
        self.config = config or HyperKGConfig()
        self.prompt_file = prompt_file
        self.temperature = temperature
        self.phase_timings: Dict[str, Dict[str, Any]] = {}

        self.embedding_service = embedding_service or build_embedding_service(self.config)
        self.vector_index = vector_index or (
            build_vector_index(self.embedding_service) if self.config.enable_vector_index else None
        )

        self.checkpoint_store = HyperKGCheckpointStore(
            self.config.checkpoint_dir or os.path.join(output_dir, "checkpoints"),
            enabled=self.config.checkpoint_enabled,
        )

        self.evidence_packet_agent = EvidencePacketAgent(
            embedding_service=self.embedding_service,
            vector_index=self.vector_index,
            entity_linking_threshold=self.config.entity_linking_threshold,
            entity_linking_top_k=self.config.entity_linking_top_k,
            max_local_entities_per_packet=self.config.max_local_entities_per_packet,
            max_local_triples_per_packet=self.config.max_local_triples_per_packet,
        )
        self.hyperedge_merger_agent = HyperedgeMergerAgent(
            client=client,
            model_name=model_name,
            embedding_service=self.embedding_service,
            vector_index=self.vector_index,
            prompt_file=prompt_file,
            temperature=temperature,
            max_retries=self.config.max_llm_retries,
            auto_merge_similarity=self.config.auto_merge_similarity,
            llm_merge_similarity=self.config.llm_merge_similarity,
            max_merge_candidates=self.config.max_merge_candidates,
        )
        self.hyperkg_writer_agent = HyperKGWriterAgent(
            output_dir=output_dir,
            embedding_service=self.embedding_service,
            vector_index=self.vector_index,
        )

    @classmethod
    def from_openrouter_env(
        cls,
        model_name: str,
        output_dir: str,
        config: Optional[HyperKGConfig] = None,
        api_key: str | None = None,
        api_base_url: str | None = None,
        env_file: str | None = None,
        prompt_file: Optional[str] = None,
        temperature: float = 0.1,
    ) -> "HyperKGPipeline":
        client = build_openrouter_client(
            api_key=api_key,
            base_url=api_base_url,
            env_file=env_file,
        )
        return cls(
            client=client,
            model_name=model_name,
            output_dir=output_dir,
            config=config,
            prompt_file=prompt_file,
            temperature=temperature,
        )

    def run(
        self,
        summaries: List[Dict[str, Any]],
        local_kgs: List[Dict[str, Any]] | Dict[str, Dict[str, Any]],
        unified_kg: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        existing_canonical_hyperedges: Optional[List[CanonicalHyperedge]] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        self.phase_timings = {}

        checkpoint_info = self.checkpoint_store.prepare(
            self._fingerprint(
                summaries=summaries,
                local_kgs=local_kgs,
                unified_kg=unified_kg,
                metadata=metadata,
                existing_canonical_hyperedges=existing_canonical_hyperedges,
            ),
            resume_enabled=self.config.resume_enabled,
        )

        packets = self._packet_phase(summaries, local_kgs, unified_kg, metadata)
        packet_map = {packet.packet_id: packet for packet in packets}

        claims, splitter_review_items = self._claim_split_phase(packets)
        evidence_hyperedges, composer_review_items = self._compose_phase(claims, packet_map)
        accepted_or_reviewed, critic_review_items = self._critic_phase(evidence_hyperedges, packet_map)
        accepted = [h for h in accepted_or_reviewed if h.decision == "ACCEPT"]
        canonical_hyperedges, merge_review_items = self._merge_phase(
            accepted,
            existing_canonical_hyperedges=existing_canonical_hyperedges,
        )

        review_items = self._renumber_review_items(
            splitter_review_items
            + composer_review_items
            + critic_review_items
            + merge_review_items
            + self._packet_review_items(packets)
        )

        phase_t0 = self._start_phase("write", 1, workers=1, skipped=0)
        run_stats = self.hyperkg_writer_agent.process(
            evidence_hyperedges=accepted_or_reviewed,
            canonical_hyperedges=canonical_hyperedges,
            packets=packets,
            review_items=review_items,
            claims=claims,
        )
        self._finish_phase("write", phase_t0, count=1, skipped=0, workers=1)

        run_stats["timing_sec"] = round(time.time() - t0, 2)
        run_stats["phase_timings"] = self.phase_timings
        run_stats["workers"] = max(1, int(self.config.max_workers))
        run_stats["checkpoint"] = checkpoint_info
        with open(os.path.join(self.output_dir, "run_stats.json"), "w", encoding="utf-8") as f:
            json.dump(run_stats, f, ensure_ascii=False, indent=2, sort_keys=True)

        return {
            "packets": [p.to_dict() for p in packets],
            "claims": [c.to_dict() for c in claims],
            "evidence_hyperedges": [h.to_dict() for h in accepted_or_reviewed],
            "canonical_hyperedges": [h.to_dict() for h in canonical_hyperedges],
            "review_items": review_items,
            "run_stats": run_stats,
        }

    def _packet_phase(
        self,
        summaries: List[Dict[str, Any]],
        local_kgs: List[Dict[str, Any]] | Dict[str, Dict[str, Any]],
        unified_kg: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> List[SegmentEvidencePacket]:
        cached = self._load_packet_checkpoint()
        if cached:
            phase_t0 = self._start_phase("packet", len(cached), workers=1, skipped=len(cached))
            self._finish_phase("packet", phase_t0, count=len(cached), skipped=len(cached), workers=1)
            return cached

        phase_t0 = self._start_phase("packet", len(summaries), workers=1, skipped=0)
        packets = self.evidence_packet_agent.process(
            summaries=summaries,
            local_kgs=local_kgs,
            unified_kg=unified_kg,
            metadata=metadata,
        )
        rows = [
            {"object_id": packet.packet_id, "index": idx, "packet": packet.to_dict()}
            for idx, packet in enumerate(packets)
        ]
        self.checkpoint_store.write_records("packets", rows)
        self._finish_phase("packet", phase_t0, count=len(packets), skipped=0, workers=1)
        return packets

    def _claim_split_phase(
        self,
        packets: List[SegmentEvidencePacket],
    ) -> Tuple[List[AtomicClaim], List[JsonDict]]:
        cached = self._load_result_rows("claim_split", "claims")
        rows: Dict[str, JsonDict] = {}
        pending: List[Tuple[int, SegmentEvidencePacket]] = []
        skipped = 0

        for idx, packet in enumerate(packets):
            row = cached.get(packet.packet_id)
            if row is None:
                pending.append((idx, packet))
            else:
                rows[packet.packet_id] = row
                skipped += 1

        phase_t0 = self._start_phase(
            "claim_split",
            len(packets),
            workers=self._effective_workers(pending),
            skipped=skipped,
        )
        for row in self._run_ordered_tasks("claim_split", pending, self._split_packet_task):
            rows[str(row["object_id"])] = row

        ordered_rows = [rows[packet.packet_id] for packet in packets if packet.packet_id in rows]
        self.checkpoint_store.write_records("claim_split", ordered_rows)
        self._finish_phase(
            "claim_split",
            phase_t0,
            count=len(packets),
            skipped=skipped,
            workers=self._effective_workers(pending),
        )

        claims: List[AtomicClaim] = []
        review_items: List[JsonDict] = []
        for row in ordered_rows:
            claims.extend(_claim_from_dict(item) for item in row.get("claims", []))
            review_items.extend(_review_items(row))
        return claims, review_items

    def _compose_phase(
        self,
        claims: List[AtomicClaim],
        packet_map: Dict[str, SegmentEvidencePacket],
    ) -> Tuple[List[EvidenceHyperedge], List[JsonDict]]:
        cached = self._load_result_rows("compose", "hyperedge")
        rows: Dict[str, JsonDict] = {}
        pending: List[Tuple[int, AtomicClaim, Dict[str, SegmentEvidencePacket]]] = []
        skipped = 0

        for idx, claim in enumerate(claims):
            row = cached.get(claim.claim_id)
            if row is None:
                pending.append((idx, claim, packet_map))
            else:
                rows[claim.claim_id] = row
                skipped += 1

        phase_t0 = self._start_phase(
            "compose",
            len(claims),
            workers=self._effective_workers(pending),
            skipped=skipped,
        )
        for row in self._run_ordered_tasks("compose", pending, self._compose_claim_task):
            rows[str(row["object_id"])] = row

        ordered_rows = [rows[claim.claim_id] for claim in claims if claim.claim_id in rows]
        self.checkpoint_store.write_records("compose", ordered_rows)
        self._finish_phase(
            "compose",
            phase_t0,
            count=len(claims),
            skipped=skipped,
            workers=self._effective_workers(pending),
        )

        hyperedges: List[EvidenceHyperedge] = []
        review_items: List[JsonDict] = []
        for row in ordered_rows:
            if isinstance(row.get("hyperedge"), dict):
                hyperedges.append(_hyperedge_from_dict(row["hyperedge"]))
            review_items.extend(_review_items(row))
        return hyperedges, review_items

    def _critic_phase(
        self,
        evidence_hyperedges: List[EvidenceHyperedge],
        packet_map: Dict[str, SegmentEvidencePacket],
    ) -> Tuple[List[EvidenceHyperedge], List[JsonDict]]:
        cached = self._load_result_rows("critic", "hyperedge")
        rows: Dict[str, JsonDict] = {}
        pending: List[Tuple[int, EvidenceHyperedge, Dict[str, SegmentEvidencePacket]]] = []
        skipped = 0

        for idx, hyperedge in enumerate(evidence_hyperedges):
            row = cached.get(hyperedge.evidence_hyperedge_id)
            if row is None:
                pending.append((idx, hyperedge, packet_map))
            else:
                rows[hyperedge.evidence_hyperedge_id] = row
                skipped += 1

        phase_t0 = self._start_phase(
            "critic",
            len(evidence_hyperedges),
            workers=self._effective_workers(pending),
            skipped=skipped,
        )
        for row in self._run_ordered_tasks("critic", pending, self._critic_hyperedge_task):
            rows[str(row["object_id"])] = row

        ordered_rows = [
            rows[hyperedge.evidence_hyperedge_id]
            for hyperedge in evidence_hyperedges
            if hyperedge.evidence_hyperedge_id in rows
        ]
        self.checkpoint_store.write_records("critic", ordered_rows)
        self._finish_phase(
            "critic",
            phase_t0,
            count=len(evidence_hyperedges),
            skipped=skipped,
            workers=self._effective_workers(pending),
        )

        reviewed: List[EvidenceHyperedge] = []
        review_items: List[JsonDict] = []
        for row in ordered_rows:
            if isinstance(row.get("hyperedge"), dict):
                reviewed.append(_hyperedge_from_dict(row["hyperedge"]))
            review_items.extend(_review_items(row))
        return reviewed, review_items

    def _merge_phase(
        self,
        accepted: List[EvidenceHyperedge],
        existing_canonical_hyperedges: Optional[List[CanonicalHyperedge]],
    ) -> Tuple[List[CanonicalHyperedge], List[JsonDict]]:
        cached = self.checkpoint_store.load_json("merge") if self.config.resume_enabled else {}
        if isinstance(cached.get("canonical_hyperedges"), list):
            canonical = [_canonical_from_dict(row) for row in cached.get("canonical_hyperedges", [])]
            reviews = _review_items(cached)
            phase_t0 = self._start_phase("merge", len(accepted), workers=1, skipped=len(accepted))
            self._finish_phase("merge", phase_t0, count=len(accepted), skipped=len(accepted), workers=1)
            return canonical, reviews

        phase_t0 = self._start_phase("merge", len(accepted), workers=1, skipped=0)
        canonical, reviews = self.hyperedge_merger_agent.process(
            evidence_hyperedges=accepted,
            existing_canonical_hyperedges=existing_canonical_hyperedges,
        )
        self.checkpoint_store.write_json(
            "merge",
            {
                "canonical_hyperedges": [row.to_dict() for row in canonical],
                "review_items": reviews,
            },
        )
        self._finish_phase("merge", phase_t0, count=len(accepted), skipped=0, workers=1)
        return canonical, reviews

    def _run_ordered_tasks(
        self,
        phase: str,
        pending: List[Any],
        task_fn: Callable[[Any], JsonDict],
    ) -> List[JsonDict]:
        if not pending:
            return []

        workers = self._effective_workers(pending)
        rows: List[JsonDict] = []
        with ProgressReporter(
            phase=phase,
            total=len(pending),
            enabled=self.config.progress_enabled,
            logger=logger,
            log_interval_sec=self.config.progress_log_interval_sec,
        ) as progress:
            if workers <= 1:
                for item in pending:
                    row = task_fn(item)
                    self.checkpoint_store.append_record(phase, row)
                    rows.append(row)
                    progress.update()
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(task_fn, item): item for item in pending}
                    for future in as_completed(futures):
                        row = future.result()
                        self.checkpoint_store.append_record(phase, row)
                        rows.append(row)
                        progress.update()
        return sorted(rows, key=lambda row: int(row.get("index", 0)))

    def _split_packet_task(self, item: Tuple[int, SegmentEvidencePacket]) -> JsonDict:
        idx, packet = item
        agent = self._new_claim_splitter()
        claims = agent.split_packet(packet)
        return {
            "object_id": packet.packet_id,
            "index": idx,
            "claims": [claim.to_dict() for claim in claims],
            "review_items": agent.review_items,
        }

    def _compose_claim_task(
        self,
        item: Tuple[int, AtomicClaim, Dict[str, SegmentEvidencePacket]],
    ) -> JsonDict:
        idx, claim, packet_map = item
        packet = packet_map.get(claim.packet_id)
        if packet is None:
            return {
                "object_id": claim.claim_id,
                "index": idx,
                "hyperedge": None,
                "review_items": [
                    {
                        "type": "MISSING_PACKET",
                        "object_id": claim.claim_id,
                        "article_id": "",
                        "segment_id": "",
                        "summary_text": "",
                        "object_json": claim.to_dict(),
                        "reason": "Claim references a packet that is not available.",
                        "suggested_action": "Check pipeline packet map construction.",
                    }
                ],
            }
        agent = self._new_composer()
        hyperedge = agent.compose_one(claim, packet, idx + 1)
        return {
            "object_id": claim.claim_id,
            "index": idx,
            "hyperedge": hyperedge.to_dict(),
            "review_items": agent.review_items,
        }

    def _critic_hyperedge_task(
        self,
        item: Tuple[int, EvidenceHyperedge, Dict[str, SegmentEvidencePacket]],
    ) -> JsonDict:
        idx, hyperedge, packet_map = item
        packet = HyperedgeCriticAgent._packet_for_hyperedge(hyperedge, packet_map)
        if packet is None:
            hyperedge.decision = "REVIEW"
            hyperedge.warnings.append("critic could not find source packet")
            review_items = [
                {
                    "type": "MISSING_PACKET",
                    "object_id": hyperedge.evidence_hyperedge_id,
                    "article_id": "",
                    "segment_id": "",
                    "summary_text": "",
                    "object_json": hyperedge.to_dict(),
                    "reason": "Critic could not find the source packet for this hyperedge.",
                    "suggested_action": "Check source ids and packet map.",
                }
            ]
        else:
            agent = self._new_critic()
            hyperedge = agent.critique_one(hyperedge, packet)
            review_items = agent.review_items
        return {
            "object_id": hyperedge.evidence_hyperedge_id,
            "index": idx,
            "hyperedge": hyperedge.to_dict(),
            "review_items": review_items,
        }

    def _new_claim_splitter(self) -> ClaimSplitterAgent:
        return ClaimSplitterAgent(
            client=self.client,
            model_name=self.model_name,
            prompt_file=self.prompt_file,
            temperature=self.temperature,
            max_retries=self.config.max_llm_retries,
            max_claims_per_summary=self.config.max_claims_per_summary,
        )

    def _new_composer(self) -> HyperedgeComposerAgent:
        return HyperedgeComposerAgent(
            client=self.client,
            model_name=self.model_name,
            prompt_file=self.prompt_file,
            temperature=self.temperature,
            max_retries=self.config.max_llm_retries,
            min_entities_per_hyperedge=self.config.min_entities_per_hyperedge,
        )

    def _new_critic(self) -> HyperedgeCriticAgent:
        return HyperedgeCriticAgent(
            client=self.client,
            model_name=self.model_name,
            prompt_file=self.prompt_file,
            temperature=self.temperature,
            max_retries=self.config.max_llm_retries,
            accept_threshold=self.config.accept_threshold,
            review_threshold=self.config.review_threshold,
        )

    def _load_packet_checkpoint(self) -> List[SegmentEvidencePacket]:
        if not self.config.resume_enabled:
            return []
        phase = self.checkpoint_store.manifest.get("phases", {}).get("packet", {})
        if phase.get("status") != "complete":
            return []
        rows = self.checkpoint_store.load_records(
            "packets",
            validate=lambda row: isinstance(row.get("packet"), dict),
        )
        if not rows:
            return []
        ordered = sorted(rows.values(), key=lambda row: int(row.get("index", 0)))
        return [_packet_from_dict(row["packet"]) for row in ordered]

    def _load_result_rows(self, phase: str, payload_key: str) -> Dict[str, JsonDict]:
        if not self.config.resume_enabled:
            return {}
        return self.checkpoint_store.load_records(
            phase,
            validate=lambda row: payload_key in row and "index" in row,
        )

    def _start_phase(self, phase: str, count: int, workers: int, skipped: int) -> float:
        logger.info(
            "HyperKG phase=%s status=start count=%d skipped=%d workers=%d checkpoint=%s",
            phase,
            count,
            skipped,
            workers,
            self.checkpoint_store.enabled,
        )
        self.checkpoint_store.mark_phase(phase, "running", count=count, skipped=skipped)
        return time.time()

    def _finish_phase(self, phase: str, started_at: float, count: int, skipped: int, workers: int) -> None:
        duration = time.time() - started_at
        self.phase_timings[phase] = {
            "duration_sec": round(duration, 2),
            "count": count,
            "skipped": skipped,
            "workers": workers,
        }
        self.checkpoint_store.mark_phase(
            phase,
            "complete",
            count=count,
            skipped=skipped,
            duration_sec=duration,
        )
        logger.info(
            "HyperKG phase=%s status=complete count=%d skipped=%d workers=%d duration_sec=%.2f",
            phase,
            count,
            skipped,
            workers,
            duration,
        )

    def _effective_workers(self, pending: List[Any]) -> int:
        if not pending:
            return 1
        return max(1, min(int(self.config.max_workers or 1), len(pending)))

    def _fingerprint(
        self,
        summaries: List[Dict[str, Any]],
        local_kgs: List[Dict[str, Any]] | Dict[str, Dict[str, Any]],
        unified_kg: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        existing_canonical_hyperedges: Optional[List[CanonicalHyperedge]],
    ) -> JsonDict:
        config_data = asdict(self.config)
        for runtime_key in (
            "max_workers",
            "checkpoint_enabled",
            "resume_enabled",
            "checkpoint_dir",
            "progress_enabled",
            "progress_log_interval_sec",
        ):
            config_data.pop(runtime_key, None)
        payload = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "config_hash": stable_hash(config_data, 16),
            "prompt_hash": stable_hash(load_prompts(self.prompt_file), 16),
            "input_hash": stable_hash(
                {
                    "summaries": summaries,
                    "local_kgs": local_kgs,
                    "unified_kg": unified_kg,
                    "metadata": metadata or {},
                    "existing_canonical_hyperedges": [
                        h.to_dict() if hasattr(h, "to_dict") else h
                        for h in (existing_canonical_hyperedges or [])
                    ],
                },
                16,
            ),
        }
        payload["fingerprint_id"] = stable_hash(payload, 16)
        return payload

    @staticmethod
    def _packet_review_items(packets: List[Any]) -> List[JsonDict]:
        items: List[JsonDict] = []
        for packet in packets:
            for unresolved in packet.unresolved_entities:
                items.append(
                    {
                        "type": "UNRESOLVED_ENTITY",
                        "object_id": packet.packet_id,
                        "article_id": packet.article_id,
                        "segment_id": packet.segment_id,
                        "summary_text": packet.summary_text,
                        "object_json": unresolved,
                        "reason": unresolved.get("reason", "entity could not be linked"),
                        "suggested_action": "Inspect canonical entity candidates or add aliases to unified KG.",
                    }
                )
            for warning in packet.warnings:
                items.append(
                    {
                        "type": "PACKET_WARNING",
                        "object_id": packet.packet_id,
                        "article_id": packet.article_id,
                        "segment_id": packet.segment_id,
                        "summary_text": packet.summary_text,
                        "object_json": packet.to_dict(),
                        "reason": warning,
                        "suggested_action": "Check upstream summary/local KG artifacts.",
                    }
                )
        return items

    @staticmethod
    def _renumber_review_items(items: List[JsonDict]) -> List[JsonDict]:
        out: List[JsonDict] = []
        for idx, item in enumerate(items, start=1):
            row = dict(item)
            row["review_item_id"] = f"REV:{idx:06d}"
            out.append(row)
        return out


def _review_items(row: JsonDict) -> List[JsonDict]:
    return [dict(item) for item in row.get("review_items", []) if isinstance(item, dict)]


def _packet_from_dict(data: JsonDict) -> SegmentEvidencePacket:
    return SegmentEvidencePacket(
        packet_id=str(data.get("packet_id") or ""),
        article_id=str(data.get("article_id") or ""),
        segment_id=str(data.get("segment_id") or ""),
        summary_id=str(data.get("summary_id") or ""),
        summary_text=str(data.get("summary_text") or ""),
        local_entities=[dict(x) for x in data.get("local_entities", []) if isinstance(x, dict)],
        local_triples=[dict(x) for x in data.get("local_triples", []) if isinstance(x, dict)],
        canonical_entity_map={str(k): str(v) for k, v in dict(data.get("canonical_entity_map") or {}).items()},
        unresolved_entities=[
            dict(x) for x in data.get("unresolved_entities", []) if isinstance(x, dict)
        ],
        metadata=dict(data.get("metadata") or {}),
        warnings=[str(x) for x in data.get("warnings", [])],
    )


def _claim_from_dict(data: JsonDict) -> AtomicClaim:
    return AtomicClaim(
        claim_id=str(data.get("claim_id") or ""),
        packet_id=str(data.get("packet_id") or ""),
        claim_text=str(data.get("claim_text") or ""),
        claim_type=str(data.get("claim_type") or "OTHER"),
        candidate_entities=[str(x) for x in data.get("candidate_entities", [])],
        candidate_triples=[dict(x) for x in data.get("candidate_triples", []) if isinstance(x, dict)],
        source_summary_id=str(data.get("source_summary_id") or ""),
        source_segment_id=str(data.get("source_segment_id") or ""),
        metadata=dict(data.get("metadata") or {}),
        raw_llm_output=str(data.get("raw_llm_output") or ""),
    )


def _hyperedge_from_dict(data: JsonDict) -> EvidenceHyperedge:
    return EvidenceHyperedge(
        evidence_hyperedge_id=str(data.get("evidence_hyperedge_id") or ""),
        claim_text=str(data.get("claim_text") or ""),
        claim_type=str(data.get("claim_type") or "OTHER"),
        entities=[
            HyperedgeEntity(
                entity_id=str(entity.get("entity_id") or ""),
                mention=str(entity.get("mention") or ""),
                role=str(entity.get("role") or "unknown"),
                entity_type=entity.get("entity_type"),
                linking_confidence=float(entity.get("linking_confidence", 1.0) or 0.0),
            )
            for entity in data.get("entities", [])
            if isinstance(entity, dict)
        ],
        qualifiers=dict(data.get("qualifiers") or {}),
        triple_projections=[
            TripleProjection(
                head_entity_id=str(projection.get("head_entity_id") or ""),
                relation=str(projection.get("relation") or ""),
                tail_entity_id=str(projection.get("tail_entity_id") or ""),
                support=str(projection.get("support") or ""),
                confidence=float(projection.get("confidence", 0.5) or 0.0),
            )
            for projection in data.get("triple_projections", [])
            if isinstance(projection, dict)
        ],
        source=dict(data.get("source") or {}),
        scores={str(k): float(v or 0.0) for k, v in dict(data.get("scores") or {}).items()},
        decision=str(data.get("decision") or "CANDIDATE"),
        warnings=[str(x) for x in data.get("warnings", [])],
        embedding_text=data.get("embedding_text"),
        vector_id=data.get("vector_id"),
        raw_llm_output=str(data.get("raw_llm_output") or ""),
    )


def _canonical_from_dict(data: JsonDict) -> CanonicalHyperedge:
    return CanonicalHyperedge(
        canonical_hyperedge_id=str(data.get("canonical_hyperedge_id") or ""),
        canonical_claim=str(data.get("canonical_claim") or ""),
        claim_type=str(data.get("claim_type") or "OTHER"),
        member_evidence_hyperedges=[str(x) for x in data.get("member_evidence_hyperedges", [])],
        support_count=int(data.get("support_count") or 0),
        core_entity_ids=[str(x) for x in data.get("core_entity_ids", [])],
        qualifier_summary=dict(data.get("qualifier_summary") or {}),
        scope_summary=dict(data.get("scope_summary") or {}),
        confidence_summary=dict(data.get("confidence_summary") or {}),
        related_hyperedges=[dict(x) for x in data.get("related_hyperedges", []) if isinstance(x, dict)],
        conflicts=[dict(x) for x in data.get("conflicts", []) if isinstance(x, dict)],
        embedding_text=data.get("embedding_text"),
        vector_id=data.get("vector_id"),
    )
