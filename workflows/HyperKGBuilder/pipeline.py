"""HyperKGBuilder end-to-end orchestrator."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from workflows.HyperKGConstruction.embedding_service import EmbeddingService
from workflows.HyperKGConstruction.llm import ChatLLM
from workflows.HyperKGConstruction.review_queue import ReviewQueue
from workflows.HyperKGConstruction.utils import stable_hash, to_plain

from .agents import (
    CandidatePackRouterAgent,
    ClaimKGEmbeddingIndexAgent,
    HyperKGWriterIndexerAgent,
    OnlineLLMHyperedgeAgent,
)
from .agents.online_hyperedge_agent import _hyperedge_from_dict
from .claim_loader import load_claims_jsonl, load_summaries_index
from .config import HyperKGRunConfig
from .schemas import CandidatePack, EvidenceHyperedge, HyperKGRunReport
from .unified_kg_index import UnifiedKGIndex

logger = logging.getLogger(__name__)


class HyperKGBuilderPipeline:
    def __init__(
        self,
        config: HyperKGRunConfig,
        embedding_service: Optional[EmbeddingService] = None,
        chat_llm: Optional[ChatLLM] = None,
    ):
        self.config = config
        self.embedding_service = embedding_service
        self.chat_llm = chat_llm
        self.review_queue = ReviewQueue()
        self.run_id = uuid.uuid4().hex[:12]
        self._setup_dirs()

    # ------------------------------------------------------------------
    def _setup_dirs(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        if not self.config.checkpoint_dir:
            self.config.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.checkpoint_dir, "online_cache"), exist_ok=True)

    # ------------------------------------------------------------------
    def _fingerprint(self, claims_size: int, kg_size: int) -> Dict[str, Any]:
        try:
            claim_stat = os.stat(self.config.claims_path)
            kg_stat = os.stat(self.config.unified_kg_path)
        except OSError:
            claim_stat = kg_stat = None
        return {
            "claims_path": self.config.claims_path,
            "claims_mtime": getattr(claim_stat, "st_mtime", 0),
            "claims_size": getattr(claim_stat, "st_size", 0),
            "claims_count": claims_size,
            "unified_kg_path": self.config.unified_kg_path,
            "kg_mtime": getattr(kg_stat, "st_mtime", 0),
            "kg_size": getattr(kg_stat, "st_size", 0),
            "kg_entity_count": kg_size,
            "model_name": self.config.model_name,
            "embedding_model_name": self.config.embedding.model_name or self.config.embedding.model_path or "",
            "routing": asdict(self.config.routing),
            "budget": asdict(self.config.budget),
        }

    def _manifest_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "manifest.json")

    def _packs_checkpoint_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "candidate_packs.jsonl")

    def _online_results_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "online_results.jsonl")

    def _online_cache_dir(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "online_cache")

    def _read_existing_manifest(self) -> Dict[str, Any]:
        path = self._manifest_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        with open(self._manifest_path(), "w", encoding="utf-8") as f:
            json.dump(to_plain(manifest), f, ensure_ascii=False, sort_keys=True, indent=2)

    # ------------------------------------------------------------------
    def run(self) -> HyperKGRunReport:
        report = HyperKGRunReport(
            run_id=self.run_id,
            started_at=time.time(),
            dry_run=self.config.dry_run,
        )

        # Phase 1: load
        claims = load_claims_jsonl(self.config.claims_path)
        unified_kg_index = UnifiedKGIndex.from_path(
            self.config.unified_kg_path, embedding_service=self.embedding_service
        )
        summaries = load_summaries_index(claims, self.config.summaries_path)
        report.phase_counts["load"] = {
            "claims": len(claims),
            "kg_entities": len(unified_kg_index.entities),
            "kg_triples": len(unified_kg_index.triples),
            "summaries": len(summaries),
        }
        logger.info(
            "phase=load claims=%d kg_entities=%d kg_triples=%d summaries=%d",
            len(claims),
            len(unified_kg_index.entities),
            len(unified_kg_index.triples),
            len(summaries),
        )

        # Fingerprint + resume bookkeeping
        fingerprint = self._fingerprint(len(claims), len(unified_kg_index.entities))
        report.fingerprint = fingerprint
        existing = self._read_existing_manifest()
        resumed = False
        if self.config.resume and existing.get("fingerprint_hash") == stable_hash(fingerprint):
            resumed = True
        else:
            # Wipe stale checkpoint artifacts.
            for path in (self._packs_checkpoint_path(), self._online_results_path()):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            cache_dir = self._online_cache_dir()
            if os.path.isdir(cache_dir):
                for name in os.listdir(cache_dir):
                    try:
                        os.remove(os.path.join(cache_dir, name))
                    except OSError:
                        pass
        report.resumed = resumed
        manifest = {
            "run_id": self.run_id,
            "fingerprint": fingerprint,
            "fingerprint_hash": stable_hash(fingerprint),
            "started_at": report.started_at,
            "phases": {},
        }
        self._write_manifest(manifest)

        # Phase 2: index
        embedding_agent = ClaimKGEmbeddingIndexAgent(embedding_service=self.embedding_service)
        adapter, index_manifest = embedding_agent.build(claims, unified_kg_index, summaries)
        manifest["phases"]["index"] = {
            "namespaces": index_manifest.namespaces,
            "embedding_model": index_manifest.embedding_model,
            "embedding_dim": index_manifest.embedding_dim,
        }
        report.phase_counts["index"] = manifest["phases"]["index"]
        self._write_manifest(manifest)
        logger.info("phase=index namespaces=%s", index_manifest.namespaces)

        # Phase 3: route
        packs: List[CandidatePack] = []
        if resumed and os.path.exists(self._packs_checkpoint_path()):
            packs = self._load_packs_from_jsonl(self._packs_checkpoint_path())
            logger.info("phase=route resumed=%d packs from checkpoint", len(packs))
        if not packs:
            router = CandidatePackRouterAgent(
                routing=self.config.routing,
                embedding_service=self.embedding_service,
            )
            packs, router_report = router.build_packs(claims, adapter, unified_kg_index, summaries)
            self._write_packs_jsonl(self._packs_checkpoint_path(), packs)
            report.routing_distribution = {
                "auto": router_report.auto_packs,
                "online_llm": router_report.online_packs,
                **router_report.routing_reason_counts,
            }
        else:
            report.routing_distribution = {
                "auto": sum(1 for p in packs if p.auto_decision == "auto"),
                "online_llm": sum(1 for p in packs if p.auto_decision == "online_llm"),
            }
        report.phase_counts["route"] = {
            "total_packs": len(packs),
            **report.routing_distribution,
        }
        manifest["phases"]["route"] = report.phase_counts["route"]
        self._write_manifest(manifest)
        logger.info("phase=route total_packs=%d %s", len(packs), report.routing_distribution)

        if self.config.dry_run:
            report.notes.append("dry_run=True; stopped after routing")
            report.budget_usage = {"online_calls_used": 0, "total_tokens_used": 0}
            self._finalize(report, packs, [], [])
            return report

        # Phase 4: online LLM (and auto) hyperedge generation
        completed_hashes, completed_edges = self._load_online_results()
        agent = OnlineLLMHyperedgeAgent(
            chat_llm=self.chat_llm,
            config=self.config,
            prompt_file=self.config.prompts_file,
            cache_dir=self._online_cache_dir(),
            review_queue=self.review_queue,
            completed_pack_hashes=set(completed_hashes),
            show_progress=getattr(self.config, "show_progress", True),
        )

        new_edges: List[EvidenceHyperedge] = []
        usage_records: List[Dict[str, Any]] = []

        def on_complete(edge: EvidenceHyperedge, usage) -> None:
            new_edges.append(edge)
            self._append_online_result(edge, usage)
            usage_records.append(usage.to_dict())

        agent.process(packs, unified_kg_index, on_complete=on_complete)

        all_edges = list(completed_edges) + new_edges
        report.budget_usage = {
            "online_calls_used": agent.online_calls_used,
            "total_tokens_used": agent.total_tokens_used,
            "max_online_work_units_per_batch": self.config.budget.max_online_work_units_per_batch,
            "max_online_work_unit_ratio": self.config.budget.max_online_work_unit_ratio,
        }
        report.phase_counts["online_llm"] = {
            "evidence_hyperedges": len(all_edges),
            "online_calls_used": agent.online_calls_used,
            "tokens_used": agent.total_tokens_used,
        }
        # Aggregate parse_status distribution.
        from collections import Counter as _Counter
        parse_counter = _Counter()
        for u in usage_records:
            parse_counter[str(u.get("parse_status") or "unknown")] += 1
        report.parse_status_distribution = dict(parse_counter)
        manifest["phases"]["online_llm"] = report.phase_counts["online_llm"]
        self._write_manifest(manifest)

        # Phase 5: write
        self._finalize(report, packs, all_edges, usage_records, summaries=summaries, unified_kg_index=unified_kg_index)
        return report

    # ------------------------------------------------------------------
    def _finalize(
        self,
        report: HyperKGRunReport,
        packs: List[CandidatePack],
        edges: List[EvidenceHyperedge],
        usage_records: List[Dict[str, Any]],
        summaries: Optional[Dict[str, Any]] = None,
        unified_kg_index: Optional[UnifiedKGIndex] = None,
    ) -> None:
        report.finished_at = time.time()
        writer = HyperKGWriterIndexerAgent(
            embedding_service=self.embedding_service,
            writer=self.config.writer,
        )
        if unified_kg_index is None:
            unified_kg_index = UnifiedKGIndex.from_path(
                self.config.unified_kg_path, embedding_service=self.embedding_service
            )
        writer.write(
            output_dir=self.config.output_dir,
            packs=packs,
            evidence_hyperedges=edges,
            unified_kg_index=unified_kg_index,
            summaries=summaries or {},
            run_report=report,
            llm_usage=usage_records,
            review_items=self.review_queue.to_list(),
        )

    # ------------------------------------------------------------------
    def _load_packs_from_jsonl(self, path: str) -> List[CandidatePack]:
        rows: List[CandidatePack] = []
        if not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                rows.append(_pack_from_dict(payload))
        return rows

    def _write_packs_jsonl(self, path: str, packs: List[CandidatePack]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in packs:
                f.write(json.dumps(to_plain(p), ensure_ascii=False, sort_keys=True) + "\n")

    def _load_online_results(self) -> tuple[set, List[EvidenceHyperedge]]:
        path = self._online_results_path()
        if not os.path.exists(path):
            return set(), []
        hashes: set = set()
        edges: List[EvidenceHyperedge] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                edge_payload = payload.get("edge") or {}
                pack_hash = (edge_payload.get("source") or {}).get("pack_hash") or payload.get("pack_hash")
                if pack_hash:
                    hashes.add(pack_hash)
                fake_pack = CandidatePack(
                    pack_id=str((edge_payload.get("source") or {}).get("pack_id") or ""),
                    representative_claim_id=str((edge_payload.get("source") or {}).get("member_claim_ids") or [""])[0:10] or "",
                    representative_claim_text="",
                    claim_type=str(edge_payload.get("claim_type") or ""),
                    member_claim_ids=list((edge_payload.get("source") or {}).get("member_claim_ids") or []),
                    entity_candidates=[],
                    triple_candidates=[],
                    summary_snippets=[],
                    local_kg_snippet={},
                    routing_reasons=[],
                    auto_decision="resumed",
                    pack_hash=pack_hash or "",
                )
                edges.append(_hyperedge_from_dict(edge_payload, fake_pack))
        return hashes, edges

    def _append_online_result(self, edge: EvidenceHyperedge, usage) -> None:
        path = self._online_results_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        row = {
            "pack_hash": edge.source.get("pack_hash"),
            "edge": edge.to_dict(),
            "usage": usage.to_dict() if hasattr(usage, "to_dict") else dict(usage),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_plain(row), ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _pack_from_dict(payload: Dict[str, Any]) -> CandidatePack:
    from .schemas import EntityCandidate, SummarySnippet, TripleCandidate

    return CandidatePack(
        pack_id=str(payload.get("pack_id") or ""),
        representative_claim_id=str(payload.get("representative_claim_id") or ""),
        representative_claim_text=str(payload.get("representative_claim_text") or ""),
        claim_type=str(payload.get("claim_type") or ""),
        member_claim_ids=list(payload.get("member_claim_ids") or []),
        entity_candidates=[EntityCandidate(**e) for e in payload.get("entity_candidates") or []],
        triple_candidates=[TripleCandidate(**t) for t in payload.get("triple_candidates") or []],
        summary_snippets=[SummarySnippet(**s) for s in payload.get("summary_snippets") or []],
        local_kg_snippet=dict(payload.get("local_kg_snippet") or {}),
        routing_reasons=list(payload.get("routing_reasons") or []),
        auto_decision=str(payload.get("auto_decision") or "auto"),
        pack_hash=str(payload.get("pack_hash") or ""),
    )
