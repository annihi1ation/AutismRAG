"""Module 4: MinimalHyperKGDiffuser — one-hop set union, optional second-hop."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from .evidence_organizer import canonical_claim_type
from .loaders import HyperRAGContext

logger = logging.getLogger(__name__)


class MinimalHyperKGDiffuser:
    def __init__(self, ctx: HyperRAGContext):
        self.ctx = ctx

    def run(
        self,
        retrieved_candidates: Dict[str, Any],
        retrieval_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        intervention_filter = set(
            retrieval_plan.get("claim_type_filters", {}).get("intervention_evidence", [])
        )

        seed_entity_ids = self._resolve_entity_ids(
            retrieved_candidates.get("entities") or [],
            retrieval_plan.get("seed_entities") or [],
        )
        seed_edge_ids: Set[str] = {
            h["id"] for h in retrieved_candidates.get("hyperedges") or [] if h.get("id")
        }

        edge_ids: Set[str] = set(seed_edge_ids)
        chunk_ids: Set[str] = {c["id"] for c in retrieved_candidates.get("chunks") or [] if c.get("id")}
        entity_ids: Set[str] = set(seed_entity_ids)

        # one-hop expansion
        for ent_id in list(seed_entity_ids):
            edge_ids.update(self.ctx.incidence.entity_to_hyperedges.get(ent_id, set()))
        for edge_id in list(edge_ids):
            entity_ids.update(self.ctx.incidence.hyperedge_to_entities.get(edge_id, set()))
            chunk_ids.update(self.ctx.incidence.hyperedge_to_chunks.get(edge_id, set()))

        provenance: Dict[str, str] = {}
        for eid in seed_edge_ids:
            provenance[eid] = "from_retrieval"
        for eid in edge_ids - seed_edge_ids:
            provenance[eid] = "from_one_hop"

        # conditional second-hop
        if not self._has_intervention_evidence(edge_ids, intervention_filter):
            logger.info("No intervention_evidence after one-hop — running second-hop expansion.")
            new_edge_ids: Set[str] = set()
            for ent_id in list(entity_ids):
                new_edge_ids.update(self.ctx.incidence.entity_to_hyperedges.get(ent_id, set()))
            second_hop_only = new_edge_ids - edge_ids
            for eid in second_hop_only:
                provenance[eid] = "from_second_hop"
            edge_ids.update(new_edge_ids)
            for edge_id in list(second_hop_only):
                entity_ids.update(self.ctx.incidence.hyperedge_to_entities.get(edge_id, set()))
                chunk_ids.update(self.ctx.incidence.hyperedge_to_chunks.get(edge_id, set()))

        # materialise full edges
        hyperedges: List[Dict[str, Any]] = []
        for eid in edge_ids:
            edge = self.ctx.hyperedge_store.get(eid)
            if edge is None:
                continue
            edge_with_prov = dict(edge)
            edge_with_prov["_provenance"] = provenance.get(eid, "from_one_hop")
            hyperedges.append(edge_with_prov)

        return {
            "hyperedges": hyperedges,
            "entities": sorted(entity_ids),
            "chunks": sorted(chunk_ids),
            "provenance": provenance,
        }

    def _resolve_entity_ids(
        self,
        retrieved_entities: List[Dict[str, Any]],
        seed_entities: List[Dict[str, Any]],
    ) -> Set[str]:
        ids: Set[str] = {e["id"] for e in retrieved_entities if e.get("id")}
        if not seed_entities:
            return ids
        # for seeds without a retrieval hit yet, do one quick lookup against the entity index
        for seed in seed_entities:
            mention = (seed.get("mention") or "").strip()
            if not mention:
                continue
            hits = self.ctx.entity_index.search(
                namespace=self.ctx.config.entity_namespace,
                query_text=mention,
                top_k=self.ctx.config.top_k_seed_entity,
            )
            for hit in hits:
                if hit.get("id"):
                    ids.add(hit["id"])
        return ids

    def _has_intervention_evidence(
        self, edge_ids: Set[str], intervention_filter: Set[str]
    ) -> bool:
        for eid in edge_ids:
            edge = self.ctx.hyperedge_store.get(eid)
            if edge is None:
                continue
            ct = canonical_claim_type(edge.get("claim_type", ""))
            if ct in intervention_filter:
                return True
        return False
