"""Incidence links: entity_id --role--> hyperedge_id."""

from __future__ import annotations

from typing import List, Sequence

from ..schemas.hyperedge import HyperEdge
from ..schemas.storage import IncidenceRecord


def build_incidence_records(
    hyperedges: Sequence[HyperEdge],
) -> List[IncidenceRecord]:
    records: List[IncidenceRecord] = []
    for he in hyperedges:
        for ent in he.entities:
            records.append(
                IncidenceRecord(
                    he_id=he.id,
                    entity_id=ent.entity_id,
                    role=ent.role,
                    mention=ent.mention,
                    confidence=ent.confidence,
                )
            )
    return records
