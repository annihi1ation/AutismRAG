"""GraphStore interface + TODO Neo4j stub.

File-based storage is the default. Neo4j is intentionally NOT implemented (the
repo does not require it). The stub documents the contract for a future
graph backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..schemas.hyperedge import HyperEdge
from ..schemas.storage import IncidenceRecord


class GraphStore(ABC):
    @abstractmethod
    def upsert_hyperedges(self, hyperedges: Sequence[HyperEdge]) -> None: ...

    @abstractmethod
    def upsert_incidence(self, records: Sequence[IncidenceRecord]) -> None: ...


class Neo4jGraphStore(GraphStore):
    """TODO: not implemented. Provided as a contract stub only."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Neo4jGraphStore is a stub. File-based storage is the default; "
            "implement this only if a graph database backend is required."
        )

    def upsert_hyperedges(self, hyperedges: Sequence[HyperEdge]) -> None:  # noqa: D102
        raise NotImplementedError

    def upsert_incidence(self, records: Sequence[IncidenceRecord]) -> None:  # noqa: D102
        raise NotImplementedError
