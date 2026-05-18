"""HyperEdge embedding index (built from final edge_text)."""

from __future__ import annotations

from typing import List

from ..schemas.hyperedge import HyperEdge
from .base import EmbeddingModel
from .vector_index import HyperEdgeVectorIndex


def build_hyperedge_index(
    hyperedges: List[HyperEdge], model: EmbeddingModel
) -> HyperEdgeVectorIndex:
    ids = [he.id for he in hyperedges]
    texts = [he.edge_text for he in hyperedges]
    vectors = model.embed_texts(texts)
    index = HyperEdgeVectorIndex()
    if ids:
        index.add(ids, vectors)
    return index
