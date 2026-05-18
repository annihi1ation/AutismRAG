"""Entity embedding index built from an entity catalog."""

from __future__ import annotations

from typing import List

from ..schemas.entity import EntityRecord
from .base import EmbeddingModel
from .vector_index import EntityVectorIndex


def build_entity_index(
    records: List[EntityRecord], model: EmbeddingModel
) -> EntityVectorIndex:
    ids = [r.entity_id for r in records]
    texts = [r.index_text() for r in records]
    vectors = model.embed_texts(texts)
    index = EntityVectorIndex()
    if ids:
        index.add(ids, vectors)
    for rec, vec in zip(records, vectors):
        rec.embedding = [float(x) for x in vec]
    return index
