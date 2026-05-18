"""Offline embedding modules."""

from __future__ import annotations

from .base import EmbeddingModel, HashingEmbeddingModel
from .entity_index import build_entity_index
from .hyperedge_index import build_hyperedge_index
from .section_index import build_section_index, section_embed_text
from .sentence_transformer_backend import SentenceTransformerBackend
from .similarity import cosine_sim, cosine_topk, normalize
from .vector_index import (
    EntityVectorIndex,
    HyperEdgeVectorIndex,
    SectionVectorIndex,
    VectorIndex,
)

__all__ = [
    "EmbeddingModel",
    "HashingEmbeddingModel",
    "SentenceTransformerBackend",
    "VectorIndex",
    "SectionVectorIndex",
    "EntityVectorIndex",
    "HyperEdgeVectorIndex",
    "build_section_index",
    "section_embed_text",
    "build_entity_index",
    "build_hyperedge_index",
    "cosine_sim",
    "cosine_topk",
    "normalize",
]


def build_embedding_model(config) -> EmbeddingModel:
    """Construct the configured embedding backend.

    Honors env override ``HKG_FAKE_EMBEDDINGS=1`` to force the deterministic
    offline hashing model (used by tests / no-model environments).
    """
    import os

    if os.environ.get("HKG_FAKE_EMBEDDINGS"):
        return HashingEmbeddingModel()
    return SentenceTransformerBackend(
        model_path=config.embedding_model_path,
        device=config.embedding_device,
        batch_size=config.batch_size,
        normalize=config.normalize_embeddings,
        trust_remote_code=config.trust_remote_code,
        max_internal_tokens=config.max_section_internal_tokens_for_embedding,
    )
