"""Optional dense embedding support for HyperKG construction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .config import EmbeddingConfig, HyperKGConfig
from .schemas import CanonicalHyperedge, EvidenceHyperedge
from .utils import compact_json

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Lazy SentenceTransformer wrapper shared by all agents."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model_name = config.model_path or config.model_name
        self._model = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.model_name)

    def _load(self) -> None:
        if not self.enabled:
            raise RuntimeError("EmbeddingService is disabled or missing model_name/model_path.")
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        logger.info("Loading embedding model '%s' on %s", self.model_name, self.config.device)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
        )

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Return a float32 matrix with shape ``(len(texts), dim)``."""
        self._load()
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)

    def encode_one(self, text: str) -> np.ndarray:
        vectors = self.encode_texts([text])
        return vectors[0]


def build_embedding_service(config: HyperKGConfig) -> Optional[EmbeddingService]:
    emb_cfg = config.embedding_config()
    if not emb_cfg.enabled:
        return None
    if not (emb_cfg.model_name or emb_cfg.model_path):
        logger.info("Embedding enabled but no model configured; using deterministic mode.")
        return None
    return EmbeddingService(emb_cfg)


def entity_embedding_text(entity: Dict[str, Any]) -> str:
    name = entity.get("canonical_name") or entity.get("name") or entity.get("entity_id") or ""
    entity_type = entity.get("entity_type") or entity.get("type") or ""
    aliases = ", ".join(str(a) for a in entity.get("aliases", []) or [])
    description = entity.get("description") or entity.get("definition") or ""
    return f"{name}. Type: {entity_type}. Aliases: {aliases}. Description: {description}."


def evidence_hyperedge_embedding_text(hyperedge: EvidenceHyperedge) -> str:
    entities = "; ".join(
        f"{e.mention} ({e.role}, {e.entity_type or 'unknown'})" for e in hyperedge.entities
    )
    return "\n".join(
        [
            f"Claim: {hyperedge.claim_text}",
            f"Type: {hyperedge.claim_type}",
            f"Entities: {entities}",
            "Qualifiers: " + compact_json(hyperedge.qualifiers),
        ]
    )


def canonical_hyperedge_embedding_text(hyperedge: CanonicalHyperedge) -> str:
    return "\n".join(
        [
            f"Canonical claim: {hyperedge.canonical_claim}",
            f"Type: {hyperedge.claim_type}",
            "Core entities: " + ", ".join(hyperedge.core_entity_ids),
            "Scope: " + compact_json(hyperedge.scope_summary),
            "Qualifiers: " + compact_json(hyperedge.qualifier_summary),
        ]
    )


def summary_embedding_text(packet: Any) -> str:
    metadata = getattr(packet, "metadata", {}) or {}
    section = metadata.get("section") or metadata.get("section_title") or ""
    return "\n".join(
        [
            f"Summary: {getattr(packet, 'summary_text', '')}",
            f"Section: {section}",
            "Metadata: " + compact_json(metadata),
        ]
    )
