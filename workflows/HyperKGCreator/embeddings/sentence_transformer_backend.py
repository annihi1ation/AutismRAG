"""SentenceTransformer embedding backend.

Loads a model from a local path or model name (no internet assumed when the
model is already cached / local). Long inputs are internally split into smaller
embedding pieces and mean-pooled into ONE vector per input. These internal
pieces never leak as evidence spans (plan critical span rule).
"""

from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

from .base import EmbeddingModel, _l2_normalize

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


def _split_for_embedding(text: str, max_tokens: int) -> List[str]:
    """Whitespace-token windows used ONLY for model length limits.

    The returned pieces are internal: callers pool them back to one vector and
    must never expose them as spans (they would carry parent_section_id only).
    """
    words = _WORD_RE.findall(text or "")
    if not words:
        return [""]
    if len(words) <= max_tokens:
        return [text]
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


class SentenceTransformerBackend(EmbeddingModel):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        trust_remote_code: bool = False,
        max_internal_tokens: int = 256,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.trust_remote_code = trust_remote_code
        self.max_internal_tokens = max_internal_tokens
        self._model = None
        self.dim = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # lazy import

        logger.info(
            "Loading embedding model '%s' on %s", self.model_path, self.device
        )
        self._model = SentenceTransformer(
            self.model_path,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Internal split: keep a mapping back to the originating input index so
        # pieces are pooled, never surfaced individually.
        pieces: List[str] = []
        owner: List[int] = []
        for i, text in enumerate(texts):
            for piece in _split_for_embedding(text, self.max_internal_tokens):
                pieces.append(piece)
                owner.append(i)

        raw = self._model.encode(
            pieces,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        pooled = np.zeros((len(texts), raw.shape[1]), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        for piece_vec, idx in zip(raw, owner):
            pooled[idx] += piece_vec
            counts[idx] += 1.0
        counts[counts == 0.0] = 1.0
        pooled /= counts[:, None]
        return _l2_normalize(pooled) if self.normalize else pooled
