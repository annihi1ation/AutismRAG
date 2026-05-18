"""Offline embedding abstraction.

No internet dependency. ``HashingEmbeddingModel`` is a deterministic backend
used for offline tests so the whole pipeline is reproducible without a model
download.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class EmbeddingModel(ABC):
    """Embeds text into L2-normalized float32 vectors."""

    dim: int

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return shape ``[len(texts), dim]`` float32."""

    def embed_text(self, text: str) -> np.ndarray:
        """Return shape ``[dim]`` float32."""
        return self.embed_texts([text])[0]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


class HashingEmbeddingModel(EmbeddingModel):
    """Deterministic hashing bag-of-words embedding (offline / test backend)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts])
        return _l2_normalize(matrix)
