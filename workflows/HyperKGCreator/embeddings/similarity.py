"""Cosine similarity helpers (numpy)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_topk(
    query: np.ndarray, corpus: np.ndarray, top_k: int
) -> List[Tuple[int, float]]:
    """Return ``[(row_index, score)]`` sorted by descending cosine."""
    if corpus.size == 0:
        return []
    q = normalize(query)[0]
    c = normalize(corpus)
    scores = c @ q
    k = min(top_k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
