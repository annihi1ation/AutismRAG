"""Save/loadable vector index.

Backend selection: FAISS if installed, else sklearn NearestNeighbors, else
numpy cosine. Persistence is a single ``.npz`` archive (ids + vectors +
namespace) so it round-trips regardless of backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .similarity import cosine_topk, normalize

logger = logging.getLogger(__name__)


def _has_faiss() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


class VectorIndex:
    """A flat cosine index keyed by string ids."""

    namespace: str = "generic"

    def __init__(self, namespace: Optional[str] = None) -> None:
        if namespace:
            self.namespace = namespace
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._faiss_index = None

    # -- build ---------------------------------------------------------------
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D [n, dim]")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        vectors = normalize(vectors)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(str(i) for i in ids)
        self._faiss_index = None  # invalidate

    # -- query ---------------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self._vectors is None or len(self._ids) == 0 or top_k <= 0:
            return []
        if _has_faiss():
            return self._search_faiss(query_vector, top_k)
        ranked = cosine_topk(query_vector, self._vectors, top_k)
        return [(self._ids[i], score) for i, score in ranked]

    def _search_faiss(self, query_vector: np.ndarray, top_k: int):
        import faiss

        if self._faiss_index is None:
            index = faiss.IndexFlatIP(self._vectors.shape[1])
            index.add(self._vectors)
            self._faiss_index = index
        q = normalize(query_vector).astype(np.float32)
        scores, idx = self._faiss_index.search(q, min(top_k, len(self._ids)))
        out: List[Tuple[str, float]] = []
        for j, score in zip(idx[0], scores[0]):
            if 0 <= j < len(self._ids):
                out.append((self._ids[int(j)], float(score)))
        return out

    # -- persistence ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> List[str]:
        return list(self._ids)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vectors = (
            self._vectors
            if self._vectors is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        np.savez(
            path,
            ids=np.array(self._ids, dtype=object),
            vectors=vectors,
            namespace=np.array(self.namespace),
        )
        # np.savez appends .npz; normalize to the requested filename.
        npz = path.with_suffix(path.suffix + ".npz") if not str(path).endswith(".npz") else path
        if npz.exists() and npz != path:
            npz.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            namespace = str(data["namespace"])
            ids = [str(x) for x in data["ids"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        index = cls(namespace=namespace)
        index._ids = ids
        index._vectors = vectors if vectors.size else None
        return index


class SectionVectorIndex(VectorIndex):
    namespace = "section"


class EntityVectorIndex(VectorIndex):
    namespace = "entity"


class HyperEdgeVectorIndex(VectorIndex):
    namespace = "hyperedge"
