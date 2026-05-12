"""In-memory vector index adapter with JSONL + NPY persistence."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .embedding_service import EmbeddingService
from .utils import stable_hash

logger = logging.getLogger(__name__)


def _cosine_topk(query: np.ndarray, corpus: np.ndarray, top_k: int) -> List[tuple[int, float]]:
    if corpus.size == 0 or query.size == 0:
        return []
    if query.ndim == 1:
        query = query[None, :]
    q = query.astype(np.float32, copy=False)
    c = corpus.astype(np.float32, copy=False)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    c_norm = np.linalg.norm(c, axis=1, keepdims=True)
    q = q / np.maximum(q_norm, 1e-12)
    c = c / np.maximum(c_norm, 1e-12)
    sims = (c @ q.T).squeeze(-1)
    k = min(max(top_k, 0), sims.shape[0])
    if k == 0:
        return []
    idx = np.argpartition(-sims, k - 1)[:k]
    ordered = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in ordered]


@dataclass
class VectorIndexAdapter:
    embedding_service: Optional[EmbeddingService] = None
    records: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)

    def add(
        self,
        namespace: str,
        ids: List[str],
        texts: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if vectors.shape[0] != len(ids):
            raise ValueError("vectors row count must match ids")
        metadata = metadata or [{} for _ in ids]
        if len(metadata) != len(ids):
            raise ValueError("metadata length must match ids")

        existing_records = self.records.setdefault(namespace, [])
        existing_vectors = self.vectors.get(namespace)
        by_id = {r["id"]: i for i, r in enumerate(existing_records)}

        if existing_vectors is None:
            existing_vectors = np.zeros((0, vectors.shape[1]), dtype=np.float32)

        for row_idx, object_id in enumerate(ids):
            record = {
                "id": object_id,
                "text": texts[row_idx],
                "metadata": metadata[row_idx],
            }
            vector = vectors[row_idx : row_idx + 1].astype(np.float32, copy=False)
            if object_id in by_id:
                pos = by_id[object_id]
                existing_records[pos] = record
                existing_vectors[pos : pos + 1] = vector
            else:
                existing_records.append(record)
                existing_vectors = np.vstack([existing_vectors, vector])

        self.vectors[namespace] = existing_vectors.astype(np.float32, copy=False)

    def search(
        self,
        namespace: str,
        query_text: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if namespace not in self.records:
            return []
        if query_vector is None:
            if query_text is None:
                raise ValueError("Either query_text or query_vector is required")
            if self.embedding_service is None:
                raise RuntimeError("query_text search requires an embedding_service")
            query_vector = self.embedding_service.encode_one(query_text)

        records = self.records[namespace]
        vectors = self.vectors.get(namespace)
        if vectors is None or vectors.size == 0:
            return []

        candidate_indices = list(range(len(records)))
        if filters:
            candidate_indices = [
                i
                for i, record in enumerate(records)
                if all(record.get("metadata", {}).get(k) == v for k, v in filters.items())
            ]
        if not candidate_indices:
            return []

        subset = vectors[candidate_indices]
        hits = _cosine_topk(query_vector, subset, top_k)
        out: List[Dict[str, Any]] = []
        for subset_idx, score in hits:
            record_idx = candidate_indices[subset_idx]
            record = dict(records[record_idx])
            record["score"] = score
            out.append(record)
        return out

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        manifest: Dict[str, Any] = {"namespaces": {}}
        for namespace, records in sorted(self.records.items()):
            safe_ns = namespace.replace("/", "_")
            index_path = os.path.join(output_dir, f"{safe_ns}.index")
            metadata_path = os.path.join(output_dir, f"{safe_ns}.metadata.jsonl")
            vectors_path = os.path.join(output_dir, f"{safe_ns}.vectors.npy")
            with open(metadata_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            np.save(vectors_path, self.vectors.get(namespace, np.zeros((0, 1), dtype=np.float32)))
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "namespace": namespace,
                        "records": len(records),
                        "metadata_file": os.path.basename(metadata_path),
                        "vectors_file": os.path.basename(vectors_path),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            manifest["namespaces"][namespace] = {
                "records": len(records),
                "index_file": os.path.basename(index_path),
                "metadata_file": os.path.basename(metadata_path),
                "vectors_file": os.path.basename(vectors_path),
            }
        manifest["manifest_id"] = stable_hash(manifest)
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)


def build_vector_index(embedding_service: Optional[EmbeddingService] = None) -> VectorIndexAdapter:
    return VectorIndexAdapter(embedding_service=embedding_service)
