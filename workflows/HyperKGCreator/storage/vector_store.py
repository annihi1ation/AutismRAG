"""Persistence wrappers for the three vector indexes."""

from __future__ import annotations

from pathlib import Path

from ..embeddings.vector_index import VectorIndex


def save_index(index: VectorIndex, path: str | Path) -> Path:
    return index.save(path)


def load_index(path: str | Path) -> VectorIndex:
    return VectorIndex.load(path)
