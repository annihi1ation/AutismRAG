"""Embedding semantic routing + non-LLM candidate preparation."""

from __future__ import annotations

from .candidate_builder import build_agent1_input
from .entity_retrieval import (
    NodeText,
    build_entity_index,
    extract_mentions,
    load_entity_catalog,
    pto_node_texts,
    retrieve_entity_candidates,
)
from .semantic_router import route_sections

__all__ = [
    "route_sections",
    "build_agent1_input",
    "load_entity_catalog",
    "build_entity_index",
    "extract_mentions",
    "retrieve_entity_candidates",
    "pto_node_texts",
    "NodeText",
]
