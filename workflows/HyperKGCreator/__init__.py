"""HyperKGCreator: section-span based Hyper Knowledge Graph construction.

A compact pipeline that builds an intellectual-disability treatment-response
Hyper Knowledge Graph from raw journal-article PDFs.

Evidence spans are SECTION-level only. Offline embeddings own routing, entity
candidate retrieval, support scoring and risk gating. Exactly three LLM stages
exist (PTO builder, HyperEdge/entity selector, conditional verifier) and all
LLM prompt files are owner-supplied placeholders.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
