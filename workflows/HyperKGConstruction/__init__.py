"""HyperKG construction workflow.

This package builds a textual-rich, source-grounded HyperKG from existing
segment summaries, local KGs, and a unified KG. It intentionally does not
perform ingestion, summarization, entity extraction, or relation extraction.
"""

from .config import EmbeddingConfig, HyperKGConfig
from .pipeline import HyperKGPipeline

__all__ = ["EmbeddingConfig", "HyperKGConfig", "HyperKGPipeline"]
