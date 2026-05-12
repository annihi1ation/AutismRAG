"""Configuration for the Autism HyperKG RAG workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUILDER_RUN = (
    _REPO_ROOT / "output" / "hyperkg_builder" / "full_flash_20260506_170236"
)


@dataclass
class HyperRAGConfig:
    builder_run_dir: Path = _DEFAULT_BUILDER_RUN

    vector_indexes_dir: Optional[Path] = None
    hyperedge_jsonl: Optional[Path] = None
    incidence_jsonl: Optional[Path] = None
    summary_links_jsonl: Optional[Path] = None

    entity_namespace: str = "canonical_entity_vector_index"
    hyperedge_namespace: str = "evidence_hyperedge_vector_index"
    chunk_namespace: str = "summary_vector_index"

    # Must match the model used to build the indexes. The full_flash_20260506_170236
    # run used sentence-transformers/all-MiniLM-L6-v2 (384-dim).
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    llm_model_name: str = "anthropic/claude-sonnet-4.5"
    llm_temperature: float = 0.1

    top_k_entities: int = 10
    top_k_hyperedges: int = 15
    top_k_chunks: int = 10
    top_k_seed_entity: int = 5

    max_hyperedges_per_bucket_in_prompt: int = 8
    max_chunks_in_prompt: int = 10

    enable_llm_rerank: bool = True
    rerank_llm_model_name: str = "anthropic/claude-haiku-4.5"
    rerank_top_k_per_bucket: int = 8
    rerank_input_top_n: int = 30
    subject_only_seeding: bool = True

    prompt_file: Optional[Path] = None

    def __post_init__(self) -> None:
        run_dir = Path(self.builder_run_dir)
        if self.vector_indexes_dir is None:
            self.vector_indexes_dir = run_dir / "vector_indexes"
        if self.hyperedge_jsonl is None:
            self.hyperedge_jsonl = run_dir / "checkpoints" / "online_results.jsonl"
        if self.incidence_jsonl is None:
            self.incidence_jsonl = run_dir / "incidence_edges.jsonl"
        if self.summary_links_jsonl is None:
            self.summary_links_jsonl = run_dir / "summary_links.jsonl"
        if self.prompt_file is None:
            self.prompt_file = Path(__file__).with_name("prompts.toml")


def load_default_config() -> HyperRAGConfig:
    cfg = HyperRAGConfig()
    if os.environ.get("HYPERRAG_FORCE_CPU"):
        cfg.embedding_device = "cpu"
    model_override = os.environ.get("HYPERRAG_LLM_MODEL")
    if model_override:
        cfg.llm_model_name = model_override
    if os.environ.get("HYPERRAG_DISABLE_RERANK"):
        cfg.enable_llm_rerank = False
    if os.environ.get("HYPERRAG_DISABLE_SUBJECT_ONLY_SEED"):
        cfg.subject_only_seeding = False
    rerank_override = os.environ.get("HYPERRAG_RERANK_LLM_MODEL")
    if rerank_override:
        cfg.rerank_llm_model_name = rerank_override
    return cfg
