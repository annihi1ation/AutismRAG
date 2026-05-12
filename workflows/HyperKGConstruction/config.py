"""Configuration objects for HyperKG construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingConfig:
    enabled: bool = True
    model_name: str = ""
    model_path: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 64
    normalize: bool = True
    max_length: int = 512
    use_fp16: bool = True
    trust_remote_code: bool = False


@dataclass
class HyperKGConfig:
    accept_threshold: float = 0.75
    review_threshold: float = 0.55

    min_entities_per_hyperedge: int = 2
    max_claims_per_summary: int = 8
    max_local_entities_per_packet: int = 80
    max_local_triples_per_packet: int = 60

    batch_size: int = 8
    max_workers: int = 8
    max_llm_retries: int = 1

    enable_neo4j_writer: bool = False
    enable_vector_index: bool = True

    merge_similarity_threshold: float = 0.85
    auto_merge_similarity: float = 0.90
    llm_merge_similarity: float = 0.78
    max_merge_candidates: int = 10

    entity_linking_threshold: float = 0.82
    entity_linking_top_k: int = 10

    embedding_enabled: bool = True
    embedding_model_name: str = ""
    embedding_model_path: Optional[str] = None
    embedding_device: str = "cuda"
    embedding_batch_size: int = 64
    embedding_normalize: bool = True
    embedding_use_fp16: bool = True
    embedding_max_length: int = 512

    checkpoint_enabled: bool = True
    resume_enabled: bool = True
    checkpoint_dir: Optional[str] = None
    progress_enabled: bool = True
    progress_log_interval_sec: float = 30.0

    def embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            enabled=self.embedding_enabled,
            model_name=self.embedding_model_name,
            model_path=self.embedding_model_path,
            device=self.embedding_device,
            batch_size=self.embedding_batch_size,
            normalize=self.embedding_normalize,
            max_length=self.embedding_max_length,
            use_fp16=self.embedding_use_fp16,
        )
