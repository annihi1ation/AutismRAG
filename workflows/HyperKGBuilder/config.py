"""Configuration objects for the HyperKGBuilder workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class EmbeddingConfig:
    enabled: bool = True
    model_name: str = ""
    model_path: Optional[str] = None
    device: str = "cuda:6"
    batch_size: int = 64
    normalize: bool = True
    max_length: int = 512
    use_fp16: bool = True
    trust_remote_code: bool = False


@dataclass
class RoutingConfig:
    entity_link_confidence_auto: float = 0.90
    triple_anchor_confidence_auto: float = 0.85
    cluster_similarity_threshold: float = 0.92
    intra_cluster_certainty_threshold: float = 0.96

    max_claims_per_work_unit: int = 20
    max_candidate_entities_per_mention: int = 5
    max_candidate_triples_per_work_unit: int = 12
    max_summary_snippets_per_work_unit: int = 5

    entity_top_k: int = 10
    triple_top_k: int = 20
    summary_top_k: int = 8

    high_impact_entity_types: Tuple[str, ...] = (
        "Medication",
        "Adverse_Effect",
        "Risk_Factor",
        "Therapeutic_Approach",
        "Healthcare_Access",
        "Institutional_System",
    )

    scope_required_claim_types: Tuple[str, ...] = (
        "SEVERITY_CLASSIFICATION",
        "RISK_FACTOR",
        "DOSE_RESPONSE",
        "INTERVENTION_OUTCOME",
        "COMPARATIVE_OUTCOME",
        "COMPARATIVE_SEVERITY",
        "COMPARATIVE_RISK",
    )


@dataclass
class BudgetConfig:
    max_online_work_units_per_batch: int = 1000
    max_online_work_unit_ratio: float = 0.10
    max_online_tokens_per_work_unit: int = 4000
    max_retries_per_work_unit: int = 2
    max_prompt_chars: int = 24_000
    online_concurrency: int = 1


@dataclass
class WriterConfig:
    canonical_similarity_threshold: float = 0.88
    canonical_jaccard_threshold: float = 0.5
    relation_supports_threshold: float = 0.92


@dataclass
class HyperKGRunConfig:
    # I/O
    claims_path: str = ""
    unified_kg_path: str = ""
    summaries_path: Optional[str] = None
    output_dir: str = ""
    checkpoint_dir: Optional[str] = None
    prompts_file: Optional[str] = None

    # Embedding
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # Online LLM
    model_name: str = "deepseek/deepseek-v4-pro"
    temperature: float = 0.1
    base_url: Optional[str] = None
    request_timeout_sec: float = 120.0
    enable_audit_pass: bool = False

    # Routing / budget / writer
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)

    # Run flags
    dry_run: bool = False
    resume: bool = True
    seed: int = 0
    progress_log_interval_sec: float = 30.0
    show_progress: bool = True
