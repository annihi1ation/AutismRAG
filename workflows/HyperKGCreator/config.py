"""Configuration for the HyperKGCreator pipeline.

Sibling-consistent pattern: a dataclass plus ``load_default_config`` env
overrides, with an optional JSON/YAML override file (``load_config``).

No long clinical LLM prompts live here. ``route_definitions`` are short
semantic labels / prototype queries only.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Allowed entity types mirror KARMA/entity_type_grouping.md exactly. Do not
# invent ontology types. Extension is only allowed when config opts in.
PATIENT_ENTITY_TYPES: List[str] = [
    "GENE",
    "NEUROBIOLOGY",
    "DEVELOPMENTAL_STAGE",
    "DISORDER",
    "CLINICAL_SEVERITY",
    "TRAUMA",
    "SYMPTOM_BEHAVIOR",
    "SYMPTOM_EMOTION",
    "FUNCTIONAL_ABILITY",
    "SOCIAL_ENVIRONMENT",
    "INTERSECTIONAL_IDENTITY",
]
TREATMENT_ENTITY_TYPES: List[str] = [
    "MEDICATION",
    "THERAPEUTIC_APPROACH",
    "EDUCATIONAL_APPROACH",
    "HEALTHCARE_ACCESS",
]
BRIDGE_ENTITY_TYPES: List[str] = [
    "ASSESSMENT_TOOL",
    "DIAGNOSTIC_FRAMEWORK",
    "PERSON_ROLE",
]
ALLOWED_ENTITY_TYPES: List[str] = (
    PATIENT_ENTITY_TYPES + TREATMENT_ENTITY_TYPES + BRIDGE_ENTITY_TYPES
)


def _default_output_paths() -> Dict[str, str]:
    return {
        "root": "output/hyperkg_creator",
        "article_documents": "article_documents",
        "pto_graphs": "pto_graphs",
        "hyperedges": "hyperedges.jsonl",
        "entities": "entities.jsonl",
        "incidence_edges": "incidence_edges.jsonl",
        "validation_reports": "validation_reports.jsonl",
        "verification_results": "verification_results.jsonl",
        "section_index": "section_embeddings.index",
        "entity_index": "entity_embeddings.index",
        "hyperedge_index": "hyperedge_embeddings.index",
    }


def _default_route_definitions() -> Dict[str, List[str]]:
    """Short semantic labels per route. NOT clinical extraction prompts."""
    return {
        "patient": [
            "study participants",
            "patient characteristics",
            "sample description",
            "diagnosis and severity",
        ],
        "treatment": [
            "intervention",
            "treatment procedure",
            "medication",
            "therapy approach",
        ],
        "outcome": [
            "outcome measures",
            "results",
            "treatment response",
            "adverse events",
        ],
        "comparator": [
            "control group",
            "comparison condition",
            "baseline",
        ],
        "study_design": [
            "study design",
            "methods",
            "procedure",
        ],
    }


@dataclass
class HKGConfig:
    """Pipeline configuration. All paths are resolved lazily by the caller."""

    # Embedding backend
    embedding_model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    batch_size: int = 16
    normalize_embeddings: bool = True
    trust_remote_code: bool = False
    max_section_internal_tokens_for_embedding: int = 256

    # Routing / retrieval
    section_router_top_k: int = 6
    entity_candidate_top_k: int = 5
    candidate_margin_threshold: float = 0.05

    # Thresholds
    support_score_threshold: float = 0.30
    risk_score_threshold: float = 0.50
    confidence_threshold: float = 0.55
    table_extraction_confidence_threshold: float = 0.50
    section_dispersion_threshold: int = 4

    # Ontology
    allowed_entity_types: List[str] = field(default_factory=lambda: list(ALLOWED_ENTITY_TYPES))
    allow_entity_type_extension: bool = False

    # Routing labels (short labels only, never long prompts)
    route_definitions: Dict[str, List[str]] = field(default_factory=_default_route_definitions)

    # Output layout
    output_paths: Dict[str, str] = field(default_factory=_default_output_paths)

    # LLM
    llm_model_name: str = "anthropic/claude-sonnet-4.5"
    llm_temperature: float = 0.1

    def output_root(self, out_dir: Path) -> Path:
        return Path(out_dir)

    def resolved_output(self, out_dir: Path, key: str) -> Path:
        return Path(out_dir) / self.output_paths[key]


def load_config(path: Optional[str | Path] = None) -> HKGConfig:
    """Load config, optionally overlaying a JSON or YAML file."""
    cfg = load_default_config()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = _read_structured(p)
    for key, value in (raw or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logger.warning("Ignoring unknown config key '%s'", key)
    return cfg


def _read_structured(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text) or {}
        except ImportError:
            logger.warning("PyYAML not installed; attempting JSON parse of %s", path)
    return json.loads(text) if text.strip() else {}


def load_default_config() -> HKGConfig:
    """Default config plus environment-variable overrides."""
    cfg = HKGConfig()
    env = os.environ
    if env.get("HKG_FORCE_CPU"):
        cfg.embedding_device = "cpu"
    if env.get("HKG_EMBEDDING_MODEL"):
        cfg.embedding_model_path = env["HKG_EMBEDDING_MODEL"]
    if env.get("HKG_EMBEDDING_DEVICE"):
        cfg.embedding_device = env["HKG_EMBEDDING_DEVICE"]
    if env.get("HKG_LLM_MODEL"):
        cfg.llm_model_name = env["HKG_LLM_MODEL"]
    return cfg
