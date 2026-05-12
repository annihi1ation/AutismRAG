"""
AggregationPipeline – orchestrates the full workflow for merging
per-article Knowledge Graphs into a single unified KG.

Stages:
  1. Load all per-article _kg.json files
  2. Classify entities via LLM (EntityTyper)
  3. Resolve entities via GPU embeddings (EntityResolver)
  4. Remap + merge + deduplicate triples (TripleMerger)
  5. Detect & resolve contradictions via LLM (ConflictResolver)
  6. Export unified_kg.json (Exporter)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from openai import OpenAI

from .loader import load_article_kgs, collect_all_entities, collect_all_triples
from .entity_typer import EntityTyper
from .entity_resolver import EntityResolver
from .triple_merger import TripleMerger
from .conflict_resolver import ConflictResolver
from .exporter import export_unified_kg

logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for the aggregation pipeline."""

    # ── Paths ───────────────────────────────────────────────────────────
    input_dir: str = ""          # KARMA output directory with _kg.json files
    output_path: str = ""        # Path for unified_kg.json

    # ── LLM ─────────────────────────────────────────────────────────────
    api_key: str = ""
    api_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "google/gemini-3-flash-preview"
    temperature: float = 0.05

    # ── Entity Typing ───────────────────────────────────────────────────
    entity_typing_batch_size: int = 40

    # ── Entity Resolution ───────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    similarity_threshold: float = 0.90
    embedding_batch_size: int = 256
    device: Optional[str] = None  # None = auto-detect

    # ── Triple Merging ──────────────────────────────────────────────────
    integration_threshold: float = 0.6
    multi_source_bonus: float = 0.03
    max_bonus: float = 0.10

    # ── Conflict Resolution ─────────────────────────────────────────────
    resolve_conflicts: bool = True

    # ── Export ──────────────────────────────────────────────────────────
    exclude_rejected: bool = True


class AggregationPipeline:
    """
    End-to-end pipeline for aggregating per-article KGs.

    Usage::

        config = AggregationConfig(
            input_dir="KARMA/output",
            output_path="KARMA/output/unified_kg/unified_kg.json",
            api_key="sk-or-v1-...",
        )
        pipeline = AggregationPipeline(config)
        result = pipeline.run()
    """

    def __init__(self, config: AggregationConfig):
        self.config = config
        self._validate_config()

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url,
        )

    def _validate_config(self):
        if not self.config.input_dir:
            raise ValueError("input_dir must be set")
        if not self.config.output_path:
            raise ValueError("output_path must be set")
        if not self.config.api_key:
            raise ValueError("api_key must be set")

    def run(self) -> Dict:
        """
        Run the full aggregation pipeline.

        Returns
        -------
        dict
            The unified KG dictionary.
        """
        t0 = time.time()
        logger.info("=" * 70)
        logger.info("KG AGGREGATION PIPELINE START")
        logger.info("=" * 70)

        # ── Stage 1: Load ───────────────────────────────────────────────
        logger.info("\n[Stage 1/6] Loading per-article KGs from %s", self.config.input_dir)
        t1 = time.time()
        article_kgs = load_article_kgs(self.config.input_dir)
        if not article_kgs:
            raise RuntimeError(f"No KG files found in {self.config.input_dir}")

        entity_sources = collect_all_entities(article_kgs)
        all_triples = collect_all_triples(article_kgs)
        article_ids = [a.article_id for a in article_kgs]

        logger.info(
            "  Loaded %d articles: %d unique entities, %d total triples (%.1fs)",
            len(article_kgs),
            len(entity_sources),
            len(all_triples),
            time.time() - t1,
        )

        # ── Stage 2: Entity Typing ──────────────────────────────────────
        logger.info("\n[Stage 2/6] Recovering entity types via LLM (%s)", self.config.model_name)
        t2 = time.time()
        typer = EntityTyper(
            client=self.client,
            model_name=self.config.model_name,
            batch_size=self.config.entity_typing_batch_size,
            temperature=self.config.temperature,
        )
        typed_entities = typer.classify_entities(entity_sources)
        logger.info("  Entity typing done (%.1fs)", time.time() - t2)

        # Log type distribution
        from collections import Counter
        type_dist = Counter(e.entity_type for e in typed_entities.values())
        for etype, count in type_dist.most_common():
            logger.info("    %s: %d", etype, count)

        # ── Stage 3: Entity Resolution ──────────────────────────────────
        logger.info(
            "\n[Stage 3/6] Resolving entities via GPU embeddings (%s, threshold=%.2f)",
            self.config.embedding_model,
            self.config.similarity_threshold,
        )
        t3 = time.time()
        resolver = EntityResolver(
            model_name=self.config.embedding_model,
            similarity_threshold=self.config.similarity_threshold,
            device=self.config.device,
            batch_size=self.config.embedding_batch_size,
        )
        canonical_entities, name_mapping = resolver.resolve(typed_entities)
        logger.info("  Entity resolution done (%.1fs)", time.time() - t3)

        # ── Stage 4: Triple Merging ─────────────────────────────────────
        logger.info("\n[Stage 4/6] Merging + deduplicating triples")
        t4 = time.time()
        merger = TripleMerger(
            name_mapping=name_mapping,
            integration_threshold=self.config.integration_threshold,
            multi_source_bonus=self.config.multi_source_bonus,
            max_bonus=self.config.max_bonus,
        )
        merged_triples = merger.merge(all_triples)
        logger.info(
            "  Triple merging done: %d merged triples (%.1fs)",
            len(merged_triples),
            time.time() - t4,
        )

        # ── Stage 5: Conflict Resolution ────────────────────────────────
        if self.config.resolve_conflicts:
            logger.info("\n[Stage 5/6] Resolving contradictions via LLM")
            t5 = time.time()
            conflict_resolver = ConflictResolver(
                client=self.client,
                model_name=self.config.model_name,
                output_dir=self.config.input_dir,
                temperature=self.config.temperature,
            )
            merged_triples = conflict_resolver.resolve(merged_triples)
            logger.info("  Conflict resolution done (%.1fs)", time.time() - t5)
        else:
            logger.info("\n[Stage 5/6] Conflict resolution SKIPPED (disabled)")

        # ── Stage 6: Export ─────────────────────────────────────────────
        logger.info("\n[Stage 6/6] Exporting unified KG to %s", self.config.output_path)
        t6 = time.time()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config.output_path) or ".", exist_ok=True)

        unified_kg = export_unified_kg(
            canonical_entities=canonical_entities,
            merged_triples=merged_triples,
            source_article_ids=article_ids,
            output_path=self.config.output_path,
            similarity_threshold=self.config.similarity_threshold,
            integration_threshold=self.config.integration_threshold,
            exclude_rejected=self.config.exclude_rejected,
        )
        logger.info("  Export done (%.1fs)", time.time() - t6)

        # ── Summary ─────────────────────────────────────────────────────
        total_time = time.time() - t0
        stats = unified_kg.get("statistics", {})
        logger.info("\n" + "=" * 70)
        logger.info("KG AGGREGATION PIPELINE COMPLETE (%.1fs)", total_time)
        logger.info("=" * 70)
        logger.info("  Source articles:      %d", len(article_ids))
        logger.info("  Canonical entities:   %d", stats.get("entity_count", 0))
        logger.info("  Merged triples:       %d", stats.get("triple_count", 0))
        logger.info("  Unique relations:     %d", stats.get("unique_relations", 0))
        logger.info("  Multi-source triples: %d", stats.get("multi_source_triples", 0))
        logger.info("  Avg confidence:       %.3f", stats.get("avg_confidence", 0))
        conflict_summary = stats.get("conflict_summary", {})
        logger.info(
            "  Conflicts: %d consistent, %d resolved, %d unresolved, %d rejected",
            conflict_summary.get("consistent", 0),
            conflict_summary.get("resolved_kept", 0),
            conflict_summary.get("unresolved", 0),
            conflict_summary.get("rejected", 0),
        )
        logger.info("  Output: %s", self.config.output_path)

        return unified_kg
