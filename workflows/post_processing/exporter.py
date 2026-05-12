"""
Exporter – assembles the final unified KG JSON file from resolved
entities and merged triples, with full provenance metadata.
"""

import json
import logging
from collections import Counter
from datetime import datetime
from typing import Dict, List

from .entity_resolver import CanonicalEntity
from .triple_merger import MergedTriple

logger = logging.getLogger(__name__)


def export_unified_kg(
    canonical_entities: Dict[str, CanonicalEntity],
    merged_triples: List[MergedTriple],
    source_article_ids: List[str],
    output_path: str,
    similarity_threshold: float = 0.82,
    integration_threshold: float = 0.6,
    exclude_rejected: bool = True,
) -> Dict:
    """
    Write the unified KG to a JSON file.

    Parameters
    ----------
    canonical_entities : dict[str, CanonicalEntity]
        Resolved entities from EntityResolver.
    merged_triples : list[MergedTriple]
        Deduplicated triples from TripleMerger + ConflictResolver.
    source_article_ids : list[str]
        All article ids that contributed to this KG.
    output_path : str
        Path to write the JSON file.
    similarity_threshold : float
        The threshold used for entity resolution (recorded in metadata).
    integration_threshold : float
        The threshold used for quality filtering (recorded in metadata).
    exclude_rejected : bool
        If True, exclude triples with conflict_status == "resolved_rejected".

    Returns
    -------
    dict
        The unified KG dictionary (same object written to disk).
    """
    # ── Filter triples ──────────────────────────────────────────────────
    if exclude_rejected:
        active_triples = [
            t for t in merged_triples if t.conflict_status != "resolved_rejected"
        ]
    else:
        active_triples = merged_triples

    # ── Build entity list ───────────────────────────────────────────────
    # Only include entities that actually appear in active triples
    used_entities = set()
    for t in active_triples:
        used_entities.add(t.head)
        used_entities.add(t.tail)

    entities_list = []
    for name in sorted(used_entities):
        ce = canonical_entities.get(name)
        if ce:
            entities_list.append({
                "canonical_name": ce.canonical_name,
                "entity_type": ce.entity_type,
                "normalized_id": ce.normalized_id,
                "aliases": ce.aliases,
                "source_articles": ce.source_articles,
                "mention_count": ce.mention_count,
            })
        else:
            entities_list.append({
                "canonical_name": name,
                "entity_type": "Other",
                "normalized_id": "N/A",
                "aliases": [],
                "source_articles": [],
                "mention_count": 0,
            })

    # ── Build triples list ──────────────────────────────────────────────
    triples_list = [t.to_dict() for t in active_triples]

    # ── Statistics ──────────────────────────────────────────────────────
    relation_counts = Counter(t.relation for t in active_triples)
    type_counts = Counter(e["entity_type"] for e in entities_list)
    multi_source = [t for t in active_triples if t.evidence_count > 1]

    statistics = {
        "entity_count": len(entities_list),
        "triple_count": len(triples_list),
        "unique_relations": len(relation_counts),
        "relation_distribution": dict(relation_counts.most_common()),
        "entity_type_distribution": dict(type_counts.most_common()),
        "avg_confidence": (
            round(sum(t.confidence for t in active_triples) / max(len(active_triples), 1), 4)
        ),
        "avg_evidence_count": (
            round(sum(t.evidence_count for t in active_triples) / max(len(active_triples), 1), 2)
        ),
        "multi_source_triples": len(multi_source),
        "single_source_triples": len(active_triples) - len(multi_source),
        "conflict_summary": {
            "consistent": sum(1 for t in active_triples if t.conflict_status == "consistent"),
            "resolved_kept": sum(1 for t in active_triples if t.conflict_status == "resolved_kept"),
            "unresolved": sum(1 for t in active_triples if t.conflict_status == "unresolved"),
            "rejected": sum(1 for t in merged_triples if t.conflict_status == "resolved_rejected"),
        },
    }

    # ── Metadata ────────────────────────────────────────────────────────
    metadata = {
        "description": "Unified Knowledge Graph aggregated from per-article KARMA KGs",
        "domain": "Intellectual Disability (ID) – challenging behaviors, interventions, outcomes",
        "total_source_articles": len(source_article_ids),
        "source_articles": sorted(source_article_ids),
        "aggregation_date": datetime.now().isoformat(),
        "entity_resolution_model": "BAAI/bge-large-en-v1.5",
        "entity_resolution_threshold": similarity_threshold,
        "integration_threshold": integration_threshold,
        "pipeline_version": "0.1.0",
    }

    # ── Assemble ────────────────────────────────────────────────────────
    unified_kg = {
        "entities": entities_list,
        "triples": triples_list,
        "metadata": metadata,
        "statistics": statistics,
    }

    # ── Write ───────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unified_kg, f, indent=2, ensure_ascii=False)

    logger.info(
        "Unified KG written to %s: %d entities, %d triples from %d articles",
        output_path,
        len(entities_list),
        len(triples_list),
        len(source_article_ids),
    )
    return unified_kg
