"""
Loader – reads all per-article _kg.json files from the KARMA output
directory and attaches provenance (source article id) to every entity
and triple.
"""

import json
import glob
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class SourceTriple:
    """A triple with provenance tracking."""
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    relevance: float = 0.0
    clarity: float = 0.0
    source_article: str = ""


@dataclass
class ArticleKG:
    """A per-article knowledge graph loaded from disk."""
    article_id: str
    entities: List[str]
    triples: List[SourceTriple]
    metadata: Dict
    statistics: Dict


def _article_id_from_path(path: str) -> str:
    """
    Derive a short human-readable article identifier from the filename.
    Example: 'Alarifi-2024-Interventions addressing challeng_kg.json'
             -> 'Alarifi-2024-Interventions addressing challeng'
    """
    basename = os.path.basename(path)
    if basename.endswith("_kg.json"):
        return basename[: -len("_kg.json")]
    return os.path.splitext(basename)[0]


def load_article_kgs(
    output_dir: str,
    pattern: str = "*_kg.json",
) -> List[ArticleKG]:
    """
    Load all per-article KG JSON files from *output_dir*.

    Parameters
    ----------
    output_dir : str
        Path to the KARMA output directory containing *_kg.json* files.
    pattern : str
        Glob pattern to match KG files (default ``*_kg.json``).

    Returns
    -------
    list[ArticleKG]
        One ``ArticleKG`` per file, sorted by article_id.
    """
    paths = sorted(glob.glob(os.path.join(output_dir, pattern)))
    # Exclude the unified KG itself to avoid reprocessing
    paths = [p for p in paths if not os.path.basename(p).startswith("unified_")]
    if not paths:
        logger.warning("No _kg.json files found in %s", output_dir)
        return []

    article_kgs: List[ArticleKG] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            article_id = _article_id_from_path(path)
            entities = data.get("entities", [])
            raw_triples = data.get("triples", [])
            metadata = data.get("metadata", {})
            statistics = data.get("statistics", {})

            triples = []
            for t in raw_triples:
                triples.append(
                    SourceTriple(
                        head=t["head"],
                        relation=t["relation"],
                        tail=t["tail"],
                        confidence=t.get("confidence", 0.0),
                        relevance=t.get("relevance", 0.0),
                        clarity=t.get("clarity", 0.0),
                        source_article=article_id,
                    )
                )

            article_kgs.append(
                ArticleKG(
                    article_id=article_id,
                    entities=entities,
                    triples=triples,
                    metadata=metadata,
                    statistics=statistics,
                )
            )
            logger.info(
                "Loaded %s: %d entities, %d triples",
                article_id,
                len(entities),
                len(triples),
            )
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)

    logger.info(
        "Loaded %d article KGs (%d total entities, %d total triples)",
        len(article_kgs),
        sum(len(a.entities) for a in article_kgs),
        sum(len(a.triples) for a in article_kgs),
    )
    return article_kgs


def load_summaries(output_dir: str, article_id: str) -> List[str]:
    """
    Load the _summaries.json for a specific article.

    Returns an empty list if the file does not exist.
    """
    path = os.path.join(output_dir, f"{article_id}_summaries.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load summaries for %s: %s", article_id, e)
        return []


def collect_all_entities(article_kgs: List[ArticleKG]) -> Dict[str, List[str]]:
    """
    Collect all unique entity strings and map each to its source articles.

    Returns
    -------
    dict[str, list[str]]
        Mapping from entity-string to list of article_ids where it appears.
    """
    entity_sources: Dict[str, List[str]] = {}
    for akg in article_kgs:
        for ent in akg.entities:
            entity_sources.setdefault(ent, []).append(akg.article_id)
    return entity_sources


def collect_all_triples(article_kgs: List[ArticleKG]) -> List[SourceTriple]:
    """Flatten all triples from all articles into a single list."""
    all_triples: List[SourceTriple] = []
    for akg in article_kgs:
        all_triples.extend(akg.triples)
    return all_triples
