#!/usr/bin/env python3
"""
Entry point for running the KG aggregation pipeline.

Loads all per-article _kg.json files from KARMA/output, merges them
into a unified KG with entity resolution, type recovery, triple
deduplication, and LLM-based conflict resolution.

Usage:
    cd /data2/leyizhao/CommTool
    python -m workflows.post_processing.run

    # Or with custom options:
    python -m workflows.post_processing.run --threshold 0.85 --no-conflicts
"""

import argparse
import logging
import os
import sys

# Add project root to path so KARMA modules are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from workflows.post_processing.pipeline import AggregationConfig, AggregationPipeline


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-article KARMA KGs into a unified Knowledge Graph"
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join(PROJECT_ROOT, "KARMA", "output"),
        help="Directory containing *_kg.json files (default: KARMA/output)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "KARMA", "output", "unified_kg.json"),
        help="Output path for the unified KG JSON",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-3-flash-preview",
        help="LLM model for entity typing and conflict resolution",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer model for entity resolution",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for entity merging (default: 0.90)",
    )
    parser.add_argument(
        "--integration-threshold",
        type=float,
        default=0.6,
        help="Minimum integration score for triples (default: 0.6)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device for embeddings (default: auto-detect, prefer cuda)",
    )
    parser.add_argument(
        "--no-conflicts",
        action="store_true",
        help="Skip LLM-based conflict resolution",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    config = AggregationConfig(
        input_dir=args.input_dir,
        output_path=args.output,
        api_key=args.api_key,
        model_name=args.model,
        embedding_model=args.embedding_model,
        similarity_threshold=args.threshold,
        integration_threshold=args.integration_threshold,
        device=args.device,
        resolve_conflicts=not args.no_conflicts,
    )

    pipeline = AggregationPipeline(config)
    unified_kg = pipeline.run()

    # Print final summary to stdout
    stats = unified_kg.get("statistics", {})
    print(f"\n{'─' * 50}")
    print(f"  Unified KG saved to: {args.output}")
    print(f"  Entities: {stats.get('entity_count', 0)}")
    print(f"  Triples:  {stats.get('triple_count', 0)}")
    print(f"  Articles: {unified_kg.get('metadata', {}).get('total_source_articles', 0)}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
