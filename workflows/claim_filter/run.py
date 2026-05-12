"""CLI entry point for the claim top-1 selector."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from workflows.HyperKGConstruction.llm import (
    build_openai_client,
    get_openrouter_api_key,
    get_openrouter_base_url,
)

from .pipeline import run

DEFAULT_INPUT = Path(
    "output/hyperkg_gpt54mini_full_gpu6/checkpoints/claim_split_results.jsonl"
)
DEFAULT_OUTPUT = Path(
    "output/hyperkg_gpt54mini_full_gpu6/claim_top1/results.jsonl"
)
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_WORKERS = 16
DEFAULT_RPM = 250


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-1 claim selector for downstream RAG.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--rpm", type=int, default=DEFAULT_RPM, help="Sliding-window cap on requests per minute. 0 disables.")
    p.add_argument("--limit", type=int, default=None, help="Process only first N records.")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--env-file", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = get_openrouter_api_key(args.api_key, args.env_file)
    base_url = get_openrouter_base_url(args.base_url, args.env_file)
    client = build_openai_client(api_key, base_url)

    run(
        input_path=args.input,
        output_path=args.output,
        client=client,
        model=args.model,
        workers=args.workers,
        limit=args.limit,
        rpm=args.rpm if args.rpm and args.rpm > 0 else None,
    )


if __name__ == "__main__":
    main()
