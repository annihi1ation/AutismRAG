"""CLI entry point for HyperKGBuilder."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

from workflows.HyperKGConstruction.embedding_service import EmbeddingService
from workflows.HyperKGConstruction.llm import (
    ChatLLM,
    OPENROUTER_API_KEY_ENV,
    build_openrouter_client,
    load_env_file,
)

from .config import EmbeddingConfig, HyperKGRunConfig
from .pipeline import HyperKGBuilderPipeline


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="HyperKGBuilder")
    parser.add_argument("--claims-path", required=True, help="Path to collected claims JSONL")
    parser.add_argument("--unified-kg-path", required=True, help="Path to unified KG JSON")
    parser.add_argument("--summaries-path", default=None, help="Optional path to segment summaries")
    parser.add_argument("--output-dir", required=True, help="Output directory for the run")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory (default: <output>/checkpoints)")
    parser.add_argument("--prompts-file", default=None, help="Override prompts.toml path")

    parser.add_argument("--model-name", default="deepseek/deepseek-v4-pro", help="OpenRouter model id")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--api-key", default=None, help="OpenRouter API key (defaults to .env)")
    parser.add_argument("--base-url", default=None, help="Override OpenRouter base URL")
    parser.add_argument("--enable-audit-pass", action="store_true")

    parser.add_argument("--embedding-model-name", default="", help="HF id or local path for embedding model")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--embedding-device", default="cuda:6")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--no-embedding-normalize", action="store_true")

    parser.add_argument("--max-online-work-units-per-batch", type=int, default=1000)
    parser.add_argument("--max-online-work-unit-ratio", type=float, default=0.10)
    parser.add_argument("--max-online-tokens-per-work-unit", type=int, default=4000)
    parser.add_argument("--max-retries-per-work-unit", type=int, default=2)
    parser.add_argument(
        "--online-concurrency",
        type=int,
        default=1,
        help="Parallel Online LLM calls (1 = sequential). OpenRouter is IO-bound; 8-32 is safe.",
    )

    parser.add_argument("--dry-run", action="store_true", help="Skip Online LLM phase")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _configure_logging(output_dir: str, level: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "hyperkg_builder.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt))
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _build_chat_llm(args, dry_run: bool) -> Optional[ChatLLM]:
    if dry_run:
        return None
    load_env_file()
    if not args.api_key and not os.environ.get(OPENROUTER_API_KEY_ENV):
        raise SystemExit(
            f"{OPENROUTER_API_KEY_ENV} is not set. Add it to .env or pass --api-key."
        )
    client = build_openrouter_client(api_key=args.api_key, base_url=args.base_url)
    return ChatLLM(
        client=client,
        model_name=args.model_name,
        temperature=args.temperature,
        max_retries=args.max_retries_per_work_unit,
    )


def _build_embedding_service(args) -> Optional[EmbeddingService]:
    if not (args.embedding_model_name or args.embedding_model_path):
        logging.getLogger(__name__).warning(
            "No embedding model configured; vector retrieval will be disabled."
        )
        return None
    cfg = EmbeddingConfig(
        enabled=True,
        model_name=args.embedding_model_name,
        model_path=args.embedding_model_path,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        normalize=not args.no_embedding_normalize,
    )
    return EmbeddingService(cfg)


_DEFAULT_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.toml")


def main(argv: Optional[list[str]] = None) -> HyperKGRunConfig:
    args = _build_argparser().parse_args(argv)
    _configure_logging(args.output_dir, args.log_level)

    config = HyperKGRunConfig(
        claims_path=args.claims_path,
        unified_kg_path=args.unified_kg_path,
        summaries_path=args.summaries_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        prompts_file=args.prompts_file or _DEFAULT_PROMPTS_FILE,
        embedding=EmbeddingConfig(
            enabled=bool(args.embedding_model_name or args.embedding_model_path),
            model_name=args.embedding_model_name,
            model_path=args.embedding_model_path,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
            normalize=not args.no_embedding_normalize,
        ),
        model_name=args.model_name,
        temperature=args.temperature,
        base_url=args.base_url,
        enable_audit_pass=args.enable_audit_pass,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    config.budget.max_online_work_units_per_batch = args.max_online_work_units_per_batch
    config.budget.max_online_work_unit_ratio = args.max_online_work_unit_ratio
    config.budget.max_online_tokens_per_work_unit = args.max_online_tokens_per_work_unit
    config.budget.max_retries_per_work_unit = args.max_retries_per_work_unit
    config.budget.online_concurrency = max(1, args.online_concurrency)

    embedding_service = _build_embedding_service(args)
    chat_llm = _build_chat_llm(args, dry_run=config.dry_run)

    pipeline = HyperKGBuilderPipeline(
        config=config,
        embedding_service=embedding_service,
        chat_llm=chat_llm,
    )
    started = time.time()
    report = pipeline.run()
    duration = time.time() - started
    logging.getLogger(__name__).info(
        "HyperKGBuilder finished in %.1fs (run_id=%s output=%s)",
        duration,
        report.run_id,
        config.output_dir,
    )
    return config


if __name__ == "__main__":
    main()
