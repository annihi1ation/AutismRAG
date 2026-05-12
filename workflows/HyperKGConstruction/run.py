"""CLI entry points for HyperKG construction.

Example:

    python -m workflows.HyperKGConstruction.run run-karma-output \
        --karma-output-dir KARMA/output \
        --unified-kg KARMA/output/unified_kg/unified_kg.json \
        --output-dir output/hyperkg \
        --api-key "$OPENROUTER_API_KEY"

The prompts in ``prompts.toml`` are intentionally blank until filled.
LLM-backed stages will raise a clear configuration error if run before
the prompt templates are populated.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from .config import HyperKGConfig
from .checkpoint import CheckpointError
from .llm import OpenRouterConfigurationError, build_openrouter_client
from .pipeline import HyperKGPipeline


def _configure_logging(level: str, output_dir: str | None = None, log_file: str | None = None) -> None:
    level_no = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level_no)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    resolved_log_file = log_file
    if resolved_log_file is None and output_dir:
        resolved_log_file = os.path.join(output_dir, "run.log")
    if resolved_log_file:
        log_parent = os.path.dirname(resolved_log_file)
        if log_parent:
            os.makedirs(log_parent, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _article_id_from_path(path: str, suffix: str) -> str:
    name = os.path.basename(path)
    return name[: -len(suffix)] if name.endswith(suffix) else os.path.splitext(name)[0]


def _karma_output_candidates(karma_output_dir: str, subdir_name: str) -> List[str]:
    subdir = os.path.join(karma_output_dir, subdir_name)
    if os.path.isdir(subdir):
        return [subdir, karma_output_dir]
    return [karma_output_dir]


def _iter_karma_output_files(karma_output_dir: str, subdir_name: str, pattern: str) -> List[str]:
    matches: List[str] = []
    seen: set[str] = set()

    for directory in _karma_output_candidates(karma_output_dir, subdir_name):
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            basename = os.path.basename(path)
            if basename in seen:
                continue
            seen.add(basename)
            matches.append(path)

    return matches


def _load_karma_output(karma_output_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    summaries: List[Dict[str, Any]] = []
    local_kgs: Dict[str, Dict[str, Any]] = {}

    for kg_path in _iter_karma_output_files(karma_output_dir, "local_kg", "*_kg.json"):
        if os.path.basename(kg_path).startswith("unified_"):
            continue
        article_id = _article_id_from_path(kg_path, "_kg.json")
        kg = _read_json(kg_path)
        kg.setdefault("article_id", article_id)
        metadata = dict(kg.get("metadata") or {})
        metadata.setdefault("article_id", article_id)
        kg["metadata"] = metadata
        local_kgs[article_id] = kg

    for summary_path in _iter_karma_output_files(karma_output_dir, "summaries", "*_summaries.json"):
        article_id = _article_id_from_path(summary_path, "_summaries.json")
        raw_summaries = _read_json(summary_path)
        for idx, raw in enumerate(raw_summaries or [], start=1):
            if isinstance(raw, dict):
                item = dict(raw)
                item.setdefault("summary_text", item.get("summary") or item.get("text") or "")
            else:
                item = {"summary_text": str(raw)}
            item.setdefault("article_id", article_id)
            item.setdefault("segment_id", f"seg{idx:04d}")
            item.setdefault("summary_id", f"SUM:{article_id}:seg{idx:04d}")
            summaries.append(item)

    return summaries, local_kgs


def _default_unified_kg_path(karma_output_dir: str) -> str:
    candidates = [
        os.path.join(karma_output_dir, "unified_kg", "unified_kg.json"),
        os.path.join(karma_output_dir, "unified_kg.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _build_config(args: argparse.Namespace) -> HyperKGConfig:
    return HyperKGConfig(
        embedding_enabled=bool(args.embedding_model),
        embedding_model_name=args.embedding_model or "",
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        enable_vector_index=not args.no_vector_index,
        accept_threshold=args.accept_threshold,
        review_threshold=args.review_threshold,
        max_local_entities_per_packet=args.max_local_entities,
        max_local_triples_per_packet=args.max_local_triples,
        max_workers=max(1, args.workers),
        checkpoint_enabled=not args.no_checkpoint,
        resume_enabled=not args.no_resume,
        checkpoint_dir=args.checkpoint_dir,
        progress_enabled=not args.no_progress,
    )


def _run_pipeline(
    args: argparse.Namespace,
    summaries: List[Dict[str, Any]],
    local_kgs: List[Dict[str, Any]] | Dict[str, Dict[str, Any]],
    unified_kg: Dict[str, Any],
) -> int:
    try:
        client = build_openrouter_client(
            api_key=args.api_key,
            base_url=args.api_base_url,
            env_file=args.env_file,
        )
    except OpenRouterConfigurationError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    pipeline = HyperKGPipeline(
        client=client,
        model_name=args.model,
        output_dir=args.output_dir,
        config=_build_config(args),
        prompt_file=args.prompts_file,
        temperature=args.temperature,
    )
    try:
        result = pipeline.run(
            summaries=summaries,
            local_kgs=local_kgs,
            unified_kg=unified_kg,
            metadata={},
        )
    except CheckpointError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    print(json.dumps(result["run_stats"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if not args.unified_kg:
        print("ERROR: --unified-kg is required for explicit run mode.", file=sys.stderr)
        return 2
    summaries = _read_json(args.summaries)
    local_kgs = _read_json(args.local_kgs)
    unified_kg = _read_json(args.unified_kg)
    if not isinstance(summaries, list):
        print("ERROR: --summaries must point to a JSON list.", file=sys.stderr)
        return 2
    return _run_pipeline(args, summaries, local_kgs, unified_kg)


def _cmd_run_karma_output(args: argparse.Namespace) -> int:
    summaries, local_kgs = _load_karma_output(args.karma_output_dir)
    unified_kg_path = args.unified_kg or _default_unified_kg_path(args.karma_output_dir)
    if not os.path.exists(unified_kg_path):
        print(f"ERROR: unified KG not found: {unified_kg_path}", file=sys.stderr)
        return 2
    unified_kg = _read_json(unified_kg_path)
    return _run_pipeline(args, summaries, local_kgs, unified_kg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workflows.HyperKGConstruction.run")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--unified-kg", default=None)
        p.add_argument("--output-dir", default="output/hyperkg")
        p.add_argument("--api-key", default=None)
        p.add_argument("--api-base-url", default=None)
        p.add_argument("--env-file", default=None)
        p.add_argument("--model", default="google/gemini-3-flash-preview")
        p.add_argument("--temperature", type=float, default=0.1)
        p.add_argument("--prompts-file", default=None)
        p.add_argument("--embedding-model", default="")
        p.add_argument("--embedding-device", default="cuda")
        p.add_argument("--embedding-batch-size", type=int, default=64)
        p.add_argument("--no-vector-index", action="store_true")
        p.add_argument("--accept-threshold", type=float, default=0.75)
        p.add_argument("--review-threshold", type=float, default=0.55)
        p.add_argument("--max-local-entities", type=int, default=80)
        p.add_argument("--max-local-triples", type=int, default=60)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--checkpoint-dir", default=None)
        p.add_argument("--no-checkpoint", action="store_true")
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("--no-progress", action="store_true")
        p.add_argument("--log-file", default=None)

    p_run = sub.add_parser("run", help="Run from explicit summary/local-KG JSON files")
    p_run.add_argument("--summaries", required=True)
    p_run.add_argument("--local-kgs", required=True)
    add_common(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_karma = sub.add_parser("run-karma-output", help="Run from a KARMA output directory")
    p_karma.add_argument("--karma-output-dir", required=True)
    add_common(p_karma)
    p_karma.set_defaults(func=_cmd_run_karma_output)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(
        args.log_level,
        output_dir=getattr(args, "output_dir", None),
        log_file=getattr(args, "log_file", None),
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
