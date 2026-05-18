"""HyperKGCreator CLI.

    python -m workflows.HyperKGCreator.cli <command> [options]

``parse``, ``index-entities``, ``index-hyperedges`` and ``validate`` never
require an LLM call.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config, load_default_config
from .embeddings import build_embedding_model
from .embeddings.hyperedge_index import build_hyperedge_index
from .embeddings.section_index import build_section_index
from .parsing.pdf_parser import parse_pdf_to_article, save_article_document
from .pipeline.build_article import build_article_hkg
from .pipeline.build_dataset import build_dataset_hkg
from .routing.entity_retrieval import build_entity_index, load_entity_catalog
from .schemas.article import ArticleDocument
from .schemas.hyperedge import HyperEdge
from .schemas.pto_graph import ArticlePTOGraphLite
from .storage.jsonl_store import read_jsonl
from .validation.schema_validator import (
    validate_article_document,
    validate_hyperedges,
    validate_pto_graph,
)

logger = logging.getLogger("hkg_creator.cli")


def _cfg(args) -> "object":
    cfg = load_config(args.config) if getattr(args, "config", None) else load_default_config()
    model_override = getattr(args, "embedding_model", None)
    if model_override:
        cfg.embedding_model_path = model_override
    return cfg


def cmd_parse(args) -> int:
    out_dir = Path(args.out_dir) / "article_documents"
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", args.pdf_dir)
    for pdf in pdfs:
        article = parse_pdf_to_article(pdf)
        path = save_article_document(article, out_dir / f"{article.metadata.article_id}.json")
        print(f"parsed {pdf.name} -> {path} ({len(article.sections)} sections)")
    return 0


def cmd_build_article(args) -> int:
    cfg = _cfg(args)
    res = build_article_hkg(
        pdf_path=Path(args.pdf),
        entity_catalog_path=Path(args.entity_catalog),
        output_dir=Path(args.out_dir),
        config=cfg,
    )
    print(
        json.dumps(
            {
                "article_id": res.article_id,
                "sections": res.n_sections,
                "pto_events": res.n_pto_events,
                "hyperedges": res.n_hyperedges,
                "incidence": res.n_incidence,
                "verified": res.n_verified,
            },
            indent=2,
        )
    )
    return 0


def cmd_build_dataset(args) -> int:
    cfg = _cfg(args)
    results = build_dataset_hkg(
        pdf_dir=Path(args.pdf_dir),
        entity_catalog_path=Path(args.entity_catalog),
        output_dir=Path(args.out_dir),
        config=cfg,
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
        selected_csv=Path(args.selected_csv) if args.selected_csv else None,
    )
    print(f"built {len(results)} articles")
    return 0


def cmd_validate(args) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    article = None
    graph = None

    if args.article_json:
        article = ArticleDocument.model_validate_json(
            Path(args.article_json).read_text(encoding="utf-8")
        )
        rep = validate_article_document(article)
        errors += rep.errors
        warnings += rep.warnings
    if args.pto_graph_json and article is not None:
        graph = ArticlePTOGraphLite.model_validate_json(
            Path(args.pto_graph_json).read_text(encoding="utf-8")
        )
        rep = validate_pto_graph(graph, article)
        errors += rep.errors
        warnings += rep.warnings
    if args.hyperedges_jsonl and article is not None and graph is not None:
        edges = [HyperEdge.model_validate(o) for o in read_jsonl(Path(args.hyperedges_jsonl))]
        rep = validate_hyperedges(edges, article, graph)
        errors += rep.errors
        warnings += rep.warnings

    status = "fail" if errors else "pass"
    print(json.dumps({"status": status, "errors": errors, "warnings": warnings}, indent=2))
    return 1 if status == "fail" else 0


def cmd_index_entities(args) -> int:
    cfg = _cfg(args)
    model = build_embedding_model(cfg)
    catalog = load_entity_catalog(
        Path(args.entity_catalog),
        allowed_types=cfg.allowed_entity_types,
        allow_extension=cfg.allow_entity_type_extension,
        for_build=False,
    )
    index = build_entity_index(catalog, model)
    out = Path(args.out_dir) / cfg.output_paths["entity_index"]
    index.save(out)
    print(f"indexed {len(index)} entities -> {out}")
    return 0


def cmd_index_hyperedges(args) -> int:
    cfg = _cfg(args)
    model = build_embedding_model(cfg)
    edges = [HyperEdge.model_validate(o) for o in read_jsonl(Path(args.hyperedges_jsonl))]
    index = build_hyperedge_index(edges, model)
    out = Path(args.out_dir) / cfg.output_paths["hyperedge_index"]
    index.save(out)
    print(f"indexed {len(index)} hyperedges -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hkg-build", description="HyperKGCreator pipeline")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("parse", help="PDFs -> ArticleDocument JSON (no LLM)")
    sp.add_argument("--pdf-dir", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_parse)

    sp = sub.add_parser("build-article", help="Full single-article build")
    sp.add_argument("--pdf", required=True)
    sp.add_argument("--entity-catalog", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--config", default=None)
    sp.set_defaults(func=cmd_build_article)

    sp = sub.add_parser("build-dataset", help="Full dataset build")
    sp.add_argument("--pdf-dir", required=True)
    sp.add_argument("--entity-catalog", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--config", default=None)
    sp.add_argument("--metadata-csv", default=None)
    sp.add_argument("--selected-csv", default=None)
    sp.set_defaults(func=cmd_build_dataset)

    sp = sub.add_parser("validate", help="Validate artifacts (no LLM)")
    sp.add_argument("--article-json", default=None)
    sp.add_argument("--pto-graph-json", default=None)
    sp.add_argument("--hyperedges-jsonl", default=None)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("index-entities", help="Build entity vector index (no LLM)")
    sp.add_argument("--entity-catalog", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--embedding-model", default=None)
    sp.add_argument("--config", default=None)
    sp.set_defaults(func=cmd_index_entities)

    sp = sub.add_parser("index-hyperedges", help="Build hyperedge vector index (no LLM)")
    sp.add_argument("--hyperedges-jsonl", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--embedding-model", default=None)
    sp.add_argument("--config", default=None)
    sp.set_defaults(func=cmd_index_hyperedges)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
