"""File-based artifact store.

Layout (under ``out_dir``)::

    article_documents/{article_id}.json
    pto_graphs/{article_id}.json
    hyperedges.jsonl
    entities.jsonl
    incidence_edges.jsonl
    validation_reports.jsonl
    verification_results.jsonl
    section_embeddings.index
    entity_embeddings.index
    hyperedge_embeddings.index
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from pydantic import BaseModel

from ..config import HKGConfig
from ..schemas.article import ArticleDocument
from ..schemas.entity import EntityRecord
from ..schemas.hyperedge import HyperEdge
from ..schemas.pto_graph import ArticlePTOGraphLite
from ..schemas.storage import IncidenceRecord, ValidationReport
from ..schemas.verification import VerificationResult


def _dump(obj: Any) -> Any:
    return obj.model_dump() if isinstance(obj, BaseModel) else obj


def write_jsonl(path: Path, rows: Iterable[Any], append: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_dump(row), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


class HKGStore:
    """Resolves and writes all pipeline artifacts under one output dir."""

    def __init__(self, out_dir: Path, config: HKGConfig) -> None:
        self.out_dir = Path(out_dir)
        self.config = config
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return self.config.resolved_output(self.out_dir, key)

    # -- per-article documents ----------------------------------------------
    def write_article_document(self, article: ArticleDocument) -> Path:
        d = self._p("article_documents")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{article.metadata.article_id}.json"
        path.write_text(article.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_pto_graph(self, graph: ArticlePTOGraphLite) -> Path:
        d = self._p("pto_graphs")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{graph.article_id}.json"
        path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
        return path

    # -- aggregate jsonl ----------------------------------------------------
    def append_hyperedges(self, edges: List[HyperEdge]) -> Path:
        return write_jsonl(self._p("hyperedges"), edges, append=True)

    def write_entities(self, records: List[EntityRecord]) -> Path:
        return write_jsonl(self._p("entities"), records, append=False)

    def append_incidence(self, records: List[IncidenceRecord]) -> Path:
        return write_jsonl(self._p("incidence_edges"), records, append=True)

    def append_validation_report(self, report: ValidationReport) -> Path:
        return write_jsonl(self._p("validation_reports"), [report], append=True)

    def append_verification_results(
        self, results: List[VerificationResult]
    ) -> Path:
        return write_jsonl(
            self._p("verification_results"), results, append=True
        )

    def index_path(self, which: str) -> Path:
        return self._p(which)
