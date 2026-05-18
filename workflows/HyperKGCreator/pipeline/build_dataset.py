"""Dataset-level build over a directory of PDFs.

Optional ``--metadata-csv`` / ``--selected-csv`` (plan R7) choose which PDFs to
process and attach ``ArticleMetadata``. The embedding model and LLM client are
constructed once and reused across articles.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..config import HKGConfig, load_default_config
from ..embeddings import build_embedding_model
from ..embeddings.base import EmbeddingModel
from ..llm.base import BaseLLMClient
from ..llm.prompt_loader import PromptLoader
from ..parsing.pdf_parser import parse_pdf_to_article
from ..schemas.article import ArticleDocument
from .build_article import BuildArticleResult, build_article_hkg

logger = logging.getLogger(__name__)

_NONALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def _norm(text: str) -> str:
    return _NONALNUM_RE.sub("_", (text or "").strip()).strip("_").lower()


@dataclass
class CsvIndex:
    by_key: Dict[str, dict]

    @classmethod
    def load(cls, path: Path) -> "CsvIndex":
        by_key: Dict[str, dict] = {}
        with Path(path).open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                for col in ("DOI", "Key"):
                    val = row.get(col)
                    if val:
                        by_key[_norm(val)] = row
                attach = row.get("File Attachments") or ""
                for piece in re.split(r"[;,]", attach):
                    stem = Path(piece.strip()).stem
                    if stem:
                        by_key[_norm(stem)] = row
        return cls(by_key)

    def lookup(self, pdf_path: Path) -> Optional[dict]:
        return self.by_key.get(_norm(pdf_path.stem))


def _apply_metadata(article: ArticleDocument, row: dict) -> None:
    m = article.metadata
    m.title = row.get("Title") or m.title
    author = row.get("Author")
    if author:
        m.authors = [a.strip() for a in re.split(r";", author) if a.strip()]
    m.journal = row.get("Publication Title") or m.journal
    m.year = (row.get("Publication Year") or m.year) or None
    m.doi = row.get("DOI") or m.doi


def build_dataset_hkg(
    pdf_dir: Path,
    entity_catalog_path: Path,
    output_dir: Path,
    config: Optional[HKGConfig] = None,
    *,
    metadata_csv: Optional[Path] = None,
    selected_csv: Optional[Path] = None,
    llm_client: Optional[BaseLLMClient] = None,
    prompt_loader: Optional[PromptLoader] = None,
    embedding_model: Optional[EmbeddingModel] = None,
) -> List[BuildArticleResult]:
    config = config or load_default_config()
    pdf_dir = Path(pdf_dir)
    model = embedding_model or build_embedding_model(config)
    prompt_loader = prompt_loader or PromptLoader()
    if llm_client is None:
        from ..llm.base import OpenRouterLLMClient

        llm_client = OpenRouterLLMClient(
            model_name=config.llm_model_name, temperature=config.llm_temperature
        )

    meta_index = CsvIndex.load(metadata_csv) if metadata_csv else None
    selected_index = CsvIndex.load(selected_csv) if selected_csv else None

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    results: List[BuildArticleResult] = []
    for pdf in pdfs:
        if selected_index is not None and selected_index.lookup(pdf) is None:
            logger.debug("Skipping %s (not in selected csv)", pdf.name)
            continue
        try:
            article = parse_pdf_to_article(pdf)
            row = meta_index.lookup(pdf) if meta_index else None
            if row:
                _apply_metadata(article, row)
            res = build_article_hkg(
                pdf_path=None,
                entity_catalog_path=entity_catalog_path,
                output_dir=output_dir,
                config=config,
                llm_client=llm_client,
                prompt_loader=prompt_loader,
                embedding_model=model,
                article=article,
            )
            results.append(res)
            logger.info(
                "Built %s: %d sections, %d hyperedges",
                res.article_id,
                res.n_sections,
                res.n_hyperedges,
            )
        except Exception as err:  # noqa: BLE001
            logger.exception("Failed to build %s: %s", pdf.name, err)
    return results
