"""Article-level HyperKG build orchestration (spec §13, 16 steps)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..config import HKGConfig, load_default_config
from ..embeddings import build_embedding_model
from ..embeddings.base import EmbeddingModel
from ..embeddings.entity_index import build_entity_index
from ..embeddings.hyperedge_index import build_hyperedge_index
from ..embeddings.section_index import build_section_index
from ..llm.agents import (
    ConditionalVerifierAgent,
    HyperEdgeBuilderEntitySelectorAgent,
    PTOBuilderAgent,
)
from ..llm.base import BaseLLMClient
from ..llm.prompt_loader import PromptLoader
from ..parsing.pdf_parser import parse_pdf_to_article
from ..routing.candidate_builder import build_agent1_input
from ..routing.entity_retrieval import (
    load_entity_catalog,
    pto_node_texts,
    retrieve_entity_candidates,
)
from ..routing.semantic_router import route_sections
from ..schemas.article import ArticleDocument
from ..schemas.hyperedge import HyperEdge
from ..schemas.verification import EvidenceQuote, VerifierInput
from ..storage.incidence_writer import build_incidence_records
from ..storage.jsonl_store import HKGStore
from ..validation.risk_gate import risk_gate
from ..validation.schema_validator import (
    validate_article_document,
    validate_hyperedges,
    validate_pto_graph,
)
from ..validation.support_scorer import score_hyperedge_support

logger = logging.getLogger(__name__)


@dataclass
class BuildArticleResult:
    article_id: str
    article: ArticleDocument
    n_sections: int
    n_pto_events: int
    n_hyperedges: int
    n_incidence: int
    n_verified: int
    output_dir: Path
    article_valid: bool
    pto_valid: bool
    hyperedges_valid: bool


def _candidate_min_margin(packs) -> Optional[float]:
    margins: List[float] = []
    for p in packs:
        if len(p.candidates) >= 2:
            margins.append(p.candidates[0].similarity - p.candidates[1].similarity)
    return min(margins) if margins else None


def build_article_hkg(
    pdf_path: Optional[Path],
    entity_catalog_path: Path,
    output_dir: Path,
    config: Optional[HKGConfig] = None,
    *,
    llm_client: Optional[BaseLLMClient] = None,
    prompt_loader: Optional[PromptLoader] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    article: Optional[ArticleDocument] = None,
) -> BuildArticleResult:
    config = config or load_default_config()
    output_dir = Path(output_dir)
    store = HKGStore(output_dir, config)
    model = embedding_model or build_embedding_model(config)
    prompt_loader = prompt_loader or PromptLoader()
    if llm_client is None:
        from ..llm.base import OpenRouterLLMClient

        llm_client = OpenRouterLLMClient(
            model_name=config.llm_model_name, temperature=config.llm_temperature
        )

    # 1. parse PDF -> ArticleDocument (or use injected article, e.g. tests)
    if article is None:
        if pdf_path is None:
            raise ValueError("Either pdf_path or article must be provided")
        article = parse_pdf_to_article(Path(pdf_path))

    article_report = validate_article_document(article)
    store.append_validation_report(article_report)

    # 2-3. embed sections + build section index
    section_index = build_section_index(article, model)

    # 4. route section-level evidence
    bundle = route_sections(article, section_index, config, model)

    # 5. build Agent 1 input
    pto_input = build_agent1_input(article, bundle)

    # 6. Agent 1: PTO builder
    pto_agent = PTOBuilderAgent(llm_client, prompt_loader)
    graph = pto_agent.build(pto_input)

    # 7. validate PTO graph
    pto_report = validate_pto_graph(graph, article)
    store.append_validation_report(pto_report)

    # 8-9. PTO node texts -> offline entity candidate retrieval
    catalog = load_entity_catalog(
        entity_catalog_path,
        allowed_types=config.allowed_entity_types,
        allow_extension=config.allow_entity_type_extension,
        for_build=True,
    )
    entity_index = build_entity_index(catalog, model)
    node_texts = pto_node_texts(graph)
    packs = retrieve_entity_candidates(
        node_texts, entity_index, model, catalog, config.entity_candidate_top_k
    )

    # 10. Agent 2: HyperEdge builder + entity selector
    he_agent = HyperEdgeBuilderEntitySelectorAgent(llm_client, prompt_loader)
    hyperedges = he_agent.build(graph, packs, config.allowed_entity_types)

    # 11. validate HyperEdges
    he_report = validate_hyperedges(hyperedges, article, graph, config.allowed_entity_types)
    store.append_validation_report(he_report)

    # 12-13. support scores + risk gate
    margin = _candidate_min_margin(packs)
    verifier_inputs: List[VerifierInput] = []
    risky_ids: set[str] = set()
    for he in hyperedges:
        support = score_hyperedge_support(he, article, section_index, model)
        he_valid = validate_hyperedges([he], article, graph, config.allowed_entity_types)
        gate = risk_gate(he, graph, article, support, he_valid, config, margin)
        if gate.needs_verification:
            risky_ids.add(he.id)
            nearest = sorted(
                set(
                    support.nearest_edge_section_ids
                    + support.nearest_treatment_section_ids
                    + support.nearest_outcome_section_ids
                )
            )
            quotes = []
            for sid in nearest[:3]:
                sec = article.get_section(sid)
                if sec and sec.text:
                    quotes.append(EvidenceQuote(section_id=sid, quote=sec.text[:300]))
            verifier_inputs.append(
                VerifierInput(
                    hyperedge=he,
                    risk_reasons=gate.risk_reasons,
                    nearest_section_ids=nearest,
                    evidence_quotes=quotes,
                )
            )

    # 14. Agent 3: conditional verifier (only risky edges)
    verifier = ConditionalVerifierAgent(llm_client, prompt_loader)
    results = verifier.verify(verifier_inputs)
    store.append_verification_results(results)

    # 15. apply repairs / statuses
    by_id: Dict[str, HyperEdge] = {he.id: he for he in hyperedges}
    for res in results:
        he = by_id.get(res.he_id)
        if he is None:
            continue
        he.verifier_status = res.status
        if res.repaired_hyperedge is not None:
            rep_valid = validate_hyperedges(
                [res.repaired_hyperedge], article, graph, config.allowed_entity_types
            )
            if rep_valid.status == "pass":
                res.repaired_hyperedge.verifier_status = res.status
                by_id[res.he_id] = res.repaired_hyperedge

    final_edges = [
        he for he in by_id.values() if he.verifier_status != "rejected"
    ]

    # 16. write artifacts
    store.write_article_document(article)
    store.write_pto_graph(graph)
    if final_edges:
        store.append_hyperedges(final_edges)
    store.write_entities(catalog)
    incidence = build_incidence_records(final_edges)
    if incidence:
        store.append_incidence(incidence)

    section_index.save(store.index_path("section_index"))
    entity_index.save(store.index_path("entity_index"))
    build_hyperedge_index(final_edges, model).save(
        store.index_path("hyperedge_index")
    )

    return BuildArticleResult(
        article_id=article.metadata.article_id,
        article=article,
        n_sections=len(article.sections),
        n_pto_events=len(graph.pto_events),
        n_hyperedges=len(final_edges),
        n_incidence=len(incidence),
        n_verified=len(results),
        output_dir=output_dir,
        article_valid=article_report.status == "pass",
        pto_valid=pto_report.status == "pass",
        hyperedges_valid=he_report.status == "pass",
    )
