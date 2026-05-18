"""Embedding support scoring.

Compares HyperEdge text fields to SECTION embeddings. Nearest evidence units
are SECTION ids only — never paragraph/chunk ids.
"""

from __future__ import annotations

from typing import List, Tuple

from ..embeddings.base import EmbeddingModel
from ..schemas.article import ArticleDocument
from ..schemas.hyperedge import HyperEdge
from ..schemas.verification import SupportScore


def _nearest(
    text: str, article: ArticleDocument, section_index, model: EmbeddingModel, top_k: int
) -> Tuple[float, List[str]]:
    if not text.strip():
        return 0.0, []
    qvec = model.embed_text(text)
    hits = section_index.search(qvec, top_k)
    if not hits:
        return 0.0, []
    section_ids = article.section_ids()
    filtered = [(sid, score) for sid, score in hits if sid in section_ids]
    if not filtered:
        return 0.0, []
    best = max(score for _, score in filtered)
    return float(best), [sid for sid, _ in filtered]


def score_hyperedge_support(
    he: HyperEdge,
    article: ArticleDocument,
    section_index,  # SectionVectorIndex
    model: EmbeddingModel,
    top_k: int = 3,
) -> SupportScore:
    p_s, p_ids = _nearest(he.patient_text, article, section_index, model, top_k)
    t_s, t_ids = _nearest(he.treatment_text, article, section_index, model, top_k)
    o_s, o_ids = _nearest(he.outcome_text, article, section_index, model, top_k)
    e_s, e_ids = _nearest(he.edge_text, article, section_index, model, top_k)
    return SupportScore(
        he_id=he.id,
        patient_support_score=p_s,
        treatment_support_score=t_s,
        outcome_support_score=o_s,
        edge_support_score=e_s,
        nearest_patient_section_ids=p_ids,
        nearest_treatment_section_ids=t_ids,
        nearest_outcome_section_ids=o_ids,
        nearest_edge_section_ids=e_ids,
    )
