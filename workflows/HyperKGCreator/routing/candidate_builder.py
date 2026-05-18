"""Non-LLM candidate builder: prepares the compact Agent 1 input.

Groups selected sections by semantic role, keeps section ids/headings/types/
page ranges/text/table markdown, and (optionally) pairwise section similarity.
It does NOT extract clinical P/T/O facts and does NOT create PTO nodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..embeddings.similarity import cosine_sim
from ..schemas.article import ArticleDocument
from ..schemas.storage import EvidenceBundle, PTOBuilderInput

_ROLE_FIELDS = {
    "patient": "patient_section_ids",
    "treatment": "treatment_section_ids",
    "outcome": "outcome_section_ids",
    "comparator": "comparator_section_ids",
    "study_design": "study_design_section_ids",
}


def build_agent1_input(
    article: ArticleDocument,
    bundle: EvidenceBundle,
    compute_pairwise: bool = True,
) -> PTOBuilderInput:
    selected_ids = list(bundle.selected_section_ids)
    selected = [s for s in article.sections if s.section_id in set(selected_ids)]
    # Preserve article order.
    order = {s.section_id: s.section_order for s in article.sections}
    selected.sort(key=lambda s: order.get(s.section_id, 1_000_000))

    # Role membership scores (binary surrogate; deterministic, non-LLM).
    role_scores: Dict[str, Dict[str, float]] = {}
    for sid in selected_ids:
        scores: Dict[str, float] = {}
        for role, field_name in _ROLE_FIELDS.items():
            scores[role] = 1.0 if sid in set(getattr(bundle, field_name)) else 0.0
        role_scores[sid] = scores

    pairwise: Dict[str, Dict[str, float]] = {}
    if compute_pairwise:
        for a in selected:
            if a.embedding is None:
                continue
            row: Dict[str, float] = {}
            for b in selected:
                if a.section_id == b.section_id or b.embedding is None:
                    continue
                row[b.section_id] = round(
                    cosine_sim(np.array(a.embedding), np.array(b.embedding)), 4
                )
            pairwise[a.section_id] = row

    return PTOBuilderInput(
        article_id=article.metadata.article_id,
        selected_sections=selected,
        section_role_scores=role_scores,
        pairwise_section_similarities=pairwise,
        metadata={
            "title": article.metadata.title,
            "doi": article.metadata.doi,
            "n_sections_total": len(article.sections),
            "n_sections_selected": len(selected),
        },
    )
