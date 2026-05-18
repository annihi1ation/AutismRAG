"""Embedding semantic router → section-level EvidenceBundle.

Selects relevant SECTIONS for Agent 1. Output uses section ids only. Routing
combines (a) section_type / heading signal and (b) embedding similarity to
short configurable route prototype labels. No clinical extraction prompts are
written here.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from ..config import HKGConfig
from ..embeddings.base import EmbeddingModel
from ..embeddings.similarity import cosine_sim
from ..schemas.article import ArticleDocument
from ..schemas.storage import EvidenceBundle

logger = logging.getLogger(__name__)

# section_type → routes it strongly belongs to (heading signal).
_TYPE_ROUTES: Dict[str, List[str]] = {
    "participants": ["patient", "study_design"],
    "abstract": ["patient", "treatment", "outcome"],
    "intervention": ["treatment"],
    "methods": ["treatment", "study_design"],
    "outcomes": ["outcome"],
    "results": ["outcome"],
    "discussion": ["outcome"],
    "introduction": ["patient"],
    "conclusion": ["outcome"],
}
_COMPARATOR_HINTS = (
    "control group",
    "comparison",
    "versus",
    " vs ",
    "placebo",
    "waitlist",
    "treatment as usual",
    "comparator",
)

_BUNDLE_FIELDS = {
    "patient": "patient_section_ids",
    "treatment": "treatment_section_ids",
    "outcome": "outcome_section_ids",
    "comparator": "comparator_section_ids",
    "study_design": "study_design_section_ids",
}


def _route_vectors(
    config: HKGConfig, model: EmbeddingModel
) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    for route, labels in config.route_definitions.items():
        if not labels:
            continue
        mat = model.embed_texts(list(labels))
        vectors[route] = mat.mean(axis=0)
    return vectors


def route_sections(
    article: ArticleDocument,
    section_index,  # SectionVectorIndex (kept for signature parity / future use)
    config: HKGConfig,
    model: EmbeddingModel,
) -> EvidenceBundle:
    """Return an EvidenceBundle of section ids per semantic role."""
    bundle = EvidenceBundle(article_id=article.metadata.article_id)
    if not article.sections:
        return bundle

    route_vecs = _route_vectors(config, model)
    top_k = max(1, config.section_router_top_k)

    # (1) Heading / section_type signal.
    role_sections: Dict[str, List[str]] = {r: [] for r in _BUNDLE_FIELDS}
    for sec in article.sections:
        for route in _TYPE_ROUTES.get(sec.section_type, []):
            role_sections[route].append(sec.section_id)
        text_l = (sec.heading + " " + sec.text).lower()
        if any(h in text_l for h in _COMPARATOR_HINTS):
            role_sections["comparator"].append(sec.section_id)

    # (2) Embedding similarity to route prototypes.
    for route, rvec in route_vecs.items():
        if route not in role_sections:
            continue
        scored = []
        for sec in article.sections:
            if sec.embedding is None:
                continue
            scored.append((sec.section_id, cosine_sim(np.array(sec.embedding), rvec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        for sid, _ in scored[:top_k]:
            role_sections[route].append(sid)

    # Dedup while preserving article (section_order) order.
    order = {s.section_id: s.section_order for s in article.sections}
    for route, field_name in _BUNDLE_FIELDS.items():
        unique = sorted(set(role_sections[route]), key=lambda s: order.get(s, 1_000_000))
        setattr(bundle, field_name, unique)

    selected = set()
    for field_name in _BUNDLE_FIELDS.values():
        selected.update(getattr(bundle, field_name))
    bundle.selected_section_ids = sorted(selected, key=lambda s: order.get(s, 1_000_000))
    return bundle
