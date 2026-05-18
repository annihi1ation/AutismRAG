"""The three (and only three) LLM agents.

Agents load an owner prompt, call the client, and schema-validate the output.
They contain NO clinical extraction prompt content.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from ..schemas.entity import EntityCandidatePack
from ..schemas.hyperedge import HyperEdge
from ..schemas.pto_graph import ArticlePTOGraphLite
from ..schemas.storage import PTOBuilderInput
from ..schemas.verification import VerificationResult, VerifierInput
from .base import BaseLLMClient
from .prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


def _join_text(*parts: Any) -> str:
    text = " ".join(str(p).strip() for p in parts if p is not None and str(p).strip())
    return text.strip()


def _evidence_ids(obj: Dict[str, Any]) -> List[str]:
    return list(obj.get("evidence_span_ids") or obj.get("evidence_section_ids") or [])


def _pto_direction(value: Any) -> str:
    mapping = {
        "adverse": "adverse_event",
        "no_change": "no_effect",
        "not_reported": "unclear",
    }
    return mapping.get(str(value or "unclear"), str(value or "unclear"))


def _normalize_pto_graph(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(raw)

    def node(obj: Dict[str, Any], *, outcome: bool = False) -> Dict[str, Any]:
        normalized = dict(obj)
        normalized["id"] = normalized.get("id") or normalized.get("node_id")
        normalized["text"] = normalized.get("text") or _join_text(
            normalized.get("label"), normalized.get("description")
        )
        normalized["evidence_span_ids"] = _evidence_ids(normalized)
        if outcome:
            normalized["direction"] = _pto_direction(normalized.get("direction"))
        return normalized

    out["patient_nodes"] = [node(o) for o in _as_list(raw, "patient_nodes")]
    out["treatment_nodes"] = [node(o) for o in _as_list(raw, "treatment_nodes")]
    out["outcome_nodes"] = [node(o, outcome=True) for o in _as_list(raw, "outcome_nodes")]
    events = []
    for obj in _as_list(raw, "pto_events"):
        normalized = dict(obj)
        normalized["id"] = normalized.get("id") or normalized.get("event_id")
        normalized["evidence_span_ids"] = _evidence_ids(normalized)
        events.append(normalized)
    out["pto_events"] = events
    return out


def _text_by_id(graph: ArticlePTOGraphLite, attr: str, ids: Sequence[str]) -> str:
    by_id = {node.id: node.text for node in getattr(graph, attr)}
    return "; ".join(by_id[i] for i in ids if i in by_id)


def _normalize_hyperedges(
    raw_edges: Sequence[Dict[str, Any]], graph: ArticlePTOGraphLite
) -> List[Dict[str, Any]]:
    normalized_edges: List[Dict[str, Any]] = []
    for obj in raw_edges:
        source = dict(obj.get("source") or {})
        pto_ids = list(source.get("pto_event_ids") or source.get("pto_event_id") or [])
        if isinstance(source.get("pto_event_id"), str):
            pto_ids = [source["pto_event_id"]]
        patient_ids = list(source.get("patient_node_ids") or [])
        treatment_ids = list(source.get("treatment_node_ids") or [])
        outcome_ids = list(source.get("outcome_node_ids") or [])
        response = dict(obj.get("response") or {})
        entities = []
        for link in obj.get("entities") or obj.get("entity_links") or []:
            role = link.get("role")
            if role not in {"PATIENT", "TREATMENT", "OUTCOME"}:
                continue
            name = link.get("name") or link.get("canonical_name") or link.get("mention") or ""
            entities.append(
                {
                    "entity_id": link.get("entity_id"),
                    "name": name,
                    "entity_type": link.get("entity_type"),
                    "mention": link.get("mention") or name,
                    "role": role,
                    "confidence": link.get("confidence")
                    or link.get("selection_confidence"),
                }
            )
        normalized_edges.append(
            {
                **obj,
                "id": obj.get("id") or obj.get("hyperedge_id"),
                "patient_text": obj.get("patient_text")
                or _text_by_id(graph, "patient_nodes", patient_ids),
                "treatment_text": obj.get("treatment_text")
                or _text_by_id(graph, "treatment_nodes", treatment_ids),
                "outcome_text": obj.get("outcome_text")
                or response.get("outcome_text")
                or _text_by_id(graph, "outcome_nodes", outcome_ids),
                "edge_text": obj.get("edge_text")
                or obj.get("summary")
                or obj.get("label")
                or response.get("outcome_text"),
                "entities": entities,
                "source": {
                    "article_id": graph.article_id,
                    "pto_event_id": pto_ids[0] if pto_ids else "",
                    "evidence_span_ids": list(
                        source.get("evidence_span_ids")
                        or source.get("evidence_section_ids")
                        or []
                    ),
                },
            }
        )
    return normalized_edges


def _normalize_verification_results(raw_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    status_map = {
        "verified": "accepted",
        "revise": "needs_review",
        "insufficient_evidence": "needs_review",
        "reject": "rejected",
        "accepted": "accepted",
        "needs_review": "needs_review",
        "rejected": "rejected",
    }
    normalized = []
    for obj in raw_results:
        notes = obj.get("verifier_notes") or []
        issues = obj.get("issues") or []
        reason = obj.get("reason") or "; ".join(
            [str(n) for n in notes] + [str(i.get("message")) for i in issues if i.get("message")]
        )
        normalized.append(
            {
                "he_id": obj.get("he_id") or obj.get("hyperedge_id"),
                "status": status_map.get(str(obj.get("status")), "needs_review"),
                "reason": reason or None,
            }
        )
    return normalized


def _as_list(raw: Dict[str, Any], *keys: str) -> List[Any]:
    for key in keys:
        val = raw.get(key)
        if isinstance(val, list):
            return val
    if isinstance(raw.get("items"), list):
        return raw["items"]
    return []


class _Agent:
    agent_key: str = ""
    prompt_name: str = ""

    def __init__(
        self,
        client: BaseLLMClient,
        prompt_loader: Optional[PromptLoader] = None,
        prompt_override: Optional[str] = None,
    ) -> None:
        self.client = client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.prompt_override = prompt_override

    def _prompt(self) -> str:
        if self.prompt_override is not None:
            return self.prompt_override
        return self.prompt_loader.load(
            self.prompt_name, allow_placeholder=self.client.is_fake
        )

    def _call(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        input_payload = {k: v for k, v in variables.items() if not k.startswith("__")}
        variables = {
            "__agent__": self.agent_key,
            "input_json": json.dumps(input_payload, ensure_ascii=False, indent=2),
            **variables,
        }
        return self.client.generate_json(self._prompt(), variables)


class PTOBuilderAgent(_Agent):
    agent_key = "pto_builder"
    prompt_name = "pto_builder"

    def build(self, pto_input: PTOBuilderInput) -> ArticlePTOGraphLite:
        raw = self._call({"pto_input": pto_input.model_dump()})
        raw.setdefault("article_id", pto_input.article_id)
        raw = _normalize_pto_graph(raw)
        return ArticlePTOGraphLite.model_validate(raw)


class HyperEdgeBuilderEntitySelectorAgent(_Agent):
    agent_key = "hyperedge_builder_entity_selector"
    prompt_name = "hyperedge_builder_entity_selector"

    def build(
        self,
        graph: ArticlePTOGraphLite,
        candidate_packs: Sequence[EntityCandidatePack],
        allowed_entity_types: Sequence[str],
    ) -> List[HyperEdge]:
        raw = self._call(
            {
                "pto_graph": graph.model_dump(),
                "entity_candidate_packs": [p.model_dump() for p in candidate_packs],
                "allowed_entity_types": list(allowed_entity_types),
            }
        )
        edges_raw = _as_list(raw, "hyperedges", "edges")
        edges_raw = _normalize_hyperedges(edges_raw, graph)
        edges: List[HyperEdge] = []
        for obj in edges_raw:
            try:
                edges.append(HyperEdge.model_validate(obj))
            except Exception as err:  # noqa: BLE001
                logger.warning("Dropping invalid HyperEdge from LLM: %s", err)
        return edges


class ConditionalVerifierAgent(_Agent):
    agent_key = "conditional_verifier"
    prompt_name = "conditional_verifier"

    def verify(
        self, verifier_inputs: Sequence[VerifierInput]
    ) -> List[VerificationResult]:
        if not verifier_inputs:
            return []
        raw = self._call(
            {"verifier_inputs": [vi.model_dump() for vi in verifier_inputs]}
        )
        results_raw = _as_list(raw, "results", "verification_results")
        results_raw = _normalize_verification_results(results_raw)
        results: List[VerificationResult] = []
        for obj in results_raw:
            try:
                results.append(VerificationResult.model_validate(obj))
            except Exception as err:  # noqa: BLE001
                logger.warning("Dropping invalid VerificationResult: %s", err)
        return results
