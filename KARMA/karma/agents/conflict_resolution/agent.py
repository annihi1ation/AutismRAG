"""Conflict Resolution Agent Implementation"""

import logging
from typing import List, Tuple, Optional
from karma.core.base_agent import BaseAgent
from karma.core.data_structures import KnowledgeTriple
from karma.agents.prompt_loader import get_agent_config

logger = logging.getLogger(__name__)


class ConflictResolutionAgent(BaseAgent):
    """Conflict Resolution Agent (CRA) for handling contradictory knowledge."""

    def __init__(self, client, model_name: str):
        config = get_agent_config("conflict_resolution")
        system_prompt = config.get("system_prompt", "")
        super().__init__(client, model_name, system_prompt)

    def process(self, new_triples: List[KnowledgeTriple], existing_triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """Resolve conflicts between new and existing triples."""
        return self.resolve_conflicts(new_triples, existing_triples)[0]

    def resolve_conflicts(self, new_triples: List[KnowledgeTriple], existing_triples: List[KnowledgeTriple]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """Check for conflicts and resolve them."""
        final_triples = []

        for new_triple in new_triples:
            conflicting_triple = self._find_contradiction(new_triple, existing_triples)

            if conflicting_triple:
                # Simple resolution: keep higher confidence triple
                if new_triple.confidence > conflicting_triple.confidence:
                    final_triples.append(new_triple)
            else:
                final_triples.append(new_triple)

        return final_triples, 0, 0, 0.0

    def _find_contradiction(self, new_triple: KnowledgeTriple, existing_triples: List[KnowledgeTriple]) -> Optional[KnowledgeTriple]:
        """Find contradicting triples."""
        contradiction_pairs = {
            ("treats", "causes"), ("inhibits", "activates"),
            ("increases", "decreases"), ("upregulates", "downregulates")
        }

        for existing in existing_triples:
            if (existing.head.lower() == new_triple.head.lower() and
                existing.tail.lower() == new_triple.tail.lower()):

                rel_pair = (existing.relation, new_triple.relation)
                if rel_pair in contradiction_pairs or rel_pair[::-1] in contradiction_pairs:
                    return existing

        return None