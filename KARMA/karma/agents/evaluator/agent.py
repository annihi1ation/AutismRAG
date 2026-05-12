"""Evaluator Agent Implementation"""

import logging
from typing import List, Tuple
from karma.core.base_agent import BaseAgent
from karma.core.data_structures import KnowledgeTriple
from karma.agents.prompt_loader import get_agent_config

logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    """Evaluator Agent (EA) for final quality assessment and integration decisions."""

    def __init__(self, client, model_name: str, integrate_threshold: float = 0.6):
        config = get_agent_config("evaluator")
        system_prompt = config.get("system_prompt", "")
        super().__init__(client, model_name, system_prompt)
        self.integrate_threshold = integrate_threshold

    def process(self, triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """Evaluate and filter triples based on quality metrics."""
        return self.finalize_triples(triples)[0]

    def finalize_triples(self, candidate_triples: List[KnowledgeTriple]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """Filter triples based on integration threshold."""
        integrated_triples = []

        for triple in candidate_triples:
            # Ensure all metrics are set
            if triple.confidence <= 0:
                triple.confidence = 0.5
            if triple.clarity <= 0:
                triple.clarity = 0.5
            if triple.relevance <= 0:
                triple.relevance = 0.5

            # Calculate integration score
            integration_score = self._aggregate_scores(triple)

            # Keep triple if it meets threshold
            if integration_score >= self.integrate_threshold:
                integrated_triples.append(triple)

        return integrated_triples, 0, 0, 0.0

    def _aggregate_scores(self, triple: KnowledgeTriple) -> float:
        """Combine quality metrics into final score."""
        # Weighted average: confidence=50%, clarity=25%, relevance=25%
        return (0.5 * triple.confidence + 0.25 * triple.clarity + 0.25 * triple.relevance)