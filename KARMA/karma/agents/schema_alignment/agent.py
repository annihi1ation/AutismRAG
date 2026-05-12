"""Schema Alignment Agent Implementation"""

import logging
from typing import List, Tuple
from karma.core.base_agent import BaseAgent
from karma.core.data_structures import KGEntity, KnowledgeTriple
from karma.agents.prompt_loader import get_agent_config

logger = logging.getLogger(__name__)


class SchemaAlignmentAgent(BaseAgent):
    """Schema Alignment Agent (SAA) for entity type classification and relation normalization."""

    def __init__(self, client, model_name: str):
        config = get_agent_config("schema_alignment")
        system_prompt = config.get("system_prompt", "")
        super().__init__(client, model_name, system_prompt)

    def process(self, entities: List[KGEntity], relationships: List[KnowledgeTriple]) -> Tuple[List[KGEntity], List[KnowledgeTriple]]:
        """Align entities and relationships to standard schema."""
        aligned_entities = self.align_entities(entities)[0]
        aligned_relationships = self.align_relationships(relationships)
        return aligned_entities, aligned_relationships

    def align_entities(self, entities: List[KGEntity]) -> Tuple[List[KGEntity], int, int, float]:
        """Classify entity types using standard biomedical categories."""
        for entity in entities:
            if entity.entity_type == "Unknown":
                entity.entity_type = self._classify_entity_type(entity.name)
        return entities, 0, 0, 0.0

    def _classify_entity_type(self, entity_name: str) -> str:
        """Simple rule-based entity type classification."""
        name_lower = entity_name.lower()

        # Drug patterns
        if any(suffix in name_lower for suffix in ['mycin', 'cillin', 'statin', 'inhibitor']):
            return "Drug"

        # Gene patterns
        if entity_name.isupper() and len(entity_name) <= 10:
            return "Gene"

        # Protein patterns
        if any(suffix in name_lower for suffix in ['ase', 'receptor', 'protein']):
            return "Protein"

        # Disease patterns
        if any(keyword in name_lower for keyword in ['cancer', 'disease', 'syndrome', 'disorder']):
            return "Disease"

        return "Chemical"  # Default

    def align_relationships(self, triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """Normalize relationship labels."""
        for triple in triples:
            triple.relation = self._normalize_relation(triple.relation)
        return triples

    def _normalize_relation(self, relation: str) -> str:
        """Standardize relation labels."""
        synonyms = {
            "inhibit": "inhibits", "inhibited": "inhibits",
            "treat": "treats", "treated": "treats",
            "cause": "causes", "caused": "causes",
            "activate": "activates", "activates": "activates",
            "associated with": "associated_with",
            "interacts with": "interacts_with"
        }
        return synonyms.get(relation.lower(), relation.lower())