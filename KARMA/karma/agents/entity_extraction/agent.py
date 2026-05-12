"""
Entity Extraction Agent Implementation

The Entity Extraction Agent identifies and classifies biomedical entities
from summarized text using LLM-based named entity recognition.
"""

import logging
from typing import List, Tuple, Dict
import json
import re

from karma.core.base_agent import BaseAgent
from karma.core.data_structures import KGEntity
from karma.agents.prompt_loader import get_agent_config

logger = logging.getLogger(__name__)


class EntityExtractionAgent(BaseAgent):
    """
    Entity Extraction Agent (EEA) for biomedical entity identification.

    This agent:
    1. Identifies biomedical entities (Disease, Drug, Gene, Protein, Chemical, etc.)
    2. Classifies entity types using domain knowledge
    3. Links entities to canonical ontology references where possible
    4. Handles entity normalization and synonym resolution
    """

    def __init__(self, client, model_name: str):
        """
        Initialize the Entity Extraction Agent.

        Args:
            client: OpenAI/OpenRouter client instance
            model_name: LLM model identifier
        """
        config = get_agent_config("entity_extraction")
        system_prompt = config.get("system_prompt", "")
        self.prompt_template = config.get("prompt_template", None)

        super().__init__(client, model_name, system_prompt)

    def process(self, summaries: List[str]) -> List[KGEntity]:
        """
        Extract entities from a list of text summaries.

        Args:
            summaries: List of text summaries to process

        Returns:
            List of extracted and deduplicated entities
        """
        all_entities = []

        for summary in summaries:
            if summary and summary not in ["[OMITTED - Low Relevance]", "[LOW CONTENT]"]:
                entities = self._extract_entities_from_text(summary)
                all_entities.extend(entities)

        # Deduplicate entities
        unique_entities = self._deduplicate_entities(all_entities)

        return unique_entities

    def _extract_entities_from_text(self, text: str) -> List[KGEntity]:
        """
        Extract entities from a single text.

        Args:
            text: Text to extract entities from

        Returns:
            List of KGEntity objects
        """
        if self.prompt_template:
            prompt = self.prompt_template.replace("{text}", text)
        else:
            prompt = f"""
            Extract biomedical entities from the text below.
            Include diseases, drugs, genes, proteins, chemicals, pathways, cells, tissues, and other relevant biomedical entities.

            For each entity found:
            1. Identify the exact mention in the text
            2. Assign the most specific entity type
            3. Provide a normalized ID if you know a standard reference (e.g., MESH:D001241 for Aspirin)
            4. Include common aliases if applicable

            Text to analyze:
            {text}

            Return only a JSON array of entities, no other text:
            """

        try:
            response, _, _, _ = self._make_llm_call(prompt, temperature=0.1)

            # Parse JSON response
            entities_data = self._parse_json_response(response)

            entity_list = []
            for ent_data in entities_data:
                if isinstance(ent_data, dict) and "mention" in ent_data:
                    entity = KGEntity(
                        entity_id=ent_data.get("mention", ""),
                        entity_type=ent_data.get("type", "Unknown"),
                        name=ent_data.get("mention", ""),
                        normalized_id=ent_data.get("normalized_id", "N/A"),
                        aliases=ent_data.get("aliases", [])
                    )
                    entity_list.append(entity)

            return entity_list

        except Exception as e:
            logger.warning(f"Entity extraction failed for text: {str(e)}")
            # Fallback: use simple regex-based extraction
            return self._fallback_entity_extraction(text)

    def _fallback_entity_extraction(self, text: str) -> List[KGEntity]:
        """
        Fallback entity extraction using pattern matching.

        Args:
            text: Text to extract entities from

        Returns:
            List of KGEntity objects
        """
        entities = []

        # Common biomedical entity patterns
        patterns = {
            'Gene': [
                r'\\b[A-Z][A-Z0-9]{2,}[a-z]?\\b',  # Gene symbols (e.g., TP53, BRCA1)
                r'\\b[A-Z]{1,2}[0-9]+[A-Z]?\\b'     # Gene IDs (e.g., IL6, TNF)
            ],
            'Protein': [
                r'\\b[A-Z][a-z]+[0-9]*\\s*receptor\\b',  # Receptors
                r'\\b[A-Z][a-z]+ase\\b',                # Enzymes ending in -ase
                r'\\b[A-Z][a-z]+in\\b'                  # Proteins ending in -in
            ],
            'Drug': [
                r'\\b[a-z]+mycin\\b',                   # Antibiotics
                r'\\b[a-z]+cillin\\b',                  # Penicillins
                r'\\b[a-z]+statin\\b'                   # Statins
            ],
            'Disease': [
                r'\\b[a-z]+\\s+cancer\\b',               # Cancers
                r'\\b[a-z]+\\s+disease\\b',              # Diseases
                r'\\b[a-z]+\\s+syndrome\\b'              # Syndromes
            ]
        }

        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    mention = match.group().strip()
                    if len(mention) > 2:  # Skip very short matches
                        entity = KGEntity(
                            entity_id=mention,
                            entity_type=entity_type,
                            name=mention,
                            normalized_id="N/A"
                        )
                        entities.append(entity)

        return entities

    def _deduplicate_entities(self, entities: List[KGEntity]) -> List[KGEntity]:
        """
        Remove duplicate entities based on name similarity.

        Args:
            entities: List of entities to deduplicate

        Returns:
            List of unique entities
        """
        if not entities:
            return []

        unique_entities = {}

        for entity in entities:
            # Create a normalized key for comparison
            key = entity.name.lower().strip()

            # If we've seen this entity before, merge information
            if key in unique_entities:
                existing = unique_entities[key]

                # Prefer more specific entity types
                if entity.entity_type != "Unknown" and existing.entity_type == "Unknown":
                    existing.entity_type = entity.entity_type

                # Merge aliases
                if entity.aliases:
                    existing.aliases.extend(entity.aliases)
                    existing.aliases = list(set(existing.aliases))  # Remove duplicates

                # Prefer non-N/A normalized IDs
                if entity.normalized_id != "N/A" and existing.normalized_id == "N/A":
                    existing.normalized_id = entity.normalized_id

            else:
                unique_entities[key] = entity

        return list(unique_entities.values())

    def extract_entities(self, text: str) -> Tuple[List[KGEntity], int, int, float]:
        """
        Legacy method for backward compatibility.

        Args:
            text: Text to extract entities from

        Returns:
            Tuple of (entity_list, prompt_tokens, completion_tokens, processing_time)
        """
        entities = self._extract_entities_from_text(text)
        return entities, 0, 0, 0.0  # Token counts handled internally now
