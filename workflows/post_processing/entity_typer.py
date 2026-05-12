"""
EntityTyper – recovers entity types for flat entity strings by calling
an LLM (Gemini Flash via OpenRouter) in batches.

The per-article _kg.json files only store entity names as plain strings;
the rich type information (Disorder, Medication, Therapeutic_Approach …)
was lost during KARMA's export step.  This agent re-classifies every
unique entity string into the 21 standardized categories from the KARMA
ontology.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── canonical entity categories (from KARMA prompts.toml) ──────────────
ENTITY_CATEGORIES = [
    "Gene",
    "Neurobiology",
    "Developmental_Stage",
    "Disorder",
    "Clinical_Severity",
    "Trauma",
    "Symptom_Behavior",
    "Symptom_Emotion",
    "Functional_Ability",
    "Assessment_Tool",
    "Diagnostic_Framework",
    "Medication",
    "Therapeutic_Approach",
    "Educational_Approach",
    "Person_Role",
    "Social_Environment",
    "Intersectional_Identity",
    "Healthcare_Access",
    "Institution_System",
    "Socioeconomic",
    "Other",
]

ENTITY_TYPING_SYSTEM_PROMPT = """You are a specialized entity classifier for an Intellectual Disability (ID) Knowledge Graph.

Given a list of entity names extracted from clinical, psychosocial, and biomedical literature on intellectual disability, classify each entity into EXACTLY ONE of the following categories:

CATEGORIES:
- Gene: genetic elements, chromosomal conditions (e.g., FMR1, trisomy 21)
- Neurobiology: brain structures, neural mechanisms (e.g., hippocampus, synaptic dysfunction)
- Developmental_Stage: temporal phases of human development (e.g., early childhood, adolescence)
- Disorder: medical, neurological, psychiatric conditions (e.g., epilepsy, ASD, intellectual disability)
- Clinical_Severity: severity levels (e.g., mild ID, severe, profound)
- Trauma: harmful events with lasting impact (e.g., neglect, domestic violence)
- Symptom_Behavior: observable behavioral manifestations (e.g., aggression, self-injury, stereotypy)
- Symptom_Emotion: internal affective states (e.g., anxiety, anger, panic)
- Functional_Ability: communication, cognition, adaptive skills (e.g., expressive language, daily living skills)
- Assessment_Tool: standardized instruments (e.g., Vineland, ABC, IQ tests, MMAT)
- Diagnostic_Framework: classification systems (e.g., DSM-5, ICD-11)
- Medication: pharmacological agents (e.g., risperidone, antipsychotics)
- Therapeutic_Approach: behavioral/psychological interventions (e.g., ABA, CBT, positive behavior support)
- Educational_Approach: educational/vocational programs (e.g., special education, early intervention)
- Person_Role: human actors (e.g., caregiver, clinician, parent, teacher)
- Social_Environment: interpersonal/caregiving contexts (e.g., family environment, peer interaction)
- Intersectional_Identity: identity aspects (e.g., gender, race, socioeconomic class)
- Healthcare_Access: service availability (e.g., therapy access, clinical support programs)
- Institution_System: organizations/systems (e.g., hospital, school system, residential care)
- Socioeconomic: economic factors (e.g., poverty, employment status)
- Other: entities that don't fit any above category

Also provide:
- normalized_id: Use MeSH, NCBI, HGNC, OMIM, UBERON, or ICD codes if confidently known; otherwise "N/A"
- aliases: known synonyms or abbreviations for this entity (only commonly used ones)

RULES:
- Classify based on the entity's PRIMARY role in ID literature
- Prefer specific categories over "Other"
- Be consistent: similar entities should get the same type
- When in doubt between two categories, pick the more specific one

OUTPUT FORMAT:
Return ONLY a valid JSON array:
[{"name": "entity_name", "type": "Category", "normalized_id": "...", "aliases": [...]}]
"""


@dataclass
class TypedEntity:
    """An entity with recovered type information."""
    name: str
    entity_type: str = "Other"
    normalized_id: str = "N/A"
    aliases: List[str] = field(default_factory=list)
    source_articles: List[str] = field(default_factory=list)
    mention_count: int = 0


class EntityTyper:
    """
    Batch-classifies entity strings into ontology categories using an LLM.

    Parameters
    ----------
    client : OpenAI
        OpenAI-compatible client (e.g. OpenRouter).
    model_name : str
        Model identifier (e.g. ``google/gemini-3-flash-preview``).
    batch_size : int
        Number of entities per LLM call (default 40).
    temperature : float
        Sampling temperature for the LLM.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str = "google/gemini-3-flash-preview",
        batch_size: int = 40,
        temperature: float = 0.05,
    ):
        self.client = client
        self.model_name = model_name
        self.batch_size = batch_size
        self.temperature = temperature

    # ── public API ──────────────────────────────────────────────────────
    def classify_entities(
        self,
        entity_sources: Dict[str, List[str]],
    ) -> Dict[str, TypedEntity]:
        """
        Classify all unique entity strings.

        Parameters
        ----------
        entity_sources : dict[str, list[str]]
            Mapping ``entity_name -> [source_article_ids]``
            (produced by ``loader.collect_all_entities``).

        Returns
        -------
        dict[str, TypedEntity]
            Mapping ``entity_name -> TypedEntity`` with recovered type info.
        """
        all_names = sorted(entity_sources.keys())
        logger.info("Classifying %d unique entities in batches of %d", len(all_names), self.batch_size)

        typed: Dict[str, TypedEntity] = {}
        for batch_start in range(0, len(all_names), self.batch_size):
            batch = all_names[batch_start : batch_start + self.batch_size]
            batch_num = batch_start // self.batch_size + 1
            total_batches = (len(all_names) + self.batch_size - 1) // self.batch_size
            logger.info("  Batch %d/%d (%d entities)", batch_num, total_batches, len(batch))

            results = self._classify_batch(batch)
            for item in results:
                name = item.get("name", "")
                if not name:
                    continue
                # Find the matching original name (case-insensitive)
                matched_name = self._match_original_name(name, batch)
                if matched_name is None:
                    matched_name = name

                entity_type = item.get("type", "Other")
                if entity_type not in ENTITY_CATEGORIES:
                    entity_type = "Other"

                typed[matched_name] = TypedEntity(
                    name=matched_name,
                    entity_type=entity_type,
                    normalized_id=item.get("normalized_id", "N/A"),
                    aliases=item.get("aliases", []),
                    source_articles=entity_sources.get(matched_name, []),
                    mention_count=len(entity_sources.get(matched_name, [])),
                )

            # Ensure every entity in the batch gets a result even if LLM missed it
            for name in batch:
                if name not in typed:
                    typed[name] = TypedEntity(
                        name=name,
                        entity_type="Other",
                        source_articles=entity_sources.get(name, []),
                        mention_count=len(entity_sources.get(name, [])),
                    )

        logger.info(
            "Entity typing complete: %d typed (%d categories used)",
            len(typed),
            len({e.entity_type for e in typed.values()}),
        )
        return typed

    # ── internals ───────────────────────────────────────────────────────
    def _classify_batch(self, names: List[str]) -> List[Dict]:
        """Call LLM to classify a batch of entity names."""
        entity_list = "\n".join(f"- {n}" for n in names)
        prompt = (
            f"Classify the following {len(names)} entities from Intellectual Disability literature.\n\n"
            f"Entities:\n{entity_list}\n\n"
            "Return ONLY a JSON array with one object per entity:\n"
            '[{"name": "...", "type": "...", "normalized_id": "...", "aliases": [...]}]'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": ENTITY_TYPING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            content = response.choices[0].message.content.strip()
            return self._parse_json_array(content)
        except Exception as e:
            logger.error("LLM batch classification failed: %s", e)
            return []

    @staticmethod
    def _parse_json_array(text: str) -> List[Dict]:
        """Extract and parse a JSON array from LLM output."""
        # Strip markdown fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        try:
            if "[" in text and "]" in text:
                json_str = text[text.find("[") : text.rfind("]") + 1]
                return json.loads(json_str)
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error: %s", e)
            return []

    @staticmethod
    def _match_original_name(returned_name: str, batch: List[str]) -> Optional[str]:
        """
        Match a name returned by the LLM to the original name in the batch
        (handles minor casing / whitespace differences).
        """
        lower = returned_name.lower().strip()
        for original in batch:
            if original.lower().strip() == lower:
                return original
        return None
