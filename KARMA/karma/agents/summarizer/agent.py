"""
Summarizer Agent Implementation

The Summarizer Agent converts high-relevance text segments into concise summaries
while preserving technical details crucial for knowledge extraction.
"""

import logging
from typing import List, Tuple

from karma.core.base_agent import BaseAgent
from karma.agents.prompt_loader import get_agent_config

logger = logging.getLogger(__name__)


class SummarizerAgent(BaseAgent):
    """
    Summarizer Agent (SA) for text segment summarization.

    This agent:
    1. Converts high-relevance segments into concise summaries
    2. Preserves technical details (gene symbols, chemical names, numeric data)
    3. Maintains entity relationships and quantitative information
    4. Filters out very low relevance content
    """

    def __init__(self, client, model_name: str):
        """
        Initialize the Summarizer Agent.

        Args:
            client: OpenAI/OpenRouter client instance
            model_name: LLM model identifier
        """
        config = get_agent_config("summarizer")
        system_prompt = config.get("system_prompt", "")
        self.prompt_template = config.get("prompt_template", "")

        super().__init__(client, model_name, system_prompt)

    def process(self, segments: List[str], relevance_threshold: float = 0.2) -> List[str]:
        """
        Summarize a list of text segments.

        Args:
            segments: List of text segments to summarize
            relevance_threshold: Minimum relevance to process segment

        Returns:
            List of summaries
        """
        summaries = []

        for segment in segments:
            if isinstance(segment, dict):
                text = segment.get('text', '')
                relevance = segment.get('score', 1.0)
            else:
                text = segment
                relevance = 1.0  # Assume relevant if no score provided

            if relevance < relevance_threshold:
                summaries.append("[OMITTED - Low Relevance]")
                continue

            summary = self._summarize_single_segment(text)
            summaries.append(summary)

        return summaries

    def _summarize_single_segment(self, text: str) -> str:
        """
        Summarize a single text segment.

        Args:
            text: Text segment to summarize

        Returns:
            Summarized text
        """
        # Skip very short segments
        if len(text.split()) < 15:
            return text  # Return as-is if too short to meaningfully summarize

        if self.prompt_template:
            prompt = self.prompt_template.replace("{text}", text)
        else:
            prompt = f"""
            Summarize the following biomedical text in 2-4 sentences, keeping it under 100 words.

            Critical Requirements:
            - Retain ALL technical terms (genes, proteins, drugs, diseases, chemicals)
            - Preserve ALL numeric data (concentrations, p-values, percentages, doses)
            - Keep relationship indicators (inhibits, activates, treats, causes, etc.)
            - Maintain scientific precision and accuracy
            - Use clear, unambiguous language

            If the text contains very little scientific information, provide a brief summary or return "[LOW CONTENT]".

            Provide only the summary with no additional text, formatting, or explanations.

            Text to summarize:
            {text}
            """

        try:
            summary, _, _, _ = self._make_llm_call(prompt, temperature=0.2)

            # Basic validation and cleanup
            summary = summary.strip()

            # Handle empty or low-quality responses
            if not summary or summary.lower() in ['[low content]', 'low content', 'n/a']:
                # Fallback: extract key sentences
                return self._extract_key_sentences(text)

            # Ensure summary is within length limit
            if len(summary.split()) > 120:  # Allow some flexibility
                # Truncate while preserving sentence structure
                sentences = summary.split('. ')
                truncated_summary = ""
                word_count = 0

                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    if word_count + sentence_words <= 100:
                        truncated_summary += sentence + ". "
                        word_count += sentence_words
                    else:
                        break

                return truncated_summary.strip() if truncated_summary else summary

            return summary

        except Exception as e:
            logger.warning(f"Summarization failed: {str(e)}")
            return self._extract_key_sentences(text)

    def _extract_key_sentences(self, text: str) -> str:
        """
        Extract key sentences from text as a fallback summarization method.

        Args:
            text: Original text

        Returns:
            Key sentences extracted from text
        """
        sentences = text.split('. ')

        # Score sentences based on biomedical content
        scored_sentences = []

        for sentence in sentences:
            score = self._score_sentence_importance(sentence)
            scored_sentences.append((sentence, score))

        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences within word limit
        selected_sentences = []
        word_count = 0

        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= 100 and score > 0.3:
                selected_sentences.append(sentence)
                word_count += sentence_words

        if selected_sentences:
            return '. '.join(selected_sentences) + '.'
        else:
            # Return first 100 words as last resort
            words = text.split()[:100]
            return ' '.join(words) + '...'

    def _score_sentence_importance(self, sentence: str) -> float:
        """
        Score the importance of a sentence for biomedical knowledge extraction.

        Args:
            sentence: Sentence to score

        Returns:
            Importance score (0-1)
        """
        sentence_lower = sentence.lower()
        score = 0.0

        # High-value biomedical terms
        high_value_terms = [
            'inhibit', 'activate', 'regulate', 'express', 'bind', 'interact',
            'cause', 'treat', 'prevent', 'induce', 'suppress', 'enhance',
            'protein', 'gene', 'enzyme', 'receptor', 'pathway', 'mechanism',
            'disease', 'cancer', 'tumor', 'therapy', 'treatment', 'drug',
            'significant', 'increase', 'decrease', 'effect', 'response'
        ]

        # Count high-value terms
        for term in high_value_terms:
            if term in sentence_lower:
                score += 0.1

        # Bonus for numeric data
        import re
        if re.search(r'\d+\.?\d*\s*(%|mg|μg|ng|mM|μM|nM|p\s*[<>=])', sentence):
            score += 0.3

        # Bonus for entity mentions (capitalized terms)
        capitalized_words = re.findall(r'\b[A-Z][A-Za-z0-9-]+\b', sentence)
        score += len(capitalized_words) * 0.05

        # Penalty for very short or very long sentences
        word_count = len(sentence.split())
        if word_count < 5:
            score *= 0.5
        elif word_count > 50:
            score *= 0.8

        return min(1.0, score)

    def summarize_segment(self, segment: str) -> Tuple[str, int, int, float]:
        """
        Legacy method for backward compatibility.

        Args:
            segment: Text segment to summarize

        Returns:
            Tuple of (summary, prompt_tokens, completion_tokens, processing_time)
        """
        summary = self._summarize_single_segment(segment)
        return summary, 0, 0, 0.0  # Token counts handled internally now