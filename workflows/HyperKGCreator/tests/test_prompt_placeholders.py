"""Prompt files are placeholders only; agents work with FakeLLMClient."""

from __future__ import annotations

import pytest

from workflows.HyperKGCreator.llm.agents import PTOBuilderAgent
from workflows.HyperKGCreator.llm.base import FakeLLMClient
from workflows.HyperKGCreator.llm.prompt_loader import (
    PROMPT_FILES,
    PromptConfigurationError,
    PromptLoader,
    is_placeholder,
)
from workflows.HyperKGCreator.routing.candidate_builder import build_agent1_input
from workflows.HyperKGCreator.routing.semantic_router import route_sections
from workflows.HyperKGCreator.config import load_default_config
from workflows.HyperKGCreator.embeddings.section_index import build_section_index


def test_prompt_files_exist_and_are_placeholders():
    loader = PromptLoader()
    for name, filename in PROMPT_FILES.items():
        path = loader.path_for(name)
        assert path.exists(), f"missing prompt file {filename}"
        text = path.read_text(encoding="utf-8")
        assert is_placeholder(text), f"{filename} must remain a placeholder"
        # No extraction instructions leaked in.
        assert "patient" not in text.lower()
        assert "extract" not in text.lower()


def test_real_client_on_placeholder_raises():
    loader = PromptLoader()
    with pytest.raises(PromptConfigurationError):
        loader.load("pto_builder", allow_placeholder=False)


def test_fake_client_allows_placeholder():
    loader = PromptLoader()
    assert is_placeholder(loader.load("pto_builder", allow_placeholder=True))


def test_agent_runs_with_fake_llm(mini_article, hashing_model, fake_llm):
    cfg = load_default_config()
    section_index = build_section_index(mini_article, hashing_model)
    bundle = route_sections(mini_article, section_index, cfg, hashing_model)
    pto_input = build_agent1_input(mini_article, bundle)

    agent = PTOBuilderAgent(fake_llm, PromptLoader())
    graph = agent.build(pto_input)
    assert graph.article_id == mini_article.metadata.article_id
    assert graph.pto_events and graph.pto_events[0].treatment_node_ids
