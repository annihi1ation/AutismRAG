"""Base classes for HyperKG construction agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..llm import ChatLLM, PromptConfigurationError
from ..prompt_store import get_agent_config
from ..review_queue import ReviewQueue


@dataclass
class PromptBackedAgent:
    """Base class for agents whose prompts must come from TOML."""

    client: Any
    model_name: str
    prompt_section: str
    prompt_file: Optional[str] = None
    temperature: float = 0.1
    max_retries: int = 1
    review_queue: ReviewQueue = field(default_factory=ReviewQueue)

    def __post_init__(self) -> None:
        config = get_agent_config(self.prompt_section, self.prompt_file)
        self.system_prompt = config.get("system_prompt", "")
        self.prompt_template = config.get("prompt_template", "")
        self.llm = ChatLLM(
            client=self.client,
            model_name=self.model_name,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

    def require_prompt_template(self) -> str:
        if not self.prompt_template.strip():
            raise PromptConfigurationError(
                f"Prompt template for '{self.prompt_section}' is empty. "
                "Fill it in prompts.toml before running this LLM-backed agent."
            )
        return self.prompt_template

    def render_prompt(self, **values: Any) -> str:
        template = self.require_prompt_template()
        try:
            return template.format(**values)
        except KeyError as err:
            raise PromptConfigurationError(
                f"Prompt template for '{self.prompt_section}' is missing placeholder value: {err}"
            ) from err

    @property
    def review_items(self) -> list[Dict[str, Any]]:
        return self.review_queue.to_list()
