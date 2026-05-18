"""Loads owner-supplied prompt text files.

Prompt files ship as placeholders containing only:
    TODO: project owner will write this prompt.
A real (non-Fake) client meeting an empty/placeholder prompt raises
``PromptConfigurationError``. No prompt content is generated in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PLACEHOLDER_TEXT = "TODO: project owner will write this prompt."

PROMPT_FILES = {
    "pto_builder": "pto_builder.txt",
    "hyperedge_builder_entity_selector": "hyperedge_builder_entity_selector.txt",
    "conditional_verifier": "conditional_verifier.txt",
}


class PromptConfigurationError(RuntimeError):
    """Raised when a required prompt is an empty placeholder."""


def is_placeholder(text: Optional[str]) -> bool:
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if stripped == PLACEHOLDER_TEXT:
        return True
    return stripped.upper().startswith("TODO")


class PromptLoader:
    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self.prompts_dir = Path(prompts_dir) if prompts_dir else (
            Path(__file__).resolve().parent.parent / "prompts"
        )

    def path_for(self, name: str) -> Path:
        filename = PROMPT_FILES.get(name, f"{name}.txt")
        return self.prompts_dir / filename

    def load(self, name: str, allow_placeholder: bool = False) -> str:
        path = self.path_for(name)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if is_placeholder(text):
            if allow_placeholder:
                return text
            raise PromptConfigurationError(
                f"Prompt '{name}' at {path} is an empty placeholder. The project "
                "owner must supply this prompt, or pass prompt_override / use a "
                "FakeLLMClient for tests."
            )
        return text
