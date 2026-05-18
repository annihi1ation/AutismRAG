"""LLM client abstraction.

- ``BaseLLMClient``: ``generate_json(prompt, variables) -> dict``.
- ``FakeLLMClient``: deterministic, for tests (no network, no real model).
- ``OpenRouterLLMClient``: reuses ``workflows.HyperKGConstruction.llm``
  (``build_openrouter_client`` + ``ChatLLM`` + ``extract_json``) — the accepted
  repo pattern (HyperRAG cross-imports it too).

No prompt content is generated in code. Owner prompts use ``{{var}}``
double-brace placeholders so JSON braces in a prompt are never touched.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
_SINGLE_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{\s*([A-Za-z0-9_]+)\s*\}(?!\})")


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    """Substitute prompt variables; non-string values are JSON-encoded.

    Owner prompts in this workflow currently use ``{input_json}``, while the
    package convention is ``{{name}}``. Support both without applying general
    Python ``str.format`` semantics, because prompt examples contain JSON
    braces.
    """

    def _sub(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in variables:
            return match.group(0)
        val = variables[key]
        return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)

    rendered = _PLACEHOLDER_RE.sub(_sub, template)
    rendered = _SINGLE_PLACEHOLDER_RE.sub(_sub, rendered)
    return rendered.replace("{{", "{").replace("}}", "}")


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


class BaseLLMClient(ABC):
    is_fake: bool = False

    @abstractmethod
    def _complete(self, prompt: str) -> str:
        """Return raw model text for a fully-rendered prompt."""

    def generate_json(self, prompt: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        rendered = render_prompt(prompt, variables)
        text = self._complete(rendered)
        from .._json import extract_json  # local import to avoid cycle

        parsed = extract_json(text)
        if parsed is None:
            raise ValueError("LLM did not return parseable JSON")
        if isinstance(parsed, list):
            return {"items": parsed}
        return parsed


class FakeLLMClient(BaseLLMClient):
    """Deterministic test client.

    ``handlers`` maps an agent key (passed by agents as ``variables["__agent__"]``)
    to either a static dict or a ``callable(variables) -> dict``.
    """

    is_fake = True

    def __init__(
        self,
        handlers: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.handlers: Dict[str, Any] = handlers or {}
        self.calls: list[Dict[str, Any]] = []

    def set_response(self, agent_key: str, payload: Any) -> None:
        self.handlers[agent_key] = payload

    def _complete(self, prompt: str) -> str:  # pragma: no cover - unused path
        return "{}"

    def generate_json(self, prompt: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        agent_key = variables.get("__agent__", "")
        self.calls.append({"agent": agent_key, "variables": variables})
        handler = self.handlers.get(agent_key)
        if handler is None:
            raise KeyError(f"FakeLLMClient has no handler for agent '{agent_key}'")
        result = handler(variables) if callable(handler) else handler
        return result


class OpenRouterLLMClient(BaseLLMClient):
    """Thin wrapper over the repo's OpenRouter helpers."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        max_retries: int = 1,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        from workflows.HyperKGConstruction.llm import (  # type: ignore
            ChatLLM,
            build_openrouter_client,
        )

        client = build_openrouter_client(api_key=api_key, base_url=base_url)
        self._chat = ChatLLM(
            client=client,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
        )
        self.model_name = model_name
        self.records: list[Dict[str, Any]] = []
        self._dump_path = os.environ.get("HKG_LLM_DUMP_PATH")

    def _complete(self, prompt: str) -> str:
        result = self._chat.chat(system_prompt="", user_prompt=prompt, want_json=True)
        text = result.get("text", "") or ""
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name,
            "prompt_chars": len(prompt),
            "response_chars": len(text),
            "response_id": result.get("id"),
            "usage": _jsonable(result.get("usage")),
            "prompt": prompt,
            "raw_response": text,
        }
        self.records.append(record)
        if self._dump_path:
            path = Path(self._dump_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return text
