"""LLM helpers for HyperKG agents."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL_ENV = "OPENROUTER_BASE_URL"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


class PromptConfigurationError(RuntimeError):
    """Raised when an LLM agent is used without a TOML prompt template."""


class OpenRouterConfigurationError(RuntimeError):
    """Raised when the OpenRouter client cannot be configured."""


def _candidate_env_files(env_file: str | None = None) -> list[Path]:
    if env_file:
        return [Path(env_file)]
    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    for directory in [cwd, *cwd.parents]:
        candidates.append(directory / ".env")
    return candidates


def load_env_file(env_file: str | None = None) -> Optional[Path]:
    """Load a .env file without requiring python-dotenv.

    Existing process environment variables win, so a shell-provided key is
    never overwritten by a file value.
    """
    for path in _candidate_env_files(env_file):
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        return path
    return None


def get_openrouter_api_key(
    api_key: str | None = None,
    env_file: str | None = None,
) -> str:
    if api_key:
        return api_key
    load_env_file(env_file)
    key = os.environ.get(OPENROUTER_API_KEY_ENV, "")
    if not key:
        raise OpenRouterConfigurationError(
            f"{OPENROUTER_API_KEY_ENV} is not set. Add it to .env or pass --api-key."
        )
    return key


def get_openrouter_base_url(
    base_url: str | None = None,
    env_file: str | None = None,
) -> str:
    if base_url:
        return base_url
    load_env_file(env_file)
    return os.environ.get(OPENROUTER_BASE_URL_ENV, DEFAULT_OPENROUTER_BASE_URL)


def extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for chunk in candidates:
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None


@dataclass
class ChatLLM:
    """Small OpenAI-compatible chat wrapper."""

    client: Any
    model_name: str
    temperature: float = 0.1
    max_retries: int = 1

    def chat(self, system_prompt: str, user_prompt: str, want_json: bool = True) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("LLM client is required for this HyperKG agent.")
        if not self.model_name:
            raise RuntimeError("model_name is required for this HyperKG agent.")

        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                text = resp.choices[0].message.content or ""
                return {"text": text, "json": extract_json(text) if want_json else None}
            except Exception as err:  # noqa: BLE001
                last_err = err
                logger.warning(
                    "HyperKG LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    err,
                )
        raise RuntimeError(f"HyperKG LLM call failed after retries: {last_err}")


def build_openai_client(api_key: str, base_url: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def build_openrouter_client(
    api_key: str | None = None,
    base_url: str | None = None,
    env_file: str | None = None,
):
    """Build an OpenRouter-compatible OpenAI client from args or .env."""
    return build_openai_client(
        api_key=get_openrouter_api_key(api_key=api_key, env_file=env_file),
        base_url=get_openrouter_base_url(base_url=base_url, env_file=env_file),
    )
