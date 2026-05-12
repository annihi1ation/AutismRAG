"""Prompt loading for the HyperKG construction workflow.

All prompt text lives in ``prompts.toml``. Agent modules should only ask
for a section and key; they should not carry fallback prompt bodies.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import tomllib


PromptTable = Dict[str, Dict[str, str]]


def _default_prompt_file() -> Path:
    return Path(__file__).with_name("prompts.toml")


def _resolve_prompt_file(prompt_file: str | None = None) -> Path:
    if prompt_file:
        return Path(prompt_file)
    env_path = os.getenv("HYPERKG_PROMPTS_FILE")
    if env_path:
        return Path(env_path)
    return _default_prompt_file()


@lru_cache(maxsize=8)
def load_prompts(prompt_file: str | None = None) -> PromptTable:
    path = _resolve_prompt_file(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt TOML not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    prompts: PromptTable = {}
    for section, value in raw.items():
        if not isinstance(value, dict):
            continue
        prompts[str(section)] = {str(k): str(v) for k, v in value.items()}
    return prompts


def get_agent_config(section: str, prompt_file: str | None = None) -> Dict[str, str]:
    prompts = load_prompts(prompt_file)
    if section not in prompts:
        available = ", ".join(sorted(prompts.keys()))
        raise KeyError(f"Prompt section '{section}' not found. Available: {available}")
    return dict(prompts[section])


def get_prompt(section: str, key: str, prompt_file: str | None = None) -> str:
    config = get_agent_config(section, prompt_file)
    if key not in config:
        available = ", ".join(sorted(config.keys()))
        raise KeyError(
            f"Prompt key '{key}' not found in section '{section}'. Available: {available}"
        )
    return config[key]


def clear_prompt_cache() -> None:
    load_prompts.cache_clear()
