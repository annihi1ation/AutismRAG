"""Robust JSON extraction from possibly-noisy LLM text.

Self-contained (mirrors the repo's ``HyperKGConstruction.llm.extract_json``)
so the Fake/test path needs no heavy imports.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


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
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return None
