"""Small shared helpers for HyperKG construction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from hashlib import sha1
from typing import Any, Iterable


_SPACE_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip()).lower()


def slugify(value: Any, max_len: int = 80) -> str:
    text = normalize_text(value)
    text = _SLUG_RE.sub("_", text).strip("_")
    return (text or "unknown")[:max_len]


def stable_hash(value: Any, length: int = 12) -> str:
    payload = compact_json(value)
    return sha1(payload.encode("utf-8")).hexdigest()[:length]


def compact_json(value: Any) -> str:
    return json.dumps(to_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(to_plain(value), ensure_ascii=False, sort_keys=True, indent=2)


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(v) for v in value]
    return value


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def unique_preserve_order(values: Iterable[Any]) -> list:
    seen: set[str] = set()
    out: list = []
    for value in values:
        key = compact_json(value) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clip01(value: Any, default: float = 0.0) -> float:
    x = safe_float(value, default)
    return max(0.0, min(1.0, x))
