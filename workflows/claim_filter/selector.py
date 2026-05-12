"""LLM-driven top-1 claim selection."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_LOW_PRIORITY_TYPES = {
    "METHOD_DESIGN",
    "OTHER",
    "PREVALENCE",
    "SERVICE_ACCESS",
}


class RpmLimiter:
    """Thread-safe sliding-window RPM limiter."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self._window: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._window and now - self._window[0] >= 60.0:
                    self._window.popleft()
                if len(self._window) < self.rpm:
                    self._window.append(now)
                    return
                wait = 60.0 - (now - self._window[0]) + 0.05
            time.sleep(wait)


def project_minimal(claims: list[dict]) -> list[dict]:
    """Strip every claim down to the three fields the selector is allowed to see."""
    minimal = []
    for c in claims:
        cid = c.get("claim_id")
        if not cid:
            continue
        minimal.append(
            {
                "claim_id": cid,
                "claim_text": (c.get("claim_text") or "").strip(),
                "claim_type": c.get("claim_type") or "UNKNOWN",
            }
        )
    return minimal


def build_user_prompt(claims_min: list[dict]) -> str:
    lines = ["Pick exactly one claim_id from the list below.", ""]
    for c in claims_min:
        text = c["claim_text"].replace("\n", " ").replace("\r", " ").strip()
        lines.append(f"{c['claim_id']} | {c['claim_type']} | {text}")
    return "\n".join(lines)


def _sanitize(raw: str, valid_ids: set[str]) -> str:
    if not raw:
        return ""
    s = raw.strip()
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break
    s = s.strip().strip("`").strip().strip('"').strip("'").strip()
    if not s:
        return ""
    token = s.split()[0]
    if token in valid_ids:
        return token
    if not token.startswith("C:") and ("C:" + token) in valid_ids:
        return "C:" + token
    return token


def _fallback_pick(claims_min: list[dict]) -> str:
    for c in claims_min:
        if c["claim_type"] not in _LOW_PRIORITY_TYPES:
            return c["claim_id"]
    return claims_min[0]["claim_id"]


def select_claim_id(
    client: Any,
    model: str,
    claims_min: list[dict],
    *,
    max_retries: int = 1,
    limiter: RpmLimiter | None = None,
) -> str | None:
    """Return the chosen claim_id, or None if there are no claims."""
    if not claims_min:
        return None
    if len(claims_min) == 1:
        return claims_min[0]["claim_id"]

    valid_ids = {c["claim_id"] for c in claims_min}
    user_prompt = build_user_prompt(claims_min)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_raw = ""
    for attempt in range(max_retries + 1):
        if limiter is not None:
            limiter.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            last_raw = resp.choices[0].message.content or ""
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "selector LLM call failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                err,
            )
            continue
        cid = _sanitize(last_raw, valid_ids)
        if cid in valid_ids:
            return cid
        logger.warning(
            "selector returned invalid claim_id %r (raw=%r); attempt %d/%d",
            cid,
            last_raw[:200],
            attempt + 1,
            max_retries + 1,
        )

    chosen = _fallback_pick(claims_min)
    logger.warning("selector falling back to deterministic pick: %s", chosen)
    return chosen
