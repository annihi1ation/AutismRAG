"""Progress reporting helpers for HyperKG construction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProgressReporter:
    phase: str
    total: int
    enabled: bool = True
    logger: logging.Logger = logging.getLogger(__name__)
    log_interval_sec: float = 30.0
    unit: str = "item"

    def __post_init__(self) -> None:
        self.done = 0
        self.started_at = time.time()
        self.last_log_at = self.started_at
        self._bar: Optional[object] = None

    def __enter__(self) -> "ProgressReporter":
        if self.enabled and self.total > 0:
            try:
                from tqdm import tqdm

                self._bar = tqdm(total=self.total, desc=self.phase, unit=self.unit)
            except Exception as err:  # noqa: BLE001
                self.logger.debug("tqdm progress disabled for phase=%s: %s", self.phase, err)
                self._bar = None
        return self

    def update(self, amount: int = 1, force_log: bool = False) -> None:
        if amount <= 0 and not force_log:
            return
        self.done += max(amount, 0)
        if self._bar is not None and amount > 0:
            self._bar.update(amount)  # type: ignore[attr-defined]
        now = time.time()
        if force_log or now - self.last_log_at >= self.log_interval_sec:
            elapsed = max(now - self.started_at, 1e-9)
            rate = self.done / elapsed
            percent = (self.done / self.total * 100.0) if self.total else 100.0
            self.logger.info(
                "HyperKG phase=%s progress=%d/%d percent=%.1f rate=%.2f/%s",
                self.phase,
                self.done,
                self.total,
                percent,
                rate,
                self.unit,
            )
            self.last_log_at = now

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.update(0, force_log=True)
        if self._bar is not None:
            self._bar.close()  # type: ignore[attr-defined]
