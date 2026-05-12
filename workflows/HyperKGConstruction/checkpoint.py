"""Checkpoint persistence for long HyperKG construction runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .utils import stable_hash, to_plain


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely reused."""


JsonDict = Dict[str, Any]


@dataclass
class HyperKGCheckpointStore:
    checkpoint_dir: str
    enabled: bool = True
    manifest: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.paths = {
            "manifest": os.path.join(self.checkpoint_dir, "manifest.json"),
            "packets": os.path.join(self.checkpoint_dir, "packets.jsonl"),
            "claim_split": os.path.join(self.checkpoint_dir, "claim_split_results.jsonl"),
            "compose": os.path.join(self.checkpoint_dir, "compose_results.jsonl"),
            "critic": os.path.join(self.checkpoint_dir, "critic_results.jsonl"),
            "merge": os.path.join(self.checkpoint_dir, "merge_state.json"),
        }

    def prepare(self, fingerprint: JsonDict, resume_enabled: bool = True) -> JsonDict:
        if not self.enabled:
            return {"enabled": False, "resumed": False}

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        existing = self._read_json(self.paths["manifest"])
        if existing and resume_enabled:
            if existing.get("fingerprint") != fingerprint:
                raise CheckpointError(
                    "Existing HyperKG checkpoint fingerprint does not match this run. "
                    "Use --no-resume or a new --output-dir/--checkpoint-dir."
                )
            self.manifest = existing
            self.manifest["resumed_at"] = time.time()
            self._write_manifest()
            return {"enabled": True, "resumed": True, "checkpoint_dir": self.checkpoint_dir}

        self._clear_artifacts()
        self.manifest = {
            "version": 1,
            "checkpoint_id": stable_hash({"fingerprint": fingerprint, "started_at": time.time()}),
            "created_at": time.time(),
            "fingerprint": fingerprint,
            "phases": {},
        }
        self._write_manifest()
        return {"enabled": True, "resumed": False, "checkpoint_dir": self.checkpoint_dir}

    def mark_phase(
        self,
        phase: str,
        status: str,
        count: int = 0,
        skipped: int = 0,
        duration_sec: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        phase_row = {
            "status": status,
            "count": count,
            "skipped": skipped,
            "updated_at": time.time(),
        }
        if duration_sec is not None:
            phase_row["duration_sec"] = round(duration_sec, 2)
        self.manifest.setdefault("phases", {})[phase] = phase_row
        self._write_manifest()

    def append_record(self, phase: str, row: JsonDict) -> None:
        if not self.enabled:
            return
        path = self.paths[phase]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_plain(row), ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_records(self, phase: str, rows: Iterable[JsonDict]) -> None:
        if not self.enabled:
            return
        path = self.paths[phase]
        self._write_jsonl_atomic(path, rows)

    def load_records(
        self,
        phase: str,
        key: str = "object_id",
        validate: Optional[Callable[[JsonDict], bool]] = None,
    ) -> Dict[str, JsonDict]:
        if not self.enabled:
            return {}
        path = self.paths[phase]
        if not os.path.exists(path):
            return {}
        rows: Dict[str, JsonDict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if validate is not None and not validate(row):
                    continue
                object_id = str(row.get(key) or "")
                if object_id:
                    rows[object_id] = row
        return rows

    def write_json(self, phase: str, value: JsonDict) -> None:
        if not self.enabled:
            return
        self._write_json_atomic(self.paths[phase], to_plain(value))

    def load_json(self, phase: str) -> JsonDict:
        if not self.enabled:
            return {}
        return self._read_json(self.paths[phase]) or {}

    def _write_manifest(self) -> None:
        self._write_json_atomic(self.paths["manifest"], self.manifest)

    def _clear_artifacts(self) -> None:
        for name, path in self.paths.items():
            if name == "manifest":
                continue
            try:
                os.remove(path)
            except FileNotFoundError:
                continue

    @staticmethod
    def _read_json(path: str) -> JsonDict:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            try:
                value = json.load(f)
            except json.JSONDecodeError:
                return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _write_json_atomic(path: str, value: JsonDict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def _write_jsonl_atomic(path: str, rows: Iterable[JsonDict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(to_plain(row), ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
