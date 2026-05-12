"""Top-1 selection pipeline over a claim_split_results jsonl."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .selector import RpmLimiter, project_minimal, select_claim_id

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "index" not in rec:
                rec["index"] = i
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    return records


def _load_done_indices(partial_path: Path) -> set[int]:
    if not partial_path.exists():
        return set()
    done: set[int] = set()
    with partial_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["index"]))
            except Exception:  # noqa: BLE001
                continue
    return done


def _process_one(
    record: dict, client: Any, model: str, limiter: RpmLimiter | None
) -> dict:
    claims = record.get("claims") or []
    claims_min = project_minimal(claims)
    chosen_id = select_claim_id(client, model, claims_min, limiter=limiter)

    out = dict(record)
    out["selected_claim_id"] = chosen_id
    if chosen_id is None:
        out["claims"] = []
    else:
        kept = next((c for c in claims if c.get("claim_id") == chosen_id), None)
        out["claims"] = [kept] if kept is not None else []
    return out


def run(
    input_path: Path,
    output_path: Path,
    *,
    client: Any,
    model: str,
    workers: int = 16,
    limit: int | None = None,
    rpm: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    records = _read_jsonl(input_path, limit)
    done = _load_done_indices(partial_path)
    pending = [r for r in records if int(r["index"]) not in done]
    logger.info(
        "loaded %d records, %d already done, %d pending (workers=%d, rpm=%s)",
        len(records),
        len(done),
        len(pending),
        workers,
        rpm,
    )

    limiter = RpmLimiter(rpm) if rpm else None
    write_lock = threading.Lock()
    partial_fh = partial_path.open("a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, rec, client, model, limiter): rec
                for rec in pending
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="claim-top1"
            ):
                rec = futures[fut]
                try:
                    out = fut.result()
                except Exception as err:  # noqa: BLE001
                    logger.error(
                        "record index=%s failed: %s", rec.get("index"), err
                    )
                    continue
                with write_lock:
                    partial_fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                    partial_fh.flush()
    finally:
        partial_fh.close()

    _finalize(partial_path, output_path)


def _finalize(partial_path: Path, output_path: Path) -> None:
    by_index: dict[int, dict] = {}
    with partial_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_index[int(rec["index"])] = rec

    with output_path.open("w", encoding="utf-8") as f:
        for idx in sorted(by_index):
            f.write(json.dumps(by_index[idx], ensure_ascii=False) + "\n")
    logger.info("wrote %d records to %s", len(by_index), output_path)


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
