#!/usr/bin/env python3
"""
Pre-filter script for text files before feeding into KARMA.

Rules:
  1. Skip obvious noise (error pages, very short files).
  2. The file must contain at least 3 recognisable sections.
  3. If an "abstract" section exists, it must have >= 100 words.
  4. At least 2 non-abstract sections must each have >= 200 words.

Usage:
    python src/filter_texts.py <text_dir> [--output-dir <dir>] [--dry-run]
"""

import argparse
import re
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple

# ── Section header patterns (case-insensitive, must start at beginning of line)
# Keys are canonical names; values are regex alternatives.
SECTION_SYNONYMS = OrderedDict([
    ("abstract",        r"abstract|summary"),
    ("introduction",    r"introduction|background"),
    ("methods",         r"methods?|methodology|materials(?:\s+and\s+methods)?|experimental"),
    ("results",         r"results?(?:/aims)?|findings|outcomes?"),
    ("discussion",      r"discussion|conclusions?|implications"),
    ("references",      r"references?|bibliography"),
    ("acknowledgments", r"acknowledg(?:e)?ments?"),
    ("funding",         r"funding(?:\s+information)?|grants?|financial"),
    ("supplementary",   r"supplement(?:ary)?|appendi(?:x|ces)|additional(?:\s+information)?"),
])

# Build one compiled regex per canonical section
_HEADER_RES = {
    canon: re.compile(
        rf"^\s*(?:{pattern})\s*$",
        re.IGNORECASE,
    )
    for canon, pattern in SECTION_SYNONYMS.items()
}


def detect_sections(text: str) -> List[Tuple[str, int, str]]:
    """Return list of (canonical_name, start_char_offset, header_line) found."""
    hits: List[Tuple[str, int, str]] = []
    for m in re.finditer(r"^(.+)$", text, re.MULTILINE):
        line = m.group(1).strip()
        if not line:
            continue
        for canon, pat in _HEADER_RES.items():
            if pat.match(line):
                hits.append((canon, m.start(), line))
                break  # first matching canon wins
    return hits


def section_word_counts(text: str, hits: List[Tuple[str, int, str]]) -> dict:
    """Map canonical section name -> word count of its body text.

    If the same canonical section appears more than once (e.g. conference
    abstracts file with many "Background" blocks), we sum all occurrences.
    """
    counts: dict = {}
    for i, (canon, start, header_line) in enumerate(hits):
        body_start = start + len(header_line)
        body_end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        body = text[body_start:body_end].strip()
        wc = len(body.split())
        counts[canon] = counts.get(canon, 0) + wc
    return counts


def passes_filter(filepath: Path) -> Tuple[bool, str]:
    """Check whether a text file passes the quality filter.

    Returns (passed: bool, reason: str).
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"read error: {e}"

    # ── Rule 0: trivially small / noise files
    word_count = len(text.split())
    if word_count < 80:
        return False, f"too short ({word_count} words)"

    # ── Detect sections
    hits = detect_sections(text)
    unique_sections = set(h[0] for h in hits)
    n_sections = len(unique_sections)

    if n_sections < 3:
        return False, f"only {n_sections} section(s) found: {sorted(unique_sections)}"

    # ── Word counts per section
    wc = section_word_counts(text, hits)

    # Rule: abstract (if present) must have >= 100 words
    if "abstract" in wc and wc["abstract"] < 100:
        return False, f"abstract too short ({wc['abstract']} words)"

    # Rule: at least 2 non-abstract sections >= 200 words
    non_abstract = {k: v for k, v in wc.items() if k != "abstract"}
    big_sections = [k for k, v in non_abstract.items() if v >= 200]
    if len(big_sections) < 2:
        return False, (
            f"only {len(big_sections)} non-abstract section(s) >= 200 words "
            f"(need >= 2); counts: {non_abstract}"
        )

    return True, f"{n_sections} sections, {len(big_sections)} substantial"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_dir", type=Path, help="Directory with .txt files")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="If set, copy passing files here (default: just report)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print results without copying",
    )
    args = parser.parse_args()

    txt_files = sorted(args.text_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {args.text_dir}")
        return

    passed_files: list = []
    failed_files: list = []

    for f in txt_files:
        ok, reason = passes_filter(f)
        if ok:
            passed_files.append((f, reason))
        else:
            failed_files.append((f, reason))

    # ── Report
    print(f"\n{'='*70}")
    print(f"  Total files scanned : {len(txt_files)}")
    print(f"  Passed              : {len(passed_files)}")
    print(f"  Filtered out        : {len(failed_files)}")
    print(f"{'='*70}\n")

    if failed_files:
        print("── Filtered out ──")
        for f, reason in failed_files:
            print(f"  ✗ {f.name}")
            print(f"      reason: {reason}")
        print()

    if passed_files:
        print("── Passed ──")
        for f, reason in passed_files:
            print(f"  ✓ {f.name}  ({reason})")
        print()

    # ── Copy if requested
    if args.output_dir and not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for f, _ in passed_files:
            shutil.copy2(f, args.output_dir / f.name)
        print(f"Copied {len(passed_files)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
