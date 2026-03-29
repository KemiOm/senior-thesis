#!/usr/bin/env python3
"""
Backfill caesura context fields in output/poems_annotated/*.json without re-annotating.

Only updates lines that already have a non-null `caesura` value.
Adds/updates:
  - caesura_punct
  - caesura_before
  - caesura_after

Usage:
  python scripts/helpers/recovery/backfill_caesura_context.py
  python scripts/helpers/recovery/backfill_caesura_context.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNOTATED_DIR = ROOT / "output" / "poems_annotated"


def compute_caesura_context_from_index(line_text: str, caesura_idx: int) -> tuple[str | None, str | None, str | None]:
    """
    Interpret legacy caesura index against token stream [word|punct] used by old annotator.
    Returns (punct, before_word, after_word), each None if unavailable.
    """
    toks = re.findall(r"[\w']+|[^\w\s]", line_text or "")
    if not toks or caesura_idx < 0 or caesura_idx >= len(toks):
        return (None, None, None)

    punct = toks[caesura_idx]
    if punct not in ",;:—-":
        return (None, None, None)

    before = None
    for j in range(caesura_idx - 1, -1, -1):
        if re.match(r"[\w']+$", toks[j]):
            before = toks[j]
            break

    after = None
    for j in range(caesura_idx + 1, len(toks)):
        if re.match(r"[\w']+$", toks[j]):
            after = toks[j]
            break

    return (punct, before, after)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill caesura context in poems_annotated JSONs")
    ap.add_argument("--dry-run", action="store_true", help="Report only; do not write files")
    args = ap.parse_args()

    files = sorted(ANNOTATED_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No files found in {ANNOTATED_DIR}")

    n_files_touched = 0
    n_lines_seen = 0
    n_lines_updated = 0

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            poem = json.load(f)

        changed = False
        for stanza in poem.get("stanzas", []):
            line_list = stanza.get("lines", stanza) if isinstance(stanza, dict) else stanza
            if not isinstance(line_list, list):
                continue
            for line in line_list:
                if not isinstance(line, dict):
                    continue
                cidx = line.get("caesura")
                if cidx is None:
                    continue
                n_lines_seen += 1
                if not isinstance(cidx, int):
                    continue
                punct, before, after = compute_caesura_context_from_index(
                    line.get("normalized", ""), cidx
                )
                old = (
                    line.get("caesura_punct"),
                    line.get("caesura_before"),
                    line.get("caesura_after"),
                )
                new = (punct, before, after)
                if old != new:
                    line["caesura_punct"] = punct
                    line["caesura_before"] = before
                    line["caesura_after"] = after
                    n_lines_updated += 1
                    changed = True

        if changed:
            n_files_touched += 1
            if not args.dry_run:
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(poem, f, indent=2, ensure_ascii=False)

    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print(
        f"{mode}: files={len(files)}, touched={n_files_touched}, "
        f"caesura_lines_seen={n_lines_seen}, updated={n_lines_updated}"
    )


if __name__ == "__main__":
    main()
