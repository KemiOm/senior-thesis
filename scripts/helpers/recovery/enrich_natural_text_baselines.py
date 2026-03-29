#!/usr/bin/env python3
"""
Add gold_stress_pm, gold_rhyme_key, gold_form_ok to existing natural_text baseline JSONs
without re-running the model (uses gold_target + evaluation.form_eval_generation).

Usage (from project root):
  python scripts/helpers/recovery/enrich_natural_text_baselines.py
  python scripts/helpers/recovery/enrich_natural_text_baselines.py --dry-run
  python scripts/helpers/recovery/enrich_natural_text_baselines.py path/to/zero_shot_natural_text.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.form_eval_generation import line_form_signature

DEFAULT_GLOB = "evaluation/results/baselines/prompt_only/*/*natural_text.json"


def enrich_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """Returns (n_updated_rows, n_skipped_already)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("task") != "natural_text":
        return 0, 0
    results = data.get("results") or []
    updated = 0
    skipped = 0
    for row in results:
        if "gold_stress_pm" in row and "gold_rhyme_key" in row:
            skipped += 1
            continue
        gsig = line_form_signature(row.get("gold_target", ""))
        row["gold_stress_pm"] = gsig["stress_pm"]
        row["gold_rhyme_key"] = gsig["rhyme_key"]
        row["gold_form_ok"] = gsig["ok"]
        updated += 1
    data["gold_form_note"] = (
        "gold_stress_pm, gold_rhyme_key, gold_form_ok: CMU-based signature of gold_target "
        "(evaluation.form_eval_generation.line_form_signature); same path as eval_natural_text_form.py."
    )
    if dry_run:
        return updated, skipped
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return updated, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich natural_text baseline JSONs with gold form fields")
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"JSON files (default: glob {DEFAULT_GLOB})",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print counts only, do not write")
    args = ap.parse_args()

    if args.paths:
        files = [p.resolve() for p in args.paths]
    else:
        files = sorted(ROOT.glob(DEFAULT_GLOB))

    if not files:
        print("No files matched.", file=sys.stderr)
        sys.exit(1)

    total_u = total_s = 0
    for p in files:
        if not p.is_file():
            print(f"Skip (not a file): {p}", file=sys.stderr)
            continue
        u, s = enrich_file(p, args.dry_run)
        total_u += u
        total_s += s
        print(f"{p.name}: updated_rows={u} already_had_fields={s}" + (" [dry-run]" if args.dry_run else ""))

    print(f"Total: updated_rows={total_u}, already_had_fields={total_s}", file=sys.stderr)


if __name__ == "__main__":
    main()
