#!/usr/bin/env python3
"""
Aggregate form metrics on natural_text baseline JSON (gold next line vs model_output).

Uses evaluation/form_eval_generation.py — same CMU phonology path for gold and pred.

Usage:
  python scripts/eval_natural_text_form.py path/to/zero_shot_natural_text.json
  python scripts/eval_natural_text_form.py path/to/file.json --n 500
  python scripts/eval_natural_text_form.py path/to/file.json --relax-oov
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.form_eval_generation import aggregate_natural_text_form_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Meter/rhyme agreement for natural_text generations")
    parser.add_argument("json_path", type=Path, help="Baseline JSON with results[].gold_target / model_output")
    parser.add_argument("--n", type=int, default=None, help="Max examples")
    parser.add_argument(
        "--relax-oov",
        action="store_true",
        help="Count lines ok if stress exists even with some CMU not_found words (see form_eval_generation).",
    )
    args = parser.parse_args()

    path = args.json_path
    if not path.is_file():
        raise SystemExit(f"Not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results") or []
    ro = True if args.relax_oov else None
    fm = aggregate_natural_text_form_results(results, max_n=args.n, relax_oov=ro)

    print(f"File: {path}")
    print(f"relax_oov: {bool(fm.get('nt_form_relaxed_oov'))}")
    print(f"Rows scanned: {fm['nt_form_rows_scanned']}")
    print(f"Evaluable (gold+pred phonology OK): {fm['nt_form_evaluable']}")
    print(f"Skipped gold phonology issues: {fm['nt_form_fail_gold']} (includes non-evaluable pairs)")
    print(f"Skipped pred phonology issues: {fm['nt_form_fail_pred']}")
    print()
    ne = fm["nt_form_evaluable"]
    print(
        f"Stress pattern match (+/- exact, same length): "
        f"{fm['nt_form_stress_hits']}/{ne} = {fm['nt_form_stress_match_pct']}%"
    )
    print(
        f"Syllable count match:                  "
        f"{fm['nt_form_syllable_hits']}/{ne} = {fm['nt_form_syllable_match_pct']}%"
    )
    rd = fm["nt_form_rhyme_denom"]
    print(
        f"Rhyme key match (where gold has key): "
        f"{fm['nt_form_rhyme_hits']}/{rd} = {fm['nt_form_rhyme_match_pct']}%"
    )


if __name__ == "__main__":
    main()
