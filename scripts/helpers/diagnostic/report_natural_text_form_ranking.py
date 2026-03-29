#!/usr/bin/env python3
"""
From evaluation/results/model_comparison.csv, filter task==natural_text and rank by
nt_form_stress_match_pct / nt_form_rhyme_match_pct (and evaluable count).

Writes: evaluation/results/natural_text_form_ranking.txt

Usage (project root):
  python scripts/helpers/diagnostic/report_natural_text_form_ranking.py
  python scripts/helpers/diagnostic/report_natural_text_form_ranking.py --csv path/to/model_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "evaluation/results/model_comparison.csv"
DEFAULT_OUT = ROOT / "evaluation/results/natural_text_form_ranking.txt"


def fnum(r: dict, k: str, default: float = 0.0) -> float:
    v = r.get(k) or ""
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"Not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with args.csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("task") != "natural_text":
                continue
            rows.append(
                {
                    "model_slug": r["model_slug"],
                    "model_id": r["model_id"],
                    "prompt_type": r["prompt_type"],
                    "n_scanned": int(r.get("nt_form_rows_scanned") or 0),
                    "evaluable": int(r.get("nt_form_evaluable") or 0),
                    "fail_gold": int(r.get("nt_form_fail_gold") or 0),
                    "fail_pred": int(r.get("nt_form_fail_pred") or 0),
                    "stress_pct": fnum(r, "nt_form_stress_match_pct"),
                    "rhyme_pct": fnum(r, "nt_form_rhyme_match_pct"),
                    "syl_pct": fnum(r, "nt_form_syllable_match_pct"),
                    "rhyme_denom": int(r.get("nt_form_rhyme_denom") or 0),
                    "exact_pct": fnum(r, "exact_match_pct"),
                }
            )

    lines: list[str] = []
    lines.append("natural_text — form metrics (from model_comparison.csv)")
    lines.append("")
    lines.append("WARNING: nt_form_* needs gold AND model_output to pass the CMU phonology path.")
    lines.append("If `eval` (evaluable pairs) is very small, stress%/rhyme% are NOT stable for ranking.")
    lines.append("Check nt_form_fail_gold / nt_form_fail_pred in the CSV — high counts mean OOV/archaic")
    lines.append("text or empty/garbled generations. Re-run summarize with more samples or inspect JSON outputs.")
    lines.append("")
    lines.append("stress% = nt_form_stress_match_pct over evaluable rows; rhyme% uses nt_form_rhyme_denom.")
    lines.append("Sort: higher evaluable first, then stress%, then rhyme%.")
    lines.append("")

    for pt in ("zero_shot", "one_shot", "few_shot"):
        sub = [x for x in rows if x["prompt_type"] == pt]
        if not sub:
            continue
        sub.sort(
            key=lambda x: (-x["evaluable"], -x["stress_pct"], -x["rhyme_pct"], x["model_slug"])
        )
        lines.append(f"=== prompt_type = {pt} ===")
        lines.append(
            f"{'model_slug':<38} {'stress%':>8} {'rhyme%':>8} {'syl%':>8} "
            f"{'eval':>6} {'scan':>6} {'f_gold':>7} {'f_pred':>7} {'rhyme_n':>8} {'exact%':>8}"
        )
        lines.append("-" * 110)
        for x in sub:
            lines.append(
                f"{x['model_slug']:<38} {x['stress_pct']:>8.2f} {x['rhyme_pct']:>8.2f} {x['syl_pct']:>8.2f} "
                f"{x['evaluable']:>6} {x['n_scanned']:>6} {x['fail_gold']:>7} {x['fail_pred']:>7} "
                f"{x['rhyme_denom']:>8} {x['exact_pct']:>8.2f}"
            )
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
