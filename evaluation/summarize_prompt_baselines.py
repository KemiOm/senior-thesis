#!/usr/bin/env python3
"""
Summarize baseline JSON results into one CSV.

Reads prompt-only baseline outputs and computes exact match
(model_output vs gold_target).

Outputs:
- evaluation/baseline_report/model_comparison.csv
- evaluation/baseline_report/model_selection_notes.txt
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# this script is for strict exact-match evaluation (baseline comparison)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# allow passing either absolute or repo-relative paths
def _resolve_repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


DEFAULT_BASELINE_DIR = ROOT / "evaluation" / "baselines"
DEFAULT_OUT_DIR = ROOT / "evaluation" / "baseline_report"

BASELINE_DIR = DEFAULT_BASELINE_DIR
OUT_CSV = DEFAULT_OUT_DIR / "model_comparison.csv"
OUT_NOTES = DEFAULT_OUT_DIR / "model_selection_notes.txt"

TASKS = ("meter_only", "rhyme_only", "natural_text", "combined")
PROMPT_ORDER = ("zero_shot", "one_shot", "few_shot")


def infer_model_type(model_id: str) -> str:
    # quick heuristic: check model name for common causal LM patterns
    m = model_id.lower()
    causal_hints = (
        "gpt2",
        "phi-",
        "llama",
        "mistral",
        "smollm",
        "qwen",
        "pythia",
        "falcon",
        "gemma",
        "olmo",
    )
    if any(h in m for h in causal_hints):
        return "causal"
    return "seq2seq"


# normalize text before exact match comparison
def normalize_for_task(task: str, s: str) -> str:
    s = (s or "").strip()
    if task == "natural_text":
        s = re.sub(r"\s+", " ", s)
    return s


# exact match = model_output == gold_target after normalization
# strict: any difference counts as wrong
def exact_match_rate(task: str, results: list) -> tuple[int, int, float]:
    n_total = 0
    n_exact = 0
    for row in results:
        gold = normalize_for_task(task, row.get("gold_target", ""))
        pred = normalize_for_task(task, row.get("model_output", ""))
        if not gold:
            continue
        n_total += 1
        if gold == pred:
            n_exact += 1
    rate = (n_exact / n_total * 100.0) if n_total else 0.0
    return n_exact, n_total, rate


def load_baseline_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"WARN: skip {path}: {e}")
        return None


# load all baseline JSONs and compute metrics per file
def collect_rows(
    baseline_dir: Path | None = None,
    slug_globs: list[str] | None = None,
    include_natural_text_form: bool = False,
    form_max_n: int | None = None,
    form_relax_oov: bool | None = None,
    include_structured_partial: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    bdir = baseline_dir if baseline_dir is not None else DEFAULT_BASELINE_DIR
    if not bdir.is_dir():
        print(f"Missing directory: {bdir}")
        return rows

    if include_natural_text_form:
        from evaluation.form_eval_generation import aggregate_natural_text_form_results
    if include_structured_partial:
        from evaluation.structured_baseline_metrics import (
            aggregate_combined_structured,
            aggregate_rhyme_only_structured,
        )

    for json_path in sorted(bdir.glob("*/*.json")):
        slug = json_path.parent.name
        if slug_globs and not any(fnmatch.fnmatch(slug, pat) for pat in slug_globs):
            continue

        data = load_baseline_json(json_path)
        if not data:
            continue

        model = data.get("model", "")
        prompt_type = data.get("prompt_type", "")
        task = data.get("task", "")
        results = data.get("results", [])

        n_exact, n_scored, rate = exact_match_rate(task, results)

        row = {
            "model_slug": slug,
            "model_id": model,
            "model_type_guess": infer_model_type(model),
            "prompt_type": prompt_type,
            "task": task,
            "n_scored": n_scored,
            "n_exact": n_exact,
            "exact_match_pct": round(rate, 3),
        }

        if include_natural_text_form and task == "natural_text":
            ro = True if form_relax_oov else None
            fm = aggregate_natural_text_form_results(
                results, max_n=form_max_n, relax_oov=ro
            )
            row.update(fm)

        if include_structured_partial:
            if task == "rhyme_only":
                row.update(aggregate_rhyme_only_structured(results))
            elif task == "combined":
                row.update(aggregate_combined_structured(results))

        rows.append(row)

    return rows


# keep runs comparable by matching n_scored
def apply_fair_n_filter(rows: list[dict], fair_n: str | int):
    if fair_n == "all":
        return rows

    if isinstance(fair_n, int):
        return [r for r in rows if int(r["n_scored"]) == fair_n]

    group_min = {}
    for r in rows:
        key = (r["prompt_type"], r["task"])
        ns = int(r["n_scored"])
        group_min[key] = min(ns, group_min.get(key, ns))

    return [
        r for r in rows
        if int(r["n_scored"]) == group_min[(r["prompt_type"], r["task"])]
    ]


# write final comparison table
def write_csv(rows: list[dict], out_path: Path | None = None) -> None:
    path = out_path or OUT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to write for {path}.")
        return

    fieldnames = sorted(set().union(*(r.keys() for r in rows)))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {path}")


def write_selection_notes(rows: list[dict], prompt_type: str):
    lines = []
    lines.append("# Model comparison notes\n")
    lines.append("Exact match = model_output must exactly equal gold_target.\n")
    lines.append(f"Focus: `{prompt_type}` results\n")

    max_rate = max((r["exact_match_pct"] for r in rows), default=0.0)

    if max_rate == 0:
        lines.append("All exact match scores are 0%.\n")
        lines.append("This is common for structured outputs — exact match is very strict.\n")

    lines.append("Next steps:")
    lines.append("- choose one seq2seq model for fine-tuning")
    lines.append("- optionally compare with one causal model\n")

    OUT_NOTES.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_NOTES}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-type", default="few_shot")
    parser.add_argument("--print-pivot", action="store_true")
    parser.add_argument("--baseline-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)

    args = parser.parse_args()

    baseline_dir = (
        _resolve_repo_path(args.baseline_dir)
        if args.baseline_dir
        else DEFAULT_BASELINE_DIR
    )
    out_dir = (
        _resolve_repo_path(args.out_dir)
        if args.out_dir
        else DEFAULT_OUT_DIR
    )

    rows = collect_rows(baseline_dir=baseline_dir)

    write_csv(rows, out_dir / "model_comparison.csv")
    write_selection_notes(rows, args.prompt_type)


if __name__ == "__main__":
    main()
