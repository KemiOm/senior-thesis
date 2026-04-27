#!/usr/bin/env python3
"""
Roll prompt-eval JSON outputs into model_comparison.csv.

Reads evaluation JSON files from ``--baseline-dir``. Supports either a flat
structure  or nested directories.

 ``evaluation/baseline_report/`` is used for pretrained
(prompt-only, Hub) runs aggregated from ``evaluation/baselines/``.
 point ``--baseline-dir`` to something like ``results`` (SFT outputs),
make sure to use a different ``--out-dir``  ``results/`` or
``evaluation/baseline_report_reeval_*`` so you don’t overwrite the
pretrained table.

"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)


DEFAULT_BASELINE_DIR = ROOT / "evaluation" / "baselines"
DEFAULT_OUT_DIR = ROOT / "evaluation" / "baseline_report"

OUT_CSV = DEFAULT_OUT_DIR / "model_comparison.csv"

TASKS = ("meter_only", "rhyme_only", "natural_text", "combined")
PROMPT_ORDER = ("zero_shot", "one_shot", "few_shot")


def infer_model_type(model_id: str) -> str:
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


def normalize_for_task(task: str, s: str) -> str:
    s = (s or "").strip()
    if task == "natural_text":
        s = re.sub(r"\s+", " ", s)
    return s


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


def collect_rows(
    baseline_dir: Path | None = None,
    slug_globs: list[str] | None = None,
    include_natural_text_form: bool = False,
    form_max_n: int | None = None,
    form_relax_oov: bool | None = None,
    include_structured_partial: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    bdir = Path(baseline_dir if baseline_dir is not None else DEFAULT_BASELINE_DIR).resolve()
    if not bdir.is_dir():
        print(f"Missing directory: {bdir}")
        return rows

    if include_natural_text_form:
        from evaluation.scoring.form_eval import aggregate_natural_text_form_results
    if include_structured_partial:
        from evaluation.scoring.struct_metrics import (
            aggregate_combined_structured,
            aggregate_rhyme_only_structured,
        )
    from evaluation.scoring.edit_distance import aggregate_string_edit_metrics

    for json_path in sorted(bdir.rglob("*.json")):
        json_path = json_path.resolve()
        if any(p.startswith("_archive") for p in json_path.parts):
            continue
        try:
            rel_parent = json_path.parent.relative_to(bdir)
        except ValueError:
            continue
        slug = str(rel_parent).replace("\\", "/")
        if slug in ("", "."):
            continue
        if slug_globs and not any(fnmatch.fnmatch(slug, pat) for pat in slug_globs):
            continue

        data = load_baseline_json(json_path)
        if not data:
            continue

        model = data.get("model", "")
        prompt_type = data.get("prompt_type", "")
        prompt_style = data.get("prompt_style", "default")
        task = data.get("task", "")
        results = data.get("results", [])

        n_exact, n_scored, rate = exact_match_rate(task, results)

        row = {
            "model_slug": slug,
            "model_id": model,
            "model_type_guess": infer_model_type(model),
            "prompt_type": prompt_type,
            "prompt_style": prompt_style,
            "task": task,
            "n_scored": n_scored,
            "n_exact": n_exact,
            "exact_match_pct": round(rate, 3),
            "json_path": str(json_path.relative_to(ROOT.resolve())),
        }
        row.update(aggregate_string_edit_metrics(task, results))

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


def main() -> None:
    parser = argparse.ArgumentParser()
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


if __name__ == "__main__":
    main()
