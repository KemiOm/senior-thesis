#!/usr/bin/env python3
"""
Roll up prompt-only baseline JSONs into one comparison table (exact match on gold vs model output).

Writes (default paths; override with ``--out-dir``):
  - ``<out-dir>/model_comparison.csv``
  - ``<out-dir>/model_selection_notes.txt``

Optional: ``--baseline-dir results`` plus ``--out-dir results``
and ``--slug-glob`` for an SFT-only rollup (CSV + notes in ``--out-dir``).

Fair rows: use `--fair-n min` keeps runs comparable when line caps differ (see `--help`).

Examples:

  python evaluation/summarize_prompt_baselines.py
  python evaluation/summarize_prompt_baselines.py --print-pivot
  python evaluation/summarize_prompt_baselines.py --prompt-type few_shot --print-pivot
  python evaluation/summarize_prompt_baselines.py --fair-n 500 --print-pivot

  python evaluation/summarize_prompt_baselines.py \\
    --baseline-dir results \\
    --out-dir results \\
    --slug-glob 'google_flan-t5-large' \\
    --slug-glob '*combined_lora*' \\
    --print-pivot

  python evaluation/summarize_prompt_baselines.py --include-natural-text-form --form-max-n 2000
  python evaluation/summarize_prompt_baselines.py --include-natural-text-form --relax-form-oov --prompt-type all

  python scripts/corpus_tools.py spotcheck-nt path/to/natural_text.json --phon-check

  python evaluation/summarize_prompt_baselines.py --skip-structured-partial
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path)
DEFAULT_BASELINE_DIR = ROOT / "evaluation" / "results" / "baselines"
DEFAULT_OUT_DIR = ROOT / "evaluation" / "results"
# Back-compat names for modules that import these (prefer DEFAULT_* in new code).
BASELINE_DIR = DEFAULT_BASELINE_DIR
OUT_CSV = DEFAULT_OUT_DIR / "model_comparison.csv"
OUT_CSV_FAIR = DEFAULT_OUT_DIR / "model_comparison_fair.csv"
OUT_NOTES = DEFAULT_OUT_DIR / "model_selection_notes.txt"

GroupPTTask = tuple[str, str]

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
    """Returns (n_exact, n_total, rate). Skips rows with empty gold."""
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

    globs = slug_globs if slug_globs else None

    for json_path in sorted(bdir.glob("*/*.json")):
        slug = json_path.parent.name
        if globs and not any(fnmatch.fnmatch(slug, pat) for pat in globs):
            continue
        data = load_baseline_json(json_path)
        if not data:
            continue
        model = data.get("model", "")
        prompt_type = data.get("prompt_type", "")
        task = data.get("task", "")
        split = data.get("split", "")
        results = data.get("results", [])
        n_samples = data.get("n_samples", len(results))

        n_exact, n_scored, rate = exact_match_rate(task, results)

        row = {
            "model_slug": slug,
            "model_id": model,
            "model_type_guess": infer_model_type(model),
            "prompt_type": prompt_type,
            "task": task,
            "split": split,
            "n_samples_meta": n_samples,
            "n_scored": n_scored,
            "n_exact": n_exact,
            "exact_match_pct": round(rate, 3),
            "json_path": str(json_path.relative_to(ROOT)),
        }
        if include_natural_text_form and task == "natural_text":
            ro = True if form_relax_oov else None
            fm = aggregate_natural_text_form_results(
                results, max_n=form_max_n, relax_oov=ro
            )
            row.update({k: fm[k] for k in fm})
        if include_structured_partial:
            if task == "rhyme_only":
                row.update(aggregate_rhyme_only_structured(results))
            elif task == "combined":
                row.update(aggregate_combined_structured(results))
        rows.append(row)
    return rows


def n_scored_mismatch_report(rows: list[dict]) -> list[str]:
    """Lines describing (prompt_type, task) groups with more than one distinct n_scored."""
    by_group: dict[GroupPTTask, set[int]] = defaultdict(set)
    for r in rows:
        by_group[(r["prompt_type"], r["task"])].add(int(r["n_scored"]))
    lines: list[str] = []
    for (pt, task) in sorted(by_group.keys()):
        vals = sorted(by_group[(pt, task)])
        if len(vals) <= 1:
            continue
        lines.append(f"- `{pt}` / `{task}`: n_scored values {vals} (not comparable without `--fair-n` or re-running baselines with the same `--n`).")
    return lines


def apply_fair_n_filter(
    rows: list[dict], fair_n: str | int
) -> tuple[list[dict], list[str]]:
    """
    fair_n: 'all' -> no filtering
            'min' -> per (prompt_type, task), keep rows whose n_scored equals min(n_scored) in that group
            int -> keep rows with n_scored == int
    Returns (filtered_rows, log_lines).
    """
    log: list[str] = []
    if fair_n == "all":
        return rows, log
    if isinstance(fair_n, int):
        target = fair_n
        kept = [r for r in rows if int(r["n_scored"]) == target]
        dropped = len(rows) - len(kept)
        log.append(
            f"Fair filter `--fair-n {target}`: kept {len(kept)} / {len(rows)} rows "
            f"(dropped {dropped} with n_scored != {target})."
        )
        return kept, log
    # min
    group_min: dict[GroupPTTask, int] = {}
    for r in rows:
        k = (r["prompt_type"], r["task"])
        ns = int(r["n_scored"])
        group_min[k] = min(ns, group_min[k]) if k in group_min else ns
    kept = [r for r in rows if int(r["n_scored"]) == group_min[(r["prompt_type"], r["task"])]]
    dropped = len(rows) - len(kept)
    log.append(
        f"Fair filter `--fair-n min`: kept {len(kept)} / {len(rows)} rows "
        f"(dropped {dropped} longer runs; per-group minimum n_scored)."
    )
    return kept, log


def write_csv(rows: list[dict], out_path: Path | None = None) -> None:
    path = out_path or OUT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No rows to write for {path}.")
        return
    preferred = [
        "model_slug",
        "model_id",
        "model_type_guess",
        "prompt_type",
        "task",
        "split",
        "n_samples_meta",
        "n_scored",
        "n_exact",
        "exact_match_pct",
        "st_rhyme_n_scored",
        "st_rhyme_relaxed_match_pct",
        "st_combined_n_scored",
        "st_combined_pred_parse_pct",
        "st_combined_meter_field_pct",
        "st_combined_meter_type_field_pct",
        "st_combined_rhyme_field_pct",
        "st_combined_end_field_pct",
        "st_combined_caesura_field_pct",
        "st_combined_mean_field_pct",
        "st_combined_all_four_pct",
    ]
    all_keys = set().union(*(r.keys() for r in rows))
    fieldnames = [k for k in preferred if k in all_keys]
    fieldnames += sorted(k for k in all_keys if k not in fieldnames)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def pivot_by_slug(rows: list[dict], prompt_type: str) -> dict[str, dict[str, float]]:
    """slug -> task -> exact_match_pct for one prompt_type."""
    p: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["prompt_type"] != prompt_type:
            continue
        p[r["model_slug"]][r["task"]] = r["exact_match_pct"]
    return dict(p)


def _print_pivot_one(rows: list[dict], prompt_type: str) -> None:
    pivot = pivot_by_slug(rows, prompt_type)
    if not pivot:
        print(f"No rows for prompt_type={prompt_type!r}.")
        return
    header = ["model_slug"] + list(TASKS) + ["mean_present"]
    print("\n" + " | ".join(header))
    print("-" * (len(header) * 14))
    for slug in sorted(pivot.keys()):
        tmap = pivot[slug]
        vals = []
        for t in TASKS:
            v = tmap.get(t)
            vals.append(f"{v:.2f}" if v is not None else "—")
        present = [tmap[t] for t in TASKS if t in tmap]
        mean4 = sum(present) / len(present) if present else 0.0
        vals.append(f"{mean4:.2f}")
        print(slug + " | " + " | ".join(vals))


def print_pivot(rows: list[dict], prompt_type: str) -> None:
    if prompt_type == "all":
        any_out = False
        for pt in PROMPT_ORDER:
            sub = [r for r in rows if r["prompt_type"] == pt]
            if not sub:
                continue
            any_out = True
            print(f"\n### prompt_type = {pt}")
            _print_pivot_one(sub, pt)
        if not any_out:
            print("No baseline rows found for any of zero_shot / one_shot / few_shot.")
        return
    _print_pivot_one(rows, prompt_type)


def _ranking_section_lines(
    pivot: dict[str, dict[str, float]], prompt_label: str
) -> tuple[list[str], list[tuple[str, float]]]:
    lines: list[str] = []

    def best_for_task(task: str) -> tuple[str, float] | None:
        best_slug, best_v = None, -1.0
        for slug, tmap in pivot.items():
            if task not in tmap:
                continue
            v = tmap[task]
            if v > best_v:
                best_v, best_slug = v, slug
        return (best_slug, best_v) if best_slug else None

    lines.append(f"## Best `{prompt_label}` by task")
    for t in TASKS:
        b = best_for_task(t)
        lines.append(f"- **{t}**: {b[0]} ({b[1]:.2f}%)" if b else f"- **{t}**: (no data)")
    lines.append("")

    complete_means: list[tuple[str, float]] = []
    for slug, tmap in pivot.items():
        if all(t in tmap for t in TASKS):
            m = sum(tmap[t] for t in TASKS) / 4.0
            complete_means.append((slug, m))
    complete_means.sort(key=lambda x: (-x[1], x[0]))
    lines.append(f"## Best mean exact-match (`{prompt_label}`, models with all four tasks)")
    for slug, m in complete_means[:8]:
        lines.append(f"- {slug}: **{m:.2f}%**")
    lines.append("")

    incomplete = [s for s in pivot if not all(t in pivot[s] for t in TASKS)]
    if incomplete:
        lines.append(f"## Incomplete folders (`{prompt_label}`, missing one or more tasks)")
        for s in sorted(incomplete):
            have = [t for t in TASKS if t in pivot[s]]
            miss = [t for t in TASKS if t not in pivot[s]]
            lines.append(f"- `{s}`: has {have}, missing {miss}")
        lines.append("")

    return lines, complete_means


def write_selection_notes(
    rows: list[dict],
    prompt_type: str,
    out_notes: Path | None = None,
    comparison_csv_basename: str = "model_comparison.csv",
) -> None:
    lines: list[str] = []
    lines.append("# Tables (auto-generated by summarize_prompt_baselines.py)")
    lines.append("")
    lines.append("Exact match = percentage of lines where model_output equals gold_target (after strip; natural_text collapses whitespace).")
    has_form = any("nt_form_evaluable" in r for r in rows)
    if has_form:
        lines.append(
            "Natural-text form columns (when present in CSV): CMU-based stress/syllable/rhyme agreement "
            "between gold_target and model_output (`aggregate_natural_text_form_results`); see `nt_form_*` fields."
        )
    has_struct = any("st_combined_mean_field_pct" in r for r in rows) or any(
        "st_rhyme_relaxed_match_pct" in r for r in rows
    )
    if has_struct:
        lines.append(
            "Structured partial columns (`st_*`): `rhyme_only` uses case/whitespace-normalized token match; "
            "`combined` parses `stress:…|meter_type:…|rhyme:…|end:…|caesura:…` (legacy `meter:…|rhyme:…|…` still supported) "
            "and reports per-field and mean-field rates (`evaluation/structured_baseline_metrics.py`)."
        )
    lines.append("")
    lines.append(
        f"Primary prompt focus for this file: **`{prompt_type}`** "
        f"(see `--prompt-type`; use `all` for one section per prompt). "
        f"`{comparison_csv_basename}` lists the JSONs included in this run (see `--slug-glob` / `--out-dir` when filtering)."
    )
    lines.append("")

    def section_for_subset(sub: list[dict], pt: str) -> tuple[list[str], list[tuple[str, float]]]:
        max_rate = max((r["exact_match_pct"] for r in sub), default=0.0)
        out: list[str] = []
        if max_rate <= 0.0 and sub:
            out.append("### Warning")
            out.append(
                "All exact-match rates are 0% for this prompt type. Common for structured targets; "
                "use architecture fit, compute constraints, and spot-checks—not rank order."
            )
            out.append("")
        pivot = pivot_by_slug(sub, pt)
        body, complete_means = _ranking_section_lines(pivot, pt)
        out.extend(body)
        return out, complete_means

    complete_means_for_heuristic: list[tuple[str, float]] = []
    if prompt_type == "all":
        first = True
        cm_by_pt: dict[str, list[tuple[str, float]]] = {}
        for pt in PROMPT_ORDER:
            sub = [r for r in rows if r["prompt_type"] == pt]
            if not sub:
                continue
            if not first:
                lines.append("---")
                lines.append("")
            first = False
            lines.append(f"## Prompt type: `{pt}`")
            lines.append("")
            sec, cm = section_for_subset(sub, pt)
            lines.extend(sec)
            cm_by_pt[pt] = cm
        for prefer in ("few_shot", "one_shot", "zero_shot"):
            if cm_by_pt.get(prefer):
                complete_means_for_heuristic = cm_by_pt[prefer]
                break
    else:
        sub = [r for r in rows if r["prompt_type"] == prompt_type]
        if not sub:
            lines.append(f"(No JSONs with prompt_type=`{prompt_type}` under the baseline JSON tree.)")
            lines.append("")
            lines.append(
                "Tip: on the cluster run `./scripts/hpc/submit_all_model_baselines.sh` (default includes "
                "`one_shot` and `few_shot`; needs `output/corpus.db`). Then rsync results and re-run this script. "
                "If only zero_shot files are present, use `--prompt-type zero_shot`."
            )
            lines.append("")
        else:
            sec, complete_means_for_heuristic = section_for_subset(sub, prompt_type)
            lines.extend(sec)

    lines.append("## Suggested next step")
    lines.append("1. Pick **one seq2seq** checkpoint for SFT when using encoder–decoder models (e.g. FLAN-T5, BART).")
    lines.append("2. Optionally pick **one causal** LM for comparison (e.g. best mean among causal models).")
    lines.append("3. Prefer **few_shot** or **one_shot** summaries when zero_shot is uninformative; see `--prompt-type`.")
    lines.append("4. Tie-break with size, speed, and stability (no repeated OOM on cluster).")
    lines.append("")

    max_all = max((r["exact_match_pct"] for r in rows), default=0.0)
    if complete_means_for_heuristic:
        top = complete_means_for_heuristic[0][0]
        lines.append(
            f"**Heuristic default SFT candidate (complete runs, best mean; tie-break: slug sort):** `{top}`"
        )
        if max_all <= 0.0:
            lines.append(
                "With 0% exact match everywhere, prefer **`google/flan-t5-base`** or **`facebook/bart-large`** for encoder–decoder SFT "
                "if that matches `scripts/sft/lora_train.py` / FLAN-T5-style seq2seq; otherwise pick a causal instruct model with stable runs."
            )
    else:
        pivot_any = {}
        for pt in PROMPT_ORDER:
            pivot_any = pivot_by_slug(rows, pt)
            if pivot_any:
                break
        if pivot_any:

            def _best_meter(piv: dict) -> tuple[str, float] | None:
                best_slug, best_v = None, -1.0
                for slug, tmap in piv.items():
                    if "meter_only" not in tmap:
                        continue
                    v = tmap["meter_only"]
                    if v > best_v:
                        best_v, best_slug = v, slug
                return (best_slug, best_v) if best_slug else None

            b = _best_meter(pivot_any)
            if b:
                lines.append(f"**Heuristic default (meter_only only):** `{b[0]}`")

    body = "\n".join(lines) + "\n"
    dest = out_notes if out_notes is not None else OUT_NOTES
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body, encoding="utf-8")
    print(f"Wrote {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-type",
        choices=["zero_shot", "one_shot", "few_shot", "all"],
        default="few_shot",
        help="Which prompt_type drives selection notes (default: few_shot). CSV always includes all JSONs. Use `all` for one section per prompt.",
    )
    parser.add_argument(
        "--print-pivot",
        action="store_true",
        help="Print task × model table(s) to stdout (same scope as --prompt-type; use all for three tables).",
    )
    parser.add_argument(
        "--include-natural-text-form",
        action="store_true",
        help=(
            "For task=natural_text JSONs, run CMU form metrics (slow: phonology per row). "
            "Adds nt_form_* columns to model_comparison.csv."
        ),
    )
    parser.add_argument(
        "--form-max-n",
        type=int,
        default=None,
        help="With --include-natural-text-form, cap rows per file (default: all rows in each JSON).",
    )
    parser.add_argument(
        "--relax-form-oov",
        action="store_true",
        help=(
            "With --include-natural-text-form: allow partial CMU coverage (stress from known words only). "
            "Raises evaluable count on archaic text; document this setting."
        ),
    )
    parser.add_argument(
        "--skip-structured-partial",
        action="store_true",
        help=(
            "Do not compute rhyme_only / combined field-level metrics (st_* columns). "
            "Default is to compute them (fast; no CMU)."
        ),
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help=(
            "Root of <model_slug>/*.json baseline trees "
            f"(default: {DEFAULT_BASELINE_DIR.relative_to(ROOT)})."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Directory for model_comparison.csv and model_selection_notes.txt "
            f"(default: {DEFAULT_OUT_DIR.relative_to(ROOT)})."
        ),
    )
    parser.add_argument(
        "--slug-glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help=(
            "If set (repeatable), only slugs matching any fnmatch pattern are included. "
            "Example: --slug-glob 'google_flan-t5-large' --slug-glob '*final_model_merged*'"
        ),
    )
    args = parser.parse_args()

    baseline_dir = _resolve_repo_path(args.baseline_dir) if args.baseline_dir else DEFAULT_BASELINE_DIR
    out_dir = _resolve_repo_path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    out_csv = out_dir / "model_comparison.csv"
    out_notes = out_dir / "model_selection_notes.txt"
    slug_globs: list[str] | None = list(args.slug_glob) if args.slug_glob else None

    if args.include_natural_text_form and baseline_dir.is_dir():
        nt_paths = sorted(baseline_dir.glob("*/*natural_text.json"))
        pair_ops = 0
        for p in nt_paths:
            data = load_baseline_json(p)
            if not data:
                continue
            n = len(data.get("results") or [])
            pair_ops += min(n, args.form_max_n) if args.form_max_n is not None else n
        print(
            f"[natural_text form] {len(nt_paths)} JSONs, ~{pair_ops} row pairs "
            f"(gold+pred phonology each; cap={'none' if args.form_max_n is None else args.form_max_n})."
        )
        if args.form_max_n is None and pair_ops > 25_000:
            print(
                "WARNING: Large form pass — expect long runtime. "
                "Use --form-max-n 5000 to cap, or run overnight.",
                file=sys.stderr,
            )
        if args.relax_form_oov:
            print("[natural_text form] relax_oov=True (partial CMU lines allowed).")

    rows = collect_rows(
        baseline_dir=baseline_dir,
        slug_globs=slug_globs,
        include_natural_text_form=args.include_natural_text_form,
        form_max_n=args.form_max_n,
        form_relax_oov=args.relax_form_oov,
        include_structured_partial=not args.skip_structured_partial,
    )
    write_csv(rows, out_csv)
    write_selection_notes(rows, args.prompt_type, out_notes)
    if args.print_pivot:
        print_pivot(rows, args.prompt_type)


if __name__ == "__main__":
    main()
