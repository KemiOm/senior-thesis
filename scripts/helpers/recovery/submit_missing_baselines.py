#!/usr/bin/env python3
"""
List or submit Slurm jobs only for baseline JSONs that are missing on disk.

Uses the same model list and slug rule as submit_all_model_baselines.sh / run_prompt_baseline.py.
Each job runs ONE task (BASELINE_TASK set) — same as SPLIT_TASKS=1.

Examples (from project root on the cluster):
  # Print sbatch lines only (review, then paste or pipe to bash)
  python scripts/helpers/recovery/submit_missing_baselines.py --n 500 --dry-run

  # Print grouped by model
  python scripts/helpers/recovery/submit_missing_baselines.py --n 500 --dry-run --by-model

  # Actually submit missing jobs
  python scripts/helpers/recovery/submit_missing_baselines.py --n 500 --execute
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Keep in sync with scripts/submit_all_model_baselines.sh SPECS
SPECS: list[tuple[str, str]] = [
    ("google/flan-t5-small", "seq2seq"),
    ("google/flan-t5-base", "seq2seq"),
    ("google/flan-t5-large", "seq2seq"),
    ("facebook/bart-base", "seq2seq"),
    ("facebook/bart-large", "seq2seq"),
    ("gpt2", "causal"),
    ("gpt2-medium", "causal"),
    ("gpt2-large", "causal"),
    ("microsoft/phi-2", "causal"),
]

PROMPTS = ("zero_shot", "one_shot", "few_shot")
TASKS = ("meter_only", "rhyme_only", "natural_text", "combined")


def slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_id)


def json_path(root: Path, model: str, prompt: str, task: str) -> Path:
    return (
        root
        / "evaluation"
        / "results"
        / "baselines"
        / "prompt_only"
        / slug(model)
        / f"{prompt}_{task}.json"
    )


def is_missing(path: Path) -> bool:
    if not path.is_file():
        return True
    try:
        return path.stat().st_size < 50
    except OSError:
        return True


def build_export(
    model: str,
    model_type: str,
    prompt: str,
    task: str,
    n: int,
    strict_eval: str,
    hf_home: str | None,
) -> str:
    parts = [
        "ALL",
        f"MODEL={model}",
        f"MODEL_TYPE={model_type}",
        f"PROMPT={prompt}",
        f"BASELINE_TASK={task}",
        f"N={n}",
        f"STRICT_EVAL={strict_eval}",
    ]
    if hf_home:
        parts.append(f"THESIS_HF_HOME={hf_home}")
    return ",".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit only missing baseline result JSONs")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root (default: repo root)",
    )
    ap.add_argument("--n", type=int, default=500, help="Sample cap per job (default: 500)")
    ap.add_argument(
        "--strict-eval",
        choices=("0", "1"),
        default="1",
        help="STRICT_EVAL passed to Slurm (default: 1)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only (default if neither --execute nor --dry-run: dry-run)",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Run sbatch for each missing combo",
    )
    ap.add_argument(
        "--by-model",
        action="store_true",
        help="Group printed commands by Hugging Face model id",
    )
    ap.add_argument(
        "--only-model",
        action="append",
        metavar="MODEL_ID",
        help="Restrict to this Hugging Face model id (repeat flag for several). E.g. --only-model gpt2-medium",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    slurm = root / "scripts" / "run_prompt_baseline_all_tasks.slurm"
    if not slurm.is_file():
        sys.exit(f"Missing Slurm script: {slurm}")

    hf_home = os.environ.get("THESIS_HF_HOME") or None
    only = set(args.only_model) if args.only_model else None
    missing: list[tuple[str, str, str, str]] = []
    for model, mtype in SPECS:
        if only is not None and model not in only:
            continue
        for prompt in PROMPTS:
            for task in TASKS:
                p = json_path(root, model, prompt, task)
                if is_missing(p):
                    missing.append((model, mtype, prompt, task))

    if not missing:
        print("No missing baseline JSONs (all present for SPECS × prompts × tasks).", file=sys.stderr)
        sys.exit(0)

    dry = args.dry_run or not args.execute
    if args.execute and args.dry_run:
        sys.exit("Use only one of --execute or --dry-run")

    by_model: dict[str, list[tuple[str, str, str, str]]] = {}
    for row in missing:
        by_model.setdefault(row[0], []).append(row)

    print(f"# Missing count: {len(missing)} (N={args.n} per job, one task per job)", file=sys.stderr)

    def one_sbatch(model: str, mtype: str, prompt: str, task: str) -> list[str]:
        exp = build_export(model, mtype, prompt, task, args.n, args.strict_eval, hf_home)
        cmd = [
            "sbatch",
            f"--chdir={root}",
            f"--export={exp}",
            str(slurm),
        ]
        return cmd

    if args.by_model and dry:
        for model in sorted(by_model.keys()):
            rows = by_model[model]
            print(f"\n# --- {model} ({len(rows)} missing) ---")
            for m, t, pr, tk in rows:
                exp = build_export(m, t, pr, tk, args.n, args.strict_eval, hf_home)
                print(
                    "sbatch "
                    f"--chdir={shlex.quote(str(root))} "
                    f"--export={shlex.quote(exp)} "
                    f"{shlex.quote(str(slurm))}"
                )
        return

    for m, t, pr, tk in missing:
        cmd = one_sbatch(m, t, pr, tk)
        if dry:
            exp = build_export(m, t, pr, tk, args.n, args.strict_eval, hf_home)
            print(
                f"sbatch --chdir={root!s} "
                f'--export="{exp}" '
                f"{slurm!s}"
            )
        else:
            subprocess.run(cmd, check=True)

    if not dry:
        print(f"Submitted {len(missing)} jobs.", file=sys.stderr)


if __name__ == "__main__":
    main()
