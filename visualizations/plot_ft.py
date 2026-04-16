"""
Fine-tuning figures for the thesis.

1. **Training** — train/eval loss (optional learning-rate panel) from trainer_state.json.
2. **Data** — input/target length histograms; **split counts** (train/dev/test sizes).
3. **Eval** — grouped exact-match bars; **heatmap** (model × eval task); **prompt-style**
   heatmap (zero/one/few × task) for one slug; **combined** structured field bars.
4. **Research narrative** (same rollup JSONs):
   - **SFT vs prompt delta** — Δ exact match (merged SFT − base prompt) per task.
   - **Cross-task transfer matrix** — training condition (slug) × eval task.
   - **Compositionality gap** — combined vs single-task SFT on meter/rhyme.

Uses the same JSON layout as evaluation.scoring.rollup.collect_rows.

Requires: matplotlib. Token lengths: transformers + tokenizer cache.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, Union

BaselineDirs = Union[Path, str, Sequence[Union[Path, str]]]

THESIS_ROOT = Path(__file__).resolve().parent.parent

if str(THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THESIS_ROOT))


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting requires matplotlib: pip install matplotlib") from e
    return plt


def load_trainer_state(run_dir: Path | str) -> dict[str, Any]:
    """Load trainer_state.json from a completed Hugging Face Trainer run."""
    path = Path(run_dir) / "trainer_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_train_eval_losses(state: dict[str, Any]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Returns (train_points, eval_points) as lists of (step, loss).
    Train points use loss; eval points use eval_loss from log_history.
    """
    history = state.get("log_history") or []
    train: list[tuple[int, float]] = []
    evals: list[tuple[int, float]] = []
    for entry in history:
        step = int(entry.get("step", -1))
        if step < 0:
            continue
        if "loss" in entry and entry["loss"] is not None and "eval_loss" not in entry:
            train.append((step, float(entry["loss"])))
        if "eval_loss" in entry and entry["eval_loss"] is not None:
            evals.append((step, float(entry["eval_loss"])))
    return train, evals


def parse_learning_rate(state: dict[str, Any]) -> list[tuple[int, float]]:
    """(step, lr) from log_history."""
    history = state.get("log_history") or []
    pts: list[tuple[int, float]] = []
    for entry in history:
        step = int(entry.get("step", -1))
        if step < 0 or "learning_rate" not in entry or entry["learning_rate"] is None:
            continue
        pts.append((step, float(entry["learning_rate"])))
    return pts


def _short_slug(slug: str, max_len: int = 28) -> str:
    if len(slug) <= max_len:
        return slug
    return slug[: max_len - 1] + "…"


def plot_train_val_loss(
    run_dir: Path | str,
    *,
    out_path: Path | str | None = None,
    title: str | None = None,
    with_learning_rate: bool = False,
) -> Path | None:
    """
    Plot training and validation loss vs step.
    If with_learning_rate, add a second panel for LR vs step.
    """
    plt = _mpl()
    state = load_trainer_state(run_dir)
    train, evals = parse_train_eval_losses(state)
    lr_pts = parse_learning_rate(state) if with_learning_rate else []

    if with_learning_rate and lr_pts:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        ax0, ax1 = axes[0], axes[1]
    else:
        fig, ax0 = plt.subplots(figsize=(8, 4))
        ax1 = None

    if train:
        ax0.plot([t[0] for t in train], [t[1] for t in train], label="train loss", alpha=0.85)
    if evals:
        ax0.plot([t[0] for t in evals], [t[1] for t in evals], label="eval loss", alpha=0.85)
    ax0.set_ylabel("loss")
    ax0.set_title(title or f"Training — {Path(run_dir).name}")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    if ax1 is not None and lr_pts:
        ax1.plot([t[0] for t in lr_pts], [t[1] for t in lr_pts], color="tab:green", alpha=0.85)
        ax1.set_xlabel("step")
        ax1.set_ylabel("learning rate")
        ax1.grid(True, alpha=0.3)
    else:
        ax0.set_xlabel("step")

    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        return outp
    plt.show()
    return None


def _load_training_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _char_lens(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    inp = [len((r.get("input") or "")) for r in rows]
    tgt = [len((r.get("target") or "")) for r in rows]
    return inp, tgt


def _token_lens(rows: list[dict[str, Any]], model_id: str) -> tuple[list[int], list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    inp = [len(tok.encode(r.get("input") or "", add_special_tokens=False)) for r in rows]
    tgt = [len(tok.encode(r.get("target") or "", add_special_tokens=False)) for r in rows]
    return inp, tgt


def length_histograms(
    task: str,
    split: str = "train",
    *,
    data_root: Path | None = None,
    model_id: str | None = None,
    out_path: Path | str | None = None,
    title_suffix: str = "",
) -> Path | None:
    """
    Overlaid histograms of input vs target lengths (characters, or tokens if model_id set).
    """
    plt = _mpl()
    root = data_root or (THESIS_ROOT / "output" / "training_data")
    path = root / task / f"{split}.json"
    rows = _load_training_json(path)
    if model_id:
        inp, tgt = _token_lens(rows, model_id)
        unit = "tokens"
    else:
        inp, tgt = _char_lens(rows)
        unit = "chars"

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = min(60, max(10, len(rows) // 30))
    ax.hist(inp, bins=bins, alpha=0.55, label=f"input ({unit})", density=True)
    ax.hist(tgt, bins=bins, alpha=0.55, label=f"target ({unit})", density=True)
    ax.set_xlabel(f"length ({unit})")
    ax.set_ylabel("density")
    ax.set_title(f"{task} / {split}{title_suffix}")
    ax.legend()
    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        return outp
    plt.show()
    return None


def plot_baseline_vs_lora(
    baseline_dir: Path | str,
    *,
    prompt_type: str = "few_shot",
    out_path: Path | str | None = None,
    tasks: tuple[str, ...] = ("meter_only", "rhyme_only", "combined"),
    slug_globs: list[str] | None = None,
    title: str | None = None,
) -> Path | None:
    """
    Grouped bar chart: exact match % per task, one cluster of bars per model slug.
    Uses the same JSON discovery as evaluation.scoring.rollup.collect_rows.
    """
    from evaluation.scoring.rollup import collect_rows

    plt = _mpl()
    rows = collect_rows(
        baseline_dir=Path(baseline_dir),
        slug_globs=slug_globs,
        include_natural_text_form=False,
        include_structured_partial=False,
    )
    rows = [r for r in rows if r.get("prompt_type") == prompt_type and r.get("task") in tasks]
    if not rows:
        raise ValueError(
            f"No rows for prompt_type={prompt_type!r} under {baseline_dir}. "
            "Check paths and that result JSONs exist."
        )

    # slug -> task -> pct
    by_slug: dict[str, dict[str, float]] = {}
    for r in rows:
        slug = r["model_slug"]
        task = r["task"]
        by_slug.setdefault(slug, {})[task] = float(r["exact_match_pct"])

    slugs = sorted(by_slug.keys())
    n_tasks = len(tasks)
    n_slugs = len(slugs)
    x = [float(i) for i in range(n_tasks)]
    width = min(0.8 / max(n_slugs, 1), 0.25)
    fig, ax = plt.subplots(figsize=(max(6, n_tasks * 2.2), max(4, n_slugs * 0.4)))

    shift = width * (n_slugs - 1) / 2
    for i, slug in enumerate(slugs):
        offsets = [xi - shift + i * width for xi in x]
        vals = [by_slug[slug].get(t, 0.0) for t in tasks]
        ax.bar(offsets, vals, width, label=slug[:40] + ("…" if len(slug) > 40 else ""))

    ax.set_xticks(x)
    ax.set_xticklabels(list(tasks))
    ax.set_ylabel("exact match %")
    ax.set_title(title or f"Exact match — prompt_type={prompt_type}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outp
    plt.show()
    return None


def _normalize_baseline_dirs(baseline_dir: BaselineDirs) -> list[Path]:
    if isinstance(baseline_dir, (str, Path)):
        return [Path(baseline_dir).resolve()]
    return [Path(p).resolve() for p in baseline_dir]


def _collect_rows_union(baseline_dir: BaselineDirs) -> list[dict]:
    from evaluation.scoring.rollup import collect_rows

    rows: list[dict] = []
    for d in _normalize_baseline_dirs(baseline_dir):
        rows.extend(
            collect_rows(
                baseline_dir=d,
                include_natural_text_form=False,
                include_structured_partial=False,
            )
        )
    return rows


def _avg_exact_by_task(rows: list[dict], slug_pattern: str) -> dict[str, float]:
    """Mean exact_match_pct per task for rows whose model_slug matches pattern (may be multiple runs)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not fnmatch.fnmatch(r["model_slug"], slug_pattern):
            continue
        buckets[r["task"]].append(float(r["exact_match_pct"]))
    return {t: sum(vs) / len(vs) for t, vs in buckets.items() if vs}


def plot_sft_delta(
    baseline_dir: BaselineDirs,
    *,
    base_slug_pattern: str,
    ft_slug_pattern: str,
    prompt_type: str = "few_shot",
    tasks: tuple[str, ...] = ("meter_only", "rhyme_only", "combined"),
    out_path: Path | str | None = None,
    title: str | None = None,
) -> Path | None:
    """
    Δ exact match % (SFT − prompt-only) per task — main “what did fine-tuning change?” figure.

    Point base_slug_pattern at the **pretrained** eval slug (e.g. google_flan-t5-large) and
    ft_slug_pattern at the **merged SFT** slug (e.g. *combined_lora* or full short slug).

    Pass multiple roots (e.g. evaluation/baselines and results) as a list or several CLI
    --baseline-dir values so pretrained and SFT JSONs can live in different folders.
    """
    plt = _mpl()
    rows = _collect_rows_union(baseline_dir)
    rows = [r for r in rows if r.get("prompt_type") == prompt_type]

    base = _avg_exact_by_task(rows, base_slug_pattern)
    ft = _avg_exact_by_task(rows, ft_slug_pattern)
    deltas = [ft.get(t, 0.0) - base.get(t, 0.0) for t in tasks]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2f855a" if d >= 0 else "#c53030" for d in deltas]
    ax.bar(list(tasks), deltas, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Δ exact match % (SFT − prompt)")
    ax.set_title(title or "Effect of SFT on structured prediction")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(deltas):
        va = "bottom" if v >= 0 else "top"
        ax.text(i, v, f"{v:+.1f}", ha="center", va=va, fontsize=10)

    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outp
    plt.show()
    return None


def plot_transfer_matrix(
    baseline_dir: BaselineDirs,
    *,
    prompt_type: str = "few_shot",
    train_tags: tuple[str, ...] = ("meter_only", "rhyme_only", "combined"),
    eval_tasks: tuple[str, ...] = ("meter_only", "rhyme_only", "combined"),
    out_path: Path | str | None = None,
    title: str | None = None,
) -> Path | None:
    """
    Rows = inferred **training condition** from model_slug (substring match), cols = **eval task**.
    Values = mean exact match % when multiple JSONs match. NaN if no data for that cell.

    Uses SFT result slugs (e.g. *meter_only_lora*, *combined_lora*). Pretrained-only slugs
    without a task token land in the "base" row only if you add "base" to train_tags.
    """
    plt = _mpl()
    rows = _collect_rows_union(baseline_dir)
    rows = [r for r in rows if r.get("prompt_type") == prompt_type]

    def infer_train_type(slug: str) -> str:
        for t in train_tags:
            if t in slug:
                return t
        return "base"

    mat: list[list[float]] = []
    for train in train_tags:
        row_vals = []
        for task in eval_tasks:
            vals = [
                float(r["exact_match_pct"])
                for r in rows
                if infer_train_type(r["model_slug"]) == train and r["task"] == task
            ]
            row_vals.append(sum(vals) / len(vals) if vals else float("nan"))
        mat.append(row_vals)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    arr = mat
    im = ax.imshow(arr, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(eval_tasks)))
    ax.set_xticklabels(list(eval_tasks), rotation=15, ha="right")
    ax.set_yticks(range(len(train_tags)))
    ax.set_yticklabels(list(train_tags))
    ax.set_xlabel("evaluation task")
    ax.set_ylabel("training condition (from slug)")
    ax.set_title(title or "Cross-task transfer (train → eval)")

    for i in range(len(train_tags)):
        for j in range(len(eval_tasks)):
            v = arr[i][j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, label="exact match %", fraction=0.046, pad=0.04)
    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outp
    plt.show()
    return None


def plot_compositionality_gap(
    baseline_dir: BaselineDirs,
    *,
    prompt_type: str = "few_shot",
    tasks: tuple[str, ...] = ("meter_only", "rhyme_only"),
    out_path: Path | str | None = None,
    title: str | None = None,
) -> Path | None:
    """
    For each task, compare best single-task SFT slug vs combined-task SFT slug (max exact % in rollup).
    """
    plt = _mpl()
    rows = _collect_rows_union(baseline_dir)
    rows = [r for r in rows if r.get("prompt_type") == prompt_type]

    def best_for(task: str, keyword: str) -> float:
        vals = [
            float(r["exact_match_pct"])
            for r in rows
            if r["task"] == task and keyword in r["model_slug"]
        ]
        return max(vals) if vals else 0.0

    combined_vals = [best_for(t, "combined") for t in tasks]
    single_vals = [best_for(t, t) for t in tasks]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(tasks)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], single_vals, width, label="single-task SFT", color="#3182ce", alpha=0.9)
    ax.bar([i + width / 2 for i in x], combined_vals, width, label="combined SFT", color="#744210", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(list(tasks))
    ax.set_ylabel("exact match % (best slug per condition)")
    ax.set_title(title or "Compositionality: combined vs single-task SFT")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outp
    plt.show()
    return None


def plot_dataset_split_counts(
    *,
    data_root: Path | None = None,
    tasks: tuple[str, ...] = ("meter_only", "rhyme_only", "combined"),
    splits: tuple[str, ...] = ("train", "dev", "test"),
    out_path: Path | str | None = None,
    title: str | None = None,
) -> Path | None:
    """Bar chart: number of training rows per task and split (what SFT is trained on)."""
    plt = _mpl()
    root = data_root or (THESIS_ROOT / "output" / "training_data")
    counts: dict[str, dict[str, int]] = {t: {} for t in tasks}
    for task in tasks:
        for sp in splits:
            p = root / task / f"{sp}.json"
            if p.is_file():
                counts[task][sp] = len(_load_training_json(p))
            else:
                counts[task][sp] = 0

    x = list(range(len(tasks)))
    n_sp = len(splits)
    width = min(0.8 / max(n_sp, 1), 0.22)
    shift = width * (n_sp - 1) / 2
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ("#2c5282", "#3182ce", "#63b3ed")
    for j, sp in enumerate(splits):
        offs = [xi - shift + j * width for xi in x]
        vals = [counts[t].get(sp, 0) for t in tasks]
        ax.bar(offs, vals, width, label=sp, color=colors[j % len(colors)], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(list(tasks))
    ax.set_ylabel("examples")
    ax.set_title(title or "SFT training data — examples per task and split")
    ax.legend(title="split")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        return outp
    plt.show()
    return None


def _collect_filtered_rows(
    baseline_dir: Path | str,
    *,
    prompt_type: str | None,
    tasks: tuple[str, ...],
    slug_globs: list[str] | None,
    include_structured_partial: bool,
) -> list[dict[str, Any]]:
    from evaluation.scoring.rollup import collect_rows

    rows = collect_rows(
        baseline_dir=Path(baseline_dir),
        slug_globs=slug_globs,
        include_natural_text_form=False,
        include_structured_partial=include_structured_partial,
    )
    out = []
    for r in rows:
        if r.get("task") not in tasks:
            continue
        if prompt_type is not None and r.get("prompt_type") != prompt_type:
            continue
        out.append(r)
    return out


def plot_eval_em_heatmap(
    baseline_dir: Path | str,
    *,
    prompt_type: str = "few_shot",
    out_path: Path | str | None = None,
    tasks: tuple[str, ...] = ("meter_only", "rhyme_only", "combined"),
    slug_globs: list[str] | None = None,
    title: str | None = None,
) -> Path | None:
    """
    Heatmap: rows = model slug, columns = eval task, cell = exact match %.
    Shows **cross-task transfer** (e.g. meter-trained model on rhyme eval).
    """
    plt = _mpl()
    rows = _collect_filtered_rows(
        baseline_dir,
        prompt_type=prompt_type,
        tasks=tasks,
        slug_globs=slug_globs,
        include_structured_partial=False,
    )
    if not rows:
        raise ValueError(f"No rows for heatmap under {baseline_dir}")

    by_slug: dict[str, dict[str, float]] = {}
    for r in rows:
        slug = r["model_slug"]
        by_slug.setdefault(slug, {})[r["task"]] = float(r["exact_match_pct"])

    slugs = sorted(by_slug.keys())
    mat = [[by_slug[s].get(t, float("nan")) for t in tasks] for s in slugs]

    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 2.2), max(4, len(slugs) * 0.45)))
    flat = [v for row in mat for v in row if v == v]
    vmax = max(100.0, max(flat) if flat else 0.0)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(list(tasks))
    ax.set_yticks(range(len(slugs)))
    ax.set_yticklabels([_short_slug(s, 36) for s in slugs], fontsize=8)
    ax.set_xlabel("eval task")
    ax.set_ylabel("model (slug)")
    ax.set_title(title or f"Exact match % — {prompt_type}")
    for i in range(len(slugs)):
        for j in range(len(tasks)):
            v = mat[i][j]
            if v != v:  # NaN
                continue
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="black" if v < 50 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="exact match %")
    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outp
    plt.show()
    return None


def plot_prompt_style_heatmap(
    baseline_dir: Path | str,
    *,
    slug_glob: str,
    out_path: Path | str | None = None,
    tasks: tuple[str, ...] = ("meter_only", "rhyme_only", "combined"),
    prompt_types: tuple[str, ...] = ("zero_shot", "one_shot", "few_shot"),
    title: str | None = None,
) -> Path | None:
    """
    For one model slug (glob), heatmap: prompt style × eval task → exact match %.
    Shows how much **prompting** alone moves the needle vs SFT (compare to LoRA slugs separately).
    """
    import fnmatch

    plt = _mpl()
    rows = _collect_filtered_rows(
        baseline_dir,
        prompt_type=None,
        tasks=tasks,
        slug_globs=None,
        include_structured_partial=False,
    )
    rows = [r for r in rows if fnmatch.fnmatch(r["model_slug"], slug_glob)]
    rows = [r for r in rows if r.get("prompt_type") in prompt_types]
    if not rows:
        raise ValueError(f"No rows for slug_glob={slug_glob!r}")

    by_pt: dict[str, dict[str, float]] = {}
    for r in rows:
        pt = r["prompt_type"]
        by_pt.setdefault(pt, {})[r["task"]] = float(r["exact_match_pct"])

    mat = [[by_pt.get(pt, {}).get(t, float("nan")) for t in tasks] for pt in prompt_types]

    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 2), 4))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(list(tasks))
    ax.set_yticks(range(len(prompt_types)))
    ax.set_yticklabels(list(prompt_types))
    ax.set_xlabel("eval task")
    ax.set_ylabel("prompt style")
    ax.set_title(title or f"Exact match % — slug matches {slug_glob!r}")
    for i, pt in enumerate(prompt_types):
        for j, _t in enumerate(tasks):
            v = mat[i][j]
            if v != v:
                continue
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > 50 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="exact match %")
    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outp
    plt.show()
    return None


def plot_combined_structured_by_slug(
    baseline_dir: Path | str,
    *,
    prompt_type: str = "few_shot",
    out_path: Path | str | None = None,
    slug_globs: list[str] | None = None,
    title: str | None = None,
) -> Path | None:
    """
    For **combined** task only: bar groups per slug for field-level metrics
    (same columns as model_comparison.csv structured block).
    """
    plt = _mpl()
    rows = _collect_filtered_rows(
        baseline_dir,
        prompt_type=prompt_type,
        tasks=("combined",),
        slug_globs=slug_globs,
        include_structured_partial=True,
    )
    rows = [r for r in rows if r.get("task") == "combined"]
    if not rows:
        raise ValueError("No combined rows with structured metrics")

    fields = [
        ("st_combined_meter_field_pct", "stress"),
        ("st_combined_meter_type_field_pct", "meter type"),
        ("st_combined_rhyme_field_pct", "rhyme"),
        ("st_combined_end_field_pct", "end"),
        ("st_combined_caesura_field_pct", "caesura"),
        ("st_combined_mean_field_pct", "mean field"),
        ("st_combined_all_four_pct", "all fields"),
    ]
    slugs = sorted({r["model_slug"] for r in rows})
    n_f = len(fields)
    x = list(range(n_f))
    n_s = len(slugs)
    width = min(0.8 / max(n_s, 1), 0.11)
    shift = width * (n_s - 1) / 2
    fig, ax = plt.subplots(figsize=(max(9, n_f * 1.4), max(4, n_s * 0.35)))

    for i, slug in enumerate(slugs):
        row = next((r for r in rows if r["model_slug"] == slug), None)
        if not row:
            continue
        offs = [xi - shift + i * width for xi in x]
        vals = [float(row.get(k, 0) or 0) for k, _ in fields]
        ax.bar(offs, vals, width, label=_short_slug(slug, 32))

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in fields], rotation=25, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(title or f"Combined task — structured field match — {prompt_type}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outp
    plt.show()
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tuning visualization helpers")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_loss = sub.add_parser("loss", help="Plot train/eval loss from trainer_state.json")
    p_loss.add_argument("--run-dir", type=Path, required=True)
    p_loss.add_argument("--out", type=Path, default=None)
    p_loss.add_argument("--with-lr", action="store_true", help="Second panel: learning rate vs step")

    p_len = sub.add_parser("lengths", help="Histogram input/target lengths")
    p_len.add_argument("--task", required=True)
    p_len.add_argument("--split", default="train")
    p_len.add_argument("--data-root", type=Path, default=None)
    p_len.add_argument("--model-id", default=None, help="If set, plot token lengths with this tokenizer")
    p_len.add_argument("--out", type=Path, default=None)

    p_bar = sub.add_parser("bars", help="Grouped exact-match bars from baseline JSONs")
    p_bar.add_argument("--baseline-dir", type=Path, required=True)
    p_bar.add_argument("--prompt-type", default="few_shot")
    p_bar.add_argument("--out", type=Path, default=None)
    p_bar.add_argument("--slug-glob", action="append", default=None)

    p_hm = sub.add_parser("heatmap", help="Model × task exact-match heatmap")
    p_hm.add_argument("--baseline-dir", type=Path, required=True)
    p_hm.add_argument("--prompt-type", default="few_shot")
    p_hm.add_argument("--out", type=Path, default=None)
    p_hm.add_argument("--slug-glob", action="append", default=None)

    p_ph = sub.add_parser("prompt-heatmap", help="Prompt style × task for one slug glob")
    p_ph.add_argument("--baseline-dir", type=Path, required=True)
    p_ph.add_argument("--slug-pattern", required=True, help="fnmatch pattern, e.g. google_flan-t5-large")
    p_ph.add_argument("--out", type=Path, default=None)

    p_sp = sub.add_parser("splits", help="Bar chart of train/dev/test sizes per task")
    p_sp.add_argument("--data-root", type=Path, default=None)
    p_sp.add_argument("--out", type=Path, default=None)

    p_cf = sub.add_parser("combined-fields", help="Combined-task structured field bars by slug")
    p_cf.add_argument("--baseline-dir", type=Path, required=True)
    p_cf.add_argument("--prompt-type", default="few_shot")
    p_cf.add_argument("--out", type=Path, default=None)
    p_cf.add_argument("--slug-glob", action="append", default=None)

    p_delta = sub.add_parser("delta", help="SFT vs prompt Δ exact match per task (main causal figure)")
    p_delta.add_argument(
        "--baseline-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more dirs with slug/*.json trees (e.g. evaluation/baselines results)",
    )
    p_delta.add_argument("--base-pattern", required=True, help="fnmatch for pretrained slug, e.g. google_flan-t5-large")
    p_delta.add_argument("--ft-pattern", required=True, help="fnmatch for merged SFT slug, e.g. *combined_lora*")
    p_delta.add_argument("--prompt-type", default="few_shot")
    p_delta.add_argument("--out", type=Path, default=None)

    p_tr = sub.add_parser("transfer", help="Cross-task transfer matrix (train condition × eval task)")
    p_tr.add_argument(
        "--baseline-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more dirs with slug/*.json (union of rows)",
    )
    p_tr.add_argument("--prompt-type", default="few_shot")
    p_tr.add_argument("--out", type=Path, default=None)

    p_comp = sub.add_parser("compositionality", help="Combined vs single-task SFT bars (meter/rhyme)")
    p_comp.add_argument(
        "--baseline-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more dirs with slug/*.json (union of rows)",
    )
    p_comp.add_argument("--prompt-type", default="few_shot")
    p_comp.add_argument("--out", type=Path, default=None)

    args = p.parse_args()
    if args.cmd == "loss":
        plot_train_val_loss(args.run_dir, out_path=args.out, with_learning_rate=args.with_lr)
    elif args.cmd == "lengths":
        length_histograms(
            args.task,
            split=args.split,
            data_root=args.data_root,
            model_id=args.model_id,
            out_path=args.out,
        )
    elif args.cmd == "bars":
        plot_baseline_vs_lora(
            args.baseline_dir,
            prompt_type=args.prompt_type,
            out_path=args.out,
            slug_globs=args.slug_glob,
        )
    elif args.cmd == "heatmap":
        plot_eval_em_heatmap(
            args.baseline_dir,
            prompt_type=args.prompt_type,
            out_path=args.out,
            slug_globs=args.slug_glob,
        )
    elif args.cmd == "prompt-heatmap":
        plot_prompt_style_heatmap(
            args.baseline_dir,
            slug_glob=args.slug_pattern,
            out_path=args.out,
        )
    elif args.cmd == "splits":
        plot_dataset_split_counts(data_root=args.data_root, out_path=args.out)
    elif args.cmd == "combined-fields":
        plot_combined_structured_by_slug(
            args.baseline_dir,
            prompt_type=args.prompt_type,
            out_path=args.out,
            slug_globs=args.slug_glob,
        )
    elif args.cmd == "delta":
        plot_sft_delta(
            list(args.baseline_dir),
            base_slug_pattern=args.base_pattern,
            ft_slug_pattern=args.ft_pattern,
            prompt_type=args.prompt_type,
            out_path=args.out,
        )
    elif args.cmd == "transfer":
        plot_transfer_matrix(
            list(args.baseline_dir),
            prompt_type=args.prompt_type,
            out_path=args.out,
        )
    elif args.cmd == "compositionality":
        plot_compositionality_gap(
            list(args.baseline_dir),
            prompt_type=args.prompt_type,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
