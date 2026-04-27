#!/usr/bin/env python3
"""
Few-shot baseline rollup + model choice (thesis / poster).

Reads evaluation/baseline_report/model_comparison.csv, focuses on the three
supervised tasks (meter_only, rhyme_only, combined), and draws:
  (1) Heatmap of exact-match % by model × task
  (2) Short definitions of exact match vs Levenshtein similarity + rationale
      for choosing FLAN-T5-large as the LoRA backbone.

Run from repo root:
  python3 visualizations/baseline_selection_diagram.py
  python3 visualizations/baseline_selection_diagram.py --dpi 300 \\
    --out visualizations/out/baseline_few_shot_selection_poster.png
"""

from __future__ import annotations

import argparse
import csv
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "evaluation/baseline_report/model_comparison.csv"

TASKS = ("meter_only", "rhyme_only", "combined")
TASK_LABELS = ("meter only", "rhyme only", "combined")

# Short labels for poster readability (order is applied after sorting by score).
SLUG_LABELS: dict[str, str] = {
    "google_flan-t5-large": "FLAN-T5-Large",
    "google_flan-t5-base": "FLAN-T5-Base",
    "google_flan-t5-small": "FLAN-T5-Small",
    "facebook_bart-base": "BART-base",
    "facebook_bart-large": "BART-large",
    "gpt2": "GPT-2",
    "gpt2-medium": "GPT-2-medium",
    "gpt2-large": "GPT-2-large",
    "microsoft_phi-2": "Phi-2",
    "meta-llama_Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "HuggingFaceTB_SmolLM-360M-Instruct": "SmolLM-360M",
}

CHOSEN_SLUG = "google_flan-t5-large"

SCORES_TEXT = (
    "Scores (rollup from per-example JSON; see evaluation/scoring/rollup.py)\n"
    "• Exact match %: fraction of test lines where the model string equals gold "
    "after task-specific normalization (character-level identity; one token off → miss).\n"
    "• Levenshtein similarity %: mean over lines of 100×(1 − normalized character edit "
    "distance); higher means closer strings when exact match is too strict "
    "(evaluation/scoring/edit_distance.py).\n"
    "• Model choice: FLAN-T5-Large is the strongest few-shot baseline on meter among "
    "candidates in this table; it is instruction-tuned seq2seq, so the same "
    "input→target trainer and LoRA setup apply to all three tasks without a "
    "separate causal decoding path."
)


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_matrix(
    rows: list[dict[str, str]],
    *,
    prompt_type: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return model slugs (sorted), exact_match matrix [n_models, 3], lev_sim same shape."""
    wanted: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        if r.get("prompt_type") != prompt_type:
            continue
        task = r.get("task") or ""
        if task not in TASKS:
            continue
        slug = r.get("model_slug") or ""
        if not slug:
            continue
        try:
            em = float(r["exact_match_pct"])
            lev = float(r["levenshtein_similarity_pct"])
        except (KeyError, ValueError):
            continue
        wanted[(slug, task)] = {"em": em, "lev": lev}

    slugs = sorted({k[0] for k in wanted.keys()})
    em = np.zeros((len(slugs), len(TASKS)), dtype=float)
    lev = np.zeros_like(em)
    for i, slug in enumerate(slugs):
        for j, task in enumerate(TASKS):
            cell = wanted.get((slug, task))
            if cell:
                em[i, j] = cell["em"]
                lev[i, j] = cell["lev"]
    # Sort rows: descending mean exact match on the three tasks (then max).
    means = em.mean(axis=1)
    maxes = em.max(axis=1)
    order = np.lexsort((-maxes, -means))
    slugs = [slugs[int(i)] for i in order]
    em = em[order]
    lev = lev[order]
    return slugs, em, lev


def _display_labels(slugs: list[str]) -> list[str]:
    return [SLUG_LABELS.get(s, s.replace("_", "/")) for s in slugs]


def draw(
    *,
    csv_path: Path,
    out_path: Path,
    dpi: int,
    prompt_type: str,
) -> Path:
    rows = _read_rows(csv_path)
    slugs, em, lev = _build_matrix(rows, prompt_type=prompt_type)
    if not slugs:
        raise SystemExit(f"No rows for prompt_type={prompt_type!r} and tasks {TASKS}")

    row_labels = _display_labels(slugs)
    chosen_idx = slugs.index(CHOSEN_SLUG) if CHOSEN_SLUG in slugs else None

    fig_h = max(7.5, 2.2 + 0.38 * len(slugs))
    fig = plt.figure(figsize=(11.2, fig_h), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.05, 1.0], hspace=0.34, wspace=0.28)

    ax_em = fig.add_subplot(gs[0, 0])
    ax_lev = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis("off")

    vmax_em = max(50.0, float(em.max()) * 1.05)
    im_em = ax_em.imshow(em, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax_em)
    ax_em.set_xticks(range(len(TASKS)))
    ax_em.set_xticklabels(TASK_LABELS, fontsize=9)
    ax_em.set_yticks(range(len(slugs)))
    ax_em.set_yticklabels(row_labels, fontsize=8.5)
    ax_em.set_title(f"Exact match % ({prompt_type.replace('_', '-')})", fontsize=10.5, fontweight="600")
    plt.colorbar(im_em, ax=ax_em, fraction=0.046, pad=0.02)
    for i in range(em.shape[0]):
        for j in range(em.shape[1]):
            v = em[i, j]
            ax_em.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="#0f172a" if v < 35 else "white")

    im_lev = ax_lev.imshow(lev, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    ax_lev.set_xticks(range(len(TASKS)))
    ax_lev.set_xticklabels(TASK_LABELS, fontsize=9)
    ax_lev.set_yticks(range(len(slugs)))
    ax_lev.set_yticklabels(row_labels, fontsize=8.5)
    ax_lev.set_title(f"Levenshtein similarity % ({prompt_type.replace('_', '-')})", fontsize=10.5, fontweight="600")
    plt.colorbar(im_lev, ax=ax_lev, fraction=0.046, pad=0.02)
    for i in range(lev.shape[0]):
        for j in range(lev.shape[1]):
            v = lev[i, j]
            ax_lev.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 55 else "#0f172a",
            )

    if chosen_idx is not None:
        for ax in (ax_em, ax_lev):
            ax.add_patch(
                Rectangle(
                    (-0.52, chosen_idx - 0.5),
                    len(TASKS) + 1.04,
                    1.0,
                    fill=False,
                    edgecolor="#b45309",
                    linewidth=2.4,
                    linestyle="-",
                )
            )

    fig.suptitle(
        "Prompt-only baselines → backbone for LoRA (three formal tasks; natural_text omitted)",
        fontsize=11.2,
        fontweight="600",
        color="#0f172a",
        y=0.98,
    )

    ax_txt.text(
        0.0,
        1.0,
        textwrap.fill(SCORES_TEXT, width=118),
        ha="left",
        va="top",
        fontsize=8.4,
        color="#334155",
        linespacing=1.35,
        transform=ax_txt.transAxes,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="model_comparison.csv path")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "visualizations/out/baseline_few_shot_selection.png",
        help="Output PNG path",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--prompt-type",
        default="few_shot",
        help="Which prompt_type rows to plot (default: few_shot)",
    )
    args = p.parse_args()
    outp = draw(csv_path=args.csv, out_path=args.out, dpi=args.dpi, prompt_type=args.prompt_type)
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
