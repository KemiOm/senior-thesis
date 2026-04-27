#!/usr/bin/env python3
"""
Three-stage SFT diagram for posters — poetry adaptation of the common
(prompt → gold → SFT) flow, using real (input, target) pairs from this repo.

Run from repo root:
  python3 visualizations/sft_supervision_diagram.py
  python3 visualizations/sft_supervision_diagram.py --task combined --dpi 300 \\
    --out visualizations/out/sft_supervision_combined.png
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

# Headless / poster-safe backend before pyplot import
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent

# Real pairs from the corpus (ECPA TEI → extract → normalize → annotate).
# This line is Night I, line 0 in output/poems_annotated/ayo19-w0010.json (Edward Young);
# combined bundle matches scripts/run_prompt_eval.py:line_row_to_labels (stress, meter_type,
# rhyme key from CMU phonology, end_stopped, caesura index).
EXAMPLES = {
    "meter_only": (
        "Tired Nature's sweet restorer, balmy Sleep!",
        "+-+-+-+-+-+",
    ),
    "rhyme_only": (
        "Tired Nature's sweet restorer, balmy Sleep!",
        "IY1 P",
    ),
    "combined": (
        "Tired Nature's sweet restorer, balmy Sleep!",
        "stress:+-+-+-+-+-+|meter_type:iambic Pentameter|rhyme:IY1 P|end:1|caesura:3",
    ),
}

# Captions under each stage (parallel to generic SFT slide wording).
STAGE1_CAPTION = (
    "Training input sampled from the annotated poetry corpus\n"
    "(ECPA TEI → line JSON → normalized text; pairs in output/training_data/)"
)
STAGE2_CAPTION = (
    "Gold structured target from the annotation pipeline\n"
    "(CMU pronunciations + Poesy/Prosodic scansion; not human-written demos)"
)
STAGE3_CAPTION = (
    "Supervised fine-tuning: LoRA adapters on FLAN-T5-large\n"
    "(seq2seq loss on input → target pairs for meter_only, rhyme_only, or combined)"
)


def _rounded_box(ax, xy, w, h, *, facecolor: str, edgecolor: str = "#334155"):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
            mutation_aspect=0.35,
        )
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color="#2563eb",
            zorder=2,
        )
    )


def _wrap(s: str, width: int = 40) -> str:
    spaced = s.replace("|", " | ")
    return textwrap.fill(spaced, width=width, break_long_words=True, break_on_hyphens=False)


def draw(*, task: str, out_path: Path, dpi: int = 200) -> Path:
    inp, tgt = EXAMPLES[task]

    fig_w, fig_h = 14.0, 4.35
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    bw, bh = 3.95, 1.55
    y_box = 1.55
    gap = 0.5
    x1 = 0.4
    x2 = x1 + bw + gap + 0.85
    x3 = x2 + bw + gap + 0.85

    _rounded_box(ax, (x1, y_box), bw, bh, facecolor="#e0f2fe")
    _rounded_box(ax, (x2, y_box), bw, bh, facecolor="#f1f5f9")
    _rounded_box(ax, (x3, y_box), bw, bh, facecolor="#f1f5f9")

    mid_y = y_box + bh / 2
    _arrow(ax, x1 + bw + 0.04, mid_y, x2 - 0.04, mid_y)
    _arrow(ax, x2 + bw + 0.04, mid_y, x3 - 0.04, mid_y)

    fs_in = 10.0 if task != "combined" else 7.6
    fs_sft = 10.0
    fs_cap = 7.6
    y_caption = y_box - 0.02

    ax.text(
        x1 + bw / 2,
        y_box + bh * 0.52,
        inp,
        ha="center",
        va="center",
        fontsize=fs_in,
        color="#0f172a",
    )
    ax.text(x1 + bw / 2, y_caption, STAGE1_CAPTION, ha="center", va="top", fontsize=fs_cap, color="#475569")

    tgt_display = _wrap(tgt, 34) if task == "combined" else tgt
    ax.text(
        x2 + bw / 2,
        y_box + bh * 0.52,
        tgt_display,
        ha="center",
        va="center",
        fontsize=fs_in,
        family="monospace" if task == "combined" else "sans-serif",
        color="#0f172a",
    )
    ax.text(x2 + bw / 2, y_caption, STAGE2_CAPTION, ha="center", va="top", fontsize=fs_cap, color="#475569")

    ax.text(
        x3 + bw / 2,
        y_box + bh * 0.58,
        "SFT",
        ha="center",
        va="center",
        fontsize=13,
        weight="700",
        color="#0f172a",
    )
    ax.text(
        x3 + bw / 2,
        y_box + bh * 0.28,
        "LoRA  ·  FLAN-T5-large",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#334155",
    )
    ax.text(x3 + bw / 2, y_caption, STAGE3_CAPTION, ha="center", va="top", fontsize=fs_cap, color="#475569")

    title = f"Supervised fine-tuning setup — {task.replace('_', ' ')} (poetry)"
    ax.text(
        fig_w / 2,
        fig_h - 0.2,
        title,
        ha="center",
        va="top",
        fontsize=11,
        weight="600",
        color="#0f172a",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        choices=tuple(EXAMPLES.keys()),
        default="meter_only",
        help="Which schema to illustrate (default: meter_only, cleanest on a poster).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "visualizations/out/sft_supervision_poetry.png",
        help="Output PNG path",
    )
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    outp = draw(task=args.task, out_path=args.out, dpi=args.dpi)
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
