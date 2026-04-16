#!/usr/bin/env python3
"""
Thesis-style protocol diagrams: (1) baseline vs SFT on identical test JSON;
(2) train/dev vs held-out test data flow; (3) structured corpus prep through annotated lines.

Outputs under visualizations/out/:
  - experiment_protocol_baseline_vs_sft.png
  - train_dev_test_schematic.png
  - structured_dataset_pipeline.png (TEI → extract → annotate; three steps)
  - structured_dataset_pipeline_column.png (vertical stack, same three steps)
  - structured_dataset_pipeline_poster.png (poster-sized three steps)

Run:  python visualizations/pipeline_figures.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Non-interactive backend first (avoids GUI-related crashes in headless / sandbox runs).
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")

THESIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "out"


def _box(ax, xy, w, h, text, *, fc, ec="#333333", fontsize=10, lw=1.2, style="round,pad=0.02"):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=style,
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=0.35,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1a1a1a",
        linespacing=1.15,
        wrap=True,
    )
    return p


def _arrow(ax, xy_start, xy_end, *, color="#333333", lw=1.5, style="->", connectionstyle="arc3,rad=0"):
    from matplotlib.patches import FancyArrowPatch

    arr = FancyArrowPatch(
        xy_start,
        xy_end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        edgecolor=color,
        facecolor=color,
        connectionstyle=connectionstyle,
        zorder=1,
    )
    ax.add_patch(arr)


def plot_dual_path_matched_eval(out_path: Path, *, dpi: int = 200) -> None:
    """
    Baseline (prompt-only) vs LoRA SFT, both feeding the same held-out test + eval harness.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=dpi)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Column headers
    ax.text(2.4, 6.55, "Supervision / selection", ha="center", fontsize=11, fontweight="600", color="#222")
    ax.text(6.3, 6.55, "Model weights", ha="center", fontsize=11, fontweight="600", color="#222")
    ax.text(11.25, 6.55, "Evaluation (identical protocol)", ha="center", fontsize=11, fontweight="600", color="#222")

    ax.text(
        7.0,
        6.95,
        "Matched evaluation: pretrained vs LoRA on the same held-out test JSON",
        ha="center",
        fontsize=12,
        fontweight="700",
        color="#0d2137",
    )

    # Colors
    c_train = "#f4d4c8"  # warm — train/dev only
    c_test = "#c5e0f0"  # cool — test
    c_neutral = "#e8e8e8"
    c_sft = "#d4e8d4"

    y_hi, y_lo = 4.85, 1.35
    row_h = 1.55

    # --- Row A: Baseline ---
    ax.text(0.35, y_hi + row_h * 0.5, "A", fontsize=12, fontweight="700", color="#555", va="center")
    ax.text(0.72, y_hi + row_h * 0.5, "Baseline", fontsize=10, fontweight="600", color="#444", va="center", rotation=90)

    _box(
        ax,
        (0.95, y_hi),
        3.0,
        row_h,
        "No task-specific\nSFT on project labels",
        fc=c_neutral,
        fontsize=9.5,
    )
    _box(
        ax,
        (4.35, y_hi),
        3.35,
        row_h,
        "FLAN-T5-Large\n(pretrained checkpoint)",
        fc=c_neutral,
        fontsize=9.5,
    )
    _box(
        ax,
        (8.35, y_hi),
        5.15,
        row_h,
        "test.json  +  same prompts\n(run_prompt_eval)",
        fc=c_test,
        fontsize=9.5,
    )

    _arrow(ax, (3.98, y_hi + row_h * 0.55), (4.32, y_hi + row_h * 0.55))
    _arrow(ax, (7.73, y_hi + row_h * 0.55), (8.32, y_hi + row_h * 0.55))

    # --- Row B: SFT ---
    ax.text(0.35, y_lo + row_h * 0.5, "B", fontsize=12, fontweight="700", color="#555", va="center")
    ax.text(0.72, y_lo + row_h * 0.5, "LoRA SFT", fontsize=10, fontweight="600", color="#444", va="center", rotation=90)

    _box(
        ax,
        (0.95, y_lo),
        3.0,
        row_h,
        "train.json  +  dev.json\n(project gold;\nseparate LoRA run per task)",
        fc=c_train,
        fontsize=9,
    )
    _box(
        ax,
        (4.35, y_lo),
        3.35,
        row_h,
        "LoRA fine-tune\n→ merge → merged\nweights for inference",
        fc=c_sft,
        fontsize=9,
    )
    _box(
        ax,
        (8.35, y_lo),
        5.15,
        row_h,
        "Same test.json  +  same prompts\n(same script, same decoding cap)",
        fc=c_test,
        fontsize=9.5,
    )

    _arrow(ax, (3.98, y_lo + row_h * 0.55), (4.32, y_lo + row_h * 0.55))
    _arrow(ax, (7.73, y_lo + row_h * 0.55), (8.32, y_lo + row_h * 0.55))

    # Bracket: same eval column
    from matplotlib.patches import FancyBboxPatch

    bracket = FancyBboxPatch(
        (8.25, y_lo - 0.15),
        5.35,
        (y_hi + row_h) - y_lo + 0.3,
        boxstyle="round,pad=0.03",
        linewidth=2.0,
        edgecolor="#2a6f97",
        facecolor="none",
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.add_patch(bracket)
    ax.text(
        13.75,
        (y_hi + y_lo + row_h) / 2 + 0.2,
        "same\ngold",
        ha="left",
        va="center",
        fontsize=8,
        color="#2a6f97",
        fontweight="600",
    )

    # Shared metrics below test boxes (conceptually one pipeline)
    _box(
        ax,
        (8.35, 0.25),
        5.15,
        0.85,
        "Metrics: exact match (+ structured / field-level where defined)",
        fc="#eef6fa",
        ec="#2a6f97",
        fontsize=9,
    )
    _arrow(ax, (10.9, y_lo), (10.9, 1.12), lw=1.8)
    _arrow(ax, (10.9, y_hi), (10.9, 1.12), lw=1.8)

    ax.text(
        7.0,
        0.08,
        "Train and dev are used only for SFT and checkpoint selection; test is held out for all headline numbers.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#444",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_train_dev_test_schematic(out_path: Path, *, dpi: int = 200) -> None:
    """
    Emphasize where gold labels are consumed: fit on train/dev only; score on test only.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=dpi)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        6.0,
        5.65,
        "Train / dev vs test: where gold labels are used",
        ha="center",
        fontsize=12,
        fontweight="700",
        color="#0d2137",
    )

    c_corpus = "#e6e0f0"
    c_split = "#fff8e6"
    c_train = "#f4d4c8"
    c_dev = "#f0c9a8"
    c_test = "#c5e0f0"
    c_forbidden = "#f0f0f0"

    _box(ax, (0.4, 2.6), 2.4, 1.35, "Annotated corpus\n(SQLite + splits)", fc=c_corpus, fontsize=9.5)
    _box(ax, (3.2, 2.6), 2.2, 1.35, "Split lists\n(train / dev / test\npoem IDs)", fc=c_split, fontsize=9.5)

    _arrow(ax, (2.82, 3.28), (3.18, 3.28))

    _box(ax, (0.35, 0.55), 2.55, 1.25, "train.json\n(project gold)", fc=c_train, fontsize=10)
    _box(ax, (3.15, 0.55), 2.35, 1.25, "dev.json\n(selection / early stop)", fc=c_dev, fontsize=9.5)
    _box(ax, (5.75, 0.55), 2.85, 1.25, "test.json\n(held out)", fc=c_test, fontsize=10)

    _arrow(ax, (4.3, 2.6), (1.6, 1.82), connectionstyle="arc3,rad=-0.15")
    _arrow(ax, (4.45, 2.6), (4.3, 1.82), connectionstyle="arc3,rad=0")
    _arrow(ax, (4.6, 2.6), (7.1, 1.82), connectionstyle="arc3,rad=0.12")

    _box(ax, (9.15, 0.45), 2.55, 1.45, "LoRA SFT\n(updates from\ntrain + dev only)", fc="#d4e8d4", fontsize=9.5)
    # Supervision flows only from train + dev into SFT (test never updates weights).
    _arrow(ax, (2.92, 1.05), (9.12, 0.85), connectionstyle="arc3,rad=-0.08", lw=1.5)
    _arrow(ax, (5.52, 1.12), (9.12, 1.05), connectionstyle="arc3,rad=0.06", lw=1.5)

    _box(
        ax,
        (9.15, 3.05),
        2.55,
        1.35,
        "Prompt eval +\nmetrics",
        fc=c_test,
        ec="#2a6f97",
        fontsize=10,
    )
    # Test split: prompts + gold for scoring only (no gradients).
    _arrow(ax, (7.05, 1.35), (9.12, 3.05), connectionstyle="arc3,rad=0.22", lw=1.8)
    ax.text(7.75, 2.42, "test prompts\n+ test gold", ha="center", fontsize=8, color="#2a6f97", fontweight="600")
    # Trained checkpoint runs forward on test inputs.
    _arrow(ax, (10.42, 1.92), (10.42, 3.02), lw=1.6)
    ax.text(10.95, 2.45, "merged\nmodel", ha="left", fontsize=7.5, color="#2d5a2d", fontweight="600")

    _box(
        ax,
        (3.0, 4.35),
        5.8,
        0.95,
        "No gradient from test — test JSON is not used to choose hyperparameters in the automated pipeline",
        fc=c_forbidden,
        ec="#999",
        fontsize=8.5,
        lw=1.0,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_structured_dataset_pipeline(out_path: Path, *, dpi: int = 200) -> None:
    """
    TEI → extract → annotate (three steps only; stops at line-level annotation).
    """
    import matplotlib.pyplot as plt

    fig_w, fig_h = 8.2, 3.35
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    cx = fig_w / 2
    ax.text(
        cx,
        fig_h - 0.22,
        "Structured corpus prep (through annotated lines)",
        ha="center",
        fontsize=11.5,
        fontweight="700",
        color="#0d2137",
    )

    c_src = "#e8e4f0"
    c_proc = "#fff4e0"
    c_ann = "#d8edd8"

    w = 2.38
    h = 1.38
    gap = 0.28
    y1 = 1.05
    x0 = 0.42
    xs1 = [x0 + i * (w + gap) for i in range(3)]

    _box(
        ax,
        (xs1[0], y1),
        w,
        h,
        "ECPA archive\nTEI P5 XML\n(works / lines)",
        fc=c_src,
        fontsize=8.8,
    )
    _box(
        ax,
        (xs1[1], y1),
        w,
        h,
        "Extract + normalize\n(batch scripts →\n`output/poems_*`)",
        fc=c_proc,
        fontsize=8.5,
    )
    _box(
        ax,
        (xs1[2], y1),
        w,
        h,
        "Annotate lines\nPoesy / Prosodic +\nCMU pronunciations",
        fc=c_ann,
        fontsize=8.5,
    )

    for i in range(2):
        xa = xs1[i] + w
        xb = xs1[i + 1]
        ym = y1 + h * 0.5
        _arrow(ax, (xa + 0.02, ym), (xb - 0.02, ym), lw=1.55)

    ax.text(
        cx,
        0.2,
        "Later steps (SQLite, splits, task JSONs) build supervised SFT/eval files from this corpus.",
        ha="center",
        fontsize=7.6,
        style="italic",
        color="#444",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_structured_dataset_pipeline_column(out_path: Path, *, dpi: int = 200) -> None:
    """
    Vertical stack through annotate lines — minimal width for a poster column.
    """
    import matplotlib.pyplot as plt

    fig_w, fig_h = 4.35, 6.15
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    w = fig_w - 0.5
    h = 1.22
    x = 0.25
    gap = 0.36
    y0 = fig_h - 0.95

    ax.text(
        x + w / 2,
        fig_h - 0.28,
        "Corpus prep\n(through annotation)",
        ha="center",
        fontsize=10.5,
        fontweight="700",
        color="#0d2137",
    )

    c_src = "#e8e4f0"
    c_proc = "#fff4e0"
    c_ann = "#d8edd8"

    labels = [
        ("ECPA archive\nTEI P5 XML\n(works / lines)", c_src),
        ("Extract + normalize\n→ `output/poems_*`", c_proc),
        ("Annotate lines\nPoesy / Prosodic +\nCMU", c_ann),
    ]

    ys = [y0 - i * (h + gap) for i in range(3)]
    for (text, fc), y in zip(labels, ys):
        _box(ax, (x, y), w, h, text, fc=fc, fontsize=7.9)

    for i in range(2):
        y_top = ys[i + 1] + h
        y_bot = ys[i]
        xm = x + w * 0.5
        _arrow(ax, (xm, y_bot - 0.02), (xm, y_top + 0.02), lw=1.45)

    ax.text(
        x + w / 2,
        0.2,
        "DB / splits / task JSONs follow.",
        ha="center",
        fontsize=7.2,
        style="italic",
        color="#444",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_structured_dataset_pipeline_poster(out_path: Path, *, dpi: int = 300) -> None:
    """
    Poster-ready three-step flow: TEI → extract/normalize → annotate (stops at annotated lines).
    """
    import matplotlib.pyplot as plt

    fig_w, fig_h = 10.2, 4.35
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    cx = fig_w / 2
    ax.text(
        cx,
        fig_h - 0.28,
        "Structured corpus prep (through annotated lines)",
        ha="center",
        fontsize=16,
        fontweight="700",
        color="#0d2137",
    )
    ax.text(
        cx,
        fig_h - 0.72,
        "ECPA → extract & normalize → line-level prosody / phonology (CMU, Poesy, Prosodic)",
        ha="center",
        fontsize=11,
        color="#2f3a4a",
    )

    c_src = "#e8e4f0"
    c_proc = "#fff4e0"
    c_ann = "#d8edd8"

    h = 1.95
    y = 0.95
    w = 2.75
    gap = 0.42
    xs = [0.55 + i * (w + gap) for i in range(3)]

    _box(ax, (xs[0], y), w, h, "ECPA TEI XML\npoems\n(lines + stanzas)", fc=c_src, fontsize=10.5)
    _box(
        ax,
        (xs[1], y),
        w,
        h,
        "Extract + normalize\ntext lines\n(output/poems_*)",
        fc=c_proc,
        fontsize=10.2,
    )
    _box(
        ax,
        (xs[2], y),
        w,
        h,
        "Annotate per line\nCMUdict +\nProsodic / Poesy",
        fc=c_ann,
        fontsize=10.2,
    )

    for i in range(2):
        x0 = xs[i] + w
        x1 = xs[i + 1]
        ym = y + h * 0.52
        _arrow(ax, (x0 + 0.04, ym), (x1 - 0.04, ym), lw=2.0)

    ax.text(
        cx,
        0.38,
        "Per-line fields include e.g. stress, meter type, rhyme, boundaries (then stored for downstream JSON tasks).",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis protocol diagrams.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for PNGs (default: visualizations/out)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p1 = args.out_dir / "experiment_protocol_baseline_vs_sft.png"
    p2 = args.out_dir / "train_dev_test_schematic.png"
    p3 = args.out_dir / "structured_dataset_pipeline.png"
    p3b = args.out_dir / "structured_dataset_pipeline_column.png"
    p4 = args.out_dir / "structured_dataset_pipeline_poster.png"
    plot_dual_path_matched_eval(p1, dpi=args.dpi)
    plot_train_dev_test_schematic(p2, dpi=args.dpi)
    plot_structured_dataset_pipeline(p3, dpi=args.dpi)
    plot_structured_dataset_pipeline_column(p3b, dpi=args.dpi)
    plot_structured_dataset_pipeline_poster(p4, dpi=max(args.dpi, 300))
    print(
        f"Wrote {p1.resolve()}\nWrote {p2.resolve()}\nWrote {p3.resolve()}\n"
        f"Wrote {p3b.resolve()}\nWrote {p4.resolve()}"
    )


if __name__ == "__main__":
    main()
