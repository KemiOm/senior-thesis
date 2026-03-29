"""Generate a meter-type distribution figure from corpus.db.

Run from project root:
  python scripts/helpers/diagnostic/meter_distribution_figure.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sqlite3

DB_PATH = ROOT / "output" / "corpus.db"
OUTPUT_DIR = ROOT / "docs" / "figures"
OUTPUT_PDF = OUTPUT_DIR / "meter_distribution.pdf"

# Maximum number of meter types to show as separate bars; rest grouped as "Other".
MAX_BARS = 12


def shorten_label(meter_type: str) -> str:
    """Shorten meter_type for x-axis if needed."""
    s = (meter_type or "").strip()
    if not s:
        return "?"
    # Optional abbreviations for long names
    if s == "iambic pentameter":
        return "Iambic pent."
    if s == "iambic tetrameter":
        return "Iambic tetr."
    if s == "trochaic pentameter":
        return "Trochaic pent."
    if s == "trochaic tetrameter":
        return "Trochaic tetr."
    return s[:18] + ("..." if len(s) > 18 else "")


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}. Run export_sqlite.py first.")
        return 1

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT meter_type, COUNT(*) AS n
        FROM lines
        WHERE meter_type IS NOT NULL AND meter_type != '' AND meter_type != 'unknown'
        GROUP BY meter_type
        ORDER BY n DESC
    """).fetchall()
    conn.close()

    if not rows:
        print("No meter_type counts found.")
        return 1

    # Build labels and counts (same logic for CSV and plot).
    if len(rows) > MAX_BARS:
        top = rows[: MAX_BARS - 1]
        other_count = sum(n for _, n in rows[MAX_BARS - 1 :])
        labels = [shorten_label(m) for m, _ in top] + ["Other"]
        counts = [n for _, n in top] + [other_count]
    else:
        labels = [shorten_label(m) for m, _ in rows]
        counts = [n for _, n in rows]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "meter_distribution.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,meter_type,count\n")
        for i, (label, count) in enumerate(zip(labels, counts)):
            f.write(f'{i},"{label}",{count}\n')
    print(f"Data for pgfplots: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. pip install matplotlib")
        return 1

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    bars = ax.bar(x, counts, color="steelblue", edgecolor="navy", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of lines")
    ax.set_xlabel("Meter type")
    ax.set_title("Meter type distribution (corpus lines with known meter)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
