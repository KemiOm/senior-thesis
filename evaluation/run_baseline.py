"""
Baseline pass: run corpus coverage metrics on ground-truth labels.

Computes meter, rhyme, CMU coverage, end-stopping, and caesura for the full corpus
and for each split (train, dev, test, held_out_poets, held_out_poems) when those files exist.

Writes `evaluation/baseline_results.json`. Those numbers describe how complete the
annotations are

Run from project root: python evaluation/run_baseline.py

Needs `output/corpus.db` (from `python scripts/export_sqlite.py`) and
`evaluation/splits/*.json` (from `evaluation/splits.py`).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import sqlite3

from evaluation.metrics import DB_PATH, compute_metrics
from evaluation.splits import SPLITS_DIR, load_split

RESULTS_PATH = Path(__file__).resolve().parent / "baseline_results.json"


def main():
    """
    Compute metrics on full corpus and each split, then save to baseline_results.json.
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: python scripts/export_sqlite.py")
        return

    conn = sqlite3.connect(DB_PATH)

    # Full corpus baseline.
    results = {"full_corpus": compute_metrics(conn)}

    # Per-split metrics. Skips splits that haven't been created yet.
    for split_name in ["train", "dev", "test", "held_out_poets", "held_out_poems"]:
        split_path = SPLITS_DIR / f"{split_name}.json"
        if split_path.exists():
            poem_ids = load_split(split_name)
            results[split_name] = compute_metrics(conn, poem_ids)
        else:
            results[split_name] = {"error": f"split {split_name} not found"}

    conn.close()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline results saved to {RESULTS_PATH}")
    print(json.dumps(results["full_corpus"], indent=2))


if __name__ == "__main__":
    main()
