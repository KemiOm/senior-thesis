"""
Annotation coverage: run corpus metrics on ground-truth labels (no models).

Computes meter, rhyme, CMU coverage, end-stopping, and caesura for the full corpus
and for each split (train, dev, test, held_out_poets, held_out_poems) when those files exist.

Writes `evaluation/annotation_coverage.json`. Those numbers describe how complete the
annotations are—not prompt-baseline or fine-tuned model accuracy.

Run from project root: python evaluation/run_annotation_coverage.py

Needs `output/corpus.db` (from `python scripts/export_sqlite.py`) and
`evaluation/splits/*.json` (from `evaluation/splits.py`).

For Hugging Face prompt-only baselines, use `scripts/run_prompt_baseline.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import sqlite3

from evaluation.metrics import DB_PATH, merge_metrics_and_diagnostics
from evaluation.splits import SPLITS_DIR, load_split

RESULTS_PATH = Path(__file__).resolve().parent / "annotation_coverage.json"


def main():
    """
    Compute metrics on full corpus and each split, then save to annotation_coverage.json.
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: python scripts/export_sqlite.py")
        return

    conn = sqlite3.connect(DB_PATH)

    results = {"full_corpus": merge_metrics_and_diagnostics(conn)}

    for split_name in ["train", "dev", "test", "held_out_poets", "held_out_poems"]:
        split_path = SPLITS_DIR / f"{split_name}.json"
        if split_path.exists():
            poem_ids = load_split(split_name)
            results[split_name] = merge_metrics_and_diagnostics(conn, poem_ids)
        else:
            results[split_name] = {"error": f"split {split_name} not found"}

    conn.close()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Annotation coverage saved to {RESULTS_PATH}")
    print(json.dumps(results["full_corpus"], indent=2))


if __name__ == "__main__":
    main()
