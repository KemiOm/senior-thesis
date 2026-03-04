"""
Evaluation metrics for the poetry corpus and model outputs.

Computes constraint-adherence metrics: meter coverage, rhyme coverage, phonology (CMU)
coverage, lineation (end-stopped), and caesura. Used for:
- Baseline: corpus ground truth (how well the annotation pipeline covers the data)
- Model evaluation: after running generated lines through the same pipeline, compare
  model output annotations to targets.

Metrics are computed from corpus.db. When evaluating model outputs, the model's
generated lines must first be annotated by the phonology pipeline (Poesy/Prosodic,
CMU dict) and loaded into a comparable structure; this module queries the database.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

# Project root (parent of evaluation/). All paths resolve relative to project root.
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "output" / "corpus.db"


def compute_metrics(conn: sqlite3.Connection, poem_ids: Optional[List[str]] = None) -> dict:
    """
    Compute evaluation metrics from corpus.db.

    Args:
        conn: SQLite connection to corpus.db.
        poem_ids: Optional list of poem IDs to restrict the computation. If None,
            metrics are computed over the full corpus.

    Returns:
        Dict with n_lines and per-metric counts/percentages:
        - meter_coverage: lines with non-unknown meter_type (from Poesy/Prosodic).
        - rhyme_coverage: lines with a rhyme_group (a, b, c, etc.).
        - cmu_coverage: lines where phonology has no "not_found" (all words in CMU).
        - end_stopped: lines ending with sentence-ending punctuation.
        - caesura: lines with mid-line punctuation (comma, semicolon, etc.).
    """
    # Build parameterized query for optional poem_id filter.
    if poem_ids:
        placeholders = ",".join("?" * len(poem_ids))
        params: tuple = tuple(poem_ids)
        n_lines = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders})", params
        ).fetchone()[0]
    else:
        params = ()
        n_lines = conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0]

    if n_lines == 0:
        return {"n_lines": 0, "error": "no lines in scope"}

    # Meter coverage: lines where meter_type is populated and not "unknown".
    # meter_type comes from Poesy/Prosodic or fallback stress-from-phonology logic.
    if poem_ids:
        meter_ok = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders}) "
            "AND (meter_type IS NOT NULL AND meter_type != '' AND meter_type != 'unknown')",
            params,
        ).fetchone()[0]
    else:
        meter_ok = conn.execute("""
            SELECT COUNT(*) FROM lines
            WHERE (meter_type IS NOT NULL AND meter_type != '' AND meter_type != 'unknown')
        """).fetchone()[0]

    # Rhyme coverage: lines with a rhyme group label (a, b, c, etc.).
    # rhyme_group comes from Poesy's rhyme_net; "-" and empty mean no rhyme.
    if poem_ids:
        rhymed = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders}) "
            "AND rhyme_group IS NOT NULL AND rhyme_group != '' AND rhyme_group != '-'",
            params,
        ).fetchone()[0]
    else:
        rhymed = conn.execute("""
            SELECT COUNT(*) FROM lines
            WHERE rhyme_group IS NOT NULL AND rhyme_group != '' AND rhyme_group != '-'
        """).fetchone()[0]

    # CMU coverage: lines where no word in phonology has source "not_found".
    # phonology is stored as JSON; "not_found" indicates the word was not in CMU.
    # Full CMU coverage means we can reliably compute stress and rhyme keys.
    if poem_ids:
        cmu_ok = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders}) "
            "AND (phonology IS NULL OR phonology NOT LIKE '%not_found%')",
            params,
        ).fetchone()[0]
    else:
        cmu_ok = conn.execute("""
            SELECT COUNT(*) FROM lines
            WHERE (phonology IS NULL OR phonology NOT LIKE '%not_found%')
        """).fetchone()[0]

    # End-stopped: lines with end_stopped = 1 (line ends with sentence-ending punctuation).
    # Derived from punctuation check in the annotation pipeline.
    if poem_ids:
        end_stopped = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders}) AND end_stopped = 1",
            params,
        ).fetchone()[0]
    else:
        end_stopped = conn.execute("SELECT COUNT(*) FROM lines WHERE end_stopped = 1").fetchone()[0]

    # Caesura: lines with mid-line punctuation (comma, semicolon, colon, em-dash).
    # caesura stores the word index of the first strong pause; NULL means none.
    if poem_ids:
        with_caesura = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders}) AND caesura IS NOT NULL",
            params,
        ).fetchone()[0]
    else:
        with_caesura = conn.execute(
            "SELECT COUNT(*) FROM lines WHERE caesura IS NOT NULL"
        ).fetchone()[0]

    return {
        "n_lines": n_lines,
        "meter_coverage": {"n_with_meter": meter_ok, "pct": 100 * meter_ok / n_lines},
        "rhyme_coverage": {"n_with_rhyme": rhymed, "pct": 100 * rhymed / n_lines},
        "cmu_coverage": {"n_full_cmu": cmu_ok, "pct": 100 * cmu_ok / n_lines},
        "end_stopped": {"n": end_stopped, "pct": 100 * end_stopped / n_lines},
        "caesura": {"n": with_caesura, "pct": 100 * with_caesura / n_lines},
    }


def main():
    """
    Compute metrics on the full corpus and print as JSON.
    Requires output/corpus.db (run export_sqlite.py first).
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run export_sqlite.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    metrics = compute_metrics(conn)
    conn.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
