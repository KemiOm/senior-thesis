"""
Corpus and model-output metrics: meter, rhyme, CMU phonology coverage,
end-stopping, and caesura.

Call `compute_metrics` to see how complete annotations are on a set of poems,
or after model outputs are written back with the same pipeline into the database.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

# Project root 
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "output" / "corpus.db"


def compute_metrics(conn: sqlite3.Connection, poem_ids: Optional[List[str]] = None) -> dict:
    """
    Compute evaluation metrics from corpus.db.

    Args:
        conn: SQLite connection to corpus.db.
        poem_ids: Optional list of poem IDs. If not given, use the full corpus.

    Returns:
        A dictionary with the total number of lines and counts/percentages for:
        - meter_coverage: lines with a known meter type
        - stress_coverage: lines with a stress pattern
        - rhyme_coverage: lines with a rhyme group
        - cmu_coverage: lines where all words were found in CMU
        - end_stopped: lines ending in sentence punctuation
        - caesura: lines with punctuation in the middle of the line
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

    # Stress coverage: lines with a non-empty stress pattern (from Poesy/Prosodic or fallback).
    if poem_ids:
        stress_ok = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders}) "
            "AND stress IS NOT NULL AND TRIM(stress) != ''",
            params,
        ).fetchone()[0]
    else:
        stress_ok = conn.execute("""
            SELECT COUNT(*) FROM lines
            WHERE stress IS NOT NULL AND TRIM(stress) != ''
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

    # End-stopped: lines with end_stopped = 1 (sentence-ending punctuation at line end).
    if poem_ids:
        end_stopped = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders}) AND end_stopped = 1",
            params,
        ).fetchone()[0]
    else:
        end_stopped = conn.execute("SELECT COUNT(*) FROM lines WHERE end_stopped = 1").fetchone()[0]

    # Caesura: lines with mid-line punctuation 
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
        "stress_coverage": {"n_with_stress": stress_ok, "pct": 100 * stress_ok / n_lines},
        "rhyme_coverage": {"n_with_rhyme": rhymed, "pct": 100 * rhymed / n_lines},
        "cmu_coverage": {"n_full_cmu": cmu_ok, "pct": 100 * cmu_ok / n_lines},
        "end_stopped": {"n": end_stopped, "pct": 100 * end_stopped / n_lines},
        "caesura": {"n": with_caesura, "pct": 100 * with_caesura / n_lines},
    }


def main():
    """
    Compute metrics on the full corpus and print as JSON.
    Requires output/corpus.db; run python scripts/export_sqlite.py first.
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: python scripts/export_sqlite.py")
        return

    conn = sqlite3.connect(DB_PATH)
    metrics = compute_metrics(conn)
    conn.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
