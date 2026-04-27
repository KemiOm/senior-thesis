"""
Corpus metrics: meter, rhyme, CMU coverage, end-stopping, caesura from corpus.db.

Used by evaluation.corpus.coverage for annotation_coverage.json.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "output" / "corpus.db"


def compute_metrics(conn: sqlite3.Connection, poem_ids: Optional[List[str]] = None) -> dict:
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

    if poem_ids:
        end_stopped = conn.execute(
            f"SELECT COUNT(*) FROM lines WHERE poem_id IN ({placeholders}) AND end_stopped = 1",
            params,
        ).fetchone()[0]
    else:
        end_stopped = conn.execute("SELECT COUNT(*) FROM lines WHERE end_stopped = 1").fetchone()[0]

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


def compute_corpus_diagnostics(conn: sqlite3.Connection, poem_ids: Optional[List[str]] = None) -> dict:
    if poem_ids:
        placeholders = ",".join("?" * len(poem_ids))
        params: tuple = tuple(poem_ids)
        line_scope = f"poem_id IN ({placeholders})"
        stanza_scope = f"poem_id IN ({placeholders})"
    else:
        params = ()
        line_scope = "1=1"
        stanza_scope = "1=1"

    n_poems = conn.execute(
        f"SELECT COUNT(DISTINCT poem_id) FROM lines WHERE {line_scope}", params
    ).fetchone()[0]
    n_stanzas = conn.execute(
        f"SELECT COUNT(*) FROM stanzas WHERE {stanza_scope}", params
    ).fetchone()[0]
    n_lines = conn.execute(f"SELECT COUNT(*) FROM lines WHERE {line_scope}", params).fetchone()[0]

    meter_rows = conn.execute(
        f"""
        SELECT meter_type, COUNT(*) AS n
        FROM lines
        WHERE {line_scope}
        GROUP BY meter_type
        ORDER BY n DESC
        LIMIT 15
        """,
        params,
    ).fetchall()
    meter_type_distribution = []
    for meter, n in meter_rows:
        label = meter if meter else "(null/empty)"
        pct = 100 * n / n_lines if n_lines else 0.0
        meter_type_distribution.append({"meter_type": label, "n": n, "pct": round(pct, 2)})

    lines_with_not_found = conn.execute(
        f"SELECT COUNT(*) FROM lines WHERE {line_scope} AND phonology LIKE '%not_found%'", params
    ).fetchone()[0]

    degraded = conn.execute(
        f"""
        SELECT poem_id FROM lines
        WHERE {line_scope}
        GROUP BY poem_id
        HAVING COUNT(*) = SUM(
            CASE WHEN meter_type IS NULL OR meter_type = 'unknown' OR meter_type = '' THEN 1 ELSE 0 END
        )
        """,
        params,
    ).fetchall()
    n_poems_all_unknown_meter = len(degraded)

    enjambed = conn.execute(
        f"SELECT COUNT(*) FROM lines WHERE {line_scope} AND enjambment = 1", params
    ).fetchone()[0]

    stanza_rows = conn.execute(
        f"""
        SELECT stanza_type, COUNT(*) AS n
        FROM stanzas
        WHERE stanza_type IS NOT NULL AND {stanza_scope}
        GROUP BY stanza_type
        ORDER BY n DESC
        LIMIT 10
        """,
        params,
    ).fetchall()
    stanza_type_distribution = [{"stanza_type": st or "(null)", "n": n} for st, n in stanza_rows]

    return {
        "n_poems": n_poems,
        "n_stanzas": n_stanzas,
        "meter_type_distribution": meter_type_distribution,
        "lines_with_phonology_not_found": lines_with_not_found,
        "n_poems_all_unknown_meter": n_poems_all_unknown_meter,
        "enjambment_lines": enjambed,
        "stanza_type_distribution": stanza_type_distribution,
    }


def merge_metrics_and_diagnostics(
    conn: sqlite3.Connection, poem_ids: Optional[List[str]] = None
) -> dict:
    out = compute_metrics(conn, poem_ids)
    diag = compute_corpus_diagnostics(conn, poem_ids)
    out.update(diag)
    return out


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: python scripts/export_sqlite.py")
        return

    conn = sqlite3.connect(DB_PATH)
    metrics = merge_metrics_and_diagnostics(conn)
    conn.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
