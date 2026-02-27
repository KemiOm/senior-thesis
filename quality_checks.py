"""
Quality checks for the annotated corpus.
Queries output/corpus.db and reports counts, coverage, and potential issues.
"""

import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "output" / "corpus.db"


def run_checks(conn: sqlite3.Connection) -> None:
    print("=" * 60)
    print("CORPUS QUALITY CHECKS")
    print("=" * 60)

    # --- Counts ---
    n_poems = conn.execute("SELECT COUNT(*) FROM poems").fetchone()[0]
    n_stanzas = conn.execute("SELECT COUNT(*) FROM stanzas").fetchone()[0]
    n_lines = conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0]
    print(f"\n--- Counts ---")
    print(f"  Poems:   {n_poems:,}")
    print(f"  Stanzas: {n_stanzas:,}")
    print(f"  Lines:   {n_lines:,}")

    # --- Meter type coverage ---
    print(f"\n--- Meter type ---")
    for row in conn.execute("""
        SELECT meter_type, COUNT(*) as n
        FROM lines
        GROUP BY meter_type
        ORDER BY n DESC
        LIMIT 15
    """):
        meter, n = row
        label = meter if meter else "(null/empty)"
        pct = 100 * n / n_lines if n_lines else 0
        print(f"  {label}: {n:,} ({pct:.1f}%)")

    # --- Rhyme coverage ---
    print(f"\n--- Rhyme group ---")
    rhymed = conn.execute("""
        SELECT COUNT(*) FROM lines
        WHERE rhyme_group IS NOT NULL AND rhyme_group != '' AND rhyme_group != '-'
    """).fetchone()[0]
    non_rhymed = conn.execute("""
        SELECT COUNT(*) FROM lines
        WHERE rhyme_group IS NULL OR rhyme_group = '' OR rhyme_group = '-'
    """).fetchone()[0]
    pct_rhymed = 100 * rhymed / n_lines if n_lines else 0
    print(f"  With rhyme group (a,b,c...): {rhymed:,} ({pct_rhymed:.1f}%)")
    print(f"  Non-rhyming or missing:       {non_rhymed:,}")

    # --- Phonology: not_found ---
    print(f"\n--- Phonology (words not in CMU) ---")
    # Phonology is stored as JSON; coverage is approximated by counting lines with "not_found".
    # Simple heuristic: sample lines and report, or use a more precise query
    with_not_found = conn.execute("""
        SELECT COUNT(*) FROM lines
        WHERE phonology LIKE '%not_found%'
    """).fetchone()[0]
    print(f"  Lines with at least one word not in CMU: {with_not_found:,}")

    # --- Poems with no Poesy (all meter_type unknown) ---
    print(f"\n--- Poems with degraded annotations ---")
    poems_all_unknown = conn.execute("""
        SELECT poem_id FROM lines
        GROUP BY poem_id
        HAVING COUNT(*) = SUM(CASE WHEN meter_type IS NULL OR meter_type = 'unknown' OR meter_type = '' THEN 1 ELSE 0 END)
    """).fetchall()
    print(f"  Poems where all lines have meter_type unknown/null: {len(poems_all_unknown):,}")

    # --- End-stopped / enjambment ---
    print(f"\n--- End-stopped vs enjambment ---")
    end_stopped = conn.execute("SELECT COUNT(*) FROM lines WHERE end_stopped = 1").fetchone()[0]
    enjambed = conn.execute("SELECT COUNT(*) FROM lines WHERE enjambment = 1").fetchone()[0]
    print(f"  End-stopped: {end_stopped:,}")
    print(f"  Enjambed:    {enjambed:,}")

    # --- Caesura ---
    with_caesura = conn.execute("SELECT COUNT(*) FROM lines WHERE caesura IS NOT NULL").fetchone()[0]
    print(f"\n--- Caesura ---")
    print(f"  Lines with caesura: {with_caesura:,} ({100*with_caesura/n_lines:.1f}%)" if n_lines else "  N/A")

    # --- Stanza types ---
    print(f"\n--- Stanza types ---")
    for row in conn.execute("""
        SELECT stanza_type, COUNT(*) as n
        FROM stanzas
        WHERE stanza_type IS NOT NULL
        GROUP BY stanza_type
        ORDER BY n DESC
        LIMIT 10
    """):
        st_type, n = row
        print(f"  {st_type}: {n:,}")

    print("\n" + "=" * 60)


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run export_sqlite.py first.")
        return
    conn = sqlite3.connect(DB_PATH)
    run_checks(conn)
    conn.close()


if __name__ == "__main__":
    main()
