"""
Export annotated poems to SQLite for querying.
Reads output/poems_annotated/*.json, creates corpus.db with poems, stanzas, lines tables.
"""

import json
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "output/poems_annotated"
DB_PATH = SCRIPT_DIR / "output" / "corpus.db"


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DROP TABLE IF EXISTS lines")
    conn.execute("DROP TABLE IF EXISTS stanzas")
    conn.execute("DROP TABLE IF EXISTS poems")
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS poems (
            id TEXT PRIMARY KEY,
            author TEXT,
            title TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stanzas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            poem_id TEXT NOT NULL REFERENCES poems(id),
            stanza_index INTEGER NOT NULL,
            stanza_type TEXT,
            rhyme_scheme TEXT,
            rhyme_pairs TEXT,
            UNIQUE(poem_id, stanza_index)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            poem_id TEXT NOT NULL REFERENCES poems(id),
            stanza_index INTEGER NOT NULL,
            line_index INTEGER NOT NULL,
            raw TEXT,
            normalized TEXT,
            rhyme_word TEXT,
            rhyme_group TEXT,
            meter_type TEXT,
            meter TEXT,
            stress TEXT,
            end_stopped INTEGER,
            caesura INTEGER,
            enjambment INTEGER,
            phonology TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_lines_poem ON lines(poem_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_lines_rhyme ON lines(rhyme_group)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_lines_meter_type ON lines(meter_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_stanzas_poem ON stanzas(poem_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_stanzas_stanza_type ON stanzas(stanza_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_poems_author ON poems(author)")

    conn.execute("DROP VIEW IF EXISTS v_lines_with_poem")
    conn.execute("""
        CREATE VIEW v_lines_with_poem AS
        SELECT l.*, p.author, p.title
        FROM lines l JOIN poems p ON l.poem_id = p.id
    """)
    conn.execute("DROP VIEW IF EXISTS v_poems_by_meter")
    conn.execute("""
        CREATE VIEW v_poems_by_meter AS
        SELECT poem_id, meter_type, COUNT(*) as line_count
        FROM lines
        WHERE meter_type IS NOT NULL AND meter_type != ''
        GROUP BY poem_id, meter_type
    """)
    conn.execute("DROP VIEW IF EXISTS v_stanzas_with_poem")
    conn.execute("""
        CREATE VIEW v_stanzas_with_poem AS
        SELECT s.*, p.author, p.title
        FROM stanzas s JOIN poems p ON s.poem_id = p.id
    """)
    conn.execute("DROP VIEW IF EXISTS v_rhyme_summary")
    conn.execute("""
        CREATE VIEW v_rhyme_summary AS
        SELECT poem_id, rhyme_group, COUNT(*) as line_count
        FROM lines
        WHERE rhyme_group IS NOT NULL AND rhyme_group != '' AND rhyme_group != '-'
        GROUP BY poem_id, rhyme_group
    """)

    files = list(INPUT_DIR.glob("*.json"))
    print(f"Exporting {len(files)} annotated poems to {DB_PATH}")

    for i, p in enumerate(files):
        with open(p, encoding="utf-8") as f:
            poem = json.load(f)
        pid = poem["id"]
        conn.execute(
            "INSERT OR REPLACE INTO poems (id, author, title) VALUES (?, ?, ?)",
            (pid, poem.get("author", ""), poem.get("title", "")),
        )
        for si, stanza in enumerate(poem.get("stanzas", [])):
            # New format: {stanza_index, stanza_type, rhyme_scheme, rhyme_pairs, lines}
            # Old format: [line, line, ...]
            if isinstance(stanza, dict):
                st_type = stanza.get("stanza_type")
                rhyme_scheme = stanza.get("rhyme_scheme")
                rhyme_pairs = json.dumps(stanza.get("rhyme_pairs", [])) if stanza.get("rhyme_pairs") else None
                line_list = stanza.get("lines", [])
            else:
                st_type = rhyme_scheme = rhyme_pairs = None
                line_list = stanza if isinstance(stanza, list) else []

            conn.execute(
                """INSERT OR REPLACE INTO stanzas (poem_id, stanza_index, stanza_type, rhyme_scheme, rhyme_pairs)
                   VALUES (?, ?, ?, ?, ?)""",
                (pid, si, st_type, rhyme_scheme, rhyme_pairs),
            )
            for li, line in enumerate(line_list):
                line = line if isinstance(line, dict) else {}
                li = line.get("line_index", li)
                phon = json.dumps(line.get("phonology", [])) if line.get("phonology") else None
                conn.execute(
                    """INSERT INTO lines (poem_id, stanza_index, line_index, raw, normalized,
                       rhyme_word, rhyme_group, meter_type, meter, stress, end_stopped, caesura, enjambment, phonology)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pid, si, li,
                        line.get("raw"), line.get("normalized"),
                        line.get("rhyme_word"), line.get("rhyme_group"),
                        line.get("meter_type"), line.get("meter"), line.get("stress"),
                        1 if line.get("end_stopped") else 0,
                        line.get("caesura") if line.get("caesura") is not None else None,
                        1 if line.get("enjambment") else 0,
                        phon,
                    ),
                )
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(files)}")

    conn.commit()
    conn.close()
    print(f"Done: {DB_PATH}")


if __name__ == "__main__":
    main()
