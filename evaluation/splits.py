"""
Create train/dev/test splits and held-out sets (poets, poems).

- train / dev: for training and tuning
- test: held-out poems for in-domain evaluation
- held_out_poets / held_out_poems: extra generalization checks

Writes poem IDs under `evaluation/splits/`. Uses a fixed random seed so splits stay stable.
Run from project root after `python scripts/export_sqlite.py` has created `output/corpus.db`.
"""

import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List

# Project root and paths.
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "output" / "corpus.db"
SPLITS_DIR = Path(__file__).resolve().parent / "splits"

# Fixed seed so splits are reproducible across runs.
RANDOM_SEED = 42


def get_poem_ids_by_author(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """
    Build a mapping from author to list of poem IDs.

    Returns:
        Dict mapping author name (or empty string) to list of poem_ids.
    """
    rows = conn.execute(
        "SELECT id, COALESCE(author, '') FROM poems"
    ).fetchall()
    by_author: Dict[str, List[str]] = {}
    for pid, author in rows:
        by_author.setdefault(author, []).append(pid)
    return by_author


def get_poem_ids_by_stanza_type(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """
    Build a mapping from stanza type (quatrain, couplet, etc.) to poem IDs.

    Poems with multiple stanza types appear in multiple lists.
    Returns deduplicated lists per stanza type.
    """
    rows = conn.execute(
        "SELECT DISTINCT poem_id, stanza_type FROM stanzas WHERE stanza_type IS NOT NULL"
    ).fetchall()
    by_type: Dict[str, List[str]] = {}
    for pid, st_type in rows:
        by_type.setdefault(st_type, []).append(pid)
    for k in by_type:
        by_type[k] = list(dict.fromkeys(by_type[k]))
    return by_type


def create_splits(
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
    test_frac: float = 0.15,
    held_out_poets_n: int = 10,
    held_out_poems_n: int = 100,
) -> None:
    """
    Create train/dev/test and held-out splits, write to evaluation/splits/.

    Strategy:
    1. Held-out poets: Select N authors with at least 3 poems; put all their
       poems in held_out_poets. Tests generalization to unseen authors.
    2. Held-out poems: From the remaining poems (not in held_out_poets),
       randomly select held_out_poems_n poems. Tests generalization to
       unseen poems within known authors.
    3. Train/dev/test: Split the rest by train_frac, dev_frac, test_frac.

    Writes train.json, dev.json, test.json, held_out_poets.json,
    held_out_poems.json, and meta.json (counts and parameters).
    """
    if abs(train_frac + dev_frac + test_frac - 1.0) > 0.001:
        raise ValueError("train_frac + dev_frac + test_frac must equal 1.0")

    conn = sqlite3.connect(DB_PATH)
    by_author = get_poem_ids_by_author(conn)
    all_poems = conn.execute("SELECT id FROM poems").fetchall()
    all_poem_ids = [r[0] for r in all_poems]
    conn.close()

    random.seed(RANDOM_SEED)

    # Held-out poets: authors with at least 3 poems; take first N authors.
    authors_with_enough = [(a, pids) for a, pids in by_author.items() if len(pids) >= 3]
    random.shuffle(authors_with_enough)
    held_out_poet_ids: List[str] = []
    for author, pids in authors_with_enough[:held_out_poets_n]:
        held_out_poet_ids.extend(pids)
    held_out_poet_ids = list(dict.fromkeys(held_out_poet_ids))

    # Held-out poems: from poems not in held_out_poets, randomly select N.
    in_domain = [p for p in all_poem_ids if p not in held_out_poet_ids]
    random.shuffle(in_domain)
    held_out_poem_ids = in_domain[:held_out_poems_n]
    rest = in_domain[held_out_poems_n:]

    # Train / dev / test from the remaining poems.
    n = len(rest)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    n_test = n - n_train - n_dev

    train_ids = rest[:n_train]
    dev_ids = rest[n_train : n_train + n_dev]
    test_ids = rest[n_train + n_dev :]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids,
        "held_out_poets": held_out_poet_ids,
        "held_out_poems": held_out_poem_ids,
    }

    for name, ids in splits.items():
        path = SPLITS_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
        print(f"  {name}: {len(ids)} poems -> {path}")

    meta = {
        "random_seed": RANDOM_SEED,
        "train_frac": train_frac,
        "dev_frac": dev_frac,
        "test_frac": test_frac,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    with open(SPLITS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta -> {SPLITS_DIR / 'meta.json'}")


def load_split(name: str) -> List[str]:
    """
    Load poem IDs for a split.

    Args:
        name: One of train, dev, test, held_out_poets, held_out_poems.

    Returns:
        List of poem IDs.

    Raises:
        FileNotFoundError: If the split file does not exist.
    """
    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}. Run create_splits first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    """
    Create all splits and write to evaluation/splits/.
    Requires output/corpus.db.
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: python scripts/export_sqlite.py")
        return

    print("Creating evaluation splits...")
    create_splits()
    print("Done.")


if __name__ == "__main__":
    main()
