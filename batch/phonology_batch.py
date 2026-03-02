"""
Batch-annotate normalized poems with phonology, meter, rhyme, end_stopped, caesura, enjambment.
Reads from output/poems_normalized/, writes to output/poems_annotated/.
Uses: pronouncing (CMU), Poesy (Prosodic). All annotations automated.

Batch optimizations: parallelism, no espeak fallback, long-line truncation, progress logging.
"""

import os
os.environ["POESY_DEBUG"] = "0"
os.environ["PHONOLOGY_BATCH"] = "1"   # disables espeak fallback
os.environ["MAX_LINE_CHARS"] = "80"    # truncate long lines; verse rarely >80 chars; prose blocks hit 30s timeout
os.environ["MAX_POEM_LINES"] = "500"  # skip Poesy for poems >500 lines (phonology-only)

import sys
import argparse
import json
from pathlib import Path

# Project root (parent of batch/). Ensures the sample package is importable in worker
# subprocesses (ProcessPoolExecutor) as well as the main process.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT_DIR = ROOT / "output/poems_normalized"
OUTPUT_DIR = ROOT / "output/poems_annotated"
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
USE_PARALLEL = os.environ.get("PHONOLOGY_PARALLEL", "1") == "1"


def _annotate_one(json_path_str: str):
    """Worker: load poem, annotate, return (ok/fail, data). Runs in subprocess. Uses
    sample.phonology_sample because each worker imports in a fresh interpreter."""
    from pathlib import Path
    from sample.phonology_sample import annotate_poem
    try:
        with open(json_path_str, encoding="utf-8") as f:
            poem = json.load(f)
        result = annotate_poem(poem)
        return ("ok", result)
    except Exception as e:
        return ("fail", (Path(json_path_str).name, str(e)))


def _poem_ids_with_valid_meter(corpus_path: Path, min_stress_len: int = 5) -> set:
    """Poem IDs that have at least one line with valid 01 stress pattern."""
    import sqlite3
    if not corpus_path.exists():
        return set()
    conn = sqlite3.connect(corpus_path)
    rows = conn.execute(
        "SELECT DISTINCT poem_id FROM lines WHERE stress IS NOT NULL AND LENGTH(TRIM(stress)) >= ? "
        "AND TRIM(stress) GLOB '[01]*' AND TRIM(stress) NOT GLOB '*[^01]*'",
        (min_stress_len,),
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def main():
    parser = argparse.ArgumentParser(description="Batch-annotate poems with phonology")
    parser.add_argument("--limit", "-n", type=int, default=0, help="Process only first N poems (0=all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip poems that already have output")
    parser.add_argument(
        "--meter-only",
        action="store_true",
        help="Only process poems that have valid 01 binary stress in corpus.db (requires prior annotation + export)",
    )
    args = parser.parse_args()

    json_files = sorted(INPUT_DIR.glob("*.json"))

    if args.meter_only:
        corpus_db = ROOT / "output" / "corpus.db"
        valid_ids = _poem_ids_with_valid_meter(corpus_db)
        if not valid_ids:
            print("No poems with valid 01 meter in corpus.db. Run export_sqlite.py first.")
            return
        json_files = [p for p in json_files if p.stem in valid_ids]
        print(f"Meter-only: {len(json_files)} poems with valid 01 stress (of {len(valid_ids)} in corpus)")

    if args.skip_existing:
        json_files = [p for p in json_files if not (OUTPUT_DIR / p.name).exists()]
        print(f"Skipping existing: {len(list(INPUT_DIR.glob('*.json')))} input, {len(json_files)} to process")
    if args.limit > 0:
        json_files = json_files[: args.limit]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(json_files)
    print(f"Found {total} normalized poem JSONs")
    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    mode = f"parallel ({MAX_WORKERS} workers)" if USE_PARALLEL else "sequential"
    limit_note = f" [limit={args.limit}]" if args.limit else ""
    skip_note = " [skip-existing]" if args.skip_existing else ""
    meter_note = " [meter-only]" if args.meter_only else ""
    max_lines = os.environ.get("MAX_POEM_LINES", "150")
    max_line_chars = os.environ.get("MAX_LINE_CHARS", "0")
    print(f"Mode: {mode} | espeak off | truncate lines >{max_line_chars} chars | skip Poesy if >{max_lines} lines{limit_note}{skip_note}{meter_note}")
    print("-" * 50)

    ok = 0
    fail = 0

    def _results():
        if not USE_PARALLEL:
            for p in json_files:
                yield _annotate_one(str(p))
            return
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(_annotate_one, str(p)) for p in json_files]
                for future in as_completed(futures):
                    yield future.result()
        except (PermissionError, OSError) as e:
            print(f"  (parallel disabled: {e}, using sequential)")
            for p in json_files:
                yield _annotate_one(str(p))

    for done, (status, data) in enumerate(_results(), 1):
        if status == "ok":
            result = data
            out_path = OUTPUT_DIR / f"{result['id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            ok += 1
        else:
            fail += 1
            name, err = data
            print(f"  FAIL [{done}/{total}] {name}: {err}")
        if done % 10 == 0:
            print(f"  ... {done}/{total} (ok={ok}, fail={fail})", flush=True)

    print("-" * 50)
    print(f"Done: {ok} annotated, {fail} failed")


if __name__ == "__main__":
    main()
