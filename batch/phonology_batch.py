"""
Batch-annotate normalized poems with phonology, meter, rhyme, end_stopped, caesura, enjambment.
Reads from output/poems_normalized/, writes to output/poems_annotated/.
Uses: pronouncing (CMU), Poesy (Prosodic). All annotations automated.

Batch optimizations: parallelism, no espeak fallback, long-line truncation, progress logging.
"""

import os
os.environ["POESY_DEBUG"] = "0"
os.environ["PHONOLOGY_BATCH"] = "1"   # disables espeak fallback
os.environ["MAX_LINE_CHARS"] = "150"   # truncate lines before Poesy to avoid 30s timeouts
os.environ["MAX_POEM_LINES"] = "150"  # skip Poesy for poems >150 lines (phonology-only)

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


def main():
    parser = argparse.ArgumentParser(description="Batch-annotate poems with phonology")
    parser.add_argument("--limit", "-n", type=int, default=0, help="Process only first N poems (0=all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip poems that already have output")
    args = parser.parse_args()

    json_files = sorted(INPUT_DIR.glob("*.json"))
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
    print(f"Mode: {mode} | espeak off | lines≤150 chars | skip Poesy if >150 lines{limit_note}{skip_note}")
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
