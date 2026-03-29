"""
Batch-normalize extracted poems.
Reads from output/poems/, applies normalize_poem, writes to output/poems_normalized/.
By default skips poems that already have normalized output (use --force to reprocess).
"""

import argparse
import json
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample.normalize_sample import normalize_poem

INPUT_DIR = ROOT / "output/poems"
OUTPUT_DIR = ROOT / "output/poems_normalized"


def main():
    parser = argparse.ArgumentParser(description="Batch-normalize extracted poems")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Reprocess all poems even if normalized output already exists",
    )
    args = parser.parse_args()
    skip_existing = not args.force

    all_files = list(INPUT_DIR.glob("*.json"))
    if skip_existing:
        json_files = [p for p in all_files if not (OUTPUT_DIR / p.name).exists()]
        if len(json_files) < len(all_files):
            print(f"Skipping {len(all_files) - len(json_files)} already normalized; processing {len(json_files)}")
    else:
        json_files = all_files

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(json_files)} poem JSONs to process")
    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print("-" * 50)

    ok = 0
    fail = 0
    for i, json_path in enumerate(json_files):
        try:
            with open(json_path, encoding="utf-8") as f:
                poem = json.load(f)
            result = normalize_poem(poem)
            out_path = OUTPUT_DIR / f"{result['id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            ok += 1
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(json_files)}")
        except Exception as e:
            fail += 1
            print(f"  FAIL {json_path.name}: {e}")

    print("-" * 50)
    print(f"Done: {ok} normalized, {fail} failed")


if __name__ == "__main__":
    main()
