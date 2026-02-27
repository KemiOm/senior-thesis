"""
Batch-normalize extracted poems.
Reads from output/poems/, applies normalize_poem, writes to output/poems_normalized/.
"""

import sys
import json
from pathlib import Path

# Project root (parent of batch/). Ensures the sample package is importable regardless
# of how the script is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample.normalize_sample import normalize_poem

INPUT_DIR = ROOT / "output/poems"
OUTPUT_DIR = ROOT / "output/poems_normalized"


def main():
    json_files = list(INPUT_DIR.glob("*.json"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(json_files)} poem JSONs")
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
