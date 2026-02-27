"""
Batch-extract 18th-c. poems from ECPA TEI XML.
Finds all poem XMLs in ECPA/web/works/, extracts each, and saves to output/poems/.
Uses the same extraction logic as sample/extract_sample.py.
"""

import sys
from pathlib import Path

# Project root (parent of batch/). Ensures the sample package is importable regardless
# of how the script is invoked (e.g. python batch/extract_batch.py or python -m batch.extract_batch).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample.extract_sample import extract_poem

ECPA_ROOT = ROOT / "ECPA/web/works"
OUTPUT_DIR = ROOT / "output/poems"


def main():
    # Find all XML files in subdirs 
    # Exclude tei_all.rnc schema file and any other non-poem XMLs
    xml_files = list(ECPA_ROOT.glob("*/*.xml"))
    xml_files = [p for p in xml_files if p.name != "tei_all.rnc"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(xml_files)} poem XMLs")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")
    print("-" * 50)

    ok = 0
    fail = 0
    for i, xml_path in enumerate(xml_files):
        poem = extract_poem(xml_path)
        if poem is None:
            fail += 1
            continue
        out_path = OUTPUT_DIR / f"{poem['id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(poem, f, indent=2, ensure_ascii=False)
        ok += 1
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(xml_files)}")

    print("-" * 50)
    print(f"Done: {ok} extracted, {fail} failed")


if __name__ == "__main__":
    main()
