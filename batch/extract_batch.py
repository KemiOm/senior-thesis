"""
Batch-extract 18th-c. poems from ECPA TEI XML.
Finds all poem XMLs in ECPA/web/works/, extracts each, and saves to output/poems/.
Uses the same extraction logic as sample/extract_sample.py.
By default skips poems that already have output (use --force to reprocess).
"""

import argparse
import json
import sys
from pathlib import Path

# Project root .
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample.extract_sample import extract_poem

ECPA_ROOT = ROOT / "ECPA/web/works"
OUTPUT_DIR = ROOT / "output/poems"


def main():
    parser = argparse.ArgumentParser(description="Batch-extract poems from ECPA TEI XML")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Reprocess all poems even if output already exists",
    )
    args = parser.parse_args()
    skip_existing = not args.force

    # Find all XML files in subdirs; exclude tei_all.rnc
    all_xml = [p for p in ECPA_ROOT.glob("*/*.xml") if p.name != "tei_all.rnc"]
    if skip_existing:
        xml_files = [p for p in all_xml if not (OUTPUT_DIR / f"{p.stem}.json").exists()]
        if len(xml_files) < len(all_xml):
            print(f"Skipping {len(all_xml) - len(xml_files)} already extracted; processing {len(xml_files)}")
    else:
        xml_files = all_xml

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(xml_files)} poem XMLs to process")
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
