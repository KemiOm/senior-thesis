"""
Normalize one extracted poem: quotes, dashes, apostrophes, whitespace.
Reads from output/poems/<id>.json, returns poem with raw + normalized per line.
"""
import json
import re
from pathlib import Path

# Project root (parent of sample/). Paths resolve relative to project root so scripts
# work from sample/ or from the project root.
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output/poems"


def normalize_line(raw: str) -> str:
    """
    Normalize a single line of verse.
    Handles: space before punctuation, curly quotes/apostrophes, whitespace.
    Preserves poetic contractions (e.g. 'T is, slumb'ring) — only standardizes the apostrophe char.
    """
    s = raw.strip()
    s = re.sub(r"\s+", " ", s)  # collapse multiple spaces
    s = s.replace(""", '"').replace(""", '"')  # curly double quotes → straight
    s = s.replace("'", "'").replace("'", "'").replace("ʼ", "'")  # curly/smart apostrophe → straight
    s = re.sub(r"[\u2013\u2014]", "-", s)  # en/em dash → hyphen
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)  # remove space before punctuation
    return s


def normalize_poem(poem: dict) -> dict:
    """
    Take extracted poem dict, add normalized lines alongside raw.
    Stanzas become lists of {raw, normalized} per line.
    """
    out = {"id": poem["id"], "author": poem["author"], "title": poem["title"], "stanzas": []}
    for stanza in poem["stanzas"]:
        lines = [{"raw": raw, "normalized": normalize_line(raw)} for raw in stanza]
        out["stanzas"].append(lines)
    return out


SAMPLE_QUATRAIN = "bah18-w0160"
SAMPLE_COUPLET = "o5156-w1237"


def main():
    for poem_id, label in [(SAMPLE_QUATRAIN, "quatrain"), (SAMPLE_COUPLET, "couplet")]:
        json_path = OUTPUT_DIR / f"{poem_id}.json"
        if not json_path.exists():
            print(f"Skip {label} ({poem_id}): not found")
            continue
        with open(json_path, encoding="utf-8") as f:
            poem = json.load(f)
        result = normalize_poem(poem)
        print("=" * 60)
        print(f"[{label}] ID: {result['id']} | {result['title'][:50]}...")
        print("=" * 60)
        for i, line in enumerate(result["stanzas"][0][:4]):
            print(f"  raw:        {line['raw']}")
            print(f"  normalized: {line['normalized']}")
            print()
        print(f"Total: {len(result['stanzas'])} stanzas, {sum(len(s) for s in result['stanzas'])} lines")


if __name__ == "__main__":
    main()
