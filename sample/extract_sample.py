"""
Extract 18th-c. poems from ECPA TEI XML into a structured corpus.
Preserves line/stanza structure. Structure verification only—no normalization yet.
"""

from pathlib import Path
from typing import Optional
from lxml import etree

# TEI P5 uses this namespace for core elements: body, lg, l, title, author, etc.
# All TEI elements are in this namespace; XPath and find() require the full namespace URI.
TEI_NS = "http://www.tei-c.org/ns/1.0"

# Project root (parent of sample/). Paths resolve relative to project root so scripts
# work when run from sample/ or from the project root.
ROOT = Path(__file__).resolve().parent.parent


def get_line_text(line_elem) -> str:
    """
    Collect all text from a line element (<l>) in document order.
    itertext() yields text from <w>, <pc>, <c> etc.; the XML often has newlines
    and spaces between each tag—whitespace is collapsed to a single space.
    """
    raw = "".join(line_elem.itertext())
    # Collapse runs of whitespace (spaces, newlines, tabs) to a single space
    return " ".join(raw.split()).strip()


def extract_poem(xml_path: Path) -> Optional[dict]:
    """
    Parse one ECPA TEI XML file and return a dict with metadata and stanzas.
    Returns None if parsing fails or body is missing.
    """
    try:
        # etree.parse() reads the XML file and builds an element tree
        tree = etree.parse(str(xml_path))
        # root is the top-level <TEI> element
        root = tree.getroot()
    except Exception as e:
        # Catch any parse error (malformed XML, encoding issues, etc.)
        print(f"Parse error {xml_path}: {e}")
        return None

    # find() returns the first match; XPath ".//" means "any descendant"
    # TEI uses xmlns; the full namespace URI is required in the tag.
    title_el = root.find(".//{http://www.tei-c.org/ns/1.0}title")
    author_el = root.find(".//{http://www.tei-c.org/ns/1.0}author")

    # Guard against None (element not found) or empty text
    title = title_el.text.strip() if title_el is not None and title_el.text else ""
    author = author_el.text.strip() if author_el is not None and author_el.text else ""

    # The poem body contains <lg> (line groups / stanzas) and <l> (lines)
    # body is under teiHeader/text/body in TEI structure
    body = root.find(f".//{{{TEI_NS}}}body")
    if body is None:
        print("No body found")
        return None
    stanzas = []

    # findall() returns all <lg> descendants of body (stanzas)
    for lg in body.findall(f".//{{{TEI_NS}}}lg"):
        lines = []
        # Each <lg> contains <l> elements (individual verse lines)
        for l in lg.findall(f"{{{TEI_NS}}}l"):
            raw = get_line_text(l)
            # Skip blank lines (some <l> may be empty or only whitespace)
            if raw:
                lines.append(raw)
        # Only add non-empty stanzas
        if lines:
            stanzas.append(lines)

    # Return structured dict: id from filename, metadata, and nested stanza/line list
    return {
        "id": xml_path.stem,
        "author": author,
        "title": title,
        "stanzas": stanzas,
    }


SAMPLE_QUATRAIN = "bah18-w0160"   # A VOW TO FORTUNE. (4-line stanzas)
SAMPLE_COUPLET = "o5156-w1237"   # BOOK XII. Ep. 23. (2-line stanzas)


def main():
    # ECPA source XML lives under project root (ECPA/web/works/<poem_id>/<poem_id>.xml).
    base = ROOT / "ECPA/web/works"

    for poem_id, label in [(SAMPLE_QUATRAIN, "quatrain"), (SAMPLE_COUPLET, "couplet")]:
        test_path = base / poem_id / f"{poem_id}.xml"
        if not test_path.exists():
            print(f"Skip {label} ({poem_id}): not found")
            continue

        poem = extract_poem(test_path)
        if poem is None:
            continue

    # Print extracted structure to verify stanza/line boundaries.
    print("=" * 60)
    print(f"ID:     {poem['id']}")
    print(f"Author: {poem['author']}")
    print(f"Title:  {poem['title']}")
    print("=" * 60)
    for i, stanza in enumerate(poem["stanzas"]):
        print(f"\n--- Stanza {i + 1} ---")
        for j, line in enumerate(stanza):
            print(f"  {j + 1}: {line}")
    print("\n" + "=" * 60)
    print(f"Total stanzas: {len(poem['stanzas'])}")
    print(f"Total lines:   {sum(len(s) for s in poem['stanzas'])}")


if __name__ == "__main__":
    main()
