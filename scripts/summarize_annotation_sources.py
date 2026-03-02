#!/usr/bin/env python3
"""
Aggregate annotation_sources across annotated poems.
Reports how much meter, stress, and rhyme_group come from Poesy/Prosodic vs fallback/empty.

Phonology (ARPAbet) is always from CMU+espeak — Poesy does not provide phonology.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="output/poems_annotated",
                    help="Directory of annotated JSON files")
    args = ap.parse_args()
    root = Path(args.dir)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    totals = {
        "stress_poesy": 0, "stress_empty": 0,
        "meter_poesy": 0, "meter_phonology": 0,
        "rhyme_poesy": 0, "rhyme_empty": 0,
        "meter_type_poesy": 0, "meter_type_phonology": 0,
    }
    poems_with_sources = 0
    poems_total = 0

    for path in sorted(root.glob("*.json")):
        poems_total += 1
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skip {path.name}: {e}", file=__import__("sys").stderr)
            continue
        src = data.get("annotation_sources")
        if not src:
            continue
        poems_with_sources += 1
        for k in totals:
            totals[k] += src.get(k, 0)

    if poems_with_sources == 0:
        print("No poems have annotation_sources. Re-run batch with ANNOTATION_SOURCES=1 (default).")
        return 0

    total_lines = totals["stress_poesy"] + totals["stress_empty"]

    print("Annotation source breakdown (Poesy/Prosodic vs fallback/empty)")
    print("=" * 60)
    print(f"Poems with annotation_sources: {poems_with_sources} / {poems_total}")
    print(f"Total lines: {total_lines}")
    print()
    # Stress: Poesy vs empty (no fallback)
    s_poesy, s_empty = totals["stress_poesy"], totals["stress_empty"]
    pct_p = 100 * s_poesy / total_lines if total_lines else 0
    pct_e = 100 * s_empty / total_lines if total_lines else 0
    print("STRESS (metrical, Poesy only — no fallback)")
    print(f"  Poesy:   {s_poesy:>8}  ({pct_p:.1f}%)")
    print(f"  Empty:   {s_empty:>8}  ({pct_e:.1f}%)")
    print()
    # Meter: Poesy vs phonology-derived
    m_poesy, m_phon = totals["meter_poesy"], totals["meter_phonology"]
    m_total = m_poesy + m_phon
    pct_mp = 100 * m_poesy / m_total if m_total else 0
    pct_mph = 100 * m_phon / m_total if m_total else 0
    print("METER (Poesy metrical vs phonology/CMU lexical)")
    print(f"  Poesy:      {m_poesy:>8}  ({pct_mp:.1f}%)")
    print(f"  Phonology:  {m_phon:>8}  ({pct_mph:.1f}%)")
    print()
    # Rhyme_group: Poesy vs empty
    r_poesy, r_empty = totals["rhyme_poesy"], totals["rhyme_empty"]
    r_total = r_poesy + r_empty
    pct_rp = 100 * r_poesy / r_total if r_total else 0
    pct_re = 100 * r_empty / r_total if r_total else 0
    print("RHYME_GROUP (Poesy only)")
    print(f"  Poesy:   {r_poesy:>8}  ({pct_rp:.1f}%)")
    print(f"  Empty:   {r_empty:>8}  ({pct_re:.1f}%)")
    print()
    # Meter_type: Poesy (poem-level) vs phonology-derived per line
    mt_poesy = totals.get("meter_type_poesy", 0)
    mt_phon = totals.get("meter_type_phonology", 0)
    mt_total = mt_poesy + mt_phon
    pct_mtp = 100 * mt_poesy / mt_total if mt_total else 0
    pct_mtph = 100 * mt_phon / mt_total if mt_total else 0
    print("METER_TYPE (Poesy poem-level vs phonology/CMU per line)")
    print(f"  Poesy:      {mt_poesy:>8}  ({pct_mtp:.1f}%)")
    print(f"  Phonology:  {mt_phon:>8}  ({pct_mtph:.1f}%)")
    print()
    print("NOTE: Phonology (ARPAbet phones) is always CMU+espeak — Poesy does not provide it.")
    return 0


if __name__ == "__main__":
    exit(main())
