#!/usr/bin/env python3
"""
Spot-check natural_text baseline JSONs: model_output quality vs gold/context.

Reports: empty preds, duplicate of context line, duplicate of gold, very short lines,
letter ratio, and random samples. Optionally gold/pred CMU ok counts (strict vs relax_oov).

Usage (project root):
  python scripts/helpers/diagnostic/spotcheck_natural_text_outputs.py path/to/few_shot_natural_text.json
  python scripts/helpers/diagnostic/spotcheck_natural_text_outputs.py path/to/file.json --samples 8 --phon-check
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def last_context_line(inp: str) -> str:
    s = (inp or "").strip()
    if not s or s == "[start]":
        return ""
    parts = [p.strip() for p in s.split("|")]
    return parts[-1] if parts else ""


def letter_ratio(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    letters = sum(1 for c in s if c.isalpha())
    return letters / max(len(s), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--samples", type=int, default=5, help="Random non-empty outputs to print")
    ap.add_argument(
        "--phon-check",
        action="store_true",
        help="Count gold/pred line_form_signature ok (strict vs relax_oov)",
    )
    args = ap.parse_args()
    path = args.json_path
    if not path.is_file():
        raise SystemExit(f"Not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results") or []
    n = len(results)
    empty = 0
    ws_only = 0
    eq_gold = 0
    eq_ctx = 0
    short10 = 0
    low_letters = 0
    nonempty = []

    for row in results:
        gold = (row.get("gold_target") or "").strip()
        pred = (row.get("model_output") or "").strip()
        inp = row.get("input", "") or ""
        if not pred:
            empty += 1
            continue
        if not pred.strip() or not any(c.isalnum() for c in pred):
            ws_only += 1
        if gold and pred == gold:
            eq_gold += 1
        ctx = last_context_line(inp)
        if ctx and pred.strip().lower() == ctx.lower():
            eq_ctx += 1
        if len(pred) < 10:
            short10 += 1
        if letter_ratio(pred) < 0.15:
            low_letters += 1
        nonempty.append((inp, gold, pred))

    print(f"File: {path}")
    print(f"Rows: {n}")
    print(f"  empty model_output:     {empty} ({100*empty/n:.1f}%)" if n else "")
    print(f"  non-alnum / odd:        {ws_only}")
    print(f"  pred == gold (exact):   {eq_gold}")
    print(f"  pred == last ctx line:  {eq_ctx}")
    print(f"  len(pred) < 10 chars:   {short10}")
    print(f"  letter_ratio < 0.15:    {low_letters}")
    print()

    if args.phon_check:
        from evaluation.form_eval_generation import line_form_signature

        g_ok_s = p_ok_s = g_ok_r = p_ok_r = 0
        for row in results:
            gold = (row.get("gold_target") or "").strip()
            pred = (row.get("model_output") or "").strip()
            if not gold or not pred:
                continue
            if line_form_signature(gold, relax_oov=False)["ok"]:
                g_ok_s += 1
            if line_form_signature(pred, relax_oov=False)["ok"]:
                p_ok_s += 1
            if line_form_signature(gold, relax_oov=True)["ok"]:
                g_ok_r += 1
            if line_form_signature(pred, relax_oov=True)["ok"]:
                p_ok_r += 1
        print("Phonology (CMU path, first N rows in file):")
        print(f"  gold ok strict:   {g_ok_s}/{n}   relax_oov: {g_ok_r}/{n}")
        print(f"  pred ok strict:   {p_ok_s}/{n}   relax_oov: {p_ok_r}/{n}")
        print()

    rng = random.Random(42)
    k = min(args.samples, len(nonempty))
    picks = rng.sample(nonempty, k=k) if nonempty else []
    print(f"Sample of {k} non-empty model_output (context / gold / pred):")
    for i, (inp, gold, pred) in enumerate(picks, 1):
        pi = re.sub(r"\s+", " ", inp)[:120]
        pg = re.sub(r"\s+", " ", gold)[:120]
        pp = re.sub(r"\s+", " ", pred)[:120]
        print(f"  --- {i} ---")
        print(f"  ctx:  {pi!r}")
        print(f"  gold: {pg!r}")
        print(f"  pred: {pp!r}")


if __name__ == "__main__":
    main()
