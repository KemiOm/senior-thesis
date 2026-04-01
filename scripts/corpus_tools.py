#!/usr/bin/env python3
"""Small CLI for CSV filters, data checks, natural-text form metrics, and extra diagnostics.

  python scripts/corpus_tools.py filter-csv --n-scored 500 -o evaluation/results/out.csv
  python scripts/corpus_tools.py filter-csv --n-scored 500 --require-strict-eval -o evaluation/results/comparison_strict.csv
  python scripts/corpus_tools.py verify-data [--check-poem-ids]
  python scripts/corpus_tools.py nt-form path/to/natural_text.json [--n 500] [--relax-oov]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "output" / "corpus.db"
OUTPUT_DIR = ROOT / "docs" / "figures"
OUTPUT_PDF = OUTPUT_DIR / "meter_distribution.pdf"
MAX_METER_BARS = 12


def _strict_eval_ok(row: dict) -> tuple[bool, str]:
    rel = (row.get("json_path") or "").strip()
    if not rel:
        return False, "no json_path"
    path = ROOT / rel
    if not path.is_file():
        return False, f"missing file {rel}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"bad json {rel}: {e}"
    se = data.get("strict_eval")
    if se is True:
        return True, ""
    if se is False:
        return False, "strict_eval=false"
    return False, "strict_eval missing (legacy)"


def cmd_filter_csv(args: argparse.Namespace) -> int:
    if (
        args.n_scored is None
        and not args.require_strict_eval
        and args.prompt_type is None
        and args.split is None
    ):
        print(
            "Specify at least one filter: --n-scored, --require-strict-eval, --prompt-type, or --split",
            file=sys.stderr,
        )
        return 2
    if not args.input.is_file():
        print(f"Missing {args.input}", file=sys.stderr)
        return 1

    kept: list[dict] = []
    dropped: list[tuple[dict, str]] = []
    with open(args.input, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Empty CSV", file=sys.stderr)
            return 1
        for row in reader:
            if args.n_scored is not None:
                try:
                    n = int(row["n_scored"])
                except (KeyError, ValueError):
                    dropped.append((row, "bad n_scored"))
                    continue
                if n != args.n_scored:
                    dropped.append((row, f"n_scored={n}"))
                    continue
            if args.prompt_type and row.get("prompt_type") != args.prompt_type:
                dropped.append((row, "prompt_type"))
                continue
            if args.split and row.get("split") != args.split:
                dropped.append((row, "split"))
                continue
            if args.require_strict_eval:
                ok, why = _strict_eval_ok(row)
                if not ok:
                    dropped.append((row, why))
                    continue
            kept.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    print(f"Wrote {len(kept)} rows to {args.output} (dropped {len(dropped)})")
    if not kept:
        print("No rows matched — relax filters or check input.", file=sys.stderr)
        if args.n_scored is None and not args.require_strict_eval:
            with open(args.input, encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                ns = sorted({int(x["n_scored"]) for x in r if str(x.get("n_scored", "")).isdigit()})
            print(f"Unique n_scored in input: {ns}", file=sys.stderr)
        return 3
    return 0


def cmd_verify_data(args: argparse.Namespace) -> int:
    db_path = ROOT / "output" / "corpus.db"
    splits_dir = ROOT / "evaluation" / "splits"
    tasks = ("meter_only", "rhyme_only", "natural_text", "combined")
    splits = ("train", "dev", "test")
    errors: list[str] = []

    if not db_path.is_file():
        errors.append(f"Missing {db_path} — run python scripts/export_sqlite.py")
    split_test = splits_dir / "test.json"
    if not split_test.is_file():
        errors.append(f"Missing {split_test} — run python evaluation/splits.py")

    counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        counts[task] = {}
        for sp in splits:
            p = ROOT / "output" / "training_data" / task / f"{sp}.json"
            if not p.is_file():
                errors.append(f"Missing {p} — run notebooks/01_prepare_training_data.ipynb")
                continue
            try:
                rows = json.loads(p.read_text(encoding="utf-8"))
                counts[task][sp] = len(rows) if isinstance(rows, list) else 0
            except Exception as e:
                errors.append(f"Bad JSON {p}: {e}")

    if errors:
        print("FAIL — prerequisites:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    db_mtime = db_path.stat().st_mtime
    newest_td = 0.0
    newest_path = ""
    for task in tasks:
        for sp in splits:
            p = ROOT / "output" / "training_data" / task / f"{sp}.json"
            if p.is_file():
                m = p.stat().st_mtime
                if m > newest_td:
                    newest_td = m
                    newest_path = str(p.relative_to(ROOT))

    print("OK — found corpus.db and training_data JSONs.\n")
    print("Line counts per task × split:")
    for task in tasks:
        parts = [f"{sp}={counts[task].get(sp, 0)}" for sp in splits]
        print(f"  {task:14s}  {'  '.join(parts)}")

    if newest_td > db_mtime + 60:
        print(
            f"\nNote: {newest_path} is newer than corpus.db by "
            f"{(newest_td - db_mtime) / 3600:.1f}h — fine if training_data JSON was rebuilt alone; "
            f"unexpected if the database changed without a fresh export."
        )

    conn = sqlite3.connect(db_path)
    n_poems = conn.execute("SELECT COUNT(*) FROM poems").fetchone()[0]
    n_lines = conn.execute("SELECT COUNT(*) FROM lines").fetchone()[0]
    conn.close()
    print(f"\nCorpus: {n_poems} poems, {n_lines} lines in {db_path.relative_to(ROOT)}")

    if args.check_poem_ids:
        test_ids = set(json.loads(split_test.read_text(encoding="utf-8")))
        bad = 0
        checked = 0
        for task in tasks:
            p = ROOT / "output" / "training_data" / task / "test.json"
            rows = json.loads(p.read_text(encoding="utf-8"))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                pid = row.get("poem_id")
                if pid is None:
                    continue
                checked += 1
                if pid not in test_ids:
                    bad += 1
                    if bad <= 5:
                        print(f"  WARN poem_id not in test split: {pid!r} ({task})")
        if checked == 0:
            print("\nWARN — no poem_id fields in test.json rows; skip alignment check or rebuild training_data.")
        elif bad:
            print(f"\nFAIL: {bad} test.json rows have poem_id outside evaluation/splits/test.json (checked {checked} rows with poem_id).")
            return 1
        else:
            print(f"\nOK — poem_id check: {checked} rows with poem_id in test.json are ⊆ test split ({len(test_ids)} poems).")

    print(
        "\nWhy task line counts differ: meter_only skips short/invalid stress; "
        "natural_text and combined use the same continuation rows (same N); "
        "rhyme_only uses one row per corpus line."
    )
    return 0


def cmd_nt_form(args: argparse.Namespace) -> int:
    from evaluation.form_eval_generation import aggregate_natural_text_form_results

    path = args.json_path
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results") or []
    ro = True if args.relax_oov else None
    fm = aggregate_natural_text_form_results(results, max_n=args.n, relax_oov=ro)

    print(f"File: {path}")
    print(f"relax_oov: {bool(fm.get('nt_form_relaxed_oov'))}")
    print(f"Rows scanned: {fm['nt_form_rows_scanned']}")
    print(f"Evaluable (gold+pred phonology OK): {fm['nt_form_evaluable']}")
    print(f"Skipped gold phonology issues: {fm['nt_form_fail_gold']} (includes non-evaluable pairs)")
    print(f"Skipped pred phonology issues: {fm['nt_form_fail_pred']}")
    print()
    ne = fm["nt_form_evaluable"]
    print(
        f"Stress pattern match (+/- exact, same length): "
        f"{fm['nt_form_stress_hits']}/{ne} = {fm['nt_form_stress_match_pct']}%"
    )
    print(
        f"Syllable count match:                  "
        f"{fm['nt_form_syllable_hits']}/{ne} = {fm['nt_form_syllable_match_pct']}%"
    )
    rd = fm["nt_form_rhyme_denom"]
    print(
        f"Rhyme key match (where gold has key): "
        f"{fm['nt_form_rhyme_hits']}/{rd} = {fm['nt_form_rhyme_match_pct']}%"
    )
    return 0


def cmd_meter_figure(_: argparse.Namespace) -> int:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}. Run: python scripts/export_sqlite.py")
        return 1

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT meter_type, COUNT(*) AS n
        FROM lines
        WHERE meter_type IS NOT NULL AND meter_type != '' AND meter_type != 'unknown'
        GROUP BY meter_type
        ORDER BY n DESC
    """).fetchall()
    conn.close()

    if not rows:
        print("No meter_type counts found.")
        return 1

    def shorten_label(meter_type: str) -> str:
        s = (meter_type or "").strip()
        if not s:
            return "?"
        if s == "iambic pentameter":
            return "Iambic pent."
        if s == "iambic tetrameter":
            return "Iambic tetr."
        if s == "trochaic pentameter":
            return "Trochaic pent."
        if s == "trochaic tetrameter":
            return "Trochaic tetr."
        return s[:18] + ("..." if len(s) > 18 else "")

    if len(rows) > MAX_METER_BARS:
        top = rows[: MAX_METER_BARS - 1]
        other_count = sum(n for _, n in rows[MAX_METER_BARS - 1 :])
        labels = [shorten_label(m) for m, _ in top] + ["Other"]
        counts = [n for _, n in top] + [other_count]
    else:
        labels = [shorten_label(m) for m, _ in rows]
        counts = [n for _, n in rows]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "meter_distribution.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,meter_type,count\n")
        for i, (label, count) in enumerate(zip(labels, counts)):
            f.write(f'{i},"{label}",{count}\n')
    print(f"Data for pgfplots: {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. pip install matplotlib")
        return 1

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    ax.bar(x, counts, color="steelblue", edgecolor="navy", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of lines")
    ax.set_xlabel("Meter type")
    ax.set_title("Meter type distribution (corpus lines with known meter)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_PDF}")
    return 0


def _last_context_line(inp: str) -> str:
    s = (inp or "").strip()
    if not s or s == "[start]":
        return ""
    parts = [p.strip() for p in s.split("|")]
    return parts[-1] if parts else ""


def _letter_ratio(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    letters = sum(1 for c in s if c.isalpha())
    return letters / max(len(s), 1)


def cmd_spotcheck_nt(args: argparse.Namespace) -> int:
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
        ctx = _last_context_line(inp)
        if ctx and pred.strip().lower() == ctx.lower():
            eq_ctx += 1
        if len(pred) < 10:
            short10 += 1
        if _letter_ratio(pred) < 0.15:
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
    return 0


def cmd_annotation_sources(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    totals = {
        "stress_poesy": 0,
        "stress_empty": 0,
        "stress_notebook": 0,
        "meter_poesy": 0,
        "meter_empty": 0,
        "meter_phonology": 0,
        "rhyme_poesy": 0,
        "rhyme_empty": 0,
        "meter_type_poesy": 0,
        "meter_type_phonology": 0,
    }
    poems_with_sources = 0
    poems_total = 0

    for path in sorted(root.glob("*.json")):
        poems_total += 1
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skip {path.name}: {e}", file=sys.stderr)
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
    s_poesy, s_empty = totals["stress_poesy"], totals["stress_empty"]
    pct_p = 100 * s_poesy / total_lines if total_lines else 0
    pct_e = 100 * s_empty / total_lines if total_lines else 0
    snb = totals.get("stress_notebook", 0)
    print("STRESS (non-empty line count; see stress_notebook for metricalgpt path)")
    print(f"  Non-empty: {s_poesy:>8}  ({pct_p:.1f}%)")
    print(f"  Empty:     {s_empty:>8}  ({pct_e:.1f}%)")
    if snb:
        print(f"  Notebook override lines (subset): {snb:>8}")
    print()
    m_poesy = totals["meter_poesy"]
    m_empty = totals.get("meter_empty", 0)
    m_phon = totals["meter_phonology"]
    m_total = m_poesy + m_empty + m_phon
    pct_mp = 100 * m_poesy / m_total if m_total else 0
    pct_me = 100 * m_empty / m_total if m_total else 0
    pct_mph = 100 * m_phon / m_total if m_total else 0
    print("METER (Prosodic pipe scansion vs empty; legacy meter_phonology = old CMU-in-meter bug)")
    print(f"  Poesy pipe:  {m_poesy:>8}  ({pct_mp:.1f}%)")
    print(f"  Empty:       {m_empty:>8}  ({pct_me:.1f}%)")
    if m_phon:
        print(f"  Legacy phon: {m_phon:>8}  ({pct_mph:.1f}%)")
    print()
    r_poesy, r_empty = totals["rhyme_poesy"], totals["rhyme_empty"]
    r_total = r_poesy + r_empty
    pct_rp = 100 * r_poesy / r_total if r_total else 0
    pct_re = 100 * r_empty / r_total if r_total else 0
    print("RHYME_GROUP (Poesy only)")
    print(f"  Poesy:   {r_poesy:>8}  ({pct_rp:.1f}%)")
    print(f"  Empty:   {r_empty:>8}  ({pct_re:.1f}%)")
    print()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_f = sub.add_parser("filter-csv", help="Filter model_comparison.csv for comparable rows")
    p_f.add_argument("--input", type=Path, default=ROOT / "evaluation/results/model_comparison.csv")
    p_f.add_argument("-o", "--output", type=Path, required=True)
    p_f.add_argument("--n-scored", type=int, default=None)
    p_f.add_argument("--prompt-type", type=str, default=None)
    p_f.add_argument("--split", type=str, default=None)
    p_f.add_argument("--require-strict-eval", action="store_true")
    p_f.set_defaults(_run=cmd_filter_csv)

    p_v = sub.add_parser("verify-data", help="Check corpus.db + training_data alignment")
    p_v.add_argument("--check-poem-ids", action="store_true")
    p_v.set_defaults(_run=cmd_verify_data)

    p_n = sub.add_parser("nt-form", help="CMU form metrics for one natural_text baseline JSON")
    p_n.add_argument("json_path", type=Path)
    p_n.add_argument("--n", type=int, default=None)
    p_n.add_argument("--relax-oov", action="store_true")
    p_n.set_defaults(_run=cmd_nt_form)

    p_m = sub.add_parser("meter-figure", help="Meter distribution CSV + PDF from corpus.db")
    p_m.set_defaults(_run=cmd_meter_figure)

    p_s = sub.add_parser("spotcheck-nt", help="Spot-check natural_text baseline JSON")
    p_s.add_argument("json_path", type=Path)
    p_s.add_argument("--samples", type=int, default=5)
    p_s.add_argument("--phon-check", action="store_true")
    p_s.set_defaults(_run=cmd_spotcheck_nt)

    p_a = sub.add_parser("annotation-sources", help="Summarize annotation_sources in poems_annotated/")
    p_a.add_argument("dir", nargs="?", default="output/poems_annotated")
    p_a.set_defaults(_run=cmd_annotation_sources)

    args = parser.parse_args()
    fn = args._run
    a = vars(args).copy()
    for k in ("cmd", "_run"):
        a.pop(k, None)
    ns = argparse.Namespace(**a)
    raise SystemExit(fn(ns))


if __name__ == "__main__":
    main()
