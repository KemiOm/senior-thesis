"""Extra metrics for rhyme_only and combined prompt baselines.

Exact match on the full string is strict; these add field-level scores where the target is
structured (rhyme token, or combined meter|rhyme|end|caesura bundles).
"""

from __future__ import annotations

import re
from typing import Any

# New: explicit stress pattern + meter label 
_COMBINED_NEW_RE = re.compile(
    r"stress:([^|]+)\|meter_type:([^|]+)\|rhyme:([^|]+)\|end:([^|]+)\|caesura:([^|]+)",
    re.IGNORECASE | re.DOTALL,
)
# What I had before 
_COMBINED_OLD_RE = re.compile(
    r"meter:([^|]+)\|rhyme:([^|]+)\|end:([^|]+)\|caesura:([^|]+)",
    re.IGNORECASE | re.DOTALL,
)


def _strip(s: str) -> str:
    return (s or "").strip()


def _normalize_ws_case(s: str) -> str:
    return re.sub(r"\s+", " ", _strip(s)).casefold()


def stress_normalize_for_compare(s: str) -> str:
    """Align meter slot: +/- string or 01 string to canonical +/-."""
    s = _strip(s)
    if not s:
        return ""
    if all(c in "+-" for c in s):
        return s
    if all(c in "01" for c in s):
        return "".join("+" if c == "1" else "-" for c in s)
    return s


def meters_equivalent(a: str, b: str) -> bool:
    return stress_normalize_for_compare(a) == stress_normalize_for_compare(b)


def parse_combined_bundle(s: str) -> dict[str, str] | None:
    """Extract fields from a combined target string ."""
    s = _strip(s)
    m = _COMBINED_NEW_RE.search(s)
    if m:
        stress = m.group(1).strip()
        mt = m.group(2).strip()
        return {
            "stress": stress,
            "meter_type": mt,
            "meter": stress,
            "rhyme": m.group(3).strip(),
            "end": m.group(4).strip(),
            "caesura": m.group(5).strip(),
        }
    m = _COMBINED_OLD_RE.search(s)
    if not m:
        return None
    return {
        "stress": "",
        "meter_type": "",
        "meter": m.group(1).strip(),
        "rhyme": m.group(2).strip(),
        "end": m.group(3).strip(),
        "caesura": m.group(4).strip(),
    }


def rhyme_tokens_equivalent(gold: str, pred: str) -> bool:
    return _normalize_ws_case(gold) == _normalize_ws_case(pred)


def aggregate_rhyme_only_structured(results: list[dict[str, Any]]) -> dict[str, float | int]:
    """
    Relaxed exact on rhyme token: case-insensitive, whitespace-normalized.
    Skips rows with empty gold 
    """
    n_scored = 0
    n_relaxed = 0
    for row in results:
        gold = _strip(row.get("gold_target", ""))
        pred = row.get("model_output", "") or ""
        if not gold:
            continue
        n_scored += 1
        if rhyme_tokens_equivalent(gold, pred):
            n_relaxed += 1

    def pct(a: int, b: int) -> float:
        return round(100.0 * a / b, 3) if b else 0.0

    return {
        "st_rhyme_n_scored": n_scored,
        "st_rhyme_relaxed_match_pct": pct(n_relaxed, n_scored),
    }


def aggregate_combined_structured(results: list[dict[str, Any]]) -> dict[str, float | int]:
    """
    Per-field match rates for combined bundles. Gold must parse; pred may fail to parse (all fields miss).
    Stress slot (reported as st_combined_meter_field_pct): +/- vs 01 normalized before compare.
    New-format bundles also score meter_type 
    """
    n_scored = 0
    n_pred_parse = 0
    n_meter = n_rhyme = n_end = n_caes = n_all_four = 0
    n_meter_type = n_meter_type_scored = 0
    sum_mean_field = 0.0

    for row in results:
        gold_s = _strip(row.get("gold_target", ""))
        pred_s = row.get("model_output", "") or ""
        if not gold_s:
            continue
        g = parse_combined_bundle(gold_s)
        if not g:
            continue
        n_scored += 1
        p = parse_combined_bundle(pred_s)
        if p:
            n_pred_parse += 1
        else:
            p = {"meter": "", "meter_type": "", "rhyme": "", "end": "", "caesura": ""}

        hm = meters_equivalent(g["meter"], p["meter"])
        hr = rhyme_tokens_equivalent(g["rhyme"], p["rhyme"])
        he = _strip(g["end"]) == _strip(p["end"])
        hc = _strip(g["caesura"]) == _strip(p["caesura"])
        gmt = _strip(g.get("meter_type", ""))
        pmt = _strip(p.get("meter_type", ""))
        hmt = True
        if gmt:
            n_meter_type_scored += 1
            hmt = _normalize_ws_case(gmt) == _normalize_ws_case(pmt)
            if hmt:
                n_meter_type += 1
        if hm:
            n_meter += 1
        if hr:
            n_rhyme += 1
        if he:
            n_end += 1
        if hc:
            n_caes += 1
        n_fields = 5 if gmt else 4
        bits = (hm, hmt, hr, he, hc) if gmt else (hm, hr, he, hc)
        sum_mean_field += sum(1 for b in bits if b) / float(n_fields)
        if gmt:
            if hm and hmt and hr and he and hc:
                n_all_four += 1
        else:
            if hm and hr and he and hc:
                n_all_four += 1

    def pct(a: int, b: int) -> float:
        return round(100.0 * a / b, 3) if b else 0.0

    return {
        "st_combined_n_scored": n_scored,
        "st_combined_pred_parse_pct": pct(n_pred_parse, n_scored),
        "st_combined_meter_field_pct": pct(n_meter, n_scored),
        "st_combined_meter_type_field_pct": pct(n_meter_type, n_meter_type_scored) if n_meter_type_scored else 0.0,
        "st_combined_rhyme_field_pct": pct(n_rhyme, n_scored),
        "st_combined_end_field_pct": pct(n_end, n_scored),
        "st_combined_caesura_field_pct": pct(n_caes, n_scored),
        "st_combined_mean_field_pct": round(100.0 * sum_mean_field / n_scored, 3) if n_scored else 0.0,
        "st_combined_all_four_pct": pct(n_all_four, n_scored),
    }
