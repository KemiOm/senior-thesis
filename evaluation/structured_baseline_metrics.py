"""Extra metrics for rhyme_only and combined prompt baselines.

Exact match on the full string is strict; these add field-level scores where the target is
structured (rhyme token, or combined meter|rhyme|end|caesura bundles).
"""

from __future__ import annotations

import re
from typing import Any

_COMBINED_RE = re.compile(
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
    """Extract meter/rhyme/end/caesura from a combined target string."""
    m = _COMBINED_RE.search(_strip(s))
    if not m:
        return None
    return {
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
    Skips rows with empty gold (same convention as exact_match_rate).
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
    Meter slot: +/- vs 01 normalized before compare.
    """
    n_scored = 0
    n_pred_parse = 0
    n_meter = n_rhyme = n_end = n_caes = n_all_four = 0
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
            p = {"meter": "", "rhyme": "", "end": "", "caesura": ""}

        hm = meters_equivalent(g["meter"], p["meter"])
        hr = rhyme_tokens_equivalent(g["rhyme"], p["rhyme"])
        he = _strip(g["end"]) == _strip(p["end"])
        hc = _strip(g["caesura"]) == _strip(p["caesura"])
        if hm:
            n_meter += 1
        if hr:
            n_rhyme += 1
        if he:
            n_end += 1
        if hc:
            n_caes += 1
        bits = (hm, hr, he, hc)
        sum_mean_field += sum(bits) / 4.0
        if all(bits):
            n_all_four += 1

    def pct(a: int, b: int) -> float:
        return round(100.0 * a / b, 3) if b else 0.0

    return {
        "st_combined_n_scored": n_scored,
        "st_combined_pred_parse_pct": pct(n_pred_parse, n_scored),
        "st_combined_meter_field_pct": pct(n_meter, n_scored),
        "st_combined_rhyme_field_pct": pct(n_rhyme, n_scored),
        "st_combined_end_field_pct": pct(n_end, n_scored),
        "st_combined_caesura_field_pct": pct(n_caes, n_scored),
        "st_combined_mean_field_pct": round(100.0 * sum_mean_field / n_scored, 3) if n_scored else 0.0,
        "st_combined_all_four_pct": pct(n_all_four, n_scored),
    }
