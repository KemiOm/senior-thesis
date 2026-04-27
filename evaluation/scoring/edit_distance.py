"""Character-level Levenshtein distance=

This complements exact match. Structured outputs can differ slightly  but still be very close overall.

Before comparing, normalize strings:
meter formats (0/1 vs +/-), rhyme casing/whitespace, and combined outputs
reordered into a consistent key:value format.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from evaluation.scoring.struct_metrics import (
    canonical_combined_for_compare,
    parse_combined_bundle,
    parse_combined_bundle_loose,
    stress_normalize_for_compare,
    _normalize_ws_case,
)


def normalize_for_eval(task: str, s: str) -> str:
    if task == "natural_text":
        s = (s or "").strip()
        return re.sub(r"\s+", " ", s)
    if task == "meter_only":
        return stress_normalize_for_compare((s or "").strip())
    if task == "rhyme_only":
        return _normalize_ws_case(s or "")
    if task == "combined":
        raw = (s or "").strip()
        g = parse_combined_bundle(raw) or parse_combined_bundle_loose(raw)
        if g:
            return canonical_combined_for_compare(g)
        return re.sub(r"\s+", " ", raw)
    return (s or "").strip()


def levenshtein(a: str, b: str) -> int:
    """Minimum single-character insertions, deletions, substitutions to turn a into b."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure a is shorter to reduce memory (still O(mn) time)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            ins = cur[j] + 1
            delete = prev[j + 1] + 1
            sub = prev[j] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def aggregate_string_edit_metrics(task: str, results: list[dict[str, Any]]) -> dict[str, float]:
    """Per-row Levenshtein vs gold; skips rows with empty gold (same convention as exact_match_rate)."""
    dists: list[int] = []
    norms: list[float] = []
    for row in results:
        gold = normalize_for_eval(task, row.get("gold_target", ""))
        pred = normalize_for_eval(task, row.get("model_output", ""))
        if not gold:
            continue
        d = levenshtein(gold, pred)
        dists.append(d)
        denom = max(len(gold), len(pred), 1)
        norms.append(d / denom)

    if not dists:
        return {
            "levenshtein_dist_mean": 0.0,
            "levenshtein_dist_median": 0.0,
            "levenshtein_norm_mean": 0.0,
            "levenshtein_similarity_pct": 0.0,
        }

    mean_norm = float(sum(norms) / len(norms))
    sim = 100.0 * (1.0 - mean_norm)
    return {
        "levenshtein_dist_mean": round(float(sum(dists) / len(dists)), 3),
        "levenshtein_dist_median": round(float(statistics.median(dists)), 3),
        "levenshtein_norm_mean": round(mean_norm, 4),
        "levenshtein_similarity_pct": round(sim, 3),
    }
