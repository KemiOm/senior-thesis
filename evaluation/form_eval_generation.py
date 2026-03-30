"""Form metrics for natural_text generations.

Compares gold vs model output with one phonology path.
Set FORM_EVAL_RELAX_OOV=1 to allow partial CMU coverage.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from sample.phonology_sample import get_phonology_for_line, line_meter_from_phonology


def rhyme_key_from_phonology(phonology_json: str) -> str:
    """Same rule as training / run_prompt_baseline (ARPAbet tail from last stressed vowel)."""
    if not phonology_json:
        return ""
    try:
        ph = json.loads(phonology_json)
    except Exception:
        return ""
    phones: list[str] = []
    for p in ph:
        arp = p.get("arpabet")
        if isinstance(arp, list) and arp:
            phones.extend(arp[0].split() if isinstance(arp[0], str) else arp[0])
        elif isinstance(arp, str):
            phones.extend(arp.split())
    if not phones:
        return ""
    for i in range(len(phones) - 1, -1, -1):
        p = phones[i]
        if len(p) > 1 and p[-1] in "12":
            return " ".join(phones[i:])
    return " ".join(phones[-3:]) if len(phones) >= 3 else " ".join(phones)


def stress_01_to_pm(s: str) -> str:
    if not s:
        return ""
    return "".join("+" if c == "1" else "-" for c in s if c in "01")


def phonology_has_not_found(phon: list[dict]) -> bool:
    for p in phon or []:
        if p.get("source") == "not_found":
            return True
    return False


def line_form_signature(line: str, *, relax_oov: bool = False) -> dict[str, Any]:
    """
    Extract comparable form features from arbitrary line text.
    Uses CMU + optional espeak (see sample.phonology_sample.get_arpabet) — same pipeline for gold and pred.

    If relax_oov is True, a line counts as ok when there is a non-empty stress string even if some
    words are CMU not_found (stress comes from pronounced words only; weaker coverage on archaic text).
    """
    text = (line or "").strip()
    if not text:
        return {
            "ok": False,
            "reason": "empty",
            "n_syllables": 0,
            "stress_01": "",
            "stress_pm": "",
            "rhyme_key": "",
            "phonology": [],
        }
    phon = get_phonology_for_line(text)
    bad = phonology_has_not_found(phon)
    j = json.dumps(phon)
    n_syl, stress_01, meter_type = line_meter_from_phonology(phon)
    rk = rhyme_key_from_phonology(j)
    if relax_oov:
        ok = bool(stress_01)
        reason = "no_stress" if not stress_01 else ("partial_oov" if bad else "")
    else:
        ok = bool(stress_01) and not bad
        reason = "not_found_word" if bad else ("no_stress" if not stress_01 else "")
    return {
        "ok": ok,
        "reason": reason,
        "n_syllables": n_syl,
        "stress_01": stress_01 or "",
        "stress_pm": stress_01_to_pm(stress_01 or ""),
        "rhyme_key": rk.strip(),
        "meter_type": meter_type,
        "phonology": phon,
    }


def compare_next_line_form(
    gold_line: str, pred_line: str, *, relax_oov: bool = False
) -> dict[str, Any]:
    """
    Gold = corpus next line; pred = model generation.
    Returns flags for tables (exact stress match, syllable count, rhyme key).
    """
    g = line_form_signature(gold_line, relax_oov=relax_oov)
    p = line_form_signature(pred_line, relax_oov=relax_oov)
    out: dict[str, Any] = {
        "gold_ok": g["ok"],
        "pred_ok": p["ok"],
        "evaluable": g["ok"] and p["ok"],
        "n_syllables_gold": g["n_syllables"],
        "n_syllables_pred": p["n_syllables"],
        "syllable_count_match": g["n_syllables"] == p["n_syllables"] and g["n_syllables"] > 0,
        "stress_exact_01": False,
        "stress_exact_pm": False,
        "rhyme_key_match": False,
    }
    if not out["evaluable"]:
        return out
    gs, ps = g["stress_01"], p["stress_01"]
    out["stress_exact_01"] = gs == ps and bool(gs)
    out["stress_exact_pm"] = g["stress_pm"] == p["stress_pm"] and bool(g["stress_pm"])
    gr, pr = g["rhyme_key"].upper(), p["rhyme_key"].upper()
    out["rhyme_key_match"] = bool(gr) and gr == pr
    return out


def _env_relax_oov() -> bool:
    return os.environ.get("FORM_EVAL_RELAX_OOV", "").strip() in ("1", "true", "yes")


def aggregate_natural_text_form_results(
    results: list,
    max_n: int | None = None,
    *,
    relax_oov: bool | None = None,
) -> dict[str, float | int]:
    """
    Batch metrics for natural_text baseline JSON results[] (gold_target vs model_output).
    Same counting rules as scripts/corpus_tools.py nt-form.
    relax_oov: if True, allow lines with partial CMU coverage (see line_form_signature).
               If None, read FORM_EVAL_RELAX_OOV=1 from the environment.
    """
    if relax_oov is None:
        relax_oov = _env_relax_oov()
    rows = results[: max_n] if max_n else results
    n_ev = 0
    n_stress = 0
    n_syl = 0
    n_rhyme = 0
    n_rhyme_gold_nonempty = 0
    n_fail_gold = 0
    n_fail_pred = 0

    for row in rows:
        gold = (row.get("gold_target") or "").strip()
        pred = (row.get("model_output") or "").strip()
        if not gold or not pred:
            continue
        cmp = compare_next_line_form(gold, pred, relax_oov=relax_oov)
        if not cmp["gold_ok"]:
            n_fail_gold += 1
        if not cmp["pred_ok"]:
            n_fail_pred += 1
        if not cmp["evaluable"]:
            continue
        n_ev += 1
        if cmp["stress_exact_pm"]:
            n_stress += 1
        if cmp["syllable_count_match"]:
            n_syl += 1
        gsig = line_form_signature(gold, relax_oov=relax_oov)
        if gsig.get("rhyme_key"):
            n_rhyme_gold_nonempty += 1
            if cmp["rhyme_key_match"]:
                n_rhyme += 1

    def pct(a: int, b: int) -> float:
        return round(100.0 * a / b, 3) if b else 0.0

    return {
        "nt_form_relaxed_oov": int(relax_oov),
        "nt_form_rows_scanned": len(rows),
        "nt_form_evaluable": n_ev,
        "nt_form_fail_gold": n_fail_gold,
        "nt_form_fail_pred": n_fail_pred,
        "nt_form_stress_hits": n_stress,
        "nt_form_syllable_hits": n_syl,
        "nt_form_rhyme_hits": n_rhyme,
        "nt_form_stress_match_pct": pct(n_stress, n_ev),
        "nt_form_syllable_match_pct": pct(n_syl, n_ev),
        "nt_form_rhyme_denom": n_rhyme_gold_nonempty,
        "nt_form_rhyme_match_pct": pct(n_rhyme, n_rhyme_gold_nonempty),
    }
