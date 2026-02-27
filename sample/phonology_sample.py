"""
Add phonology, meter, rhyme, and punctuation annotations to one normalized poem.
Uses: Poesy/Prosodic (rhyme, meter, stress), CMU dict (phonology/ARPAbet), programmatic checks (end_stopped, caesura).
All annotations are automated—no manual tagging.

Install first:
  brew install espeak
  pip install pronouncing
  pip install git+https://github.com/quadrismegistus/poesy
"""

import json
import re
import threading
from pathlib import Path

# Project root (parent of sample/). Paths resolve relative to project root so scripts
# work from sample/ or from the project root.
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "output/poems_normalized"
OUTPUT_DIR = ROOT / "output/poems_annotated"

# Patch multiprocessing.Manager BEFORE Prosodic/hashstash can import it.
# hashstash creates Manager() at import; its subprocess fails in sandbox/restricted envs.
# Fallback: in-process fake manager so Poesy rhyme/meter can run.
def _patch_manager_for_hashstash():
    import multiprocessing as mp
    _orig = mp.Manager

    class _FakeManager:
        def dict(self):
            return {}

        def Lock(self):
            return threading.Lock()

    def _patched_manager():
        try:
            return _orig()
        except (EOFError, OSError, PermissionError, ConnectionError):
            return _FakeManager()

    mp.Manager = _patched_manager


_patch_manager_for_hashstash()

# Project-local NLTK data (Prosodic needs punkt); avoids ~/nltk_data permission issues.
_nltk_data = ROOT / "data" / "nltk_data"
_nltk_data.mkdir(parents=True, exist_ok=True)
import os
os.environ.setdefault("NLTK_DATA", str(_nltk_data))


def _ensure_nltk_punkt():
    """Download NLTK punkt if missing (required by Prosodic tokenizer)."""
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", download_dir=str(_nltk_data), quiet=True)
        except Exception:
            try:
                nltk.download("punkt", download_dir=str(_nltk_data), quiet=True)
            except Exception:
                pass


def _patch_prosodic_config():
    """Prosodic 2.x may not expose config; Poesy expects prosodic.config."""
    try:
        import prosodic
        if not hasattr(prosodic, "config"):
            prosodic.config = {}
    except Exception:
        pass


def _patch_prosodic_text_meter():
    """Prosodic 2.x Text() has fixed signature; Poesy passes meter= string."""
    try:
        import prosodic
        from prosodic.texts.texts import TextModel
        from prosodic.ents import Entity

        def _patched_text(txt="", fn="", lang=None, parent=None, children=None, tokens_df=None, **kwargs):
            kwargs.pop("meter", None)  # Poesy passes string; Prosodic needs Meter object or None
            return TextModel(
                txt=txt, fn=fn, lang=lang, parent=parent,
                children=children or [], tokens_df=tokens_df, **kwargs
            )

        prosodic.Text = _patched_text
        prosodic.texts.texts.Text = _patched_text

        # Entity.get_meter: Poesy calls line.parse(meter="..."); Prosodic expects Meter object
        _orig_get_meter = Entity.get_meter

        def _patched_get_meter(self, meter=None, **meter_kwargs):
            if isinstance(meter, str):
                meter = None  # use Prosodic default
            return _orig_get_meter(self, meter=meter, **meter_kwargs)

        Entity.get_meter = _patched_get_meter

        # parse/parse_iter: Poesy passes meter= string; Prosodic expects Meter object
        import prosodic.texts.texts as _texts_mod
        _orig_parse = _texts_mod.TextModel.parse

        def _patched_parse(self, combine_by=None, num_proc=None, force=False, lim=None, meter=None, **meter_kwargs):
            if isinstance(meter, str):
                meter = None
            return _orig_parse(
                combine_by=combine_by, num_proc=num_proc, force=force,
                lim=lim, meter=meter, **meter_kwargs
            )

        _texts_mod.TextModel.parse = _patched_parse
    except Exception:
        pass


_ensure_nltk_punkt()
_patch_prosodic_config()
_patch_prosodic_text_meter()

# Import pronouncing first (before Poesy) so it loads cleanly
try:
    import pronouncing
    HAS_PRONOUNCING = True
except Exception as e:
    HAS_PRONOUNCING = False
    _PRONOUNCING_ERR = str(e)

from poesy import Poem as PoesyPoem


def _arpabet_from_espeak(word: str) -> list:
    """Use espeak subprocess to get IPA; best-effort conversion to ARPAbet. Returns [] if espeak fails."""
    import subprocess
    try:
        clean = re.sub(r"^[^\w']+|[^\w']+$", "", word)
        clean = clean.replace("'", "").lower()
        if not clean:
            return []
        out = subprocess.run(
            ["espeak", "-q", "-x", "-v", "en", clean],
            capture_output=True, text=True, timeout=2
        )
        if out.returncode != 0 or not out.stdout.strip():
            return []
        ipa = out.stdout.strip()
        # Minimal IPA->ARPAbet: common vowels (add stress 1), consonants
        v = {"ɑ": "AA1", "æ": "AE1", "ʌ": "AH1", "ɔ": "AO1", "ə": "AH0", "ɛ": "EH1",
             "eɪ": "EY1", "ɪ": "IH1", "i": "IY1", "aɪ": "AY1", "oʊ": "OW1",
             "ɔɪ": "OY1", "ʊ": "UH1", "u": "UW1", "aʊ": "AW1"}
        c = {"p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G",
             "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
             "ʃ": "SH", "ʒ": "ZH", "h": "HH", "m": "M", "n": "N", "ŋ": "NG",
             "l": "L", "r": "R", "w": "W", "j": "Y", "tʃ": "CH", "dʒ": "JH"}
        phones = []
        i = 0
        while i < len(ipa):
            for k in sorted(list(v.keys()) + list(c.keys()), key=len, reverse=True):
                if ipa[i:i + len(k)] == k:
                    phones.append(v.get(k) or c.get(k, k))
                    i += len(k)
                    break
            else:
                i += 1
        return phones if phones else []
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def get_arpabet(word: str) -> tuple:
    """
    Look up ARPAbet pronunciation. Returns (phones_list, source).
    Tries: CMU dict, apostrophe variants, then espeak fallback (disabled when PHONOLOGY_BATCH=1).
    """
    skip_espeak = os.environ.get("PHONOLOGY_BATCH", "").strip() == "1"
    if not HAS_PRONOUNCING:
        phones = _arpabet_from_espeak(word)
        return (phones, "espeak" if phones else "not_found")
    clean = re.sub(r"^[^\w']+|[^\w']+$", "", word)
    if not clean:
        return ([], "not_found")
    phones = pronouncing.phones_for_word(clean.lower())
    if phones:
        return (phones, "cmudict")
    if "'" in clean:
        alt = clean.replace("'", "").lower()
        phones = pronouncing.phones_for_word(alt)
        if phones:
            return (phones, "cmudict")
    phones = _arpabet_from_espeak(word)
    return (phones, "espeak" if phones else "not_found")


def get_phonology_for_line(line_text: str) -> list:
    """Build per-word phonology list: {word, arpabet, source}."""
    words = re.findall(r"[\w']+", line_text)
    out = []
    for w in words:
        phones, src = get_arpabet(w)
        out.append({"word": w, "arpabet": phones, "source": src})
    return out


def get_rhyme_word(phonology: list):
    """
    The word that carries the rhyme: last content word (line-final rhyme word).
    Uses last word with arpabet; falls back to last word.
    """
    if not phonology:
        return None
    for p in reversed(phonology):
        word = p.get("word", "")
        if not word:
            continue
        if p.get("arpabet"):
            return word
    return phonology[-1].get("word") if phonology else None


def phones_to_stress(phones: list) -> list:
    """Extract stress digits from CMU phones: AH0->0, IY1->1, OW2->2."""
    stresses = []
    for ph in (phones or []):
        p = ph if isinstance(ph, str) else (ph[0] if isinstance(ph, (list, tuple)) and ph else None)
        if p and len(p) > 1 and p[-1] in "012":
            stresses.append(int(p[-1]))
    return stresses


FEET_NAMES = {1: "monometer", 2: "dimeter", 3: "trimeter", 4: "tetrameter", 5: "pentameter",
              6: "hexameter", 7: "heptameter", 8: "octameter"}


def line_meter_from_phonology(phonology: list) -> tuple:
    """Compute syllable count, binary stress pattern, and meter_type from phonology. Returns (syllable_count, stress_pattern, meter_type)."""
    stresses = []
    for p in phonology:
        arp = p.get("arpabet") or []
        pron = arp[0] if arp else []
        if isinstance(pron, str):
            pron = pron.split()
        elif isinstance(pron, list) and pron and isinstance(pron[0], str):
            pass
        else:
            continue
        stresses.extend(phones_to_stress(pron))
    stress_binary = "".join("1" if s > 0 else "0" for s in stresses)
    n = len(stresses)
    if n == 0:
        return (0, "", "unknown")
    feet = n // 2
    if n == 10 and stress_binary.startswith("01"):
        meter_type = "iambic pentameter"
    elif n == 10 and stress_binary.startswith("10"):
        meter_type = "trochaic pentameter"
    elif n == 8 and stress_binary.startswith("01"):
        meter_type = "iambic tetrameter"
    elif n == 8 and stress_binary.startswith("10"):
        meter_type = "trochaic tetrameter"
    elif feet >= 2 and stress_binary.startswith("01"):
        meter_type = "iambic " + FEET_NAMES.get(feet, f"{feet} feet")
    elif feet >= 2 and stress_binary.startswith("10"):
        meter_type = "trochaic " + FEET_NAMES.get(feet, f"{feet} feet")
    else:
        meter_type = FEET_NAMES.get(feet, f"{n} syllable")
    return (n, stress_binary, meter_type)


END_STOP_PUNCT = {".", "!", "?", ";", ":", ")", "]", "}", ","}


def get_end_stopped(line_text: str) -> bool:
    """True if line ends with sentence-ending punctuation (matches shared build_tables)."""
    s = line_text.strip()
    return (s[-1] if s else "") in END_STOP_PUNCT


def get_caesura(line_text: str):
    """
    Index of first strong mid-line punctuation (comma, semicolon, colon, em-dash).
    Returns 0-based word index or None if no caesura.
    """
    words = re.findall(r"[\w']+|[^\w\s]", line_text)
    for i, tok in enumerate(words):
        if tok in ",;:—-" and i > 0 and i < len(words) - 1:
            return i
    return None


def build_poem_string(poem: dict) -> str:
    """Convert normalized stanzas to newline-separated string for Poesy. Truncates long lines when MAX_LINE_CHARS set."""
    max_chars = 0
    try:
        max_chars = int(os.environ.get("MAX_LINE_CHARS", "0") or "0")
    except ValueError:
        pass
    lines = []
    for stanza in poem["stanzas"]:
        for line_dict in stanza:
            norm = line_dict["normalized"]
            if max_chars > 0 and len(norm) > max_chars:
                norm = "…" + norm[-max_chars:]  # keep end (rhyme word) for Poesy
            lines.append(norm)
        lines.append("")  # blank between stanzas
    return "\n".join(lines).strip()


def _log_poesy(msg: str) -> None:
    """Print diagnostic message to stderr (for troubleshooting rhyme_group). Suppressed when POESY_DEBUG=0."""
    if os.environ.get("POESY_DEBUG", "1") != "0":
        print(f"[poesy-debug] {msg}", file=__import__("sys").stderr)


def annotate_with_poesy(poem: dict) -> dict:
    """
    Use Poesy (Prosodic) for rhyme, meter, stress.
    Returns {per_line: {(si,li): {...}}, meter_type: "iambic pentameter"}.
    Skips Poesy when MAX_POEM_LINES exceeded (batch mode) to avoid long runs on huge poems.
    """
    import sys
    num_lines = sum(len(s) for s in poem.get("stanzas", []))
    max_poem_lines = 0
    try:
        max_poem_lines = int(os.environ.get("MAX_POEM_LINES", "0") or "0")
    except ValueError:
        pass
    if max_poem_lines > 0 and num_lines > max_poem_lines:
        _log_poesy(f"skip Poesy: poem has {num_lines} lines (max={max_poem_lines})")
        return {"per_line": {}, "meter_type": None}
    text = build_poem_string(poem)
    if not text.strip():
        _log_poesy("empty text, returning {}")
        return {"per_line": {}, "meter_type": None}
    try:
        _log_poesy(f"building Poem from {len(text)} chars, {text.count(chr(10))} newlines")
        p = PoesyPoem(text)
        _log_poesy("calling p.parse()...")
        p.parse()  # trigger metrical parsing
        _log_poesy("accessing p.rhymes (triggers rhyme_net)...")
        rhymes_raw = p.rhymes  # trigger rhyme_net computation
        _log_poesy(f"p.rhymes type={type(rhymes_raw).__name__}, len={len(rhymes_raw) if rhymes_raw else 0}")
        if rhymes_raw:
            sample = list(rhymes_raw.items())[:5]
            _log_poesy(f"p.rhymes sample: {sample}")
        keys_sorted = sorted(p.lined.keys())  # canonical line order (linenum, stanzanum)
        rhymes_list = [rhymes_raw.get(k, "") if rhymes_raw else "" for k in keys_sorted]
        _log_poesy(f"keys_sorted[:8]={keys_sorted[:8]} rhymes_list[:8]={rhymes_list[:8]}")
        prosodic_list = []
        for k in keys_sorted:
            line_obj = p.prosodic.get(k)
            if line_obj is None:
                prosodic_list.append((None, None))
                continue
            parse_str = getattr(line_obj, "parse_str", None)
            parse_str = parse_str(viols=False) if callable(parse_str) else None
            bp = None
            for attr in ("bestParses", "best_parses"):
                val = getattr(line_obj, attr, None)
                if val is not None:
                    bps = val() if callable(val) else val
                    if hasattr(bps, "__iter__") and not isinstance(bps, str):
                        bps = list(bps) if bps else []
                        bp = bps[0] if bps else None
                    break
            meter = getattr(bp, "meter_str", None) or getattr(bp, "parse_meter", None) or (str(bp) if bp else parse_str)
            stress = getattr(bp, "stress_str", None) or getattr(bp, "parse_stress", None) or getattr(bp, "parse_stress_str", None)
            if callable(stress):
                stress = stress()
            prosodic_list.append((meter, stress))
        ann = {}
        pos = 0
        for si, stanza in enumerate(poem["stanzas"]):
            for li in range(len(stanza)):
                ann[(si, li)] = {}
                if pos < len(rhymes_list):
                    ann[(si, li)]["rhyme_group"] = rhymes_list[pos]
                if pos < len(prosodic_list):
                    m, s = prosodic_list[pos]
                    ann[(si, li)]["meter"] = m
                    ann[(si, li)]["stress"] = s
                pos += 1
        non_empty_rhymes = sum(1 for v in ann.values() if v.get("rhyme_group") and str(v.get("rhyme_group", "")).strip() not in ("", "?"))
        _log_poesy(f"ann built: {len(ann)} entries, {non_empty_rhymes} non-empty rhyme_groups")
        # meter_type from Poesy: statd has meter_type_scheme (iambic, trochaic, etc.) + beat_scheme_repr (Pentameter, etc.)
        meter_type = None
        try:
            statd = getattr(p, "statd", None) or {}
            scheme = (statd.get("meter_type_scheme") or "").strip()
            beat = (statd.get("beat_scheme_repr") or "").strip()
            if scheme:
                meter_type = (scheme + " " + beat).strip() if beat else scheme
            if not meter_type:
                meterd = getattr(p, "meterd", None) or {}
                scheme = (meterd.get("type_scheme") or "").strip()
                if scheme:
                    meter_type = scheme
        except Exception:
            pass
        _log_poesy(f"meter_type from Poesy: {meter_type!r}")
        return {"per_line": ann, "meter_type": meter_type}
    except Exception as e:
        import traceback
        _log_poesy(f"EXCEPTION: {type(e).__name__}: {e}")
        _log_poesy(traceback.format_exc())
        # Root cause: Prosodic->hashstash creates multiprocessing.Manager() at import.
        # Manager subprocess fails with PermissionError binding socket -> parent gets EOFError.
        # Fix: run outside sandbox; check macOS Full Disk Access / Terminal permissions; or patch hashstash.
        _log_poesy("ROOT_CAUSE: (1) hashstash Manager fails in restricted envs; (2) Poesy/Prosodic 2.x API mismatch")
        _log_poesy("  (meter=str vs Meter object) -> parse fails before rhyme_net runs.")
        import warnings
        warnings.warn(f"Poesy annotation failed: {e}")
        return {"per_line": {}, "meter_type": None}


STANZA_NAMES = {2: "couplet", 4: "quatrain", 6: "sestet", 8: "octet", 3: "tercet", 5: "cinquain"}


def _rhyme_pairs(rhyme_groups: list) -> list:
    """Pairs of line indices that rhyme, e.g. [(0,1)] for couplet, [(0,2),(1,3)] for abab."""
    groups = {}
    for i, rg in enumerate(rhyme_groups):
        if rg and str(rg).strip() and str(rg).strip() != "?" and str(rg) != "-":
            g = str(rg)
            groups.setdefault(g, []).append(i)
    pairs = []
    for indices in groups.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pairs.append([indices[a], indices[b]])
    return pairs


def annotate_poem(poem: dict) -> dict:
    """
    Add phonology, meter, rhyme, end_stopped, caesura, enjambment to each line.
    Output: stanzas list of {stanza_index, stanza_type, rhyme_scheme, rhyme_pairs, lines}.
    Each line: raw, normalized, stanza_index, line_index, phonology, rhyme_word, rhyme_group, meter, stress, end_stopped, caesura, enjambment.
    """
    poesy_result = annotate_with_poesy(poem)
    poesy_ann = poesy_result.get("per_line", poesy_result) if isinstance(poesy_result, dict) and "per_line" in poesy_result else poesy_result
    poesy_meter_type = poesy_result.get("meter_type") if isinstance(poesy_result, dict) else None
    out = {
        "id": poem["id"],
        "author": poem["author"],
        "title": poem["title"],
        "stanzas": [],
    }
    for si, stanza in enumerate(poem["stanzas"]):
        lines = []
        rhyme_groups = []
        for li, line_dict in enumerate(stanza):
            norm = line_dict["normalized"]
            end_stopped = get_end_stopped(norm)
            rec = {
                "raw": line_dict["raw"],
                "normalized": norm,
                "stanza_index": si,
                "line_index": li,
            }
            pa = poesy_ann.get((si, li), {}) if isinstance(poesy_ann, dict) else {}
            rg = pa.get("rhyme_group")
            rg = rg if rg and str(rg).strip() and str(rg).strip() != "?" else None
            rec["end_stopped"] = end_stopped
            rec["caesura"] = get_caesura(norm)
            rec["enjambment"] = not end_stopped
            phon = get_phonology_for_line(norm)
            rec["phonology"] = phon
            rec["rhyme_word"] = get_rhyme_word(phon)
            rec["rhyme_group"] = rg
            rhyme_groups.append(rg)
            rec["meter"] = pa.get("meter")
            rec["stress"] = pa.get("stress")
            syll, stress_pattern, meter_type_fallback = line_meter_from_phonology(phon)
            if rec["meter"] is None or rec["meter"] == "":
                rec["meter"] = stress_pattern or ""
            if rec["stress"] is None or rec["stress"] == "":
                rec["stress"] = stress_pattern or ""
            rec["meter_type"] = poesy_meter_type if poesy_meter_type else meter_type_fallback
            lines.append(rec)
        n = len(stanza)
        stanza_type = STANZA_NAMES.get(n, f"{n}_line" if n else "variable")
        rhyme_scheme = "".join(str(rg) if rg else "-" for rg in rhyme_groups)
        pairs = _rhyme_pairs(rhyme_groups)
        out["stanzas"].append({
            "stanza_index": si,
            "stanza_type": stanza_type,
            "rhyme_scheme": rhyme_scheme,
            "rhyme_pairs": pairs,
            "lines": lines,
        })
    return out


SAMPLE_QUATRAIN = "bah18-w0160"   # A VOW TO FORTUNE. (4-line stanzas)
SAMPLE_COUPLET = "o5156-w1237"   # BOOK XII. Ep. 23. (2-line stanzas)


def main():
    print("Poesy (Prosodic): loaded")
    if HAS_PRONOUNCING:
        print("CMU (pronouncing): loaded")
    else:
        err = globals().get("_PRONOUNCING_ERR", "pip install pronouncing")
        print(f"CMU (pronouncing): no - {err}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for poem_id, label in [(SAMPLE_QUATRAIN, "quatrain"), (SAMPLE_COUPLET, "couplet")]:
        json_path = INPUT_DIR / f"{poem_id}.json"
        if not json_path.exists():
            print(f"Skip {label} ({poem_id}): not found")
            continue

        with open(json_path, encoding="utf-8") as f:
            poem = json.load(f)

        result = annotate_poem(poem)
        out_path = OUTPUT_DIR / f"{result['id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("=" * 60)
        print(f"Annotated [{label}]: {result['id']} | {result['title']}")
        print("=" * 60)
        for stanza in result["stanzas"]:
            si = stanza["stanza_index"]
            st_type = stanza.get("stanza_type", "")
            rhyme_scheme = stanza.get("rhyme_scheme", "")
            print(f"\n--- Stanza {si + 1} ({st_type}, {rhyme_scheme}) ---")
            for line in stanza["lines"]:
                li = line["line_index"]
                print(f"  [{li + 1}] {line['normalized']}")
                print(f"      rhyme_word: {line.get('rhyme_word')} | rhyme: {line['rhyme_group']} | meter_type: {line.get('meter_type', 'N/A')} | meter: {line.get('meter', 'N/A')} | end_stopped: {line['end_stopped']} | caesura: {line['caesura']} | enjambment: {line['enjambment']}")
                ph = line['phonology'][:5]
                print(f"      phonology: {ph}")
        print("\n" + "=" * 60)
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
