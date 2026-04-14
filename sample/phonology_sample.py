"""
Turn one normalized poem JSON into an annotated poem JSON.

What gets added per line: CMU-style pronunciations, stress and meter hints from the
Poesy library which wraps Prosodic, rhyme groups from Poesy when parsing succeeds, plus
simple rules for end-of-line punctuation (end-stopped vs run-on) and caesura (a mid-line
pause). 

Dependencies: pronouncing (CMU Pronouncing Dictionary), Poesy (Prosodic + rhyme), and
optionally espeak for words missing from CMU when not in batch mode.

Install (typical):
  brew install espeak
  pip install pronouncing
  pip install git+https://github.com/quadrismegistus/poesy

some changes made to handle stress
"""

import json
import re
import threading
from pathlib import Path

# Project root 
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "output/poems_normalized"
OUTPUT_DIR = ROOT / "output/poems_annotated"

def _patch_manager_for_hashstash():
    """Prosodic’s dependency chain can spawn a multiprocessing Manager at import time.

    sometimes that subprocess fails. Swaped in a tiny fake Manager so imports still succeed; 
    rhyme/meter code can run in the main process.
    """
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

# was getting errors about nltk data not being found, so added this
_nltk_data = ROOT / "data" / "nltk_data"
_nltk_data.mkdir(parents=True, exist_ok=True)
import os
os.environ.setdefault("NLTK_DATA", str(_nltk_data))


def _ensure_nltk_punkt():
    """Download NLTK punkt data if Prosodic/Poesy will run wasn't needed for CMU-only helper
    """
    try:
        import nltk
    except ImportError:
        return
    try:
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
    """Older Poesy code expects ``prosodic.config`` to exist.

    Newer Prosodic builds omit it; attach an empty dict so imports do not crash
    got an error when not attaching an empty dict
    """
    try:
        import prosodic
        if not hasattr(prosodic, "config"):
            prosodic.config = {}
    except Exception:
        pass


def _patch_prosodic_text_meter():
    """Bridge Poesy (older API) to Prosodic 2.x (stricter types).

    Poesy still passes meter names as strings and sometimes leaves poem-level meter as a
    string on the text object. Prosodic expects a real Meter object and will throw if it
    gets a bare string. wrap three hooks: Text construction, get_meter: if the
    inherited value is still a string, replace it with a default Meter, and parse if
    callers pass meter= as a string.  If this block
    fails silently  Poesy errors out 
    """
    import importlib
    import warnings

    try:
        import prosodic
        from prosodic.texts.texts import TextModel
        from prosodic.ents import Entity
        from prosodic.parsing import Meter

        # ``import prosodic.texts.texts as _m`` can raise ImportError on some Python/Prosodic
        # this was for error handling had to search to find this solution
        _texts_mod = importlib.import_module("prosodic.texts.texts")

        def _patched_text(txt="", fn="", lang=None, parent=None, children=None, tokens_df=None, **kwargs):
            kwargs.pop("meter", None)  # Poesy passes p.Text(line, meter=...); TextModel has no meter kw
            return TextModel(
                txt=txt, fn=fn, lang=lang, parent=parent,
                children=children or [], tokens_df=tokens_df, **kwargs
            )

        prosodic.Text = _patched_text
        _texts_mod.Text = _patched_text

        def _coerce_meter_value(res, meter_kwargs):
            """Poesy stores poem ``meter`` as a label string on ``text._mtr``; Prosodic needs Meter."""
            if isinstance(res, str) or (res is not None and not hasattr(res, "parse_unit")):
                m = Meter(**meter_kwargs) if meter_kwargs else Meter()
                return m
            return res

        # Entity.get_meter: Poesy passes meter= str to parse(); Prosodic 2.x expects a Meter instance.
        # Also, Poesy sets poem.meter to a string; get_meter() inherits self.text._mtr and returns
        _orig_get_meter = Entity.get_meter

        def _patched_get_meter(self, meter=None, **meter_kwargs):
            if isinstance(meter, str):
                meter = None
            res = _orig_get_meter(self, meter=meter, **meter_kwargs)
            res = _coerce_meter_value(res, meter_kwargs)
            if res is not getattr(self, "_mtr", None):
                self._mtr = res
                tx = getattr(self, "text", None)
                if tx is not None and isinstance(getattr(tx, "_mtr", None), str):
                    tx._mtr = res
            return res

        Entity.get_meter = _patched_get_meter

        # parse_iter: must never see meter as str (parse_unit). Sanitize poem text._mtr before get_meter.
        _orig_parse_iter = _texts_mod.TextModel.parse_iter

        def _patched_parse_iter(self, combine_by=None, num_proc=None, lim=None, force=False, meter=None, **meter_kwargs):
            if isinstance(meter, str):
                meter = None
            tx = getattr(self, "text", None)
            if tx is not None and isinstance(getattr(tx, "_mtr", None), str):
                tx._mtr = Meter(**meter_kwargs) if meter_kwargs else Meter()
            if isinstance(getattr(self, "_mtr", None), str):
                self._mtr = Meter(**meter_kwargs) if meter_kwargs else Meter()
            yield from _orig_parse_iter(
                self,
                combine_by=combine_by,
                num_proc=num_proc,
                lim=lim,
                force=force,
                meter=meter,
                **meter_kwargs,
            )

        _texts_mod.TextModel.parse_iter = _patched_parse_iter

        # parse: Poesy passes meter= string; Prosodic expects Meter object
        _orig_parse = _texts_mod.TextModel.parse

        def _patched_parse(self, combine_by=None, num_proc=None, force=False, lim=None, meter=None, **meter_kwargs):
            if isinstance(meter, str):
                meter = None
            return _orig_parse(
                self,
                combine_by=combine_by, num_proc=num_proc, force=force,
                lim=lim, meter=meter, **meter_kwargs
            )

        _texts_mod.TextModel.parse = _patched_parse

        # Poesy calls ``text.words()`` on each ``p.Text(line, ...)`` object. In Prosodic 2.x,
        # ``words`` is usually resolved via ``Entity.__getattr__`` to a ``WordTokenList``
        # instance, so ``text.words`` is that list and ``text.words()`` crashes (not callable).
        def _textmodel_words(self):
            return self.wordtokens

        _texts_mod.TextModel.words = _textmodel_words  # type: ignore[assignment]

        # For ``Line`` objects (``WordTokenList``), the old API was ``.words()`` too.
        from prosodic.words.wordtokenlist import WordTokenList

        def _words_callable(self):
            return self

        WordTokenList.words = _words_callable  # type: ignore[assignment]
    except Exception as e:
        warnings.warn(f"prosodic/poesy compatibility patch failed: {e}", RuntimeWarning)


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

# Poesy is imported lazily inside ``annotate_with_poesy`` so that CMU-only callers
# (e.g. ``get_phonology_for_line`` / ``evaluation.form_eval_generation`` on HPC) do not
# require ``pip install poesy`` for prompt baselines.


def _arpabet_from_espeak(word: str) -> list:
    """Fallback pronunciation when CMU does not know a word.

    Calls the ``espeak`` command-line tool to get IPA, then maps common symbols into a rough
    ARPAbet-like phone list. Returns an empty list if espeak is
    missing or the word cannot be handled.
    """
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
    """Return how a single token is pronounced, plus where that guess came from.

    Order of attempts: (1) CMU Pronouncing Dictionary via ``pronouncing``, (2) same word with
    apostrophes stripped if that fails, (3) espeak fallback

    Returns:
        Tuple of ``(phones, source)`` where ``phones`` is a list of ARPAbet strings (one
        pronunciation variant) and ``source`` is ``"cmudict"``, ``"espeak"``, or ``"not_found"``.
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
    """Split a line into word tokens and attach CMU (or fallback) phones to each.

    Output is a list of dicts: ``{"word": str, "arpabet": list of str, "source": str}``. Those
    ARPAbet strings carry **lexical** stress digits on vowels; later turn them into a stress
    pattern with ``phones_to_stress`` / ``line_meter_from_phonology`` when need a fallback.
    """
    words = re.findall(r"[\w']+", line_text)
    out = []
    for w in words:
        phones, src = get_arpabet(w)
        out.append({"word": w, "arpabet": phones, "source": src})
    return out


def get_rhyme_word(phonology: list):
    """
    The word that carries the rhyme: last content word
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
    """Read stress numbers from CMU dictionary phones (lexical stress only).

    CMU marks each vowel with a digit at the end of the phone: 0 = unstressed, 1 = main stress,
    2 = secondary stress. Consonants have no stress digit, so skip them. This is *dictionary*
    stress (how the word is usually said), not yet a full poetic scansion.
    """
    stresses = []
    for ph in (phones or []):
        p = ph if isinstance(ph, str) else (ph[0] if isinstance(ph, (list, tuple)) and ph else None)
        if p and len(p) > 1 and p[-1] in "012":
            stresses.append(int(p[-1]))
    return stresses


# Human-readable foot names when  guess meter from syllable count alone.
FEET_NAMES = {1: "monometer", 2: "dimeter", 3: "trimeter", 4: "tetrameter", 5: "pentameter",
              6: "hexameter", 7: "heptameter", 8: "octameter"}


def line_meter_from_phonology(phonology: list) -> tuple:
    """Build a stress string and a rough meter *label* using only CMU pronunciations.

    **What this evaluates:** For each word  already looked up in CMU,  collect stress on
    every vowel in order (left to right through the line). That gives a simple unstressed/stressed
    pattern for the whole line.

    **What this is *not*:** It is not Prosodic’s metrical parse. It ignores poetry-specific
    effects (elision, promoted beats, headless lines). Use this as a fallback when Poesy/Prosodic
    did not give us a ``stress`` string.

    Turn CMU’s 0/1/2 into a binary string: any stressed vowel (1 or 2) becomes ``1``,
    unstressed (0) becomes ``0``. then guess a friendly label like “iambic pentameter” only
    when the syllable count and the start of the pattern match simple templates.

    Returns:
        ``(syllable_count, stress_binary_string, meter_type_label)``. If there are no vowels,
        returns ``(0, "", "unknown")``.
    """
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
        # Collect one stress level per vowel phone, in order, for this line.
        stresses.extend(phones_to_stress(pron))
    # Collapse to 0/1 for downstream code that expects binary weak/strong.
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


def get_caesura(line_text: str) -> dict:
    """Find the first plausible caesura: a comma-like pause between two words mid-line.

    tokenize into words and punctuation. The first mark in the set comma, semicolon, colon,
    em-dash, or hyphen that sits between two real words counts as a hit. store the word
    index before the break and the words on either side. If nothing matches, every value in
    the returned dict is None.

    Returns:
        Dict with keys index, punct, before, after (each an int/str or None).
    """
    toks = re.findall(r"[\w']+|[^\w\s]", line_text)
    word_i = -1
    for i, tok in enumerate(toks):
        if re.match(r"[\w']+$", tok):
            word_i += 1
            continue
        if tok not in ",;:—-":
            continue
        # find neighboring words around punctuation
        before = after = None
        b = i - 1
        while b >= 0:
            if re.match(r"[\w']+$", toks[b]):
                before = toks[b]
                break
            b -= 1
        a = i + 1
        while a < len(toks):
            if re.match(r"[\w']+$", toks[a]):
                after = toks[a]
                break
            a += 1
        if before and after:
            return {"index": word_i, "punct": tok, "before": before, "after": after}
    return {"index": None, "punct": None, "before": None, "after": None}


def build_poem_string(poem: dict) -> str:
    """Flatten a poem dict into one string for Poesy: one verse line per text line.

    Blank lines separate stanzas. If MAX_LINE_CHARS is a positive integer, lines longer than
    that are cut from the left and prefixed with an ellipsis so the  rhyme word
    stays intact for Prosodic.
    """
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
    """Debug helper for Poesy: prints to stderr unless ``POESY_DEBUG=0`` (batch runs often disable it)."""
    if os.environ.get("POESY_DEBUG", "1") != "0":
        print(f"[poesy-debug] {msg}", file=__import__("sys").stderr)


def _best_parse_from_prosodic_line(line_obj) -> object:
    """Pick the best Prosodic ``Parse`` for one line (Prosodic 2.x ``parses[0].best_parse`` or older API)."""
    parses = getattr(line_obj, "parses", None)
    if parses is not None:
        try:
            if len(parses) > 0:
                pl0 = parses[0]
                bp = getattr(pl0, "best_parse", None)
                if bp is not None:
                    return bp
        except (TypeError, IndexError, AttributeError):
            pass
    for attr in ("bestParses", "best_parses"):
        val = getattr(line_obj, attr, None)
        if val is not None:
            bps = val() if callable(val) else val
            if hasattr(bps, "__iter__") and not isinstance(bps, str):
                bps = list(bps) if bps else []
                return bps[0] if bps else None
    return None


def _pipe_meter_from_parse(bp) -> str:
    """Turn Parse ``txt`` (space-separated scansion tokens) into ``a|B|c`` like stored JSON examples."""
    if bp is None:
        return ""
    t = getattr(bp, "txt", None)
    if not (t and isinstance(t, str)):
        return ""
    parts = t.split()
    return "|".join(parts) if parts else ""


def annotate_with_poesy(poem: dict) -> dict:
    """Run Poesy’s Poem parser to get rhyme groups and Prosodic line parses for one poem.

    Poesy wraps Prosodic: it builds a metrical parse and a rhyme graph. Read, for each line,
    the best available stress string and meter string from the Prosodic line object, plus a
    rhyme label per line position. Tries to read an overall meter label for the poem from Poesy’s stats when present.

    If anything in that stack throws version mismatch, huge poem timeout,  return
    empty annotations and ``annotate_poem`` falls back to CMU-only fields.
        MAX_POEM_LINES: if set and the poem has more lines, skip Poesy entirely (batch safety).

    Returns:
        Dict with keys:
            ``per_line``: map ``(stanza_index, line_index)`` -> ``{"rhyme_group", "meter", "stress"}``
            (values may be None where missing).
            ``meter_type``: poem-level string or None.
    """
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
        from poesy import Poem as PoesyPoem

        _log_poesy(f"building Poem from {len(text)} chars, {text.count(chr(10))} newlines")
        p = PoesyPoem(text)
        _log_poesy("calling p.parse()...")
        p.parse()  # trigger metrical parsing
        _log_poesy("accessing p.rhymes (triggers rhyme_net)...")
        try:
            rhymes_raw = p.rhymes  # trigger rhyme_net computation
        except Exception as er:
            _log_poesy(
                f"p.rhymes/rhyme_net failed (Poesy vs Prosodic 2.x API): {type(er).__name__}: {er}; "
                "continuing without rhyme groups"
            )
            rhymes_raw = {}
        _log_poesy(f"p.rhymes type={type(rhymes_raw).__name__}, len={len(rhymes_raw) if rhymes_raw else 0}")
        if rhymes_raw:
            sample = list(rhymes_raw.items())[:5]
            _log_poesy(f"p.rhymes sample: {sample}")
        keys_sorted = sorted(p.lined.keys())  # line order 
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
            bp = _best_parse_from_prosodic_line(line_obj)
            m_raw = None
            stress = None
            if bp is not None:
                # **Metrical stress (Prosodic):** the parser picks a best scan for the line.
                # Try several attribute names across Prosodic versions; some are methods.
                sget = (
                    getattr(bp, "stress_str", None)
                    or getattr(bp, "stress", None)
                    or getattr(bp, "parse_stress", None)
                    or getattr(bp, "parse_stress_str", None)
                )
                if callable(sget):
                    sget = sget()
                stress = sget if isinstance(sget, str) else None
                m_raw = getattr(bp, "parse_meter", None)
                if isinstance(m_raw, str) and m_raw.strip():
                    pass
                else:
                    m_raw = _pipe_meter_from_parse(bp)
                    if not m_raw:
                        ms = getattr(bp, "meter_str", None)
                        if callable(ms):
                            try:
                                m_raw = ms("|")
                            except TypeError:
                                m_raw = ms()
                        elif isinstance(ms, str) and ms.strip():
                            if all(c in "+-" for c in ms.strip()):
                                m_raw = None
                            else:
                                m_raw = ms
                _m = getattr(bp, "meter", None)
                if _m is not None and isinstance(_m, str) and not m_raw:
                    m_raw = _m
            meter = m_raw if (m_raw is not None and isinstance(m_raw, str) and m_raw.strip()) else parse_str
            # still have no stress string, derive it from Prosodic’s pipe-separated scan:
            # uppercase letters in a slot mean “stressed” for that metrical position.
            if not stress and parse_str and isinstance(parse_str, str) and "|" in parse_str:
                stress = _parse_str_to_stress(parse_str)
            if not stress and meter and isinstance(meter, str) and "|" in meter:
                stress = _parse_str_to_stress(meter)
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
                    m_str = None if m is None else (m if isinstance(m, str) else str(m))
                    ann[(si, li)]["meter"] = m_str
                    s_str = None if s is None else (s if isinstance(s, str) else str(s))
                    # Last chance: build +/− from the pipe scansion stored as ``meter`` text.
                    if not s_str and m_str and "|" in m_str:
                        s_str = _parse_str_to_stress(m_str)
                    ann[(si, li)]["stress"] = s_str or None
                pos += 1
        non_empty_rhymes = sum(1 for v in ann.values() if v.get("rhyme_group") and str(v.get("rhyme_group", "")).strip() not in ("", "?"))
        _log_poesy(f"ann built: {len(ann)} entries, {non_empty_rhymes} non-empty rhyme_groups")
        # meter_type from Poesy: statd has meter_type_scheme  + beat_scheme_repr 
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
        # Manager subprocess fails with Permission this was the error message I had before so log incase it happens again
        _log_poesy("ROOT_CAUSE: (1) hashstash Manager fails in restricted envs; (2) Poesy/Prosodic 2.x API mismatch")
        _log_poesy("  (meter=str vs Meter object) -> parse fails before rhyme_net runs.")
        import warnings
        warnings.warn(f"Poesy annotation failed: {e}")
        return {"per_line": {}, "meter_type": None}


# Short names for stanza sizes  label stanza_type in the output JSON.
STANZA_NAMES = {2: "couplet", 4: "quatrain", 6: "sestet", 8: "octet", 3: "tercet", 5: "cinquain"}


def _parse_str_to_stress(parse_str: str) -> str:
    """Turn Prosodic’s pipe-separated line into a simple strong/weak beat string.

    **Rule:** Split on ``|``. For each piece, if any letter in that piece is uppercase, treat
    that metrical position as stressed (``+``); otherwise weak (``-``). Example: ``a|B`` → ``-+``.
    recover stress when Prosodic only gives us the scanned text, not a ready-made
    ``stress_str``. Empty input returns ``""``.
    """
    if not parse_str or not isinstance(parse_str, str):
        return ""
    parts = parse_str.split("|")
    return "".join("+" if any(c.isupper() for c in p if c.isalpha()) else "-" for p in parts if p.strip())


def normalize_stress_to_plus_minus(s: str) -> str:
    """Make sure stored ``stress`` uses only ``+`` and ``-`` so all tasks agree.

    Different steps above may produce ``+``/``-`` (Prosodic) or ``0``/``1`` (CMU-style binary).
    convert the latter here so one line’s ``stress`` field always looks the same in JSON.
    """
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    if all(c in "+-" for c in s):
        return s
    if all(c in "01" for c in s):
        return "".join("+" if c == "1" else "-" for c in s)
    return s


# ---  Single-line Prosodic stress (PROSODIC_NOTEBOOK_STRESS mode)
_NOTEBOOK_STYLE_METER = None
_PROSODIC_LOG_QUIET = False


def _quiet_prosodic_logging_once() -> None:
    """Turn down Prosodic’s log noise -> lets batch log be simple"""
    global _PROSODIC_LOG_QUIET
    if _PROSODIC_LOG_QUIET:
        return
    try:
        import logging
        import prosodic

        if hasattr(prosodic, "logger"):
            prosodic.logger.setLevel(logging.ERROR)
        logging.getLogger("prosodic").setLevel(logging.ERROR)
    except Exception:
        pass
    _PROSODIC_LOG_QUIET = True


def _metricalgpt_notebook_meter():
    """Return the default bundled meter (Prosodic 1.x: ``prosodic.METER``, “Ryan’s Meter”).

    Older code used ``Meter(constraints=...)`` (Prosodic 2.x-style); this environment expects
    ``Meter(config)`` or the prebuilt ``prosodic.METER`` singleton.
    """
    global _NOTEBOOK_STYLE_METER
    if _NOTEBOOK_STYLE_METER is not None:
        return _NOTEBOOK_STYLE_METER
    import prosodic

    _quiet_prosodic_logging_once()
    _NOTEBOOK_STYLE_METER = prosodic.METER
    return _NOTEBOOK_STYLE_METER


def _text_lines_list(text_obj) -> list:
    """``Text.lines`` may be a method (this repo’s Prosodic) or a list in other versions."""
    lines = getattr(text_obj, "lines", None)
    if lines is None:
        return []
    return lines() if callable(lines) else (lines if isinstance(lines, list) else list(lines))


def _line_best_parse(line_obj):
    """Best parse for one line: ``bestParse`` may be method or attribute."""
    bp = getattr(line_obj, "bestParse", None)
    if bp is None:
        bp = getattr(line_obj, "best_parse", None)
    if bp is None:
        return None
    return bp() if callable(bp) else bp


def _parse_stress_from_prosodic_parse(bp) -> str:
    """Derive ``+``/``-`` from a Prosodic ``Parse`` (pipe- or dot-separated scansion tokens)."""
    if bp is None:
        return ""
    stress = getattr(bp, "stress_str", None) or getattr(bp, "parse_stress", None)
    if callable(stress):
        try:
            stress = stress()
        except TypeError:
            stress = stress()
    if isinstance(stress, str) and stress.strip() and all(c in "+-" for c in stress.strip()):
        return stress.strip()
    parse_str = getattr(bp, "parse_str", None)
    if callable(parse_str):
        try:
            parse_str = parse_str(viols=False)
        except TypeError:
            parse_str = parse_str()
    if not (parse_str and isinstance(parse_str, str)):
        parse_str = getattr(bp, "posString", None)
        if callable(parse_str):
            parse_str = parse_str()
    if not (parse_str and isinstance(parse_str, str) and parse_str.strip()):
        parse_str = str(bp).strip()
    if not parse_str:
        return ""
    return _parse_str_to_stress(parse_str)


def stress_from_notebook_style_prosodic(line_text: str) -> str:
    """Get metrical stress for one line using Prosodic alone (optional path).

    Uses ``prosodic.Text`` + default ``prosodic.METER`` (matches the Prosodic build in this
    project’s venv). Does not use Poesy’s rhyme graph. On failure, returns ``""``.

    **Prosodic 2.x:** If ``prosodic.texts.texts.TextModel`` exists, it is tried as a fallback
    after the ``Text`` path.

    Returns ``+``/``-`` string, or ``""`` if Prosodic fails or the line exceeds
    ``PROSODIC_NOTEBOOK_MAX_CHARS``.
    """
    s = (line_text or "").strip()
    if not s:
        return ""
    try:
        max_chars = int(os.environ.get("PROSODIC_NOTEBOOK_MAX_CHARS", "2000") or "2000")
    except ValueError:
        max_chars = 2000
    if len(s) > max_chars:
        return ""

    _quiet_prosodic_logging_once()

    def _from_text_api() -> str:
        import prosodic

        meter = _metricalgpt_notebook_meter()
        t = prosodic.Text(s)
        t.parse(meter=meter, line_lim=1)
        lines = _text_lines_list(t)
        if not lines:
            return ""
        bp = _line_best_parse(lines[0])
        return _parse_stress_from_prosodic_parse(bp)

    def _from_textmodel_api() -> str:
        try:
            from prosodic.texts.texts import TextModel
        except Exception:
            return ""
        meter = _metricalgpt_notebook_meter()
        out = TextModel(txt=s).parse(meter=meter, lim=1)
        if not out:
            return ""
        pl = out[0]
        bp = getattr(pl, "best_parse", None)
        if bp is None:
            return ""
        return _parse_stress_from_prosodic_parse(bp)

    try:
        stress = _from_text_api()
        if stress:
            return stress
        stress = _from_textmodel_api()
        return stress if stress else ""
    except Exception:
        return ""


def _rhyme_pairs(rhyme_groups: list) -> list:
    """List undirected pairs of line indices that share the same non-empty rhyme label.
    Input is parallel to lines in one stanza: each entry is a rhyme group id 
    Used only to store ``rhyme_pairs`` in the stanza record for convenience.
    """
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
    """Main entry: take a normalized poem and return the same poem shape with labels filled in.

    Steps in order: call ``annotate_with_poesy`` for rhyme + Prosodic **meter** (pipe scansion)
    when possible; for each line, attach CMU phonology, rhyme word, end-stopped and caesura
    flags, and enjambment.

    **Stress (two modes):** If ``PROSODIC_NOTEBOOK_STRESS`` is set to ``1``, ``force``, ``yes``,
    ``true``, or ``prefer``, the **only** source for line ``stress`` is
    ``stress_from_notebook_style_prosodic`` (single-line Prosodic parse with the fixed notebook
    ``Meter``). Poesy line stress and CMU lexical stress are **not** used for ``stress`` in
    that mode. If the env var is unset, ``stress`` comes from Poesy first, then CMU fallback.

    Line-level ``meter_type`` uses Poesy’s poem-wide label if possible; otherwise the heuristic
    from ``line_meter_from_phonology`` (syllable-count / pattern label—still computed for the
    label even in notebook-stress mode).

    Field roles: ``meter`` = Poesy/Prosodic pipe scansion when available; ``stress`` = ``+``/``-``
    beats; ``meter_type`` = abstract label. CMU lexical stress never fills ``meter``—only the
    ``stress`` field in default mode when Poesy omits it.

    The counter dict ``annotation_sources`` is for checking, not rigorous stats.

    Returns:
        Dict with ``id``, ``author``, ``title``, and ``stanzas``. Each stanza has index,
        type label, rhyme scheme string, rhyme_pairs list, and ``lines`` with the per-field
        records described in your export script.
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
    # Track sources: Poesy vs fallback (phonology/CMU)
    src = {"stress_poesy": 0, "stress_empty": 0, "stress_notebook": 0, "meter_poesy": 0, "meter_empty": 0,
           "meter_phonology": 0, "rhyme_poesy": 0, "rhyme_empty": 0, "meter_type_poesy": 0, "meter_type_phonology": 0}
    track_sources = os.environ.get("ANNOTATION_SOURCES", "1") == "1"

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
            c = get_caesura(norm)
            rec["caesura"] = c["index"]
            rec["caesura_punct"] = c["punct"]
            rec["caesura_before"] = c["before"]
            rec["caesura_after"] = c["after"]
            rec["enjambment"] = not end_stopped
            phon = get_phonology_for_line(norm)
            rec["phonology"] = phon
            rec["rhyme_word"] = get_rhyme_word(phon)
            rec["rhyme_group"] = rg
            rhyme_groups.append(rg)
            rec["meter"] = pa.get("meter")
            # ``meter_type`` fallback label still needs syllable/heuristic info from phonology.
            syll, stress_pattern, meter_type_fallback = line_meter_from_phonology(phon)

            nb_mode = (os.environ.get("PROSODIC_NOTEBOOK_STRESS") or "").strip().lower()
            # Notebook mode: ONLY ``stress_from_notebook_style_prosodic`` may set ``stress`` (not
            # Poesy line stress, not CMU). Export PROSODIC_NOTEBOOK_STRESS=1|force|yes|prefer|true.
            notebook_only_stress = nb_mode in ("1", "force", "yes", "true", "prefer")

            if notebook_only_stress:
                nb = stress_from_notebook_style_prosodic(norm)
                rec["stress"] = nb or ""
                if track_sources:
                    if rec["stress"]:
                        src["stress_notebook"] += 1
                    else:
                        src["stress_empty"] += 1
            else:
                rec["stress"] = pa.get("stress")
                if not (rec.get("stress") or "").strip():
                    rec["stress"] = stress_pattern or ""
                if track_sources:
                    if rec["stress"]:
                        src["stress_poesy"] += 1
                    else:
                        src["stress_empty"] += 1

            # Final shape for JSON: always ``+``/``-`` per beat (never mix with ``0``/``1``).
            rec["stress"] = normalize_stress_to_plus_minus(rec.get("stress") or "")

            if track_sources:
                if rg:
                    src["rhyme_poesy"] += 1
                else:
                    src["rhyme_empty"] += 1
            if rec["meter"] is None or rec["meter"] == "":
                rec["meter"] = ""
                if track_sources:
                    src["meter_empty"] += 1
            else:
                if track_sources:
                    src["meter_poesy"] += 1
        
            # When Poesy/Prosodic fails, leave stress empty; meter_only training filters these out.
            rec["meter_type"] = poesy_meter_type if poesy_meter_type else meter_type_fallback
            if track_sources:
                if poesy_meter_type:
                    src["meter_type_poesy"] += 1
                else:
                    src["meter_type_phonology"] += 1
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
    if track_sources:
        out["annotation_sources"] = src
    return out


SAMPLE_QUATRAIN = "bah18-w0160"   # A VOW TO FORTUNE. (4-line stanzas)
SAMPLE_COUPLET = "o5156-w1237"   # BOOK XII. Ep. 23. (2-line stanzas)


def main():
    """Small demo: load two built-in sample poem IDs from ``output/poems_normalized`` and print labels.

    Writes matching JSON under ``output/poems_annotated``. Handy for a quick smoke test after
    changing this module; batch jobs use ``batch/phonology_batch.py`` instead.
    """
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
                print(
                    f"      rhyme_word: {line.get('rhyme_word')} | rhyme: {line['rhyme_group']} "
                    f"| meter_type: {line.get('meter_type', 'N/A')} | meter: {line.get('meter', 'N/A')} "
                    f"| stress: {line.get('stress', 'N/A')} "
                    f"| end_stopped: {line['end_stopped']} | caesura: {line['caesura']} "
                    f"({line.get('caesura_before')} {line.get('caesura_punct')} {line.get('caesura_after')}) "
                    f"| enjambment: {line['enjambment']}"
                )
                ph = line['phonology'][:5]
                print(f"      phonology: {ph}")
        print("\n" + "=" * 60)
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
