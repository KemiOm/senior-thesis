#!/usr/bin/env python3
"""
Run prompt-only baseline: load test lines, build prompts, call models, save outputs.

Supports:
  - Encoder-decoder (seq2seq): T5, FLAN-T5, BART, PEGASUS, etc. (--model_type seq2seq, default)
  - Decoder-only (causal): GPT-2, Pythia, LLaMA, Mistral, etc. (--model_type causal)

Few-shot prompts use a full example stanza (quatrain): each line is paired with its label in the
same format as training (from corpus.db), mirroring line-level meter/rhyme supervision.

Usage:
  python scripts/run_prompt_baseline.py --model google/flan-t5-large --prompt zero_shot --task meter_only
  python scripts/run_prompt_baseline.py --model google/flan-t5-large --prompt few_shot --task meter_only
  python scripts/run_prompt_baseline.py --model gpt2-medium --model_type causal --prompt few_shot --task meter_only --n 100

Requires: pip install transformers torch (or pip install transformers accelerate)

Output: evaluation/results/baselines/prompt_only/{model_slug}/{prompt_type}_{task}.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "output" / "training_data"
RESULTS_DIR = ROOT / "evaluation" / "results" / "baselines" / "prompt_only"

# Few-shot: we generate an example quatrain dynamically from `output/corpus.db`,
# so the example line labels (stress/rhyme/combined tokens) exactly match the
# same supervision logic used to build `output/training_data/...`.
FEW_SHOT_SEPARATOR = "\n\n---\n\n"
_EXAMPLE_QUATRAIN = None


def stress_to_plus_minus(s: str) -> str:
    """Same as notebooks/01_prepare_training_data.ipynb: corpus stress -> +/- for meter_only targets."""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    if len(s) < 5:
        return ""
    if all(c in "+-" for c in s):
        return s
    if all(c in "01" for c in s):
        return "".join("+" if c == "1" else "-" for c in s)
    return ""


def rhyme_key_from_phonology(phonology_json: str) -> str:
    """
    Derive rhyme key from phonology using the same logic as
    `notebooks/01_prepare_training_data.ipynb`.
    Returns '' if unavailable.
    """
    if not phonology_json:
        return ""
    try:
        ph = json.loads(phonology_json)
    except Exception:
        return ""
    phones = []
    for p in ph:
        arp = p.get("arpabet")
        if isinstance(arp, list) and arp:
            phones.extend(arp[0].split() if isinstance(arp[0], str) else arp[0])
        elif isinstance(arp, str):
            phones.extend(arp.split())
    if not phones:
        return ""
    # Last stressed vowel onwards (ARPAbet vowel stress digits: 0/1/2).
    for i in range(len(phones) - 1, -1, -1):
        p = phones[i]
        if len(p) > 1 and p[-1] in "12":
            return " ".join(phones[i:])
    return " ".join(phones[-3:]) if len(phones) >= 3 else " ".join(phones)


def _caesura_tok(caesura) -> str:
    return str(caesura) if caesura is not None else "-"


def input_normalize_key(s: str) -> str:
    """Match training JSON `input` to corpus.db `normalized` (whitespace-normalized)."""
    return " ".join((s or "").strip().split())


MIN_STRESS_PATTERN_LEN = 5


def line_row_to_labels(row: tuple) -> dict:
    """
    One DB line row -> label dict (same fields as former make_label_dict in get_example_quatrain).
    row: (line_index, normalized, stress, meter_type, rhyme_group, phonology, end_stopped, caesura)
    """
    (
        line_index,
        normalized,
        stress,
        meter_type,
        rhyme_group,
        phonology,
        end_stopped,
        caesura,
    ) = row
    stress_raw = (stress or "").strip()
    stress_pm = stress_to_plus_minus(stress_raw)
    rhyme_key = rhyme_key_from_phonology(phonology or "")
    rg = rhyme_group or ""
    rhyme_only_target = rhyme_key if rhyme_key else (rg if rg and rg != "-" else "none")

    meter_tok = stress_raw or (meter_type or "unknown")
    rg_combined = rhyme_group or "-"
    rhyme_tok = rhyme_key if rhyme_key else (rg_combined if rg_combined != "-" else "none")
    end_tok = 1 if end_stopped else 0
    caes_tok = _caesura_tok(caesura)
    combined_target = f"meter:{meter_tok}|rhyme:{rhyme_tok}|end:{end_tok}|caesura:{caes_tok}"

    # Prefer stress string in the combined meter slot when available.
    combined_target_stress = (
        f"meter:{stress_pm}|rhyme:{rhyme_tok}|end:{end_tok}|caesura:{caes_tok}" if stress_pm else combined_target
    )

    return {
        "line_index": line_index,
        "normalized": (normalized or "").strip(),
        "stress": stress_raw,
        "stress_pm": stress_pm,
        "meter_type": (meter_type or "").strip(),
        "rhyme_key": rhyme_key,
        "rhyme_group": rhyme_group,
        "rhyme_only_target": rhyme_only_target,
        "combined_target": combined_target,
        "combined_target_stress": combined_target_stress,
        "end_tok": end_tok,
        "caes_tok": caes_tok,
    }


def line_ok_for_strict_quatrain(d: dict) -> bool:
    """All four lines in a stanza must pass this for --strict-eval membership."""
    if not d.get("stress_pm") or len(d["stress_pm"]) < MIN_STRESS_PATTERN_LEN:
        return False
    mt = (d.get("meter_type") or "").strip().lower()
    if not mt or mt == "unknown":
        return False
    if d.get("rhyme_only_target") == "none":
        return False
    return True


def iter_quatrain_line_rows(conn):
    """Yields list of 4 row tuples for each stanza that has exactly 4 lines."""
    cur = conn.execute(
        """
        SELECT poem_id, stanza_index FROM lines
        GROUP BY poem_id, stanza_index
        HAVING COUNT(*) = 4
        ORDER BY poem_id, stanza_index
        """
    )
    for poem_id, stanza_index in cur.fetchall():
        rows = conn.execute(
            """
            SELECT line_index, normalized, stress, meter_type, rhyme_group, phonology, end_stopped, caesura
            FROM lines
            WHERE poem_id = ? AND stanza_index = ?
            ORDER BY line_index
            """,
            (poem_id, stanza_index),
        ).fetchall()
        if len(rows) == 4:
            yield rows


def build_unambiguous_label_map(conn) -> dict[str, dict]:
    """normalized key -> labels; keys with multiple DB rows (any mismatch) are omitted."""
    from collections import defaultdict

    buckets: dict[str, list] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT line_index, normalized, stress, meter_type, rhyme_group, phonology, end_stopped, caesura
        FROM lines
        WHERE normalized IS NOT NULL AND TRIM(normalized) != ''
        """
    ):
        d = line_row_to_labels(row)
        k = input_normalize_key(d["normalized"])
        if not k:
            continue
        buckets[k].append(d)

    out: dict[str, dict] = {}
    for k, group in buckets.items():
        if len(group) == 1:
            out[k] = group[0]
            continue
        # Same stress_pm + rhyme + combined_stress → treat as unambiguous
        sigs = {
            (
                g["stress_pm"],
                g["rhyme_only_target"],
                g["combined_target_stress"],
            )
            for g in group
        }
        if len(sigs) == 1:
            out[k] = group[0]
    return out


def build_strict_quatrain_input_keys(conn) -> set[str]:
    """Input keys that appear in at least one fully reliable 4-line stanza."""
    keys: set[str] = set()
    for rows in iter_quatrain_line_rows(conn):
        lines = [line_row_to_labels(r) for r in rows]
        if not all(line_ok_for_strict_quatrain(l) for l in lines):
            continue
        for l in lines:
            k = input_normalize_key(l["normalized"])
            if k:
                keys.add(k)
    return keys


def is_reliable_gold_json(task: str, target: str) -> bool:
    """Filter using JSON target only (no DB). Used when --gold-filter reliable without --strict-eval."""
    t = (target or "").strip()
    if not t:
        return False
    if task == "meter_only":
        if all(c in "+-" for c in t):
            return len(t) >= MIN_STRESS_PATTERN_LEN
        pm = stress_to_plus_minus(t)
        return len(pm or "") >= MIN_STRESS_PATTERN_LEN
    if task == "rhyme_only":
        return t.lower() != "none"
    if task == "natural_text":
        return True
    if task == "combined":
        m = re.match(r"meter:([^|]+)\|rhyme:([^|]+)\|", t)
        if not m:
            return False
        meter_part, rhyme_part = m.group(1).strip(), m.group(2).strip()
        if not meter_part or meter_part.lower() == "unknown":
            return False
        if rhyme_part.lower() in ("none", ""):
            return False
        mp = meter_part
        if all(c in "+-" for c in mp):
            return len(mp) >= MIN_STRESS_PATTERN_LEN
        if all(c in "01" for c in mp):
            pm = stress_to_plus_minus(mp)
            return len(pm or "") >= MIN_STRESS_PATTERN_LEN
        return len(meter_part) >= MIN_STRESS_PATTERN_LEN
    return True


def apply_regold_target(task: str, d: dict) -> str | None:
    """Gold string from DB label dict; None if task requirements not met."""
    if not d.get("stress_pm") or len(d["stress_pm"]) < MIN_STRESS_PATTERN_LEN:
        return None
    if (d.get("meter_type") or "").strip().lower() in ("", "unknown"):
        return None
    if d.get("rhyme_only_target") == "none":
        return None
    if task == "meter_only":
        return d["stress_pm"]
    if task == "rhyme_only":
        return d["rhyme_only_target"]
    if task == "natural_text":
        return None  # keep JSON target
    if task == "combined":
        return d["combined_target_stress"]
    return None


def filter_and_regold_data(
    data: list[dict],
    task: str,
    *,
    strict_eval: bool,
    gold_filter: str,
    corpus_db: Path,
) -> tuple[list[dict], dict]:
    """
    Returns (filtered_data, meta) where meta has counts and flags for JSON sidecar.
    """
    meta = {
        "strict_eval": strict_eval,
        "gold_filter": gold_filter,
        "n_input": len(data),
        "n_after": 0,
        "n_dropped_no_key": 0,
        "n_dropped_ambiguous_regold": 0,
    }
    if not strict_eval and gold_filter == "none":
        meta["n_after"] = len(data)
        return data, meta

    if gold_filter == "reliable" and not strict_eval:
        out: list[dict] = []
        for item in data:
            if not is_reliable_gold_json(task, item.get("target", "")):
                meta["n_dropped_no_key"] += 1
                continue
            out.append(dict(item))
        meta["n_after"] = len(out)
        return out, meta

    import sqlite3

    if not corpus_db.exists():
        raise FileNotFoundError(
            f"--strict-eval requires {corpus_db}. Run export_sqlite.py first."
        )
    conn = sqlite3.connect(corpus_db)
    try:
        label_map = build_unambiguous_label_map(conn)
        strict_keys = build_strict_quatrain_input_keys(conn)
    finally:
        conn.close()

    out = []
    for item in data:
        k = input_normalize_key(item.get("input", item.get("line", "")))

        if not k or k not in strict_keys:
            meta["n_dropped_no_key"] += 1
            continue
        d = label_map.get(k)
        if d is None:
            meta["n_dropped_ambiguous_regold"] += 1
            continue
        if task == "natural_text":
            out.append(dict(item))
            continue
        new_target = apply_regold_target(task, d)
        if new_target is None:
            meta["n_dropped_ambiguous_regold"] += 1
            continue
        new_item = dict(item)
        new_item["target"] = new_target
        out.append(new_item)

    meta["n_after"] = len(out)
    return out, meta


def get_example_quatrain_lines():
    """
    Choose one deterministic stanza (first that satisfies constraints) and
    return 4 lines with precomputed labels for:
      - meter_only ICL: stress as +/- via stress_to_plus_minus (same as training targets)
      - rhyme_only: rhyme_key if available else rhyme_group else 'none'
      - combined target: meter/token/rhyme/end/caesura bundled string

    This is used only for prompt construction (few-shot/one-shot).
    """
    global _EXAMPLE_QUATRAIN
    if _EXAMPLE_QUATRAIN is not None:
        return _EXAMPLE_QUATRAIN

    corpus_db = ROOT / "output" / "corpus.db"
    if not corpus_db.exists():
        raise FileNotFoundError(
            f"Required for example prompts: {corpus_db}. Run export_sqlite.py first."
        )

    import sqlite3

    conn = sqlite3.connect(corpus_db)
    chosen = None
    for rows in iter_quatrain_line_rows(conn):
        lines = [line_row_to_labels(r) for r in rows]

        # Prefer: strict quatrain (reliable meter + known meter_type + rhyme), same as --strict-eval.
        if all(line_ok_for_strict_quatrain(l) for l in lines):
            chosen = lines
            break

        # Then: first 4-line stanza where every line has a valid +/- stress string.
        if chosen is None and all(l["stress_pm"] for l in lines):
            chosen = lines

        # Last resort: first stanza where every line has non-empty raw stress.
        if chosen is None and all(l["stress"] for l in lines):
            chosen = lines

    conn.close()
    if chosen is None:
        raise RuntimeError("Could not find an example stanza for few-shot prompts.")

    _EXAMPLE_QUATRAIN = chosen
    return _EXAMPLE_QUATRAIN


def render_few_shot_example_prefix(task: str, example_lines: list[dict]) -> str:
    """Render a quatrain example prefix (full stanza) for the given task."""
    def _meter_pattern_display(line_dict: dict) -> str:
        """Match gold meter_only targets: +/- string from stress_to_plus_minus."""
        return line_dict["stress_pm"] or line_dict["stress"]

    if task == "meter_only":
        out = [
            "Example poem (quatrain). For each line, output the metrical stress pattern (same as training):",
            "one character per metrical position: + (stressed) or - (unstressed).",
            "",
        ]
        for l in example_lines:
            out.append(f"Line: {l['normalized']}")
            out.append(f"Pattern: {_meter_pattern_display(l)}")
        return "\n".join(out)

    if task == "rhyme_only":
        out = [
            "Example poem (quatrain). For each line, output the rhyme label (same as training):",
            "prefer ARPAbet phones from the last stressed vowel through the end of the word;",
            "if unavailable, a single rhyme-group letter; if none, the token none.",
            "",
        ]
        for l in example_lines:
            out.append(f"Line: {l['normalized']}")
            out.append(f"Rhyme: {l['rhyme_only_target']}")
        return "\n".join(out)

    if task == "natural_text":
        l1, l2, l3, l4 = example_lines[0], example_lines[1], example_lines[2], example_lines[3]
        return (
            "Example: continue in the same style through one quatrain.\n\n"
            f"Context: {l1['normalized']}\nNext line: {l2['normalized']}\n\n"
            f"Context: {l2['normalized']}\nNext line: {l3['normalized']}\n\n"
            f"Context: {l3['normalized']}\nNext line: {l4['normalized']}\n"
        )

    if task == "combined":
        out = [
            "Example poem (quatrain). Output one bundled line per input line:",
            "meter:X|rhyme:Y|end:Z|caesura:C",
            "",
        ]
        for l in example_lines:
            out.append(f"Line: {l['normalized']}")
            out.append(l.get("combined_target_stress") or l["combined_target"])
        return "\n".join(out)

    raise ValueError(f"Unsupported task for few-shot prefix: {task}")


def render_one_shot_example(task: str, example_lines: list[dict]) -> str:
    """Render the labeled example used for one_shot prompts (2 lines)."""
    l1, l2 = example_lines[0], example_lines[1]
    if task == "meter_only":
        p1 = l1["stress_pm"] or l1["stress"]
        p2 = l2["stress_pm"] or l2["stress"]
        return f"Line: {l1['normalized']}\nPattern: {p1}\nLine: {l2['normalized']}\nPattern: {p2}"
    if task == "rhyme_only":
        return f"Line: {l1['normalized']}\nRhyme: {l1['rhyme_only_target']}\nLine: {l2['normalized']}\nRhyme: {l2['rhyme_only_target']}"
    if task == "natural_text":
        return f"Context: {l1['normalized']}\nNext line: {l2['normalized']}"
    if task == "combined":
        c1 = l1.get("combined_target_stress") or l1["combined_target"]
        c2 = l2.get("combined_target_stress") or l2["combined_target"]
        return f"Line: {l1['normalized']}\n{c1}\nLine: {l2['normalized']}\n{c2}"
    raise ValueError(f"Unsupported task for one_shot example: {task}")

# Prompt templates: task -> prompt_type -> template (use {line} or {input} placeholder).
PROMPTS = {
    "meter_only": {
        "zero_shot": (
            "Given this line of poetry, output only its metrical stress label: a string of + (stressed) and "
            "- (unstressed), one character per metrical position, exactly as in the training data. "
            "No words, no explanation.\n\n"
            "Line: {line}\n"
        ),
        "one_shot": (
            "Task: metrical stress. Output only + and - (one per metrical position), matching the style of the example.\n\n"
            "Example (two labeled lines):\n"
            "{EXAMPLE}\n\n"
            "Line: {line}\nPattern:\n"
        ),
        "few_shot": (
            "Output only the stress pattern for the line below: + and - only, one per metrical position, same as the examples.\n\n"
            "Line: {line}\nPattern:\n"
        ),
    },
    "rhyme_only": {
        "zero_shot": (
            "Given this line of poetry, output only its rhyme label, exactly as in the training data: "
            "prefer a short ARPAbet phone sequence (from the last stressed vowel through the end of the word, "
            "spaces between phones); if that is not available, output a single rhyme-group letter; "
            "if neither applies, output none. No other text.\n\n"
            "Line: {line}\n"
        ),
        "one_shot": (
            "Task: rhyme label for the line (ARPAbet tail, or rhyme-group letter, or none — same as the example).\n\n"
            "Example (two labeled lines):\n"
            "{EXAMPLE}\n\n"
            "Line: {line}\nRhyme:\n"
        ),
        "few_shot": (
            "Output only the rhyme label for the line below (same format as the examples).\n\n"
            "Line: {line}\nRhyme:\n"
        ),
    },
    "natural_text": {
        "zero_shot": (
            "Continue the poem: write the single next line that follows the context. "
            "Context uses one previous line, or several joined with \" | \", or [start] for the opening line.\n\n"
            "Context: {line}\n\n"
            "Next line:\n"
        ),
        "one_shot": (
            "Continue the poem: one next line only, in the same style as the example "
            "(one line of context, then the following line).\n\n"
            "Example:\n{EXAMPLE}\n\n"
            "Context: {line}\n\nNext line:\n"
        ),
        "few_shot": (
            "Continue the poem: output only the next line after the context (same style as the examples).\n\n"
            "Context: {line}\n\nNext line:\n"
        ),
    },
    "combined": {
        "zero_shot": (
            "Given this line of poetry, output a single line with four fields (training format):\n"
            "meter:X — stress string from the corpus if present, otherwise the abstract meter type;\n"
            "rhyme:Y — ARPAbet rhyme tail, else rhyme-group letter, else none;\n"
            "end:Z — 1 if the line is end-stopped, else 0;\n"
            "caesura:C — caesura annotation from the corpus (e.g. word index), or - if absent.\n"
            "Use exactly this template and no other text: meter:X|rhyme:Y|end:Z|caesura:C\n\n"
            "Line: {line}\n"
        ),
        "one_shot": (
            "Task: bundle meter, rhyme, end-stopping, and caesura in one string: meter:X|rhyme:Y|end:Z|caesura:C "
            "(same meaning as the example).\n\n"
            "Example:\n"
            "{EXAMPLE}\n\n"
            "Line: {line}\n"
        ),
        "few_shot": (
            "Output meter:X|rhyme:Y|end:Z|caesura:C for this line only, same format as the examples.\n\n"
            "Line: {line}\n"
        ),
    },
}


def slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_id)


def load_test_data(task: str, split: str = "test") -> list:
    """Load test data from output/training_data/{task}/{split}.json."""
    path = DATA_DIR / task / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Test data not found: {path}. Run notebooks/01_prepare_training_data.ipynb first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_prompt(item: dict, task: str, prompt_type: str) -> str:
    """Build prompt from item. Uses 'input' or 'line' key for the line text."""
    line = item.get("input", item.get("line", ""))
    template = PROMPTS.get(task, {}).get(prompt_type, PROMPTS["meter_only"]["zero_shot"])
    if prompt_type in ("one_shot", "few_shot"):
        example_lines = get_example_quatrain_lines()
    else:
        example_lines = None

    if prompt_type == "one_shot":
        example_text = render_one_shot_example(task, example_lines)
        return template.format(line=line, input=line, EXAMPLE=example_text)

    if prompt_type == "few_shot":
        prefix = render_few_shot_example_prefix(task, example_lines)
        body = template.format(line=line, input=line)
        return prefix.rstrip() + FEW_SHOT_SEPARATOR + body

    # zero_shot
    return template.format(line=line, input=line)


def run_inference_seq2seq(
    prompts: list,
    model_id: str,
    max_new_tokens: int = 64,
    device: int = -1,
    max_input_length: int = 512,
) -> list:
    """Encoder-decoder (T5, BART, PEGASUS, etc.): prompt as encoder input, decode output."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Install transformers: pip install transformers torch")

    dev = torch.device("cuda", device) if device >= 0 else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(dev)
    model.eval()

    outputs = []
    n = len(prompts)
    for i, p in enumerate(prompts):
        if i and i % 500 == 0:
            print(f"  ... generated {i}/{n} prompts", flush=True)
        inputs = tokenizer(
            p, return_tensors="pt", truncation=True, max_length=max_input_length
        ).to(dev)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


def run_inference_causal(
    prompts: list, model_id: str, max_new_tokens: int = 64, device: int = -1, max_input_length: int = 512
) -> list:
    """Decoder-only (GPT-2, Pythia, LLaMA, etc.): prompt as context, return only the generated continuation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Install transformers: pip install transformers torch")

    dev = torch.device("cuda", device) if device >= 0 else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(dev)
    model.eval()

    # Many causal LMs need a padding side for batch decode; left-pad so we decode only new tokens
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    n = len(prompts)
    for i, p in enumerate(prompts):
        if i and i % 500 == 0:
            print(f"  ... generated {i}/{n} prompts", flush=True)
        enc = tokenizer(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(dev)
        input_ids = enc["input_ids"]
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the new tokens (continuation), not the prompt
        new_ids = generated[0][input_ids.shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run prompt-only baseline on test data")
    parser.add_argument("--model", default="google/flan-t5-large", help="Hugging Face model ID")
    parser.add_argument(
        "--model_type",
        choices=["seq2seq", "causal"],
        default="seq2seq",
        help="seq2seq = encoder-decoder (T5, BART, etc.); causal = decoder-only (GPT-2, LLaMA, etc.)",
    )
    parser.add_argument(
        "--prompt",
        choices=["zero_shot", "one_shot", "few_shot"],
        default="zero_shot",
        help="Prompt variant",
    )
    parser.add_argument(
        "--task",
        choices=["meter_only", "rhyme_only", "natural_text", "combined"],
        default="meter_only",
        help="Task / condition",
    )
    parser.add_argument("--split", default="test", help="Data split (test or dev)")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit samples (default: all). Use 0 to mean no limit (same as omitting).",
    )
    parser.add_argument("--max_tokens", type=int, default=64, help="Max output tokens")
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=None,
        help="Tokenizer max length for input (default: 1024 for few_shot, else 512)",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU device (0, 1, ...); -1 for CPU")
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    parser.add_argument(
        "--strict-eval",
        action="store_true",
        help=(
            "Keep only test lines that appear in a fully reliable 4-line stanza in corpus.db "
            "(stress +/- length≥5, meter_type not unknown, rhyme not none); "
            "regold targets from DB (meter/rhyme/combined use stress-based meter slot). "
            "Requires output/corpus.db. Recommended for strict baseline runs."
        ),
    )
    parser.add_argument(
        "--gold-filter",
        choices=("none", "reliable"),
        default="none",
        help=(
            "none=all test lines; reliable=drop degenerate JSON targets (no DB). "
            "When --strict-eval is set, the DB/quatrain path is used instead (this flag is only for non-strict runs)."
        ),
    )
    args = parser.parse_args()

    corpus_db = ROOT / "output" / "corpus.db"
    data = load_test_data(args.task, args.split)
    filter_meta: dict | None = None
    data, filter_meta = filter_and_regold_data(
        data,
        args.task,
        strict_eval=args.strict_eval,
        gold_filter=args.gold_filter,
        corpus_db=corpus_db,
    )
    if filter_meta["n_input"] != filter_meta["n_after"]:
        print(
            f"Gold filter: {filter_meta['n_input']} -> {filter_meta['n_after']} lines "
            f"(dropped not-in-strict-quatrain={filter_meta['n_dropped_no_key']}, "
            f"dropped ambiguous/regold-fail={filter_meta['n_dropped_ambiguous_regold']})"
        )
    if args.n is not None and args.n > 0:
        data = data[: args.n]
    if not data:
        raise SystemExit(
            "No samples left after filtering. With --strict-eval, ensure output/corpus.db exists and "
            "test lines overlap reliable 4-line stanzas (export_sqlite + training_data from same corpus). "
            "Try STRICT_EVAL=0 or --gold-filter reliable without --strict-eval for a larger (noisier) set."
        )
    print(f"Loaded {len(data)} samples for task={args.task}, prompt={args.prompt}")

    prompts = [build_prompt(item, args.task, args.prompt) for item in data]
    max_in = (
        args.max_prompt_length
        if args.max_prompt_length is not None
        else (1024 if args.prompt == "few_shot" else 512)
    )
    print(f"Running {args.model} ({args.model_type}), max_prompt_length={max_in}...")
    if args.model_type == "causal":
        outputs = run_inference_causal(
            prompts,
            args.model,
            max_new_tokens=args.max_tokens,
            device=args.device,
            max_input_length=max_in,
        )
    else:
        outputs = run_inference_seq2seq(
            prompts,
            args.model,
            max_new_tokens=args.max_tokens,
            device=args.device,
            max_input_length=max_in,
        )

    def strip_prompt_echo(raw_out: str, prompt_used: str) -> str:
        """If the model echoed the prompt (or a prefix of it), strip it so we keep only the answer."""
        if not prompt_used or not raw_out:
            return raw_out
        if raw_out.startswith(prompt_used):
            return raw_out[len(prompt_used) :].strip()
        # Find longest prefix of prompt that output starts with (handles truncated echo, e.g. BART)
        for end in range(min(len(prompt_used), len(raw_out)), 0, -1):
            if raw_out.startswith(prompt_used[:end]):
                return raw_out[end:].strip()
        return raw_out

    if args.task == "natural_text":
        from evaluation.form_eval_generation import line_form_signature

    results = []
    for i, item in enumerate(data):
        raw_out = outputs[i] if i < len(outputs) else ""
        raw_out = strip_prompt_echo(raw_out, prompts[i])
        row = {
            "input": item.get("input", item.get("line", "")),
            "gold_target": item.get("target", ""),
            "prompt": prompts[i],
            "model_output": raw_out,
        }
        if args.task == "natural_text":
            gsig = line_form_signature(item.get("target", ""))
            row["gold_stress_pm"] = gsig["stress_pm"]
            row["gold_rhyme_key"] = gsig["rhyme_key"]
            row["gold_form_ok"] = gsig["ok"]
        results.append(row)

    out_dir = RESULTS_DIR / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (out_dir / f"{args.prompt}_{args.task}.json")
    payload = {
        "model": args.model,
        "prompt_type": args.prompt,
        "task": args.task,
        "split": args.split,
        "n_samples": len(results),
        "strict_eval": args.strict_eval,
        "gold_filter": args.gold_filter,
        "gold_filter_meta": filter_meta,
        "results": results,
    }
    if args.task == "natural_text":
        payload["gold_form_note"] = (
            "gold_stress_pm, gold_rhyme_key, gold_form_ok: CMU-based signature of gold_target "
            "(evaluation.form_eval_generation.line_form_signature); same path as eval_natural_text_form.py."
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
