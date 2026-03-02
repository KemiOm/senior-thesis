#!/usr/bin/env python3
"""
Run prompt-only baseline: load test lines, build prompts, call models, save outputs.

Usage:
  python scripts/run_prompt_baseline.py --model google/flan-t5-large --prompt zero_shot --task meter_only
  python scripts/run_prompt_baseline.py --model google/flan-t5-base --prompt one_shot --task meter_only --n 100

Requires: pip install transformers torch (or pip install transformers accelerate)

Output: evaluation/results/baselines/prompt_only/{model_slug}/{prompt_type}_{task}.json
"""

import argparse
import json
import re
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output" / "training_data"
RESULTS_DIR = ROOT / "evaluation" / "results" / "baselines" / "prompt_only"

# Prompt templates: task -> prompt_type -> template (use {line} or {input} placeholder)
PROMPTS = {
    "meter_only": {
        "zero_shot": (
            "Given this line of poetry, output its metrical stress pattern as a string of + (stressed) and - (unstressed) for each syllable. Output nothing else.\n\n"
            "Line: {line}\n"
        ),
        "one_shot": (
            "Given a line of poetry, output its metrical stress pattern as + (stressed) and - (unstressed).\n\n"
            "Example:\nLine: The cat sat on the mat\nPattern: -+-+-+-+\n\n"
            "Line: {line}\nPattern:\n"
        ),
        "few_shot": (
            "Output the metrical stress pattern (+/-) for each line.\n\n"
            "Line: To be or not to be, that is the question\nPattern: -+-+-+-+-+-+-+-+-+\n\n"
            "Line: Shall I compare thee to a summer's day\nPattern: -+-+-+-+-+-+-+-+-+\n\n"
            "Line: {line}\nPattern:\n"
        ),
    },
    "rhyme_only": {
        "zero_shot": (
            "Given this line of poetry, output the rhyme sound (phoneme rime of the last stressed syllable onward). Output nothing else.\n\n"
            "Line: {line}\n"
        ),
        "one_shot": (
            "Given a line of poetry, output the rhyme sound (phoneme rime from the last stressed syllable).\n\n"
            "Example:\nLine: The cat sat on the mat\nRhyme: AE1 T\n\n"
            "Line: {line}\nRhyme:\n"
        ),
        "few_shot": (
            "Output the rhyme sound (phoneme rime) for each line.\n\n"
            "Line: To be or not to be\nRhyme: IY1\n\n"
            "Line: Shall I compare thee to a summer's day\nRhyme: EY1\n\n"
            "Line: {line}\nRhyme:\n"
        ),
    },
    "natural_text": {
        "zero_shot": (
            "Continue this poem. Given the previous line(s), write the next line.\n\n"
            "Context: {line}\n\n"
            "Next line:\n"
        ),
        "one_shot": (
            "Continue this poem. Given the context, write the next line.\n\n"
            "Example:\nContext: The sun sets in the west\nNext line: The stars will light the rest\n\n"
            "Context: {line}\n\nNext line:\n"
        ),
        "few_shot": (
            "Continue this poem.\n\n"
            "Context: The sun sets in the west\nNext line: The stars will light the rest\n\n"
            "Context: When in disgrace with fortune\nNext line: And men's eyes\n\n"
            "Context: {line}\n\nNext line:\n"
        ),
    },
    "combined": {
        "zero_shot": (
            "Given this line of poetry, output: meter (stress pattern or type), rhyme (phoneme rime or class), end (1 if line ends with punctuation, 0 otherwise), caesura (word index of mid-line pause or -). Format: meter:X|rhyme:Y|end:Z|caesura:C\n\n"
            "Line: {line}\n"
        ),
        "one_shot": (
            "Given a line of poetry, output meter, rhyme, end, caesura in format: meter:X|rhyme:Y|end:Z|caesura:C\n\n"
            "Example:\nLine: The cat sat on the mat.\nmeter:+-+-+-+-|rhyme:AE1 T|end:1|caesura:-\n\n"
            "Line: {line}\n"
        ),
        "few_shot": (
            "Output meter, rhyme, end, caesura for each line. Format: meter:X|rhyme:Y|end:Z|caesura:C\n\n"
            "Line: The cat sat on the mat.\nmeter:+-+-+-+-|rhyme:AE1 T|end:1|caesura:-\n\n"
            "Line: To be or not to be\nmeter:-+-+-+-+-+|rhyme:IY1|end:0|caesura:-\n\n"
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
    if "{line}" in template:
        return template.format(line=line)
    return template.format(input=line)


def run_inference(prompts: list, model_id: str, max_new_tokens: int = 64, device: int = -1) -> list:
    """Run T5/FLAN-T5 on prompts. device=-1 means CPU, 0+ means GPU."""
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
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run prompt-only baseline on test data")
    parser.add_argument("--model", default="google/flan-t5-large", help="Hugging Face model ID")
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
    parser.add_argument("--n", type=int, default=None, help="Limit samples (default: all)")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max output tokens")
    parser.add_argument("--device", type=int, default=0, help="GPU device (0, 1, ...); -1 for CPU")
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    args = parser.parse_args()

    data = load_test_data(args.task, args.split)
    if args.n:
        data = data[: args.n]
    print(f"Loaded {len(data)} samples for task={args.task}, prompt={args.prompt}")

    prompts = [build_prompt(item, args.task, args.prompt) for item in data]
    print(f"Running {args.model}...")
    outputs = run_inference(prompts, args.model, max_new_tokens=args.max_tokens, device=args.device)

    results = []
    for i, item in enumerate(data):
        results.append({
            "input": item.get("input", item.get("line", "")),
            "gold_target": item.get("target", ""),
            "prompt": prompts[i],
            "model_output": outputs[i] if i < len(outputs) else "",
        })

    out_dir = RESULTS_DIR / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (out_dir / f"{args.prompt}_{args.task}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "prompt_type": args.prompt,
                "task": args.task,
                "split": args.split,
                "n_samples": len(results),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
