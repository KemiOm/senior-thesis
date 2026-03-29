#!/usr/bin/env python3
"""Minimal seq2seq SFT runner for one task.

Input JSON: output/training_data/{task}/train.json with {"input", "target"} rows.
Use sample caps for smoke tests; scale on HPC for full runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output" / "training_data"


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["meter_only", "rhyme_only", "natural_text", "combined"],
        required=True,
    )
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output" / "sft_runs" / "sample",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = DATA_DIR / args.task / "train.json"
    dev_path = DATA_DIR / args.task / "dev.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run notebooks/01_prepare_training_data.ipynb.")

    train_rows = load_json(train_path)
    eval_rows = load_json(dev_path) if dev_path.exists() else train_rows[:500]

    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples:
        eval_rows = eval_rows[: args.max_eval_samples]

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(f"Install deps: pip install transformers datasets torch accelerate\n{e}") from e

    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def to_hf(rows):
        return Dataset.from_dict(
            {
                "input_text": [r["input"] for r in rows],
                "target_text": [r["target"] for r in rows],
            }
        )

    train_ds = to_hf(train_rows)
    eval_ds = to_hf(eval_rows)

    def preprocess(batch):
        ins = tokenizer(
            batch["input_text"],
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            batch["target_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        ins["labels"] = labels["input_ids"]
        return ins

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        seed=args.seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "final_model"))
    tokenizer.save_pretrained(str(args.output_dir / "final_model"))
    print(f"Saved: {args.output_dir / 'final_model'}")


if __name__ == "__main__":
    main()
