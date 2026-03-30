#!/usr/bin/env python3
"""Train a seq2seq model on one task (meter_only, rhyme_only, natural_text, or combined).

Reads `output/training_data/{task}/train.json` rows with `input` and `target`.
By default writes under `sft/<task>/`. Use `--max_steps 2` for a tiny CPU smoke test (skips eval hooks).
For four matching runs (one per task), see `scripts/hpc/run_sft_four_conditions.sh`.
"""

from __future__ import annotations

import argparse
import inspect
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
        default=None,
        help="Defaults to sft/<task>/",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Linear LR warmup fraction of total training steps.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping (0 to disable).",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop if eval_loss does not improve for this many evals; 0 disables (full runs only).",
    )
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 (recommended on recent NVIDIA GPUs with enough memory).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Trade compute for memory (helpful for large models on one GPU).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0, stop after this many optimizer steps (smoke test; overrides num_train_epochs).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (ROOT / "sft" / args.task)

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
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(f"Install deps: pip install transformers datasets torch accelerate\n{e}") from e

    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

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

    output_dir.mkdir(parents=True, exist_ok=True)
    _cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    _cfg["output_dir_resolved"] = str(output_dir)
    (output_dir / "sft_run_config.json").write_text(
        json.dumps(_cfg, indent=2, default=str),
        encoding="utf-8",
    )

    smoke = args.max_steps > 0
    use_es = (not smoke) and args.early_stopping_patience > 0
    # HF: max_grad_norm 0 disables clipping
    max_grad = args.max_grad_norm if args.max_grad_norm > 0 else 0.0

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        warmup_ratio=0.0 if smoke else args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=max_grad,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="no" if smoke else "steps",
        eval_steps=args.eval_steps,
        save_strategy="no" if smoke else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False if smoke else True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1 if smoke else 50,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        seed=args.seed,
        report_to="none",
        bf16=args.bf16 and (not smoke) and torch.cuda.is_available(),
    )

    callbacks = []
    if use_es:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    # transformers >= 4.46 uses processing_class; older versions use tokenizer
    _sig = inspect.signature(Seq2SeqTrainer.__init__)
    _tok_kw = (
        "processing_class"
        if "processing_class" in _sig.parameters
        else "tokenizer"
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        callbacks=callbacks,
        **{_tok_kw: tokenizer},
    )

    trainer.train()
    final = output_dir / "final_model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"Saved: {final}")


if __name__ == "__main__":
    main()
