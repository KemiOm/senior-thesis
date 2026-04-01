#!/usr/bin/env python3
# Run full fine-tuning for the rhyme_only task.
# This script is meant to be simple and explicit, not heavily optimized.

import argparse
import json
from pathlib import Path
from datetime import datetime 
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,

)

def load_sjon(path: Path):
    """Open a JSON file and return parsed content"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_hf_dataset(rows):
    """convert rows like {"input": "...", "target": "..."} into a Hugging Face Dataset"""
    return Dataset.from_list(rows)

def tokenize_function(examples, tokenizer, max_input_len, max_target_len):
    """Tokenizer model inputs"""
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_len,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=max_target_len,
        truncation=True,
    )
    # add tokenized labels to model_inputs so Trainer can computer loss
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # Parse command-line arguments so each run is reproducible and configurable.
    parser = argparse.ArgumentParser(description="Full fine-tuning for rhyme_only task")
    parser.add_argument("--model", type=str, default="google/flan-t5-base")
    # Base model checkpoint to fine-tune.
    parser.add_argument("--train_file", type=str, default="output/training_data/rhyme_only/train.json")
    # Path to training examples generated from your corpus pipeline.
    parser.add_argument("--dev_file", type=str, default="output/training_data/rhyme_only/dev.json")
    # Path to dev/validation examples for periodic eval during training.
    parser.add_argument("--output_root", type=str, default="sft_full/rhyme_only")
    # Root output directory where checkpoints, logs, and params are saved.
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    # Number of full passes over the training dataset.
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    # AdamW learning rate; keep explicit for run tracking.
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # Weight decay regularization.
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    # Fraction of total steps used for LR warmup.
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    # Batch size per GPU/CPU for training.
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    # Batch size per GPU/CPU for evaluation.
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # Accumulate gradients across steps to simulate larger effective batch size.
    parser.add_argument("--max_input_len", type=int, default=256)
    # Max token length for inputs.
    parser.add_argument("--max_target_len", type=int, default=64)
    # Max token length for targets (rhyme labels are short).
    parser.add_argument("--eval_steps", type=int, default=200)
    # Evaluate every N update steps.
    parser.add_argument("--save_steps", type=int, default=200)
    # Save checkpoint every N update steps.
    parser.add_argument("--seed", type=int, default=42)
    # Random seed for reproducibility.
    args = parser.parse_args()
    # Build a timestamped run directory so each run stays separate and traceable.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{Path(args.model).name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save all run arguments to disk for experiment tracking.
    params_path = run_dir / "run_params.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Load train/dev data rows from JSON files.
    train_rows = load_json(Path(args.train_file))
    dev_rows = load_json(Path(args.dev_file))

    # Convert raw rows into Hugging Face Dataset objects.
    train_ds = build_hf_dataset(train_rows)
    dev_ds = build_hf_dataset(dev_rows)

    # Load tokenizer and base seq2seq model.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Tokenize both datasets with the same function and limits.
    train_tokenized = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.max_input_len, args.max_target_len),
        batched=True,
    )
    dev_tokenized = dev_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.max_input_len, args.max_target_len),
        batched=True,
    )

    # Dynamic padding collator for seq2seq tasks.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Define training behavior, logging, evaluation cadence, and checkpoint policy.
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,
        bf16=False,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Create Trainer object to handle train/eval loop.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start fine-tuning.
    trainer.train()

    # Save final model and tokenizer.
    final_dir = run_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Write final evaluation metrics.
    metrics = trainer.evaluate()
    with (run_dir / "final_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()