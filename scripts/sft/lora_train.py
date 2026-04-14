#!/usr/bin/env python3
"""LoRA / QLoRA fine-tuning for seq2seq models on structured tasks (meter_only, rhyme_only, combined).

Uses JSON with `input` / `target` fields from output/training_data/<task>/*.json.
Set --train_file, --dev_file, --output_root per task. Slurm: scripts/hpc/lora_train.slurm or scripts/hpc/submit_lora3.sh.

QLoRA (--qlora) requires CUDA + bitsandbytes; otherwise use LoRA only.

Inference:
- Raw adapters in final_model/ are not directly usable with from_pretrained.
- Prefer --merge_and_save and load final_model_merged/.
- Otherwise, load base model (e.g., google/flan-t5-large) and attach adapters.
- Eval with scripts/run_prompt_eval.py expects merged weights or Hub ids (plain from_pretrained).

Cluster notes:
- Defaults: eval/save every 1000 steps; train=8, eval=4 (~32GB GPU); reduce batch if OOM; scale up on larger GPUs.
- Slurm defaults BF16 on CUDA (--bf16); NO_BF16=1 for fp16; if NaNs or loss=0 use full FP32 (NO_FP16=1 NO_BF16=1).
- Resume with --resume_from_checkpoint pointing at checkpoint-* under the run directory.
"""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError as e:
    raise SystemExit(
        "Install PEFT:  pip install peft\n"
        "For QLoRA on GPU:  pip install bitsandbytes\n"
        f"{e}"
    ) from e


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_hf_dataset(rows):
    return Dataset.from_list(rows)


def tokenize_function(examples, tokenizer, max_input_len, max_target_len):
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
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def debug_batch_inspection(args, train_rows: list, train_tokenized, data_collator) -> None:
    print("=== debug_batch_only ===")
    for i in range(min(5, len(train_rows))):
        inp = train_rows[i].get("input", "")
        tgt = train_rows[i].get("target", "")
        print(f"raw[{i}] input_len={len(inp)} target_len={len(tgt)} target_empty={not str(tgt).strip()}")
    for i in range(min(5, len(train_tokenized))):
        row = train_tokenized[i]
        lab = row["labels"]
        lab = lab.tolist() if hasattr(lab, "tolist") else list(lab)
        n_sup = sum(1 for x in lab if x != -100)
        print(f"tok[{i}] len(labels)={len(lab)} supervised_tokens(≠-100)={n_sup}")
    from torch.utils.data import DataLoader

    bs = max(1, min(args.per_device_train_batch_size, len(train_tokenized)))
    dl = DataLoader(
        train_tokenized,
        batch_size=bs,
        collate_fn=data_collator,
        shuffle=False,
    )
    batch = next(iter(dl))
    labels = batch["labels"]
    print(f"one batch: labels shape={tuple(labels.shape)} batch_size={bs}")
    for bi in range(labels.size(0)):
        row = labels[bi].tolist()
        n100 = sum(1 for x in row if x == -100)
        n_sup = len(row) - n100
        print(f"  item {bi}: pad(-100)={n100} supervised={n_sup}")
    print("=== end debug ===")


def load_base_model(args: argparse.Namespace, model_id: str):
    """Load seq2seq base (optionally 4-bit quantized for QLoRA)."""
    if args.qlora:
        if not torch.cuda.is_available() or args.use_cpu:
            raise SystemExit("--qlora requires CUDA (omit --use_cpu).")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise SystemExit("QLoRA needs bitsandbytes on CUDA: pip install bitsandbytes") from e
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=not args.no_qlora_double_quant,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return model


def attach_lora(model, args: argparse.Namespace):
    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    if not targets:
        raise SystemExit("--lora_target_modules must list at least one module (e.g. q,v).")
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=targets,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    # Frozen base + checkpointing: gradients must flow into LoRA layers (avoids grad_norm nan).
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # T5: disable KV cache during training (recommended with gradient checkpointing / PEFT).
    try:
        inner = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(inner, "config"):
            inner.config.use_cache = False
    except Exception:
        pass
    return model


def main():
    parser = argparse.ArgumentParser(description="LoRA / QLoRA fine-tuning (seq2seq; task via --train_file paths)")
    parser.add_argument("--model", type=str, default="google/flan-t5-large")
    parser.add_argument("--train_file", type=str, default="output/training_data/rhyme_only/train.json")
    parser.add_argument("--dev_file", type=str, default="output/training_data/rhyme_only/dev.json")
    parser.add_argument("--output_root", type=str, default="sft/rhyme_only_lora")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Eval batch per device; use 2 if eval OOMs (YCRC suggests >1 for throughput).",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=64)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluate every N steps (larger = less eval overhead, spikier metrics).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (for resume after wall-time kill).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--predict_with_generate", action="store_true")
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--no_eval_during_training", action="store_true")
    parser.add_argument("--final_eval_at_end", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="bf16 training (NVIDIA with bf16 support).")
    parser.add_argument("--debug_batch_only", action="store_true")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="checkpoint-XXXX dir from a prior LoRA run (under that run's output_dir).",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="4-bit NF4 base + LoRA (CUDA + bitsandbytes).",
    )
    parser.add_argument(
        "--no_qlora_double_quant",
        action="store_true",
        help="QLoRA only: set bitsandbytes double_quant=False (sometimes stabilizes NaN grads; uses a bit more VRAM).",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,v",
        help="Comma-separated T5 attention linear names (FLAN-T5: q,v is typical).",
    )
    parser.add_argument(
        "--merge_and_save",
        action="store_true",
        help="After training, merge LoRA into base and save to final_model_merged/ for plain from_pretrained().",
    )
    args = parser.parse_args()

    resume_ckpt: Path | None = None
    if str(args.resume_from_checkpoint).strip():
        resume_ckpt = Path(args.resume_from_checkpoint).expanduser().resolve()
        if not resume_ckpt.is_dir():
            raise SystemExit(f"--resume_from_checkpoint not found: {resume_ckpt}")
        run_dir = resume_ckpt.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Resuming from {resume_ckpt} (run_dir={run_dir})", flush=True)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "qlora" if args.qlora else "lora"
        run_dir = Path(args.output_root) / f"{Path(args.model).name}_{tag}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_params.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_rows = load_json(Path(args.train_file))
    dev_rows = load_json(Path(args.dev_file))
    train_ds = build_hf_dataset(train_rows)
    dev_ds = build_hf_dataset(dev_rows)

    tokenizer = (
        AutoTokenizer.from_pretrained(str(resume_ckpt))
        if resume_ckpt is not None
        else AutoTokenizer.from_pretrained(args.model)
    )

    model_id = args.model
    model = load_base_model(args, model_id)
    model = attach_lora(model, args)

    train_tokenized = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.max_input_len, args.max_target_len),
        batched=True,
    )
    dev_tokenized = dev_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.max_input_len, args.max_target_len),
        batched=True,
    )
    if args.max_eval_samples and args.max_eval_samples > 0:
        n = min(args.max_eval_samples, len(dev_tokenized))
        dev_tokenized = dev_tokenized.select(range(n))

    _drop = [c for c in ("input", "target") if c in train_tokenized.column_names]
    if _drop:
        train_tokenized = train_tokenized.remove_columns(_drop)
        dev_tokenized = dev_tokenized.remove_columns(
            [c for c in _drop if c in dev_tokenized.column_names]
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    smoke = args.max_steps > 0
    use_eval = (not smoke) and (not args.no_eval_during_training)
    use_cuda = torch.cuda.is_available() and not args.use_cpu
    # QLoRA: HF recipes use fp16 or bf16 for adapter training; disabling both caused grad_norm nan / loss 0.
    use_fp16 = use_cuda and not args.no_fp16 and not args.bf16
    use_bf16 = use_cuda and args.bf16 and not args.no_fp16
    if use_bf16 and use_cuda and not torch.cuda.is_bf16_supported():
        print("WARN: --bf16 not supported on this GPU; using fp16 for QLoRA/LoRA.", flush=True)
        use_bf16 = False
        use_fp16 = use_cuda and not args.no_fp16
    print(
        f"Precision: fp16={use_fp16} bf16={use_bf16} qlora={args.qlora} cuda={use_cuda}",
        flush=True,
    )

    ta_kwargs = dict(
        output_dir=str(run_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if use_eval else "no",
        save_steps=min(args.save_steps, args.max_steps) if smoke and args.max_steps > 0 else args.save_steps,
        logging_steps=min(50, args.max_steps) if smoke and args.max_steps > 0 else 50,
        save_total_limit=5,
        predict_with_generate=args.predict_with_generate,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=use_eval,
        greater_is_better=False,
        use_cpu=args.use_cpu,
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    if use_eval:
        ta_kwargs["eval_steps"] = args.eval_steps
        ta_kwargs["metric_for_best_model"] = "eval_loss"
    training_args = Seq2SeqTrainingArguments(**ta_kwargs)

    _tk_kw = {}
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        _tk_kw["processing_class"] = tokenizer
    else:
        _tk_kw["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        data_collator=data_collator,
        **_tk_kw,
    )

    if args.debug_batch_only:
        debug_batch_inspection(args, train_rows, train_tokenized, data_collator)
        print("Exiting (--debug_batch_only).")
        return

    train_kw: dict = {}
    if resume_ckpt is not None:
        train_kw["resume_from_checkpoint"] = str(resume_ckpt)
    trainer.train(**train_kw)

    final_dir = run_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.merge_and_save and hasattr(trainer.model, "merge_and_unload"):
        merged_dir = run_dir / "final_model_merged"
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Merged weights saved to {merged_dir}", flush=True)

    if smoke:
        run_final_eval = args.final_eval_at_end
    elif args.no_eval_during_training:
        run_final_eval = args.final_eval_at_end
    else:
        run_final_eval = True
    if run_final_eval:
        metrics = trainer.evaluate()
        with (run_dir / "final_eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(f"Done. Adapter + tokenizer: {final_dir}")
    if args.merge_and_save:
        print(f"Merged (optional): {run_dir / 'final_model_merged'}")


if __name__ == "__main__":
    main()
