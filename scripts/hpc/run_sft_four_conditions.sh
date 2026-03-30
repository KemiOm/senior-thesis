#!/usr/bin/env bash
# Run four full SFT jobs   with identical hyperparameters.
# Each task writes to sft/<task>/final_model and sft_run_config.json



set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-google/flan-t5-large}"
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-0.1}"
WD="${WD:-0.01}"
CLIP="${CLIP:-1.0}"
ES_PATIENCE="${ES_PATIENCE:-3}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BS="${TRAIN_BS:-4}"
EVAL_BS="${EVAL_BS:-4}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SMOKE="${SMOKE:-0}"

EXTRA=()
if [[ "${GRAD_CKPT:-0}" == "1" ]]; then
  EXTRA+=(--gradient_checkpointing)
fi
if [[ "${BF16:-0}" == "1" ]]; then
  EXTRA+=(--bf16)
fi

run_one() {
  local task="$1"
  local out="$ROOT/sft/${task}"
  echo "========== ${task} -> ${out} =========="
  if [[ "$SMOKE" == "1" ]]; then
    python scripts/train_sft_seq2seq_sample.py \
      --task "$task" \
      --model "$MODEL" \
      --output_dir "$out" \
      --max_steps 2 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --max_train_samples 64 \
      "${EXTRA[@]}"
  else
    python scripts/train_sft_seq2seq_sample.py \
      --task "$task" \
      --model "$MODEL" \
      --output_dir "$out" \
      --num_train_epochs "$EPOCHS" \
      --learning_rate "$LR" \
      --warmup_ratio "$WARMUP" \
      --weight_decay "$WD" \
      --max_grad_norm "$CLIP" \
      --early_stopping_patience "$ES_PATIENCE" \
      --eval_steps "$EVAL_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --per_device_train_batch_size "$TRAIN_BS" \
      --per_device_eval_batch_size "$EVAL_BS" \
      "${EXTRA[@]}"
  fi
}

for TASK in meter_only rhyme_only natural_text combined; do
  run_one "$TASK"
done

echo "Done. Checkpoints under sft/<task>/final_model"
