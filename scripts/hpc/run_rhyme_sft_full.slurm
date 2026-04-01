#!/bin/bash
#SBATCH --job-name=rhyme-sft-full
#SBATCH --account=span9810
#SBATCH --partition=day
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=rhyme_sft_full_%j.out

set -euo pipefail

# Move to repo root
cd "${SLURM_SUBMIT_DIR:-$HOME/Senior-Thesis}"

# Activate env
if [ -d "venv" ]; then
  source venv/bin/activate
elif command -v conda &>/dev/null; then
  conda activate base
fi

# Optional shared HF cache
HF_CACHE_ROOT="${CLUSTER_HF_HOME:-${THESIS_HF_HOME:-}}"
if [ -n "${HF_CACHE_ROOT:-}" ]; then
  export HF_HOME="$HF_CACHE_ROOT"
  export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
  mkdir -p "$TRANSFORMERS_CACHE" || true
fi

# currently don't need one may later
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN set"

# Run full rhyme SFT (override via env vars at submit time)
python3 scripts/full_sft/rhyme_sft.py \
  --model "${MODEL:-google/flan-t5-base}" \
  --train_file "${TRAIN_FILE:-output/training_data/rhyme_only/train.json}" \
  --dev_file "${DEV_FILE:-output/training_data/rhyme_only/dev.json}" \
  --output_root "${OUTPUT_ROOT:-sft_full/rhyme_only}" \
  --num_train_epochs "${EPOCHS:-3}" \
  --learning_rate "${LR:-5e-5}" \
  --weight_decay "${WD:-0.01}" \
  --warmup_ratio "${WARMUP:-0.1}" \
  --per_device_train_batch_size "${TRAIN_BS:-4}" \
  --per_device_eval_batch_size "${EVAL_BS:-4}" \
  --gradient_accumulation_steps "${GRAD_ACC:-1}" \
  --eval_steps "${EVAL_STEPS:-200}" \
  --save_steps "${SAVE_STEPS:-200}" \
  --seed "${SEED:-42}"

echo "Done. Check sft_full/rhyme_only/"