#!/bin/bash
#SBATCH --job-name=lora-train
#SBATCH --account=span9810

# Runs LoRA / QLoRA training using scripts/sft/lora_train.py
# Training data and output paths are controlled by TRAIN_FILE, DEV_FILE, OUTPUT_ROOT

# Default setup:
# - 1 GPU (RTX 5000 Ada)
# - Up to 48 hours runtime
# - Periodic evaluation and checkpoint saving
#
# Common fixes:
# - NaN errors: set NO_FP16=1 NO_BF16=1
# - Out of memory: reduce TRAIN_BS and EVAL_BS
# - QLoRA instability: set NO_QLORA_DOUBLE_QUANT=1

# Short test runs can use gpu_devel with lower time

#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --mem=48G

#SBATCH --output=lora_train_%j.out
#SBATCH --error=lora_train_%j.err

# Safer bash settings
set -euo pipefail
export PYTHONUNBUFFERED=1

echo "Start $(date)"

# Check write access
_SUBMIT="${SLURM_SUBMIT_DIR:-.}"
touch "$_SUBMIT/test_file" || { echo "Cannot write to submit directory"; exit 1; }
rm -f "$_SUBMIT/test_file"

# Move to project directory
THESIS_ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/Senior-Thesis}}"
cd "$THESIS_ROOT" || { echo "Project directory not found"; exit 1; }

# Ensure training script exists
if [ ! -f scripts/sft/lora_train.py ]; then
  echo "Missing training script"
  exit 1
fi

# Activate environment
if [ -d "venv" ]; then
  source venv/bin/activate
elif command -v conda &>/dev/null; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate base || true
fi

# Set Hugging Face cache location
HF_CACHE_ROOT="${CLUSTER_HF_HOME:-${THESIS_HF_HOME:-}}"
if [ -z "${HF_CACHE_ROOT:-}" ] && [ -d "${HOME}/scratch" ]; then
  HF_CACHE_ROOT="${HOME}/scratch/hf"
fi

if [ -n "${HF_CACHE_ROOT:-}" ]; then
  export HF_HOME="$HF_CACHE_ROOT"
  export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
  mkdir -p "$TRANSFORMERS_CACHE"
fi

# Select python
PYTHON="python3"
[ -x "venv/bin/python" ] && PYTHON="venv/bin/python"

# Check required packages
$PYTHON -c "import datasets, transformers, peft" 2>/dev/null || {
  echo "Missing required packages"
  exit 1
}

# Optional flags
EXTRA_ARGS=()
[ "${SMOKE:-0}" = "1" ] && EXTRA_ARGS+=(--max_steps 50)
[ "${NO_FP16:-0}" = "1" ] && EXTRA_ARGS+=(--no_fp16)
[ "${NO_BF16:-0}" != "1" ] && EXTRA_ARGS+=(--bf16)
[ "${QLORA:-0}" = "1" ] && EXTRA_ARGS+=(--qlora)
[ "${NO_QLORA_DOUBLE_QUANT:-0}" = "1" ] && EXTRA_ARGS+=(--no_qlora_double_quant)
[ "${MERGE_AND_SAVE:-0}" = "1" ] && EXTRA_ARGS+=(--merge_and_save)

# Run training
"$PYTHON" -u scripts/sft/lora_train.py \
  --model "${MODEL:-google/flan-t5-large}" \
  --train_file "${TRAIN_FILE:-output/training_data/rhyme_only/train.json}" \
  --dev_file "${DEV_FILE:-output/training_data/rhyme_only/dev.json}" \
  --output_root "${OUTPUT_ROOT:-sft_runs/rhyme_only_lora}" \
  --num_train_epochs "${EPOCHS:-3}" \
  --learning_rate "${LR:-1e-4}" \
  --per_device_train_batch_size "${TRAIN_BS:-8}" \
  --per_device_eval_batch_size "${EVAL_BS:-4}" \
  --gradient_accumulation_steps "${GRAD_ACC:-1}" \
  --eval_steps "${EVAL_STEPS:-1000}" \
  --save_steps "${SAVE_STEPS:-1000}" \
  --seed "${SEED:-42}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --lora_target_modules "${LORA_TARGET_MODULES:-q,v}" \
  "${EXTRA_ARGS[@]}"

echo "Done. Outputs saved in OUTPUT_ROOT"
