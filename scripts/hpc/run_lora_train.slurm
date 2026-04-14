#!/bin/bash
#SBATCH --job-name=lora-train
#SBATCH --account=span9810
# LoRA / QLoRA for any structured task — runs scripts/sft/lora_train.py (set TRAIN_FILE, DEV_FILE, OUTPUT_ROOT).
# GPU docs: https://docs.ycrc.yale.edu/clusters/bouchet/#partitions-and-hardware
#
# Default below: production `gpu` queue, RTX 5000 Ada, 48h wall (YCRC: gpu partition max ~48h).
# Eval/save every 1000 steps; train_bs=8 / eval_bs=4; bf16 on by default (NO_BF16=1 for fp16).
# If logs show grad_norm nan: use NO_FP16=1 NO_BF16=1 for full fp32 (NO_BF16 alone still uses fp16).
# QLoRA still unstable: NO_QLORA_DOUBLE_QUANT=1 (disables bitsandbytes double quant).
# OOM on RTX 5000 Ada: TRAIN_BS=6 EVAL_BS=2
# H200: TRAIN_BS=32 EVAL_BS=16 (tune down if OOM). RTX Pro 6000: ~24 / ~12.
#
# Quick test / debug (~6h max on gpu_devel, often starts faster):
#   #SBATCH --partition=gpu_devel
#   #SBATCH --gpus=rtx_5000_ada:1
#   #SBATCH --cpus-per-gpu=4
#   #SBATCH --time=6:00:00
#   #SBATCH --mem=48G
#
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#
# H200 (~4x VRAM vs RTX 5000 Ada — try TRAIN_BS / EVAL_BS ~4x; comment block above, use e.g.):
#   #SBATCH --partition=gpu_h200
#   #SBATCH --gpus=h200:1
#   #SBATCH --cpus-per-gpu=8
#   #SBATCH --time=48:00:00
#   #SBATCH --mem=64G
#   #  TRAIN_BS=32 EVAL_BS=16 sbatch scripts/hpc/run_lora_train.slurm
#
# RTX PRO 6000 Blackwell (CUDA 12.8+ in your env; ~3x VRAM — try batches ~3x):
#   #SBATCH --partition=gpu_rtx6000
#   #SBATCH --gpus=rtx_pro_6000_blackwell:1
#   #SBATCH --cpus-per-gpu=4
#   #SBATCH --time=48:00:00
#   #SBATCH --mem=64G
#
#SBATCH --output=lora_train_%j.out
#SBATCH --error=lora_train_%j.err
# Single-GPU Trainer; multi-GPU DDP would need torchrun + code changes.

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTHONFAULTHANDLER=1

echo "=== lora_train start $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname -s) ==="
_SUBMIT="${SLURM_SUBMIT_DIR:-.}"
if ! touch "$_SUBMIT/.slurm_write_test_$$" 2>/dev/null; then
  echo "ERROR: cannot create files in SLURM_SUBMIT_DIR=$_SUBMIT"
  exit 1
fi
rm -f "$_SUBMIT/.slurm_write_test_$$"

THESIS_ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/Senior-Thesis}}"
cd "$THESIS_ROOT" || { echo "ERROR: cd failed: $THESIS_ROOT"; exit 1; }
if [ ! -f scripts/sft/lora_train.py ]; then
  echo "ERROR: missing scripts/sft/lora_train.py. pwd=$(pwd)"
  exit 1
fi

if [ -d "venv" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
elif [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-base}" || true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-base}" || true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-base}" || true
fi

HF_CACHE_ROOT="${CLUSTER_HF_HOME:-${THESIS_HF_HOME:-}}"
if [ -z "${HF_CACHE_ROOT:-}" ] && [ -d "${HOME}/scratch" ]; then
  HF_CACHE_ROOT="${HOME}/scratch/hf"
fi
if [ -n "${HF_CACHE_ROOT:-}" ]; then
  export HF_HOME="$HF_CACHE_ROOT"
  export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
  mkdir -p "$TRANSFORMERS_CACHE" || true
  echo "HF_HOME=$HF_HOME"
fi

if [ -x "${THESIS_ROOT}/venv/bin/python" ]; then
  PYTHON="${THESIS_ROOT}/venv/bin/python"
elif [ -x "venv/bin/python" ]; then
  PYTHON="venv/bin/python"
else
  PYTHON="python3"
fi
echo "Using: $PYTHON"
$PYTHON -c "import datasets, transformers, peft" 2>/dev/null || {
  echo "ERROR: pip install -r requirements_sft.txt (needs peft)"
  exit 1
}
if [ "${QLORA:-0}" = "1" ]; then
  $PYTHON -c "import bitsandbytes" 2>/dev/null || {
    echo "ERROR: QLORA=1 requires: pip install bitsandbytes"
    exit 1
  }
fi

EXTRA_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--max_steps "${SMOKE_MAX_STEPS:-50}")
fi
if [ -n "${MAX_EVAL_SAMPLES:-}" ] && [ "${MAX_EVAL_SAMPLES}" != "0" ]; then
  EXTRA_ARGS+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi
if [ "${NO_EVAL_DURING_TRAINING:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no_eval_during_training)
fi
if [ "${NO_FP16:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no_fp16)
fi
# bf16 on Ampere+ unless NO_BF16=1 or BF16=0 (QLoRA path ignores bf16 in Python)
if [ "${NO_BF16:-0}" != "1" ] && [ "${BF16:-1}" = "1" ]; then
  EXTRA_ARGS+=(--bf16)
fi
if [ "${DEBUG_BATCH_ONLY:-0}" = "1" ]; then
  EXTRA_ARGS+=(--debug_batch_only)
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [ "${QLORA:-0}" = "1" ]; then
  EXTRA_ARGS+=(--qlora)
fi
if [ "${NO_QLORA_DOUBLE_QUANT:-0}" = "1" ]; then
  EXTRA_ARGS+=(--no_qlora_double_quant)
fi
if [ "${MERGE_AND_SAVE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--merge_and_save)
fi
if [ -n "${MAX_INPUT_LEN:-}" ]; then
  EXTRA_ARGS+=(--max_input_len "${MAX_INPUT_LEN}")
fi
if [ -n "${MAX_TARGET_LEN:-}" ]; then
  EXTRA_ARGS+=(--max_target_len "${MAX_TARGET_LEN}")
fi

"$PYTHON" -u scripts/sft/lora_train.py \
  --model "${MODEL:-google/flan-t5-large}" \
  --train_file "${TRAIN_FILE:-output/training_data/rhyme_only/train.json}" \
  --dev_file "${DEV_FILE:-output/training_data/rhyme_only/dev.json}" \
  --output_root "${OUTPUT_ROOT:-sft/rhyme_only_lora}" \
  --num_train_epochs "${EPOCHS:-3}" \
  --learning_rate "${LR:-1e-4}" \
  --weight_decay "${WD:-0.01}" \
  --warmup_ratio "${WARMUP:-0.1}" \
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

echo "Done. Adapters under OUTPUT_ROOT (default sft/rhyme_only_lora/ when TRAIN_FILE unset)."
