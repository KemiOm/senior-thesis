#!/bin/bash
#SBATCH --job-name=eval-ckpt
#SBATCH --account=span9810
#SBATCH --partition=gpu_devel
#SBATCH --qos=normal
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_ckpt_%j.out
#SBATCH --error=logs/eval_ckpt_%j.err
#
# Single run of run_prompt_eval.py on one checkpoint or merged dir.
# Submit from repo root with env vars (do not paste this file into the shell line-by-line):
#
#   MODEL=/path/to/checkpoint-NNNN TASK=meter_only PROMPT=zero_shot MAX_TOKENS=16 \
#     sbatch scripts/hpc/eval_ckpt.slurm
#
#   MODEL=/path/to/final_model_merged TASK=combined PROMPT=zero_shot MAX_TOKENS=64 \
#     sbatch scripts/hpc/eval_ckpt.slurm
#
# Default PROMPT_STYLE=line_only (raw JSON input, matches SFT). Use PROMPT_STYLE=default for long prompts.
#
# Optional: N=200 SFT_RESULTS=~/Senior-Thesis/results THESIS_ROOT=~/Senior-Thesis

set -euo pipefail
export PYTHONUNBUFFERED=1

THESIS_ROOT="${THESIS_ROOT:-}"
if [ -z "$THESIS_ROOT" ]; then
  for d in "${SLURM_SUBMIT_DIR:-}" "$HOME/Senior-Thesis" "$HOME/Senior Thesis" "."; do
    [ -n "$d" ] && [ -f "${d}/scripts/run_prompt_eval.py" ] && THESIS_ROOT="$d" && break
  done
fi
[ -z "$THESIS_ROOT" ] && THESIS_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "$THESIS_ROOT" || { echo "ERROR: cd $THESIS_ROOT"; exit 1; }
if [ ! -f scripts/run_prompt_eval.py ]; then
  echo "ERROR: Cannot find project root. Set THESIS_ROOT=... pwd=$(pwd)"
  exit 1
fi

if [ -d ".venv" ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
elif [ -d "venv" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

MODEL="${MODEL:-}"
TASK="${TASK:-}"
PROMPT="${PROMPT:-zero_shot}"
PROMPT_STYLE="${PROMPT_STYLE:-line_only}"
N="${N:-500}"
MAX_TOKENS="${MAX_TOKENS:-64}"
DEVICE="${DEVICE:-0}"
SPLIT="${SPLIT:-test}"
SFT_RESULTS="${SFT_RESULTS:-$THESIS_ROOT/results}"

if [ -z "$MODEL" ]; then
  echo "ERROR: Set MODEL to a checkpoint or merged weights directory, e.g."
  echo "  MODEL=\$HOME/scratch/Senior-Thesis/sft_full/.../checkpoint-NNNN   # or repo: sft_runs/round3/.../checkpoint-NNNN"
  exit 1
fi
if [ -z "$TASK" ]; then
  echo "ERROR: Set TASK=meter_only | rhyme_only | combined | natural_text"
  exit 1
fi
if [ ! -d "$MODEL" ]; then
  echo "ERROR: MODEL is not a directory: $MODEL"
  exit 1
fi

mkdir -p logs "$SFT_RESULTS"

HF_CACHE_ROOT="${CLUSTER_HF_HOME:-${THESIS_HF_HOME:-}}"
if [ -z "${HF_CACHE_ROOT:-}" ] && [ -d "${HOME}/scratch" ]; then
  HF_CACHE_ROOT="${HOME}/scratch/Senior-Thesis/hf"
fi
if [ -n "${HF_CACHE_ROOT:-}" ]; then
  export HF_HOME="$HF_CACHE_ROOT"
  export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
  mkdir -p "$TRANSFORMERS_CACHE" || true
  echo "HF_HOME=$HF_HOME"
fi

echo "MODEL=$MODEL"
echo "TASK=$TASK PROMPT=$PROMPT PROMPT_STYLE=$PROMPT_STYLE N=$N MAX_TOKENS=$MAX_TOKENS DEVICE=$DEVICE"
echo "SFT_RESULTS=$SFT_RESULTS"

python3 -u scripts/run_prompt_eval.py \
  --results-dir "$SFT_RESULTS" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --prompt-style "$PROMPT_STYLE" \
  --task "$TASK" \
  --split "$SPLIT" \
  --n "$N" \
  --device "$DEVICE" \
  --max_tokens "$MAX_TOKENS"

echo "Done eval_ckpt job ${SLURM_JOB_ID:-local}"
