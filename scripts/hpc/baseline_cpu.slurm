#!/bin/bash
#
# Slurm batch script to run scripts/run_prompt_eval.py on a compute node.
# Heavy jobs should run through Slurm (not the login node) to ensure proper
# resource allocation, time limits, and logging.
#
#SBATCH --job-name=poetry-baseline-all
#SBATCH --account=span9810
# Default partition (can override at submit time)
#SBATCH --partition=day
# Default walltime (can override via sbatch)
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=poetry_baseline_all_%j.out
#
# Requires: output/training_data/{task}/test.json
# Run:
#   sbatch --chdir=$(pwd) scripts/hpc/baseline_cpu.slurm
#   ./scripts/hpc/submit_baselines.sh  (for batch submission)

# Locate project root (prioritize SLURM_SUBMIT_DIR, then common paths)
for d in "${SLURM_SUBMIT_DIR}" "$HOME/Senior-Thesis" "$HOME/Senior Thesis" "."; do
  [ -n "$d" ] && [ -f "${d}/scripts/run_prompt_eval.py" ] && cd "$d" && break
done

if [ ! -f "scripts/run_prompt_eval.py" ]; then
  echo "ERROR: Could not locate project root (missing scripts/run_prompt_eval.py)"
  exit 1
fi

# Activate environment: prefer venv, fallback to conda base
if [ -d "venv" ]; then
  source venv/bin/activate
elif command -v conda &>/dev/null; then
  conda activate base
fi

# Ensure required packages (skip if already installed)
pip install --quiet transformers torch accelerate sentencepiece tiktoken nltk 2>/dev/null || true

# Optional: configure shared Hugging Face cache
HF_CACHE_ROOT="${CLUSTER_HF_HOME:-${THESIS_HF_HOME:-}}"
if [ -n "${HF_CACHE_ROOT:-}" ]; then
  export HF_HOME="$HF_CACHE_ROOT"
  export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
  mkdir -p "$TRANSFORMERS_CACHE" 2>/dev/null || true
  echo "Using HF cache: $HF_HOME"
fi

# Optional: authentication for higher Hub rate limits
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN detected"
else
  echo "HF_TOKEN not set (may hit rate limits)"
fi

# Job parameters (can be overridden via sbatch --export)
MODEL=${MODEL:-google/flan-t5-base}
PROMPT=${PROMPT:-zero_shot}
MODEL_TYPE=${MODEL_TYPE:-seq2seq}
N=${N:-}
STRICT_EVAL=${STRICT_EVAL:-1}

# Build argument list
EXTRA_ARGS=()
[ "$MODEL_TYPE" = "causal" ] && EXTRA_ARGS+=(--model_type causal)
[ "$STRICT_EVAL" = "1" ] && EXTRA_ARGS+=(--strict-eval)

# Task selection: single task if BASELINE_TASK set, else run all
if [ -n "${BASELINE_TASK:-}" ]; then
  TASK_LIST=("$BASELINE_TASK")
else
  TASK_LIST=(meter_only rhyme_only natural_text combined)
fi

# Run evaluation (CPU mode; optional --n limit)
for TASK in "${TASK_LIST[@]}"; do
  echo "=============================================="
  echo "Running: $MODEL | $PROMPT | $TASK | $MODEL_TYPE"
  echo "=============================================="

  if [ -n "$N" ]; then
    python scripts/run_prompt_eval.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --task "$TASK" \
      --n "$N" \
      --device -1 \
      "${EXTRA_ARGS[@]}"
  else
    python scripts/run_prompt_eval.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --task "$TASK" \
      --device -1 \
      "${EXTRA_ARGS[@]}"
  fi

  echo ""
done

echo "Done. Results saved to evaluation/baselines/<model_slug>/"
