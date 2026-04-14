#!/bin/bash
#
# Slurm batch script: run scripts/run_prompt_baseline.py on a compute node.
#
# The login node is for editing and submitting jobs; heavy model runs go through
# Slurm so work runs on compute nodes with a time limit and logged output.
# Docs: https://slurm.schedmd.com/sbatch.html
# Yale cluster overview: https://docs.ycrc.yale.edu/
#
#SBATCH --job-name=poetry-baseline-all
#SBATCH --account=span9810
# Override partition when submitting: sbatch --partition=week ... or SLURM_PARTITION=week
#SBATCH --partition=day
# Default walltime; submit_all_model_baselines.sh passes --time
# On Bouchet, week often needs walltime >= 24h; day is for shorter jobs
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=poetry_baseline_all_%j.out
#
# Needs: output/training_data/{task}/test.json from notebooks/01_prepare_training_data.ipynb
#
# One job: sbatch --chdir=$(pwd) scripts/hpc/run_prompt_baseline_all_tasks.slurm
# Many jobs: ./scripts/hpc/submit_all_model_baselines.sh

# Find project root: Slurm sets SLURM_SUBMIT_DIR; fallbacks cover common clone paths.
for d in "${SLURM_SUBMIT_DIR}" "$HOME/Senior-Thesis" "$HOME/Senior Thesis" "."; do
  [ -n "$d" ] && [ -f "${d}/scripts/run_prompt_baseline.py" ] && cd "$d" && break
done
if [ ! -f "scripts/run_prompt_baseline.py" ]; then
  echo "ERROR: Cannot find project root (scripts/run_prompt_baseline.py). SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  exit 1
fi

# Prefer local venv; else conda base if available.
if [ -d "venv" ]; then
  source venv/bin/activate
elif command -v conda &>/dev/null; then
  conda activate base
fi
# Minimal deps on the compute node if the image lacks them; ignore failure if already present.
pip install --quiet transformers torch accelerate sentencepiece tiktoken nltk 2>/dev/null || true

# Shared Hugging Face cache on cluster storage (set on login node or in job prologue).
HF_CACHE_ROOT="${CLUSTER_HF_HOME:-${THESIS_HF_HOME:-}}"
if [ -n "${HF_CACHE_ROOT:-}" ]; then
  export HF_HOME="$HF_CACHE_ROOT"
  export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
  mkdir -p "$TRANSFORMERS_CACHE" 2>/dev/null || true
  echo "Using HF_HOME=$HF_HOME"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set (authenticated Hub requests)"
else
  echo "Note: export HF_TOKEN on the login node for higher Hub rate limits."
fi

# Job env from sbatch --export (see submit_all_model_baselines.sh).
MODEL=${MODEL:-google/flan-t5-base}
PROMPT=${PROMPT:-zero_shot}
MODEL_TYPE=${MODEL_TYPE:-seq2seq}
N=${N:-}
STRICT_EVAL=${STRICT_EVAL:-1}

# Flags for run_prompt_baseline.py: causal vs seq2seq; strict eval when requested.
EXTRA_ARGS=()
[ "$MODEL_TYPE" = "causal" ] && EXTRA_ARGS+=(--model_type causal)
if [ "${STRICT_EVAL:-0}" = "1" ]; then
  EXTRA_ARGS+=(--strict-eval)
fi

# BASELINE_TASK set => one task only; unset => loop all four (used when SPLIT_TASKS=0).
if [ -n "${BASELINE_TASK:-}" ]; then
  TASK_LIST=("$BASELINE_TASK")
else
  TASK_LIST=(meter_only rhyme_only natural_text combined)
fi

# CPU inference (--device -1); optional --n caps test lines when N is non-empty.
for TASK in "${TASK_LIST[@]}"; do
  echo "=============================================="
  echo "Running $MODEL | $PROMPT | $TASK | $MODEL_TYPE"
  echo "=============================================="
  if [ -n "$N" ]; then
    python scripts/run_prompt_baseline.py --model "$MODEL" --prompt "$PROMPT" --task "$TASK" --n "$N" --device -1 "${EXTRA_ARGS[@]}"
  else
    python scripts/run_prompt_baseline.py --model "$MODEL" --prompt "$PROMPT" --task "$TASK" --device -1 "${EXTRA_ARGS[@]}"
  fi
  echo ""
done

echo "Done. Results in evaluation/baselines/<model_slug>/"
