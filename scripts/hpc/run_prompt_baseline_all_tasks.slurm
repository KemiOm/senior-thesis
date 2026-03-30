#!/bin/bash
#SBATCH --job-name=poetry-baseline-all
#SBATCH --account=span9810
#SBATCH --partition=day
# YCRC Bouchet `day` QoS often enforces a 4h walltime cap even when the partition MaxTime is longer.
# Match that here so expectations match sacct. For longer limits: sbatch --time=12:00:00 ...
#SBATCH --time=04:00:00
# submit_all_model_baselines.sh now defaults to SPLIT_TASKS=1 and N=10000 so each job usually finishes within this window.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=poetry_baseline_all_%j.out
# Run all four training conditions (meter_only, rhyme_only, natural_text, combined) in one job.
# Prerequisite: 01_prepare_training_data.ipynb has been run so output/training_data/{task}/test.json exist.
#
# Submit from project root:
#   sbatch --chdir=$(pwd) scripts/hpc/run_prompt_baseline_all_tasks.slurm
#
# Run ALL candidate models in parallel (one job per model):
#   rm -rf evaluation/results/baselines/prompt_only/*   # optional: clear old JSONs
#   ./scripts/hpc/submit_all_model_baselines.sh
#   # Full test (no line cap — needs long walltime / SPLIT_TASKS=1 strongly recommended):  N=0 ./scripts/hpc/submit_all_model_baselines.sh
#   # or with a sample cap:  N=500 ./scripts/hpc/submit_all_model_baselines.sh
#   # or:  PROMPT=few_shot N=500 ./scripts/hpc/submit_all_model_baselines.sh
#   # or:  PROMPTS="zero_shot one_shot few_shot" N=500 ./scripts/hpc/submit_all_model_baselines.sh
#
# To run full test set for all tasks, omit N or set N= (empty).

for d in "${SLURM_SUBMIT_DIR}" "$HOME/Senior-Thesis" "$HOME/Senior Thesis" "."; do
  [ -n "$d" ] && [ -f "${d}/scripts/run_prompt_baseline.py" ] && cd "$d" && break
done
if [ ! -f "scripts/run_prompt_baseline.py" ]; then
  echo "ERROR: Cannot find project root (scripts/run_prompt_baseline.py). SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  exit 1
fi

if [ -d "venv" ]; then
  source venv/bin/activate
elif command -v conda &>/dev/null; then
  conda activate base
fi
# Some models (e.g. certain tokenizers / phi) need extras; install quietly if missing.
pip install --quiet transformers torch accelerate sentencepiece tiktoken 2>/dev/null || true

# Shared Hugging Face cache on the cluster (before sbatch: export CLUSTER_HF_HOME=/path/to/scratch/hf_cache)
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

MODEL=${MODEL:-google/flan-t5-base}
PROMPT=${PROMPT:-zero_shot}
MODEL_TYPE=${MODEL_TYPE:-seq2seq}
# N from submit script (default 10000). Empty = full test. N=0 = full test (passed as --n 0; see run_prompt_baseline.py).
N=${N:-}
# STRICT_EVAL=1 (default): only lines in reliable 4-line stanzas + regold from corpus.db (see run_prompt_baseline.py --strict-eval)
STRICT_EVAL=${STRICT_EVAL:-1}

EXTRA_ARGS=()
[ "$MODEL_TYPE" = "causal" ] && EXTRA_ARGS+=(--model_type causal)
if [ "${STRICT_EVAL:-0}" = "1" ]; then
  EXTRA_ARGS+=(--strict-eval)
fi

# If BASELINE_TASK is set (e.g. SPLIT_TASKS=1 submit), run only that task (smaller jobs / easier retries).
if [ -n "${BASELINE_TASK:-}" ]; then
  TASK_LIST=("$BASELINE_TASK")
else
  TASK_LIST=(meter_only rhyme_only natural_text combined)
fi

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

echo "Done. Results in evaluation/results/baselines/prompt_only/<model_slug>/"
