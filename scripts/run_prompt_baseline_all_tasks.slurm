#!/bin/bash
#SBATCH --job-name=poetry-baseline-all
#SBATCH --account=span9810
#SBATCH --partition=day
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=poetry_baseline_all_%j.out
# NOTE: No GPU requested - runs on CPU. For GPU, use partition=gpu_h200 and add: #SBATCH --gres=gpu:1
#
# Run all four training conditions (meter_only, rhyme_only, natural_text, combined) in one job.
# Prerequisite: 01_prepare_training_data.ipynb has been run so output/training_data/{task}/test.json exist.
#
# Submit from project root:
#   sbatch --chdir=$(pwd) scripts/run_prompt_baseline_all_tasks.slurm
# Optional env vars (defaults shown):
#   MODEL=google/flan-t5-base  PROMPT=zero_shot  N=500  MODEL_TYPE=seq2seq
#   MODEL=gpt2-medium  MODEL_TYPE=causal  N=500   # decoder-only (GPT-2, LLaMA, etc.)
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
  pip install --quiet transformers torch 2>/dev/null || true
elif command -v conda &>/dev/null; then
  conda activate base
fi

MODEL=${MODEL:-google/flan-t5-base}
PROMPT=${PROMPT:-zero_shot}
MODEL_TYPE=${MODEL_TYPE:-seq2seq}
# N empty = full test set; set N=500 for a quick run
N=${N:-}

EXTRA_ARGS=()
[ "$MODEL_TYPE" = "causal" ] && EXTRA_ARGS+=(--model_type causal)

for TASK in meter_only rhyme_only natural_text combined; do
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
