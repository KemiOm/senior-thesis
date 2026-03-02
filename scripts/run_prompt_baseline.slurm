#!/bin/bash
#SBATCH --job-name=poetry-baseline
#SBATCH --account=span9810
#SBATCH --partition=day
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=poetry_baseline_%j.out
# NOTE: No GPU requested - runs on CPU. For GPU, use partition=gpu_h200 and add: #SBATCH --gres=gpu:1

# Run from project root. SLURM_SUBMIT_DIR can sometimes point to spool on Bouchet.
# Try submit dir first, then common locations. Submit with: sbatch --chdir=$(pwd) ... for reliability.
for d in "${SLURM_SUBMIT_DIR}" "$HOME/Senior-Thesis" "$HOME/Senior Thesis" "."; do
  [ -n "$d" ] && [ -f "${d}/scripts/run_prompt_baseline.py" ] && cd "$d" && break
done
if [ ! -f "scripts/run_prompt_baseline.py" ]; then
  echo "ERROR: Cannot find project root (scripts/run_prompt_baseline.py). SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  exit 1
fi

# Activate env
if [ -d "venv" ]; then
  source venv/bin/activate
  # Install deps on compute node (login node has limited mem - pip install torch gets Killed)
  pip install --quiet transformers torch 2>/dev/null || true
elif command -v conda &>/dev/null; then
  conda activate base
fi

# Default: flan-t5-large, zero_shot, meter_only, 500 samples
# Submit from project root: sbatch scripts/run_prompt_baseline.slurm
# Or explicitly set working dir: sbatch --chdir=$(pwd) scripts/run_prompt_baseline.slurm
# Or edit MODEL, PROMPT, TASK, N below
MODEL=${MODEL:-google/flan-t5-large}
PROMPT=${PROMPT:-zero_shot}
TASK=${TASK:-meter_only}
N=${N:-500}

python scripts/run_prompt_baseline.py \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --task "$TASK" \
  --n "$N" \
  --device -1
