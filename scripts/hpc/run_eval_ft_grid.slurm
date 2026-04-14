#!/bin/bash
#SBATCH --job-name=eval-ft-grid
#SBATCH --account=span9810
#
# GPU eval: 4 models × 3 tasks = 12 runs of scripts/run_prompt_baseline.py
#   (base flan-t5-large + 3 merged LoRA dirs: meter_only, rhyme_only, combined).
#
# Submit from repo root:
#   cd ~/Senior-Thesis && sbatch scripts/hpc/run_eval_ft_grid.slurm
#
# If Slurm reports "Invalid qos specification", try:
#   sbatch --qos=normal scripts/hpc/run_eval_ft_grid.slurm
#
# Override merged paths:
#   METER_FT=... RHYME_FT=... COMBINED_FT=... sbatch scripts/hpc/run_eval_ft_grid.slurm
#
# Shorter smoke test:
#   N=50 sbatch scripts/hpc/run_eval_ft_grid.slurm
#
# JSON output root (default: results/) — eval generations only, not training weights.
#   SFT_RESULTS=/path/to/dir sbatch scripts/hpc/run_eval_ft_grid.slurm
#
# Merged weights below default to sft/...;
#
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=eval_ft_grid_%j.out
#SBATCH --error=eval_ft_grid_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

THESIS_ROOT="${THESIS_ROOT:-}"
if [ -z "$THESIS_ROOT" ]; then
  for d in "${SLURM_SUBMIT_DIR:-}" "$HOME/Senior-Thesis" "$HOME/Senior Thesis" "."; do
    [ -n "$d" ] && [ -f "${d}/scripts/run_prompt_baseline.py" ] && THESIS_ROOT="$d" && break
  done
fi
[ -z "$THESIS_ROOT" ] && THESIS_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "$THESIS_ROOT" || { echo "ERROR: cd $THESIS_ROOT"; exit 1; }
if [ ! -f scripts/run_prompt_baseline.py ]; then
  echo "ERROR: Cannot find project root (scripts/run_prompt_baseline.py). Set THESIS_ROOT=... pwd=$(pwd)"
  exit 1
fi

if [ -d "venv" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
elif command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
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

BASE="${BASE_MODEL:-google/flan-t5-large}"
METER_FT="${METER_FT:-$THESIS_ROOT/sft/meter_only_lora/flan-t5-large_lora_20260411_174009/final_model_merged}"
RHYME_FT="${RHYME_FT:-$THESIS_ROOT/sft/rhyme_only_lora/flan-t5-large_lora_20260411_174908/final_model_merged}"
COMBINED_FT="${COMBINED_FT:-$THESIS_ROOT/sft/combined_lora/flan-t5-large_lora_20260411_175033/final_model_merged}"

PROMPT="${PROMPT:-few_shot}"
N="${N:-500}"
DEVICE="${DEVICE:-0}"
SPLIT="${SPLIT:-test}"

# SFT eval JSONs default to results/ (not evaluation/baselines/).
SFT_RESULTS="${SFT_RESULTS:-$THESIS_ROOT/results}"
mkdir -p "$SFT_RESULTS"

for d in "$METER_FT" "$RHYME_FT" "$COMBINED_FT"; do
  if [ ! -f "$d/config.json" ]; then
    echo "WARN: missing $d/config.json — check METER_FT/RHYME_FT/COMBINED_FT paths."
  fi
done

echo "=== eval_ft_grid start $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname -s) ==="
echo "BASE_MODEL=$BASE"
echo "PROMPT=$PROMPT SPLIT=$SPLIT N=$N DEVICE=$DEVICE"
echo "SFT_RESULTS=$SFT_RESULTS"

if [ -x "${THESIS_ROOT}/venv/bin/python" ]; then
  PYTHON="${THESIS_ROOT}/venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
"$PYTHON" -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count())" || true

RUN=0
for MODEL in "$BASE" "$METER_FT" "$RHYME_FT" "$COMBINED_FT"; do
  for TASK in meter_only rhyme_only combined; do
    RUN=$((RUN + 1))
    echo ""
    echo "========== [$RUN/12] MODEL=$MODEL TASK=$TASK =========="
    "$PYTHON" -u scripts/run_prompt_baseline.py \
      --results-dir "$SFT_RESULTS" \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --task "$TASK" \
      --split "$SPLIT" \
      --n "$N" \
      --device "$DEVICE"
  done
done

echo ""
echo "=== Done $(date -Is). Results: ${SFT_RESULTS}/<slug>/${PROMPT}_<task>.json ==="
