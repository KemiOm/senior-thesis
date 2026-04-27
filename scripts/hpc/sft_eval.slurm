#!/bin/bash
#SBATCH --job-name=eval-ft-grid
#SBATCH --account=span9810
#
# GPU eval: 4 models × 3 tasks = 12 runs of scripts/run_prompt_eval.py
#   (base flan-t5-large + 3 merged LoRA dirs: meter_only, rhyme_only, combined).
#
# Submit from repo root:
#   cd ~/Senior-Thesis && sbatch scripts/hpc/sft_eval.slurm
#
# If Slurm reports "Invalid qos specification", try:
#   sbatch --qos=normal scripts/hpc/sft_eval.slurm
#
# Override merged paths:
#   METER_FT=... RHYME_FT=... COMBINED_FT=... sbatch scripts/hpc/sft_eval.slurm
#
# Shorter smoke test:
#   N=50 sbatch scripts/hpc/sft_eval.slurm
#
# JSON output root (default: results/) — eval generations only, not training weights.
#   SFT_RESULTS=/path/to/dir sbatch scripts/hpc/sft_eval.slurm
#
# Merged weights default to this thesis layout: sft_runs/round2/<task>/final_model_merged
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
    [ -n "$d" ] && [ -f "${d}/scripts/run_prompt_eval.py" ] && THESIS_ROOT="$d" && break
  done
fi
[ -z "$THESIS_ROOT" ] && THESIS_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "$THESIS_ROOT" || { echo "ERROR: cd $THESIS_ROOT"; exit 1; }
if [ ! -f scripts/run_prompt_eval.py ]; then
  echo "ERROR: Cannot find project root (scripts/run_prompt_eval.py). Set THESIS_ROOT=... pwd=$(pwd)"
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
METER_FT="${METER_FT:-$THESIS_ROOT/sft_runs/round2/meter_only/final_model_merged}"
RHYME_FT="${RHYME_FT:-$THESIS_ROOT/sft_runs/round2/rhyme_only/final_model_merged}"
COMBINED_FT="${COMBINED_FT:-$THESIS_ROOT/sft_runs/round2/combined/final_model_merged}"

# zero_shot + line_only matches lora_train.py (raw input only). Use PROMPT=few_shot PROMPT_STYLE=default for Hub-style ICL.
PROMPT="${PROMPT:-zero_shot}"
# Empty PROMPT_STYLE = auto: local checkpoint dirs → line_only; Hub model id → default (needs instructions).
PROMPT_STYLE="${PROMPT_STYLE:-}"
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
    extra_ps=()
    if [ -n "${PROMPT_STYLE}" ]; then
      extra_ps=(--prompt-style "$PROMPT_STYLE")
    elif [ -d "$MODEL" ]; then
      extra_ps=(--prompt-style line_only)
    else
      extra_ps=(--prompt-style default)
    fi
    "$PYTHON" -u scripts/run_prompt_eval.py \
      --results-dir "$SFT_RESULTS" \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      "${extra_ps[@]}" \
      --task "$TASK" \
      --split "$SPLIT" \
      --n "$N" \
      --device "$DEVICE"
  done
done

echo ""
echo "=== Done $(date -Is). Results under ${SFT_RESULTS}/<slug>/ (e.g. zero_shot_*_line_only.json when auto style) ==="
