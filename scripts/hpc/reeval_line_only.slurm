#!/bin/bash
#SBATCH --job-name=reeval-line
#SBATCH --account=span9810
#SBATCH --partition=gpu_devel
#SBATCH --qos=normal
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48G
# gpu_devel often caps walltime (e.g. 4h). Override if your QOS allows.
#SBATCH --time=04:00:00
#SBATCH --output=logs/reeval_line_only_%j.out
#SBATCH --error=logs/reeval_line_only_%j.err
#
# Single entry point for line_only re-eval + rollup (see scripts/rerun_eval_line_only_and_score.sh).
#
# Pick *one* way to select what to run:
#
# 1) REEVAL_TARGET (recommended) — maps to SFT rounds or FLAN-only:
#      baseline | flan | hub     — Hub FLAN-T5-large only (SKIP_SFT=1)
#      round1 | round2 | round3 | round4
#      all | all_sft              — every sft_runs/**/final_model_merged
#      r12                        — round1 + round2
#      r123                       — round1 + round2 + round3
#      r1234                      — round1–4
#
# 2) Set SFT_ROUNDS_FILTER yourself (space-separated round names). If set, REEVAL_TARGET is ignored.
#
# Common overrides (any submit):
#   N=500 RESULTS_DIR=... OUT_DIR=... SKIP_BASELINE=1 DEVICE=0 sbatch scripts/hpc/reeval_line_only.slurm
#
# Examples:
#   cd ~/scratch/Senior-Thesis
#   REEVAL_TARGET=flan N=500 sbatch scripts/hpc/reeval_line_only.slurm
#   REEVAL_TARGET=round3 N=500 sbatch scripts/hpc/reeval_line_only.slurm
#   REEVAL_TARGET=r1234 N=500 sbatch scripts/hpc/reeval_line_only.slurm
#   SFT_ROUNDS_FILTER="round3 round4" N=500 sbatch scripts/hpc/reeval_line_only.slurm

set -euo pipefail
export PYTHONUNBUFFERED=1

THESIS_ROOT="${THESIS_ROOT:-}"
if [ -z "$THESIS_ROOT" ]; then
  for d in "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/Senior-Thesis" "$HOME/Senior-Thesis" "."; do
    [ -n "$d" ] && [ -f "${d}/scripts/rerun_eval_line_only_and_score.sh" ] && THESIS_ROOT="$d" && break
  done
fi
[ -z "$THESIS_ROOT" ] && THESIS_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "$THESIS_ROOT" || { echo "ERROR: cd $THESIS_ROOT"; exit 1; }

mkdir -p logs

if [ -d ".venv" ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
elif [ -d "venv" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

export HF_HOME="${HF_HOME:-${HOME}/scratch/hf}"
mkdir -p "${HF_HOME}/transformers"

export DEVICE="${DEVICE:-0}"

# Map REEVAL_TARGET → SFT_ROUNDS_FILTER / SKIP_SFT (unless user already set SFT_ROUNDS_FILTER).
effective="${REEVAL_TARGET:-r12}"
rt_lc="$(printf '%s' "$effective" | tr '[:upper:]' '[:lower:]')"
if [ -z "${SFT_ROUNDS_FILTER:-}" ]; then
  unset SKIP_SFT
  case "$rt_lc" in
    baseline|flan|hub)
      export SKIP_SFT=1
      ;;
    round1) export SFT_ROUNDS_FILTER="round1" ;;
    round2) export SFT_ROUNDS_FILTER="round2" ;;
    round3) export SFT_ROUNDS_FILTER="round3" ;;
    round4) export SFT_ROUNDS_FILTER="round4" ;;
    all|all_sft)
      unset SFT_ROUNDS_FILTER
      ;;
    r12) export SFT_ROUNDS_FILTER="round1 round2" ;;
    r123) export SFT_ROUNDS_FILTER="round1 round2 round3" ;;
    r1234) export SFT_ROUNDS_FILTER="round1 round2 round3 round4" ;;
    *)
      echo "ERROR: Unknown REEVAL_TARGET='$REEVAL_TARGET' (effective='$effective'). Use baseline, round1–4, all, r12, r123, r1234, or set SFT_ROUNDS_FILTER."
      exit 1
      ;;
  esac
else
  unset SKIP_SFT
fi

slug="$rt_lc"
if [ -n "${SKIP_SFT:-}" ]; then
  slug="baseline_flan"
elif [ -n "${SFT_ROUNDS_FILTER:-}" ] && [ -z "${REEVAL_TARGET:-}" ]; then
  slug="custom"
fi
export RESULTS_DIR="${RESULTS_DIR:-$THESIS_ROOT/results/reeval_${slug}_line_only}"
export OUT_DIR="${OUT_DIR:-$THESIS_ROOT/evaluation/baseline_report_reeval_${slug}}"

echo "THESIS_ROOT=$THESIS_ROOT"
echo "REEVAL_TARGET=${REEVAL_TARGET:-} SFT_ROUNDS_FILTER=${SFT_ROUNDS_FILTER:-} SKIP_SFT=${SKIP_SFT:-}"
echo "N=${N:-} RESULTS_DIR=$RESULTS_DIR OUT_DIR=$OUT_DIR DEVICE=$DEVICE"
hostname
nvidia-smi || true

bash scripts/rerun_eval_line_only_and_score.sh

echo "Done $(date -Is)"
