#!/usr/bin/env bash
# Submit one Slurm job per model so all run in parallel on the cluster.
#
# Prerequisite: same as run_prompt_baseline_all_tasks.slurm (training_data, splits, etc.)
#
# Usage (from project root on the login node):
#   chmod +x scripts/submit_all_model_baselines.sh
#
#   # All models × default PROMPTS; default SPLIT_TASKS=1 and N=10000 (fits ~4h jobs)
#   ./scripts/submit_all_model_baselines.sh
#
#   # Quick smoke test (500 lines per task)
#   N=500 ./scripts/submit_all_model_baselines.sh
#
#   # Legacy: four tasks in one job (often TIMEOUT on 4h partitions — use SPLIT_TASKS=0 only if you have long walltime)
#   SPLIT_TASKS=0 ./scripts/submit_all_model_baselines.sh
#   SPLIT_TASKS=0 N=5000 ./scripts/submit_all_model_baselines.sh
#
#   # Legacy: single prompt
#   PROMPT=few_shot N=500 ./scripts/submit_all_model_baselines.sh
#
#   # Zero-shot only (fewer jobs; ICL needs corpus.db for one_shot/few_shot)
#   PROMPTS=zero_shot ./scripts/submit_all_model_baselines.sh
#
# Default PROMPTS=zero_shot one_shot few_shot (few_shot/one_shot require output/corpus.db on the cluster).
#
# Recommended on Bouchet (see OVERVIEW.MD HPC section + this file’s comments):
#   export HF_TOKEN=hf_xxx          # huggingface.co/settings/tokens — fewer rate-limit warnings
#   export THESIS_HF_HOME=$HOME/scratch/hf_cache   # optional: shared HF cache path (mkdir first)
#   STAGGER_SEC=45 ./scripts/submit_all_model_baselines.sh   # optional: seconds between sbatch calls
#
# Clear old JSONs first (optional):
#   rm -rf evaluation/results/baselines/prompt_only/*
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Defaults tuned for YCRC Bouchet-style limits: day partition often kills jobs at 4h.
# SPLIT_TASKS=1 → one Slurm job per (model × prompt × task) so each run is short enough to finish.
# N=10000 → cap lines per JSON (raise if your QoS allows longer jobs; N=0 = full test set, no --n).
SPLIT_TASKS="${SPLIT_TASKS:-1}"
N="${N:-10000}"

SLURM_SCRIPT="$ROOT/scripts/run_prompt_baseline_all_tasks.slurm"
if [ ! -f "$SLURM_SCRIPT" ]; then
  echo "ERROR: Missing $SLURM_SCRIPT"
  exit 1
fi

# Space-separated. Default = all three prompt types (zero_shot + in-context one_shot/few_shot).
# PROMPT= is still supported if PROMPTS is unset (backward compatible).
# For zero_shot only (faster / fewer jobs): PROMPTS=zero_shot ./scripts/submit_all_model_baselines.sh
PROMPTS="${PROMPTS:-${PROMPT:-zero_shot one_shot few_shot}}"

# ---------------------------------------------------------------------------
# Edit this list: "huggingface_model_id|seq2seq|causal"
# seq2seq = T5, FLAN-T5, BART, PEGASUS, ...
# causal   = GPT-2, LLaMA, Phi, Mistral, ...
# ---------------------------------------------------------------------------
SPECS=(
  "google/flan-t5-small|seq2seq"
  "google/flan-t5-base|seq2seq"
  "google/flan-t5-large|seq2seq"
  "facebook/bart-base|seq2seq"
  "facebook/bart-large|seq2seq"
  "gpt2|causal"
  "gpt2-medium|causal"
  "gpt2-large|causal"
  "microsoft/phi-2|causal"
)

build_export() {
  local model="$1"
  local mtype="$2"
  local prompt="$3"
  local task="${4:-}"
  # ALL = copy submit-shell env into job (so HF_TOKEN, THESIS_HF_HOME, etc. apply if exported)
  local exp="ALL,MODEL=${model},MODEL_TYPE=${mtype},PROMPT=${prompt}"
  if [ -n "${N:-}" ]; then
    exp+=",N=${N}"
  fi
  if [ -n "$task" ]; then
    exp+=",BASELINE_TASK=${task}"
  fi
  if [ -n "${THESIS_HF_HOME:-}" ]; then
    exp+=",THESIS_HF_HOME=${THESIS_HF_HOME}"
  fi
  # STRICT_EVAL=0 to run on full test JSON without quatrain filter (legacy / more lines, noisier gold)
  exp+=",STRICT_EVAL=${STRICT_EVAL:-1}"
  printf '%s' "$exp"
}

# Optional: sleep between sbatch calls to spread Hub downloads (default: no pause)
STAGGER_SEC="${STAGGER_SEC:-0}"
stagger() {
  case "${STAGGER_SEC}" in '' | *[!0-9]*) return ;; esac
  [ "${STAGGER_SEC}" -gt 0 ] && sleep "${STAGGER_SEC}"
}

TASKS=(meter_only rhyme_only natural_text combined)
n_prompts=0
for _ in $PROMPTS; do n_prompts=$((n_prompts + 1)); done
if [ "${SPLIT_TASKS}" = "1" ]; then
  n_task_mul=${#TASKS[@]}
else
  n_task_mul=1
fi
total=$(( ${#SPECS[@]} * n_prompts * n_task_mul ))
if [ "${N}" = "0" ]; then
  n_disp="0 (full test set — needs long walltime per job)"
else
  n_disp="$N"
fi
echo "Submitting $total Slurm jobs | PROMPTS=$PROMPTS | N=$n_disp | SPLIT_TASKS=$SPLIT_TASKS | STAGGER_SEC=${STAGGER_SEC}"
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN is set in this shell (passed via --export=ALL)." || echo "Tip: export HF_TOKEN before running this script for Hub auth."

for PROMPT in $PROMPTS; do
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r MODEL MODEL_TYPE <<< "$spec"
    if [ "${SPLIT_TASKS}" = "1" ]; then
      for BTASK in "${TASKS[@]}"; do
        exp="$(build_export "$MODEL" "$MODEL_TYPE" "$PROMPT" "$BTASK")"
        echo "  -> $MODEL ($MODEL_TYPE) | $PROMPT | $BTASK"
        sbatch --chdir="$ROOT" --export="$exp" "$SLURM_SCRIPT"
        stagger
      done
    else
      exp="$(build_export "$MODEL" "$MODEL_TYPE" "$PROMPT" "")"
      echo "  -> $MODEL ($MODEL_TYPE) | $PROMPT"
      sbatch --chdir="$ROOT" --export="$exp" "$SLURM_SCRIPT"
      stagger
    fi
  done
done

echo "Done. Check: squeue -u \"\$USER\""
echo "Logs: poetry_baseline_all_<jobid>.out in the directory where you ran sbatch (usually project root)."
