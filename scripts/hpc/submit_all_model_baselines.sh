#!/usr/bin/env bash
# Submit one Slurm job per model so all run in parallel on the cluster.
#   # All models × default PROMPTS; default SPLIT_TASKS=1 and N=10000 (fits ~4h jobs)
#   ./scripts/hpc/submit_all_model_baselines.sh
#
# Default PROMPTS=zero_shot one_shot few_shot few_shot/one_shot require output/corpus.db on the cluster
#
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# SPLIT_TASKS=1 → one Slurm job per (model × prompt × task) for easier retries and parallelism.
# N=10000 → cap lines per JSON (N=0 = full test set — needs enough walltime).
# Longer walltime / different partition (bypass 4h day cap): e.g.
#   SLURM_TIME=24:00:00 SLURM_PARTITION=week ./scripts/hpc/submit_all_model_baselines.sh
# (Check your cluster: scontrol show partition; partition names vary.)
SPLIT_TASKS="${SPLIT_TASKS:-1}"
N="${N:-10000}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
SLURM_PARTITION="${SLURM_PARTITION:-}"

SLURM_SCRIPT="$ROOT/scripts/hpc/run_prompt_baseline_all_tasks.slurm"
if [ ! -f "$SLURM_SCRIPT" ]; then
  echo "ERROR: Missing $SLURM_SCRIPT"
  exit 1
fi

# Space-separated. Default = all three prompt types (zero_shot + in-context one_shot/few_shot).
# PROMPT= is still supported if PROMPTS is unset (backward compatible).
# For zero_shot only (faster / fewer jobs): PROMPTS=zero_shot ./scripts/hpc/submit_all_model_baselines.sh
PROMPTS="${PROMPTS:-${PROMPT:-zero_shot one_shot few_shot}}"

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
  # ALL = pass through the submit shell environment (HF_TOKEN, cache paths, etc.)
  local exp="ALL,MODEL=${model},MODEL_TYPE=${mtype},PROMPT=${prompt}"
  if [ -n "${N:-}" ]; then
    exp+=",N=${N}"
  fi
  if [ -n "$task" ]; then
    exp+=",BASELINE_TASK=${task}"
  fi
  local hf_cache="${CLUSTER_HF_HOME:-${THESIS_HF_HOME:-}}"
  if [ -n "$hf_cache" ]; then
    exp+=",CLUSTER_HF_HOME=${hf_cache}"
  fi
  # STRICT_EVAL=0 runs the full test JSON without the quatrain filter (noisier gold, more lines).
  exp+=",STRICT_EVAL=${STRICT_EVAL:-1}"
  printf '%s' "$exp"
}

# Sleep between sbatch calls to spread Hub downloads (default: no pause)
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
echo "Submitting $total Slurm jobs | PROMPTS=$PROMPTS | N=$n_disp | SPLIT_TASKS=$SPLIT_TASKS | time=${SLURM_TIME}${SLURM_PARTITION:+ partition=$SLURM_PARTITION} | STAGGER_SEC=${STAGGER_SEC}"
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN is set in this shell (passed via --export=ALL)." || echo "Tip: export HF_TOKEN before running this script for Hub auth."

for PROMPT in $PROMPTS; do
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r MODEL MODEL_TYPE <<< "$spec"
    if [ "${SPLIT_TASKS}" = "1" ]; then
      for BTASK in "${TASKS[@]}"; do
        exp="$(build_export "$MODEL" "$MODEL_TYPE" "$PROMPT" "$BTASK")"
        echo "  -> $MODEL ($MODEL_TYPE) | $PROMPT | $BTASK"
        sbatch --chdir="$ROOT" --time="$SLURM_TIME" ${SLURM_PARTITION:+--partition="$SLURM_PARTITION"} --export="$exp" "$SLURM_SCRIPT"
        stagger
      done
    else
      exp="$(build_export "$MODEL" "$MODEL_TYPE" "$PROMPT" "")"
      echo "  -> $MODEL ($MODEL_TYPE) | $PROMPT"
      sbatch --chdir="$ROOT" --time="$SLURM_TIME" ${SLURM_PARTITION:+--partition="$SLURM_PARTITION"} --export="$exp" "$SLURM_SCRIPT"
      stagger
    fi
  done
done

echo "Done. Check: squeue -u \"\$USER\""
echo "Logs: poetry_baseline_all_<jobid>.out in the working directory used for sbatch (often the project root)."
