#!/usr/bin/env bash
#
# Calls sbatch once per baseline run. Each job runs scripts/hpc/run_prompt_baseline_all_tasks.slurm
# with MODEL, PROMPT, BASELINE_TASK, N, STRICT_EVAL, etc. passed via --export.
#
# SPLIT_TASKS=1 (default): one job per (model x prompt x task). Easier retries and parallel runs.
# SPLIT_TASKS=0: one job per (model x prompt); all four tasks run inside that single job.
#
# N: max lines per test JSON. N=0 means full test set; needs long walltime.
#
# Partitions (Bouchet / YCRC—confirm on current docs):
#   day: shorter jobs; QoS may cap walltime around a few hours.
#   week: often requires walltime >= 24h; shorter requests can be rejected. This script bumps
#   12:00:00 and similar to 24:00:00 when SLURM_PARTITION=week.
#
# Slurm: https://slurm.schedmd.com/sbatch.html
# Cluster: https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/
# Cluster: https://docs.ycrc.yale.edu/clusters/bouchet/
# Examples:
#   ./scripts/hpc/submit_all_model_baselines.sh
#   N=500 SLURM_PARTITION=day ./scripts/hpc/submit_all_model_baselines.sh
#   SLURM_PARTITION=week ./scripts/hpc/submit_all_model_baselines.sh
#   SLURM_TIME=12:00:00 SLURM_PARTITION=day ./scripts/hpc/submit_all_model_baselines.sh
#   N=500 ONLY_TASKS="natural_text combined" SLURM_PARTITION=day ./scripts/hpc/submit_all_model_baselines.sh
#   ONLY_MODEL_SPEC='meta-llama/Meta-Llama-3-8B-Instruct|causal' N=500 ./scripts/hpc/submit_all_model_baselines.sh
#
# Exit on first error, unset variables, or pipe failure.
set -euo pipefail
# Repo root (parent of scripts/) so sbatch --chdir and paths stay correct.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SPLIT_TASKS="${SPLIT_TASKS:-1}"
N="${N:-10000}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
# week partition: bump short default walltimes to 24h so the scheduler accepts the job.
if [ "${SLURM_PARTITION}" = "week" ]; then
  case "${SLURM_TIME}" in
  12:00:00|04:00:00|4:00:00|08:00:00|8:00:00|06:00:00|6:00:00)
    echo "submit_all_model_baselines: partition week needs walltime >= 24h. Using SLURM_TIME=24:00:00 (was ${SLURM_TIME})." >&2
    SLURM_TIME=24:00:00
    ;;
  esac
fi

SLURM_SCRIPT="$ROOT/scripts/hpc/run_prompt_baseline_all_tasks.slurm"
if [ ! -f "$SLURM_SCRIPT" ]; then
  echo "ERROR: Missing $SLURM_SCRIPT"
  exit 1
fi

# Prompt styles: PROMPT overrides PROMPTS if set (single value); else list all three.
PROMPTS="${PROMPTS:-${PROMPT:-zero_shot one_shot few_shot}}"

# model|MODEL_TYPE pairs for seq2seq vs causal LM baselines.
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
  "meta-llama/Meta-Llama-3-8B-Instruct|causal"
)

# Optional: run a single "model|type" instead of the full list (debug or one-off).
if [ -n "${ONLY_MODEL_SPEC:-}" ]; then
  SPECS=("${ONLY_MODEL_SPEC}")
fi

# Builds the sbatch --export string: MODEL, optional N and BASELINE_TASK, cache path, STRICT_EVAL.
build_export() {
  local model="$1"
  local mtype="$2"
  local prompt="$3"
  local task="${4:-}"
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
  exp+=",STRICT_EVAL=${STRICT_EVAL:-1}"
  printf '%s' "$exp"
}

# Optional delay between sbatch calls to reduce burst load on the scheduler or shared FS.
STAGGER_SEC="${STAGGER_SEC:-0}"
stagger() {
  case "${STAGGER_SEC}" in '' | *[!0-9]*) return ;; esac
  [ "${STAGGER_SEC}" -gt 0 ] && sleep "${STAGGER_SEC}"
}

# ONLY_TASKS: space-separated subset; default is all four baseline tasks.
if [ -n "${ONLY_TASKS:-}" ]; then
  # shellcheck disable=SC2206
  TASKS=(${ONLY_TASKS})
else
  TASKS=(meter_only rhyme_only natural_text combined)
fi
# Count words in PROMPTS for total job estimate (handles multi-word prompt names if added later).
n_prompts=0
for _ in $PROMPTS; do n_prompts=$((n_prompts + 1)); done
if [ "${SPLIT_TASKS}" = "1" ]; then
  n_task_mul=${#TASKS[@]}
else
  n_task_mul=1
fi
# Expected number of sbatch invocations (for the summary line).
total=$(( ${#SPECS[@]} * n_prompts * n_task_mul ))
if [ "${N}" = "0" ]; then
  n_disp="0 (full test; needs long walltime)"
else
  n_disp="$N"
fi
echo "Submitting $total Slurm jobs | PROMPTS=$PROMPTS | N=$n_disp | SPLIT_TASKS=$SPLIT_TASKS | time=${SLURM_TIME}${SLURM_PARTITION:+ partition=$SLURM_PARTITION} | STAGGER_SEC=${STAGGER_SEC}${ONLY_MODEL_SPEC:+ | ONLY_MODEL_SPEC=1 model}"
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN is set in this shell (passed via --export=ALL)." || echo "Tip: export HF_TOKEN before running for Hub auth."

for PROMPT in $PROMPTS; do
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r MODEL MODEL_TYPE <<< "$spec"
    if [ "${SPLIT_TASKS}" = "1" ]; then
      # One Slurm job per task; BASELINE_TASK selects a single task inside the .slurm script.
      for BTASK in "${TASKS[@]}"; do
        exp="$(build_export "$MODEL" "$MODEL_TYPE" "$PROMPT" "$BTASK")"
        echo "  -> $MODEL ($MODEL_TYPE) | $PROMPT | $BTASK"
        sbatch --chdir="$ROOT" --time="$SLURM_TIME" ${SLURM_PARTITION:+--partition="$SLURM_PARTITION"} --export="$exp" "$SLURM_SCRIPT"
        stagger
      done
    else
      # One job runs all tasks: omit BASELINE_TASK so the .slurm script loops over four tasks.
      exp="$(build_export "$MODEL" "$MODEL_TYPE" "$PROMPT" "")"
      echo "  -> $MODEL ($MODEL_TYPE) | $PROMPT"
      sbatch --chdir="$ROOT" --time="$SLURM_TIME" ${SLURM_PARTITION:+--partition="$SLURM_PARTITION"} --export="$exp" "$SLURM_SCRIPT"
    fi
  done
done

echo "Done. Check: squeue -u \"\$USER\""
echo "Logs: poetry_baseline_all_<jobid>.out in the project root (or sbatch working directory)."
