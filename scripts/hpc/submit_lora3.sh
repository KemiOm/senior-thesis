#!/usr/bin/env bash
# Submit three LoRA jobs: meter_only, rhyme_only, combined (no natural_text).
#
# Uses scripts/hpc/lora_train.slurm → scripts/sft/lora_train.py for each task.
# Prerequisites: output/training_data/<task>/{train,dev}.json from notebooks/01_prepare_training_data.ipynb
#
# Usage (from repo root on Bouchet):
#   bash scripts/hpc/submit_lora3.sh
#
# Optional env (applied to every submission), e.g. fp32 if you see NaN grads:
#   NO_FP16=1 NO_BF16=1 MERGE_AND_SAVE=1 bash scripts/hpc/submit_lora3.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SLURM="${SLURM:-scripts/hpc/lora_train.slurm}"
if [[ ! -f "$SLURM" ]]; then
  echo "ERROR: missing $SLURM"
  exit 1
fi

for TASK in meter_only rhyme_only combined; do
  echo "========== Submitting LoRA: $TASK =========="
  sbatch --job-name="lora-${TASK}" \
    --export=ALL,"TRAIN_FILE=output/training_data/${TASK}/train.json","DEV_FILE=output/training_data/${TASK}/dev.json","OUTPUT_ROOT=sft_runs/${TASK}_lora" \
    "$SLURM"
done

echo "Done. Three jobs submitted: meter_only, rhyme_only, combined."
echo "Outputs: sft_runs/<task>_lora/ under the repo."
