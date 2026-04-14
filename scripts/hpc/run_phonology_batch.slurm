#!/bin/bash
#SBATCH --job-name=phonology-batch
#SBATCH --account=span9810
# CPU + RAM: phonology uses Prosodic/Poesy (no GPU). Login-node OOM is common; use a compute node + large --mem.
# Partitions / limits: https://docs.ycrc.yale.edu/clusters/bouchet/#partitions-and-hardware
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# Bouchet `day` often caps walltime (e.g. 12h). Do not exceed partition max or sbatch rejects.
# Longer runs: use `week` and e.g. #SBATCH --time=2-00:00:00 (see YCRC Bouchet partition limits).
#SBATCH --time=12:00:00
#SBATCH --output=phonology_batch_%j.out
#SBATCH --error=phonology_batch_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

THESIS_ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/Senior-Thesis}}"
cd "$THESIS_ROOT" || { echo "ERROR: cd $THESIS_ROOT"; exit 1; }

if [ -d "venv" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

PYTHON="${THESIS_ROOT}/venv/bin/python"
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi

echo "=== phonology_batch start $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname -s) ==="
echo "THESIS_ROOT=$THESIS_ROOT  pwd=$(pwd)"

# PROSODIC_NOTEBOOK_STRESS=1 → stress *only* from single-line Prosodic (no CMU fallback). If Bouchet yields 0% non-empty stress, Prosodic is failing — rerun with PROSODIC_NOTEBOOK_STRESS=0 for Poesy+CMU stress.
export PROSODIC_NOTEBOOK_STRESS="${PROSODIC_NOTEBOOK_STRESS:-1}"
export PHONOLOGY_BATCH="${PHONOLOGY_BATCH:-1}"
export MAX_LINE_CHARS="${MAX_LINE_CHARS:-2000}"
# Parallelism: 0 = sequential (safest). On a fat node try 4–8:
export PHONOLOGY_PARALLEL="${PHONOLOGY_PARALLEL:-0}"
export PHONOLOGY_MAX_WORKERS="${PHONOLOGY_MAX_WORKERS:-4}"

echo "PROSODIC_NOTEBOOK_STRESS=$PROSODIC_NOTEBOOK_STRESS  MAX_LINE_CHARS=$MAX_LINE_CHARS"
echo "PHONOLOGY_PARALLEL=$PHONOLOGY_PARALLEL  PHONOLOGY_MAX_WORKERS=$PHONOLOGY_MAX_WORKERS"

EXTRA=(--force)
# Smoke test:  LIMIT=50 sbatch --export=ALL,LIMIT=50 scripts/hpc/run_phonology_batch.slurm
if [ -n "${LIMIT:-}" ] && [ "${LIMIT}" != "0" ]; then EXTRA+=(--limit "${LIMIT}"); fi

"$PYTHON" -u batch/phonology_batch.py "${EXTRA[@]}"

echo "=== done $(date -Is) ==="
echo "Optional next step: python3 scripts/export_sqlite.py"
