#!/usr/bin/env bash
# Re-annotate all normalized poems with Poesy (Prosodic, via sample.phonology_sample) and rebuild corpus.db.
# Requires: poesy + pronouncing (see requirements.txt). No separate prosodic install needed.
#
# Optional: export PROSODIC_NOTEBOOK_STRESS=force  # 1|force|yes|prefer|true — stress field ONLY from
#   stress_from_notebook_style_prosodic (Poesy/CMU not used for stress); meter & meter_type still Poesy (slow)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PHONOLOGY_BATCH="${PHONOLOGY_BATCH:-1}"
python3 batch/phonology_batch.py --force
python3 scripts/export_sqlite.py
echo "Done. Re-run notebooks/01_prepare_training_data.ipynb to refresh output/training_data/."
