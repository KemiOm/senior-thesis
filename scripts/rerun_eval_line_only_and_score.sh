#!/usr/bin/env bash
# Re-run prompt eval after prompt-style + scoring changes:
#   - Every sft_runs/**/final_model_merged: zero_shot + line_only (matches SFT training inputs).
#   - Hub baseline (google/flan-t5-large by default): zero_shot + default (instructions help cold Hub).
# Then regenerate model_comparison.csv (includes Levenshtein columns from rollup).
#
# From repo root:
#   bash scripts/rerun_eval_line_only_and_score.sh
# Smoke (fast):
#   N=25 DEVICE=-1 bash scripts/rerun_eval_line_only_and_score.sh
# Roll up SFT JSONs into results/ (do not set OUT_DIR to evaluation/baseline_report —
# that folder is for pretrained baselines from evaluation/baselines only):
#   RESULTS_DIR="$PWD/results" OUT_DIR="$PWD/results" bash scripts/rerun_eval_line_only_and_score.sh
# Baseline + round 1 and 2 SFT only:
#   SFT_ROUNDS_FILTER="round1 round2" bash scripts/rerun_eval_line_only_and_score.sh
#
# Env:
#   RESULTS_DIR   (default: <repo>/results/reeval_line_only)
#   OUT_DIR       (default: <repo>/evaluation/baseline_report_reeval_line_only)
#   N             if set, passed as --n (cap lines per run)
#   SPLIT         (default: test)
#   DEVICE        (default: 0; use -1 for CPU)
#   BASE_MODEL    (default: google/flan-t5-large)
#   BASE_PROMPT   (default: zero_shot)
#   BASE_PROMPT_STYLE (default: default)
#   PYTHON        (default: python3)
#   SFT_ROUNDS_FILTER  e.g. "round1 round2" — only those sft_runs/<round>/**/final_model_merged;
#                      unset = all rounds under sft_runs/
#   SKIP_SFT=1         — skip all LoRA merges (Hub baseline + rollup only; use for FLAN baseline-only)
#   SKIP_BASELINE=1  — only run SFT merges + rollup (no Hub baseline)

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Match scripts/hpc/reeval_line_only.slurm: prefer project venv over system python3.
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
  elif [ -x "$ROOT/venv/bin/python" ]; then
    PYTHON="$ROOT/venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

if ! "$PYTHON" -c "import transformers, torch" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON cannot import transformers/torch."
  echo "  Create or activate a venv under this repo (see .venv/ or venv/), e.g.:"
  echo "    python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install transformers torch accelerate peft"
  exit 1
fi
echo "Using PYTHON=$PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/reeval_line_only}"
OUT_DIR="${OUT_DIR:-$ROOT/evaluation/baseline_report_reeval_line_only}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-0}"
BASE_MODEL="${BASE_MODEL:-google/flan-t5-large}"
BASE_PROMPT="${BASE_PROMPT:-zero_shot}"
BASE_PROMPT_STYLE="${BASE_PROMPT_STYLE:-default}"

mkdir -p "$RESULTS_DIR" "$OUT_DIR"

n_args=()
if [ -n "${N:-}" ]; then
  n_args=(--n "$N")
fi

if [ -n "${SKIP_SFT:-}" ]; then
  echo "=== SKIP_SFT set — skipping LoRA / merged checkpoints ==="
else
  echo "=== Re-eval SFT (line_only) → $RESULTS_DIR ==="
  tmpf="$(mktemp)"
  : >"$tmpf"
  if [ -n "${SFT_ROUNDS_FILTER:-}" ]; then
    for r in $SFT_ROUNDS_FILTER; do
      if [ -d "$ROOT/sft_runs/$r" ]; then
        # Symlinks to merged weights are -type l, not -type d (GNU find).
        find "$ROOT/sft_runs/$r" \( -type d -o -type l \) -name final_model_merged 2>/dev/null >>"$tmpf" || true
      else
        echo "WARN: no directory $ROOT/sft_runs/$r"
      fi
    done
  else
    find "$ROOT/sft_runs" \( -type d -o -type l \) -name final_model_merged 2>/dev/null >>"$tmpf" || true
  fi
  LC_ALL=C sort -u "$tmpf" -o "$tmpf"
  while IFS= read -r merged; do
    [ -n "$merged" ] || continue
    [ -f "$merged/config.json" ] || { echo "WARN: skip (no config): $merged"; continue; }
    for task in meter_only rhyme_only combined; do
      echo ""
      echo "--- model=$merged task=$task (zero_shot line_only) ---"
      "$PYTHON" -u scripts/run_prompt_eval.py \
        --results-dir "$RESULTS_DIR" \
        --model "$merged" \
        --prompt zero_shot \
        --prompt-style line_only \
        --task "$task" \
        --split "$SPLIT" \
        --device "$DEVICE" \
        "${n_args[@]}"
    done
  done <"$tmpf"
  rm -f "$tmpf"
fi

if [ -z "${SKIP_BASELINE:-}" ]; then
  echo ""
  echo "=== Re-eval baseline Hub ($BASE_MODEL, $BASE_PROMPT $BASE_PROMPT_STYLE) → $RESULTS_DIR ==="
  for task in meter_only rhyme_only combined; do
    echo ""
    echo "--- model=$BASE_MODEL task=$task ---"
    "$PYTHON" -u scripts/run_prompt_eval.py \
      --results-dir "$RESULTS_DIR" \
      --model "$BASE_MODEL" \
      --prompt "$BASE_PROMPT" \
      --prompt-style "$BASE_PROMPT_STYLE" \
      --task "$task" \
      --split "$SPLIT" \
      --device "$DEVICE" \
      "${n_args[@]}"
  done
else
  echo ""
  echo "=== SKIP_BASELINE set — skipping Hub baseline runs ==="
fi

echo ""
echo "=== Rollup CSV → $OUT_DIR ==="
"$PYTHON" "$ROOT/evaluation/summarize_prompt_baselines.py" \
  --baseline-dir "$RESULTS_DIR" \
  --out-dir "$OUT_DIR"

echo "Done. Open: $OUT_DIR/model_comparison.csv"
