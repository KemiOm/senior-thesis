# Evaluation protocol (gold labels, metrics, prompts, splits)

Short reference for **metrics**, **where labels come from**, and **how to keep baseline comparisons fair**. File paths point to the code.

---

## 0. Evaluation layout (when you need each)

**`evaluation/corpus/`** — gold / SQLite (no model generations):

| Module | Role | Typical use |
|--------|------|---------------|
| `corpus/splits.py` | Fixed poem IDs for train / dev / test / held-out → `evaluation/splits/*.json` | `python evaluation/splits.py` |
| `corpus/metrics.py` | SQLite aggregation for **annotation coverage** (not model accuracy) | Used by `corpus/coverage.py` |
| `corpus/coverage.py` | CLI → `evaluation/annotation_coverage.json` | `python evaluation/run_annotation_coverage.py` |

**`evaluation/scoring/`** — model JSON outputs:

| Module | Role | Typical use |
|--------|------|---------------|
| `scoring/slug.py` | Short result directory name from Hub id or checkpoint path | `run_prompt_baseline.py` |
| `scoring/struct_metrics.py` | Field-level partial metrics (rhyme / combined bundles) | Rollup + baseline script |
| `scoring/form_eval.py` | CMU form / stress for **natural_text** | Rollup, `corpus_tools.py` |
| `scoring/rollup.py` | Walk `*/*.json` trees → `baseline_report/model_comparison.csv` | `python evaluation/summarize_prompt_baselines.py` |

Shims under `evaluation/` keep those two-word CLI paths stable; implementation lives in `corpus/` and `scoring/`.

---

## 1. What “gold” means

**Gold** = labels produced by the annotation pipeline from the source poems. They live in `output/corpus.db` (SQLite) and are exported into line-level JSON for training and evaluation (`output/training_data/{task}/…`).

| Concept | Plain-language meaning | Where it shows up |
|--------|-------------------------|-------------------|
| **Stress pattern** | Strong vs weak syllables along the line, as a string (e.g. `+` / `-` or `0`/`1` encodings, normalized to a common form). Used for **meter-only** targets and the meter part of **combined** targets. | DB: `lines.stress`; meter-only supervision: `evaluation/corpus/metrics.py`, `scripts/run_prompt_baseline.py` (`stress_to_plus_minus`) |
| **Meter type** | High-level label from Poesy/Prosodic (e.g. whether the line is treated as iambic pentameter vs unknown). | DB: `lines.meter_type` |
| **Rhyme group** | Letter (or symbol) grouping lines that rhyme together in the poem’s scheme. | DB: `lines.rhyme_group` |
| **Rhyme key (phonetic)** | Derived from **ARPAbet** phonology: tail phones from the **last stressed vowel** (same rule for training and evaluation). Used for **rhyme-only** targets and form checks. | `scripts/run_prompt_baseline.py` (`rhyme_key_from_phonology`), `evaluation/scoring/form_eval.py` |
| **Phonology / CMU** | Pronunciation entries per word; lines are flagged when a word is missing from CMU (`not_found`). | DB: `lines.phonology` JSON |
| **End-stopped** | Whether the line ends at a sentence-like boundary (punctuation heuristic). | DB: `lines.end_stopped` |
| **Caesura** | Position of a strong mid-line pause (when present). | DB: `lines.caesura` |

**Annotation coverage metrics** (how complete the corpus labels are, not model accuracy): `evaluation/corpus/metrics.py` → `compute_metrics` plus `compute_corpus_diagnostics` (meter-type histogram, lines with CMU gaps, poems that are all-unknown meter, enjambment counts, stanza-type counts). `python evaluation/run_annotation_coverage.py` runs `evaluation/corpus/coverage.py` and merges both into `evaluation/annotation_coverage.json` for the full corpus and each split file that exists. This supersedes the old `scripts/quality_checks.py` printout.

---

## 2. Train / dev / test (and held-out) boundaries

**Poem-level splits** are defined once in `evaluation/splits.py` (fixed seed **42** for reproducibility):

| Split | Purpose |
|-------|---------|
| **train** | Used to train models. |
| **dev** | Used for tuning and model selection. |
| **test** | Final in-domain evaluation (unseen poems in this split). |
| **held_out_poets** | Authors held out from train/dev/test (tests new-author generalization). |
| **held_out_poems** | Poems held out for unseen-poem generalization (authors may appear elsewhere). |

Files: `evaluation/splits/{train,dev,test,held_out_poets,held_out_poems}.json` (poem IDs) + `meta.json`.

**Line-level JSON** for seq2seq (`output/training_data/{task}/train.json`, `dev.json`, `test.json`) is built from the corpus **after** splits exist (`notebooks/01_prepare_training_data.ipynb`). Each row includes `input` and `target` (and baseline JSONs store the gold value as `gold_target`).

For prompt baselines, `scripts/run_prompt_baseline.py` loads `output/training_data/{task}/{split}.json` (default `--split test`).

---

## 3. Four conditions (tasks)

| Task | Typical `input` | What the model predicts (`gold_target` / `target`) |
|------|-----------------|------------------------------------------------------|
| **meter_only** | One line of verse | Stress / meter pattern (e.g. `+/-` string). |
| **rhyme_only** | One line of verse | Rhyme key from the phonology pipeline. |
| **natural_text** | Prior line(s) in the **same poem** (or `[start]`) | **Next line** surface text (continuation). |
| **combined** | **Same as natural_text** | **Next line’s** bundled string (`meter:…|rhyme:…|end:…|caesura:…`). Rows may include `next_line` for strict-eval tooling. |

**Comparability rule:** Rows come from the same corpus split; **natural_text** and **combined** share the same continuation `input` per position. Compare models on the same task and `n_scored`.

---

## 4. Prompt templates (prompt-only baselines)

Implemented in `scripts/run_prompt_baseline.py` (`build_prompt`).

| Variant | Meaning |
|---------|---------|
| **zero_shot** | Instructions + the current line only (no example). |
| **one_shot** | One example pair (input → target format). |
| **few_shot** | Multiple lines, e.g. a **quatrain** example where labels match the same format as training data (`FEW_SHOT_SEPARATOR`). |

**Defaults:** `--split test`, plus `--model_type {seq2seq,causal}` to switch encoder–decoder vs causal models.



---

## 5. Metrics for model outputs (baseline comparisons)

Scoring uses saved JSON with **per-line** `gold_target` and `model_output` (as produced by `run_prompt_baseline`).

### 5.1 Exact string match (all tasks)

`evaluation/summarize_prompt_baselines.py` → `exact_match_rate`: after task-specific whitespace normalization, **percentage of lines where `model_output` equals `gold_target`**. Skips rows with empty gold.

- **meter_only / natural_text / rhyme_only / combined:** all get **`exact_match_pct`** for the primary headline comparison.

### 5.2 Natural text — form metrics (phonology-based)

For natural_text outputs, CMU-based form checks compare the **gold line** to the **predicted line** (`evaluation/scoring/form_eval.py`):

- **`nt_form_stress_match_pct`**: stress pattern match (evaluable lines only).
- **`nt_form_syllable_match_pct`**: syllable count match.
- **`nt_form_rhyme_match_pct`**: rhyme-key match, restricted to rows where the gold rhyme key is non-empty (denominator in `nt_form_rhyme_denom`).

### 5.3 Rhyme-only — relaxed token match

`evaluation/scoring/struct_metrics.py` → `st_rhyme_relaxed_match_pct`: case-insensitive, whitespace-normalized rhyme token agreement.

### 5.4 Combined — per-field rates

Same module → `st_combined_*`: parse success, per-field match (meter with `+/-` vs `01` normalization), **all-four** match rate, etc.

---

## 6. Running prompt baselines on Bouchet (YCRC)

Baseline jobs use **CPU** inference (`--device -1`) in `scripts/hpc/run_prompt_baseline_all_tasks.slurm`, matching your other poetry-baseline runs: **`partition=day`**, account **`span9810`**, **`--qos=normal`** when your association requires it. Submit from the repo root so paths and `evaluation/baselines/…` resolve. GPU SFT eval grids (`scripts/hpc/run_eval_ft_grid.slurm`) write **eval JSON** under `results/<short_slug>/` by default (see `evaluation/scoring/slug.py`); that is separate from training checkpoints under your LoRA **`OUTPUT_ROOT`** (Slurm defaults to `sft/<task>_lora/`).

**Environment**

- **`HF_TOKEN`**: export on the login node before `sbatch` if a model is gated (e.g. Meta Llama). Slurm passes your environment when using `--export=ALL` (as in `submit_all_model_baselines.sh`).
- **Venv**: `source venv/bin/activate` in the job or before submit.

**Meta Llama 3 8B — `meter_only` + `rhyme_only`, few-shot only**

```bash
cd ~/Senior-Thesis   # or: cd "$HOME/Senior Thesis"
source venv/bin/activate
export HF_TOKEN=…   # Hugging Face token with access to meta-llama/Meta-Llama-3-8B-Instruct

ONLY_MODEL_SPEC='meta-llama/Meta-Llama-3-8B-Instruct|causal' \
ONLY_TASKS="meter_only rhyme_only" \
PROMPTS=few_shot \
N=500 \
SLURM_PARTITION=day \
./scripts/hpc/submit_all_model_baselines.sh
```

Adjust **`N`** (omit for full test; use a longer **`SLURM_TIME`** in that script or override **`SLURM_TIME=24:00:00`** if needed). After jobs finish: `python evaluation/summarize_prompt_baselines.py` → refreshes `evaluation/baseline_report/model_comparison.csv`.

**Logs:** `poetry_baseline_all_<jobid>.out` in the project directory (or the directory you passed to `--chdir`).
