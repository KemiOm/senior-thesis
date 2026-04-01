# Evaluation protocol (gold labels, metrics, prompts, splits)

Short reference for **metrics**, **where labels come from**, and **how to keep baseline comparisons fair**. File paths point to the code.

---

## 1. What “gold” means

**Gold** = labels produced by the annotation pipeline from the source poems. They live in `output/corpus.db` (SQLite) and are exported into line-level JSON for training and evaluation (`output/training_data/{task}/…`).

| Concept | Plain-language meaning | Where it shows up |
|--------|-------------------------|-------------------|
| **Stress pattern** | Strong vs weak syllables along the line, as a string (e.g. `+` / `-` or `0`/`1` encodings, normalized to a common form). Used for **meter-only** targets and the meter part of **combined** targets. | DB: `lines.stress`; meter-only supervision: `evaluation/metrics.py`, `scripts/run_prompt_baseline.py` (`stress_to_plus_minus`) |
| **Meter type** | High-level label from Poesy/Prosodic (e.g. whether the line is treated as iambic pentameter vs unknown). | DB: `lines.meter_type` |
| **Rhyme group** | Letter (or symbol) grouping lines that rhyme together in the poem’s scheme. | DB: `lines.rhyme_group` |
| **Rhyme key (phonetic)** | Derived from **ARPAbet** phonology: tail phones from the **last stressed vowel** (same rule for training and evaluation). Used for **rhyme-only** targets and form checks. | `scripts/run_prompt_baseline.py` (`rhyme_key_from_phonology`), `evaluation/form_eval_generation.py` |
| **Phonology / CMU** | Pronunciation entries per word; lines are flagged when a word is missing from CMU (`not_found`). | DB: `lines.phonology` JSON |
| **End-stopped** | Whether the line ends at a sentence-like boundary (punctuation heuristic). | DB: `lines.end_stopped` |
| **Caesura** | Position of a strong mid-line pause (when present). | DB: `lines.caesura` |

**Annotation coverage metrics** (how complete the corpus labels are, not model accuracy): `evaluation/metrics.py` → `compute_metrics` — coverage of meter, stress, rhyme, CMU, end-stopping, caesura. Run on the full DB or on a **list of poem IDs** (e.g. a split). `evaluation/run_baseline.py` writes `evaluation/baseline_results.json` for the full corpus plus each split file that exists.

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

For natural_text outputs, CMU-based form checks compare the **gold line** to the **predicted line** (`evaluation/form_eval_generation.py`):

- **`nt_form_stress_match_pct`**: stress pattern match (evaluable lines only).
- **`nt_form_syllable_match_pct`**: syllable count match.
- **`nt_form_rhyme_match_pct`**: rhyme-key match, restricted to rows where the gold rhyme key is non-empty (denominator in `nt_form_rhyme_denom`).

### 5.3 Rhyme-only — relaxed token match

`evaluation/structured_baseline_metrics.py` → `st_rhyme_relaxed_match_pct`: case-insensitive, whitespace-normalized rhyme token agreement.

### 5.4 Combined — per-field rates

Same module → `st_combined_*`: parse success, per-field match (meter with `+/-` vs `01` normalization), **all-four** match rate, etc.

