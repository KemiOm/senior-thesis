# 18th-Century Poetry Corpus Pipeline

A pipeline for extracting, normalizing, and annotating English poetry from the Eighteenth-Century Poetry Archive (ECPA). Outputs a SQLite corpus with phonology, meter, rhyme, and line-level annotations for training and evaluating language models on formal poetic constraints.

---

## Documentation

- **[OVERVIEW.MD](OVERVIEW.MD)** — Full project description: pipeline stages, terminology (meter, stress, rhyme, TEI), phonology tools (Poesy, Prosodic, CMU), evaluation workflow (splits → training JSONs → baselines → metrics → **SFT** → checkpoint eval), structured metrics, and **HPC / Bouchet** usage (`day` vs `gpu_devel`, QoS, checkpoints vs `final_model`).
- **Evaluation** — Splits, metrics, and prompt-only baselines: see [evaluation/EVALUATION_PROTOCOL.md](evaluation/EVALUATION_PROTOCOL.md) and the “Evaluation and choosing a model” section in OVERVIEW.MD.

---

## Source Data

**Primary corpus:** **Eighteenth-Century Poetry Archive (ECPA)** — English-language poetry from the long eighteenth century (1660–1800).

- **ECPA GitHub repository:** https://github.com/alhuber1502/ECPA  
- **ECPA website:** https://www.eighteenthcenturypoetry.org/  
- **License:** CC BY-SA (Creative Commons Attribution-ShareAlike)

The pipeline expects TEI P5 XML files under `ECPA/web/works/<poem_id>/<poem_id>.xml`. Clone or download the ECPA repository and place it in the project root.


---

## Overview

The pipeline has four stages:

1. **Extract** — Parse TEI XML into structured JSON (metadata, stanzas, lines).
2. **Normalize** — Standardize quotes, dashes, apostrophes, and whitespace.
3. **Annotate** — Add phonology (CMU/ARPAbet), meter, rhyme, end-stopping, caesura, enjambment.
4. **Export** — Build SQLite database and run annotation coverage (replaces a separate quality-check script).

---

## Prerequisites

- **Python 3.9+**
- **espeak** (required for Poesy/Prosodic phonology):
  ```bash
  brew install espeak   # macOS
  ```
- **NLTK** (downloaded automatically on first phonology run)

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/KemiOm/senior-thesis.git
cd senior-thesis
```

### 2. Clone the ECPA corpus

```bash
git clone https://github.com/alhuber1502/ECPA.git
```

The `ECPA/` folder must sit in the project root. Poems are read from `ECPA/web/works/<poem_id>/<poem_id>.xml`.

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── sample/                  # Single-poem scripts (for testing)
│   ├── extract_sample.py
│   ├── normalize_sample.py
│   └── phonology_sample.py
├── batch/                   # Full-corpus scripts
│   ├── extract_batch.py
│   ├── normalize_batch.py
│   └── phonology_batch.py
├── notebooks/               # Training data prep, evaluation metrics, SFT overview
├── scripts/                 # export_sqlite, run_prompt_eval, corpus_tools; scripts/sft/lora_train.py; scripts/hpc/ = Slurm
├── evaluation/              # corpus/ (splits, coverage), scoring/ (rollup, slugs, metrics-on-JSON), baselines/, baseline_report/
├── sft/                     # Default LoRA training root (--output-root); heavy dirs gitignored
├── results/                 # SFT **eval** JSON only (`run_prompt_eval.py` on merged weights); not checkpoints
├── data/                    # Optional local data (samples, metadata)
├── docs/                    # Data overview, debug transcripts
├── requirements.txt
└── ECPA/                    # Source data (clone separately, not in repo)
    └── web/works/           # Poem XMLs
```

---

## Running the Pipeline

Run from the **project root**. Sample scripts process a few poems for verification; batch scripts process the full corpus. Use the sample scripts first to validate outputs before running the full batch.

### Step 1: Extract (TEI XML → JSON)

```bash
# Test on 2 sample poems
python sample/extract_sample.py

# Full corpus
python batch/extract_batch.py
```

Output: `output/poems/<id>.json`

### Step 2: Normalize

```bash
python sample/normalize_sample.py   # Sample
python batch/normalize_batch.py     # Full corpus
```

Output: `output/poems_normalized/<id>.json`

### Step 3: Annotate 

```bash
python sample/phonology_sample.py   # Sample
python batch/phonology_batch.py     # Full corpus
```

Batch options:
- `--limit N` — Process only the first N poems
- `--skip-existing` — Skip poems that already have output

Output: `output/poems_annotated/<id>.json`

### Step 4: Export to SQLite

```bash
python scripts/export_sqlite.py
```

Output: `output/corpus.db`

### Step 5: Quality checks

```bash
python evaluation/run_annotation_coverage.py
```

Reports: poem/line counts, meter distribution, rhyme coverage, phonology (CMU) coverage, degraded poems, end-stopping, caesura, stanza types.

---

## Evaluation and baselines

After the corpus is built: (1) splits, (2) task-specific train/dev/test JSONs, (3) prompt-only baselines, (4) metrics, (5) supervised fine-tuning per task and evaluation of checkpoints. Details and code paths are in **OVERVIEW.MD** (evaluation flow, **Supervised fine-tuning**, and **Running on HPC**).

**Quick check** that each model has all four task outputs:

```bash
for d in evaluation/baselines/*/; do
  n=$(ls "$d"zero_shot_*.json 2>/dev/null | wc -l)
  echo "$d: $n/4 files"
done
```

Results live under `evaluation/baselines/<model_slug>/` for pretrained prompt baselines, and `results/<short_slug>/` for **SFT eval** JSON from `scripts/hpc/sft_eval.slurm` (e.g. `few_shot_meter_only.json`). Training **weights** live under whatever `--output-root` you used (Slurm defaults to `sft/<task>_lora/`); you do not need both `sft/` and `sft_full/`—the latter is only an alternate folder name some runs used. Short slugs come from `evaluation/scoring/slug.py`. If one task is missing (e.g. 3/4), re-run that task for that model; see OVERVIEW.MD for the single-task command.

---

## Output Structure

| Directory                  | Contents                                          |
|---------------------------|---------------------------------------------------|
| `output/poems/`           | Extracted JSON (raw text)                         |
| `output/poems_normalized/`| Normalized JSON                                   |
| `output/poems_annotated/` | JSON with phonology, meter, rhyme                 |
| `output/corpus.db`        | SQLite database                                   |
| `output/training_data/`   | Task-specific train/dev/test JSONs (from notebook)|
| `evaluation/splits/`      | Train/dev/test and held-out poem ID lists         |
| `evaluation/baselines/`   | Prompt-eval JSON per `<model_slug>/` (see `run_prompt_eval.py`) |
| `evaluation/baseline_report/` | Roll-up tables: `model_comparison.csv`, selection notes (`summarize_prompt_baselines.py`) |
| `results/`                | SFT **eval** JSON per `<short_slug>/` (not weights); roll-up with `summarize_prompt_baselines.py --baseline-dir results --out-dir results` |
| `sft/`                      | Default **`OUTPUT_ROOT`** for LoRA: checkpoints, adapters, merged weights (often gitignored). Same role as any other output root (e.g. `sft_full/`), not a separate pipeline. |
| `data/nltk_data/`         | NLTK data (created automatically)                 |

---

## Database Schema

- **poems** — `id`, `author`, `title`
- **stanzas** — `poem_id`, `stanza_index`, `stanza_type`, `rhyme_scheme`, `rhyme_pairs`
- **lines** — `poem_id`, `stanza_index`, `line_index`, `raw`, `normalized`, `rhyme_word`, `rhyme_group`, `meter_type`, `meter`, `stress`, `end_stopped`, `caesura`, `enjambment`, `phonology`

Views: `v_lines_with_poem`, `v_poems_by_meter`, and others for common queries.

---

## Dependencies

Key libraries:

- **lxml** — TEI XML parsing
- **pronouncing** — CMU Pronouncing Dictionary (ARPAbet)
- **Poesy** — Prosodic-based rhyme and meter (https://github.com/quadrismegistus/poesy)
- **trafilatura** (and related) — Text extraction utilities

---



