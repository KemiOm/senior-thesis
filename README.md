# 18th-Century Poetry Corpus Pipeline

A pipeline for extracting, normalizing, and annotating English poetry from the Eighteenth-Century Poetry Archive (ECPA). Outputs a SQLite corpus with phonology, meter, rhyme, and line-level annotations for training and evaluating language models on formal poetic constraints.

---

## Documentation

- **[OVERVIEW.MD](OVERVIEW.MD)** — Full project description: pipeline stages, terminology (meter, stress, rhyme, TEI), phonology tools (Poesy, Prosodic, CMU), evaluation workflow (splits → prepare data → run baselines → compute metrics → choose model), and HPC usage.
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
4. **Export** — Build SQLite database and run quality checks.

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
├── scripts/                 # export_sqlite, quality_checks, run_prompt_baseline, corpus_tools; scripts/hpc/ = cluster jobs
├── evaluation/              # splits, metrics, summarize_prompt_baselines.py, baseline JSON + CSV outputs
├── sft/                     # Fine-tuned model checkpoints (gitignored weights; see scripts/train_sft_seq2seq_sample.py)
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
python scripts/quality_checks.py
```

Reports: poem/line counts, meter distribution, rhyme coverage, phonology (CMU) coverage, degraded poems, end-stopping, caesura, stanza types.

---

## Evaluation and baselines

After the corpus is built  (1) create train/dev/test splits, (2) prepare task-specific test data (natural_text, meter_only, rhyme_only, combined), (3) run prompt-only baselines on candidate models (e.g. via Slurm on HPC), (4) compute metrics and compare to gold, (5) choose a model for fine-tuning. Details and code paths are in **OVERVIEW.MD** (“Evaluation and choosing a model” and “Running on HPC (summary)”).

**Quick check** that each model has all four task outputs:

```bash
for d in evaluation/results/baselines/prompt_only/*/; do
  n=$(ls "$d"zero_shot_*.json 2>/dev/null | wc -l)
  echo "$d: $n/4 files"
done
```

Results live under `evaluation/results/baselines/prompt_only/<model_slug>/` (e.g. `zero_shot_meter_only.json`). If one task is missing (e.g. 3/4), re-run that task for that model; see OVERVIEW.MD for the single-task command.

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
| `evaluation/results/`     | Baseline run outputs (gitignored; see OVERVIEW.MD)|
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



