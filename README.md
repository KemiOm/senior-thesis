# 18th-Century Poetry Corpus Pipeline

A pipeline for extracting, normalizing, and annotating English poetry from the Eighteenth-Century Poetry Archive (ECPA). Outputs a SQLite corpus with phonology, meter, rhyme, and line-level annotations for training and evaluating language models on formal poetic constraints.

---

## Documentation

- **[OVERVIEW.MD](OVERVIEW.MD)** — Full project description: pipeline stages, terminology (meter, stress, rhyme, TEI), phonology tools (Poesy, Prosodic, CMU), evaluation workflow (splits → training JSONs → baselines → metrics → **SFT** → checkpoint eval), structured metrics, and **HPC / Bouchet** usage (partitions, QoS, checkpoints vs final merged weights).
- **Evaluation** — Splits, annotation coverage, scoring, and baselines: code under **evaluation/** (see layout below). Roll-up tables and model-selection notes live in **evaluation/baseline_report/** (**model_comparison.csv**, **MODEL_SELECTION.MD**). Stable CLI entry points at the **evaluation/** package root (**splits.py**, **run_annotation_coverage.py**, **summarize_prompt_baselines.py**) are thin shims; implementations are in **evaluation/corpus/** and **evaluation/scoring/**. Step-by-step narrative: “Evaluation and choosing a model” and **Running on HPC** in OVERVIEW.MD.

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

That file includes corpus / extraction pins and **SFT** stack (`transformers`, `peft`, etc.). Install a matching **PyTorch** build for your machine or cluster separately if `pip` does not pull one you want.

---

## Project Structure

High-level layout. Large or machine-local artifacts are often **gitignored** (check **.gitignore**): e.g. most of **output/** except **corpus.db** and **training_data/**, **ECPA/**, **venv/**, checkpoint/weight trees under **sft/**, **sft_full/**, and **sft_runs/**, plus **visualizations/out/**.

```
.
├── OVERVIEW.MD              # Long-form pipeline + evaluation + HPC narrative
├── README.md
├── requirements.txt
├── sample/                  # Single-poem: extract_sample, normalize_sample, phonology_sample
├── batch/                   # Full corpus: extract_batch, normalize_batch, phonology_batch
├── notebooks/               # 01_prepare_training_data.ipynb → output/training_data/{task}/{split}.json
├── scripts/
│   ├── export_sqlite.py     # Annotated poem JSON → output/corpus.db
│   ├── run_prompt_eval.py   # Prompt-based eval on task JSON (Hub or merged checkpoints)
│   ├── eval_cli.py          # Baseline CSV filters, corpus.db alignment checks, natural-text form tools
│   ├── sft/lora_train.py    # LoRA / QLoRA fine-tuning
│   └── hpc/                 # Slurm + shell helpers (baseline_cpu, sft_eval, lora_train, submit_*.sh)
├── evaluation/
│   ├── corpus/              # Splits, annotation coverage, SQLite metrics (implementation)
│   ├── scoring/             # Rollup, slugs, form_eval, struct_metrics (implementation)
│   ├── splits/              # Poem ID lists: train.json, test.json, held-out sets, meta.json
│   ├── baselines/           # Per-model slug dirs of prompt-eval JSON (when versioned)
│   ├── baseline_report/     # model_comparison.csv, MODEL_SELECTION.MD, filtered CSVs
│   ├── splits.py            # CLI: python evaluation/splits.py
│   ├── run_annotation_coverage.py
│   └── summarize_prompt_baselines.py
├── output/                  # corpus.db, training_data/; intermediate poem JSON trees usually not in git
├── results/                 # SFT eval JSON per short slug (local or cluster; not training weights)
├── sft/, sft_full/          # Alternate LoRA output roots from training jobs (choose one convention)
├── sft_runs/                # Experiment registry (round-based runs, params/metrics, selected merged models)
├── visualizations/          # plot_ft.py, ft_figures.ipynb → visualizations/out/ (figures gitignored)
├── data/, docs/             # Optional local data; notes / figure sources
└── ECPA/                    # Source TEI corpus (clone separately; not tracked by default)
    └── web/works/<id>/<id>.xml
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

After the corpus is built: (1) splits, (2) task-specific train/dev/test JSONs, (3) prompt-based evaluation (pretrained or merged checkpoints), (4) metrics and rollups, (5) supervised fine-tuning per task and checkpoint eval. Details and code paths are in **OVERVIEW.MD** (evaluation flow, **Supervised fine-tuning**, and **Running on HPC**).

**Quick check** that each model has all four task outputs:

```bash
for d in evaluation/baselines/*/; do
  n=$(ls "$d"zero_shot_*.json 2>/dev/null | wc -l)
  echo "$d: $n/4 files"
done
```

Results live under `evaluation/baselines/<model_slug>/` for pretrained prompt baselines, and `results/<short_slug>/` for **SFT eval** JSON from `scripts/hpc/sft_eval.slurm` (e.g. `few_shot_meter_only.json`). Training **weights** live under whatever `--output-root` you used (Slurm defaults to `sft/<task>_lora/`); you do not need both `sft/` and `sft_full/`—the latter is only an alternate folder name some runs used. Short slugs come from `evaluation/scoring/slug.py`. If one task is missing (e.g. 3/4), re-run that task for that model; see OVERVIEW.MD for the single-task command.

---

### `sft_full` vs `sft_runs` (do you need both?)

Short answer: **no, you do not need both**.

- Keep `sft_runs/` if you want the round-based experiment history used for model selection (`run_params.json`, `final_eval_metrics.json`, and selected merged checkpoints).
- Keep `results/` for evaluation JSON used in rollups/tables/figures.
- `sft_full/` is only an alternate training output root used by some older runs. If equivalent checkpoints are already represented in `sft_runs/` (or uploaded to Hugging Face), `sft_full/` can be archived or removed.

Recommended thesis setup:
- Keep: `results/` + lightweight metadata in `sft_runs/`.
- Optional local-only: heavyweight checkpoints/adapters.
- Do not maintain duplicate weight trees in both `sft_full/` and `sft_runs/` unless you need redundancy.

---

## Publishing best checkpoints to Hugging Face

Use merged checkpoints (`final_model_merged/`) for Hub upload so each repo loads directly with `AutoModelForSeq2SeqLM.from_pretrained(...)`.

Recommended paths currently used for the three task repos:

- `combined`: `sft_runs/round1/combined/final_model_merged`
- `meter_only`: `sft_runs/round3/meter_only_lr1e4/final_model_merged`
- `rhyme_only`: `sft_runs/round2/rhyme_only/final_model_merged`

### One-time auth setup

```bash
source venv/bin/activate
python -m pip install "huggingface-hub>=0.34,<1.0"
hf auth login
```

Token scope should include write/create permissions for model repos under the `KemiOm` user namespace.

### Create repos (once)

```bash
hf repo create KemiOm/poetry-combined-best --repo-type model --exist-ok
hf repo create KemiOm/poetry-meter-best --repo-type model --exist-ok
hf repo create KemiOm/poetry-rhyme-best --repo-type model --exist-ok
```

### Upload merged folders

```bash
hf upload-large-folder KemiOm/poetry-combined-best "sft_runs/round1/combined/final_model_merged" --repo-type=model
hf upload-large-folder KemiOm/poetry-meter-best "sft_runs/round3/meter_only_lr1e4/final_model_merged" --repo-type=model
hf upload-large-folder KemiOm/poetry-rhyme-best "sft_runs/round2/rhyme_only/final_model_merged" --repo-type=model
```

Notes:
- Some `hf` versions require `--repo-type=model` for `upload-large-folder` or the command fails.
- Run uploads sequentially (not in parallel) and rerun the same command to resume interrupted uploads.
- If your `hf repo` subcommand does not support `files`, verify via the Hub web UI or Python:

```bash
python -c "from huggingface_hub import list_repo_files; print(list_repo_files('KemiOm/poetry-rhyme-best', repo_type='model'))"
```

---

## Output Structure

| Location | Contents |
|----------|----------|
| output/poems/ | Extracted JSON (raw text); often local-only |
| output/poems_normalized/ | Normalized JSON |
| output/poems_annotated/ | JSON with phonology, meter, rhyme |
| output/corpus.db | SQLite database (tracked when committed for review) |
| output/training_data/ | Per-task train.json, dev.json, test.json (from notebook) |
| evaluation/splits/ | Train/dev/test and held-out poem ID lists |
| evaluation/annotation_coverage.json | Gold-label coverage report (run_annotation_coverage.py) |
| evaluation/baselines/ | Prompt-eval JSON per model_slug/ (run_prompt_eval.py default --results-dir) |
| evaluation/baseline_report/ | model_comparison.csv, MODEL_SELECTION.MD, filtered exports |
| results/ | SFT eval JSON per short_slug/ (not weights); roll up with summarize_prompt_baselines.py --baseline-dir results --out-dir results |
| sft/, sft_full/ | Alternate LoRA training roots (`--output-root`); mostly large local artifacts |
| sft_runs/ | Round-based run registry used for model selection; keep params/metrics, avoid duplicate weight trees |
| visualizations/out/ | Figures from plot_ft.py / ft_figures.ipynb (gitignored) |
| data/nltk_data/ | NLTK data (created automatically; gitignored) |

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



