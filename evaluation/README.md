# Evaluation framework

Run from project root.

---

## Quick start

```bash
# Create splits (train/dev/test, held-out poets/poems)
python3 evaluation/splits.py

# Compute metrics on full corpus
python3 evaluation/metrics.py

# Run baseline (corpus ground truth) → baseline_results.json
python3 evaluation/run_baseline.py
```

**Prompt-only baselines (HPC):** See **[BASELINE_MODELS_TO_RUN.md](./BASELINE_MODELS_TO_RUN.md)** for the list of 9 models and copy-paste `sbatch` commands.

---

## Outputs

| Path | Description |
|------|-------------|
| `evaluation/splits/*.json` | Poem IDs per split |
| `evaluation/baseline_results.json` | Metrics per split (corpus ground truth) |
| `evaluation/results/baselines/prompt_only/<model_slug>/` | Per-model prompt-only JSONs (from HPC) |
| `evaluation/results/model_comparison.csv` | Summary: model × prompt × metrics |

---

## Syncing results to your machine

Baseline runs live on the cluster. To pull them to your Mac (replace with your cluster user/host if different):

```bash
rsync -avz span9810_ato22@bouchet.ycrc.yale.edu:~/Senior-Thesis/evaluation/results/ "/Users/kemiomoniyi/Senior Thesis/evaluation/results/"
```

---

## Next steps (HPC)

1. Run prompt-only baselines (see BASELINE_MODELS_TO_RUN.md).
2. Annotate model outputs with phonology pipeline.
3. Compute metrics on model outputs; compare to baseline; fill model_comparison.csv.
