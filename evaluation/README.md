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

---

## Outputs

| Path | Description |
|------|-------------|
| `evaluation/splits/*.json` | Poem IDs per split |
| `evaluation/baseline_results.json` | Metrics per split (corpus ground truth) |
| `evaluation/results/baselines/prompt_only/<model_slug>/` | Per-model prompt-only JSONs (from HPC) |

---

## Syncing results 

Baseline runs live on the cluster. To pull them I used.

```bash
rsync -avz span9810_ato22@bouchet.ycrc.yale.edu:~/Senior-Thesis/evaluation/results/ "/Users/kemiomoniyi/Senior Thesis/evaluation/results/"
```
