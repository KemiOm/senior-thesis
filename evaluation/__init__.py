"""
Evaluation framework for poetry constraint adherence and fine-tuning experiments.

This package provides:
- metrics: Constraint-based metrics (meter, rhyme, CMU coverage, lineation, caesura)
  computed from corpus.db for baseline and model evaluation.
- splits: Train/dev/test and held-out splits (poets, poems) for reproducible evaluation.
- run_baseline: Run metrics on corpus ground truth and save baseline_results.json.

Run from project root. See evaluation/README.md for usage.
"""
