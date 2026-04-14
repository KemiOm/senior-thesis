"""CLI shim: python evaluation/summarize_prompt_baselines.py to evaluation.scoring.rollup."""

from evaluation.scoring.rollup import collect_rows, main

__all__ = ["collect_rows", "main"]

if __name__ == "__main__":
    main()
