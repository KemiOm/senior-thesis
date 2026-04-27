"""CLI shim: python evaluation/summarize_prompt_baselines.py to evaluation.scoring.rollup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scoring.rollup import collect_rows, main

__all__ = ["collect_rows", "main"]

if __name__ == "__main__":
    main()
