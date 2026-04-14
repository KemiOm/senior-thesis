"""short python file for naming conventions during fine-tuning"""

from __future__ import annotations

import re
from pathlib import Path


def baseline_save_slug(model_id: str) -> str:
    """
    Creates a folder name for storing model results.
    Model names from Hugging Face (like "google/flan-t5-large") are converted to "google_flan-t5-large".
    For local checkpoints, it doesn't use the full file path. Instead, use a short name based on the task folder and run folder.
    Example:
    ".../sft/combined_lora/flan-t5-large_lora_20260411_175033/final_model_merged"
    becomes:
    "combined_lora_flan-t5-large_lora_20260411_175033"
    """
    s = (model_id or "").strip()
    if not s:
        return "unknown_model"
    norm = s.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    if parts:
        last = parts[-1]
        if last in ("final_model_merged", "final_model") or (
            last.startswith("checkpoint-") and last != "checkpoint"
        ):
            parent = parts[-2] if len(parts) >= 2 else ""
            gp = parts[-3] if len(parts) >= 3 else ""
            if parent:
                raw = f"{gp}_{parent}" if gp else parent
                return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)

    p = Path(s)
    tail_special = (
        p.name in ("final_model_merged", "final_model")
        or (p.name.startswith("checkpoint-") and p.name != "checkpoint")
    )
    try:
        if p.is_dir():
            rp = p.resolve()
            if tail_special and rp.parent.name:
                parent = rp.parent
                gp = parent.parent
                raw = f"{gp.name}_{parent.name}" if gp.name else parent.name
                return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
            return re.sub(r"[^a-zA-Z0-9_-]", "_", rp.name)
    except OSError:
        pass
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)
