"""Filesystem-safe directory names for baseline / SFT eval JSON trees."""

from __future__ import annotations

import re
from pathlib import Path


def baseline_save_slug(model_id: str) -> str:
    """Directory name under a baseline results root.

    Hub-style ids become google_flan-t5-large.

    Local checkpoints: use parent folders before final_model_merged.
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
