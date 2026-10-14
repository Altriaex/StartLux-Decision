"""Model folders for tests: config.json and decision_config.json as released folders have them (no weights)."""
from __future__ import annotations

import json
import os
import tempfile

LETTERS = list(range(65, 91))           # the letter ids of FakeTokenizer
TEMPERATURES = {"noul": 1.376, "choice": 0.9503, "score": 1.0108}


def make_folder(kind: str = "auto_think", extra: dict | None = None, config: dict | None = None) -> str:
    """A temporary folder: "auto_think" (causal) or "instant" (prompt-bidirectional)."""
    path = tempfile.mkdtemp(prefix="startlux_test_")
    cfg = {}
    dec = {"letter_token_ids": LETTERS, "temperature_by_type": dict(TEMPERATURES), "max_options_per_pass": 26,
           "wide_choice": {"group": 25, "keep": 3, "residual": 0.001}}
    if kind == "instant":
        cfg["jev_prompt_attention"] = "bidirectional"
        dec["prompt_attention"] = "bidirectional"
    cfg.update(config or {})
    dec.update(extra or {})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(path, "decision_config.json"), "w") as f:
        json.dump(dec, f)
    return path
