"""Tiny config loader. Loads the YAML once and exposes it with attribute access,
so the rest of the code reads `cfg.audio["sample_rate"]`."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    ns = SimpleNamespace(**raw)
    ns._raw = raw
    return ns
