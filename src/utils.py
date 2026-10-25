"""Shared utilities: class mapping, config, reproducibility."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Authoritative class mapping (do NOT alphabetize)
CLASS_TO_ID = {
    "PLAY_MUSIC": 0,
    "WEATHER": 1,
    "TIME": 2,
    "LIGHT_ON": 3,
    "LIGHT_OFF": 4,
    "LIGHT_DIM": 5,
    "SET_TIMER": 6,
    "SET_ALARM": 7,
    "SET_TEMPERATURE": 8,
    "MEDIA_PAUSE": 9,
    "MEDIA_STOP": 10,
    "MEDIA_NEXT": 11,
    "VOLUME_UP": 12,
    "VOLUME_DOWN": 13,
    "UNKNOWN": 14,
    "SILENCE": 15,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
assert len(CLASS_TO_ID) == 16


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "safe_mix_run1.yaml"
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return _expand(cfg)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
