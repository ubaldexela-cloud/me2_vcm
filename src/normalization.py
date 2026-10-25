"""Global per-mel-bin mean/std normalization (train-only stats)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


class MelNormalizer:
    """Apply (x - mean) / std with mean/std shaped (1, n_mels, 1) for broadcast."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        if mean.dim() != 1:
            raise ValueError("mean must be 1-D (n_mels,)")
        self.mean = mean.view(1, -1, 1).float()
        self.std = std.view(1, -1, 1).float()

    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (1, n_mels, T) or (B, 1, n_mels, T)"""
        mean = self.mean
        std = self.std
        if feat.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (feat - mean.to(feat.device)) / (std.to(feat.device) + 1e-8)

    @staticmethod
    def from_file(path: str | Path) -> "MelNormalizer":
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        return MelNormalizer(obj["mean"], obj["std"])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.view(-1).cpu(),
            "std": self.std.view(-1).cpu(),
        }


def validate_norm_stats(mean: torch.Tensor, std: torch.Tensor) -> None:
    if not torch.isfinite(mean).all():
        raise ValueError("mean has NaN/Inf")
    if not torch.isfinite(std).all():
        raise ValueError("std has NaN/Inf")
    if (std <= 0).any():
        raise ValueError("std has non-positive values")
