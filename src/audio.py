"""Waveform loading and fixed-length pad/crop (no amplitude normalization)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_waveform(
    path: str | Path,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio as mono float32 in approximately [-1, 1].

    PCM_16: values are decoded by soundfile to float in [-1, 1] (equivalent to /32768).
    Already-float files: used as-is (clipped only if needed later by features).
    Resamples to target_sr if needed; mixes down to mono if multi-channel.
    Does NOT apply peak/RMS/loudness normalization.
    """
    path = Path(path)
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # data shape: (samples, channels), float32 in [-1, 1] for PCM_16
    wav = torch.from_numpy(data.T.copy())  # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.shape[0] == 1:
        pass
    else:
        raise ValueError(f"empty audio: {path}")

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    wav = wav.squeeze(0).contiguous()  # (samples,)
    return wav, sr


def fix_length(
    wav: torch.Tensor,
    num_samples: int = 64000,
    mode: str = "train",
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    mode='train': random crop if longer; right-pad if shorter.
    mode='eval': center crop if longer; right-pad if shorter (deterministic).
    """
    n = wav.numel()
    if n == num_samples:
        return wav
    if n < num_samples:
        out = torch.zeros(num_samples, dtype=wav.dtype)
        out[:n] = wav
        return out

    # n > num_samples
    if mode == "train":
        max_start = n - num_samples
        if rng is None:
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start = int(torch.randint(0, max_start + 1, (1,), generator=rng).item())
    elif mode == "eval":
        start = (n - num_samples) // 2
    else:
        raise ValueError(f"unknown mode {mode}")
    return wav[start : start + num_samples].contiguous()
