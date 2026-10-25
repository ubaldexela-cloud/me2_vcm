"""Log-Mel feature extraction (locked parameters)."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torchaudio


class LogMelExtractor(torch.nn.Module):
    """
    Produces log-Mel spectrogram with shape (1, n_mels, n_frames).

    Locked: n_fft=512, hop=160, n_mels=40, f_min=0, f_max=8000, center=False
    For 64000 samples: n_frames = 1 + floor((64000-512)/160) = 397
    log: log(mel + eps) with eps=1e-6
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.sample_rate = int(cfg["sample_rate"])
        self.n_fft = int(cfg["n_fft"])
        self.win_length = int(cfg.get("win_length", cfg["n_fft"]))
        self.hop_length = int(cfg["hop_length"])
        self.n_mels = int(cfg["n_mels"])
        self.f_min = float(cfg["f_min"])
        self.f_max = float(cfg["f_max"])
        self.power = float(cfg.get("power", 2.0))
        self.log_eps = float(cfg.get("log_eps", 1e-6))
        self.center = bool(cfg.get("center", False))
        self.expected_frames = int(cfg.get("n_frames", 397))

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=self.power,
            center=self.center,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
            window_fn=torch.hann_window,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (T,) or (B, T) float32
        returns: (1, n_mels, frames) or (B, 1, n_mels, frames)
        """
        squeeze = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeeze = True
        # MelSpectrogram expects (B, T)
        mel = self.mel(wav)  # (B, n_mels, frames)
        log_mel = torch.log(mel + self.log_eps)
        log_mel = log_mel.unsqueeze(1)  # (B, 1, n_mels, frames)
        if squeeze:
            return log_mel.squeeze(0)
        return log_mel

    def check_frames(self, num_samples: int = 64000) -> int:
        """Return frame count for a fixed-length waveform."""
        wav = torch.zeros(num_samples)
        feat = self.forward(wav)
        return int(feat.shape[-1])
