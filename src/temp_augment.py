"""
SET_TEMPERATURE-targeted augmentations for Run #3.

- Mild speed perturbation (0.95–1.05)
- Mild pitch shift (±1 semitone)
- Mild SpecAugment on log-Mel (before normalization)

Only applied to SET_TEMPERATURE during training.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torchaudio

from .audio import fix_length


class SetTemperatureSpeakerAugmentor:
    """Waveform + feature augmentations exclusive to SET_TEMPERATURE."""

    def __init__(
        self,
        sample_rate: int = 16000,
        num_samples: int = 64000,
        speed_p: float = 0.50,
        speed_min: float = 0.95,
        speed_max: float = 1.05,
        pitch_p: float = 0.50,
        pitch_semitones: float = 1.0,
        spec_p: float = 0.30,
        freq_mask_param: int = 3,
        time_mask_param: int = 15,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        seed: int = 42,
    ):
        self.sample_rate = int(sample_rate)
        self.num_samples = int(num_samples)
        self.speed_p = float(speed_p)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.pitch_p = float(pitch_p)
        self.pitch_semitones = float(pitch_semitones)
        self.spec_p = float(spec_p)
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.n_freq_masks = int(n_freq_masks)
        self.n_time_masks = int(n_time_masks)
        self.gen = torch.Generator()
        self.gen.manual_seed(int(seed) + 7)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "target_class": "SET_TEMPERATURE",
            "speed_p": self.speed_p,
            "speed_min": self.speed_min,
            "speed_max": self.speed_max,
            "pitch_p": self.pitch_p,
            "pitch_semitones": self.pitch_semitones,
            "spec_p": self.spec_p,
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "n_freq_masks": self.n_freq_masks,
            "n_time_masks": self.n_time_masks,
            "notes": "SET_TEMPERATURE only; not applied to SILENCE/UNKNOWN/other classes.",
        }

    def _rand(self) -> float:
        return float(torch.rand(1, generator=self.gen).item())

    def _uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self._rand()

    def _randint(self, low: int, high: int) -> int:
        if high <= low:
            return low
        return int(torch.randint(low, high, (1,), generator=self.gen).item())

    def apply_speed(self, wav: torch.Tensor) -> torch.Tensor:
        speed = self._uniform(self.speed_min, self.speed_max)
        if abs(speed - 1.0) < 1e-6:
            return wav
        # Resample at speed*sr then interpret at sr → duration changes
        new_sr = max(int(round(self.sample_rate * speed)), 1)
        x = wav.unsqueeze(0)
        y = torchaudio.functional.resample(x, self.sample_rate, new_sr).squeeze(0)
        return fix_length(y, self.num_samples, mode="eval")

    def apply_pitch(self, wav: torch.Tensor) -> torch.Tensor:
        # Uniform in [-semitones, +semitones]
        n_steps = self._uniform(-self.pitch_semitones, self.pitch_semitones)
        if abs(n_steps) < 1e-6:
            return wav
        x = wav.unsqueeze(0)
        y = torchaudio.functional.pitch_shift(x, self.sample_rate, n_steps).squeeze(0)
        return fix_length(y, self.num_samples, mode="eval")

    def apply_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        if self._rand() < self.speed_p:
            wav = self.apply_speed(wav)
        if self._rand() < self.pitch_p:
            wav = self.apply_pitch(wav)
        return wav.clamp(-1.0, 1.0)

    def apply_spec_mask(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Mild SpecAugment on log-Mel.
        feat: (1, n_mels, T) — applied BEFORE normalization.
        """
        if self._rand() >= self.spec_p:
            return feat
        out = feat.clone()
        # out shape (1, F, T)
        _, n_mels, n_frames = out.shape
        for _ in range(self.n_freq_masks):
            f = self._randint(0, self.freq_mask_param + 1)
            if f == 0 or n_mels <= 1:
                continue
            f0 = self._randint(0, max(n_mels - f, 1))
            out[:, f0 : f0 + f, :] = 0.0
        for _ in range(self.n_time_masks):
            t = self._randint(0, self.time_mask_param + 1)
            if t == 0 or n_frames <= 1:
                continue
            t0 = self._randint(0, max(n_frames - t, 1))
            out[:, :, t0 : t0 + t] = 0.0
        return out
