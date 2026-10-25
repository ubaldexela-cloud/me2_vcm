"""Training-only waveform acoustic augmentation for robustness_run2.

Never writes files. Never used on val/test.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from .audio import load_waveform


class WaveformAugmentor:
    """
    Mild acoustic augmentations for smart-device microphone realism.

    Config (defaults match robustness_run2 lock):
      gain: p=0.50, range 0.70–1.30
      time_shift: p=0.50, ±80 ms, zero-pad (no circular wrap)
      noise_mix: p=0.50, SNR 10–25 dB (weighted bands)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_paths: Optional[Sequence[str | Path]] = None,
        gain_p: float = 0.50,
        gain_min: float = 0.70,
        gain_max: float = 1.30,
        shift_p: float = 0.50,
        shift_ms: float = 80.0,
        noise_p: float = 0.50,
        snr_bands: Optional[List[tuple]] = None,
        seed: int = 42,
    ):
        self.sample_rate = int(sample_rate)
        self.gain_p = float(gain_p)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.shift_p = float(shift_p)
        self.shift_samples = int(round(shift_ms * 1e-3 * sample_rate))
        self.noise_p = float(noise_p)
        # (low_snr, high_snr, weight)
        self.snr_bands = snr_bands or [
            (10.0, 15.0, 0.20),  # occasional difficult
            (15.0, 20.0, 0.40),  # common
            (20.0, 25.0, 0.40),  # common
        ]
        self.noise_waves: List[torch.Tensor] = []
        self.noise_paths: List[str] = []
        if noise_paths:
            for p in noise_paths:
                wav, sr = load_waveform(p, target_sr=self.sample_rate)
                if wav.numel() < self.sample_rate:  # skip tiny
                    continue
                self.noise_waves.append(wav.contiguous())
                self.noise_paths.append(str(p))
        if not self.noise_waves:
            raise RuntimeError("WaveformAugmentor: no usable noise files loaded")

        # Independent generator for reproducibility with global seed
        self.gen = torch.Generator()
        self.gen.manual_seed(int(seed))

    def config_dict(self) -> dict:
        return {
            "gain_p": self.gain_p,
            "gain_min": self.gain_min,
            "gain_max": self.gain_max,
            "shift_p": self.shift_p,
            "shift_samples": self.shift_samples,
            "shift_ms": self.shift_samples * 1000.0 / self.sample_rate,
            "noise_p": self.noise_p,
            "snr_bands": self.snr_bands,
            "noise_files": self.noise_paths,
            "n_noise_files": len(self.noise_paths),
            "sample_rate": self.sample_rate,
            "notes": (
                "No pitch shift, time stretch, reverb, or SpecAugment. "
                "SILENCE uses ambient noise level mix when signal energy is near zero."
            ),
        }

    def _rand(self) -> float:
        return float(torch.rand(1, generator=self.gen).item())

    def _randint(self, low: int, high: int) -> int:
        """Inclusive low, exclusive high."""
        if high <= low:
            return low
        return int(torch.randint(low, high, (1,), generator=self.gen).item())

    def _uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self._rand()

    def _choose_snr(self) -> float:
        weights = torch.tensor([b[2] for b in self.snr_bands], dtype=torch.float64)
        idx = int(torch.multinomial(weights, 1, generator=self.gen).item())
        lo, hi, _ = self.snr_bands[idx]
        return self._uniform(lo, hi)

    def _sample_noise(self, n: int) -> torch.Tensor:
        wav = self.noise_waves[self._randint(0, len(self.noise_waves))]
        if wav.numel() >= n:
            start = self._randint(0, wav.numel() - n + 1)
            return wav[start : start + n].clone()
        # tile if shorter than needed
        reps = int(math.ceil(n / wav.numel()))
        tiled = wav.repeat(reps)[:n].clone()
        return tiled

    def apply_gain(self, wav: torch.Tensor) -> torch.Tensor:
        g = self._uniform(self.gain_min, self.gain_max)
        return wav * g

    def apply_time_shift(self, wav: torch.Tensor) -> torch.Tensor:
        max_shift = self.shift_samples
        if max_shift <= 0:
            return wav
        shift = self._randint(-max_shift, max_shift + 1)
        if shift == 0:
            return wav
        out = torch.zeros_like(wav)
        if shift > 0:
            out[shift:] = wav[:-shift]
        else:
            out[:shift] = wav[-shift:]
        return out

    def apply_noise_mix(self, wav: torch.Tensor, is_silence: bool = False) -> torch.Tensor:
        noise = self._sample_noise(wav.numel())
        eps = 1e-10
        if is_silence:
            # Near-zero energy: add ambient noise at random RMS (-45..-20 dBFS)
            target_db = self._uniform(-45.0, -20.0)
            target_rms = 10.0 ** (target_db / 20.0)
            n_rms = torch.sqrt(torch.mean(noise ** 2) + eps)
            noise = noise * (target_rms / (n_rms + eps))
            return wav + noise

        snr_db = self._choose_snr()
        sig_pow = torch.mean(wav ** 2).clamp_min(eps)
        noi_pow = torch.mean(noise ** 2).clamp_min(eps)
        snr_lin = 10.0 ** (snr_db / 10.0)
        scale = torch.sqrt(sig_pow / (noi_pow * snr_lin))
        return wav + noise * scale

    def __call__(self, wav: torch.Tensor, class_name: str = "") -> torch.Tensor:
        """
        Apply stochastic augmentations to a fixed-length mono float waveform.
        class_name used only for SILENCE-specific noise mixing behavior.
        """
        is_silence = class_name == "SILENCE"
        # Always allow independent draws; SILENCE gets same ops but noise mix handles low energy
        if self._rand() < self.gain_p:
            wav = self.apply_gain(wav)
        if self._rand() < self.shift_p:
            wav = self.apply_time_shift(wav)
        if self._rand() < self.noise_p:
            wav = self.apply_noise_mix(wav, is_silence=is_silence)
        # mild clip to float range (no hard distortion amp)
        return wav.clamp(-1.0, 1.0)


def discover_noise_files(noise_dir: str | Path) -> List[Path]:
    root = Path(noise_dir)
    files = sorted(
        [
            p
            for p in root.rglob("*")
            if p.suffix.lower() in {".wav", ".flac", ".ogg"} and p.is_file()
        ]
    )
    return files
