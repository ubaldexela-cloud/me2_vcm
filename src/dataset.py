"""PyTorch Dataset for ME2 VCM manifests."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from .audio import fix_length, load_waveform
from .augment import WaveformAugmentor
from .features import LogMelExtractor
from .normalization import MelNormalizer
from .temp_augment import SetTemperatureSpeakerAugmentor
from .utils import CLASS_TO_ID


class VCMDataset(Dataset):
    """
    Reads CSV manifests. Labels come from CLASS_TO_ID (never alphabetical).

    Dataset B train.csv columns differ slightly from Dataset A manifests.
    Optional WaveformAugmentor applies ONLY when split_mode='train'.
    Optional SetTemperatureSpeakerAugmentor applies ONLY to SET_TEMPERATURE in train.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        audio_root: str | Path,
        cfg: Dict[str, Any],
        split_mode: str = "train",
        normalizer: Optional[MelNormalizer] = None,
        extractor: Optional[LogMelExtractor] = None,
        cache_waveforms: bool = False,
        augmentor: Optional[WaveformAugmentor] = None,
        temp_augmentor: Optional[SetTemperatureSpeakerAugmentor] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.audio_root = Path(audio_root)
        self.cfg = cfg
        self.split_mode = split_mode  # 'train' | 'eval'
        self.normalizer = normalizer
        self.extractor = extractor or LogMelExtractor(cfg)
        self.sample_rate = int(cfg["sample_rate"])
        self.num_samples = int(cfg["num_samples"])
        self.cache_waveforms = bool(cache_waveforms)
        self.augmentor = augmentor
        self.temp_augmentor = temp_augmentor
        if self.augmentor is not None and self.split_mode != "train":
            raise ValueError("Augmentor must not be attached to eval/val/test datasets")
        if self.temp_augmentor is not None and self.split_mode != "train":
            raise ValueError("temp_augmentor must not be attached to eval/val/test datasets")
        self._wav_cache: Optional[list] = None

        self.rows = []
        with open(self.manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        if not self.rows:
            raise RuntimeError(f"empty manifest: {self.manifest_path}")

        # Detect schema
        sample = self.rows[0]
        if "class" in sample:
            self.label_key = "class"
            self.path_key = "filepath"
        elif "label" in sample:
            self.label_key = "label"
            self.path_key = "filepath"
        else:
            raise KeyError(f"no class/label column in {self.manifest_path}")

        if self.cache_waveforms:
            self._preload_waveforms()

    def _preload_waveforms(self) -> None:
        """Load waveforms into RAM once (does not modify source files)."""
        cache = []
        n = len(self.rows)
        for i, row in enumerate(self.rows):
            path = self._resolve_path(row[self.path_key])
            wav, sr = load_waveform(path, target_sr=self.sample_rate)
            cache.append((wav, sr))
            if (i + 1) % 2000 == 0 or (i + 1) == n:
                print(f"  waveform cache [{self.split_mode}]: {i+1}/{n}", flush=True)
        self._wav_cache = cache

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        return self.audio_root / rel

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        label_name = row[self.label_key]
        if label_name not in CLASS_TO_ID:
            raise KeyError(f"unknown class {label_name}")
        class_id = CLASS_TO_ID[label_name]
        path = self._resolve_path(row[self.path_key])

        if self._wav_cache is not None:
            wav, sr = self._wav_cache[idx]
        else:
            wav, sr = load_waveform(path, target_sr=self.sample_rate)
        mode = "train" if self.split_mode == "train" else "eval"
        wav = fix_length(wav, self.num_samples, mode=mode)
        # Training-only acoustic augmentation (never on val/test)
        if self.augmentor is not None and self.split_mode == "train":
            wav = self.augmentor(wav, class_name=label_name)
        # SET_TEMPERATURE-only speaker variation (speed/pitch) — train only
        if (
            self.temp_augmentor is not None
            and self.split_mode == "train"
            and label_name == "SET_TEMPERATURE"
        ):
            wav = self.temp_augmentor.apply_waveform(wav)

        feat = self.extractor(wav)  # (1, 40, 397)
        # Mild SpecAugment for SET_TEMPERATURE only — BEFORE normalization
        if (
            self.temp_augmentor is not None
            and self.split_mode == "train"
            and label_name == "SET_TEMPERATURE"
        ):
            feat = self.temp_augmentor.apply_spec_mask(feat)
        if self.normalizer is not None:
            feat = self.normalizer(feat)

        meta = {
            "filepath": str(path),
            "relpath": row[self.path_key],
            "class_name": label_name,
            "class_id": class_id,
            "source_type": row.get("source_type") or row.get("source") or "",
            "original_or_augmented": row.get("original_or_augmented") or "original",
            "source_dataset": row.get("source_dataset") or row.get("original_dataset") or "",
            "speaker": row.get("speaker_id") or row.get("speaker") or "",
            "sample_rate": sr,
        }
        return feat, class_id, meta


def collate_batch(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)  # (B, 1, 40, 397)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metas = [b[2] for b in batch]
    return feats, labels, metas
