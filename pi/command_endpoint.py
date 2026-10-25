#!/usr/bin/env python3
"""End recording after trailing silence; keep the 4.0 s Safe Mix window.

Does not change COMMAND_SECONDS of the model. Captured audio is right-padded
with zeros to 64,000 samples. No new packages. No microphone is opened unless
record_command() is called with a device.

Copy this file beside the experimental dashboard on the Pi. Do not put it
inside pi_package_safe_mix/ and do not overwrite the original dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

SAMPLE_RATE = 16000
MODEL_SAMPLES = 64000  # 4.0 s — do not shrink
MIN_SECONDS = 1.20
HANGOVER_MS = 900
FRAME_MS = 20
ABS_SPEECH_RMS = 0.004  # onset ~-48 dBFS RMS; quiet SET_TEMPERATURE
ABS_HOLD_RMS = 0.0015  # stay in speech through dips ~-56 dBFS
NOISE_FRAMES = 8
SPEECH_RATIO = 3.5
HOLD_RATIO = 2.0
MIN_SPEECH_FRAMES = 4


@dataclass(frozen=True)
class EndpointConfig:
    sample_rate: int = SAMPLE_RATE
    model_samples: int = MODEL_SAMPLES
    min_seconds: float = MIN_SECONDS
    hangover_ms: float = HANGOVER_MS
    frame_ms: float = FRAME_MS
    abs_speech_rms: float = ABS_SPEECH_RMS
    abs_hold_rms: float = ABS_HOLD_RMS
    noise_frames: int = NOISE_FRAMES
    speech_ratio: float = SPEECH_RATIO
    hold_ratio: float = HOLD_RATIO
    min_speech_frames: int = MIN_SPEECH_FRAMES

    @property
    def frame_samples(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))

    @property
    def min_samples(self) -> int:
        return int(round(self.min_seconds * self.sample_rate))

    @property
    def hangover_frames(self) -> int:
        return max(1, int(round(self.hangover_ms / self.frame_ms)))


class CommandEndpoint:
    """Streaming endpoint detector. Push int16 or float32 mono chunks."""

    def __init__(self, cfg: Optional[EndpointConfig] = None):
        self.cfg = cfg or EndpointConfig()
        self._buf = np.zeros(0, dtype=np.float32)
        self._frame_rms: list[float] = []
        self._speech_run = 0
        self._heard_speech = False
        self._silent_run = 0
        self._stop_at: Optional[int] = None

    def reset(self) -> None:
        self.__init__(self.cfg)

    def push(self, chunk: np.ndarray) -> bool:
        """Append samples. Return True when capture should stop."""
        if self._stop_at is not None:
            return True
        x = np.asarray(chunk)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / np.float32(32768.0)
        else:
            x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1).astype(np.float32)
        remain = self.cfg.model_samples - int(self._buf.shape[0])
        if remain <= 0:
            self._stop_at = self.cfg.model_samples
            return True
        if x.shape[0] > remain:
            x = x[:remain]
        self._buf = np.concatenate([self._buf, x])
        self._consume_frames()
        if int(self._buf.shape[0]) >= self.cfg.model_samples:
            self._stop_at = self.cfg.model_samples
            return True
        return self._stop_at is not None

    def _noise_rms(self) -> float:
        """Room tone from frames below the onset floor. Ignore speech-leading clips."""
        quiet = [r for r in self._frame_rms if r < self.cfg.abs_speech_rms]
        if len(quiet) < self.cfg.noise_frames:
            return 0.0
        return float(np.median(quiet[: self.cfg.noise_frames]))

    def _onset_threshold(self) -> float:
        noise = self._noise_rms()
        if noise <= 0.0:
            return self.cfg.abs_speech_rms
        return max(self.cfg.abs_speech_rms, self.cfg.speech_ratio * noise)

    def _hold_threshold(self) -> float:
        noise = self._noise_rms()
        if noise <= 0.0:
            return self.cfg.abs_hold_rms
        return max(self.cfg.abs_hold_rms, self.cfg.hold_ratio * noise)

    def _consume_frames(self) -> None:
        fs = self.cfg.frame_samples
        n_frames = int(self._buf.shape[0]) // fs
        while len(self._frame_rms) < n_frames:
            i = len(self._frame_rms)
            frame = self._buf[i * fs : (i + 1) * fs]
            rms = float(np.sqrt(np.mean(np.square(frame))))
            self._frame_rms.append(rms)
            if self._heard_speech:
                is_speech = rms >= self._hold_threshold()
            else:
                is_speech = rms >= self._onset_threshold()
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.cfg.min_speech_frames:
                    self._heard_speech = True
                self._silent_run = 0
            else:
                self._speech_run = 0
                if self._heard_speech:
                    self._silent_run += 1
            t = (i + 1) * fs
            if (
                self._heard_speech
                and t >= self.cfg.min_samples
                and self._silent_run >= self.cfg.hangover_frames
            ):
                self._stop_at = min(int(self._buf.shape[0]), t)
                return

    @property
    def stopped(self) -> bool:
        return self._stop_at is not None

    def captured_unpadded(self) -> np.ndarray:
        n = self._stop_at if self._stop_at is not None else int(self._buf.shape[0])
        n = min(n, int(self._buf.shape[0]), self.cfg.model_samples)
        return np.ascontiguousarray(self._buf[:n], dtype=np.float32)

    def captured_model_input(self) -> np.ndarray:
        return pad_to_model_length(self.captured_unpadded(), self.cfg.model_samples)


def pad_to_model_length(wav: np.ndarray, num_samples: int = MODEL_SAMPLES) -> np.ndarray:
    """Right-pad or truncate to the locked 4.0 s window. No loop, no extra speech."""
    x = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
    if x.ndim > 1:
        x = x.reshape(-1)
    out = np.zeros(num_samples, dtype=np.float32)
    n = min(int(x.shape[0]), num_samples)
    if n:
        out[:n] = x[:n]
    return out


def apply_endpoint(wav: np.ndarray, cfg: Optional[EndpointConfig] = None) -> np.ndarray:
    """Offline: stream a file as if it were a live capture, then pad to 4.0 s."""
    cfg = cfg or EndpointConfig()
    det = CommandEndpoint(cfg)
    x = np.ascontiguousarray(np.asarray(wav, dtype=np.float32)).reshape(-1)
    fs = cfg.frame_samples
    for start in range(0, int(x.shape[0]), fs):
        if det.push(x[start : start + fs]):
            break
    if not det.stopped:
        det._stop_at = min(int(det._buf.shape[0]), cfg.model_samples)
    return det.captured_model_input()


def record_command(
    chunks: Iterable[np.ndarray],
    cfg: Optional[EndpointConfig] = None,
) -> np.ndarray:
    """Consume an existing chunk iterator (tests or dashboard capture). No mic."""
    det = CommandEndpoint(cfg)
    for chunk in chunks:
        if det.push(chunk):
            break
    if not det.stopped:
        det._stop_at = min(int(det._buf.shape[0]), det.cfg.model_samples)
    return det.captured_model_input()


def arecord_chunks(device: str, frame_samples: int = 320) -> Iterable[np.ndarray]:
    """Yield int16 mono frames from arecord. Caller must not use this on HPC."""
    import subprocess

    cmd = [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(SAMPLE_RATE),
        "-t",
        "raw",
        "-q",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_samples * 2)
    try:
        while True:
            buf = proc.stdout.read(frame_samples * 2)
            if not buf or len(buf) < frame_samples * 2:
                break
            yield np.frombuffer(buf, dtype=np.int16)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()


def record_command_from_mic(device: str, cfg: Optional[EndpointConfig] = None) -> np.ndarray:
    """Pi-only capture. Do not call on the HPC login node."""
    cfg = cfg or EndpointConfig()
    return record_command(arecord_chunks(device, cfg.frame_samples), cfg)
