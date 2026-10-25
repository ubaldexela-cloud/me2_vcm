"""NumPy log-mel frontend matched to ME2 VCM TorchAudio eval preprocessing.

This is a deployment copy. It does not replace ME2_VCM/src/features.py.

TorchAudio path that this follows (torchaudio 2.8):

- soundfile float32 decode, mono mean, no peak or RMS normalization
- eval length: center crop if longer than 64000, right-pad with zeros if shorter
- periodic Hann window of length 512: 0.5 - 0.5 * cos(2 * pi * n / 512)
- torch.stft with center=False, normalized=False, onesided=True, hop 160, n_fft 512
- pad_mode "reflect" is stored by MelSpectrogram but is not applied when center is False
- power spectrogram is |STFT|^2
- HTK mel filterbank, norm=None, f_min=0, f_max=8000, n_mels=40
  hz_to_mel = 2595 * log10(1 + hz / 700)
  mel_to_hz = 700 * (10^(mel / 2595) - 1)
  triangular filters from melscale_fbanks, no Slaney area normalization
- natural log(mel + 1e-6)
- normalize (feat - mean) / (std + 1e-8)

Files that are not already 16 kHz are rejected. TorchAudio's resampler is not
reimplemented here, so a different sample rate cannot silently change the features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 160
N_MELS = 40
N_FREQS = N_FFT // 2 + 1
F_MIN = 0.0
F_MAX = 8000.0
NUM_SAMPLES = 64000
LOG_EPS = 1e-6
NORM_EPS = 1e-8


def periodic_hann(length: int = WIN_LENGTH) -> np.ndarray:
    n = np.arange(length, dtype=np.float32)
    return (np.float32(0.5) - np.float32(0.5) * np.cos(
        (np.float32(2.0) * np.float32(np.pi) * n) / np.float32(length)
    )).astype(np.float32)


def mel_filterbank() -> np.ndarray:
    """HTK triangular filterbank, shape (n_freqs, n_mels), float32. norm is None."""
    all_freqs = np.linspace(F_MIN, SAMPLE_RATE // 2, N_FREQS, dtype=np.float64)
    m_min = 2595.0 * np.log10(1.0 + (F_MIN / 700.0))
    m_max = 2595.0 * np.log10(1.0 + (F_MAX / 700.0))
    m_pts = np.linspace(m_min, m_max, N_MELS + 2, dtype=np.float64)
    f_pts = 700.0 * (np.power(10.0, m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]
    down = (-slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    return np.ascontiguousarray(fb, dtype=np.float32)


_WINDOW = periodic_hann()
_FB = mel_filterbank()


def load_mono_f32(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if data.shape[1] == 0 or data.shape[0] == 0:
        raise ValueError(f"empty audio: {path}")
    if data.shape[1] == 1:
        wav = data[:, 0]
    else:
        wav = data.mean(axis=1).astype(np.float32)
    if int(sr) != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample rate {sr}; this frontend does not resample. "
            "The locked TorchAudio path expects 16000 Hz."
        )
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def fix_length_eval(wav: np.ndarray, num_samples: int = NUM_SAMPLES) -> np.ndarray:
    n = int(wav.shape[0])
    if n == num_samples:
        return np.ascontiguousarray(wav, dtype=np.float32)
    if n < num_samples:
        out = np.zeros(num_samples, dtype=np.float32)
        out[:n] = wav
        return out
    start = (n - num_samples) // 2
    return np.ascontiguousarray(wav[start : start + num_samples], dtype=np.float32)


def power_spectrogram(wav: np.ndarray) -> np.ndarray:
    """|STFT|^2 with center=False. Shape (n_freqs, n_frames)."""
    if wav.shape[0] < N_FFT:
        raise ValueError(f"need at least {N_FFT} samples, got {wav.shape[0]}")
    frames = np.lib.stride_tricks.sliding_window_view(wav, N_FFT)[::HOP_LENGTH]
    spec = np.fft.rfft(frames * _WINDOW, n=N_FFT, axis=1)
    power = (spec.real.astype(np.float32) ** 2) + (spec.imag.astype(np.float32) ** 2)
    return np.ascontiguousarray(power.T, dtype=np.float32)


def log_mel(wav: np.ndarray) -> np.ndarray:
    """Natural log mel, shape (n_mels, n_frames), before mean/std normalization."""
    power = power_spectrogram(wav)
    mel = power.T @ _FB
    mel = np.ascontiguousarray(mel.T, dtype=np.float32)
    return np.log(mel + np.float32(LOG_EPS)).astype(np.float32)


def normalize(logmel: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32).reshape(N_MELS, 1)
    std = np.asarray(std, dtype=np.float32).reshape(N_MELS, 1)
    return ((logmel - mean) / (std + np.float32(NORM_EPS))).astype(np.float32)


def waveform_to_model_input(wav: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_mel[40, T], model_input[1, 1, 40, T])."""
    feat = log_mel(wav)
    norm = normalize(feat, mean, std)
    model_in = norm.reshape(1, 1, N_MELS, norm.shape[1]).astype(np.float32)
    return feat, model_in


def file_to_model_input(path: str | Path, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    wav, sr = load_mono_f32(path)
    raw_n = int(wav.shape[0])
    peak = float(np.max(np.abs(wav))) if raw_n else 0.0
    fixed = fix_length_eval(wav, NUM_SAMPLES)
    feat, model_in = waveform_to_model_input(fixed, mean, std)
    meta = {
        "sample_rate": sr,
        "raw_samples": raw_n,
        "fixed_samples": int(fixed.shape[0]),
        "peak_abs": peak,
        "feature_shape": list(feat.shape),
        "model_input_shape": list(model_in.shape),
    }
    return feat, model_in, meta



import sys

import onnxruntime as ort


def _softmax(logits):
    z = logits.astype(np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python inference.py sample.wav", file=sys.stderr)
        return 2
    import json
    pkg = Path(__file__).resolve().parent
    cfg = json.loads((pkg / "feature_config.json").read_text())
    class_map = json.loads((pkg / "class_map.json").read_text())
    names = [class_map["id_to_class"][str(i)] for i in range(int(cfg["num_classes"]))]
    mean = np.load(pkg / "mean.npy")
    std = np.load(pkg / "std.npy")
    _feat, model_in, _meta = file_to_model_input(Path(sys.argv[1]), mean, std)
    if tuple(model_in.shape) != tuple(cfg["model_input_shape"]):
        raise RuntimeError(f"expected {cfg['model_input_shape']}, got {list(model_in.shape)}")
    if not np.isfinite(model_in).all():
        raise RuntimeError("features contain NaN or Inf")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(pkg / "model.onnx"), so, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {sess.get_inputs()[0].name: model_in})[0][0]
    probs = _softmax(logits)
    order = np.argsort(-probs)
    pred = int(order[0])
    print(f"Predicted class: {names[pred]}")
    print(f"Confidence: {float(probs[pred]):.6f}")
    print("Top-5:")
    for rank, i in enumerate(order[:5], start=1):
        print(f"  {rank}. {names[int(i)]}  {float(probs[int(i)]):.6f}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
