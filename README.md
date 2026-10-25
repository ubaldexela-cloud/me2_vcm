# ICY — ME2 Voice Controlled Smart Device

Wake-word plus 16-class spoken command recognition for a Raspberry Pi smart-device demo (AI231 ME2).

Repository: https://github.com/ubaldexela-cloud/me2_vcm

This tree contains **training/evaluation code, recorded metrics, feature metadata, and Raspberry Pi dashboard source**. It does **not** contain dataset audio, Vosk model files, wake-word training data, PyTorch `.pt` checkpoints, or the ONNX graph (`export/model.onnx` is local-only until a weight license is chosen).

## Architecture

- **VCM-DS-Tiny** depthwise-separable CNN, **95,376** trainable parameters.
- Input: 16 kHz mono, 4.0 s (64,000 samples), log-Mel `(B, 1, 40, 397)`.
- Output: 16 logits (softmax outside the graph).
- Classes: PLAY_MUSIC, WEATHER, TIME, LIGHT_ON, LIGHT_OFF, LIGHT_DIM, SET_TIMER, SET_ALARM, SET_TEMPERATURE, MEDIA_PAUSE, MEDIA_STOP, MEDIA_NEXT, VOLUME_UP, VOLUME_DOWN, UNKNOWN, SILENCE.
- Deployment: ONNX Runtime on Raspberry Pi **5**, USB mic `plughw:2,0`, openWakeWord (`icy.onnx`, threshold 0.10), Vosk only for timer/alarm/temperature/brightness slots.

## Two result families (do not mix)

### Offline locked test (classifier only, not Pi)

Recorded on NVIDIA **A100 80GB PCIe**, PyTorch 2.8.0+cu128, seed **42**, best epoch **33** (highest **val macro F1 0.797492080662753**). Train wall time **28,535.6 s**.

| Metric | Value |
|---|---|
| Test accuracy | **0.8400588482555695** (84.01%) |
| Test macro F1 | **0.7785515388212305** |
| n_test | **4,758** Dataset A held-out clips |
| Speaker/SHA overlap vs train | **0** (`reports/mix_summary.json`) |

HPC job **66249** re-evaluated the released checkpoint and matched these numbers with **Δ = 0**. That used cluster PyTorch and local `VCM_MASTER` audio. It is **not** a clean-laptop or from-scratch training test. See `reports/REPRO_TEST.md` and `reports/VERIFIED_RUN.md`.

### Live acoustic benchmark (Raspberry Pi **5**, not Pi 4)

Course harness playback, 24 Oct 2026, seed 82456, 109 wake trials + 16 no-wake. Mic check skipped.

| Metric | Value | Definition |
|---|---|---|
| Keyword / wake detect | **92.7%** | wake log or command event |
| 19-intent accuracy | **40.4%** | course 19-way labels (ICY trains 16) |
| False accept | **62.5% (10/16)** | out-of-scope fired |
| Latency p50 / p95 | **6.16 / 10.89 s** | **end of played command → Pi log**, not ONNX ms |
| ONNX infer mean / p95 | 12.8 / 19.5 ms | `infer_ms` |

Full write-up: `reports/live_pi5_20261024/`. **Pi 4 latency is PENDING.** Do not label these as Pi 4 numbers.

### Baseline (architecture-matched, not hardware-matched)

`baseline_run1`: same VCM-DS-Tiny, Dataset B train, locked A val/test, **CPU** training, no extra augmentation.

| Metric | Value |
|---|---|
| Test accuracy | **0.664985** (~66.50%) |
| Test macro F1 | **0.527704** |

This is **not** a controlled A100 vs CPU study. See `reports/baselines/`.

## Installation (training/eval code)

Python 3.9+ with the versions in `requirements.txt` (PyTorch 2.8 CUDA wheels from pytorch.org). Audio datasets are **not** in Git.

```bash
git clone https://github.com/ubaldexela-cloud/me2_vcm.git
cd me2_vcm
python -m venv .venv
# then install torch/torchaudio from the official CUDA index, plus requirements.txt
```

Raspberry Pi runtime versions (observed 25 Oct 2026): Python 3.13.5, onnxruntime 1.30.0, numpy **2.4.3**, flask 3.1.3, openwakeword 0.4.0, vosk 0.3.45, soundfile 0.14.0. The 24 Oct live bench recorded numpy **2.2.4** — the venv drifted; see `requirements-pi.txt`.

## Laptop command that is actually tested here

Checksum + optional ONNX dummy forward (no audio, no 84% claim):

```bash
python scripts/verify_release.py
```

Expected: `OK` for json/npy/csv rows in `reports/CHECKSUMS.txt`; `MISSING` for gitignored `artifacts/*.pt` and `export/model.onnx` until those files are supplied locally.

## Locked-test evaluation (not one-command on a laptop)

Requires: locked **VCM_MASTER** test audio, CUDA, and `artifacts/best_model.pt` + `artifacts/mel_norm.pt` (SHA-256 in `reports/CHECKSUMS.txt`; **not in Git** until a weight license is chosen).

```bash
export VCM_MASTER=/path/to/VCM_MASTER
export MEL_NORM=$PWD/artifacts/mel_norm.pt
python3 scripts/eval_locked_test.py \
  --checkpoint artifacts/best_model.pt \
  --out-dir reports/repro_eval
```

Expected only if that environment matches the A100 job: accuracy 0.8400588, macro F1 0.7785515. The script **refuses CPU**. Training from scratch is **not** one-command and was **not** re-run for this submission.

## Raspberry Pi deployment

See `docs/DEPLOYMENT.md` and `pi/`. Do not unpack over a live demo. Obtain `export/model.onnx` from the HPC package (SHA `3dd066e9…` in `reports/CHECKSUMS.txt`); it is **not** in this Git tree. The wake model is separate and also not in Git. Vosk `vosk-model-small-en-us-0.15` is third-party and not shipped.

## Licensing

- Original code: MIT (`LICENSE`).
- Datasets: **not included**. FSC audio must not be redistributed; SLURP audio is CC BY-NC 4.0; SNIPS terms **unverified**; student recordings need consent (`DATASET_LICENSES.md`).
- Weights: license **undecided** (`MODEL_LICENSE.md`). Neither `.pt` nor ONNX is in Git.

## Reviewer checklist

`docs/REVIEWER_CHECKLIST.md`
