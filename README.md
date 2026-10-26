# ICY — ME2 Voice Controlled Smart Device

Wake-word plus 16-class spoken command recognition for a Raspberry Pi smart-device demo (AI231 ME2).

**Selected and tested deployment device: Raspberry Pi 5 Model B Rev 1.1.** The course example lists Raspberry Pi 4 or Raspberry Pi 5 as alternatives; this project selected Pi 5. A Pi 4 was not used and was not benchmarked.

Repository: https://github.com/ubaldexela-cloud/me2_vcm

This tree contains **training/evaluation code, recorded metrics, feature metadata, and Raspberry Pi dashboard source**. It does **not** contain dataset audio, Vosk model files, wake-word training data, PyTorch `.pt` checkpoints, or the ONNX graph. Weight redistribution is **HOLD — permission not established** (`MODEL_LICENSE.md`).

## Architecture

- **VCM-DS-Tiny** depthwise-separable CNN, **95,376** trainable parameters.
- Input: 16 kHz mono, 4.0 s (64,000 samples), log-Mel `(B, 1, 40, 397)`.
- Output: 16 logits (softmax outside the graph).
- Classes: PLAY_MUSIC, WEATHER, TIME, LIGHT_ON, LIGHT_OFF, LIGHT_DIM, SET_TIMER, SET_ALARM, SET_TEMPERATURE, MEDIA_PAUSE, MEDIA_STOP, MEDIA_NEXT, VOLUME_UP, VOLUME_DOWN, UNKNOWN, SILENCE.
- Deployment: ONNX Runtime on Raspberry Pi **5 Model B Rev 1.1**, USB mic `plughw:2,0`, openWakeWord (`icy.onnx`, threshold 0.10), Vosk only for timer/alarm/temperature/brightness slots.

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

### Live acoustic benchmark (selected hardware: Raspberry Pi **5 Model B Rev 1.1**)

**Not applicable — Raspberry Pi 5 was selected as the target hardware; Pi 4 was not used.** Device: Raspberry Pi **5 Model B Rev 1.1**, Debian 13, aarch64. These are Pi 5 measurements and are not claimed to transfer to Pi 4.

Course-harness playback on **24 Oct 2026** (not re-run in the post-publication audit). Seed **82456**, 109 wake trials + 16 no-wake. Mic check skipped. The dashboard was **not** running during the 25 Oct SSH audit; no new acoustic trial was started.

| Metric | Value | Definition |
|---|---|---|
| Keyword / wake detect | **92.7%** | wake log or command event |
| 19-intent accuracy | **40.4%** | course 19-way labels (ICY trains 16) |
| False accept | **62.5% (10/16)** | out-of-scope fired |
| End-to-end p50 / p95 | **6.16 / 10.89 s** | **end of played command audio → Pi log line** |
| ONNX infer mean / p95 | 12.8 / 19.5 ms | Pi-reported `infer_ms` only |

Write-up: `reports/live_pi5_20261024/` (copied from the 24 Oct course run). The playback harness is **not** in this repository; a GitHub clone cannot replay that test.

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

## Public verification (checksums only — not 84.01% accuracy)

A clone of this repository can check published file hashes. That does **not** reproduce locked-test accuracy **0.8400588482555695** (84.01%).

```bash
python scripts/verify_release.py
```

Expected from a GitHub clone: `OK` for json/npy/csv rows in `reports/CHECKSUMS.txt`; **`MISSING`** for gitignored `artifacts/*.pt` and `export/model.onnx`. Optional ONNX dummy-forward needs a local ONNX file and `onnxruntime` (not claimed as a clean-room eval).

## Locked-test evaluation (private data + CUDA)

Requires locked **VCM_MASTER** audio (not redistributed), CUDA, and `artifacts/best_model.pt` + `artifacts/mel_norm.pt` (SHA-256 in `reports/CHECKSUMS.txt`; **not in Git**; **HOLD — redistribution permission not established**).

```bash
export VCM_MASTER=/path/to/VCM_MASTER
export MEL_NORM=$PWD/artifacts/mel_norm.pt
python3 scripts/eval_locked_test.py \
  --checkpoint artifacts/best_model.pt \
  --out-dir reports/repro_eval
```

Expected only if that environment matches A100 job 66249: accuracy 0.8400588, macro F1 0.7785515. The script **refuses CPU**. Training from scratch is **not** available from the public tree (no audio, no checkpoint).

## Raspberry Pi deployment

See `docs/DEPLOYMENT.md` and `pi/`. Do not unpack over a live demo. Obtain `export/model.onnx` from the HPC package (SHA `3dd066e9…` in `reports/CHECKSUMS.txt`); it is **not** in this Git tree. The wake model is separate and also not in Git. Vosk `vosk-model-small-en-us-0.15` is third-party and not shipped.

## Licensing

- Original code and documentation: MIT (`LICENSE`). That grant does **not** cover datasets, third-party recordings, or model weights.
- Datasets: **not included**. FSC audio must not be redistributed; SLURP audio is CC BY-NC 4.0; SNIPS terms **unresolved**; Multi-Sensor custom terms **unresolved**; contributor consent evidence was **not found** in inspected project directories (`DATASET_LICENSES.md`).
- Weights: **HOLD — redistribution permission not established** (`MODEL_LICENSE.md`). Identified checkpoint SHA-256 `4c653cf5710ab46da4a94301045d47cb574a508d30b626f8158304825aa43e7f`; ONNX SHA-256 `3dd066e96e4cc9e17a1e1a571b1a6132e1b55e74870bb854268118e38555a0fb`. Neither file is published. No download links.

## Reviewer checklist

`docs/REVIEWER_CHECKLIST.md`
