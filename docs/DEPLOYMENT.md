# Raspberry Pi deployment

Selected and tested hardware: **Raspberry Pi 5 Model B Rev 1.1**, aarch64, Debian GNU/Linux 13 (trixie), Python 3.13.5.

The course example lists Raspberry Pi 4 or Raspberry Pi 5 as alternatives. This project selected Pi 5. **Not applicable — Raspberry Pi 5 was selected as the target hardware; Pi 4 was not used.** Live latency and conditions are in `reports/live_pi5_20261024/`. Those numbers are Pi 5 only and are not claimed to transfer to Pi 4.

## Live latency (24 Oct 2026)

Documented in `reports/live_pi5_20261024/` from the course-harness run (seed 82456). That harness is **not** shipped here.

- **End-to-end** p50 / p95 **6.16 / 10.89 s**: time from the **end of played command audio** to the **Pi log line**.
- **ONNX inference** mean / p95 **12.8 / 19.5 ms**: Pi-reported `infer_ms` only.

A 25 Oct SSH audit did **not** start the dashboard and did **not** collect a new acoustic run. Numpy was **2.2.4** in the 24 Oct `pi_specs.json` and **2.4.3** in the venv on 25 Oct.

## Layout

Intent ONNX (local/HPC copy, **not** in Git): SHA `3dd066e96e4cc9e17a1e1a571b1a6132e1b55e74870bb854268118e38555a0fb`, 376,229 bytes.

Wake model is separate (`icy.onnx`, 14,018 bytes, SHA `ed4870d3…`, threshold 0.10). Not trained in the Safe Mix job; **not** in Git.

Mic: USB PnP Sound Device, ALSA `plughw:2,0`.

## Do not

- Restart or overwrite a live demo from this repo without a rollback copy.
- Commit Vosk, ONNX, `.pt`, or audio.
- Treat 6.16 / 10.89 s as neural-net latency.
