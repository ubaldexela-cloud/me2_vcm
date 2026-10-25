# Raspberry Pi deployment

Verified hardware: **Raspberry Pi 5 Model B Rev 1.1**, aarch64, Debian GNU/Linux 13, Python 3.13.5.

**Pi 4 is not verified.** Latency item 4 in the course checklist is PENDING.

## Layout

Intent package files match HPC `export/` (ONNX SHA `3dd066e96e4cc9e17a1e1a571b1a6132e1b55e74870bb854268118e38555a0fb`, 376,229 bytes).

Wake model is separate (`icy.onnx`, 14,018 bytes, SHA `ed4870d3a129d51669da77fd9197df9e54157cf76cfb22ea75ca5efa86d5567f`, threshold 0.10). Not trained in the Safe Mix job.

Mic: USB PnP Sound Device, ALSA `plughw:2,0`.

## Do not

- Restart or overwrite a live demo from this repo without a rollback copy.
- Commit Vosk or `.pt` files.
- Treat 6.16 / 10.89 s p50/p95 as neural-net latency.
