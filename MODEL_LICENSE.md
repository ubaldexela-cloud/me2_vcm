# Model-weight license (not yet published)

This package **does not publish** trained weights.

- `export/model.onnx` is **not** committed. Keep a local copy from the HPC package and verify SHA-256 `3dd066e96e4cc9e17a1e1a571b1a6132e1b55e74870bb854268118e38555a0fb`. Redistribution terms are undecided.
- `artifacts/best_model.pt` is a local copy for HPC transfer only. It is listed in `.gitignore` and must **not** be committed to ordinary Git history.

A public weight license has **not** been chosen in the original training tree. Before a GitHub Release:

1. Do not attach dataset audio.
2. Prefer a **non-commercial** weight license (for example CC BY-NC 4.0) because the training mix includes Fluent Speech Commands (academic / non-commercial public license) and SLURP **audio** (CC BY-NC 4.0).
3. Host `best_model.pt` (~403 KiB) and optionally `mel_norm.pt` (~2.6 KiB) as a **GitHub Release asset** or a Hugging Face model repo. Do not put `.pt` files in the default branch.
4. Publish the SHA-256 values from `reports/CHECKSUMS.txt`.

This file is a preparation note, not a granted license for redistribution of weights.
