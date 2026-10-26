# Model-weight license — HOLD

**Status: HOLD — redistribution permission not established.**

This repository **does not publish** trained weights and **does not grant** a license to copy or redistribute them.

The repository MIT license (`LICENSE`) covers **original code and documentation only**. It does **not** cover datasets, third-party recordings, or model weights (`artifacts/best_model.pt`, `export/model.onnx`, `artifacts/mel_norm.pt`).

## Identified artifacts (not in Git)

| File | SHA-256 |
|---|---|
| 84.01% locked-test checkpoint `artifacts/best_model.pt` | `4c653cf5710ab46da4a94301045d47cb574a508d30b626f8158304825aa43e7f` |
| Associated ONNX export `export/model.onnx` | `3dd066e96e4cc9e17a1e1a571b1a6132e1b55e74870bb854268118e38555a0fb` |
| `artifacts/mel_norm.pt` | `3a67c19aa467941fc55cdc7dabf4a6faa08deea3c28bcfd8c750ba66da9ec2ed` |

These hashes match `reports/CHECKSUMS.txt` and `reports/TRAINING_METADATA.json`. There is **no public download link**. Do not commit or attach these files.

## Why redistribution is on HOLD

The Safe Mix training index includes clips that are not covered by a verified public grant, including **120** `SET_TEMPERATURE_REAL`, **278** `real_voice`, and **3** `xela_SET_TEMPERATURE_REAL` rows. Contributor consent evidence was **not found** in the inspected project directories. SNIPS licensing and Multi-Sensor custom terms remain **unresolved**. The mix also includes Fluent Speech Commands (academic / non-Share audio terms) and SLURP **audio** (CC BY-NC 4.0), as already recorded in `DATASET_LICENSES.md`.

This file is a no-grant notice, not a license for weights.
