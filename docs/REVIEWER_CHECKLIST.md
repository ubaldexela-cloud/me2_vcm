# Reviewer checklist

Public repository: https://github.com/ubaldexela-cloud/me2_vcm \
Review against the current published `main` branch (this file does not pin its own commit hash).

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1a | Public repository | **VERIFIED** | https://github.com/ubaldexela-cloud/me2_vcm current published `main`. |
| 1b | One-command **locked-test** (84.01% acc) | **BLOCKED** | `python scripts/verify_release.py` only checks published checksums. Locked-test needs private `VCM_MASTER` audio, gitignored `.pt` files, and CUDA (`reports/REPRO_TEST.md`). Not claimed as 84% reproduction. |
| 2 | Dataset location, sources, citations/DOIs, licenses | **INCOMPLETE / BLOCKED** pending license and consent resolution | Documented in `DATASET_LICENSES.md` (copy-ready location entry + 30,254-clip source table). Audio **not** in Git; Safe Mix is **not** a public download. Recorded conflicts, SNIPS identity, Multi-Sensor custom terms, unverified tags, and **missing consent for 401 contributor-tagged clips** remain. Do **not** treat this item as complete. |
| 3 | Training logs and final checkpoint | **PARTIAL** / **All Rights Reserved — no public license granted** | Logs: `reports/training_log.csv` (epoch 33, val macro F1 0.797492). Identified checkpoint SHA-256 `4c653cf5710ab46da4a94301045d47cb574a508d30b626f8158304825aa43e7f`; ONNX SHA-256 `3dd066e96e4cc9e17a1e1a571b1a6132e1b55e74870bb854268118e38555a0fb`. **`best_model.pt`, `model.onnx`, and `mel_norm.pt` are not in Git and are not publicly released.** Training index includes 120 `SET_TEMPERATURE_REAL`, 278 `real_voice`, and 3 `xela_SET_TEMPERATURE_REAL` clips. Consent records were not found in inspected directories. All Rights Reserved does **not** override third-party rights or resolve missing consent. |
| 4 | Raspberry Pi **5 Model B Rev 1.1** latency (selected target hardware) | **VERIFIED** (documented run) | Selected and tested board: Raspberry Pi **5 Model B Rev 1.1**. Results: `reports/live_pi5_20261024/`. Course harness is **not** in this repo; the live run is **not** independently reproducible from a GitHub clone. A Pi 4 was **not** used and is **not** a remaining project requirement. |
| 5 | Held-out test, unseen speakers | **VERIFIED** (offline protocol) | n=4758; overlap 0 in `reports/mix_summary.json` / `reports/SPLIT_EVIDENCE.md`. Live HF holdout is a different set. |
| 6 | Comparable-size baseline | **VERIFIED** with limitation | Same architecture, Dataset B, **CPU**: acc 0.664985, F1 0.527704. Not hardware-matched to A100. |
