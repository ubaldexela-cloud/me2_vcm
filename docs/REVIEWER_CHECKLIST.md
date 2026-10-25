# Reviewer checklist

Public repository: https://github.com/ubaldexela-cloud/me2_vcm \
Published commit: `494a04f37af68b8d52d9dbfdeb6303e87c75593e`

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1a | Public repository | **VERIFIED** | https://github.com/ubaldexela-cloud/me2_vcm `main` at `494a04f`. |
| 1b | One-command **locked-test** (84.01% acc) | **BLOCKED** | `python scripts/verify_release.py` only checks published checksums. Locked-test needs private `VCM_MASTER` audio, gitignored `.pt` files, and CUDA (`reports/REPRO_TEST.md`). Not claimed as 84% reproduction. |
| 2 | Dataset sources, citations/DOIs, licenses | **VERIFIED** (documentation) | `DATASET_LICENSES.md`, `THIRD_PARTY.md`. Audio not in Git. SNIPS **unverified**. Student recordings need consent. |
| 3 | Training logs and final checkpoint | **PARTIAL** | Logs: `reports/training_log.csv` (epoch 33, val macro F1 0.797492). Checkpoint SHA `4c653cf5…` is documented; **`.pt` is not in Git**. Weight license **undecided**. |
| 4 | Raspberry Pi **5** latency (measured hardware) | **VERIFIED** (documented run) | Raspberry Pi **5 Model B Rev 1.1**, 24 Oct 2026. Metrics in `reports/live_pi5_20261024/`. Course harness is **not** in this repo; the live run is **not** independently reproducible from a GitHub clone. No on-disk course PDF found that requires Pi 4; if the syllabus still names Pi 4, this is a **hardware mismatch** (Pi 5 was tested). |
| 5 | Held-out test, unseen speakers | **VERIFIED** (offline protocol) | n=4758; overlap 0 in `reports/mix_summary.json`. Live HF holdout is a different set. |
| 6 | Comparable-size baseline | **VERIFIED** with limitation | Same architecture, Dataset B, **CPU**: acc 0.664985, F1 0.527704. Not hardware-matched to A100. |
