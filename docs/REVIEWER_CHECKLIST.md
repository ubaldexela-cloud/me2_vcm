# Reviewer checklist

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Public repository and one-command reproduction | **PENDING** until first approved push. **BLOCKED** as “one-command 84% repro” | Empty public repo `ubaldexela-cloud/me2_vcm`. Laptop command `python scripts/verify_release.py` checks hashes only. Locked-test needs `VCM_MASTER` + CUDA + gitignored `.pt` (`reports/REPRO_TEST.md`). |
| 2 | Dataset sources, citations/DOIs, licenses | **VERIFIED** (documentation) | `DATASET_LICENSES.md`, `THIRD_PARTY.md`. Audio not in Git. SNIPS **unverified**. |
| 3 | Training logs and final checkpoint | **VERIFIED** locally / **PENDING** public checkpoint | `reports/training_log.csv`, epoch 33, val F1 0.797492. `best_model.pt` SHA `4c653cf5…` gitignored. Weight license undecided. |
| 4 | Raspberry Pi **4** latency from published script | **PENDING** | Measured board is Pi **5**. |
| 5 | Held-out test, unseen speakers | **VERIFIED** (offline) | n=4758; overlap 0 in `mix_summary.json`. Live HF holdout is a different set. |
| 6 | Comparable-size baseline | **VERIFIED** with limitation | Same architecture, Dataset B, **CPU**: acc 0.664985, F1 0.527704. Not hardware-matched to A100. |
