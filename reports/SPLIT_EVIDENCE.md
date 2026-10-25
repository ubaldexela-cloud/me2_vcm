# Speaker / SHA split evidence (Safe Mix train vs locked A val/test)

Primary records: job 66143 stdout `leak_check=...` and `reports/mix_summary.json`.
Public train index: `manifests/safe_mix_train_index.csv` (SHA-256 + class + pool only; **no filepaths, no speaker IDs**).

| Check | Count | Primary record |
|---|---:|---|
| Train clips | 30254 | mix_summary `n_train`; job log `mix n=30254` |
| Unique train SHA-256 | 30254 | mix_summary `unique_sha` |
| Train speakers | 418 | mix_summary `speakers` |
| Train speakers ∩ A val speakers | **0** | mix_summary / job `speaker_overlap_val` |
| Train speakers ∩ A test speakers | **0** | mix_summary / job `speaker_overlap_test` |
| Train SHA ∩ A val SHA | **0** | mix_summary / job `sha_overlap_val` |
| Train SHA ∩ A test SHA | **0** | mix_summary / job `sha_overlap_test` |
| Locked val clips | 4734 | Dataset A `val.csv` (audio not in Git) |
| Locked test clips | 4758 | Dataset A `test.csv`; `best_test_metrics.csv` |

Pools: `vcm_balanced` 15266 + `a_train_leftover` 13329 + `hf_usable` 1659.

Image hashes: **not applicable**. This project stores audio SHA-256, not image hashes.

## What a GitHub-only reviewer can recompute

- **Cannot** recompute speaker overlap from the public tree: the published train index omits `speaker_id`.
- **Cannot** recompute SHA overlap without the locked Dataset A val/test SHA columns (those manifests live in private `VCM_MASTER`).
- The zeros above are recorded from the training-job leak check, which aborted if any overlap was nonzero.

Locked val/test audio is **not** in this repository.
