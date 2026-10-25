# Held-out speakers (offline Dataset A)

From `reports/mix_summary.json` and job 66143 `leak_check` (manifest mix, no audio in Git). Full table: `reports/SPLIT_EVIDENCE.md`.

- Train clips: 30254 unique SHA-256
- Speakers in mix: 418
- `speaker_overlap_val`: 0
- `speaker_overlap_test`: 0
- `sha_overlap_val`: 0
- `sha_overlap_test`: 0
- Locked val/test: Dataset A (`n_val` 4734 / `n_test` 4758)

Image hashes are **not applicable** (audio SHA-256 only).

A GitHub-only clone **cannot** recompute speaker overlap: `manifests/safe_mix_train_index.csv` omits speaker IDs and filepaths. SHA overlap also needs private Dataset A val/test SHA columns.

The live course benchmark uses Hugging Face holdout clips, **not** Dataset A test.
