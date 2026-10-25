# Local weight copies (not for Git)

| File | Role |
|---|---|
| `best_model.pt` | Safe Mix best checkpoint (epoch 33). Gitignored. |
| `mel_norm.pt` | Locked per-mel mean/std. Gitignored if large policy applies; 2.6 KiB. |

Obtain these from the HPC package transfer or a future GitHub Release. Verify SHA-256 against `../reports/CHECKSUMS.txt`.
