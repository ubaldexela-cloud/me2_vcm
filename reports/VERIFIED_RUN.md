# Verified Safe Mix training run

Checked against job 66143 stdout, `sacct -j 66143`, recorded configs, and metric CSVs. Rounded slide figures are shown next to full recorded values. Machine-readable copy: `reports/TRAINING_METADATA.json`.

| Claim | File value | Status |
|---|---|---|
| Model VCM-DS-Tiny / VCMDSTiny | `src/model.py`, config `model: VCM-DS-Tiny` | Match |
| Input `(B, 1, 40, 397)` | config `feature_shape`, model docstring | Match |
| Output 16 logits | `forward` returns `self.fc(x)`; no Softmax | Match |
| Trainable parameters 95,376 | log `params=95376`; export report | Match |
| MACs 32,890,816 | log `macs=32890816` | Match |
| Seed 42 | `configs` / recorded training yaml | Match |
| AdamW, lr 0.001, wd 0.0001 | recorded yaml | Match |
| Batch size 64 | recorded yaml | Match |
| Cross-entropy | `nn.CrossEntropyLoss()` | Match |
| ReduceLROnPlateau on val macro F1 | scheduler `mode=max` + `scheduler.step(val_f1)` | Match |
| Max epochs 50; patience 7; stop epoch 40 | log `early_stopping_patience_7` after epoch 40 | Match |
| Best epoch 33 | `training_log.csv` epoch 33 | Match |
| Val accuracy (epoch 33) | **0.834178284748627** | Match (slide rounding 0.8342) |
| Val macro F1 (epoch 33) | **0.797492080662753** | Match (slide rounding 0.7975) |
| Locked-test acc | **0.8400588482555695** | Match (slide rounding 0.8401) |
| Locked-test macro F1 | **0.7785515388212305** | Match (slide rounding 0.7786) |
| Locked-test weighted F1 | **0.835083840264589** | Match (`best_test_metrics.csv`) |
| Script train_time_sec | **28535.5549929142** (logged 28,535.6) | Match. Excludes init, data load, locked test |
| Slurm wall job 66143 | **29181 s** (08:06:21; 2026-10-23T03:12:16 → 11:18:37) | Match (`sacct`) |
| Node | `uperdfi-a100gpu0` | Match (`sacct` NodeList) |
| GPU used | one NVIDIA A100 80GB PCIe | Match (job log device name; `--gres=...:1`). AllocTRES has no GPU TRES |
| PyTorch | 2.8.0+cu128 | Match (job log) |
| CUDA driver / `torch.version.cuda` | not printed | **Unverified** |
| FP32 | **No `amp` / `autocast` / precision key.** Train loop uses default float tensors. | **Implicit, not an explicit recorded setting** |
| Checkpoint SHA-256 | `4c653cf5710ab46da4a94301045d47cb574a508d30b626f8158304825aa43e7f` | Match (`sha256sum`) |
| `mel_norm.pt` SHA-256 | `3a67c19aa467941fc55cdc7dabf4a6faa08deea3c28bcfd8c750ba66da9ec2ed` | Match (`sha256sum` + job `locked_sha.norm`) |
| Speaker / audio-SHA overlap | 0 / 0 / 0 / 0 | Match (job `leak_check` + `mix_summary.json`) |

Checkpoint-selection criterion: highest **validation macro F1**. Test was evaluated **once** after selection.

Endpointing is **not** part of this training run. It is an inference-time capture helper (`inference/command_endpoint.py`) that still feeds 64,000 samples to the same model.

The public GitHub tree does not include `VCM_MASTER` audio or `.pt` weights, so it cannot reproduce the locked-test scores or retrain from scratch.
