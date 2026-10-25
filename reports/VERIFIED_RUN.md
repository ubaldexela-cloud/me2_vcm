# Verified Safe Mix training run

Checked against the original logs and configs. Rounded slide figures are shown next to full recorded values.

| Claim | File value | Status |
|---|---|---|
| Model VCM-DS-Tiny / VCMDSTiny | `src/model.py`, config `model: VCM-DS-Tiny` | Match |
| Input `(B, 1, 40, 397)` | config `feature_shape`, model docstring | Match |
| Output 16 logits | `forward` returns `self.fc(x)`; no Softmax | Match |
| Trainable parameters 95,376 | log `params=95376`; export report | Match |
| Seed 42 | `configs` / recorded training yaml | Match |
| AdamW, lr 0.001, wd 0.0001 | recorded yaml | Match |
| Batch size 64 | recorded yaml | Match |
| Cross-entropy | `nn.CrossEntropyLoss()` | Match |
| ReduceLROnPlateau on val macro F1 | scheduler `mode=max` + `scheduler.step(val_f1)` | Match |
| Max epochs 50; patience 7; stop epoch 40 | log `early_stopping_patience_7` after epoch 40 | Match |
| Best epoch 33, val macro F1 0.7975 | `0.797492080662753` | Match (rounded) |
| Locked-test acc 0.8401 | `0.8400588482555695` | Match (rounded) |
| Locked-test macro F1 0.7786 | `0.7785515388212305` | Match (rounded) |
| Train time 28,535.6 s | `28535.5549929142` | Match (rounded) |
| NVIDIA A100 80GB PCIe | training log | Match |
| Seed 42 | yaml | Match |
| FP32 | **No `amp` / `autocast` / precision key.** Train loop uses default float tensors. | **Implicit, not an explicit recorded setting** |

Checkpoint-selection criterion: highest **validation macro F1**. Test was evaluated **once** after selection.

Endpointing is **not** part of this training run. It is an inference-time capture helper (`inference/command_endpoint.py`) that still feeds 64,000 samples to the same model.
