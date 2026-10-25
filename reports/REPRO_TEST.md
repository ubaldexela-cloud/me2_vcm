# Reproduction test log

Status: **tested on HPC A100** (Slurm job 66249, 2026-10-25, 4 min 58 s).\
This is **evaluation of released weights**, not training from scratch.

## Command actually run

```bash
export VCM_MASTER=/path/to/VCM_MASTER          # locked test audio required
export MEL_NORM=$PWD/artifacts/mel_norm.pt
export PYTHONPATH=/tmp/me2_pydeps:${PYTHONPATH}
python3 scripts/eval_locked_test.py \
  --checkpoint artifacts/best_model.pt \
  --out-dir reports/repro_eval
```

(The submitted job used the cluster path for `VCM_MASTER` and `scripts/eval_locked_test.sbatch`.)

## Outcome

| Item | Result |
|---|---|
| Device | NVIDIA A100 80GB PCIe |
| PyTorch | 2.8.0+cu128 |
| Checkpoint epoch / source | 33 / `best` |
| Parameters | 95,376 |
| n_test | 4,758 |
| Accuracy | 0.8400588482555695 |
| Macro F1 | 0.7785515388212305 |
| Δ vs recorded | **0.0 / 0.0** |
| Exit | `REPRO_EVAL_MATCH` |

Outputs: `reports/repro_eval/repro_summary.json` and the regenerated metrics CSVs.

## Limitations

- Not a clean-laptop test. Torch came from the existing `/tmp/me2_pydeps` + system Python 3.9 used for Safe Mix.
- Locked test audio is **not** in this package. Without `VCM_MASTER`, the command fails.
- CPU is refused by the script (cluster GPU-only rule).
- Training from scratch was **not** re-run.
- ONNX Runtime locked-test was **not** re-run in this job (export parity already exists in the original export report).
