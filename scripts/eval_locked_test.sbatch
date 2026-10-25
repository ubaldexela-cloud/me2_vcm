#!/bin/bash
#SBATCH --job-name=icy-vcm-repro-eval
#SBATCH --partition=uperdfi-gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=reports/repro_eval_%j.out
#SBATCH --error=reports/repro_eval_%j.err

# Cluster-specific defaults were removed. Export these before sbatch:
#   VCM_MASTER  MEL_NORM  PYTHONPATH
set -euo pipefail
: "${VCM_MASTER:?set VCM_MASTER to the locked test dataset root}"
: "${MEL_NORM:?set MEL_NORM to artifacts/mel_norm.pt}"
cd "${SLURM_SUBMIT_DIR:-.}"
python3 scripts/eval_locked_test.py \
  --checkpoint artifacts/best_model.pt \
  --out-dir reports/repro_eval
