#!/usr/bin/env python3
"""Evaluate a VCM-DS-Tiny checkpoint on locked VCM_MASTER test.

This is evaluation of released weights, not training from scratch.
Requires CUDA. Does not write into the original ME2_VCM tree.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import VCMDataset, collate_batch
from src.metrics import classification_report_text
from src.model import VCMDSTiny, count_parameters
from src.normalization import MelNormalizer
from src.utils import ID_TO_CLASS, load_config, set_seed
from train_targeted_temp_run3 import evaluate, precompute_eval, save_cm_csv, write_metrics_csv

RECORDED = {
    "accuracy": 0.8400588482555695,
    "macro_f1": 0.7785515388212305,
    "best_epoch": 33,
    "trainable": 95376,
}


def pack_metrics(res):
    out = {
        "accuracy": float(res["macro"]["accuracy"]),
        "macro_f1": float(res["macro"]["macro_f1"]),
        "weighted_f1": float(res["weighted"]["weighted_f1"]),
    }
    for i in range(16):
        out[ID_TO_CLASS[i]] = float(res["prf"]["f1"][i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "safe_mix_run1.yaml"))
    ap.add_argument("--checkpoint", default=str(ROOT / "artifacts" / "best_model.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "repro_eval"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    if not torch.cuda.is_available():
        raise SystemExit("STOP: CUDA required (do not fall back to CPU)")
    device = torch.device("cuda")

    test_manifest = Path(cfg["test_manifest"])
    test_root = Path(cfg["test_dataset"])
    if not test_manifest.is_file():
        raise SystemExit(f"STOP: missing test manifest {test_manifest} (set VCM_MASTER)")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"STOP: missing checkpoint {ckpt_path}")

    normalizer = MelNormalizer.from_file(cfg["norm_path"])
    test_raw = VCMDataset(str(test_manifest), str(test_root), cfg, "eval", normalizer, cache_waveforms=True)
    test_ds = precompute_eval(test_raw)
    loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_batch)

    model = VCMDSTiny(num_classes=16, dropout=float(cfg.get("dropout", 0.20)))
    pstats = count_parameters(model)
    if pstats["trainable"] != RECORDED["trainable"]:
        raise SystemExit(f"STOP: architecture mismatch {pstats['trainable']}")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_res = evaluate(model, loader, criterion, device)
    got = pack_metrics(test_res)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(test_res, out_dir / "best_test_metrics.csv")
    save_cm_csv(test_res["cm"], out_dir / "best_test_confusion_matrix.csv")
    (out_dir / "test_classification_report.txt").write_text(classification_report_text(test_res["cm"]))

    delta_acc = got["accuracy"] - RECORDED["accuracy"]
    delta_f1 = got["macro_f1"] - RECORDED["macro_f1"]
    match = abs(delta_acc) < 1e-6 and abs(delta_f1) < 1e-6
    summary = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": blob.get("epoch"),
        "checkpoint_source": blob.get("source"),
        "trainable_parameters": pstats["trainable"],
        "n_test": len(test_ds),
        "got": got,
        "recorded": RECORDED,
        "delta_accuracy": delta_acc,
        "delta_macro_f1": delta_f1,
        "matches_recorded": match,
        "evaluation_not_training": True,
    }
    (out_dir / "repro_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not match:
        print("NOTE: reproduced metrics differ from recorded Safe Mix numbers.", flush=True)
        return 2
    print("REPRO_EVAL_MATCH", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
