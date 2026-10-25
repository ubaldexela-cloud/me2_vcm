#!/usr/bin/env python3
"""
ME2 VCM Targeted SET_TEMPERATURE Run #3.

Preserves Run #2 acoustic augmentation.
Adds: real-focused WeightedRandomSampler + SET_TEMPERATURE-only speed/pitch/SpecAugment.

Does NOT modify Dataset A/B, mel_norm, run1, or run2.
Does NOT start Run #4.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_extra = os.environ.get("ICY_EXTRA_PYTHONPATH", "")
if _extra:
    sys.path.insert(0, _extra)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.augment import WaveformAugmentor, discover_noise_files
from src.dataset import VCMDataset, collate_batch
from src.features import LogMelExtractor
from src.metrics import (
    classification_report_text,
    confidence_stats,
    confusion_matrix,
    macro_scores,
    per_class_prf,
    special_class_confusion,
    weighted_scores,
)
from src.model import VCMDSTiny, count_parameters
from src.normalization import MelNormalizer
from src.temp_augment import SetTemperatureSpeakerAugmentor
from src.utils import CLASS_TO_ID, ID_TO_CLASS, load_config, set_seed

COMMAND_IDS = list(range(14))
TEMP_ID = 8
UNKNOWN_ID = 14
SILENCE_ID = 15


def estimate_macs(num_mels=40, num_frames=397) -> int:
    def conv_out(L, k=3, s=1, p=1):
        return (L + 2 * p - k) // s + 1

    def macs_conv(cin, cout, k, hout, wout, groups=1):
        return (cin // groups) * k * k * cout * hout * wout

    H, W = num_mels, num_frames
    macs = macs_conv(1, 32, 3, H, W)
    H, W = conv_out(H, 3, 2, 1), conv_out(W, 3, 2, 1)
    macs += macs_conv(32, 32, 3, H, W, groups=32) + macs_conv(32, 64, 1, H, W)
    H, W = conv_out(H, 3, 2, 1), conv_out(W, 3, 2, 1)
    macs += macs_conv(64, 64, 3, H, W, groups=64) + macs_conv(64, 128, 1, H, W)
    H, W = conv_out(H, 3, 2, 1), conv_out(W, 3, 2, 1)
    macs += macs_conv(128, 128, 3, H, W, groups=128) + macs_conv(128, 192, 1, H, W)
    H, W = conv_out(H, 3, 2, 1), conv_out(W, 3, 2, 1)
    macs += macs_conv(192, 192, 3, H, W, groups=192) + macs_conv(192, 256, 1, H, W)
    return macs + 256 * 16


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_audio(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


def count_manifest(path: Path) -> int:
    return sum(1 for _ in open(path)) - 1


def integrity_check(cfg, log, phase: str) -> dict:
    a_root = Path(os.environ.get("VCM_MASTER", "VCM_MASTER"))
    b_root = Path(os.environ.get("VCM_BALANCED", "VCM_BALANCED"))
    norm = Path(cfg["norm_path"])
    r1 = Path(cfg["run1_ckpt"])
    r2 = Path(cfg["run2_ckpt"])
    info = {
        "phase": phase,
        "A_audio": count_audio(a_root / "audio"),
        "B_audio": count_audio(b_root / "audio"),
        "A_train": count_manifest(a_root / "manifests/train.csv"),
        "A_val": count_manifest(a_root / "manifests/val.csv"),
        "A_test": count_manifest(a_root / "manifests/test.csv"),
        "B_train": count_manifest(b_root / "manifests/train.csv"),
        "A_train_sha": file_sha256(a_root / "manifests/train.csv"),
        "A_val_sha": file_sha256(a_root / "manifests/val.csv"),
        "A_test_sha": file_sha256(a_root / "manifests/test.csv"),
        "B_train_sha": file_sha256(b_root / "manifests/train.csv"),
        "norm_sha": file_sha256(norm),
        "run1_sha": file_sha256(r1),
        "run2_sha": file_sha256(r2),
        "run1_mtime": os.path.getmtime(r1),
        "run2_mtime": os.path.getmtime(r2),
    }
    ok_a = info["A_audio"] == 36622 and info["A_train"] == 27130 and info["A_val"] == 4734 and info["A_test"] == 4758
    ok_b = info["B_audio"] == 15268 and info["B_train"] == 15268
    ok_norm = info["norm_sha"].startswith("3a67c19aa467941f")
    info["ok"] = ok_a and ok_b and ok_norm and r1.is_file() and r2.is_file()
    info["Dataset A modified"] = "NO" if ok_a else "YES"
    info["Dataset B modified"] = "NO" if ok_b else "YES"
    info["Normalization modified"] = "NO" if ok_norm else "YES"
    log(f"integrity[{phase}] ok={info['ok']} A={info['A_audio']} B={info['B_audio']}")
    return info


def audit_set_temperature(train_ds: VCMDataset) -> dict:
    real_orig = real_aug = synth_orig = synth_aug = 0
    for r in train_ds.rows:
        if r[train_ds.label_key] != "SET_TEMPERATURE":
            continue
        is_real = (r.get("source_type") == "real") or (r.get("source_dataset") == "SET_TEMPERATURE_REAL")
        oa = r.get("original_or_augmented", "original")
        if is_real and oa == "original":
            real_orig += 1
        elif is_real:
            real_aug += 1
        elif oa == "original":
            synth_orig += 1
        else:
            synth_aug += 1
    return {
        "total": real_orig + real_aug + synth_orig + synth_aug,
        "real_original": real_orig,
        "real_augmented": real_aug,
        "synthetic_original": synth_orig,
        "synthetic_augmented": synth_aug,
        "real_derived": real_orig + real_aug,
        "synthetic_derived": synth_orig + synth_aug,
        "expected_match": (
            real_orig == 60
            and real_aug == 60
            and synth_orig == 300
            and synth_aug == 180
            and (real_orig + real_aug + synth_orig + synth_aug) == 600
        ),
    }


def build_sample_weights(train_ds: VCMDataset, cfg: dict) -> tuple:
    st = cfg["set_temperature"]
    w_real = float(st["real_sample_weight"])
    w_synth = float(st["synthetic_sample_weight"])
    w_other = float(st["other_class_weight"])
    weights = []
    n_real = n_synth = n_other = 0
    for r in train_ds.rows:
        if r[train_ds.label_key] == "SET_TEMPERATURE":
            is_real = (r.get("source_type") == "real") or (r.get("source_dataset") == "SET_TEMPERATURE_REAL")
            if is_real:
                weights.append(w_real)
                n_real += 1
            else:
                weights.append(w_synth)
                n_synth += 1
        else:
            weights.append(w_other)
            n_other += 1
    tw = float(sum(weights))
    n = len(weights)
    # Expected draws per epoch (with replacement, num_samples=n)
    exp_temp_real = n * (n_real * w_real) / tw
    exp_temp_synth = n * (n_synth * w_synth) / tw
    exp_temp = exp_temp_real + exp_temp_synth
    pct_real_of_temp = 100.0 * exp_temp_real / max(exp_temp, 1e-12)
    stats = {
        "n_dataset": n,
        "n_temp_real_files": n_real,
        "n_temp_synth_files": n_synth,
        "n_other_files": n_other,
        "weight_real": w_real,
        "weight_synth": w_synth,
        "weight_other": w_other,
        "total_weight": tw,
        "expected_temp_draws_per_epoch": exp_temp,
        "expected_temp_real_draws_per_epoch": exp_temp_real,
        "expected_temp_synth_draws_per_epoch": exp_temp_synth,
        "pct_temp_exposure_from_real": pct_real_of_temp,
        "unweighted_pct_temp_real_files": 100.0 * n_real / max(n_real + n_synth, 1),
    }
    return torch.tensor(weights, dtype=torch.double), stats


class MetaFeatureDataset(Dataset):
    def __init__(self, feats, labels, metas):
        self.feats, self.labels, self.metas = feats, labels, metas

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, idx):
        return self.feats[idx], int(self.labels[idx]), self.metas[idx]


def precompute_eval(ds: VCMDataset) -> MetaFeatureDataset:
    assert ds.augmentor is None and ds.temp_augmentor is None
    assert ds.split_mode == "eval"
    feats, labels, metas = [], [], []
    n = len(ds)
    for i in range(n):
        f, y, m = ds[i]
        feats.append(f)
        labels.append(y)
        metas.append(m)
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"  precompute eval: {i+1}/{n}", flush=True)
    return MetaFeatureDataset(torch.stack(feats), torch.tensor(labels, dtype=torch.long), metas)


def save_cm_csv(cm, path: Path):
    names = [ID_TO_CLASS[i] for i in range(16)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + names)
        for i, row in enumerate(cm):
            w.writerow([names[i]] + [int(x) for x in row])


def save_cm_png(cm, path: Path, title: str, normalize=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [ID_TO_CLASS[i] for i in range(16)]
    data = cm.astype(np.float64)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(16), yticks=range(16), xticklabels=names, yticklabels=names, title=title, ylabel="True", xlabel="Pred")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = data.max() / 2
    for i in range(16):
        for j in range(16):
            txt = f"{data[i,j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", color="white" if data[i, j] > thresh else "black", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(rows, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(ep, [r["train_loss"] for r in rows], label="train")
    axes[0, 0].plot(ep, [r["val_loss"] for r in rows], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(ep, [r["train_accuracy"] for r in rows], label="train")
    axes[0, 1].plot(ep, [r["val_accuracy"] for r in rows], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[1, 0].plot(ep, [r["val_macro_f1"] for r in rows], label="val_macro_f1")
    axes[1, 0].set_title("Val macro F1")
    axes[1, 0].legend()
    axes[1, 1].plot(ep, [r["learning_rate"] for r in rows], label="lr")
    axes[1, 1].set_title("LR")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_details=False):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred, max_probs, details = [], [], [], []
    for feats, labels, metas in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        max_probs.extend(conf.cpu().tolist())
        if collect_details:
            for i in range(bs):
                details.append(
                    {
                        "file": metas[i]["filepath"],
                        "true_class": ID_TO_CLASS[int(labels[i])],
                        "predicted_class": ID_TO_CLASS[int(pred[i])],
                        "confidence": float(conf[i]),
                        "source_dataset": metas[i].get("source_dataset", ""),
                        "source_type": metas[i].get("source_type", ""),
                        "speaker": metas[i].get("speaker", ""),
                    }
                )
    cm = confusion_matrix(y_true, y_pred, 16)
    return {
        "loss": total_loss / max(n, 1),
        "cm": cm,
        "macro": macro_scores(cm),
        "weighted": weighted_scores(cm),
        "prf": per_class_prf(cm),
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "max_probs": np.array(max_probs, dtype=np.float64),
        "details": details,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    temp_seen = real_temp_seen = 0
    for feats, labels, metas in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((logits.argmax(1) == labels).sum().item())
        n += bs
        for m, y in zip(metas, labels.cpu().tolist()):
            if y == TEMP_ID:
                temp_seen += 1
                if m.get("source_type") == "real" or m.get("source_dataset") == "SET_TEMPERATURE_REAL":
                    real_temp_seen += 1
    return total_loss / max(n, 1), correct / max(n, 1), {"temp_seen": temp_seen, "real_temp_seen": real_temp_seen}


def write_failures(details, path: Path):
    with open(path, "w", newline="") as f:
        fields = ["file", "true_class", "predicted_class", "confidence", "source_dataset", "source_type", "speaker"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in details:
            if d["true_class"] != d["predicted_class"]:
                row = {k: d[k] for k in fields}
                row["confidence"] = f"{d['confidence']:.6f}"
                w.writerow(row)


def write_metrics_csv(res, path: Path):
    prf = res["prf"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
        for i in range(16):
            w.writerow([i, ID_TO_CLASS[i], f"{prf['precision'][i]:.6f}", f"{prf['recall'][i]:.6f}", f"{prf['f1'][i]:.6f}", int(prf["support"][i])])
        w.writerow([])
        for k, v in [
            ("accuracy", res["macro"]["accuracy"]),
            ("macro_precision", res["macro"]["macro_precision"]),
            ("macro_recall", res["macro"]["macro_recall"]),
            ("macro_f1", res["macro"]["macro_f1"]),
            ("weighted_precision", res["weighted"]["weighted_precision"]),
            ("weighted_recall", res["weighted"]["weighted_recall"]),
            ("weighted_f1", res["weighted"]["weighted_f1"]),
        ]:
            w.writerow([k, v])


def write_confidence(res, path: Path):
    y_true, y_pred, mp = res["y_true"], res["y_pred"], res["max_probs"]
    lines = ["CONFIDENCE ANALYSIS", ""]

    def block(name, mask):
        st = confidence_stats(mp[mask])
        lines.append(f"=== {name} ===")
        for k, v in st.items():
            lines.append(f"  {k}: {v}" if k == "count" else f"  {k}: {v:.6f}" if v == v else f"  {k}: nan")
        lines.append("")

    correct = y_true == y_pred
    block("all", np.ones(len(mp), dtype=bool))
    block("correct", correct)
    block("incorrect", ~correct)
    block("SET_TEMPERATURE (true)", y_true == TEMP_ID)
    block("UNKNOWN (true)", y_true == UNKNOWN_ID)
    block("SILENCE (true)", y_true == SILENCE_ID)
    path.write_text("\n".join(lines) + "\n")


def set_temperature_analysis(val_res, test_res, composition, sampler_stats, path: Path):
    lines = ["SET_TEMPERATURE ANALYSIS — targeted_temp_run3", ""]
    lines.append("## Training composition (Dataset B, unchanged on disk)")
    for k, v in composition.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("## Sampler / effective exposure (expected per epoch)")
    for k, v in sampler_stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    for split, res in [("VALIDATION", val_res), ("TEST", test_res)]:
        prf = res["prf"]
        cm = res["cm"]
        lines.append(f"## {split} SET_TEMPERATURE metrics")
        lines.append(f"  precision: {prf['precision'][TEMP_ID]:.6f}")
        lines.append(f"  recall:    {prf['recall'][TEMP_ID]:.6f}")
        lines.append(f"  f1:        {prf['f1'][TEMP_ID]:.6f}")
        lines.append(f"  support:   {int(prf['support'][TEMP_ID])}")
        lines.append(f"  Where true SET_TEMPERATURE was classified:")
        for j in range(16):
            lines.append(f"    SET_TEMPERATURE → {ID_TO_CLASS[j]} = {int(cm[TEMP_ID, j])}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def measure_latency(model, device, extractor, normalizer, n_warmup=10, n_iter=50):
    model.eval()
    wav = torch.zeros(64000)
    for _ in range(n_warmup):
        _ = normalizer(extractor(wav))
    times_pre = []
    for _ in range(n_iter):
        t1 = time.perf_counter()
        feat = normalizer(extractor(wav))
        times_pre.append(time.perf_counter() - t1)
    x = feat.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        times_inf = []
        for _ in range(n_iter):
            t1 = time.perf_counter()
            _ = model(x)
            times_inf.append(time.perf_counter() - t1)
    pre = 1000 * float(np.median(times_pre))
    inf = 1000 * float(np.median(times_inf))
    return {"preprocess_ms": pre, "model_ms": inf, "total_ms": pre + inf, "device": str(device)}


def load_run2_metrics(run2_dir: Path):
    def load(csv_path):
        rows, overall = {}, {}
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6 and parts[0].isdigit():
                    rows[int(parts[0])] = {"f1": float(parts[4]), "precision": float(parts[2]), "recall": float(parts[3])}
                elif len(parts) == 2:
                    try:
                        overall[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
        return rows, overall

    return load(run2_dir / "validation_metrics.csv"), load(run2_dir / "test_metrics.csv")


def build_comparison(run2_dir, val_res, test_res, out_path: Path):
    (r2_val_rows, r2_val_o), (r2_test_rows, r2_test_o) = load_run2_metrics(run2_dir)
    lines = ["RUN2 vs RUN3 COMPARISON", "=" * 72, ""]
    lines.append(f"{'Metric':<32} {'Run2':>12} {'Run3':>12} {'Delta':>12}")
    lines.append("-" * 72)

    def row(name, a, b):
        lines.append(f"{name:<32} {a:12.6f} {b:12.6f} {b-a:+12.6f}")

    row("Validation Accuracy", r2_val_o["accuracy"], val_res["macro"]["accuracy"])
    row("Validation Macro F1", r2_val_o["macro_f1"], val_res["macro"]["macro_f1"])
    row("Test Accuracy", r2_test_o["accuracy"], test_res["macro"]["accuracy"])
    row("Test Macro F1", r2_test_o["macro_f1"], test_res["macro"]["macro_f1"])
    row("Test Weighted F1", r2_test_o["weighted_f1"], test_res["weighted"]["weighted_f1"])
    row("SET_TEMPERATURE Precision (test)", r2_test_rows[8]["precision"], float(test_res["prf"]["precision"][8]))
    row("SET_TEMPERATURE Recall (test)", r2_test_rows[8]["recall"], float(test_res["prf"]["recall"][8]))
    row("SET_TEMPERATURE F1 (test)", r2_test_rows[8]["f1"], float(test_res["prf"]["f1"][8]))
    row("SILENCE F1 (test)", r2_test_rows[15]["f1"], float(test_res["prf"]["f1"][15]))
    row("UNKNOWN F1 (test)", r2_test_rows[14]["f1"], float(test_res["prf"]["f1"][14]))
    lines.append("")
    lines.append("Per-class TEST F1:")
    improved, degraded = [], []
    for i in range(16):
        a, b = r2_test_rows[i]["f1"], float(test_res["prf"]["f1"][i])
        st = "improved" if b > a + 1e-6 else ("degraded" if b < a - 1e-6 else "similar")
        if st == "improved":
            improved.append(ID_TO_CLASS[i])
        elif st == "degraded":
            degraded.append(ID_TO_CLASS[i])
        lines.append(f"  {ID_TO_CLASS[i]:<18} {a:8.4f} → {b:8.4f} ({b-a:+.4f}) {st}")
    lines.append("")
    lines.append(f"Improved: {improved}")
    lines.append(f"Degraded: {degraded}")
    d_macro = test_res["macro"]["macro_f1"] - r2_test_o["macro_f1"]
    sil_ok = float(test_res["prf"]["f1"][15]) > 0.90
    unk_ok = float(test_res["prf"]["f1"][14]) > 0.80
    temp_f1 = float(test_res["prf"]["f1"][8])
    temp_improved = temp_f1 > 0.0
    macro_ok = d_macro >= -0.03
    lines.append("")
    lines.append(f"SET_TEMPERATURE test F1: {r2_test_rows[8]['f1']:.4f} → {temp_f1:.4f}")
    lines.append(f"macro-F1 delta: {d_macro:+.6f}")
    lines.append(f"SILENCE remains high: {sil_ok} ({float(test_res['prf']['f1'][15]):.4f})")
    lines.append(f"UNKNOWN remains high: {unk_ok} ({float(test_res['prf']['f1'][14]):.4f})")
    out_path.write_text("\n".join(lines) + "\n")
    return {
        "temp_f1_r2": r2_test_rows[8]["f1"],
        "temp_f1_r3": temp_f1,
        "temp_improved": temp_improved,
        "d_macro": d_macro,
        "sil_ok": sil_ok,
        "unk_ok": unk_ok,
        "macro_ok": macro_ok,
        "improved": improved,
        "degraded": degraded,
        "r2_test_macro": r2_test_o["macro_f1"],
        "r2_sil": r2_test_rows[15]["f1"],
        "r2_unk": r2_test_rows[14]["f1"],
    }


def main():
    cfg = load_config(ROOT / "configs" / "targeted_temp_run3.yaml")
    set_seed(int(cfg["seed"]))
    ckpt_dir = Path(cfg["checkpoint_dir"])
    res_dir = Path(cfg["results_dir"])
    assert "targeted_temp_run3" in str(ckpt_dir)
    assert "targeted_temp_run3" in str(res_dir)
    assert "baseline_run1" not in str(ckpt_dir) and "robustness_run2" not in str(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "configs" / "targeted_temp_run3.yaml", ckpt_dir / "training_config.yaml")
    shutil.copy(ROOT / "configs" / "targeted_temp_run3.yaml", res_dir / "training_config.yaml")

    log_lines = []

    def log(msg):
        line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    pre = integrity_check(cfg, log, "before")
    if not pre["ok"]:
        log("STOP: integrity failure")
        (res_dir / "integrity_FAILURE.txt").write_text(json.dumps(pre, indent=2))
        sys.exit(1)
    # Snapshot run1/run2 mtimes to verify unchanged later
    r1_mtime0, r2_mtime0 = pre["run1_mtime"], pre["run2_mtime"]
    r1_sha0, r2_sha0 = pre["run1_sha"], pre["run2_sha"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__}")

    # Noise + Run2 augmentor
    noise_files = discover_noise_files(cfg["augmentation"]["noise_dir"])
    if not noise_files:
        log("STOP: no noise files")
        sys.exit(1)
    bands = [tuple(x) for x in cfg["augmentation"]["snr_bands"]]
    augmentor = WaveformAugmentor(
        sample_rate=16000,
        noise_paths=noise_files,
        gain_p=cfg["augmentation"]["gain_p"],
        gain_min=cfg["augmentation"]["gain_min"],
        gain_max=cfg["augmentation"]["gain_max"],
        shift_p=cfg["augmentation"]["shift_p"],
        shift_ms=cfg["augmentation"]["shift_ms"],
        noise_p=cfg["augmentation"]["noise_p"],
        snr_bands=bands,
        seed=int(cfg["seed"]),
    )
    st = cfg["set_temperature"]
    temp_aug = SetTemperatureSpeakerAugmentor(
        sample_rate=16000,
        num_samples=64000,
        speed_p=st["speed_p"],
        speed_min=st["speed_min"],
        speed_max=st["speed_max"],
        pitch_p=st["pitch_p"],
        pitch_semitones=st["pitch_semitones"],
        spec_p=st["spec_p"],
        freq_mask_param=st["freq_mask_param"],
        time_mask_param=st["time_mask_param"],
        n_freq_masks=st["n_freq_masks"],
        n_time_masks=st["n_time_masks"],
        seed=int(cfg["seed"]),
    )
    aug_cfg = {"run2_acoustic": augmentor.config_dict(), "set_temperature": temp_aug.config_dict()}
    (res_dir / "augmentation_config.json").write_text(json.dumps(aug_cfg, indent=2) + "\n")
    (ckpt_dir / "augmentation_config.json").write_text(json.dumps(aug_cfg, indent=2) + "\n")

    normalizer = MelNormalizer.from_file(cfg["norm_path"])
    log("loading train with Run2 aug + temp-targeted aug...")
    train_ds = VCMDataset(
        cfg["train_manifest"],
        cfg["train_dataset"],
        cfg,
        split_mode="train",
        normalizer=normalizer,
        cache_waveforms=True,
        augmentor=augmentor,
        temp_augmentor=temp_aug,
    )
    composition = audit_set_temperature(train_ds)
    log(f"SET_TEMPERATURE composition={composition}")
    if not composition["expected_match"]:
        log("STOP: SET_TEMPERATURE composition mismatch")
        sys.exit(1)

    weights, sampler_stats = build_sample_weights(train_ds, cfg)
    (res_dir / "sampler_config.json").write_text(json.dumps(sampler_stats, indent=2) + "\n")
    (ckpt_dir / "sampler_config.json").write_text(json.dumps(sampler_stats, indent=2) + "\n")
    log(f"sampler expected_temp/epoch={sampler_stats['expected_temp_draws_per_epoch']:.1f} "
        f"real_pct_of_temp={sampler_stats['pct_temp_exposure_from_real']:.1f}%")

    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    # generator for sampler reproducibility
    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))

    log("loading val (NO augment)...")
    val_raw = VCMDataset(cfg["val_manifest"], cfg["val_dataset"], cfg, "eval", normalizer, cache_waveforms=True)
    val_ds = precompute_eval(val_raw)
    val_raw._wav_cache = None

    bs = int(cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, shuffle=False, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_batch)

    model = VCMDSTiny(num_classes=16, dropout=float(cfg["dropout"]))
    pstats = count_parameters(model)
    macs = estimate_macs()
    if pstats["trainable"] != 95376 or macs != 32890816:
        log(f"STOP: params/macs mismatch {pstats['trainable']} {macs}")
        sys.exit(1)
    model = model.to(device)
    log(f"params={pstats['trainable']} macs={macs}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(cfg["scheduler_factor"]), patience=int(cfg["scheduler_patience"]), min_lr=float(cfg["min_lr"])
    )

    best_f1, best_epoch, no_improve = -1.0, -1, 0
    stopping_reason = "reached_max_epochs"
    csv_rows = []
    t0_all = time.time()
    temp_epoch_stats = []

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        t0 = time.time()
        tr_loss, tr_acc, temp_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = evaluate(model, val_loader, criterion, device)
        val_f1 = val_res["macro"]["macro_f1"]
        lr_now = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_f1)
        et = time.time() - t0
        temp_epoch_stats.append({"epoch": epoch, **temp_stats})
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_accuracy": tr_acc,
            "val_loss": val_res["loss"],
            "val_accuracy": val_res["macro"]["accuracy"],
            "val_macro_precision": val_res["macro"]["macro_precision"],
            "val_macro_recall": val_res["macro"]["macro_recall"],
            "val_macro_f1": val_f1,
            "val_set_temperature_f1": float(val_res["prf"]["f1"][TEMP_ID]),
            "learning_rate": lr_now,
            "epoch_time": et,
            "temp_seen": temp_stats["temp_seen"],
            "real_temp_seen": temp_stats["real_temp_seen"],
        }
        csv_rows.append(row)
        log(
            f"epoch={epoch} train_loss={tr_loss:.4f} val_macro_f1={val_f1:.4f} "
            f"val_temp_f1={row['val_set_temperature_f1']:.4f} temp_seen={temp_stats['temp_seen']} "
            f"real_temp={temp_stats['real_temp_seen']} lr={lr_now:.6g} time={et:.1f}s"
        )
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_macro_f1": val_f1}, ckpt_dir / "last_model.pt")
        if val_f1 > best_f1 + 1e-12:
            best_f1, best_epoch, no_improve = val_f1, epoch, 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_macro_f1": val_f1, "val_metrics": val_res["macro"]},
                ckpt_dir / "best_model.pt",
            )
            log(f"  NEW BEST val_macro_f1={best_f1:.6f}")
        else:
            no_improve += 1
            log(f"  no_improve={no_improve}/{cfg['early_stopping_patience']}")
            if no_improve >= int(cfg["early_stopping_patience"]):
                stopping_reason = f"early_stopping_patience_{cfg['early_stopping_patience']}"
                log(f"EARLY STOP: {stopping_reason}")
                break

    train_time = time.time() - t0_all
    with open(res_dir / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    (res_dir / "temp_exposure_per_epoch.json").write_text(json.dumps(temp_epoch_stats, indent=2) + "\n")
    plot_curves(csv_rows, res_dir)

    blob = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    best_val = evaluate(model, val_loader, criterion, device, collect_details=True)
    write_metrics_csv(best_val, res_dir / "validation_metrics.csv")
    save_cm_csv(best_val["cm"], res_dir / "validation_confusion_matrix.csv")
    save_cm_png(best_val["cm"], res_dir / "validation_confusion_matrix.png", f"Run3 Val CM ep{best_epoch}")
    save_cm_png(best_val["cm"], res_dir / "normalized_validation_confusion_matrix.png", "Run3 Val CM row-norm", True)
    write_failures(best_val["details"], res_dir / "validation_failure_cases.csv")
    write_confidence(best_val, res_dir / "validation_confidence.txt")
    (res_dir / "best_model_validation_report.txt").write_text(classification_report_text(best_val["cm"]))

    log("ONE-TIME test eval...")
    test_raw = VCMDataset(cfg["test_manifest"], cfg["test_dataset"], cfg, "eval", normalizer, cache_waveforms=True)
    test_ds = precompute_eval(test_raw)
    test_raw._wav_cache = None
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    test_res = evaluate(model, test_loader, criterion, device, collect_details=True)
    log(
        f"TEST acc={test_res['macro']['accuracy']:.4f} macro_f1={test_res['macro']['macro_f1']:.4f} "
        f"temp_f1={test_res['prf']['f1'][8]:.4f} sil_f1={test_res['prf']['f1'][15]:.4f} unk_f1={test_res['prf']['f1'][14]:.4f}"
    )
    write_metrics_csv(test_res, res_dir / "test_metrics.csv")
    save_cm_csv(test_res["cm"], res_dir / "test_confusion_matrix.csv")
    save_cm_png(test_res["cm"], res_dir / "test_confusion_matrix.png", "Run3 Test CM")
    save_cm_png(test_res["cm"], res_dir / "normalized_test_confusion_matrix.png", "Run3 Test CM row-norm", True)
    write_failures(test_res["details"], res_dir / "test_failure_cases.csv")
    write_confidence(test_res, res_dir / "test_confidence.txt")
    (res_dir / "test_classification_report.txt").write_text(classification_report_text(test_res["cm"]))
    set_temperature_analysis(best_val, test_res, composition, sampler_stats, res_dir / "set_temperature_analysis.txt")

    lat = measure_latency(model, device, LogMelExtractor(cfg), normalizer)
    ckpt_size = (ckpt_dir / "best_model.pt").stat().st_size
    (res_dir / "model_resource_report.txt").write_text(
        "\n".join(
            [
                "MODEL RESOURCE — targeted_temp_run3",
                f"trainable_parameters: {pstats['trainable']}",
                f"MACs: {macs}",
                f"FP32_KiB: {pstats['fp32_bytes']/1024:.2f}",
                f"checkpoint_MiB: {ckpt_size/1024/1024:.3f}",
                f"preprocess_ms: {lat['preprocess_ms']:.3f}",
                f"model_ms: {lat['model_ms']:.3f}",
                f"total_ms: {lat['total_ms']:.3f}",
                "NOTE: host CPU measurement; not Raspberry Pi.",
            ]
        )
        + "\n"
    )

    cmp = build_comparison(Path(cfg["run2_dir"]), best_val, test_res, res_dir / "RUN2_VS_RUN3_COMPARISON.txt")

    post = integrity_check(cfg, log, "after")
    run1_ok = post["run1_mtime"] == r1_mtime0 and post["run1_sha"] == r1_sha0
    run2_ok = post["run2_mtime"] == r2_mtime0 and post["run2_sha"] == r2_sha0
    hash_ok = all(pre[k] == post[k] for k in ["A_train_sha", "A_val_sha", "A_test_sha", "B_train_sha", "norm_sha", "A_audio", "B_audio"])
    log(f"Run1 unmodified={run1_ok} Run2 unmodified={run2_ok} datasets_hash_ok={hash_ok}")

    temp_f1 = float(test_res["prf"]["f1"][8])
    # Answer the 8 required questions
    q1 = temp_f1 > 0.0
    q2 = "Likely speaker/domain-related if val remains high and test improves; see analysis." if q1 else "Intervention did not demonstrate improved speaker/domain generalization to S5."
    q3 = float(test_res["prf"]["f1"][15]) >= 0.90
    q4 = float(test_res["prf"]["f1"][14]) >= 0.80
    # substantial degrade: any class drop > 0.05 F1 or SILENCE/UNKNOWN collapse
    substantial = []
    (r2_val_rows, _), (r2_test_rows, _) = load_run2_metrics(Path(cfg["run2_dir"]))
    for i in range(16):
        drop = r2_test_rows[i]["f1"] - float(test_res["prf"]["f1"][i])
        if drop > 0.05:
            substantial.append(f"{ID_TO_CLASS[i]} ({drop:+.3f})")
    q5 = len(substantial) > 0
    q6_delta = cmp["d_macro"]
    q7 = (
        "Yes — residual gap is consistent with speaker/domain mismatch (val>test for SET_TEMPERATURE) if test F1 rises under real-focused training; "
        "if test F1 stays 0, evidence that available S1–S3 real data is insufficient for S5."
    )
    success = q1 and q3 and q4 and cmp["macro_ok"] and not (float(test_res["prf"]["f1"][15]) < 0.9)
    if success and not q5:
        q8 = "YES — Run #3 preferable to Run #2 for demo (SET_TEMPERATURE recovered without major regressions)."
    elif q1 and q3 and q4:
        q8 = "MIXED — SET_TEMPERATURE improved but some class regressions; choose carefully for demo."
    elif q3 and q4 and not q1:
        q8 = "NO — prefer Run #2 for demo; SET_TEMPERATURE still fails (need more real speakers, not more tuning)."
    else:
        q8 = "NO — Run #3 not preferable given the measured profile."

    with open(res_dir / "RUN2_VS_RUN3_COMPARISON.txt", "a") as f:
        f.write("\n## Success criteria\n")
        f.write(f"temp_test_f1_above_zero: {q1}\n")
        f.write(f"silence_robust: {q3}\n")
        f.write(f"unknown_robust: {q4}\n")
        f.write(f"macro_not_substantially_degraded: {cmp['macro_ok']} (delta={q6_delta:+.6f})\n")
        f.write(f"preferable_to_run2: {q8}\n")

    report = f"""ME2 VCM — FINAL TARGETED TEMPERATURE RUN #3 REPORT
========================================================================
Generated: {datetime.datetime.now().isoformat(timespec='seconds')}

## Experiment
Hypothesis: SET_TEMPERATURE failure is speaker/domain (S5) rather than noise robustness.
Interventions: real-weight=4.0 sampler; SET_TEMPERATURE-only speed 0.95–1.05, pitch ±1 st, mild SpecAugment.
Preserved: Run #2 gain/shift/noise for all classes. Architecture unchanged (95376 / 32.89M MACs).

## SET_TEMPERATURE composition (verified)
{json.dumps(composition, indent=2)}

## Sampler exposure
{json.dumps(sampler_stats, indent=2)}

## Training
epochs={len(csv_rows)} best_epoch={best_epoch} best_val_macro_f1={best_f1:.6f}
stopping={stopping_reason} train_time_sec={train_time:.1f} device={device} seed={cfg['seed']}

## Best Validation
acc={best_val['macro']['accuracy']:.6f} macro_f1={best_val['macro']['macro_f1']:.6f}
SET_TEMPERATURE P/R/F1={best_val['prf']['precision'][8]:.4f}/{best_val['prf']['recall'][8]:.4f}/{best_val['prf']['f1'][8]:.4f}
SILENCE F1={best_val['prf']['f1'][15]:.4f} UNKNOWN F1={best_val['prf']['f1'][14]:.4f}

## Test (one-shot)
acc={test_res['macro']['accuracy']:.6f} macro_f1={test_res['macro']['macro_f1']:.6f} weighted_f1={test_res['weighted']['weighted_f1']:.6f}
SET_TEMPERATURE P/R/F1={test_res['prf']['precision'][8]:.4f}/{test_res['prf']['recall'][8]:.4f}/{test_res['prf']['f1'][8]:.4f}
SILENCE F1={test_res['prf']['f1'][15]:.4f} UNKNOWN F1={test_res['prf']['f1'][14]:.4f}

## Required answers
1. Did SET_TEMPERATURE test F1 improve above zero? {q1} (F1={temp_f1:.4f}; Run2={cmp['temp_f1_r2']:.4f})
2. Speaker/domain robustness interpretation: {q2}
3. Did SILENCE remain robust? {q3} (F1={float(test_res['prf']['f1'][15]):.4f})
4. Did UNKNOWN remain robust? {q4} (F1={float(test_res['prf']['f1'][14]):.4f})
5. Did any other class substantially degrade (>0.05 F1)? {q5} {substantial if substantial else '(none)'}
6. Overall macro F1 change vs Run2: {q6_delta:+.6f}
7. Evidence about speaker/domain mismatch: {q7}
8. Is Run #3 preferable to Run #2 for demo? {q8}

## Integrity
Dataset A modified: {post['Dataset A modified']}
Dataset B modified: {post['Dataset B modified']}
Normalization modified: {post['Normalization modified']}
Run #1 modified: {'NO' if run1_ok else 'YES'}
Run #2 modified: {'NO' if run2_ok else 'YES'}

## Resources
params={pstats['trainable']} MACs={macs} checkpoint_MiB={ckpt_size/1024/1024:.3f}
CPU total_ms={lat['total_ms']:.2f} (not Pi)

## Conclusion
{'SET_TEMPERATURE still fails on S5 despite targeted training — recommend additional real-speaker data rather than further model-only tuning.' if not q1 else 'SET_TEMPERATURE improved under targeted training; see comparison for demo preference.'}
No Run #4 started.
"""
    (res_dir / "FINAL_TARGETED_TEMP_RUN3_REPORT.txt").write_text(report)
    (res_dir / "baseline_training_log.txt").write_text("\n".join(log_lines) + "\n")
    (res_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "stopping_reason": stopping_reason,
                "train_time_sec": train_time,
                "val": best_val["macro"],
                "test": {**test_res["macro"], **test_res["weighted"]},
                "test_set_temperature_f1": temp_f1,
                "test_silence_f1": float(test_res["prf"]["f1"][15]),
                "test_unknown_f1": float(test_res["prf"]["f1"][14]),
                "comparison": cmp,
                "answers": {"q1": q1, "q3": q3, "q4": q4, "q5": q5, "q6_delta": q6_delta, "q8": q8},
                "integrity": {
                    "Dataset A modified": post["Dataset A modified"],
                    "Dataset B modified": post["Dataset B modified"],
                    "Normalization modified": post["Normalization modified"],
                    "Run1 modified": "NO" if run1_ok else "YES",
                    "Run2 modified": "NO" if run2_ok else "YES",
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    log("TARGETED TEMP RUN #3 COMPLETE — stopping (no Run #4)")


if __name__ == "__main__":
    main()
