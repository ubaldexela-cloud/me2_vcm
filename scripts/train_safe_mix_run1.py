#!/usr/bin/env python3
"""From-scratch retrain on leak-safe mix (B unique + leftover A-train + on-disk HF).

Run 3 recipe, random init. Writes only to checkpoints/results/safe_mix_run1.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/tmp/me2_pydeps")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.augment import WaveformAugmentor, discover_noise_files
from src.dataset import VCMDataset, collate_batch
from src.metrics import classification_report_text
from src.model import VCMDSTiny, count_parameters
from src.normalization import MelNormalizer
from src.temp_augment import SetTemperatureSpeakerAugmentor
from src.utils import ID_TO_CLASS, load_config, set_seed
from train_targeted_temp_run3 import (
    audit_set_temperature,
    build_sample_weights,
    estimate_macs,
    evaluate,
    precompute_eval,
    save_cm_csv,
    train_one_epoch,
    write_metrics_csv,
)

TEMP_ID = 8
LOCKED_DIRS = ("targeted_temp_run3", "scratch_combined_run1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(lines, msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    lines.append(line)


def read_metrics_csv(path: Path):
    metrics = {}
    with open(path, newline="") as f:
        for rec in csv.reader(f):
            if len(rec) >= 5 and rec[0].isdigit():
                metrics[rec[1]] = float(rec[4])
            elif len(rec) == 2 and rec[0] in ("accuracy", "macro_f1", "weighted_f1"):
                metrics[rec[0]] = float(rec[1])
    return metrics


def pack_metrics(res):
    out = {
        "accuracy": float(res["macro"]["accuracy"]),
        "macro_f1": float(res["macro"]["macro_f1"]),
        "weighted_f1": float(res["weighted"]["weighted_f1"]),
    }
    for i in range(16):
        out[ID_TO_CLASS[i]] = float(res["prf"]["f1"][i])
    return out


def load_split_keys(manifest: Path):
    speakers, shas = set(), set()
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            speakers.add(r.get("speaker_id") or r.get("speaker") or "")
            shas.add(r.get("sha256") or "")
    speakers.discard("")
    shas.discard("")
    return speakers, shas


def snapshot_locked(cfg):
    a = Path(cfg["a_root"])
    b = Path(cfg["balanced_root"])
    return {
        "A_train": sha256_file(a / "manifests/train.csv"),
        "A_val": sha256_file(a / "manifests/val.csv"),
        "A_test": sha256_file(a / "manifests/test.csv"),
        "B_train": sha256_file(b / "manifests/train.csv"),
        "norm": sha256_file(Path(cfg["norm_path"])),
        "run3": sha256_file(Path(cfg["run3_checkpoint"])),
        "scratch": sha256_file(Path(cfg["scratch_checkpoint"])),
    }


def main():
    cfg = load_config(ROOT / "configs" / "safe_mix_run1.yaml")
    set_seed(int(cfg["seed"]))
    ckpt_dir = Path(cfg["checkpoint_dir"])
    res_dir = Path(cfg["results_dir"])
    for guard in (ckpt_dir, res_dir):
        if any(name in str(guard) for name in LOCKED_DIRS):
            raise SystemExit("STOP: refusing to write locked checkpoint/results dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "configs" / "safe_mix_run1.yaml", ckpt_dir / "training_config.yaml")

    lines = []
    locked0 = snapshot_locked(cfg)
    log(lines, f"locked_sha={json.dumps(locked0)}")

    rows = list(csv.DictReader(open(cfg["train_manifest"], newline="")))
    pools = Counter(r["pool"] for r in rows)
    log(lines, f"mix n={len(rows)} pools={dict(pools)}")
    if len(rows) != int(cfg["expected_train"]):
        raise SystemExit(f"STOP: train count {len(rows)}")
    if pools.get("vcm_balanced") != int(cfg["expected_balanced"]):
        raise SystemExit("STOP: balanced count")
    if pools.get("a_train_leftover") != int(cfg["expected_leftover"]):
        raise SystemExit("STOP: leftover count")
    if pools.get("hf_usable") != int(cfg["expected_hf"]):
        raise SystemExit("STOP: hf count")
    if len({r["sha256"] for r in rows}) != len(rows):
        raise SystemExit("STOP: duplicate SHA in mix")

    val_spk, val_sha = load_split_keys(Path(cfg["val_manifest"]))
    test_spk, test_sha = load_split_keys(Path(cfg["test_manifest"]))
    mix_spk = {r["speaker_id"] for r in rows if r.get("speaker_id")}
    mix_sha = {r["sha256"] for r in rows if r.get("sha256")}
    leak = {
        "speaker_overlap_val": len(mix_spk & val_spk),
        "speaker_overlap_test": len(mix_spk & test_spk),
        "sha_overlap_val": len(mix_sha & val_sha),
        "sha_overlap_test": len(mix_sha & test_sha),
    }
    log(lines, f"leak_check={leak}")
    if any(leak.values()):
        raise SystemExit("STOP: mix leaks into locked val/test")

    if not torch.cuda.is_available():
        raise SystemExit("STOP: CUDA required")
    device = torch.device("cuda")
    log(lines, f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    noise_files = discover_noise_files(cfg["augmentation"]["noise_dir"])
    if len(noise_files) < 6:
        raise SystemExit(f"STOP: noise files missing ({len(noise_files)})")
    st = cfg["set_temperature"]
    augmentor = WaveformAugmentor(
        sample_rate=16000,
        noise_paths=noise_files,
        gain_p=cfg["augmentation"]["gain_p"],
        gain_min=cfg["augmentation"]["gain_min"],
        gain_max=cfg["augmentation"]["gain_max"],
        shift_p=cfg["augmentation"]["shift_p"],
        shift_ms=cfg["augmentation"]["shift_ms"],
        noise_p=cfg["augmentation"]["noise_p"],
        snr_bands=[tuple(x) for x in cfg["augmentation"]["snr_bands"]],
        seed=int(cfg["seed"]),
    )
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
    normalizer = MelNormalizer.from_file(cfg["norm_path"])
    log(lines, "loading train (cache_waveforms=True)...")
    train_ds = VCMDataset(
        cfg["train_manifest"],
        cfg["train_dataset"],
        cfg,
        "train",
        normalizer,
        cache_waveforms=True,
        augmentor=augmentor,
        temp_augmentor=temp_aug,
    )
    composition = audit_set_temperature(train_ds)
    log(lines, f"SET_TEMPERATURE composition={composition}")
    weights, sampler_stats = build_sample_weights(train_ds, cfg)
    (res_dir / "sampler_config.json").write_text(json.dumps(sampler_stats, indent=2) + "\n")
    (res_dir / "set_temperature_composition.json").write_text(json.dumps(composition, indent=2) + "\n")
    log(
        lines,
        f"sampler expected_temp/epoch={sampler_stats['expected_temp_draws_per_epoch']:.1f} "
        f"real_pct_of_temp={sampler_stats['pct_temp_exposure_from_real']:.1f}%",
    )

    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True, generator=g)

    log(lines, "loading val (no augment)...")
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
        raise SystemExit(f"STOP: architecture mismatch {pstats['trainable']} {macs}")
    model.to(device)
    log(lines, f"params={pstats['trainable']} macs={macs} init=random")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg["scheduler_factor"]),
        patience=int(cfg["scheduler_patience"]),
        min_lr=float(cfg["min_lr"]),
    )

    history = []
    best_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    stopping = "reached_max_epochs"
    t0 = time.time()
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        tr_loss, tr_acc, temp_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = evaluate(model, val_loader, criterion, device)
        val_f1 = val_res["macro"]["macro_f1"]
        lr_now = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_f1)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_accuracy": tr_acc,
            "val_macro_f1": val_f1,
            "val_accuracy": val_res["macro"]["accuracy"],
            "val_temp_f1": float(val_res["prf"]["f1"][TEMP_ID]),
            "val_unknown_f1": float(val_res["prf"]["f1"][14]),
            "val_silence_f1": float(val_res["prf"]["f1"][15]),
            "learning_rate": lr_now,
            "temp_seen": temp_stats["temp_seen"],
            "real_temp_seen": temp_stats["real_temp_seen"],
        }
        history.append(row)
        log(
            lines,
            f"epoch={epoch} train_loss={tr_loss:.4f} val_macro_f1={val_f1:.4f} "
            f"temp={row['val_temp_f1']:.4f} unk={row['val_unknown_f1']:.4f} "
            f"sil={row['val_silence_f1']:.4f} lr={lr_now:.2e}",
        )
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "val_macro_f1": val_f1, "source": "last"},
            ckpt_dir / "last_model.pt",
        )
        if val_f1 > best_f1 + 1e-12:
            best_f1, best_epoch, no_improve = val_f1, epoch, 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_macro_f1": val_f1, "source": "best"},
                ckpt_dir / "best_model.pt",
            )
            log(lines, f"  NEW BEST {best_f1:.6f}")
        else:
            no_improve += 1
            log(lines, f"  no_improve={no_improve}/{cfg['early_stopping_patience']}")
            if no_improve >= int(cfg["early_stopping_patience"]):
                stopping = f"early_stopping_patience_{cfg['early_stopping_patience']}"
                log(lines, stopping)
                break
        (res_dir / "training_log.txt").write_text("\n".join(lines) + "\n")

    train_time = time.time() - t0
    with (res_dir / "training_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    log(lines, "ONE-TIME locked test eval...")
    test_raw = VCMDataset(cfg["test_manifest"], cfg["test_dataset"], cfg, "eval", normalizer, cache_waveforms=True)
    test_ds = precompute_eval(test_raw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    best_blob = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_blob["model_state_dict"])
    test_res = evaluate(model, test_loader, criterion, device)
    write_metrics_csv(test_res, res_dir / "best_test_metrics.csv")
    save_cm_csv(test_res["cm"], res_dir / "best_test_confusion_matrix.csv")
    (res_dir / "test_classification_report.txt").write_text(classification_report_text(test_res["cm"]))

    got = pack_metrics(test_res)
    run3_metrics = read_metrics_csv(Path(cfg["run3_results"]) / "test_metrics.csv")
    names = [ID_TO_CLASS[i] for i in range(16)]
    cmp = ["metric,run3,safe_mix,delta_vs_run3"]
    for key in ["accuracy", "macro_f1", "weighted_f1"] + names:
        cmp.append(f"{key},{run3_metrics[key]:.6f},{got[key]:.6f},{got[key] - run3_metrics[key]:+.6f}")
    (res_dir / "RUN3_VS_SAFE_MIX.txt").write_text("\n".join(cmp) + "\n")

    locked1 = snapshot_locked(cfg)
    if locked0 != locked1:
        raise SystemExit(f"STOP: locked file changed {locked0} -> {locked1}")

    summary = {
        "n_train": len(rows),
        "pools": dict(pools),
        "leak_check": leak,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_accuracy": got["accuracy"],
        "test_macro_f1": got["macro_f1"],
        "stopping": stopping,
        "train_time_sec": train_time,
        "init": "random",
        "device": torch.cuda.get_device_name(0),
        "checkpoint_dir": str(ckpt_dir),
        "locked_unmodified": True,
    }
    (res_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(
        lines,
        f"TEST acc={got['accuracy']:.4f} macro_f1={got['macro_f1']:.4f} "
        f"best_epoch={best_epoch} train_time_sec={train_time:.1f}",
    )
    log(lines, "SAFE_MIX_RUN1_DONE")
    (res_dir / "training_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
