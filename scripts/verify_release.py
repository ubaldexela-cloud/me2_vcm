#!/usr/bin/env python3
"""Laptop-safe release check: hashes + optional ONNX dummy forward.

This does NOT evaluate locked-test accuracy. That requires VCM_MASTER audio,
CUDA, and artifacts/best_model.pt (see reports/REPRO_TEST.md).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = ROOT / "reports" / "CHECKSUMS.txt"

EXPECTED_ONNX_SHAPE = (1, 1, 40, 397)
EXPECTED_CLASSES = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        rows.append((digest, rel.replace("\\", "/")))
    return rows


def main() -> int:
    rows = load_manifest(CHECKSUMS)
    missing = 0
    mismatch = 0
    ok = 0
    for digest, rel in rows:
        path = ROOT / rel
        if not path.is_file():
            print(f"MISSING  {rel}")
            missing += 1
            continue
        got = sha256(path)
        if got != digest:
            print(f"MISMATCH {rel}")
            print(f"  expected {digest}")
            print(f"  got      {got}")
            mismatch += 1
        else:
            print(f"OK       {rel}")
            ok += 1

    onnx_path = ROOT / "export" / "model.onnx"
    onnx_status = "skipped"
    if onnx_path.is_file():
        try:
            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            dummy = np.zeros(EXPECTED_ONNX_SHAPE, dtype=np.float32)
            out = sess.run(None, {inp.name: dummy})[0]
            if tuple(out.shape) != (1, EXPECTED_CLASSES):
                print(f"ONNX_BAD_SHAPE {out.shape}")
                return 2
            print(f"ONNX_OK  input={inp.name} {inp.shape} output={out.shape}")
            onnx_status = "ok"
        except ImportError as exc:
            print(f"ONNX_SKIP {exc}")
            onnx_status = "skip_import"
        except Exception as exc:
            print(f"ONNX_FAIL {exc}")
            return 3

    summary = {
        "ok": ok,
        "missing": missing,
        "mismatch": mismatch,
        "onnx": onnx_status,
        "note": "Missing .pt files are expected in a Git tree; they are gitignored.",
    }
    print(json.dumps(summary))
    if mismatch:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
