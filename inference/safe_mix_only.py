#!/usr/bin/env python3
"""Experimental VCM inference: Safe Mix as the sole classifier.

Does not load Run 3 or Scratch. Does not average or gate their outputs.
Wake-word behavior is not handled here. Live Pi is not started.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = Path(__file__).resolve().parent / "experimental_config.json"

FORBIDDEN = (
    "targeted_temp_run3",
    "scratch_combined_run1",
    "pi_package_lightweight",
    "pi_package_scratch",
)


def load_experimental_config() -> dict:
    cfg = json.loads(CFG_PATH.read_text())
    if cfg.get("classifier") != "safe_mix_run1" or not cfg.get("sole_classifier"):
        raise SystemExit("STOP: experimental path is not Safe Mix only")
    if cfg.get("combine_with"):
        raise SystemExit("STOP: experimental path must not combine models")
    for key in ("checkpoint", "onnx", "package_dir"):
        path = str(cfg[key])
        if any(name in path for name in FORBIDDEN):
            raise SystemExit(f"STOP: experimental path points at a dual-model artifact: {path}")
    return cfg


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python inference/safe_mix_only.py sample.wav", file=sys.stderr)
        return 2
    cfg = load_experimental_config()
    pkg = Path(cfg["package_dir"])
    infer = pkg / "inference.py"
    if not infer.is_file():
        print(f"Safe Mix package not ready: {infer}", file=sys.stderr)
        print("Run export_safe_mix_run1.py on the GPU node first.", file=sys.stderr)
        return 1
    sys.path.insert(0, str(pkg))
    import inference as pkg_inference

    sys.argv = [str(infer), sys.argv[1]]
    return int(pkg_inference.main())


if __name__ == "__main__":
    raise SystemExit(main())
