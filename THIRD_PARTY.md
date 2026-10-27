# Third-party provenance

Code in this repository is MIT-licensed original work except where noted. **Audio is not shipped.**

Canonical dataset location, source-count table, citations, and unresolved permissions: `DATASET_LICENSES.md`. Safe Mix is a private 30,254-clip mix; audio is not shipped.

| Asset | Terms (as documented) | Redistribute? |
|---|---|---|
| Fluent Speech Commands | Academic public license; reproduce but **not Share** licensed audio | **No** audio |
| SLURP audio (Zenodo) | **CC BY-NC 4.0** | **No** as a free corpus |
| SLURP text | CC BY 4.0 | Follow GitHub SLURP repo |
| Timers and Such | Paper/site CC0 vs Zenodo “Other (Open)” — **unresolved** | Do not treat as cleared |
| Multi-Sensor Voice Command | DOI **10.48804/IEKKVZ**; course note CC-BY-4.0 + tracking; Custom Dataset Terms **not captured** | **No** until RDR terms are read |
| Google Speech Commands v0.02 noise | Project notes CC BY 4.0 (text not re-read here); SILENCE background noise only | Do not upload the mixed VCM tree |
| Course HF `airimonda/ai231-me2-voice-commands` | Card: no single license; DOI 10.57967/hf/10723 | **No** as a bundle |
| SNIPS | Corpus identity and licensing **unresolved** | Do not assume a grant |
| Contributor tags (401 clips: 278 `real_voice`, 120 `SET_TEMPERATURE_REAL`, 3 `xela_SET_TEMPERATURE_REAL`) | Consent required; HPC report found **no consent documentation** | **No** |
| Vosk small en-us | Alpha Cephei terms | Do not rehost if restricted |
| openWakeWord | Upstream license | Separate from ICY MIT |
| ONNX Runtime, Flask, NumPy, PyTorch | Their OSS licenses | |

Model weights (`best_model.pt`, `model.onnx`, `mel_norm.pt`): **All Rights Reserved — no public license granted**. They are **not publicly released**. The repository MIT license does not automatically license the weights, third-party datasets, or contributor recordings. All Rights Reserved does **not** override third-party rights or resolve missing contributor consent. Training mix includes FSC, SLURP NC audio, unresolved SNIPS/Multi-Sensor terms, and student/contributor clips without consent evidence in inspected directories (`MODEL_LICENSE.md`).
