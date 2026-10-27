# Third-party provenance

Code in this repository is MIT-licensed original work except where noted. **Audio is not shipped.**

Canonical dataset table: `DATASET_LICENSES.md`.

| Asset | Terms (as documented) | Redistribute? |
|---|---|---|
| Fluent Speech Commands | Academic public license; reproduce but **not Share** licensed audio | **No** audio |
| SLURP audio (Zenodo) | **CC BY-NC 4.0** | **No** as a free corpus |
| SLURP text | CC BY 4.0 | Follow GitHub SLURP repo |
| Timers and Such | Paper/site CC0; Zenodo “Other (Open)” | Do not upload the mixed VCM tree |
| Multi-Sensor Voice Command | DOI **10.48804/IEKKVZ**, CC-BY-4.0 + GDPR download-tracking | Attribution + tracking |
| Google Speech Commands v0.02 noise | CC BY 4.0 | Attribution; keywords unused as classes |
| Course HF `airimonda/ai231-me2-voice-commands` | Card: no single license; DOI 10.57967/hf/10723 | **No** as a bundle |
| SNIPS | **Not verified** in this package | Do not assume a grant |
| Student real_voice / S1–S5 | Consent required | **No** without consent |
| Vosk small en-us | Alpha Cephei terms | Do not rehost if restricted |
| openWakeWord | Upstream license | Separate from ICY MIT |
| ONNX Runtime, Flask, NumPy, PyTorch | Their OSS licenses | |

Model weights (`best_model.pt`, `model.onnx`, `mel_norm.pt`): **All Rights Reserved — no public license granted**. They are **not publicly released**. The repository MIT license does not automatically license the weights, third-party datasets, or contributor recordings. All Rights Reserved does **not** override third-party rights or resolve missing contributor consent. Training mix includes FSC, SLURP NC audio, unresolved SNIPS/Multi-Sensor terms, and student/contributor clips without consent evidence in inspected directories (`MODEL_LICENSE.md`).
