# Dataset sources and redistribution limits

**This package contains no audio.** Do not upload VCM_MASTER, VCM_BALANCED, Hugging Face extracts, contributor recordings, or private full manifests to GitHub.

The repository MIT license covers **original code and documentation only**. It does not grant rights to any dataset below, to contributor recordings, or to the combined Safe Mix.

## Dataset location (copy-ready)

Safe Mix Run 1 is a **private 30,254-clip training mix**, not a public downloadable dataset. This repository contains **code and documentation**, not the audio.

Public source citations and access links are listed in this file. **Each source’s license and access restrictions apply independently.** There is **no single DOI** for the combined mix.

Recorded license conflicts, unknown source tags, and **missing contributor consent** prevent treating the full mix as cleared for redistribution. Reviewers can use the citations below to locate the public corpora; they cannot download Safe Mix from GitHub.

## Safe Mix Run 1 composition (30,254 clips)

Counts are from the published train index `manifests/safe_mix_train_index.csv` (`sha256`, `class`, `pool`, `source_dataset`, `source_type`, `original_or_augmented` only; **no filepaths, no speaker IDs**). They match the HPC report of the private training index. These are **index counts**, not redistributed audio.

| Source tag | Clips |
|---|---:|
| SLURP | 14,521 |
| Fluent Speech Commands | 11,593 |
| Multi-Sensor | 956 |
| Timers and Such | 873 |
| Google Speech Commands background noise | 632 |
| `group_synthetic` | 600 |
| `SET_TEMPERATURE_SYNTHETIC` | 479 |
| `real_voice` | 278 |
| SNIPS | 152 |
| `SET_TEMPERATURE_REAL` | 120 |
| Common Voice English | 36 |
| `SpeechCommands_v2` | 11 |
| `xela_SET_TEMPERATURE_REAL` | 3 |
| **Total** | **30,254** |

Pool split (unchanged): Dataset B unique 15,266 + leftover Dataset A train 13,329 + leak-safe on-disk HF clips 1,659. Locked val/test remain Dataset A only (`reports/mix_summary.json`).

**401 contributor-tagged clips** (278 `real_voice` + 120 `SET_TEMPERATURE_REAL` + 3 `xela_SET_TEMPERATURE_REAL`). The HPC report found **no consent documentation**. Do not publish personal identifiers or contributor audio.

Local course notes (not in Git): `VCM_MASTER/reports/dataset_sources_and_licenses.txt`. Access links below are **as recorded in project/HPC notes**. This update does **not** re-fetch official websites and does **not** claim a new license audit. Where a course note and an official page disagree, **do not redistribute**.

## Recorded access and license notes (not a clearance)

| Source | Role in Safe Mix | Recorded access | Recorded license / restriction | Redistribute audio? |
|---|---|---|---|---|
| Fluent Speech Commands (Lugosch et al.; Interspeech 2019) | Commands in A/B (play, lights, volume, pause/stop); 11,593 clips | Paper DOI [10.21437/Interspeech.2019-2396](https://doi.org/10.21437/Interspeech.2019-2396); https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/ | Recorded as academic / non-commercial. A public-license PDF is cited in prior notes; **it was not available on HPC for reinspection** and **full terms were not reverified** for this report. | **No** |
| SLURP (Bastianelli et al.; EMNLP 2020) | Weather, time, dim, some alarms, UNKNOWN; 14,521 clips | Paper DOI [10.18653/v1/2020.emnlp-main.588](https://doi.org/10.18653/v1/2020.emnlp-main.588); text https://github.com/pswietojanski/slurp ; audio DOI [10.5281/zenodo.4274930](https://doi.org/10.5281/zenodo.4274930) | Project notes: text **CC BY 4.0**; audio **CC BY-NC 4.0**. **Zenodo was not re-fetched** for this report. | **No** (audio) |
| Timers and Such v1.0 (Lugosch et al.) | SET_TIMER, some SET_ALARM; 873 clips | Paper https://arxiv.org/abs/2104.01604 ; dataset https://zenodo.org/records/4623772 | **Unresolved conflict:** project notes record **CC0**; Zenodo rights field recorded as **“Other (Open)”**. | **Do not treat as cleared** |
| Multi-Sensor Voice Command (Rusci, Van hamme, Tuytelaars) | MEDIA_NEXT; 956 clips | DOI [10.48804/IEKKVZ](https://doi.org/10.48804/IEKKVZ); https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/IEKKVZ | Course note: **CC-BY-4.0** plus GDPR download-tracking. RDR **Custom Dataset Terms** were **not captured**. Custom terms are **not cleared**. | **No** until current RDR terms are read |
| Google Speech Commands v0.02 (Warden 2018) | **Background-noise audio for the SILENCE class** (632 clips); keywords unused as VCM classes | Paper https://arxiv.org/abs/1804.03209 | Project notes: **CC BY 4.0**. License text was **not re-read** for this report. | Do not upload the mixed VCM tree |
| `airimonda/ai231-me2-voice-commands` | HF pool in Safe Mix (1,659 leak-safe train clips) | https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands ; DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723) | Card: **no single license**; research/education only; includes FSC, SLURP, SNIPS, group recordings, student `real_voice`. | **No** as a bundle |
| SNIPS (tag) | 152 clips | Corpus identity **unresolved** | Licensing **unresolved**. Do not assume FSC or CC-BY covers SNIPS. | **No** |
| Common Voice English (`CommonVoice_en`) | 36 clips | Tag in train index | Clip licenses **not independently verified** against this extract. | **No** as part of the mix |
| `SpeechCommands_v2` | 11 clips | Tag in train index | Clip licenses **not independently verified**. Distinct from the GSC **background-noise** SILENCE rows. | **No** as part of the mix |
| `group_synthetic` | 600 clips | Class-internal TTS / cloned voices | Applicable TTS/project terms **need clarification**; not a public redistribution grant. | **No** |
| `SET_TEMPERATURE_SYNTHETIC` | 479 clips | Project Piper TTS (train-only thermostat) | Project-specific; Piper/voice terms apply; **need clarification**. | **No** as a public corpus |
| Contributor tags | 401 clips (see composition) | Not a public corpus | Consent **required**. HPC report found **no consent documentation**. | **No** |

## Bibliography (recorded bibliographic fields)

Page numbers and DOIs are from previously recorded paper/dataset records, not a new website re-fetch for this note.

1. Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, Yoshua Bengio. “Speech Model Pre-Training for End-to-End Spoken Language Understanding.” *Interspeech 2019*, pp. 814–818. DOI: [10.21437/Interspeech.2019-2396](https://doi.org/10.21437/Interspeech.2019-2396). (Introduces Fluent Speech Commands.)
2. Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, Verena Rieser. “SLURP: A Spoken Language Understanding Resource Package.” *EMNLP 2020*, pp. 7252–7262. DOI: [10.18653/v1/2020.emnlp-main.588](https://doi.org/10.18653/v1/2020.emnlp-main.588). Audio dataset: DOI [10.5281/zenodo.4274930](https://doi.org/10.5281/zenodo.4274930).
3. Loren Lugosch, Piyush Papreja, Mirco Ravanelli, Abdelwahab Heba, Titouan Parcollet. “Timers and Such: A Practical Benchmark for Spoken Language Understanding with Numbers.” arXiv:2104.01604. Dataset: https://zenodo.org/records/4623772 .
4. Pete Warden. “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.” arXiv:1804.03209 (2018). Identified Safe Mix use: `_background_noise_` for SILENCE.
5. Manuele Rusci, Hugo Van hamme, Tinne Tuytelaars. “Multi-Sensor Voice Command Dataset.” DOI: [10.48804/IEKKVZ](https://doi.org/10.48804/IEKKVZ) (published 2024-10-22 on KU Leuven RDR).
6. `airimonda/ai231-me2-voice-commands`. Hugging Face dataset DOI: [10.57967/hf/10723](https://doi.org/10.57967/hf/10723).

Related Multi-Sensor paper listed on the RDR page (not independently re-read for this note): M. Rusci, H. Van Hamme and T. Tuytelaars, “Self-Incremental Training for Personalized Voice Command Recognition in a Wireless Audio Sensor Network,” ICASSP 2025, DOI [10.1109/ICASSP49660.2025.10887592](https://doi.org/10.1109/ICASSP49660.2025.10887592).

## Unresolved permissions (dataset item remains blocked)

- SNIPS corpus identity and licensing
- Multi-Sensor Custom Dataset Terms (text not captured)
- Timers and Such CC0 vs Zenodo “Other (Open)”
- `CommonVoice_en` and `SpeechCommands_v2` clip licenses not independently verified
- `group_synthetic` and `SET_TEMPERATURE_SYNTHETIC` TTS/project terms
- Contributor consent missing for 401 tagged clips
- Fluent Speech Commands full public-license PDF not reinspected on HPC
- SLURP Zenodo page not re-fetched for this report

Model weights (`best_model.pt`, `model.onnx`, `mel_norm.pt`) are **All Rights Reserved — no public license granted** and are **not publicly released** (`MODEL_LICENSE.md`). That no-grant notice does **not** override the third-party terms or missing-consent issues above.

## Split verification (no audio)

- Train speakers ∩ A val = 0; ∩ A test = 0
- Train SHA ∩ A val = 0; ∩ A test = 0
- Train unique SHA = 30,254 = clip count

See `reports/SPLIT_EVIDENCE.md`, `reports/mix_summary.json`, and `manifests/safe_mix_train_index.csv` (**no filepaths, no speaker IDs**). A GitHub-only clone cannot recompute speaker overlap from the public index.
