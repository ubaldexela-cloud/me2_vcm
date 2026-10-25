# Dataset sources and redistribution limits

**This package contains no audio.** Do not upload VCM_MASTER, VCM_BALANCED, or Hugging Face extracts to GitHub.

The repository MIT license covers **original code and documentation only**. It does not grant rights to any dataset below.

Safe Mix train is a **manifest mix** of three pools (30,254 clips): Dataset B unique (15,266) + leftover Dataset A train (13,329) + leak-safe on-disk HF clips (1,659). Locked val/test are Dataset A only.

Local course notes: `VCM_MASTER/reports/dataset_sources_and_licenses.txt`. Official pages below were checked for this submission prep. Where the course note and the official page disagree, **do not redistribute**; follow the stricter official term.

## Components actually used

| Source | Role in Safe Mix | Official access | License / restriction | Redistribute audio? |
|---|---|---|---|---|
| Fluent Speech Commands (Lugosch, Ravanelli, Ignoto, Tomar, Bengio; Interspeech 2019) | Commands in A/B (play, lights, volume, pause/stop) | Paper DOI [10.21437/Interspeech.2019-2396](https://doi.org/10.21437/Interspeech.2019-2396); site https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/ ; license PDF https://fluent.ai/wp-content/uploads/2021/04/Fluent_Speech_Commands_Public_License.pdf | Fluent Speech Commands Public License: non-commercial academic use; **reproduce but not Share** the licensed material. Site text also cites CC BY-NC-ND 4.0. | **No** |
| SLURP (Bastianelli, Vanzo, Swietojanski, Rieser; EMNLP 2020) | Weather, time, dim, some alarms, UNKNOWN | Paper DOI [10.18653/v1/2020.emnlp-main.588](https://doi.org/10.18653/v1/2020.emnlp-main.588); text https://github.com/pswietojanski/slurp ; audio DOI [10.5281/zenodo.4274930](https://doi.org/10.5281/zenodo.4274930) | Text CC BY 4.0. **Audio CC BY-NC 4.0** (Zenodo). A class HF card incorrectly lists SLURP as CC BY 4.0. | **No** (audio) |
| Timers and Such v1.0 (Lugosch, Papreja, Ravanelli, Heba, Parcollet) | SET_TIMER, some SET_ALARM | Paper arXiv:2104.01604; dataset https://zenodo.org/records/4623772 | Paper/site: **CC0**. Zenodo rights field is “Other (Open)”. | Allowed under the paper’s CC0 statement; still do not upload the mixed VCM tree |
| Multi-Sensor Voice Command (Rusci, Van hamme, Tuytelaars) | MEDIA_NEXT | DOI **10.48804/IEKKVZ** https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/IEKKVZ | Course note: **CC-BY-4.0** plus GDPR obligation to track downloads of derivatives. KU Leuven RDR landing page (checked 2026-10-25) also labels **Custom Dataset Terms**. Full custom-terms text was not captured here. | Do not upload without reading the current RDR terms and completing any required tracking |
| Google Speech Commands v0.02 (Warden 2018) | SILENCE from `_background_noise_` only; also train-time noise files | Paper arXiv:1804.03209 (title: *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*) | Paper states **CC BY 4.0** | Allowed with attribution; keywords were not used as VCM classes |
| Project SET_TEMPERATURE real (S1–S5) | Thermostat commands | Not a public corpus | Course recordings. **Consent is required.** Consent records were **not inspected** in the 2026-10-25 audit; this file does **not** claim that consent exists or is on file. | **No** unless each speaker’s consent is separately confirmed |
| Project SET_TEMPERATURE synthetic (Piper TTS) | Train-only thermostat | Generated for the course | Project-specific; Piper/voice terms apply | Do not treat as a public corpus |
| `airimonda/ai231-me2-voice-commands` (1,659 leak-safe train clips) | HF pool in Safe Mix | https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723) | Card: **no single license**; research/education only; includes FSC, SLURP, SNIPS, group recordings, student `real_voice`. Card: “do not share [group recordings] outside the class without each speaker's consent.” | **No** as a bundle |

## Bibliography (verified bibliographic fields)

Use these when citing sources. Page numbers and DOIs are from the official paper/dataset records checked for this note.

1. Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, Yoshua Bengio. “Speech Model Pre-Training for End-to-End Spoken Language Understanding.” *Interspeech 2019*, pp. 814–818. DOI: [10.21437/Interspeech.2019-2396](https://doi.org/10.21437/Interspeech.2019-2396). (Introduces Fluent Speech Commands.)
2. Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, Verena Rieser. “SLURP: A Spoken Language Understanding Resource Package.” *EMNLP 2020*, pp. 7252–7262. DOI: [10.18653/v1/2020.emnlp-main.588](https://doi.org/10.18653/v1/2020.emnlp-main.588). Audio dataset: DOI [10.5281/zenodo.4274930](https://doi.org/10.5281/zenodo.4274930).
3. Loren Lugosch, Piyush Papreja, Mirco Ravanelli, Abdelwahab Heba, Titouan Parcollet. “Timers and Such: A Practical Benchmark for Spoken Language Understanding with Numbers.” arXiv:2104.01604. Dataset: Zenodo record 4623772.
4. Pete Warden. “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.” arXiv:1804.03209 (2018).
5. Manuele Rusci, Hugo Van hamme, Tinne Tuytelaars. “Multi-Sensor Voice Command Dataset.” DOI: [10.48804/IEKKVZ](https://doi.org/10.48804/IEKKVZ) (published 2024-10-22 on KU Leuven RDR). Course note authors match this landing page.
6. `airimonda/ai231-me2-voice-commands`. Hugging Face dataset DOI: [10.57967/hf/10723](https://doi.org/10.57967/hf/10723).

Related Multi-Sensor paper listed on the RDR page (not independently re-read for this note): M. Rusci, H. Van Hamme and T. Tuytelaars, “Self-Incremental Training for Personalized Voice Command Recognition in a Wireless Audio Sensor Network,” ICASSP 2025, DOI [10.1109/ICASSP49660.2025.10887592](https://doi.org/10.1109/ICASSP49660.2025.10887592).

## Other HF tags present in the 1,659-clip pool

From the Safe Mix `hf_usable` rows (no audio shipped here): SLURP, `group_synthetic`, `real_voice`, SNIPS, FluentSpeechCommands, CommonVoice_en, SpeechCommands_v2, `xela_SET_TEMPERATURE_REAL`.

- Common Voice (en): Mozilla’s corpus is commonly released as **CC0**. The exact license of the specific `CommonVoice_en` rows in this HF extract was **not re-verified clip-by-clip**.
- SNIPS: official terms for the **clips actually used** remain **unverified**. A candidate Sonos/Snips audio license exists (`sonos/spoken-language-understanding-research-datasets`: academic/research only, no commercial use), but this package cannot confirm that the HF `SNIPS` tags are that corpus (a different Snips `nlu-benchmark` repo is CC0 text). Do not assume FSC or CC-BY covers SNIPS.
- `group_synthetic`: class-internal TTS / cloned voices; not a public redistribution grant.
- `real_voice` / `xela_SET_TEMPERATURE_REAL`: student or project recordings. Consent is required. Consent records were **not inspected**; this file does not claim consent exists.

## Split verification (no audio)

- Train speakers ∩ A val = 0; ∩ A test = 0
- Train SHA ∩ A val = 0; ∩ A test = 0
- Train unique SHA = 30,254 = clip count

See `reports/SPLIT_EVIDENCE.md`, `reports/mix_summary.json`, and `manifests/safe_mix_train_index.csv` (SHA + class + pool only; **no filepaths, no speaker IDs**). A GitHub-only clone cannot recompute speaker overlap from the public index.
