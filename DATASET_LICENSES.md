# Dataset sources and redistribution limits

**This package contains no audio.** Do not upload VCM_MASTER, VCM_BALANCED, or Hugging Face extracts to GitHub.

Safe Mix train is a **manifest mix** of three pools (30,254 clips): Dataset B unique (15,266) + leftover Dataset A train (13,329) + leak-safe on-disk HF clips (1,659). Locked val/test are Dataset A only.

Local course notes: `VCM_MASTER/reports/dataset_sources_and_licenses.txt`. Official pages below were checked for this submission prep. Where the course note and the official page disagree, **do not redistribute**; follow the stricter official term.

## Components actually used

| Source | Role in Safe Mix | Official access | License / restriction | Redistribute audio? |
|---|---|---|---|---|
| Fluent Speech Commands (Lugosch et al., Interspeech 2019) | Commands in A/B (play, lights, volume, pause/stop) | https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/ ; license PDF https://fluent.ai/wp-content/uploads/2021/04/Fluent_Speech_Commands_Public_License.pdf | Fluent Speech Commands Public License: non-commercial academic use; **reproduce but not Share** the licensed material. Site text also cites CC BY-NC-ND 4.0. | **No** |
| SLURP (Bastianelli et al., EMNLP 2020) | Weather, time, dim, some alarms, UNKNOWN | Text: https://github.com/pswietojanski/slurp ; audio: https://doi.org/10.5281/zenodo.4274930 | Text CC BY 4.0. **Audio CC BY-NC 4.0**. A class HF card incorrectly lists SLURP as CC BY 4.0; the Zenodo audio record is CC BY-NC 4.0. | **No** (audio) |
| Timers and Such v1.0 (Lugosch et al.) | SET_TIMER, some SET_ALARM | https://zenodo.org/records/4623772 ; paper arXiv:2104.01604 | Paper/site: **CC0**. Zenodo rights field is “Other (Open)”. | Allowed under CC0; still do not upload the mixed VCM tree |
| Multi-Sensor Voice Command | MEDIA_NEXT | https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/IEKKVZ DOI **10.48804/IEKKVZ** | **CC-BY-4.0** plus GDPR obligation to track downloads of derivatives (course license note) | Attribution + download-tracking; do not upload without that process |
| Google Speech Commands v0.02 (Warden 2018) | SILENCE from `_background_noise_` only; also train-time noise files | https://arxiv.org/abs/1804.03209 ; TF download `speech_commands_v0.02.tar.gz` | **CC BY 4.0** | Allowed with attribution; keywords were not used as VCM classes |
| Project SET_TEMPERATURE real (S1–S5) | Thermostat commands | Not a public corpus | Course recordings; retain consent separately | **No** without each speaker’s consent |
| Project SET_TEMPERATURE synthetic (Piper TTS) | Train-only thermostat | Generated for the course | Project-specific; Piper/voice terms apply | Do not treat as a public corpus |
| `airimonda/ai231-me2-voice-commands` (1,659 leak-safe train clips) | HF pool in Safe Mix | https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands DOI 10.57967/hf/10723 | Card: **no single license**; research/education only; includes FSC, SLURP, SNIPS, group recordings, student `real_voice`. Card: “do not share [group recordings] outside the class without each speaker's consent.” | **No** as a bundle |

## Other HF tags present in the 1,659-clip pool

From the Safe Mix `hf_usable` rows (no audio shipped here): SLURP, `group_synthetic`, `real_voice`, SNIPS, FluentSpeechCommands, CommonVoice_en, SpeechCommands_v2, `xela_SET_TEMPERATURE_REAL`.

- Common Voice (en): typically **CC0** (card and Mozilla terms).
- SNIPS SLU: official terms were **not** on disk in this HPC tree; consult the SNIPS / Sonos release. Do not assume FSC or CC-BY covers SNIPS.
- `group_synthetic`: class-internal TTS / cloned voices; not a public redistribution grant.

## Split verification (no audio)

- Train speakers ∩ A val = 0; ∩ A test = 0
- Train SHA ∩ A val = 0; ∩ A test = 0
- Train unique SHA = 30,254 = clip count

See `reports/mix_summary.json` and `manifests/safe_mix_train_index.csv` (SHA + class + pool only; **no filepaths, no speaker IDs**).
