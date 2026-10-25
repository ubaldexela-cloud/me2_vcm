# VCM benchmark - Xela Ubalde - 20261024-075005

**Hardware selection:** The selected and tested deployment device is **Raspberry Pi 5 Model B Rev 1.1**. The course example lists Raspberry Pi 4 or Raspberry Pi 5 as alternatives; this project selected Pi 5. **Not applicable — Raspberry Pi 5 was selected as the target hardware; Pi 4 was not used.** Metrics below are Pi 5 only and are not claimed to transfer to Pi 4.

Wake word: **Icy** - trials: 109 with the wake word + 16 without - shuffle seed: 82456 - connection: ssh - holdout: huggingface

Pi log file: ~/vcm_benchmark/benchmark_20261024-065959.log

Mic check: skipped (--skip-mic-check). No SNR was recorded, and the check is not treated as a pass.

## At a glance

|                                   | overall       | real voice   | synthetic voice |
|-----------------------------------|---------------|--------------|-----------------|
| intent accuracy (19)              | 40.4%         | 33.8%        | 50.0%           |
| command accuracy (93)             | 33.9%         | 29.2%        | 40.9%           |
| false accept (out of scope fired) | 62.5% (10/16) | 70.0% (7/10) | 50.0% (3/6)     |
| false reject (command ignored)    | 26.9%         | 21.8%        | 34.2%           |
| false wake (no wake word, fired)  | 12.5% (2/16)  | 0.0% (0/6)   | 20.0% (2/10)    |
| slot exact                        | 68.2%         | 66.7%        | 69.2%           |
| latency p95                       | 10.89 s       | 10.87 s      | 9.77 s          |

**Pi:** real-time factor 0.003 (p95 0.005), inference 13 ms, CPU temp max 55.1 C, runtime CPU 13% mean, runtime RAM 426 MB peak

**Check:**

- 3 extra fire(s): more than one command for one utterance

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 40.4%                | 33.9%                 |
| balanced accuracy                                 | 40.8%                | 33.4%                 |
| precision (macro)                                 | 34.7%                | 28.5%                 |
| recall (macro)                                    | 40.8%                | 33.4%                 |
| F1 (macro)                                        | 33.8%                | 29.3%                 |
| F2 (macro)                                        | 36.1%                | 30.6%                 |
| false accept rate (OOS fired)                     | 62.5%                | 62.5%                 |
| false reject rate (in-scope silent/rejected)      | 26.9%                | 26.9%                 |
| misfire rate (wrong command fired)                | 32.3%                | 39.8%                 |
| accuracy 95% CI                                   | [32-50%]             | [26-43%]              |
| false accept 95% CI                               | [39-82%] (10/16)     | [39-82%]              |
| false wake rate (command without wake word fired) | 12.5% [3-36%] (2/16) | 12.5% [3-36%] (2/16)  |

Responses: 90.8% of trials fired a command; no response: 10; extra fires: 3; wake detect rate: 92.7%

**False wakes (no wake word, Pi fired):** Change color to red -> LIGHT_ON; Tell me the weather -> WEATHER

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 109            | 65             | 44              |
| **19 intents** accuracy        | 40.4% [32-50%] | 33.8% [24-46%] | 50.0% [36-64%]  |
| balanced accuracy              | 40.8%          | 39.3%          | 47.9%           |
| F1 (macro)                     | 33.8%          | 30.9%          | 39.2%           |
| F2 (macro)                     | 36.1%          | 33.2%          | 42.8%           |
| false accept rate              | 62.5% (10/16)  | 70.0% (7/10)   | 50.0% (3/6)     |
| false reject rate              | 26.9%          | 21.8%          | 34.2%           |
| misfire rate                   | 32.3%          | 43.6%          | 15.8%           |
| **93 commands** accuracy       | 33.9%          | 29.2%          | 40.9%           |
| balanced accuracy              | 33.4%          | 29.1%          | 39.7%           |
| F1 (macro)                     | 29.3%          | 29.0%          | 35.1%           |
| F2 (macro)                     | 30.6%          | 29.1%          | 37.0%           |
| misfire rate                   | 39.8%          | 49.1%          | 26.3%           |
| slot exact (intent right)      | 68.2% (n=22)   | 66.7% (n=9)    | 69.2% (n=13)    |
| latency p50 / p95              | 6.16 / 10.89 s | 6.04 / 10.87 s | 6.27 / 9.77 s   |
| false wake rate (no wake word) | 12.5% (2/16)   | 0.0% (0/6)     | 20.0% (2/10)    |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent      | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM       | 8  | 75.0%  | 8.6 min        | 0.010          | 0.641         | 0.606     |
| TEMPERATURE | 7  | 28.6%  | 1.2 deg        | 0.156          | 0.509         | 0.527     |
| TIMER       | 7  | 100.0% | 0.0 s          | 0.000          | 0.200         | 0.264     |
| ALL         | 22 | 68.2%  | -              | 0.038          | 0.459         | 0.472     |

## Raspberry Pi

- **Raspberry Pi 5 Model B Rev 1.1**, 4 cores  up to 2400.0 MHz, RAM 4049.1 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5
- packages: numpy 2.2.4

| metric                                      | mean / p95 / max          |
|---------------------------------------------|---------------------------|
| response latency (command end -> Pi output) | 5.144 / 10.886 / 12.033 s |
| latency p50 / p99                           | 6.164 / 11.114 s          |
| inference time (Pi-reported)                | 12.8 / 19.5 / 22.0 ms     |
| real-time factor (infer / audio window)     | 0.003 / 0.005 / 0.005     |
| CPU temperature                             | 52.3 / 53.5 / 55.1 C      |
| CPU use, whole Pi                           | 3.6 / 7.7 / 29.8 %        |
| CPU use, your runtime process               | 13.1 / 29.7 / 109.6 %     |
| RAM (RSS), your runtime process             | 426.0 / 426.3 / 426.3 MB  |
| RAM used, whole Pi                          | 868.8 / 871.4 / 876.2 MB  |
| CPU clock                                   | 1766 / 2400 / 2400 MHz    |
| load average (1 min)                        | 0.18 / 0.35 / 0.50        |
| runtime CPU-seconds per second of speech    | 1.333                     |
| runtime CPU share of wall time              | 13.1%                     |
| throttling flags seen                       | none                      |
| test wall time                              | 34.8 min                  |

## Most frequent confusions

**intent level:** COLOR -> REJECT (7); BRIGHTNESS -> LIGHT_ON (5); CREATE_REMINDER -> LIGHT_ON (4); BRIGHTNESS -> REJECT (4); REJECT -> LIGHT_OFF (4); CREATE_REMINDER -> REJECT (3); REJECT -> LIGHT_ON (2); MESSAGE -> REJECT (2); NEXT -> REJECT (2); TIME -> VOLUME_UP (2)

**command level:** REJECT -> Lights out (4); REJECT -> Lights on (2); REJECT -> Weather (2); Reminder Study -> Lights on (1); Create a reminder to Exercise -> Volume down (1); Send my message -> REJECT (1); Change the temperature to 22 degrees -> Lights on (1); Brightness level 100 percent -> REJECT (1); Change color to Green -> REJECT (1); Reminders -> TIMER:? (1)

## Per-intent scores

| class           | n  | precision | recall | F1    | F2    |
|-----------------|----|-----------|--------|-------|-------|
| ALARM           | 9  | 80.0%     | 88.9%  | 84.2% | 87.0% |
| BRIGHTNESS      | 9  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| CALL            | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| COLOR           | 9  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| CREATE_REMINDER | 9  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| LIGHT_OFF       | 3  | 14.3%     | 33.3%  | 20.0% | 26.3% |
| LIGHT_ON        | 3  | 15.8%     | 100.0% | 27.3% | 48.4% |
| LIST_REMINDERS  | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| MESSAGE         | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| NEXT            | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| PAUSE           | 3  | 100.0%    | 66.7%  | 80.0% | 71.4% |
| PLAY_MUSIC      | 3  | 66.7%     | 66.7%  | 66.7% | 66.7% |
| REJECT          | 16 | 19.4%     | 37.5%  | 25.5% | 31.6% |
| STOP            | 3  | 100.0%    | 33.3%  | 50.0% | 38.5% |
| TEMPERATURE     | 9  | 100.0%    | 77.8%  | 87.5% | 81.4% |
| TIME            | 3  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| TIMER           | 9  | 77.8%     | 77.8%  | 77.8% | 77.8% |
| VOLUME_DOWN     | 3  | 37.5%     | 100.0% | 54.5% | 75.0% |
| VOLUME_UP       | 3  | 33.3%     | 66.7%  | 44.4% | 55.6% |
| WEATHER         | 3  | 50.0%     | 66.7%  | 57.1% | 62.5% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
