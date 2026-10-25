# Raspberry Pi dashboard source

Copies of the ICY_final dashboard used on Raspberry Pi **5** (Debian 13). This is not a GPIO appliance tree.

| File | Role |
|---|---|
| `icy_dashboard_ICY_final.py` | Flask UI + wake + ONNX + Vosk slots |
| `icy_final_slots.py` | Parameter parse after PARAMETER intents |
| `command_endpoint.py` | RMS endpointing; still pads to 64,000 samples |

Place next to `pi_package_ICY_final/` (from `export/`) and `wakeword/icy_outputs/icy.onnx`. Install Vosk separately.

Normal start (on the Pi, after you copy files there yourself):

```bash
cd ~/me2_vcm
unset VCM_BENCHMARK
~/.venv or /home/<user>/ME2_VCM/.venv/bin/python -u icy_dashboard_ICY_final.py
```

Dashboard: port **8765**. Do not commit Wi-Fi, SSH, or mic recordings.

`inference/command_endpoint.py` is the HPC copy of the same helper.
