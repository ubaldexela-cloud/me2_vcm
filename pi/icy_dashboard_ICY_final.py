#!/usr/bin/env python3
"""ICY_final dashboard.

Safe Mix with endpointing is the intent classifier. Vosk reads only the
parameter of a timer, alarm, temperature, or brightness command. This file
does not replace icy_dashboard.py or icy_dashboard_safe_mix_experimental.py,
and importing it does not start the assistant.
"""
import importlib.util
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import urllib.request
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import onnxruntime as ort
from flask import Flask, jsonify, render_template_string, request
from openwakeword.model import Model

HOME = Path.home()
ROOT = HOME / "me2_vcm"
WAKE_MODEL = ROOT / "wakeword/icy_outputs/icy.onnx"
PACKAGE = ROOT / "pi_package_ICY_final"
VOSK_MODEL = ROOT / "vosk-model-small-en-us-0.15"
VOSK_RECOGNIZER = None
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from icy_final_slots import extract_parameter
READY_WAV = ROOT / "command_ready.wav"
ACK_WAV = ROOT / "yess.wav"
ACK_AUDIO = None
RETRY_WAV = ROOT / "try_again.wav"
FAILED_WAV = ROOT / "not_understood.wav"
PIPER_PYTHON = HOME / "me2_vcm/piper_env/bin/python"
PIPER_VOICE = HOME / "me2_vcm/piper_voices/en_US-amy-medium.onnx"
MIC = "plughw:2,0"
SAMPLE_RATE = 16000
COMMAND_SECONDS = 4
WAKE_THRESHOLD = 0.10
MAX_ATTEMPTS = 3
WAKE_COOLDOWN = 1.0
INVALID_CLASSES = {"UNKNOWN", "SILENCE"}
ACTION_CLASSES = {
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "LIGHT_DIM", "SET_TIMER", "SET_ALARM", "SET_TEMPERATURE",
    "MEDIA_PAUSE", "MEDIA_STOP", "MEDIA_NEXT", "VOLUME_UP", "VOLUME_DOWN",
}
# These intents can be recognized, but their values are not guessed.
# The dashboard controls are the input method until a non-ASR slot path exists.
PARAMETER_COMMANDS = {
    "LIGHT_DIM": "I recognized dim the light. Set the brightness percentage on the dashboard.",
    "SET_TIMER": "I recognized set a timer. Enter the duration on the dashboard.",
    "SET_ALARM": "I recognized set an alarm. Enter the clock time on the dashboard.",
    "SET_TEMPERATURE": "I recognized set the temperature. Enter a value from 0 to 50 on the dashboard.",
}
COMMAND_TEXT = {
    "PLAY_MUSIC": "play music", "WEATHER": "check the weather",
    "TIME": "tell the time", "LIGHT_ON": "turn the light on",
    "LIGHT_OFF": "turn the light off", "LIGHT_DIM": "dim the light",
    "SET_TIMER": "set a timer", "SET_ALARM": "set an alarm",
    "SET_TEMPERATURE": "set the temperature", "MEDIA_PAUSE": "pause the media",
    "MEDIA_STOP": "stop the media", "MEDIA_NEXT": "play the next track",
    "VOLUME_UP": "turn the volume up", "VOLUME_DOWN": "turn the volume down",
}
# Names the course harness already understands. Slot values are never invented.
BENCHMARK_INTENT = {
    "PLAY_MUSIC": "PLAY_MUSIC",
    "WEATHER": "WEATHER",
    "TIME": "TIME",
    "LIGHT_ON": "LIGHT_ON",
    "LIGHT_OFF": "LIGHT_OFF",
    "VOLUME_UP": "VOLUME_UP",
    "VOLUME_DOWN": "VOLUME_DOWN",
    "MEDIA_PAUSE": "PAUSE",
    "MEDIA_STOP": "STOP",
    "MEDIA_NEXT": "NEXT",
    "LIGHT_DIM": "BRIGHTNESS",
    "SET_TIMER": "TIMER",
    "SET_ALARM": "ALARM",
    "SET_TEMPERATURE": "TEMPERATURE",
}
BENCHMARK_MODE = os.environ.get("VCM_BENCHMARK", "").strip().lower() in {"1", "true", "yes", "on"}
BENCHMARK_LOG = None
app = Flask(__name__)
STATE_LOCK = threading.Lock()
STATE = {
    "status": "Starting assistant…", "wake_status": "Starting",
    "model_a": {"label": "—", "confidence": 0.0},
    "model_b": {"label": "—", "confidence": 0.0},
    "decision": "Waiting", "last_command": "—", "action_status": "No action yet",
    "history": [], "running": True, "updated": time.time(),
}
ACTION_QUEUE = queue.Queue()

# Software-only dashboard device state; no extra hardware is required.
DEVICE_STATE = {
    "light_on": False,
    "brightness": 50,
    "temperature": 22,
    "media": "stopped",
    "track": 1,
    "weather": "Weather source not configured",
    "timers": [],
    "alarms": [],
}
DEVICE_LOCK = threading.Lock()

# Actual music playback. Track numbers are 1-based.
MEDIA_DIR = ROOT / "media" / "ME2_Music"
MEDIA_FILES = sorted(MEDIA_DIR.glob("*.mp3"))
MEDIA_PROCESS = None


def stop_media_process():
    """Stop the current music process, if one is running."""
    global MEDIA_PROCESS
    proc = MEDIA_PROCESS
    MEDIA_PROCESS = None
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def start_media_track(index):
    """Start the selected MP3 through the default PipeWire output."""
    global MEDIA_PROCESS

    if not MEDIA_FILES:
        return False, f"No MP3 files found in {MEDIA_DIR}"

    index %= len(MEDIA_FILES)
    stop_media_process()

    try:
        MEDIA_PROCESS = subprocess.Popen(
            [
                "ffplay", "-nodisp", "-autoexit",
                "-loglevel", "error", "-nostats",
                str(MEDIA_FILES[index]),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        MEDIA_PROCESS = None
        return False, f"Could not start music playback: {exc}"

    DEVICE_STATE["track"] = index + 1
    DEVICE_STATE["media"] = "playing"
    return True, f"Playing track {index + 1}: {MEDIA_FILES[index].name}"



def update_state(**kwargs):
    with STATE_LOCK:
        STATE.update(kwargs)
        STATE["updated"] = time.time()

def add_history(message, kind="info"):
    if "timer" in str(message).lower():
        return
    with STATE_LOCK:
        STATE["history"].insert(0, {"time": datetime.now().strftime("%H:%M:%S"),
                                    "message": str(message), "kind": kind})
        STATE["history"] = STATE["history"][:30]
        STATE["updated"] = time.time()

def add_prediction_history(label):
    """Show the predicted class only. This bypasses the timer-text history filter."""
    message = f"Model predicted: {label}"
    kind = "success" if command_is_actionable(label) else "warning"
    with STATE_LOCK:
        STATE["history"].insert(0, {"time": datetime.now().strftime("%H:%M:%S"),
                                    "message": message, "kind": kind})
        STATE["history"] = STATE["history"][:30]
        STATE["updated"] = time.time()
    return message

def require_files():
    paths = [WAKE_MODEL, PACKAGE / "inference.py", PACKAGE / "model.onnx",
             PACKAGE / "mean.npy", PACKAGE / "std.npy", PACKAGE / "class_map.json",
             PACKAGE / "feature_config.json",
             READY_WAV, RETRY_WAV, FAILED_WAV, ACK_WAV,
             PIPER_PYTHON, PIPER_VOICE]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files/dependencies:\n" + "\n".join(missing))

def load_frontend():
    """Load Safe Mix preprocessing and its own normalization vectors."""
    path = PACKAGE / "inference.py"
    spec = importlib.util.spec_from_file_location("icy_safe_mix_frontend", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, np.load(PACKAGE / "mean.npy"), np.load(PACKAGE / "std.npy")

def load_model_contract():
    """Confirm the package expects 16 kHz, four seconds, and 16 classes."""
    with open(PACKAGE / "feature_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    shape = tuple(int(x) for x in cfg["model_input_shape"])
    n_classes = int(cfg["num_classes"])
    if int(cfg["sample_rate"]) != SAMPLE_RATE:
        raise ValueError("classifier sample rate is not 16000 Hz")
    if int(cfg["num_samples"]) != SAMPLE_RATE * COMMAND_SECONDS:
        raise ValueError("classifier input is not a four-second clip")
    if shape != (1, 1, 40, 397) or n_classes != 16:
        raise ValueError("classifier contract is not the 16-class log-mel input")
    return shape, n_classes

def load_class_map(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw = raw.get("id_to_class", raw)
    mapping = {int(k): str(v).upper() for k, v in raw.items() if str(k).isdigit()}
    if not mapping:
        raise ValueError(f"Could not interpret class map: {path}")
    return mapping

def load_classifier(package_dir):
    session = ort.InferenceSession(str(package_dir / "model.onnx"),
                                   providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name, load_class_map(package_dir / "class_map.json")

def speak(wav_path):
    subprocess.run(
        ["pw-play", str(wav_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )

def load_wake_ack():
    """Read the Yess? clip once. Later playback must not wait on disk."""
    global ACK_AUDIO
    ACK_AUDIO = ACK_WAV.read_bytes()

def start_wake_ack(detected_at):
    """Request Yess? as soon as the wake word is accepted. Never raises."""
    proc = None
    if not ACK_AUDIO:
        add_history("Wake acknowledgment is not loaded.", "error")
        return None
    try:
        proc = subprocess.Popen(
            ["pw-play", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(ACK_AUDIO)
        proc.stdin.close()
        elapsed_ms = (time.monotonic() - detected_at) * 1000.0
        add_history(f"Wake acknowledgment requested {elapsed_ms:.0f} ms after detection.")
        return proc
    except Exception as exc:
        add_history(f"Wake acknowledgment failed: {exc}", "error")
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        return None

def finish_wake_ack(proc):
    """Wait for the short clip so the command microphone does not record it."""
    if proc is None:
        return
    try:
        proc.wait(timeout=3)
    except Exception as exc:
        add_history(f"Wake acknowledgment did not finish: {exc}", "warning")
        try:
            proc.kill()
        except Exception:
            pass

def say_text(text):
    """Synthesize dynamic text with the already-installed Piper voice."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    try:
        subprocess.run([
            str(PIPER_PYTHON), "-m", "piper", "-m", str(PIPER_VOICE),
            "--data-dir", str(PIPER_VOICE.parent), "-f", str(wav_path), "--", text
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        speak(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)

def record_audio(seconds, output_path=None):
    created = output_path is None
    if created:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)
    try:
        subprocess.run(["arecord", "-q", "-D", MIC, "-d", str(seconds), "-t", "wav",
                        "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1",
                        str(output_path)], check=True)
        audio, rate = sf.read(output_path, dtype="float32")
        if rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {rate}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if not audio.size:
            raise ValueError("Empty microphone recording")
        return audio
    finally:
        if created:
            output_path.unlink(missing_ok=True)

def _stop_process(proc):
    """Terminate arecord and reap it. A second wait is also bounded."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _feed_endpoint_bytes(detector, pending, payload):
    """Accumulate short reads. An empty payload is end of stream, not a short frame."""
    if not payload:
        return pending, False
    pending.extend(payload)
    frame_bytes = detector.cfg.frame_samples * 2
    while len(pending) >= frame_bytes and not detector.stopped:
        frame = bytes(pending[:frame_bytes])
        del pending[:frame_bytes]
        detector.push(np.frombuffer(frame, dtype=np.int16).copy())
    return pending, True


def record_command():
    """Endpoint the command mic, then right-pad to 64,000 samples.

    Returns (model_audio, captured_audio). model_audio is the Safe Mix input.
    captured_audio is the span before synthetic padding and is the silence-gate
    input. Wake recording is unchanged. A stalled arecord stops at 6.0 s
    wall clock: 4.0 s of audio plus 2.0 s of device slack.
    """
    import os
    import select
    from command_endpoint import MODEL_SAMPLES, CommandEndpoint

    detector = CommandEndpoint()
    frame_bytes = detector.cfg.frame_samples * 2
    proc = subprocess.Popen(
        ["arecord", "-q", "-D", MIC, "-f", "S16_LE", "-c", "1",
         "-r", str(SAMPLE_RATE), "-t", "raw"],
        stdout=subprocess.PIPE, bufsize=0,
    )
    deadline = time.monotonic() + (MODEL_SAMPLES / float(SAMPLE_RATE)) + 2.0
    pending = bytearray()
    try:
        stream = proc.stdout
        if stream is None:
            raise RuntimeError("command recorder has no audio stream")
        fd = stream.fileno()
        while not detector.stopped and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                break
            try:
                payload = os.read(fd, frame_bytes)
            except OSError:
                break
            pending, alive = _feed_endpoint_bytes(detector, pending, payload)
            if not alive:
                break
    finally:
        _stop_process(proc)
    if not detector.stopped:
        detector._stop_at = min(int(detector._buf.shape[0]), detector.cfg.model_samples)
    captured = detector.captured_unpadded()
    model_audio = detector.captured_model_input()
    if tuple(model_audio.shape) != (MODEL_SAMPLES,):
        from command_endpoint import pad_to_model_length
        model_audio = pad_to_model_length(model_audio, MODEL_SAMPLES)
    return model_audio, captured


def predict(session, input_name, class_map, frontend, mean, std, audio,
            expected_shape, n_classes):
    """Run Safe Mix preprocessing and one ONNX session.

    Exact four-second audio is unchanged. A short or long buffer is fit with
    the package's own fix_length_eval so the fixed 397-frame input is honored.
    """
    fixed = frontend.fix_length_eval(
        np.asarray(audio, dtype=np.float32), frontend.NUM_SAMPLES
    )
    _, model_input = frontend.waveform_to_model_input(fixed, mean, std)
    model_input = np.asarray(model_input, dtype=np.float32)
    if tuple(model_input.shape) != tuple(expected_shape):
        raise RuntimeError(f"unexpected model input shape {tuple(model_input.shape)}")
    if not np.isfinite(model_input).all():
        raise RuntimeError("features contain NaN or Inf")
    output = session.run(None, {input_name: model_input})[0]
    logits = np.asarray(output, dtype=np.float64).reshape(-1)
    if logits.size != n_classes or not np.isfinite(logits).all():
        raise RuntimeError("malformed classifier output")
    logits -= np.max(logits)
    probs = np.exp(logits)
    total = np.sum(probs)
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("malformed classifier probabilities")
    probs /= total
    index = int(np.argmax(probs))
    label = class_map.get(index)
    if not isinstance(label, str) or not label:
        raise RuntimeError("classifier index is not in the class map")
    return label, float(probs[index])


def confidence_is_usable(value):
    """True for a finite probability in [0, 1]. This does not compare the two models."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and 0.0 <= number <= 1.0


def format_confidence(value):
    if not confidence_is_usable(value):
        return "n/a"
    return f"{float(value):.1%}"


def command_is_actionable(label):
    """Recognized command class. UNKNOWN, SILENCE, and unknown labels are not."""
    return label not in INVALID_CLASSES and label in ACTION_CLASSES


def select_command(label, confidence):
    """Accept one actionable class. There is no second model and no extra cutoff."""
    if not confidence_is_usable(confidence):
        return None, "rejected: confidence value is missing or not a probability"
    if not command_is_actionable(label):
        return None, "rejected: prediction is not an actionable command"
    return label, "accepted"


def voice_attempt_result(label, confidence, attempt, attempt_limit):
    """Decide one attempt without speaking or queueing.

    A parameter command is recognized and is not queued. Its value is not invented.
    Playback is 'retry' only for a rejected attempt that is not the last one.
    The final failure clip is selected after the whole sequence.
    """
    candidate, reason = select_command(label, confidence)
    if candidate in PARAMETER_COMMANDS:
        return {
            "candidate": candidate,
            "reason": reason,
            "queued": None,
            "accepted": True,
            "playback": None,
            "notice": True,
        }
    queued = action_for_command(candidate)
    if queued is None:
        playback = "retry" if attempt < attempt_limit else None
        return {
            "candidate": candidate,
            "reason": reason,
            "queued": None,
            "accepted": False,
            "playback": playback,
            "notice": False,
        }
    return {
        "candidate": candidate,
        "reason": reason,
        "queued": queued,
        "accepted": True,
        "playback": None,
        "notice": False,
    }


def action_for_command(command):
    """Return a queue item for one selected intent. Slot commands are not queued."""
    if not command_is_actionable(command) or command in PARAMETER_COMMANDS:
        return None
    return (command, {})


def selection_validation(candidate, queued):
    """Describe the existing slot and action check. This does not execute anything."""
    if candidate in PARAMETER_COMMANDS:
        return "validation: recognized; required slot is not taken from speech"
    if queued is None:
        return "validation: rejected"
    return "validation: passed"


def attempts_for_mode(benchmark_mode):
    """Benchmark trials get one decision. Normal use keeps the three-attempt retry."""
    return 1 if benchmark_mode else MAX_ATTEMPTS


def final_failure_requested(accepted, benchmark_mode):
    """Play the final clip only after a normal sequence with nothing accepted."""
    return (not accepted) and (not benchmark_mode)


def audio_duration_ms(audio, sample_rate=SAMPLE_RATE):
    return float(len(audio)) / float(sample_rate) * 1000.0


def classify_command(session, input_name, class_map, frontend, mean, std, audio,
                     expected_shape, n_classes):
    """Time Safe Mix preprocessing and the single ONNX run."""
    started = time.perf_counter()
    label, confidence = predict(
        session, input_name, class_map, frontend, mean, std, audio,
        expected_shape, n_classes,
    )
    infer_ms = (time.perf_counter() - started) * 1000.0
    return label, confidence, infer_ms


def benchmark_decision_record(command, infer_ms, audio_ms, slot=""):
    """One decision. A missing parameter stays blank and is never guessed."""
    intent = BENCHMARK_INTENT.get(command, "OUT_OF_SCOPE") if command else "OUT_OF_SCOPE"
    return {
        "intent": intent,
        "slot": slot or "",
        "infer_ms": round(float(infer_ms), 1),
        "audio_ms": int(round(float(audio_ms))),
    }


def vosk_model():
    """Load the local Vosk model once. This does not open the microphone."""
    global VOSK_RECOGNIZER
    if VOSK_RECOGNIZER is None:
        from vosk import Model
        if not VOSK_MODEL.is_dir():
            raise FileNotFoundError(f"Vosk model is not available: {VOSK_MODEL}")
        VOSK_RECOGNIZER = Model(str(VOSK_MODEL))
    return VOSK_RECOGNIZER


def transcribe_captured(samples):
    """Turn captured command audio into text. Empty or invalid audio stays empty."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        return ""
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    from vosk import KaldiRecognizer
    recognizer = KaldiRecognizer(vosk_model(), SAMPLE_RATE)
    recognizer.AcceptWaveform(pcm.tobytes())
    result = json.loads(recognizer.FinalResult())
    return str(result.get("text") or "").strip()


def parameter_from_capture(intent, samples):
    """Read and validate the one value required by a Safe Mix parameter intent."""
    try:
        transcript = transcribe_captured(samples)
    except Exception as exc:
        print(f"Value recognition error: {exc}", flush=True)
        return None, "The spoken value could not be read.", ""
    return extract_parameter(intent, transcript)


def wake_benchmark_record():
    return {"event": "wake"}


def benchmark_lines_for_wake(command, infer_ms, audio_ms):
    """Exactly one wake event and one decision for a trial."""
    return [
        wake_benchmark_record(),
        benchmark_decision_record(command, infer_ms, audio_ms),
    ]


def write_benchmark_line(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def open_benchmark_log(directory=None):
    """Create the benchmark directory and log as soon as the assistant is ready."""
    directory = Path(directory) if directory is not None else Path.home() / "vcm_benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"benchmark_{stamp}.log"
    return open(path, "a", encoding="utf-8", buffering=1)


def log_benchmark(record):
    global BENCHMARK_LOG
    if not BENCHMARK_MODE:
        return
    if BENCHMARK_LOG is None or BENCHMARK_LOG.closed:
        BENCHMARK_LOG = open_benchmark_log()
    write_benchmark_line(BENCHMARK_LOG, record)


def wake_trigger_ready(score, now, last_trigger):
    """Apply the existing score check and cooldown. The detector output is unchanged."""
    return score >= WAKE_THRESHOLD and now - last_trigger >= WAKE_COOLDOWN


def listen_for_wake(detector):
    update_state(status="Listening for “Icy”…", wake_status="Listening")
    process = subprocess.Popen(["arecord", "-q", "-D", MIC, "-t", "raw",
        "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    frame_bytes = 2560
    buffer = bytearray()
    last_trigger = 0.0
    try:
        while True:
            chunk = process.stdout.read(frame_bytes)
            if not chunk:
                raise RuntimeError("Microphone stream ended unexpectedly")
            buffer.extend(chunk)
            while len(buffer) >= frame_bytes:
                frame = bytes(buffer[:frame_bytes]); del buffer[:frame_bytes]
                scores = detector.predict(np.frombuffer(frame, dtype=np.int16))
                score = max((float(v) for v in scores.values()), default=0.0)
                with STATE_LOCK:
                    STATE["wake_score"] = score
                    STATE["updated"] = time.time()
                now = time.monotonic()
                if wake_trigger_ready(score, now, last_trigger):
                    add_history(f"Wake word detected ({score:.3f})", "success")
                    update_state(status="Wake word detected", wake_status="Triggered")
                    if BENCHMARK_MODE:
                        return None
                    return start_wake_ack(time.monotonic())
    finally:
        process.terminate()
        try: process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()


def fetch_quezon_city_weather():
    """Fetch current weather for Quezon City using Open-Meteo."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=14.6760&longitude=121.0437"
        "&current=temperature_2m,relative_humidity_2m,"
        "apparent_temperature,weather_code,wind_speed_10m"
        "&timezone=Asia%2FManila"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Icy-Voice-Assistant/1.0"}
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        data = json.loads(response.read().decode("utf-8"))

    current = data.get("current")
    if not isinstance(current, dict):
        raise ValueError("Weather service returned no current conditions")

    descriptions = {
        0: "clear sky",
        1: "mostly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "foggy with rime",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "heavy drizzle",
        56: "light freezing drizzle",
        57: "heavy freezing drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "light snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "light rain showers",
        81: "moderate rain showers",
        82: "heavy rain showers",
        85: "light snow showers",
        86: "heavy snow showers",
        95: "a thunderstorm",
        96: "a thunderstorm with light hail",
        99: "a thunderstorm with heavy hail",
    }

    try:
        temperature = float(current["temperature_2m"])
        feels_like = float(current["apparent_temperature"])
        humidity = int(current["relative_humidity_2m"])
        wind = float(current["wind_speed_10m"])
        code = int(current["weather_code"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Weather service returned incomplete data") from exc

    condition = descriptions.get(code, "conditions that are not classified")
    speech = (
        f"In Quezon City, it's {condition}, "
        f"{temperature:.0f} degrees Celsius. "
        f"It feels like {feels_like:.0f} degrees, "
        f"humidity is {humidity} percent, "
        f"and wind speed is {wind:.0f} kilometers per hour."
    )
    detail = (
        f"Weather in Quezon City: {condition}; "
        f"{temperature:.1f}°C, feels like {feels_like:.1f}°C, "
        f"humidity {humidity}%, wind {wind:.1f} km/h."
    )
    return speech, detail


def dashboard_action(command, payload=None):
    """Execute a software action and report its real outcome."""
    payload = payload or {}
    with DEVICE_LOCK:
        if command == "LIGHT_ON":
            DEVICE_STATE["light_on"] = True
            return True, "Simulated light turned on"
        if command == "LIGHT_OFF":
            DEVICE_STATE["light_on"] = False
            return True, "Simulated light turned off"
        if command == "LIGHT_DIM":
            if "brightness" not in payload:
                return False, "Set a brightness percentage on the dashboard."
            value = payload.get("brightness")
            try:
                value = int(value)
            except (TypeError, ValueError):
                return False, "Brightness must be a number from 0 to 100"
            if not 0 <= value <= 100:
                return False, "Brightness must be between 0 and 100"
            DEVICE_STATE["brightness"] = value
            return True, f"Simulated brightness set to {value}%"
        if command == "SET_TEMPERATURE":
            if "temperature" not in payload:
                return False, "Set a temperature from 0 to 50 on the dashboard."
            value = payload.get("temperature")
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False, "Temperature must be a number"
            if not 0 <= value <= 50:
                return False, "Temperature must be between 0 and 50 degrees"
            DEVICE_STATE["temperature"] = value
            return True, f"Simulated temperature set to {value:g} degrees"
        if command == "PLAY_MUSIC":
            if not MEDIA_FILES:
                return False, f"No MP3 files found in {MEDIA_DIR}"

            if DEVICE_STATE["media"] == "paused" and MEDIA_PROCESS is not None:
                if MEDIA_PROCESS.poll() is None:
                    MEDIA_PROCESS.send_signal(signal.SIGCONT)
                    DEVICE_STATE["media"] = "playing"
                    return True, f"Resumed track {DEVICE_STATE['track']}"

            if (DEVICE_STATE["media"] == "playing"
                    and MEDIA_PROCESS is not None
                    and MEDIA_PROCESS.poll() is None):
                return True, f"Track {DEVICE_STATE['track']} is already playing"

            return start_media_track(DEVICE_STATE["track"] - 1)

        if command == "MEDIA_PAUSE":
            if (DEVICE_STATE["media"] != "playing"
                    or MEDIA_PROCESS is None
                    or MEDIA_PROCESS.poll() is not None):
                return False, "No music is currently playing"
            MEDIA_PROCESS.send_signal(signal.SIGSTOP)
            DEVICE_STATE["media"] = "paused"
            return True, f"Paused track {DEVICE_STATE['track']}"

        if command == "MEDIA_STOP":
            stop_media_process()
            DEVICE_STATE["media"] = "stopped"
            return True, "Music stopped"

        if command == "MEDIA_NEXT":
            if not MEDIA_FILES:
                return False, f"No MP3 files found in {MEDIA_DIR}"
            return start_media_track(DEVICE_STATE["track"] % len(MEDIA_FILES))
        if command == "WEATHER":
            try:
                speech, detail = fetch_quezon_city_weather()
                DEVICE_STATE["weather"] = detail
                say_text(speech)
                return True, detail
            except Exception as exc:
                message = f"Weather unavailable in Quezon City: {exc}"
                DEVICE_STATE["weather"] = message
                return False, message
        if command == "TIME":
            now = datetime.now().strftime("%-I:%M %p")
            say_text(f"The time is {now}.")
            return True, f"Spoke current time: {now}"
        if command == "VOLUME_UP":
            result = subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "10%+"],
                capture_output=True, text=True)
            if result.returncode == 0:
                return True, "System volume increased"
            return False, "Volume control failed: " + (result.stderr.strip() or result.stdout.strip())
        if command == "VOLUME_DOWN":
            result = subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "10%-"],
                capture_output=True, text=True)
            if result.returncode == 0:
                return True, "System volume decreased"
            return False, "Volume control failed: " + (result.stderr.strip() or result.stdout.strip())
        if command == "SET_ALARM":
            clock = str(payload.get("time", ""))
            try:
                target = datetime.strptime(clock, "%H:%M")
            except ValueError:
                return False, "Please say a clock time with AM or PM"
            now = datetime.now()
            target = target.replace(
                year=now.year, month=now.month, day=now.day,
                second=0, microsecond=0
            )
            if target <= now:
                from datetime import timedelta
                target += timedelta(days=1)
            label = str(payload.get("label", "Voice alarm")).strip()[:80] or "Voice alarm"
            item = {
                "id": str(time.time_ns()),
                "label": label,
                "deadline": time.time() + (target - now).total_seconds(),
                "done": False,
            }
            DEVICE_STATE["alarms"].append(item)
            add_history(f"Alarm set for {target.strftime('%I:%M %p')}: {label}", "success")
            return True, f"Alarm set for {target.strftime('%I:%M %p')}"
        if command == "SET_TIMER":
            try:
                seconds = int(payload["seconds"])
            except (KeyError, TypeError, ValueError):
                return False, "Please specify a timer duration in seconds"
            if not 1 <= seconds <= 604800:
                return False, "Timer must be between 1 second and 7 days"
            label = str(payload.get("label", "Voice countdown timer")).strip()[:80]
            item = {"id": str(time.time_ns()), "label": label or "Voice countdown timer",
                    "deadline": time.time() + seconds, "done": False}
            DEVICE_STATE["timers"].append(item)
            add_history(f"Timer created: {item['label']} ({seconds} seconds)", "success")
            return True, f"Timer set for {seconds} seconds"
        return False, f"{COMMAND_TEXT.get(command, command)} is not connected yet"


def run_action(command, payload=None):
    """Execute one selected voice command. Slot values are never guessed."""
    payload = payload or {}
    return dashboard_action(command, payload)


def timer_scheduler():
    """Trigger in-memory dashboard timers and alarms in the background."""
    while True:
        now = time.time()
        expired = []
        with DEVICE_LOCK:
            for timer in DEVICE_STATE["timers"]:
                if not timer["done"] and now >= timer["deadline"]:
                    timer["done"] = True
                    expired.append(("Timer finished", timer["label"]))
            for alarm in DEVICE_STATE["alarms"]:
                if not alarm["done"] and now >= alarm["deadline"]:
                    alarm["done"] = True
                    expired.append(("Alarm triggered", alarm["label"]))
        for title, label in expired:
            if title == "Timer finished":
                update_state(action_status="Timer done")
                try:
                    say_text("Timer done.")
                except Exception:
                    pass
            else:
                add_history("Alarm ringing", "success")
                update_state(action_status="Alarm ringing")
                try:
                    say_text("Alarm ringing.")
                except Exception as exc:
                    add_history(f"Alert speech failed: {exc}", "warning")
        time.sleep(0.5)


def action_worker():
    while True:
        item = ACTION_QUEUE.get()
        if item is None: return
        if isinstance(item, tuple):
            command, payload = item
        else:
            command, payload = item, {}
        update_state(status=f"Executing {COMMAND_TEXT.get(command, command)}…",
                     decision="Accepted", last_command=COMMAND_TEXT.get(command, command))
        ok, message = run_action(command, payload)
        add_history(message, "success" if ok else "warning")
        update_state(action_status=message, status="Command completed" if ok else "Action unavailable",
                     decision="Action succeeded" if ok else "Action unavailable")
        if not ok:
            try:
                if (ROOT / "action_unavailable.wav").is_file():
                    speak(ROOT / "action_unavailable.wav")
                else:
                    say_text("That action is not connected yet.")
            except Exception:
                pass

def voice_loop():
    try:
        require_files()
        frontend, mean, std = load_frontend()
        expected_shape, n_classes = load_model_contract()
        session, input_name, class_map = load_classifier(PACKAGE)
        if len(class_map) != n_classes:
            raise ValueError("class map does not contain 16 classes")
        detector = Model(wakeword_model_paths=[str(WAKE_MODEL)])
        update_state(status="Ready — say “Icy”", wake_status="Listening")
        load_wake_ack()
        add_history("Assistant initialized.", "success")
        global BENCHMARK_LOG
        if BENCHMARK_MODE:
            BENCHMARK_LOG = open_benchmark_log()
            add_history(f"Benchmark log ready: {BENCHMARK_LOG.name}", "success")
            print(f"Benchmark log: {BENCHMARK_LOG.name}", flush=True)
        while True:
            ack = listen_for_wake(detector)
            wake_returned = time.monotonic()
            add_history("Latency | wake accepted")
            try:
                log_benchmark(wake_benchmark_record())
                detector.reset()
                update_state(status="Wake detected — preparing to record", wake_status="Triggered",
                             decision="Waiting for command")
            finally:
                finish_wake_ack(ack)
            accepted = False
            decision_logged = False
            last_infer_ms = 0.0
            last_audio_ms = 0.0
            attempt_limit = attempts_for_mode(BENCHMARK_MODE)
            attempt = 0
            while attempt < attempt_limit:
                attempt += 1
                update_state(status=f"Recording command ({attempt}/{attempt_limit})",
                             decision=f"Attempt {attempt}/{attempt_limit}")
                infer_ms = 0.0
                audio_ms = 0.0
                record_done = None
                classified = False
                pred = "UNKNOWN"
                conf = 0.0
                try:
                    record_started = time.monotonic()
                    audio, captured = record_command()
                    record_done = time.monotonic()
                    add_history(
                        f"Latency | recording complete {(record_done - record_started) * 1000:.0f} ms; "
                        f"{(record_done - wake_returned) * 1000:.0f} ms after wake return"
                    )
                    audio_ms = audio_duration_ms(audio)
                    gate_audio = np.asarray(captured, dtype=np.float64).reshape(-1)
                    if gate_audio.size == 0 or not np.isfinite(gate_audio).all():
                        rms_dbfs = float("-inf")
                    else:
                        rms = float(np.sqrt(np.mean(np.square(gate_audio))))
                        rms_dbfs = 20.0 * np.log10(max(rms, 1e-12))

                    add_history(
                        f"Command audio level: {rms_dbfs:.1f} dBFS"
                    )

                    if rms_dbfs < -32.0:
                        pred = "SILENCE"
                        conf = 0.0
                        add_history(
                            f"Silence rejected by audio gate ({rms_dbfs:.1f} dBFS)",
                            "warning"
                        )
                    else:
                        pred, conf, infer_ms = classify_command(
                            session, input_name, class_map, frontend, mean, std, audio,
                            expected_shape, n_classes,
                        )
                        classified = True
                except Exception as exc:
                    print(f"Command processing error: {exc}", flush=True)
                    add_history("Command processing error", "error")
                    pred = "UNKNOWN"
                    conf = 0.0
                    infer_ms = 0.0
                    classified = False
                last_infer_ms = infer_ms
                last_audio_ms = audio_ms
                if classified:
                    add_prediction_history(pred)
                shown = COMMAND_TEXT.get(pred, "—") if command_is_actionable(pred) else "—"
                update_state(last_command=shown, decision="Prediction ready")
                result = voice_attempt_result(pred, conf, attempt, attempt_limit)
                candidate = result["candidate"]
                reason = result["reason"]
                queued = result["queued"]
                validation = selection_validation(candidate, queued)
                if record_done is not None:
                    add_history(
                        f"Latency | inference {infer_ms:.0f} ms | "
                        f"decision {(time.monotonic() - record_done) * 1000:.0f} ms after recording"
                    )
                if candidate in PARAMETER_COMMANDS:
                    payload, error, slot_text = parameter_from_capture(candidate, captured)
                    if error or payload is None:
                        notice = error or "The spoken value could not be read."
                        update_state(
                            last_command=COMMAND_TEXT.get(candidate, candidate),
                            status=notice,
                            decision="Value rejected",
                            action_status=notice,
                        )
                        add_history(notice, "warning")
                        if BENCHMARK_MODE or attempt >= attempt_limit:
                            log_benchmark(benchmark_decision_record(
                                candidate, infer_ms, audio_ms, slot=""))
                            decision_logged = True
                        if BENCHMARK_MODE:
                            break
                        if attempt < attempt_limit:
                            speak(RETRY_WAV)
                        continue
                    if BENCHMARK_MODE:
                        log_benchmark(benchmark_decision_record(
                            candidate, infer_ms, audio_ms, slot=slot_text))
                        decision_logged = True
                        update_state(
                            last_command=COMMAND_TEXT.get(candidate, candidate),
                            decision="Benchmark decision logged",
                            status="Benchmark decision logged",
                            action_status="Value recognized; benchmark mode did not change the device.",
                        )
                        accepted = True
                        break
                    ok, message = dashboard_action(candidate, payload)
                    if not ok:
                        update_state(
                            last_command=COMMAND_TEXT.get(candidate, candidate),
                            status=message,
                            decision="Value rejected",
                            action_status=message,
                        )
                        add_history(message, "warning")
                        if attempt >= attempt_limit:
                            log_benchmark(benchmark_decision_record(
                                candidate, infer_ms, audio_ms, slot=""))
                            decision_logged = True
                        else:
                            speak(RETRY_WAV)
                        continue
                    log_benchmark(benchmark_decision_record(
                        candidate, infer_ms, audio_ms, slot=slot_text))
                    decision_logged = True
                    add_history(message, "success")
                    update_state(
                        last_command=COMMAND_TEXT.get(candidate, candidate),
                        decision="Accepted — value applied",
                        status=message,
                        action_status=message,
                    )
                    accepted = True
                    break
                if BENCHMARK_MODE:
                    log_benchmark(benchmark_decision_record(candidate, infer_ms, audio_ms))
                    decision_logged = True
                    if queued is not None:
                        add_history(
                            f"Selected action: {candidate} | {reason} | {validation}",
                            "success",
                        )
                        update_state(last_command=COMMAND_TEXT.get(candidate, candidate),
                                     decision="Benchmark decision logged",
                                     status="Benchmark decision logged")
                        accepted = True
                    else:
                        update_state(status="Command not understood",
                                     decision="Benchmark decision logged")
                    break
                if queued is None:
                    if result["playback"] == "retry":
                        update_state(status=f"Didn't get a valid command ({attempt}/{attempt_limit})",
                                     decision="Repeat requested")
                        speak(RETRY_WAV)
                    continue
                log_benchmark(benchmark_decision_record(queued[0], infer_ms, audio_ms))
                decision_logged = True
                add_history(f"Selected action: {queued[0]} | {reason} | {validation}", "success")
                update_state(last_command=COMMAND_TEXT.get(queued[0], queued[0]),
                             decision="Accepted — executing", status="Executing command")
                add_history(
                    f"Latency | action dispatched {(time.monotonic() - wake_returned) * 1000:.0f} ms after wake return"
                )
                ACTION_QUEUE.put(queued)
                accepted = True
                break
            if not decision_logged:
                log_benchmark(benchmark_decision_record(None, last_infer_ms, last_audio_ms))
            if final_failure_requested(accepted, BENCHMARK_MODE):
                update_state(status="Command not understood", decision="3 attempts exhausted")
                add_history("Three attempts failed.", "warning")
                speak(FAILED_WAV)
            elif not accepted:
                add_history("Benchmark decision logged as OUT_OF_SCOPE.", "warning")
            time.sleep(0.5)
    except Exception as exc:
        print(f"Startup/runtime error: {exc}", flush=True)
        update_state(status="Startup/runtime error", wake_status="Error", decision="Error")
        add_history("Assistant error", "error")

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICY_final</title>
<style>
:root {
  --bg0:#031127; --bg1:#061a38; --text:#f0f7ff; --muted:#a7c9ee;
  --blue:#168cff; --cyan:#4ddcff; --glow:#78e8ff; --green:#36f4ba;
  --line:rgba(73,143,213,.18); --border:rgba(91,177,255,.72);
  --panel:rgba(6,25,53,.76);
}
* { box-sizing:border-box; }
html,body { margin:0; height:100%; }
body {
  color:var(--text);
  font-family:"Segoe UI","Avenir Next","Trebuchet MS",sans-serif;
  background:
    radial-gradient(ellipse 48% 42% at 31% 36%, rgba(22,140,255,.20), transparent 62%),
    radial-gradient(ellipse 70% 28% at 36% 100%, rgba(18,92,190,.28), transparent 60%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}
body::before {
  content:""; position:fixed; inset:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(77,160,230,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(77,160,230,.045) 1px, transparent 1px);
  background-size:46px 46px;
  mask-image:radial-gradient(circle at 30% 38%, #000 0%, transparent 72%);
}
button,input { font:inherit; color:inherit; }
button {
  border:1px solid rgba(91,177,255,.45); background:rgba(8,28,62,.72);
  border-radius:999px; padding:6px 10px; cursor:pointer; color:#d7ecff;
}
button:hover { border-color:var(--cyan); box-shadow:0 0 12px rgba(77,220,255,.28); }
input {
  width:100%; border:1px solid rgba(91,177,255,.28); background:rgba(3,16,38,.72);
  border-radius:8px; padding:5px 7px; color:var(--text);
}
.shell {
  position:relative; z-index:1; height:100vh; display:grid;
  grid-template-columns:minmax(0,1fr) 428px; gap:16px;
  padding:20px;
}
.main { min-width:0; min-height:0; display:grid; grid-template-rows:auto minmax(0,1fr) 340px; gap:14px; overflow:hidden; }
.top { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; padding:8px 8px 0 28px; }
.brand { font-weight:300; font-size:66px; letter-spacing:.22em; line-height:.9; color:#8aefff;
  text-shadow:0 0 18px rgba(77,220,255,.55), 0 0 2px #e9fbff; }
.subtitle { margin-top:8px; font-size:13px; letter-spacing:.28em; color:#9fd8ff; font-weight:500; }
.pills { display:flex; gap:12px; align-items:center; padding-top:10px; }
.pill {
  min-width:148px; border:1px solid var(--border); background:rgba(6,24,52,.62);
  border-radius:999px; padding:8px 16px 8px 14px; box-shadow:0 0 18px rgba(22,140,255,.12), inset 0 0 12px rgba(77,220,255,.05);
  display:flex; align-items:center; gap:10px;
}
.pill.time { display:block; text-align:right; padding-right:18px; }
.dot { width:9px; height:9px; border-radius:50%; background:var(--green); box-shadow:0 0 10px var(--green); }
.pill b { display:block; font-weight:600; font-size:14px; }
.pill small { display:block; color:var(--green); font-size:12px; margin-top:1px; }
.pill .date { color:var(--muted); font-size:12px; letter-spacing:.02em; }
.pill .clock { font-size:22px; letter-spacing:.03em; font-weight:500; }
.stage { display:grid; grid-template-columns:minmax(0,1fr) 220px; align-items:center; min-height:0; height:100%; overflow:hidden; }
.orb-wrap { display:grid; place-items:center; min-height:0; height:100%; overflow:hidden; }
.orb { position:relative; width:min(420px, 100%); max-height:100%; aspect-ratio:1; }
.orb svg { width:100%; height:100%; overflow:hidden; filter:drop-shadow(0 0 18px rgba(77,220,255,.28)); }
.spin-slow,.spin-rev { transform-box:fill-box; transform-origin:center; }
.spin-slow { animation:spin 32s linear infinite; }
.spin-rev { animation:spin 22s linear infinite reverse; }
.ring-pulse { transform-origin:center; animation:breathe 3.6s ease-in-out infinite; }
.orb-core {
  position:absolute; inset:33%; border-radius:50%; display:grid; place-content:center; text-align:center;
  background:radial-gradient(circle, rgba(4,16,40,.2), rgba(4,16,40,.55) 70%);
}
.orb-core strong { font-size:28px; font-weight:500; letter-spacing:.02em; }
.orb-core span { color:#d5ebff; font-size:14px; margin-top:4px; }
.stack { display:grid; align-content:center; gap:0; padding-right:8px; min-height:0; max-height:100%; overflow:hidden; }
.stack-row { display:grid; grid-template-columns:42px 1fr; gap:12px; align-items:center; padding:13px 0; border-bottom:1px solid var(--line); }
.stack-row:last-child { border-bottom:0; }
.stack-row svg { width:28px; height:28px; }
.stack-row b { display:block; font-size:15px; font-weight:600; }
.stack-row small { color:var(--green); font-size:13px; }
.stack-row small.idle { color:var(--muted); }
.cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; min-height:0; height:340px; position:relative; z-index:2; }
.cards { align-items:stretch; }
.card {
  min-width:0; border:1px solid rgba(91,177,255,.62); border-radius:20px;
  background:linear-gradient(180deg, rgba(16,48,92,.42), var(--panel));
  box-shadow:0 0 24px rgba(22,140,255,.12), inset 0 1px 0 rgba(160,220,255,.28);
  padding:16px 14px 14px; display:flex; flex-direction:column; gap:8px; overflow:hidden;
}
.card h2 { margin:0; font-size:22px; font-weight:600; text-align:center; }
.card .sub { text-align:center; color:#8ec6ff; font-size:14px; margin-top:2px; }
.art { flex:1 1 auto; min-height:0; display:grid; place-items:center; position:relative; overflow:hidden; }
.art svg { width:auto; height:124px; max-width:100%; display:block; }
.bulb-caption {
  position:absolute; left:50%; top:46%; transform:translate(-50%,-50%);
  font-size:12px; font-weight:700; letter-spacing:.08em; color:#fff4be;
  text-shadow:0 0 12px #ffcb55; opacity:0; transition:opacity .2s; pointer-events:none;
}
.bulb-caption.show { opacity:1; }
.weather-read { text-align:center; }
.weather-read strong { display:block; font-size:26px; font-weight:560; letter-spacing:.02em; }
.weather-read span { display:block; color:#c5e2ff; font-size:13px; line-height:1.3; min-height:18px; }
.metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:8px; text-align:center; }
.metrics small { display:block; color:var(--muted); font-size:10px; }
.metrics b { font-size:12px; font-weight:600; }
.sched-rows { display:grid; gap:8px; margin-top:4px; }
.sched-rows div { display:flex; align-items:center; justify-content:center; gap:8px; color:#d7ecff; font-size:14px; }
.sched-rows svg { width:16px; height:16px; }
.controls { margin-top:auto; display:grid; gap:6px; }
.controls label { color:var(--muted); font-size:11px; display:grid; gap:3px; }
#weather-state:empty { display:none; }
.row { display:flex; gap:6px; flex-wrap:wrap; }
.items { display:grid; gap:4px; max-height:72px; overflow:auto; }
.item { display:flex; justify-content:space-between; gap:6px; font-size:11px; color:#cfe6ff; }
.item button { padding:2px 7px; font-size:10px; }
#control-message { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); }
.side {
  min-height:0; border:1px solid rgba(91,177,255,.55); border-radius:24px;
  background:linear-gradient(180deg, rgba(10,36,74,.55), rgba(5,20,46,.82));
  box-shadow:0 0 28px rgba(22,140,255,.16), inset 0 1px 0 rgba(170,225,255,.22);
  padding:22px 18px 16px; display:flex; flex-direction:column;
}
.side h2 { margin:0 0 14px; font-size:21px; font-weight:500; letter-spacing:.18em; text-align:center; }
.side h2::after { content:""; display:block; height:1px; margin-top:14px; background:linear-gradient(90deg, transparent, rgba(91,177,255,.55), transparent); }
.filters { display:flex; gap:8px; justify-content:center; margin-bottom:16px; }
.filters button { border-radius:999px; padding:6px 12px; font-size:13px; color:#9ec8ef; background:rgba(4,18,40,.35); }
.filters button.active { color:#f4fbff; background:rgba(22,140,255,.28); border-color:#7adfff; box-shadow:0 0 14px rgba(77,220,255,.35); }
.history { position:relative; overflow:auto; min-height:0; padding-left:6px; }
.history::-webkit-scrollbar { width:8px; }
.history::-webkit-scrollbar-thumb { background:rgba(120,190,255,.35); border-radius:99px; }
.timeline { position:relative; display:grid; gap:0; }
.timeline::before { content:""; position:absolute; left:7px; top:8px; bottom:8px; width:2px; background:linear-gradient(#49b6ff, #1d4e86); }
.event { position:relative; display:grid; grid-template-columns:22px 74px 22px 1fr; gap:8px; align-items:start; padding:9px 0 9px 0; }
.marker { width:10px; height:10px; margin-top:4px; border-radius:50%; background:#4ddcff; box-shadow:0 0 8px #4ddcff; }
.event.success .marker { background:var(--green); box-shadow:0 0 8px var(--green); }
.event.warning .marker, .event.error .marker { background:#ffc46b; box-shadow:0 0 8px #ffc46b; }
.event time { color:#8eb6de; font-size:13px; padding-top:1px; }
.event .icon { width:18px; height:18px; margin-top:1px; }
.event .icon svg { width:18px; height:18px; display:block; }
.event b { display:block; font-size:14px; font-weight:600; }
.event span { display:block; color:#b7d4f4; font-size:13px; line-height:1.45; margin-top:2px; }
.empty { color:var(--muted); text-align:center; padding:28px 8px; }
@media (max-width:1100px) {
  .shell { grid-template-columns:1fr; height:auto; min-height:100vh; }
  .side { height:72vh; }
  .stage { grid-template-columns:1fr; }
  .stack { grid-template-columns:1fr 1fr; padding:0 12px 8px; }
  .cards { grid-template-columns:1fr 1fr; }
  .orb { width:min(420px, 78vw); }
}
@media (max-width:720px) {
  .top,.pills { flex-direction:column; align-items:flex-start; }
  .brand { font-size:48px; }
  .cards,.stack { grid-template-columns:1fr; }
  .shell { padding:12px; }
}
@media (prefers-reduced-motion:reduce) {
  .spin-slow,.spin-rev,.ring-pulse,.orbit { animation:none !important; }
}
@keyframes spin { to { transform:rotate(360deg); } }
@keyframes breathe { 50% { opacity:.55; } }
</style>
</head>
<body>
<div class="shell">
  <div class="main">
    <header class="top">
      <div>
        <div class="brand">ICY</div>
        <div class="subtitle">ME2 - Voice Controlled Smart Device</div>
      </div>
      <div class="pills">
        <div class="pill" id="wakepill">
          <span class="dot" id="online-dot"></span>
          <div><b id="online-label">Online</b><small id="online-sub">Assistant Ready</small></div>
        </div>
        <div class="pill time">
          <div class="date" id="now-date">—</div>
          <div class="clock" id="now-time">—</div>
        </div>
      </div>
    </header>
    <section class="stage">
      <div class="orb-wrap">
        <div class="orb" id="visualizer">
          <svg viewBox="0 0 200 200" aria-hidden="true">
            <defs>
              <radialGradient id="coreglow" cx="50%" cy="50%" r="50%">
                <stop offset="0" stop-color="#9af6ff" stop-opacity=".9"/>
                <stop offset="1" stop-color="#168cff" stop-opacity="0"/>
              </radialGradient>
              <linearGradient id="spike" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#e9fdff"/><stop offset="1" stop-color="#168cff"/>
              </linearGradient>
            </defs>
            <circle cx="100" cy="100" r="78" fill="none" stroke="rgba(77,220,255,.18)" stroke-width=".6"/>
            <g class="spin-slow" id="spike-layer"></g>
            <g class="spin-rev">
              <circle cx="100" cy="100" r="62" fill="none" stroke="#7ee7ff" stroke-width=".7" stroke-dasharray="8 10" opacity=".8"/>
              <circle cx="100" cy="100" r="70" fill="none" stroke="#4ddcff" stroke-width=".5" stroke-dasharray="2 7"/>
              <path d="M100 24 A76 76 0 0 1 166 78" fill="none" stroke="#d8fbff" stroke-width="1.1"/>
              <path d="M40 142 A74 74 0 0 1 96 176" fill="none" stroke="#49bfff" stroke-width=".8"/>
            </g>
            <circle class="ring-pulse" cx="100" cy="100" r="48" fill="none" stroke="#f4fdff" stroke-width="1.4"/>
            <circle cx="100" cy="100" r="46" fill="url(#coreglow)" opacity=".18"/>
          </svg>
          <div class="orb-core">
            <strong id="viz-title">Listening</strong>
            <span id="viz-subtitle">for "Icy"...</span>
          </div>
        </div>
      </div>
      <div class="stack">
        <div class="stack-row">
          <svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M6 10v4M10 7v10M14 4v16M18 8v8"/></svg>
          <div><b>Wake Word</b><small id="wake-row-value">Active</small></div>
        </div>
        <div class="stack-row">
          <svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4"/></svg>
          <div><b>Microphone</b><small id="mic-row-value">Listening</small></div>
        </div>
        <div class="stack-row">
          <svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="3" fill="#7ae7ff"/></svg>
          <div><b>Recording</b><small id="rec-row-value" class="idle">Idle</small></div>
        </div>
        <div class="stack-row">
          <svg viewBox="0 0 24 24" fill="#7ae7ff"><circle cx="8" cy="9" r="1.4"/><circle cx="12" cy="6" r="1.4"/><circle cx="16" cy="9" r="1.4"/><circle cx="10" cy="14" r="1.4"/><circle cx="15" cy="15" r="1.4"/></svg>
          <div><b>Assistant</b><small id="asst-row-value">Ready</small></div>
        </div>
        <div id="control-message" role="status">Ready.</div>
      </div>
    </section>
    <section class="cards">
      <article class="card">
        <h2>Lighting</h2>
        <div class="sub">Power &amp; Brightness</div>
        <div class="art" id="bulb-scene">
          <svg viewBox="0 0 180 210">
            <defs>
              <radialGradient id="amber" cx="50%" cy="42%" r="55%">
                <stop offset="0" stop-color="#fff6d2"/><stop offset=".45" stop-color="#ffd36a"/><stop offset="1" stop-color="#ff9a2e"/>
              </radialGradient>
              <linearGradient id="glass" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#d7ecff" stop-opacity=".35"/><stop offset="1" stop-color="#67b7ff" stop-opacity=".05"/>
              </linearGradient>
            </defs>
            <ellipse cx="90" cy="168" rx="46" ry="10" fill="#49bfff" opacity=".35"/>
            <ellipse cx="90" cy="96" rx="62" ry="62" fill="#7ecbff" opacity=".08"/>
            <path d="M90 28c-34 0-58 26-58 58 0 22 12 38 28 52 6 6 8 12 9 20h42c1-8 3-14 9-20 16-14 28-30 28-52 0-32-24-58-58-58z" fill="url(#glass)" stroke="#b9e4ff" stroke-width="1.4"/>
            <path d="M90 42c-24 0-40 18-40 42 0 14 7 26 16 36 4 4 6 8 7 14h34c1-6 3-10 7-14 9-10 16-22 16-36 0-24-16-42-40-42z" fill="url(#amber)" opacity=".92"/>
            <path d="M78 78c8-16 22-16 28-4" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" opacity=".7"/>
            <path d="M74 158h32v8H74zM76 168h28v7H76zM79 177h22v7H79z" fill="#9fb4c9"/>
            <path d="M84 186h12l-2 8H86z" fill="#6e849c"/>
          </svg>
          <div class="bulb-caption" id="bulb-caption"></div>
        </div>
        <div class="sub" id="light-state">Off</div>
      </article>
      <article class="card">
        <h2>Media</h2>
        <div class="sub">Music Playback</div>
        <div class="art">
          <svg viewBox="0 0 180 180">
            <ellipse class="orbit" cx="90" cy="96" rx="62" ry="22" fill="none" stroke="#7adfff" stroke-width="1.2" opacity=".8"/>
            <ellipse cx="90" cy="96" rx="44" ry="14" fill="none" stroke="#d8fbff" stroke-width=".7" opacity=".7"/>
            <path d="M78 118V62l46-12v48" fill="none" stroke="#9aefff" stroke-width="4" stroke-linejoin="round"/>
            <circle cx="70" cy="118" r="12" fill="#49d6ff"/>
            <circle cx="116" cy="100" r="12" fill="#7af0ff"/>
            <ellipse cx="90" cy="142" rx="36" ry="8" fill="#49bfff" opacity=".45"/>
          </svg>
        </div>
        <div class="sub" id="media-state">Stopped</div>
      </article>
      <article class="card">
        <h2>Weather</h2>
        <div class="sub">Quezon City</div>
        <div class="art">
          <svg viewBox="0 0 180 120">
            <ellipse cx="90" cy="58" rx="34" ry="22" fill="#8fd7ff" opacity=".85"/>
            <ellipse cx="66" cy="64" rx="24" ry="18" fill="#bfe9ff"/>
            <ellipse cx="112" cy="66" rx="26" ry="18" fill="#6eb6ee"/>
            <ellipse cx="90" cy="72" rx="40" ry="16" fill="#d7f3ff" opacity=".35"/>
            <g stroke="#7adfff" stroke-width="2" stroke-linecap="round">
              <path d="M62 96l-6 16"/><path d="M82 98l-6 16"/><path d="M102 98l-6 16"/><path d="M122 96l-6 16"/>
            </g>
          </svg>
        </div>
        <div class="weather-read">
          <strong id="weather-temp">—</strong>
          <span id="weather-cond">Waiting for current conditions</span>
        </div>
        <div class="metrics">
          <div><small>Feels like</small><b id="weather-feels">—</b></div>
          <div><small>Humidity</small><b id="weather-hum">—</b></div>
          <div><small>Wind</small><b id="weather-wind">—</b></div>
        </div>
        <p id="weather-state" class="sub"></p>
      </article>
      <article class="card">
        <h2>Time &amp; Schedules</h2>
        <div class="sub">Timers and Alarms</div>
        <div class="art">
          <svg viewBox="0 0 180 180">
            <circle cx="90" cy="90" r="58" fill="none" stroke="#7adfff" stroke-width="3"/>
            <circle cx="90" cy="90" r="68" fill="none" stroke="rgba(77,220,255,.25)" stroke-width="8"/>
            <g id="clock-ticks" stroke="#d8fbff" stroke-width="1.4"></g>
            <line id="hand-hour" x1="90" y1="90" x2="90" y2="58" stroke="#eafcff" stroke-width="3" stroke-linecap="round"/>
            <line id="hand-minute" x1="90" y1="90" x2="90" y2="42" stroke="#7af0ff" stroke-width="2" stroke-linecap="round"/>
            <circle cx="90" cy="90" r="3.5" fill="#f4fdff"/>
          </svg>
        </div>
        <div class="sched-rows">
          <div>
            <svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><circle cx="12" cy="13" r="6"/><path d="M9 6h6M12 13l3-2"/></svg>
            <span id="alarm-count">0 Alarms</span>
          </div>
          <div>
            <svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><circle cx="12" cy="13" r="7"/><path d="M12 13V9"/></svg>
            <span id="timer-count">0 Timers</span>
          </div>
        </div>
        <div class="sub" id="schedule-state"></div>
      </article>
    </section>
  </div>
  <aside class="side">
    <h2>Recent Activity</h2>
    <div class="filters" id="filters">
      <button type="button" class="active" data-filter="all">All</button>
      <button type="button" data-filter="commands">Commands</button>
      <button type="button" data-filter="system">System</button>
      <button type="button" data-filter="warnings">Warnings</button>
    </div>
    <div class="history" id="history"><div class="empty">No activity yet.</div></div>
  </aside>
</div>
<script>
function pct(v){return (100*(Number(v)||0)).toFixed(1)}
async function postJSON(url,data){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const d=await r.json();
  if(!r.ok||!d.ok)throw new Error(d.message||'Request failed');
  return d;
}
function showControl(text){document.getElementById('control-message').textContent=text;}
async function action(name,payload={}){
  try{const d=await postJSON('/api/action',{action:name,...payload});showControl(d.message);await refresh();}
  catch(e){showControl(e.message)}
}
async function setBrightness(){
  try{
    const d=await postJSON('/api/action',{action:'LIGHT_DIM',brightness:Number(document.getElementById('brightness-value').value)});
    showControl(d.message);await refresh();
  }catch(e){showControl(e.message)}
}
async function setTemperature(){
  try{
    const d=await postJSON('/api/action',{action:'SET_TEMPERATURE',temperature:Number(document.getElementById('temperature-value').value)});
    showControl(d.message);await refresh();
  }catch(e){showControl(e.message)}
}
async function createTimer(){
  try{
    const d=await postJSON('/api/timer',{seconds:document.getElementById('timer-seconds').value,label:document.getElementById('timer-label').value});
    showControl(d.message);refresh();
  }catch(e){showControl(e.message)}
}
async function createAlarm(){
  try{
    const d=await postJSON('/api/alarm',{time:document.getElementById('alarm-time').value,label:document.getElementById('alarm-label').value});
    showControl(d.message);refresh();
  }catch(e){showControl(e.message)}
}
async function cancelItem(group,id){
  try{const d=await postJSON('/api/cancel',{group,id});showControl(d.message);refresh();}
  catch(e){showControl(e.message)}
}
function renderItems(id,group,items){
  const root=document.getElementById(id);root.replaceChildren();
  const active=(items||[]).filter(x=>!x.done);
  if(!active.length)return;
  for(const x of active){
    const row=document.createElement('div');row.className='item';
    const label=document.createElement('span');
    label.textContent=x.label+' · '+(x.done?'Triggered':x.remaining+'s left');
    const b=document.createElement('button');b.type='button';b.textContent='Cancel';
    b.onclick=()=>cancelItem(group,x.id);
    row.append(label,b);root.append(row);
  }
}
let lastLightState=null,lastBrightness=null,bulbLabelTimeout=null;
function flashBulbLabel(text){
  const el=document.getElementById('bulb-caption');
  el.textContent=text;el.classList.add('show');
  if(bulbLabelTimeout)clearTimeout(bulbLabelTimeout);
  bulbLabelTimeout=setTimeout(()=>{el.classList.remove('show')},1600);
}
function setRow(id,text,active){
  const el=document.getElementById(id);
  el.textContent=text;el.classList.toggle('idle',!active);
}
function updateVisualizer(s){
  const status=String(s.status||'');
  const low=status.toLowerCase();
  const wake=String(s.wake_status||'');
  const recording=/record/.test(low);
  const processing=/execut|process|confirm|analy/.test(low+' '+String(s.decision||'').toLowerCase());
  const listening=/listen|ready|wake/.test(low)||wake==='Listening'||wake==='Triggered';
  const root=document.getElementById('visualizer');
  root.classList.toggle('processing',processing);
  root.classList.toggle('listening',!processing&&listening);
  const title=document.getElementById('viz-title');
  const sub=document.getElementById('viz-subtitle');
  if(processing){title.textContent='Processing';sub.textContent='command';}
  else if(recording){title.textContent='Recording';sub.textContent='command';}
  else if(listening){title.textContent='Listening';sub.textContent='for "Icy"...';}
  else {title.textContent=status?status.slice(0,22):'Standing by';sub.textContent='say "Icy"';}
  const online=wake!=='Error';
  document.getElementById('online-dot').style.background=online?'#36f4ba':'#ff7c98';
  document.getElementById('online-label').textContent=online?'Online':'Offline';
  document.getElementById('online-sub').textContent=listening?'Assistant Ready':(wake||'Starting');
  document.getElementById('online-sub').style.color=online?'#36f4ba':'#ffb4c4';
  setRow('wake-row-value',(wake==='Listening'||wake==='Triggered')?'Active':(wake||'Starting'),wake==='Listening'||wake==='Triggered');
  setRow('mic-row-value',recording?'Recording':listening?'Listening':'Idle',recording||listening);
  setRow('rec-row-value',recording?'4 seconds':'Idle',recording);
  setRow('asst-row-value',listening?'Ready':(status||'Starting'),listening);
}
function makeSpikes(){
  const svg=document.getElementById('spike-layer');
  const ns='http://www.w3.org/2000/svg';
  for(let i=0;i<72;i++){
    const a=(i/72)*Math.PI*2;
    const inner=i%3===0?74:84;
    const outer=i%6===0?99:i%2===0?93:88;
    const line=document.createElementNS(ns,'line');
    line.setAttribute('x1',(100+Math.cos(a)*inner).toFixed(2));
    line.setAttribute('y1',(100+Math.sin(a)*inner).toFixed(2));
    line.setAttribute('x2',(100+Math.cos(a)*outer).toFixed(2));
    line.setAttribute('y2',(100+Math.sin(a)*outer).toFixed(2));
    line.setAttribute('stroke',i%4===0?'#eafcff':'#49d6ff');
    line.setAttribute('stroke-width',i%6===0?'1.3':'.6');
    svg.append(line);
    if(i%6===0){
      const dot=document.createElementNS(ns,'circle');
      dot.setAttribute('cx',(100+Math.cos(a)*96).toFixed(2));
      dot.setAttribute('cy',(100+Math.sin(a)*96).toFixed(2));
      dot.setAttribute('r','1.5');dot.setAttribute('fill','#eafcff');
      svg.append(dot);
    }
  }
  for(let i=0;i<16;i++){
    const a=(i/16)*Math.PI*2;
    const len=i%2?18:28;
    const poly=document.createElementNS(ns,'polygon');
    const tip=100+len;
    poly.setAttribute('points',[
      [100+Math.cos(a-.05)*70,100+Math.sin(a-.05)*70],
      [100+Math.cos(a)*tip,100+Math.sin(a)*tip],
      [100+Math.cos(a+.05)*70,100+Math.sin(a+.05)*70]
    ].map(p=>p.map(n=>n.toFixed(2)).join(',')).join(' '));
    poly.setAttribute('fill','rgba(120,220,255,.18)');
    poly.setAttribute('stroke','#bdf4ff');
    poly.setAttribute('stroke-width','.6');
    svg.append(poly);
  }
}
function clockTicks(){
  const g=document.getElementById('clock-ticks');
  for(let i=0;i<12;i++){
    const a=(i/12)*Math.PI*2-Math.PI/2;
    const line=document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',(90+Math.cos(a)*48).toFixed(1));
    line.setAttribute('y1',(90+Math.sin(a)*48).toFixed(1));
    line.setAttribute('x2',(90+Math.cos(a)*54).toFixed(1));
    line.setAttribute('y2',(90+Math.sin(a)*54).toFixed(1));
    g.append(line);
  }
}
function tickClock(){
  const now=new Date();
  document.getElementById('now-date').textContent=now.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'});
  document.getElementById('now-time').textContent=now.toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit'});
  const h=now.getHours()%12,m=now.getMinutes(),s=now.getSeconds();
  document.getElementById('hand-hour').setAttribute('transform','rotate('+(h*30+m*.5)+' 90 90)');
  document.getElementById('hand-minute').setAttribute('transform','rotate('+(m*6+s*.1)+' 90 90)');
}
const ICONS={
  weather:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M7 16h10a4 4 0 0 0 0-8 5 5 0 0 0-9.5-1A3.5 3.5 0 0 0 7 16z"/></svg>',
  gear:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><circle cx="12" cy="12" r="3"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M18.4 5.6 17 7M7 17l-1.4 1.4"/></svg>',
  doc:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
  mic:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4"/></svg>',
  bars:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M6 14v4M10 10v8M14 6v12M18 11v7"/></svg>',
  level:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M4 16h3v3H4zM10 11h3v8h-3zM16 6h3v13h-3z"/></svg>',
  spark:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5z"/></svg>',
  clock:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><circle cx="12" cy="12" r="8"/><path d="M12 8v5l3 2"/></svg>',
  note:'<svg viewBox="0 0 24 24" fill="none" stroke="#7ae7ff" stroke-width="1.6"><path d="M9 18V6l10-2v10"/><circle cx="7" cy="18" r="2"/><circle cx="17" cy="14" r="2"/></svg>',
  warn:'<svg viewBox="0 0 24 24" fill="none" stroke="#ffc46b" stroke-width="1.6"><path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>'
};
function eventIcon(message,kind){
  const m=String(message||'').toLowerCase();
  if(kind==='error'||kind==='warning'||/fail|error|reject|unavailable/.test(m))return ICONS.warn;
  if(/weather/.test(m))return ICONS.weather;
  if(/selected action|dashboard:/.test(m))return ICONS.gear;
  if(/transcript|analysis|command audio|latency/.test(m))return ICONS.doc;
  if(/model predicted/.test(m))return ICONS.bars;
  if(/dbfs|audio level/.test(m))return ICONS.level;
  if(/wake word/.test(m))return ICONS.spark;
  if(/time|alarm/.test(m))return ICONS.clock;
  if(/music|media|volume|track/.test(m))return ICONS.note;
  if(/recogn|value|spoken/.test(m))return ICONS.mic;
  return ICONS.doc;
}
function splitMessage(message){
  const text=String(message||'');
  const idx=text.indexOf(':');
  if(idx>0&&idx<48)return [text.slice(0,idx),text.slice(idx+1).trim()];
  return [text,''];
}
let currentFilter='all', latestHistory=[];
function classifyEvent(e){
  const msg=String(e.message||'').toLowerCase();
  if(e.kind==='error'||e.kind==='warning'||/failed|unavailable|error|rejected|warning/.test(msg))return 'warnings';
  if(/model predicted|command|selected action|transcription|weather in|timer|alarm|light|volume|music|temperature|media|forecast|wake word detected/.test(msg))return 'commands';
  return 'system';
}
function renderHistory(){
  const hist=document.getElementById('history');
  const keep=hist.scrollTop;
  hist.replaceChildren();
  const shown=latestHistory.filter(e=>currentFilter==='all'||classifyEvent(e)===currentFilter);
  if(!shown.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No matching activity.';hist.append(empty);return}
  const list=document.createElement('div');list.className='timeline';
  for(const e of shown){
    const row=document.createElement('div');row.className='event '+(e.kind||'');
    const marker=document.createElement('div');marker.className='marker';
    const time=document.createElement('time');time.textContent=e.time||'';
    const icon=document.createElement('div');icon.className='icon';icon.innerHTML=eventIcon(e.message,e.kind);
    const body=document.createElement('div');
    const parts=splitMessage(e.message);
    const title=document.createElement('b');title.textContent=parts[0];
    body.append(title);
    if(parts[1]){const detail=document.createElement('span');detail.textContent=parts[1];body.append(detail);}
    row.append(marker,time,icon,body);list.append(row);
  }
  hist.append(list);
  if(keep>8)hist.scrollTop=keep;
}
document.getElementById('filters').addEventListener('click',e=>{
  const b=e.target.closest('button[data-filter]');if(!b)return;
  currentFilter=b.dataset.filter;
  document.querySelectorAll('#filters button').forEach(x=>x.classList.toggle('active',x===b));
  renderHistory();
});
function parseWeather(text){
  const raw=String(text||'').trim();
  if(!raw||/not configured/i.test(raw))return null;
  const grab=(re)=>{const m=raw.match(re);return m?m[1]:null};
  const head=raw.split(';')[0].replace(/^Weather in Quezon City:\s*/i,'').trim();
  return {
    temp:grab(/(-?\d+(?:\.\d+)?)°C/),
    feels:grab(/feels like\s+(-?\d+(?:\.\d+)?)°C/i),
    hum:grab(/humidity\s+(\d+)%/i),
    wind:grab(/wind\s+(-?\d+(?:\.\d+)?)\s*km\/h/i),
    cond:head||'Current conditions'
  };
}
async function refresh(){
 try{
  const r=await fetch('/api/state',{cache:'no-store'});const s=await r.json();
  updateVisualizer(s);
  showControl(s.action_status||'Ready.');
  const d=s.devices||{};
  const bulb=document.getElementById('bulb-scene');
  bulb.classList.toggle('is-on',!!d.light_on);
  const brightness=Math.max(0,Math.min(100,Number(d.brightness??50)));
  if(lastLightState!==null&&lastLightState!==!!d.light_on)flashBulbLabel(d.light_on?'LIGHT ON':'LIGHT OFF');
  else if(lastBrightness!==null&&lastBrightness!==brightness)flashBulbLabel('BRIGHTNESS '+brightness+'%');
  lastLightState=!!d.light_on;lastBrightness=brightness;
  document.getElementById('light-state').textContent=(d.light_on?'On':'Off')+' · '+brightness+'%';
  document.getElementById('media-state').textContent=(d.media||'stopped')+' · Track '+(d.track||1);
  const weather=parseWeather(d.weather);
  document.getElementById('weather-temp').textContent=weather&&weather.temp?weather.temp+'°C':'—';
  document.getElementById('weather-cond').textContent=weather?weather.cond:'Waiting for current conditions';
  document.getElementById('weather-feels').textContent=weather&&weather.feels?weather.feels+'°C':'—';
  document.getElementById('weather-hum').textContent=weather&&weather.hum?weather.hum+'%':'—';
  document.getElementById('weather-wind').textContent=weather&&weather.wind?weather.wind+' km/h':'—';
  document.getElementById('weather-state').textContent='';
  const timerCount=(d.timers||[]).filter(x=>!x.done).length;
  const alarmCount=(d.alarms||[]).filter(x=>!x.done).length;
  document.getElementById('timer-count').textContent=timerCount+' Timers';
  document.getElementById('alarm-count').textContent=alarmCount+' Alarms';
  document.getElementById('schedule-state').textContent='';
  latestHistory=s.history||[];renderHistory();
 }catch(e){
  document.getElementById('online-label').textContent='Offline';
  document.getElementById('online-sub').textContent='No connection';
  showControl('Dashboard cannot reach the assistant.');
 }
}
makeSpikes();clockTicks();tickClock();setInterval(tickClock,1000);refresh();setInterval(refresh,800);
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML)

@app.get("/api/state")
def api_state():
    with STATE_LOCK:
        snapshot = dict(STATE)
        snapshot["history"] = list(STATE["history"])
    with DEVICE_LOCK:
        snapshot["devices"] = {
            "light_on": DEVICE_STATE["light_on"],
            "brightness": DEVICE_STATE["brightness"],
            "temperature": DEVICE_STATE["temperature"],
            "media": DEVICE_STATE["media"],
            "track": DEVICE_STATE["track"],
            "weather": DEVICE_STATE["weather"],
            "timers": [dict(x) for x in DEVICE_STATE["timers"]],
            "alarms": [dict(x) for x in DEVICE_STATE["alarms"]],
        }
    now = time.time()
    for group in ("timers", "alarms"):
        for item in snapshot["devices"][group]:
            item["remaining"] = max(0, int(item["deadline"] - now))
            item.pop("deadline", None)
    return jsonify(snapshot)


@app.post("/api/action")
def api_action():
    data = request.get_json(silent=True) or {}
    command = str(data.get("action", "")).upper()
    if command not in ACTION_CLASSES:
        return jsonify({"ok": False, "message": "Unknown action"}), 400
    try:
        ok, message = dashboard_action(command, data)
    except Exception as exc:
        ok, message = False, f"Action failed: {exc}"
    add_history(f"Dashboard: {message}", "success" if ok else "warning")
    update_state(last_command=COMMAND_TEXT.get(command, command),
                 action_status=message,
                 decision="Dashboard action succeeded" if ok else "Dashboard action unavailable")
    return jsonify({"ok": ok, "message": message}), (200 if ok else 400)


@app.post("/api/timer")
def api_timer():
    data = request.get_json(silent=True) or {}
    try:
        seconds = int(data.get("seconds", 60))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Timer duration must be a number"}), 400
    if not 1 <= seconds <= 604800:
        return jsonify({"ok": False, "message": "Timer must be between 1 second and 7 days"}), 400
    label = str(data.get("label", "Countdown timer")).strip()[:80] or "Countdown timer"
    item = {"id": str(time.time_ns()), "label": label,
            "deadline": time.time() + seconds, "done": False}
    with DEVICE_LOCK:
        DEVICE_STATE["timers"].append(item)
    add_history(f"Timer created: {label} ({seconds} seconds)", "success")
    return jsonify({"ok": True, "message": f"Timer created for {seconds} seconds"})


@app.post("/api/alarm")
def api_alarm():
    data = request.get_json(silent=True) or {}
    clock = str(data.get("time", ""))
    try:
        target = datetime.strptime(clock, "%H:%M")
    except ValueError:
        return jsonify({"ok": False, "message": "Choose an alarm time in HH:MM format"}), 400
    now = datetime.now()
    target = target.replace(year=now.year, month=now.month, day=now.day, second=0, microsecond=0)
    if target <= now:
        from datetime import timedelta
        target += timedelta(days=1)
    label = str(data.get("label", "Dashboard alarm")).strip()[:80] or "Dashboard alarm"
    item = {"id": str(time.time_ns()), "label": label,
            "deadline": time.time() + (target - now).total_seconds(), "done": False}
    with DEVICE_LOCK:
        DEVICE_STATE["alarms"].append(item)
    add_history(f"Alarm set for {target.strftime('%I:%M %p')}: {label}", "success")
    return jsonify({"ok": True, "message": f"Alarm set for {target.strftime('%I:%M %p')}"})


@app.post("/api/cancel")
def api_cancel():
    data = request.get_json(silent=True) or {}
    item_id = str(data.get("id", ""))
    group = data.get("group")
    if group not in ("timers", "alarms"):
        return jsonify({"ok": False, "message": "Invalid item group"}), 400
    with DEVICE_LOCK:
        before = len(DEVICE_STATE[group])
        DEVICE_STATE[group] = [x for x in DEVICE_STATE[group] if x["id"] != item_id]
        removed = len(DEVICE_STATE[group]) < before
    if removed:
        add_history(f"Cancelled {group[:-1]}: {item_id}", "warning")
    return jsonify({"ok": removed, "message": "Cancelled" if removed else "Item not found"}), (200 if removed else 404)


def main():
    """Start the experimental assistant. Call this only from a manual launch."""
    action_thread = threading.Thread(target=action_worker, daemon=True)
    action_thread.start()
    scheduler = threading.Thread(target=timer_scheduler, daemon=True)
    scheduler.start()
    worker = threading.Thread(target=voice_loop, daemon=True)
    worker.start()
    # Local-only binding: accessible from the Pi itself by default.
    app.run(host="0.0.0.0", port=8765, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
