# Hey, Azurion — wakeword demo & integration guide

Self-contained "Hey, Azurion" wakeword package built on
[openWakeWord](https://github.com/dscripka/openWakeWord).

This folder is everything you need to:

1. Run a live microphone test on Windows/Linux/macOS.
2. Drop the wakeword detector into your own application.

---

## Contents

| File | Purpose |
| --- | --- |
| `hey_azurion.onnx` | The trained wakeword classifier (recommended on desktop). |
| `hey_azurion.tflite` | Same model, TensorFlow Lite format (smaller; better for edge/ARM). |
| `requirements.txt` | Python dependencies. |
| `setup.bat` | One-click Windows setup: creates `.venv` and installs everything. |
| `mic_demo.py` | Live microphone tester (CLI score bar + detection banner). |
| `run_mic_demo.bat` | Convenience launcher for `mic_demo.py`. |
| `integrate_example.py` | Minimal `HeyAzurionDetector` class — copy/adapt for your app. |

---

## Quick start (Windows)

```powershell
cd hey_azurion_demo
setup.bat                       # one-time: creates .venv, installs deps
run_mic_demo.bat --list-devices # see your microphones
run_mic_demo.bat                # start listening on the default input device
run_mic_demo.bat --device 1 --threshold 0.7
```

Say "Hey, Azurion" — you should see the score bar fill and a
`>>> WAKEWORD DETECTED` line appear.

## Quick start (any OS)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import openwakeword; openwakeword.utils.download_models()"
.venv/bin/python mic_demo.py
```

The first run downloads two small preprocessor models (~5 MB total) into the
`openwakeword` package cache: the mel-spectrogram extractor and Google's
speech-embedding network. The actual wakeword classifier is the local
`.onnx` / `.tflite` file in this folder.

---

## How the model works (in one paragraph)

openWakeWord runs a fixed pipeline on every 80 ms (1280-sample) chunk of
**16 kHz mono int16 PCM** audio:

```
mic → mel-spectrogram → Google speech embedding → hey_azurion.onnx → score ∈ [0,1]
```

You only own the last stage; the first two are shared and downloaded
automatically. The classifier looks at a sliding 1.5-second window of
embeddings and outputs a single sigmoid score per frame. Cross a threshold
(typically 0.5 – 0.8) and you've got a wakeword.

---

## Integrating into your own app

The whole integration is ~3 lines. Copy `integrate_example.py` into your
project (or import from it) and use it like this:

```python
from integrate_example import HeyAzurionDetector

detector = HeyAzurionDetector(
    model_path="hey_azurion.onnx",   # or .tflite
    threshold=0.5,                   # raise to 0.7+ for fewer false positives
    cooldown_sec=1.5,                # min seconds between two triggers
)

# Inside your audio loop. `frame` must be a numpy int16 array of EXACTLY
# 1280 samples (80 ms) at 16 kHz mono.
if detector.process(frame):
    on_wake_word()
```

### Audio requirements (non-negotiable)

| Property | Value |
| --- | --- |
| Sample rate | **16 000 Hz** |
| Channels | **1 (mono)** |
| Sample format | **int16** |
| Frame size fed to `process()` | **1280 samples** (80 ms) |

If your capture source is different, resample to 16 kHz first
(`scipy.signal.resample_poly` or `soxr`) and split the stream into 1280-sample
frames.

### Choosing the threshold

Run `eval_testset.py` on a held-out set (or just listen with `mic_demo.py`)
and pick the threshold that gives you the recall/false-trigger trade-off you
want. For this build:

| Threshold | Recall | False trigger rate on the *adversarial* negative set |
| --- | --- | --- |
| 0.30 | 1.000 | 0.110 |
| 0.50 | 1.000 | 0.090 |
| 0.70 | 1.000 | 0.090 |

Real-world (non-adversarial) false-trigger rates will be lower. **Start at
0.5** and only increase if you see false wakes in your environment.

### State & reset

`HeyAzurionDetector` keeps an internal feature buffer (the sliding embedding
window). If your audio stream pauses or restarts (e.g. user opens a new
session), call `detector.reset()` to clear it — otherwise stale audio could
contaminate the first second of the next session.

### Choosing ONNX vs TFLite

| Backend | When to use |
| --- | --- |
| `hey_azurion.onnx` (1.4 MB) | Desktop, server, anything with `onnxruntime`. Slightly faster on x86. |
| `hey_azurion.tflite` (0.7 MB) | Embedded / Raspberry Pi / Android. Lower memory footprint. |

Both produce identical detection scores in practice.

### Performance

CPU cost is dominated by the shared embedding model, not by the wakeword
classifier itself. Expect ~1–3 % of a single x86 core when feeding real-time
16 kHz audio. The classifier itself is 4 layers (128 units) — negligible.

### Multiple wakewords

To run several wakewords (e.g. "Hey, Azurion" + a sleep word), just pass a
list:

```python
from openwakeword.model import Model
m = Model(wakeword_models=["hey_azurion.onnx", "go_to_sleep.onnx"],
          inference_framework="onnx")
scores = m.predict(frame)        # -> {'hey_azurion': 0.97, 'go_to_sleep': 0.02}
```

You only pay the embedding cost once regardless of how many classifiers you
load.

### Optional add-ons

`Model(..., vad_threshold=0.5)` enables Silero VAD: predictions are zeroed
out when no speech is present. Cuts false triggers from non-speech noise at
the cost of ~1 ms extra latency per frame.

`Model(..., enable_speex_noise_suppression=True)` runs SpeexDSP NS on the
incoming audio. Useful in stationary noisy environments (HVAC, server room).

---

## Troubleshooting

**"No module named 'openwakeword'"**  
Run `setup.bat`, or activate the venv before running anything:
`.venv\Scripts\activate`.

**"No input device" / silence**  
List devices with `mic_demo.py --list-devices` and pass the right index with
`--device N`.

**Score never goes above ~0.2 when I speak**  
Check your mic is being captured by the *right* device (use the score bar
with random noise — it should react). Confirm the stream is 16 kHz mono.
Hold the mic ~30 cm away and speak clearly: "Hey. Azurion."

**Lots of false triggers from TV / radio**  
Raise `--threshold` to 0.7 or 0.8, or enable `vad_threshold=0.5` in
`HeyAzurionDetector` (edit the `Model(...)` call).

---

## License

The wakeword model is your own training artifact. The openWakeWord runtime
and Google speech-embedding preprocessor are Apache-2.0.
