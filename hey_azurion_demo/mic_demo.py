"""Minimal microphone demo for the 'Hey, Azurion' wakeword model.

Captures 16 kHz mono audio from the default input device, runs it through
openwakeword in 80 ms chunks, prints a live score bar, and announces every
detection that crosses the threshold.

    python mic_demo.py                       # default device, threshold 0.5
    python mic_demo.py --list-devices        # see available input devices
    python mic_demo.py --device 1 --threshold 0.7
    python mic_demo.py --model hey_azurion.tflite

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


SAMPLE_RATE = 16000   # openwakeword is fixed at 16 kHz mono int16
CHUNK = 1280          # 80 ms — openwakeword's native frame size
COOLDOWN_SEC = 1.5    # min seconds between two consecutive triggers


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(here, "hey_azurion.onnx"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=int, default=None,
                   help="input device index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    framework = "onnx" if args.model.endswith(".onnx") else "tflite"
    print(f"Loading model: {args.model}  (backend: {framework})")
    model = Model(wakeword_models=[args.model], inference_framework=framework)
    wakeword_key = os.path.splitext(os.path.basename(args.model))[0]
    print(f"Listening for '{wakeword_key.replace('_', ' ')}' "
          f"at threshold {args.threshold}.  Ctrl+C to quit.\n")

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        # indata is float32 in [-1, 1]; openwakeword wants int16
        pcm16 = (indata[:, 0] * 32767.0).clip(-32768, 32767).astype(np.int16)
        audio_q.put(pcm16.copy())

    buf = np.zeros(0, dtype=np.int16)
    last_trigger = 0.0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK,
        device=args.device,
        callback=callback,
    ):
        try:
            while True:
                buf = np.concatenate([buf, audio_q.get()])
                while len(buf) >= CHUNK:
                    frame, buf = buf[:CHUNK], buf[CHUNK:]
                    scores = model.predict(frame)
                    score = float(scores.get(wakeword_key, 0.0))

                    bar = "#" * int(score * 40) + "-" * (40 - int(score * 40))
                    sys.stdout.write(f"\r[{bar}] {score:0.3f}")
                    sys.stdout.flush()

                    now = time.time()
                    if score >= args.threshold and (now - last_trigger) > COOLDOWN_SEC:
                        last_trigger = now
                        sys.stdout.write(
                            f"\n>>> WAKEWORD DETECTED  score={score:0.3f}  "
                            f"({time.strftime('%H:%M:%S')})\n"
                        )
                        sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
