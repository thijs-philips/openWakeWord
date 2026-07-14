"""Minimal microphone demo for the hey_azurion wakeword model.

Captures 16 kHz mono audio from the default input device, feeds it into
openwakeword in 1280-sample (80 ms) chunks, and prints a banner when the
detection score crosses the threshold. Press Ctrl+C to quit.

Usage:
    python mic_test.py
    python mic_test.py --threshold 0.7 --model hey_azurion.tflite --list-devices
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


CHUNK = 1280            # 80 ms @ 16 kHz — openwakeword's native chunk size
SAMPLE_RATE = 16000
COOLDOWN_SEC = 1.5      # min seconds between consecutive triggers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "hey_azurion.onnx"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    print(f"Loading model: {args.model}")
    framework = "onnx" if args.model.endswith(".onnx") else "tflite"
    model = Model(wakeword_models=[args.model], inference_framework=framework)
    wakeword = os.path.splitext(os.path.basename(args.model))[0]
    print(f"Listening for '{wakeword.replace('_', ' ')}' at threshold {args.threshold} ...")
    print("Speak into the microphone. Ctrl+C to quit.\n")

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        # indata is float32 in [-1, 1]; openwakeword wants int16
        pcm16 = (indata[:, 0] * 32767.0).clip(-32768, 32767).astype(np.int16)
        q.put(pcm16.copy())

    last_trigger = 0.0
    buf = np.zeros(0, dtype=np.int16)

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
                chunk = q.get()
                buf = np.concatenate([buf, chunk])
                while len(buf) >= CHUNK:
                    frame, buf = buf[:CHUNK], buf[CHUNK:]
                    scores = model.predict(frame)
                    score = float(scores.get(wakeword, 0.0))

                    bar_len = int(score * 40)
                    bar = "#" * bar_len + "-" * (40 - bar_len)
                    sys.stdout.write(f"\r[{bar}] {score:0.3f}")
                    sys.stdout.flush()

                    now = time.time()
                    if score >= args.threshold and (now - last_trigger) > COOLDOWN_SEC:
                        last_trigger = now
                        sys.stdout.write(f"\n>>> WAKEWORD DETECTED  score={score:0.3f}  "
                                         f"({time.strftime('%H:%M:%S')})\n")
                        sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
