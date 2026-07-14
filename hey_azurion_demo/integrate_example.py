"""Minimal embedding example: drop a HeyAzurionDetector into your own app.

This shows the smallest possible wrapper around openwakeword for use as a
library inside a larger application (chatbot, robot controller, kiosk, etc.).

    from integrate_example import HeyAzurionDetector

    det = HeyAzurionDetector(threshold=0.5)
    # ... in your audio loop, for every 80 ms / 1280-sample int16 mono frame:
    if det.process(frame):
        on_wake_word()   # your callback
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from openwakeword.model import Model


class HeyAzurionDetector:
    """Stateful detector. Feed it consecutive int16 mono 16 kHz frames of
    exactly 1280 samples (80 ms) — openwakeword's native frame size — and it
    returns True on the frame that crosses the threshold."""

    CHUNK = 1280
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        cooldown_sec: float = 1.5,
    ):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "hey_azurion.onnx")
        framework = "onnx" if model_path.endswith(".onnx") else "tflite"
        self._model = Model(wakeword_models=[model_path], inference_framework=framework)
        self._key = os.path.splitext(os.path.basename(model_path))[0]
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self._last_trigger = 0.0
        self.last_score = 0.0

    def process(self, frame_int16: np.ndarray) -> bool:
        """Feed exactly 1280 int16 samples. Returns True on detection."""
        if frame_int16.dtype != np.int16:
            raise ValueError(f"frame must be int16, got {frame_int16.dtype}")
        if len(frame_int16) != self.CHUNK:
            raise ValueError(f"frame must be {self.CHUNK} samples, got {len(frame_int16)}")

        scores = self._model.predict(frame_int16)
        self.last_score = float(scores.get(self._key, 0.0))

        now = time.time()
        if self.last_score >= self.threshold and (now - self._last_trigger) > self.cooldown_sec:
            self._last_trigger = now
            return True
        return False

    def reset(self) -> None:
        """Clear internal feature buffers — call this between sessions."""
        self._model.reset()
        self._last_trigger = 0.0
        self.last_score = 0.0


if __name__ == "__main__":
    # Tiny smoke test against a WAV file.
    import sys
    import wave

    if len(sys.argv) < 2:
        print("Usage: python integrate_example.py path\\to\\some.wav")
        sys.exit(2)

    det = HeyAzurionDetector()
    with wave.open(sys.argv[1], "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1, \
            "WAV must be 16 kHz mono"
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    peak = 0.0
    for i in range(0, len(pcm) - det.CHUNK + 1, det.CHUNK):
        triggered = det.process(pcm[i:i + det.CHUNK])
        peak = max(peak, det.last_score)
        if triggered:
            print(f"[{i / 16000:0.2f}s] DETECTED  score={det.last_score:0.3f}")
    print(f"\nPeak score over file: {peak:0.3f}")
