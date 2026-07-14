"""Import the hospital operating-room background MP3 into the training corpus.

Reads a long MP3 of OR background sound, downmixes to mono, resamples to
16 kHz, and slices into ~10 s WAV chunks under ./backgrounds/hospital_or/.

These are domain-specific backgrounds (alarms, monitors, suction, staff
movement at distance) that perfectly match the device's real deployment
environment, so they're more valuable than generic AudioSet ambience and
should get a HIGHER duplication rate in the trainer YAML.
"""

from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import soundfile as sf
import librosa

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = Path(
    r"C:\Users\nly96630\OneDrive - Philips\SharePoints\NextGen Test Lab - Documents"
    r"\Feature integration\2. Feature info\Voice control\Background sounds"
    r"\background-sounds.mp3"
)
OUT_DIR = HERE / "backgrounds" / "hospital_or"
TARGET_SR = 16_000


def _resample(arr: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return arr
    g = gcd(sr, TARGET_SR)
    return scipy.signal.resample_poly(arr, TARGET_SR // g, sr // g)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="Source MP3 (default: Philips OneDrive path)")
    p.add_argument("--chunk-sec", type=float, default=10.0,
                   help="Chunk length in seconds (default 10)")
    p.add_argument("--overlap-sec", type=float, default=0.0,
                   help="Overlap between consecutive chunks (default 0)")
    p.add_argument("--min-chunk-sec", type=float, default=4.0,
                   help="Drop final chunk if shorter than this")
    args = p.parse_args()

    src = args.src.resolve()
    if not src.is_file():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 2

    print(f"loading {src.name} ({src.stat().st_size/1e6:.1f} MB) ...")
    # librosa handles MP3 via audioread / soundfile; force mono + 16 kHz.
    arr, sr = librosa.load(str(src), sr=TARGET_SR, mono=True)
    print(f"  total duration: {len(arr)/sr:.1f} s @ {sr} Hz mono")

    # Normalise once globally so chunk amplitudes match each other.
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / peak * 0.95

    chunk_n = int(args.chunk_sec * sr)
    step_n = max(1, chunk_n - int(args.overlap_sec * sr))
    min_n = int(args.min_chunk_sec * sr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for start in range(0, len(arr), step_n):
        chunk = arr[start:start + chunk_n]
        if len(chunk) < min_n:
            break
        pcm = (chunk * 32767.0).astype(np.int16)
        out = OUT_DIR / f"or_{written:04d}.wav"
        wavfile.write(out, sr, pcm)
        written += 1

    print(f"\nwrote {written} chunks of ~{args.chunk_sec:.0f}s to {OUT_DIR}")
    print(f"\nYAML snippet:")
    print(f"  background_paths:")
    print(f"    - \"{OUT_DIR}\"           # hospital OR — duplicate aggressively")
    print(f"  background_paths_duplication_rate:")
    print(f"    - 5                                  # heavy: this is the deployment domain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
