"""Stage 3b — download the MIT environmental impulse-response corpus.

The HF dataset `davidscripka/MIT_environmental_impulse_responses` is
already 16 kHz mono and only exposes an `audio` field — there is no
environment/location label exposed in the parquet, so we cannot filter
by room type at this point.

That turns out to be acceptable for openWakeWord training: the trainer
randomly samples one RIR per positive clip during augmentation, and
mixing in some out-of-distribution rooms (cathedrals, halls, etc.)
empirically improves generalisation to unseen rooms more than it hurts
in-room recall. If we later want a curated subset, drop unwanted files
from ./rirs/ by hand and the trainer will simply not see them.

USAGE
-----
    python fetch_rirs.py                  # fetch the whole corpus (~270)
    python fetch_rirs.py --max 100        # cap to first 100

Output: ./rirs/rir_{idx:04d}.wav  (16 kHz mono int16 PCM)
Plug into YAML:
    rir_paths:
      - "./training/hey_azurion/rirs"
"""

from __future__ import annotations

import argparse
import os
import sys
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "rirs"
TARGET_SR = 16_000

# Pull HF_TOKEN from the azure-ai-foundry skill's .env so HF downloads run
# authenticated (higher rate limits, gated dataset access).
_SKILL_ENV = (HERE.parent.parent / ".github" / "skills" /
              "azure-ai-foundry" / "scripts" / ".env")
if _SKILL_ENV.is_file() and not os.environ.get("HF_TOKEN"):
    for _line in _SKILL_ENV.read_text(encoding="utf-8").splitlines():
        if _line.startswith("HF_TOKEN="):
            _tok = _line.split("=", 1)[1].strip()
            os.environ["HF_TOKEN"] = _tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
            break


def _to_pcm16_mono_16k(arr: np.ndarray, sr: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        arr = scipy.signal.resample_poly(arr, TARGET_SR // g, sr // g)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / max(peak, 1e-6) * 0.95
    return (arr * 32767.0).astype(np.int16)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max", type=int, default=None,
                   help="Cap fetched count (default: all)")
    args = p.parse_args()

    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in OUT_DIR.glob("*.wav")}
    print(f"existing in {OUT_DIR}: {len(existing)}")

    ds = load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )

    written = 0
    for idx, ex in enumerate(ds):
        if args.max is not None and written >= args.max:
            break
        name = f"rir_{idx:04d}.wav"
        if name in existing:
            continue
        audio = ex.get("audio") or {}
        arr = audio.get("array")
        sr = int(audio.get("sampling_rate") or 0)
        if arr is None or sr == 0:
            continue
        try:
            pcm = _to_pcm16_mono_16k(arr, sr)
            wavfile.write(OUT_DIR / name, TARGET_SR, pcm)
            written += 1
            if written % 50 == 0:
                print(f"  written {written}")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name}: {e}", file=sys.stderr)

    total = len(list(OUT_DIR.glob("*.wav")))
    print(f"\ndone. new={written}  total_on_disk={total}")
    print(f"\nYAML snippet:")
    print(f"  rir_paths:")
    print(f"    - \"{OUT_DIR}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
