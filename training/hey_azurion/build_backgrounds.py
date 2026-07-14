"""Build a filtered background-noise corpus from agkphysics/AudioSet.

Stage 3a of training. Produces ~3,000 short 16 kHz mono WAVs under
./backgrounds/ that the openWakeWord trainer will mix into positive
clips during augmentation (`background_paths` in the YAML).

WHY FILTER
----------
AudioSet contains a lot of clips labelled `Speech`, `Conversation`,
`Narration` etc. Mixing those into positive samples teaches the model
to fire whenever someone is talking nearby — the opposite of what we
want. We also DOWN-WEIGHT (don't exclude) `Television` so the model
learns to ignore TV chatter, which is openWakeWord's most common
real-world false-positive trigger.

LABEL POLICY
------------
A clip is KEPT iff at least one of its labels is in KEEP and none of
its labels are in REJECT.

A clip is additionally classified as "tv-like" if any label matches
TV_LIKE — those clips are written to a separate sub-directory so the
training YAML can give them their own (lower) duplication rate.

OUTPUT LAYOUT
-------------
./backgrounds/
    household/      <- general home ambience: ~80% of the corpus
    tv_radio/       <- TV/radio babble: ~20%, low duplication
    manifest.csv    <- one row per kept clip with labels + duration

USAGE
-----
    python build_backgrounds.py                # default: 3000 clips
    python build_backgrounds.py --n 5000       # bigger corpus
    python build_backgrounds.py --dry-run      # list, don't download

The script uses the HuggingFace `datasets` streaming API so we never
materialise the full ~50 GB dataset on disk. Only kept clips are
written, and they're downsampled to 16 kHz mono on the fly.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "backgrounds"

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

TARGET_SR = 16_000  # openWakeWord operates on 16 kHz audio

# --- Label sets --------------------------------------------------------------
# AudioSet uses human-readable label strings in the agkphysics dataset.
# These come from the AudioSet ontology (https://research.google.com/audioset/ontology/).

# Reject = "this clip is too speech-heavy to be useful as background"
# A single match here is enough to drop the clip.
REJECT = {
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Whispering",
    "Shout",
    "Yell",
    "Children shouting",
    "Screaming",
    "Whoop",
    "Battle cry",
    "Crying, sobbing",
    "Singing",
    "Choir",
    "Yodeling",
    "Chant",
    "Rapping",
    "Humming",
    "Beatboxing",
}

# Keep = household / indoor ambience that a smart-speaker will actually hear.
TV_LIKE = {
    "Television",
    "Radio",
}

KEEP = {
    # rooms / ambience
    "Inside, small room",
    "Inside, large room or hall",
    "Inside, public space",
    # household sounds
    "Domestic sounds, home sounds",
    "Door",
    "Doorbell",
    "Cupboard open or close",
    "Drawer open or close",
    "Dishes, pots, and pans",
    "Cutlery, silverware",
    "Chopping (food)",
    "Frying (food)",
    "Microwave oven",
    "Blender",
    "Kettle whistle",
    "Water tap, faucet",
    "Sink (filling or washing)",
    "Bathtub (filling or washing)",
    "Toilet flush",
    "Toothbrush",
    "Vacuum cleaner",
    "Hair dryer",
    "Washing machine",
    "Dishwasher",
    # HVAC / appliances
    "Mechanical fan",
    "Air conditioning",
    "Refrigerator",
    # background tech
    "Typing",
    "Computer keyboard",
    "Mouse click",
    "Printer",
    "Telephone bell ringing",
    "Telephone",
    "Ringtone",
    "Notification sound",
    "Alarm clock",
    # music (instrumental-ish, not vocal-led — already de-risked by REJECT)
    "Music",
    "Background music",
    "Musical instrument",
    "Piano",
    "Guitar",
    "Drum kit",
    "Drum",
    # pets / small animals (often present at home)
    "Dog",
    "Cat",
    "Bark",
    "Meow",
    "Purr",
    # passing traffic that bleeds in through windows
    "Vehicle",
    "Car",
    "Traffic noise, roadway noise",
    "Motor vehicle (road)",
    # TV/radio kept in own bucket
    *TV_LIKE,
}


# --- HF dataset loader -------------------------------------------------------
def _load_dataset(split: str):
    from datasets import load_dataset  # local import so --help works without deps

    # The agkphysics/AudioSet dataset is a wrapper around the official AudioSet
    # CSVs with the audio resolved from YouTube. Streaming avoids the full
    # download.
    return load_dataset(
        "agkphysics/AudioSet",
        "balanced",        # ~22k clips total; richer label coverage per clip
        split=split,
        streaming=True,
        trust_remote_code=True,
    )


def _classify(labels: list[str]) -> str | None:
    """Return 'household' | 'tv_radio' | None (reject)."""
    label_set = set(labels)
    if label_set & REJECT:
        return None
    if not (label_set & KEEP):
        return None
    if label_set & TV_LIKE:
        return "tv_radio"
    return "household"


# --- Audio I/O ---------------------------------------------------------------
def _to_pcm16_mono_16k(audio: dict) -> np.ndarray:
    """Take a HF Audio dict {'array': np.float32, 'sampling_rate': int}
    and return int16 mono samples at TARGET_SR."""
    import scipy.signal

    arr = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])

    if arr.ndim > 1:
        arr = arr.mean(axis=-1)

    if sr != TARGET_SR:
        # polyphase resample — handles arbitrary sr ratios cleanly
        from math import gcd
        g = gcd(sr, TARGET_SR)
        up, down = TARGET_SR // g, sr // g
        arr = scipy.signal.resample_poly(arr, up, down)

    # Normalise to int16 without clipping headroom
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / max(peak, 1e-6) * 0.95
    return (arr * 32767.0).astype(np.int16)


def _write_wav(path: Path, pcm: np.ndarray) -> int:
    import scipy.io.wavfile as wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, TARGET_SR, pcm)
    return path.stat().st_size


# --- Main --------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=3000, help="Total clips to keep")
    p.add_argument("--tv-fraction", type=float, default=0.20,
                   help="Max fraction of tv_radio clips (default 0.20)")
    p.add_argument("--split", default="train",
                   help="HF split: train | test (default train)")
    p.add_argument("--min-sec", type=float, default=4.0,
                   help="Drop clips shorter than this")
    p.add_argument("--max-sec", type=float, default=12.0,
                   help="Cap clips to this duration")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't download/write — just classify and count")
    args = p.parse_args()

    tv_cap = int(args.n * args.tv_fraction)
    household_cap = args.n - tv_cap

    counts = {"household": 0, "tv_radio": 0, "rejected": 0, "too_short": 0,
              "errors": 0}
    rows: list[dict] = []

    print(f"target: {household_cap} household + {tv_cap} tv_radio = {args.n}")
    print(f"streaming agkphysics/AudioSet[{args.split}] ...")

    ds = _load_dataset(args.split)
    for ex in ds:
        if counts["household"] >= household_cap and counts["tv_radio"] >= tv_cap:
            break

        labels = ex.get("human_labels") or ex.get("labels") or []
        if isinstance(labels, str):
            labels = [labels]

        bucket = _classify(labels)
        if bucket is None:
            counts["rejected"] += 1
            continue
        if bucket == "household" and counts["household"] >= household_cap:
            continue
        if bucket == "tv_radio" and counts["tv_radio"] >= tv_cap:
            continue

        try:
            audio = ex["audio"]
            sr = int(audio["sampling_rate"])
            dur = len(audio["array"]) / sr if sr else 0
            if dur < args.min_sec:
                counts["too_short"] += 1
                continue
        except Exception as e:  # noqa: BLE001
            counts["errors"] += 1
            print(f"  [skip] audio read failed: {e}", file=sys.stderr)
            continue

        ytid = ex.get("video_id") or ex.get("id") or f"clip_{sum(counts.values())}"
        out_name = f"{ytid}.wav"
        out_path = OUT_ROOT / bucket / out_name

        if not args.dry_run:
            try:
                pcm = _to_pcm16_mono_16k(audio)
                # cap duration
                max_n = int(args.max_sec * TARGET_SR)
                if len(pcm) > max_n:
                    pcm = pcm[:max_n]
                _write_wav(out_path, pcm)
            except Exception as e:  # noqa: BLE001
                counts["errors"] += 1
                print(f"  [skip] write failed for {ytid}: {e}", file=sys.stderr)
                continue

        counts[bucket] += 1
        rows.append({
            "bucket": bucket,
            "ytid": ytid,
            "duration_sec": round(dur, 2),
            "labels": "|".join(labels),
            "path": str(out_path.relative_to(OUT_ROOT)) if not args.dry_run else "",
        })

        if (counts["household"] + counts["tv_radio"]) % 100 == 0:
            print(f"  kept: household={counts['household']}/{household_cap} "
                  f"tv_radio={counts['tv_radio']}/{tv_cap} "
                  f"rejected={counts['rejected']}")

    # Manifest
    if not args.dry_run:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        manifest = OUT_ROOT / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["bucket", "ytid", "duration_sec",
                                              "labels", "path"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {manifest}")

    print("\nsummary:")
    for k, v in counts.items():
        print(f"  {k:<12} {v}")
    print(f"\nUse in your YAML:")
    print(f"  background_paths:")
    print(f"    - \"{OUT_ROOT / 'household'}\"")
    print(f"    - \"{OUT_ROOT / 'tv_radio'}\"")
    print(f"  background_paths_duplication_rate:")
    print(f"    - 2   # household: oversample slightly")
    print(f"    - 1   # tv_radio: keep low so we don't bias toward TV-only ambience")
    return 0


if __name__ == "__main__":
    sys.exit(main())
