"""gpt-4o-mini-tts deep-dive for the "Hey, Azurion" wake word.

The first pronunciation_test.py only swept 4 voices x 3 tones on Group D.
The user liked those clips a lot and asked for more variation:
  - more voices
  - more tones
  - more emotions
  - speed variations
  - slur / mumble / whisper / etc.

gpt-4o-mini-tts has no SSML. We steer it via:
  * `voice`         — one of the 11 Azure-shipped voices
  * `speed`         — 0.25..4.0 (we stay in 0.8..1.25 for realism)
  * `instructions`  — free-form natural-language style prompt; this is
                      where tone, emotion, slur, accent, pacing live
  * `input`         — the raw text. Because the model doesn't read IPA,
                      we ALSO sneak the canonical respellings from the
                      Stage-1 ipa_mix.json into a subset of clips, so we
                      can hear whether respelled input + voice instruction
                      converges on the same pronunciation as Azure Speech
                      with `<phoneme ph="...">`.

All clips land under ./out_gpt_tts/<bucket>/<voice>__<label>.wav
plus a manifest.csv with one row per clip so we can scribble verdicts.

Auth/endpoints/SSL reused from the azure-ai-foundry skill.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- plumb the azure-ai-foundry skill helpers ------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SKILL_SCRIPTS = REPO_ROOT / ".github" / "skills" / "azure-ai-foundry" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from common import (  # noqa: E402
    AZURE_OPENAI_TTS_DEPLOYMENT,
    fail,
    header,
    make_azure_openai_tts_client,
    ok,
)

OUT_ROOT = HERE / "out_gpt_tts"

# ---------------------------------------------------------------------------
# Configuration matrices
# ---------------------------------------------------------------------------
# All 11 voices Azure currently ships for gpt-4o-mini-tts. Keep the comments
# loose — exact "personality" depends on tone+speed+instructions too.
ALL_VOICES = [
    "alloy",     # balanced neutral
    "ash",       # warm baritone
    "ballad",    # mellow male
    "coral",     # bright female
    "echo",      # smooth male
    "fable",     # narrator male
    "onyx",      # deep male
    "nova",      # bright female
    "sage",      # thoughtful female
    "shimmer",   # gentle female
    "verse",     # expressive
]

# A handful of voices the user already flagged in Stage 1 (alloy/nova/sage/
# shimmer) — used for the deeper "speed / emotion / slur" sweeps so we don't
# explode the matrix while still varying the *delivery*.
CORE_VOICES = ["alloy", "nova", "sage", "shimmer", "ash", "verse"]

# Phrase text. gpt-4o-mini-tts cannot read IPA, so we provide:
#  - "plain"   : the literal target text the wake-word model will encounter
#  - "respell" : phonetic respelling that biases the model toward the
#                Stage-1 ipa_mix.json winners (v02 ə'zʊəri-un, v08 ə'zuːri-un)
PHRASES = {
    "plain":         "Hey, Azurion.",
    "respell_zure":  "Hey, a-ZURE-ree-un.",   # mirrors stem v02 əˈzʊəri…
    "respell_zoo":   "Hey, a-ZOO-ree-un.",    # mirrors stem v08 əˈzuːri…
}


@dataclass
class Clip:
    bucket: str
    voice: str
    label: str
    phrase_key: str
    text: str
    instructions: str
    speed: float
    path: Path
    notes: str = ""


# ---------------------------------------------------------------------------
# Bucket 1 — voice x tone (broad survey)
# ---------------------------------------------------------------------------
# Cover every voice across a small but well-spread tone set. This is the
# direct expansion of the original 4x3 matrix to 11x6 = 66 clips.
TONES_BROAD = [
    ("neutral",        "Speak in a neutral, clearly-articulated tone."),
    ("friendly",       "Speak warmly and naturally, as if greeting a smart-home assistant."),
    ("urgent",         "Speak quickly and with mild urgency, like calling out to get attention."),
    ("calm",           "Speak in a calm, low-energy, almost meditative tone."),
    ("cheerful",       "Speak in a bright, cheerful, upbeat tone."),
    ("conversational", "Speak casually, mid-sentence, as if talking to a friend nearby."),
]


# ---------------------------------------------------------------------------
# Bucket 2 — emotion sweep (core voices only)
# ---------------------------------------------------------------------------
# Stronger emotional colours. Some of these will be too theatrical to keep,
# but the trainer benefits from outliers.
EMOTIONS = [
    ("excited",      "Speak with genuine excitement, slightly higher pitch, energetic."),
    ("annoyed",      "Speak with mild annoyance, like the assistant just ignored you."),
    ("sleepy",       "Speak sleepily, low energy, slightly drawn-out vowels."),
    ("commanding",   "Speak with firm authority, as if issuing a command."),
    ("hesitant",     "Speak hesitantly, as if unsure whether the assistant is listening."),
    ("amused",       "Speak with quiet amusement, a hint of smile in the voice."),
    ("frustrated",   "Speak with light frustration, exhaling slightly before the word."),
    ("relieved",     "Speak with relief, exhaling, as if you just remembered the wake word."),
]


# ---------------------------------------------------------------------------
# Bucket 3 — pacing / speed (same voice, different rates)
# ---------------------------------------------------------------------------
# `speed` is a first-class OpenAI TTS knob. We pair each rate with an
# instruction that matches, so the model doesn't fight the parameter.
SPEEDS = [
    ("slow",     0.85, "Speak slowly and deliberately, every syllable separated."),
    ("normal",   1.00, "Speak at a natural pace."),
    ("brisk",    1.15, "Speak briskly, slightly faster than normal."),
    ("fast",     1.25, "Speak quickly but still clearly."),
]


# ---------------------------------------------------------------------------
# Bucket 4 — articulation / "slur" effects
# ---------------------------------------------------------------------------
# Real-world wake-word triggers are sloppy. Train on sloppy data.
ARTICULATIONS = [
    ("whisper",    "Speak in a soft whisper, barely above breath."),
    ("mumble",     "Speak with a slight mumble, lips not fully opening, as if half-asleep."),
    ("slurred",    "Speak as if mildly tipsy: slightly slurred, vowels running together, but still recognisable."),
    ("breathy",    "Speak with a breathy, airy voice, exhaling through the word."),
    ("clipped",    "Speak crisply with clipped, staccato consonants."),
    ("singsong",   "Speak in a light sing-song lilt, slight melodic rise and fall."),
    ("yawning",    "Speak as if mid-yawn, vowels stretched and lazy."),
    ("through_smile", "Speak as if smiling broadly while talking, voice slightly brighter."),
]


# ---------------------------------------------------------------------------
# Bucket 5 — accents / personae (core voices only, light touch)
# ---------------------------------------------------------------------------
# These won't be perfect (gpt-4o-mini-tts accent control is best-effort) but
# they add useful timbral variety for the trainer.
ACCENTS = [
    ("british_rp",   "Speak with a neutral British Received Pronunciation accent."),
    ("scottish",     "Speak with a soft Scottish accent."),
    ("australian",   "Speak with a relaxed Australian accent."),
    ("indian_en",    "Speak with a clear Indian English accent."),
    ("dutch_en",     "Speak English with a noticeable Dutch accent."),
    ("southern_us",  "Speak with a mild US Southern accent."),
    ("ny_us",        "Speak with a casual New York accent."),
]


# ---------------------------------------------------------------------------
# Bucket 6 — distance / environment cosplay
# ---------------------------------------------------------------------------
# Steers the *delivery* (not real acoustics — RIRs handle the room).
DISTANCES = [
    ("across_room", "Speak as if calling across a quiet living room, slightly raised volume."),
    ("close_mic",   "Speak very close to the microphone, intimate, low volume, no projection."),
    ("hands_busy",  "Speak as if your hands are full and you're calling out without looking up."),
    ("from_kitchen","Speak as if calling out from the next room while cooking."),
]


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------
def _path(bucket: str, voice: str, label: str) -> Path:
    return OUT_ROOT / bucket / f"{voice}__{label}.wav"


def build_plan() -> list[Clip]:
    plan: list[Clip] = []

    # Bucket 1: 11 voices x 6 tones x phrase=plain = 66
    for voice in ALL_VOICES:
        for tone_label, tone_instr in TONES_BROAD:
            plan.append(Clip(
                bucket="01_voice_x_tone",
                voice=voice,
                label=tone_label,
                phrase_key="plain",
                text=PHRASES["plain"],
                instructions=tone_instr,
                speed=1.0,
                path=_path("01_voice_x_tone", voice, tone_label),
            ))

    # Bucket 2: 6 core voices x 8 emotions x phrase=plain = 48
    for voice in CORE_VOICES:
        for emo_label, emo_instr in EMOTIONS:
            plan.append(Clip(
                bucket="02_emotion",
                voice=voice,
                label=emo_label,
                phrase_key="plain",
                text=PHRASES["plain"],
                instructions=emo_instr,
                speed=1.0,
                path=_path("02_emotion", voice, emo_label),
            ))

    # Bucket 3: 4 core voices x 4 speeds x phrase=plain = 16
    for voice in ["alloy", "nova", "ash", "shimmer"]:
        for speed_label, speed_val, speed_instr in SPEEDS:
            plan.append(Clip(
                bucket="03_speed",
                voice=voice,
                label=speed_label,
                phrase_key="plain",
                text=PHRASES["plain"],
                instructions=speed_instr,
                speed=speed_val,
                path=_path("03_speed", voice, speed_label),
            ))

    # Bucket 4: 6 core voices x 8 articulations x phrase=plain = 48
    for voice in CORE_VOICES:
        for art_label, art_instr in ARTICULATIONS:
            plan.append(Clip(
                bucket="04_articulation",
                voice=voice,
                label=art_label,
                phrase_key="plain",
                text=PHRASES["plain"],
                instructions=art_instr,
                speed=1.0,
                path=_path("04_articulation", voice, art_label),
            ))

    # Bucket 5: 4 voices x 7 accents x phrase=plain = 28
    for voice in ["alloy", "nova", "ash", "verse"]:
        for acc_label, acc_instr in ACCENTS:
            plan.append(Clip(
                bucket="05_accent",
                voice=voice,
                label=acc_label,
                phrase_key="plain",
                text=PHRASES["plain"],
                instructions=acc_instr,
                speed=1.0,
                path=_path("05_accent", voice, acc_label),
            ))

    # Bucket 6: 3 voices x 4 distances x phrase=plain = 12
    for voice in ["alloy", "nova", "ash"]:
        for dist_label, dist_instr in DISTANCES:
            plan.append(Clip(
                bucket="06_distance",
                voice=voice,
                label=dist_label,
                phrase_key="plain",
                text=PHRASES["plain"],
                instructions=dist_instr,
                speed=1.0,
                path=_path("06_distance", voice, dist_label),
            ))

    # Bucket 7: respelling check — top voices x 2 respellings x neutral tone
    # Validates whether gpt-4o-mini-tts honours phonetic respelling.
    for voice in ["alloy", "nova", "sage", "shimmer", "ash", "verse"]:
        for ph_key in ("respell_zure", "respell_zoo"):
            plan.append(Clip(
                bucket="07_respell",
                voice=voice,
                label=ph_key,
                phrase_key=ph_key,
                text=PHRASES[ph_key],
                instructions="Speak naturally, as if it were a real word, "
                             "and pronounce it exactly as spelled with the hyphens "
                             "indicating syllable boundaries and capitals indicating stress.",
                speed=1.0,
                path=_path("07_respell", voice, ph_key),
            ))

    return plan


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(plan: list[Clip]) -> int:
    client = make_azure_openai_tts_client(timeout=60)
    fails = 0
    current_bucket = None
    for clip in plan:
        if clip.bucket != current_bucket:
            header(f"Bucket {clip.bucket}")
            current_bucket = clip.bucket
        label = f"{clip.bucket}/{clip.voice}/{clip.label}"
        try:
            resp = client.audio.speech.create(
                model=AZURE_OPENAI_TTS_DEPLOYMENT,
                input=clip.text,
                voice=clip.voice,
                response_format="wav",
                speed=clip.speed,
                instructions=clip.instructions,
            )
            audio = resp.read()
            if not audio or len(audio) < 1024:
                fail(label, f"audio too small ({len(audio)} bytes)")
                fails += 1
                continue
            clip.path.parent.mkdir(parents=True, exist_ok=True)
            clip.path.write_bytes(audio)
            ok(label, f"{clip.path.relative_to(OUT_ROOT)} ({len(audio)} bytes)")
        except Exception as e:  # noqa: BLE001
            fail(label, str(e))
            fails += 1
    return fails


def write_manifest(plan: list[Clip]) -> Path:
    manifest = OUT_ROOT / "manifest.csv"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "bucket", "voice", "label", "phrase_key", "speed",
            "text", "instructions",
            "relative_path", "exists",
            "verdict (good|ok|bad)", "notes",
        ])
        for c in plan:
            rel = (
                str(c.path.relative_to(OUT_ROOT))
                if c.path.is_relative_to(OUT_ROOT) else str(c.path)
            )
            w.writerow([
                c.bucket, c.voice, c.label, c.phrase_key, c.speed,
                c.text, c.instructions,
                rel, "yes" if c.path.exists() else "no",
                "", c.notes,
            ])
    return manifest


def main() -> int:
    plan = build_plan()
    header(f"gpt-4o-mini-tts variants — {len(plan)} clips queued")
    by_bucket: dict[str, int] = {}
    for c in plan:
        by_bucket[c.bucket] = by_bucket.get(c.bucket, 0) + 1
    for b, n in by_bucket.items():
        print(f"  {b:20s} {n:3d}")

    fails = run(plan)
    manifest = write_manifest(plan)

    header("done")
    wrote = sum(1 for c in plan if c.path.exists())
    print(f"  wrote {wrote}/{len(plan)} clips")
    print(f"  base dir: {OUT_ROOT}")
    print(f"  manifest: {manifest}")
    if fails:
        print(f"  WARNING: {fails} clip(s) failed — see logs above.")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
