"""Stage 2 — bulk training-clip generation for "Hey, Azurion".

Produces the four directories that `python -m openwakeword.train` expects:
    tts/positive_train/   target phrase, training split
    tts/positive_test/    target phrase, held-out split
    tts/negative_train/   adversarial / near-miss phrases, training split
    tts/negative_test/    adversarial / near-miss phrases, held-out split

All clips are 16 kHz mono 16-bit PCM WAV — the native format consumed by
openWakeWord, so no resampling at train time. Azure Speech delivers that
directly via `riff-16khz-16bit-mono-pcm`; gpt-4o-mini-tts returns 24 kHz
WAV which we polyphase-resample to 16 kHz before writing.

VOICE MIX (per positive clip)
-----------------------------
A weighted recipe pool is sampled per clip. Each recipe is a
(`engine`, `voice`, `phrase_strategy`, `delivery`) tuple. Weights below
are illustrative; see `RECIPES` at the bottom of the file for the source
of truth.

    Azure Speech en-US, IPA forced via <phoneme ph="...">   ~70%
        (16 IPA combinations sampled from ipa_mix.json,
         crossed with ~15 en-US neural voices and 3 prosody rates)
    Azure Speech en-US, plain text "Hey, Azurion."          ~10%
    Azure Speech en-US, with mstts:express-as styles        ~5%
        (cheerful/friendly/whispering — only on voices that support it)
    Azure Speech nl-NL, plain                               ~5%
    gpt-4o-mini-tts, full voice/tone/articulation matrix    ~10%

NEGATIVES
---------
A hand-curated seed list of ~80 adversarial phrases (other wake words,
phonetic neighbours of Azurion, sentences that contain Azurion in a
non-trigger context) is sampled and synthesized through the same voice
mix. This is more reliable than pulling DeepPhonemizer at runtime, and
the trainer additionally consumes 2000 h of ACAV100M speech as bulk
negatives — so the burden on this list is small.

RESUMABILITY
------------
Filenames are deterministic (`{idx:06d}_*.wav`). Re-running picks up
where the previous run left off by counting existing files in each
output dir.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Callable, Iterable
from xml.sax.saxutils import escape, quoteattr

import httpx
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import soundfile as sf

# --- plumb the azure-ai-foundry skill helpers ------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SKILL_SCRIPTS = REPO_ROOT / ".github" / "skills" / "azure-ai-foundry" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from common import (  # noqa: E402
    AZURE_OPENAI_TTS_DEPLOYMENT,
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    SSL_VERIFY,
    make_azure_openai_tts_client,
)

TARGET_SR = 16_000
OUT_ROOT = HERE / "tts"
SPLITS = ("positive_train", "positive_test", "negative_train", "negative_test")

# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------
# A curated en-US neural roster. Kept broad on purpose — diverse timbre is
# the biggest single contributor to generalisation in wake-word models.
EN_US_VOICES = [
    "en-US-AvaNeural", "en-US-AndrewNeural", "en-US-AriaNeural",
    "en-US-GuyNeural", "en-US-JennyMultilingualNeural", "en-US-DavisNeural",
    "en-US-AmberNeural", "en-US-AnaNeural", "en-US-AshleyNeural",
    "en-US-BrandonNeural", "en-US-ChristopherNeural", "en-US-CoraNeural",
    "en-US-ElizabethNeural", "en-US-EricNeural", "en-US-JacobNeural",
    "en-US-MichelleNeural", "en-US-MonicaNeural", "en-US-RogerNeural",
    "en-US-SaraNeural", "en-US-SteffanNeural", "en-US-TonyNeural",
    "en-US-NancyNeural", "en-US-JaneNeural",
]

# Voices known to support mstts:express-as styles.
EXPRESS_VOICES = {
    "en-US-AriaNeural":  ["cheerful", "friendly", "whispering", "newscast-casual"],
    "en-US-JennyNeural": ["cheerful", "friendly", "whispering", "assistant"],
    "en-US-GuyNeural":   ["cheerful", "newscast"],
    "en-US-DavisNeural": ["chat", "cheerful", "whispering"],
    "en-US-SaraNeural":  ["cheerful", "whispering", "shouting"],
    "en-US-TonyNeural":  ["cheerful", "sad", "angry"],
}

NL_NL_VOICES = ["nl-NL-FennaNeural", "nl-NL-MaartenNeural", "nl-NL-ColetteNeural"]

GPT_TTS_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable",
                  "onyx", "nova", "sage", "shimmer", "verse"]

GPT_TTS_INSTRUCTIONS = [
    "Speak in a neutral, clearly-articulated tone.",
    "Speak warmly and naturally, as if greeting a smart-home assistant.",
    "Speak quickly and with mild urgency, like calling out to get attention.",
    "Speak in a calm, low-energy tone.",
    "Speak in a bright, cheerful, upbeat tone.",
    "Speak casually, mid-sentence, as if talking to a friend nearby.",
    "Speak with a slight mumble, lips not fully opening, as if half-asleep.",
    "Speak in a soft whisper, barely above breath.",
    "Speak with a breathy, airy voice.",
    "Speak as if calling out from the next room.",
    "Speak with a noticeable Dutch accent.",
    "Speak with a neutral British accent.",
    "Speak with quiet excitement.",
    "Speak with mild annoyance.",
]
GPT_TTS_SPEEDS = [0.9, 1.0, 1.1, 1.2]

PROSODY_RATES = ["-15%", "-5%", "0%", "+5%", "+15%"]

# ---------------------------------------------------------------------------
# Adversarial seed phrases — hand-curated near-misses for "Hey Azurion"
# ---------------------------------------------------------------------------
ADVERSARIAL_SEEDS = [
    # Other commercial wake words (anti-collision — these MUST not trigger us)
    "Hey Alexa", "Hey Google", "Hey Siri", "Hey Portal", "Hey Cortana",
    "OK Google", "Computer", "Alexa", "Hey Mycroft", "Hey Jarvis",
    "Hey Amazon", "Hey Echo",
    # Phonetic neighbours of "Azurion"
    "Hey azure", "Hey arion", "Hey marion", "Hey orion", "Hey junior",
    "Hey aurora", "Hey ozarion", "Hey ozarian", "Hey aurion", "Hey azora",
    "Hey azaria", "Hey azoria", "Hey adrian", "Hey asuncion",
    "Hey osiris", "Hey valerian", "Hey decurion",
    # Same target word but in non-trigger linguistic context — the model
    # should *not* fire when the word appears mid-sentence without "Hey"
    "Azurion is offline", "Open azurion", "Close azurion",
    "About azurion", "Talk about azurion later", "Where is azurion",
    "Azurion update", "Azurion configuration", "Restart azurion",
    "Azurion documentation", "The azurion system",
    "An azurion procedure", "Azurion image guided therapy",
    # Same prefix "Hey" + non-target word — model must learn that "hey" alone
    # is not enough
    "Hey there", "Hey listen", "Hey buddy", "Hey doctor",
    "Hey nurse", "Hey machine", "Hey computer system",
    "Hey can you help me", "Hey is anyone there", "Hey what time is it",
    "Hey turn the lights on", "Hey play some music", "Hey set a timer",
    # "Azurion" without "Hey" prefix — partial match suppression
    "Azurion", "An azurion", "The new azurion", "Buying an azurion",
    "Setting up azurion", "Login to azurion",
    # Generic noisy English
    "Good morning", "How are you doing today", "What's the weather",
    "Can you hear me", "Please call me back", "I need some help",
    "Turn off the lights", "Lock the door", "Start the coffee maker",
    "Open the garage", "Pause the music", "Skip this track",
    "What's on my calendar", "Read my messages",
]

# ---------------------------------------------------------------------------
# Recipe definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Recipe:
    code: str           # short tag used in filenames + manifest
    engine: str         # "azure_speech" | "gpt_tts"
    weight: int         # relative weight in the recipe pool

@dataclass
class Job:
    idx: int
    split: str
    recipe: Recipe
    text: str           # raw phrase (used for negatives + plain-mode positives)
    voice: str | None = None     # resolved at sample time
    delivery: dict = field(default_factory=dict)
    out_path: Path = field(default_factory=Path)


RECIPES: list[Recipe] = [
    Recipe("az_ipa",        "azure_speech", weight=70),
    Recipe("az_plain",      "azure_speech", weight=10),
    Recipe("az_express",    "azure_speech", weight=5),
    Recipe("az_nl",         "azure_speech", weight=5),
    Recipe("gpt_tts",       "gpt_tts",      weight=10),
]
RECIPE_BY_CODE = {r.code: r for r in RECIPES}

# ---------------------------------------------------------------------------
# IPA mix loader
# ---------------------------------------------------------------------------
def load_ipa_mix() -> tuple[list[str], list[float]]:
    """Return (ipa_strings, probs) for weighted random sampling."""
    data = json.loads((HERE / "ipa_mix.json").read_text(encoding="utf-8"))
    ipas, weights = [], []
    for combo in data["combinations"]:
        ipas.append(combo["ipa"])
        weights.append(combo["weight"])
    total = sum(weights)
    probs = [w / total for w in weights]
    return ipas, probs


# ---------------------------------------------------------------------------
# Azure Speech engine
# ---------------------------------------------------------------------------
def make_speech_client(timeout: float = 60.0) -> httpx.Client:
    if not AZURE_SPEECH_KEY:
        raise RuntimeError("AZURE_SPEECH_KEY missing from .env")
    return httpx.Client(
        base_url=f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com",
        headers={
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            "User-Agent": "openwakeword-hey-azurion-bulk-gen",
        },
        timeout=timeout,
        verify=SSL_VERIFY,
    )


def _ssml_plain(voice: str, lang: str, text: str, rate: str | None = None) -> str:
    inner = escape(text)
    if rate and rate != "0%":
        inner = f'<prosody rate="{escape(rate)}">{inner}</prosody>'
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xml:lang="{escape(lang)}">'
        f'<voice name="{escape(voice)}">{inner}</voice>'
        f'</speak>'
    )


def _ssml_ipa_azurion(voice: str, lang: str, ipa: str, rate: str | None = None) -> str:
    """SSML for the positive phrase 'Hey, Azurion.' with IPA forced."""
    phoneme = (
        f'<phoneme alphabet="ipa" ph={quoteattr(ipa)}>Azurion</phoneme>'
    )
    inner = f"Hey, {phoneme}."
    if rate and rate != "0%":
        inner = f'<prosody rate="{escape(rate)}">{inner}</prosody>'
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xml:lang="{escape(lang)}">'
        f'<voice name="{escape(voice)}">{inner}</voice>'
        f'</speak>'
    )


def _ssml_express(voice: str, lang: str, style: str, text: str) -> str:
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{escape(lang)}">'
        f'<voice name="{escape(voice)}">'
        f'<mstts:express-as style="{escape(style)}">{escape(text)}</mstts:express-as>'
        f'</voice></speak>'
    )


def azure_post_ssml(client: httpx.Client, ssml: str) -> bytes:
    r = client.post("/cognitiveservices/v1", content=ssml.encode("utf-8"))
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    if not r.content or len(r.content) < 1024:
        raise RuntimeError(f"audio too small ({len(r.content)} bytes)")
    return r.content


# ---------------------------------------------------------------------------
# gpt-4o-mini-tts engine
# ---------------------------------------------------------------------------
def gpt_tts_synth(client, voice: str, text: str, speed: float, instructions: str) -> bytes:
    """Return raw 16 kHz mono PCM WAV bytes."""
    resp = client.audio.speech.create(
        model=AZURE_OPENAI_TTS_DEPLOYMENT,
        input=text,
        voice=voice,
        response_format="wav",
        speed=speed,
        instructions=instructions,
    )
    raw = resp.read()
    return _to_16k_mono_wav_bytes(raw)


def _to_16k_mono_wav_bytes(wav_bytes: bytes) -> bytes:
    arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        up, down = TARGET_SR // g, int(sr) // g
        arr = scipy.signal.resample_poly(arr, up, down)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / max(peak, 1e-6) * 0.95
    pcm = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, TARGET_SR, pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Recipe resolution — turns a recipe code + (split, text) into Azure/GPT call
# ---------------------------------------------------------------------------
def resolve_job(job: Job, ipa_pool: tuple[list[str], list[float]], rng: random.Random) -> None:
    """Fill in job.voice and job.delivery based on the recipe."""
    code = job.recipe.code
    if code == "az_ipa":
        job.voice = rng.choice(EN_US_VOICES)
        job.delivery = {
            "kind": "ipa",
            "ipa": rng.choices(ipa_pool[0], weights=ipa_pool[1], k=1)[0],
            "rate": rng.choice(PROSODY_RATES),
        }
    elif code == "az_plain":
        job.voice = rng.choice(EN_US_VOICES)
        job.delivery = {
            "kind": "plain",
            "lang": "en-US",
            "rate": rng.choice(PROSODY_RATES),
        }
    elif code == "az_express":
        voice = rng.choice(list(EXPRESS_VOICES.keys()))
        job.voice = voice
        job.delivery = {
            "kind": "express",
            "lang": "en-US",
            "style": rng.choice(EXPRESS_VOICES[voice]),
        }
    elif code == "az_nl":
        job.voice = rng.choice(NL_NL_VOICES)
        job.delivery = {"kind": "plain", "lang": "nl-NL", "rate": "0%"}
    elif code == "gpt_tts":
        job.voice = rng.choice(GPT_TTS_VOICES)
        job.delivery = {
            "kind": "gpt",
            "speed": rng.choice(GPT_TTS_SPEEDS),
            "instructions": rng.choice(GPT_TTS_INSTRUCTIONS),
        }
    else:
        raise ValueError(f"unknown recipe code {code}")


def synth_job(job: Job, speech_client: httpx.Client, gpt_client) -> bytes:
    d = job.delivery
    kind = d["kind"]
    if kind == "ipa":
        # job.text is the canonical "Hey, Azurion." for positives
        ssml = _ssml_ipa_azurion(job.voice, "en-US", d["ipa"], d.get("rate"))
        return azure_post_ssml(speech_client, ssml)
    if kind == "plain":
        ssml = _ssml_plain(job.voice, d["lang"], job.text, d.get("rate"))
        return azure_post_ssml(speech_client, ssml)
    if kind == "express":
        ssml = _ssml_express(job.voice, d["lang"], d["style"], job.text)
        return azure_post_ssml(speech_client, ssml)
    if kind == "gpt":
        return gpt_tts_synth(gpt_client, job.voice, job.text,
                             d["speed"], d["instructions"])
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def sample_recipe(rng: random.Random, exclude_ipa: bool) -> Recipe:
    """For negatives we never use the IPA-forced positive template — drop it."""
    pool = [r for r in RECIPES if not (exclude_ipa and r.code == "az_ipa")]
    weights = [r.weight for r in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def build_plan(targets: dict[str, int], rng: random.Random) -> list[Job]:
    ipa_pool = load_ipa_mix()
    jobs: list[Job] = []
    # Resume by counting existing files per split
    for split, n_target in targets.items():
        split_dir = OUT_ROOT / split
        split_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(split_dir.glob("*.wav"))
        n_existing = len(existing)
        if n_existing >= n_target:
            print(f"[plan] {split:<16} already has {n_existing}/{n_target} — skipping")
            continue
        is_neg = split.startswith("negative")
        n_seeds = len(ADVERSARIAL_SEEDS)
        for idx in range(n_existing, n_target):
            recipe = sample_recipe(rng, exclude_ipa=is_neg)
            text = (
                ADVERSARIAL_SEEDS[idx % n_seeds] if is_neg
                else "Hey, Azurion."
            )
            job = Job(idx=idx, split=split, recipe=recipe, text=text)
            resolve_job(job, ipa_pool, rng)
            voice_tag = (job.voice or "").replace("-", "")[:18] or "x"
            job.out_path = split_dir / (
                f"{idx:06d}_{recipe.code}_{voice_tag}.wav"
            )
            jobs.append(job)
        print(f"[plan] {split:<16} queued {n_target - n_existing} new jobs"
              f" (existing {n_existing}, target {n_target})")
    return jobs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(jobs: list[Job], workers_az: int, workers_gpt: int) -> tuple[int, int]:
    speech_clients: list[httpx.Client] = [
        make_speech_client() for _ in range(workers_az)
    ]
    gpt_client = make_azure_openai_tts_client(timeout=60) if workers_gpt > 0 else None

    az_jobs = [j for j in jobs if j.recipe.engine == "azure_speech"]
    gpt_jobs = [j for j in jobs if j.recipe.engine == "gpt_tts"]

    print(f"[run] {len(az_jobs)} Azure Speech jobs, {len(gpt_jobs)} gpt-tts jobs")
    print(f"[run] workers: azure={workers_az}, gpt={workers_gpt}")

    ok = 0
    bad = 0
    t0 = time.time()

    def _do_az(idx_job):
        i, job = idx_job
        client = speech_clients[i % workers_az]
        try:
            audio = synth_job(job, client, None)
            job.out_path.write_bytes(audio)
            return True, job, None
        except Exception as e:  # noqa: BLE001
            return False, job, str(e)

    def _do_gpt(job):
        try:
            audio = synth_job(job, None, gpt_client)
            job.out_path.write_bytes(audio)
            return True, job, None
        except Exception as e:  # noqa: BLE001
            return False, job, str(e)

    # Azure pool
    if az_jobs:
        with ThreadPoolExecutor(max_workers=workers_az) as ex:
            futs = [ex.submit(_do_az, (i, j)) for i, j in enumerate(az_jobs)]
            for n, fut in enumerate(as_completed(futs), 1):
                success, job, err = fut.result()
                if success:
                    ok += 1
                else:
                    bad += 1
                    print(f"  [az fail] {job.out_path.name}: {err}", file=sys.stderr)
                if n % 200 == 0:
                    rate = n / max(time.time() - t0, 1e-3)
                    print(f"  [az] {n}/{len(az_jobs)}  ok={ok}  bad={bad}"
                          f"  rate={rate:.1f}/s")

    # gpt-tts pool (lower concurrency to respect rate limits)
    if gpt_jobs:
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=workers_gpt) as ex:
            futs = [ex.submit(_do_gpt, j) for j in gpt_jobs]
            for n, fut in enumerate(as_completed(futs), 1):
                success, job, err = fut.result()
                if success:
                    ok += 1
                else:
                    bad += 1
                    print(f"  [gpt fail] {job.out_path.name}: {err}", file=sys.stderr)
                if n % 50 == 0:
                    rate = n / max(time.time() - t1, 1e-3)
                    print(f"  [gpt] {n}/{len(gpt_jobs)}  ok={ok}  bad={bad}"
                          f"  rate={rate:.1f}/s")

    for c in speech_clients:
        c.close()
    return ok, bad


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(jobs: list[Job]) -> Path:
    manifest = OUT_ROOT / "manifest.csv"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["split", "idx", "recipe", "voice", "text",
                        "delivery_json", "path", "exists"])
        for j in jobs:
            w.writerow([
                j.split, j.idx, j.recipe.code, j.voice or "", j.text,
                json.dumps(j.delivery, ensure_ascii=False),
                str(j.out_path.relative_to(OUT_ROOT)),
                "yes" if j.out_path.exists() else "no",
            ])
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-train", type=int, default=5000,
                   help="positive_train target (default 5000)")
    p.add_argument("--n-test",  type=int, default=500,
                   help="positive_test target (default 500)")
    p.add_argument("--n-neg-train", type=int, default=5000,
                   help="negative_train target (default 5000)")
    p.add_argument("--n-neg-test", type=int, default=500,
                   help="negative_test target (default 500)")
    p.add_argument("--workers-az", type=int, default=8,
                   help="concurrent Azure Speech requests (default 8)")
    p.add_argument("--workers-gpt", type=int, default=2,
                   help="concurrent gpt-4o-mini-tts requests (default 2)")
    p.add_argument("--seed", type=int, default=20260519,
                   help="random seed for recipe sampling")
    p.add_argument("--dry-run", action="store_true",
                   help="plan only — print counts and exit")
    args = p.parse_args()

    targets = {
        "positive_train": args.n_train,
        "positive_test":  args.n_test,
        "negative_train": args.n_neg_train,
        "negative_test":  args.n_neg_test,
    }
    rng = random.Random(args.seed)

    print("=" * 72)
    print("Stage 2 — bulk training-clip generation for 'Hey, Azurion'")
    print("=" * 72)
    print(f"  output root: {OUT_ROOT}")
    for k, v in targets.items():
        print(f"  target {k:<16} {v}")

    jobs = build_plan(targets, rng)
    if not jobs:
        print("nothing to do — all splits at target.")
        return 0

    # Distribution sanity report
    by_recipe = {}
    for j in jobs:
        by_recipe[j.recipe.code] = by_recipe.get(j.recipe.code, 0) + 1
    print("\n[plan] recipe distribution across queued jobs:")
    for code, n in sorted(by_recipe.items(), key=lambda x: -x[1]):
        print(f"  {code:<12} {n:6d}  ({n / len(jobs) * 100:.1f}%)")

    if args.dry_run:
        print("\n--dry-run: no synthesis performed.")
        return 0

    t0 = time.time()
    ok, bad = run(jobs, args.workers_az, args.workers_gpt)
    elapsed = time.time() - t0

    write_manifest(jobs)
    print("\n" + "=" * 72)
    print(f"done in {elapsed/60:.1f} min — ok={ok} bad={bad}")
    print(f"manifest: {OUT_ROOT / 'manifest.csv'}")
    print("=" * 72)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
