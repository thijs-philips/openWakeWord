"""Round 3: lock in the IPA mix recipe for Stage 2 bulk generation.

User decisions so far:

  Endings  (final vowel before 'n'):
    un  = best, dominant
    in  = some, secondary
    uhn = some, secondary
    on  = even some, sprinkle

  Stems  (start of word, up to '-ree-'):
    v02 əˈzʊəri  = best  (a-ZURE-ree-)
    v08 əˈzuːri  = best  (a-ZOO-ree-)
    v07 əˈzɜːri  = mid   (a-ZURR-ree-)  — round-1 'fine'
    v09 əˈzʊri   = mid   (a-ZUR-ree-, short vowel) — round-1 'fine'

This script does TWO things:

  1. Generates ALL stem x ending combinations (4 x 4 = 16 clips) for
     en-US-AndrewNeural / "Hey Azurion." so we can sanity-check what
     the bulk run will actually produce.

  2. Writes  ./ipa_mix.json  — the weighted recipe consumed by the
     Stage 2 bulk generator. Each (stem, ending) pair has a sampling
     weight = stem_weight x ending_weight.

Tier weights (10 / 5 / 1):

         |  un (10) | in (5) | uhn (5) | on (1) |
  v02 10 |   100    |   50   |   50    |   10   |
  v08 10 |   100    |   50   |   50    |   10   |
  v07  1 |    10    |    5   |    5    |    1   |
  v09  1 |    10    |    5   |    5    |    1   |

Totals:  un=220 (50.7%), in=110 (25.3%), uhn=110 (25.3%), on=22 (5.1%)
         v02=210 (48.4%), v08=210 (48.4%), v07=22 (5.1%), v09=22 (5.1%)

(Numbers add to 434; probabilities are weight/434.)
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / ".github" / "skills"
                       / "azure-ai-foundry" / "scripts"))
from common import (  # noqa: E402
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    SSL_VERIFY,
    fail,
    header,
    ok,
)

VOICE = "en-US-AndrewNeural"
LANG = "en-US"
PHRASE_PREFIX = "Hey "
PHRASE_SUFFIX = "."
OUT_DIR = HERE / "out_ipa_mix_preview"
MIX_FILE = HERE / "ipa_mix.json"
OUTPUT_FORMAT = "riff-16khz-16bit-mono-pcm"


@dataclass
class Stem:
    code: str
    ipa: str     # full stem up to and including 'i' (the "-ree-")
    weight: int  # tier weight
    label: str

@dataclass
class Ending:
    code: str
    ipa: str     # vowel + 'n'
    weight: int
    label: str

STEMS = [
    Stem("v02", "əˈzʊəri", 10, "a-ZURE-ree-  (z + CURE)"),
    Stem("v08", "əˈzuːri",  10, "a-ZOO-ree-   (z + GOOSE)"),
    Stem("v07", "əˈzɜːri",  1,  "a-ZURR-ree-  (z + NURSE)"),
    Stem("v09", "əˈzʊri",   1,  "a-ZUR-ree-   (z + short FOOT)"),
]

ENDINGS = [
    Ending("un",  "ən",  10, "'-un'  (schwa, dominant)"),
    Ending("in",  "ɪn",  5,  "'-in'  (KIT vowel)"),
    Ending("uhn", "ʌn",  5,  "'-uhn' (STRUT vowel)"),
    Ending("on",  "ɒn",  1,  "'-on'  (LOT vowel, sprinkle)"),
]


def _ssml(ipa: str) -> str:
    inner = (
        f'{escape(PHRASE_PREFIX)}'
        f'<phoneme alphabet="ipa" ph={quoteattr(ipa)}>Azurion</phoneme>'
        f'{escape(PHRASE_SUFFIX)}'
    )
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xml:lang="{escape(LANG)}">'
        f'<voice name="{escape(VOICE)}">{inner}</voice>'
        f'</speak>'
    )


def _client() -> httpx.Client:
    if not AZURE_SPEECH_KEY:
        print("ERROR: AZURE_SPEECH_KEY is not set.", file=sys.stderr)
        sys.exit(2)
    return httpx.Client(
        base_url=f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com",
        headers={
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
            "User-Agent": "openwakeword-azurion-ipa-mix",
        },
        timeout=60.0,
        verify=SSL_VERIFY,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header(f"IPA MIX preview — {VOICE}, {len(STEMS)}x{len(ENDINGS)} = "
           f"{len(STEMS) * len(ENDINGS)} clips")

    rows: list[dict] = []
    mix_entries: list[dict] = []
    total_weight = 0
    fails = 0

    with _client() as c:
        for s in STEMS:
            for e in ENDINGS:
                ipa = s.ipa + e.ipa
                vid = f"{s.code}__{e.code}"
                fname = f"{vid}.wav"
                w = s.weight * e.weight
                total_weight += w
                desc = f"{s.label}  +  {e.label}  (weight={w})"
                try:
                    r = c.post("/cognitiveservices/v1",
                               content=_ssml(ipa).encode("utf-8"))
                    if r.status_code >= 400:
                        fail(vid, f"HTTP {r.status_code}: {r.text[:200]}")
                        fails += 1
                    elif not r.content or len(r.content) < 1024:
                        fail(vid, f"audio too small ({len(r.content)} bytes)")
                        fails += 1
                    else:
                        (OUT_DIR / fname).write_bytes(r.content)
                        ok(vid, f"{fname} w={w} — {desc}")
                except Exception as ex:  # noqa: BLE001
                    fail(vid, str(ex))
                    fails += 1

                rows.append({
                    "id": vid, "stem": s.code, "ending": e.code, "ipa": ipa,
                    "stem_weight": s.weight, "end_weight": e.weight,
                    "combined_weight": w, "file": fname,
                    "description": desc,
                })
                mix_entries.append({
                    "id": vid, "stem": s.code, "ending": e.code,
                    "ipa": ipa, "weight": w,
                })

    # Normalize for the JSON recipe
    for entry in mix_entries:
        entry["probability"] = entry["weight"] / total_weight

    mix = {
        "voice_hint": VOICE,
        "phrase_template": f"{PHRASE_PREFIX}<phoneme ipa='...'>Azurion</phoneme>{PHRASE_SUFFIX}",
        "total_weight": total_weight,
        "stems": [{"code": s.code, "ipa": s.ipa, "weight": s.weight, "label": s.label}
                  for s in STEMS],
        "endings": [{"code": e.code, "ipa": e.ipa, "weight": e.weight, "label": e.label}
                    for e in ENDINGS],
        "combinations": mix_entries,
    }
    MIX_FILE.write_text(json.dumps(mix, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = OUT_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "stem", "ending", "ipa",
                                          "stem_weight", "end_weight",
                                          "combined_weight", "file",
                                          "description"])
        w.writeheader()
        w.writerows(rows)

    header("done")
    print(f"  wrote {len(rows) - fails}/{len(rows)} preview clips")
    print(f"  preview folder: {OUT_DIR}")
    print(f"  manifest:       {manifest}")
    print(f"  mix recipe:     {MIX_FILE}")
    print()
    print("  expected distribution (per 1000 clips at this mix):")
    for entry in sorted(mix_entries, key=lambda x: -x["weight"]):
        n = round(entry["probability"] * 1000)
        print(f"    {entry['id']:<14} {n:>4}  ({entry['probability'] * 100:5.2f}%)  ipa={entry['ipa']}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
