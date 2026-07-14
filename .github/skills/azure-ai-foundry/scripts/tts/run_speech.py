"""Native Azure Cognitive Services Speech — direct SSML to the Speech REST API.

The proxy version of this script JSON-posts to the tts-sidecar, which
builds SSML server-side. Here we build the SSML on the client and POST
it directly to::

    https://<region>.tts.speech.microsoft.com/cognitiveservices/v1

with ``Ocp-Apim-Subscription-Key`` and ``Content-Type: application/ssml+xml``.

Exercises the same three patterns as the proxy sample so you can A/B
the audio output:
  1. Plain text
  2. IPA phonetic override via ``<phoneme>``
  3. Rich SSML with ``<prosody>``, ``<break>``, ``<say-as>``, ``<phoneme>``

References
----------
* Azure Speech REST API — text-to-speech (endpoint, headers,
  ``X-Microsoft-OutputFormat`` values, region hostnames):
  https://learn.microsoft.com/azure/ai-services/speech-service/rest-text-to-speech
* Speech Synthesis Markup Language (SSML) reference — every element used
  here (``voice``, ``prosody``, ``break``, ``say-as``, ``phoneme``):
  https://learn.microsoft.com/azure/ai-services/speech-service/speech-synthesis-markup
* Neural voice gallery (voice names like ``en-US-JennyNeural``):
  https://learn.microsoft.com/azure/ai-services/speech-service/language-support?tabs=tts
"""

import sys
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    SSL_VERIFY,
    fail,
    header,
    ok,
)

VOICE = "en-US-JennyNeural"
LANG = "en-US"
OUTPUT_FORMAT = "audio-24khz-160kbitrate-mono-mp3"
HERE = Path(__file__).resolve().parent

OUT = HERE / "out.mp3"
OUT_PHONETIC = HERE / "out_phonetic.mp3"
OUT_ADVANCED = HERE / "out_advanced.mp3"

TEXT = "Hello from the AI router native samples. This audio was synthesised by Azure Cognitive Services Speech."
PHONETIC_TEXT = "Hello, Azure!"
PHONETIC_IPA = "hɛloʊ ˈæʒər"

# Same JsonML payload as the proxy sample, but serialised here.
ADVANCED_SSML = [
    "prosody", {"rate": "slow"},
    "Welcome. ",
    ["break", {"time": "500ms"}],
    "Today is ",
    ["say-as", {"interpret-as": "date", "format": "dmy"}, "11/05/2026"],
    ". ",
    ["break", {"time": "300ms"}],
    "Let me say hello: ",
    ["phoneme", {"alphabet": "ipa", "ph": "hɛloʊ"}, "hello"],
    "!",
]


def jsonml_to_xml(node) -> str:
    """JsonML → XML serialiser (mirrors tts-sidecar/app.py)."""
    if isinstance(node, str):
        return escape(node)
    if not isinstance(node, list) or not node:
        raise ValueError(f"invalid JsonML node: {node!r}")
    tag = node[0]
    attrs: dict = {}
    children_start = 1
    if len(node) > 1 and isinstance(node[1], dict):
        attrs = node[1]
        children_start = 2
    attr_str = "".join(f" {k}={quoteattr(str(v))}" for k, v in attrs.items())
    children = node[children_start:]
    if not children:
        return f"<{tag}{attr_str}/>"
    inner = "".join(jsonml_to_xml(c) for c in children)
    return f"<{tag}{attr_str}>{inner}</{tag}>"


def _wrap_ssml(inner_xml: str) -> str:
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xml:lang="{escape(LANG)}">'
        f'<voice name="{escape(VOICE)}">{inner_xml}</voice>'
        f'</speak>'
    )


def _synth(client: httpx.Client, label: str, ssml: str, out_path: Path) -> bool:
    try:
        r = client.post(
            "/cognitiveservices/v1",
            content=ssml.encode("utf-8"),
            headers={"Content-Type": "application/ssml+xml"},
        )
        if r.status_code >= 400:
            fail(label, f"HTTP {r.status_code}: {r.text[:200]}")
            return False
        audio = r.content
        if not audio or len(audio) < 1024:
            fail(label, f"audio too small: {len(audio)} bytes")
            return False
        out_path.write_bytes(audio)
        ok(label, f"wrote {out_path.name} ({len(audio)} bytes)")
        return True
    except Exception as e:  # noqa: BLE001
        fail(label, str(e))
        return False


def main() -> int:
    if not AZURE_SPEECH_KEY:
        fail("azure-speech", "AZURE_SPEECH_KEY is not set")
        return 1

    header(f"tts (native Azure Speech) — {VOICE}, region={AZURE_SPEECH_REGION}")

    base_url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
        "User-Agent": "ai-router-native-sample",
    }
    all_ok = True
    with httpx.Client(base_url=base_url, headers=headers, timeout=60, verify=SSL_VERIFY) as client:
        # 1. Plain text
        if not _synth(client, VOICE, _wrap_ssml(escape(TEXT)), OUT):
            all_ok = False

        # 2. IPA phonetics
        phoneme_xml = (
            f'<phoneme alphabet="ipa" ph="{escape(PHONETIC_IPA)}">'
            f"{escape(PHONETIC_TEXT)}</phoneme>"
        )
        if not _synth(client, f"{VOICE} (IPA)", _wrap_ssml(phoneme_xml), OUT_PHONETIC):
            all_ok = False

        # 3. Full SSML via JsonML
        if not _synth(client, f"{VOICE} (JsonML)",
                      _wrap_ssml(jsonml_to_xml(ADVANCED_SSML)), OUT_ADVANCED):
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
