"""Native gpt-4o-mini-tts — direct against the East US 2 Azure OpenAI resource.

Uses the ``AzureOpenAI`` SDK pointed at ``iexigtsazurioneye-openai`` with
the dedicated ``AZURE_API_KEY_TTS``. The deployment name on that
resource is ``gpt-4o-mini-tts-2`` (override via ``AZURE_OPENAI_TTS_DEPLOYMENT``).

References
----------
* Azure OpenAI text-to-speech quickstart (deployment names, supported
  voices and response formats, regional availability):
  https://learn.microsoft.com/azure/ai-services/openai/text-to-speech-quickstart
* OpenAI text-to-speech guide (voices, ``response_format``, streaming):
  https://platform.openai.com/docs/guides/text-to-speech
* OpenAI Python SDK — ``client.audio.speech.create`` (incl. ``.read()``
  vs ``.iter_bytes()`` for streaming):
  https://github.com/openai/openai-python#audio
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_OPENAI_TTS_DEPLOYMENT,
    fail,
    header,
    make_azure_openai_tts_client,
    ok,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_openai.mp3"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Test gpt-4o-mini-tts (native Azure OpenAI)")
    parser.add_argument("--voice", default="alloy", help="Voice name (default: alloy)")
    parser.add_argument(
        "--text",
        default="Hello from the AI router native samples. This is gpt-4o-mini-tts speaking directly to Azure OpenAI.",
        help="Text to speak",
    )
    args = parser.parse_args()

    client = make_azure_openai_tts_client(timeout=30)
    header(f"tts (native OpenAI) — {AZURE_OPENAI_TTS_DEPLOYMENT}, voice={args.voice}")
    try:
        resp = client.audio.speech.create(
            model=AZURE_OPENAI_TTS_DEPLOYMENT,
            input=args.text,
            voice=args.voice,
        )
        audio = resp.read()
        if not audio or len(audio) < 1024:
            fail(AZURE_OPENAI_TTS_DEPLOYMENT, f"audio too small: {len(audio)} bytes")
            return 1
        OUT.write_bytes(audio)
        ok(AZURE_OPENAI_TTS_DEPLOYMENT, f"wrote {OUT.name} ({len(audio)} bytes)")
        return 0
    except Exception as e:  # noqa: BLE001
        fail(AZURE_OPENAI_TTS_DEPLOYMENT, str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
