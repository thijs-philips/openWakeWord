"""Native realtime transcription (Whisper) — direct WebSocket to Azure (Canada East).

Transcription uses ``intent=transcription`` on the standard realtime
endpoint and a ``transcription_session.*`` event protocol (model is in
the session config, not the URL).

References
----------
* Azure OpenAI Realtime audio quickstart (transcription mode, ``intent=transcription``,
  ``transcription_session.update`` shape):
  https://learn.microsoft.com/azure/ai-services/openai/realtime-audio-quickstart
* OpenAI Realtime API events — transcription session schema
  (``input_audio_transcription``, ``turn_detection`` / server-VAD knobs):
  https://platform.openai.com/docs/api-reference/realtime
* websockets library docs:
  https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_OPENAI_API_VERSION_REALTIME,
    AZURE_REALTIME_API_KEY,
    AZURE_REALTIME_ENDPOINT,
    SSL_VERIFY,
    fail,
    header,
    ok,
)

import ssl as _ssl  # noqa: E402
import websockets  # noqa: E402

_SSL_CTX = None
if not SSL_VERIFY:
    _SSL_CTX = _ssl.create_default_context()
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = _ssl.CERT_NONE

MODEL = "gpt-realtime-whisper"


async def run(language: str = "en") -> int:
    url = (
        f"wss://{AZURE_REALTIME_ENDPOINT}"
        f"/openai/realtime?api-version={AZURE_OPENAI_API_VERSION_REALTIME}"
        f"&intent=transcription"
    )
    header(f"realtime-whisper (native, lang={language})")
    print(f"  Endpoint: {url}")
    try:
        async with websockets.connect(
            url,
            additional_headers={"api-key": AZURE_REALTIME_API_KEY},
            open_timeout=15,
            close_timeout=5,
            ssl=_SSL_CTX,
        ) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            event = json.loads(raw)
            if event.get("type") != "transcription_session.created":
                fail(MODEL, f"expected transcription_session.created, got: {event.get('type')}")
                return 1
            session = event["session"]
            print(f"  Session ID: {session['id']}")
            print(f"  Input format: {session.get('input_audio_format')}")
            print(f"  Turn detection: {session.get('turn_detection', {}).get('type')}")

            session_update = {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": MODEL,
                        "language": language,
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                },
            }
            await ws.send(json.dumps(session_update))
            raw2 = await asyncio.wait_for(ws.recv(), timeout=5)
            event2 = json.loads(raw2)
            if event2.get("type") == "error":
                fail(MODEL, f"session update error: {event2['error']['message']}")
                return 1

            print(f"  session update response: {event2.get('type')}")
            if event2.get("type") == "transcription_session.updated":
                txn = event2.get("session", {}).get("input_audio_transcription", {})
                print(f"  Configured model:    {txn.get('model')}")
                print(f"  Configured language: {txn.get('language')}")

            ok(MODEL, f"connected, model={MODEL}, lang={language}")
            return 0
    except Exception as e:  # noqa: BLE001
        fail(MODEL, str(e))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Test gpt-realtime-whisper (native)")
    parser.add_argument("--language", "-l", default="en", help="Language hint (default: en)")
    args = parser.parse_args()
    return asyncio.run(run(language=args.language))


if __name__ == "__main__":
    sys.exit(main())
