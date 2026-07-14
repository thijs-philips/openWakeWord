"""Native realtime translation — direct WebSocket to Azure (Canada East).

Translation lives at the dedicated path ``/openai/v1/realtime/translations``
on ``iexigtsazurioneyeai-can-resource.openai.azure.com`` and uses the
``api-key`` header for auth. Model is in the URL.

Sends ``session.update`` to set the target output language, then disconnects.

References
----------
* Azure OpenAI Realtime audio — overview (regional availability for
  ``gpt-realtime-translate`` and ``gpt-realtime-whisper``):
  https://learn.microsoft.com/azure/ai-services/openai/realtime-audio-quickstart
* OpenAI Realtime API events (``session.created``, ``session.update``,
  ``error`` shapes — same on Azure):
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

MODEL = "gpt-realtime-translate"


async def run(target_language: str = "es") -> int:
    url = (
        f"wss://{AZURE_REALTIME_ENDPOINT}"
        f"/openai/v1/realtime/translations?model={MODEL}"
    )
    header(f"realtime-translate (native) → {target_language}")
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
            if event.get("type") != "session.created":
                fail(MODEL, f"expected session.created, got: {event.get('type')}")
                return 1
            session = event["session"]
            print(f"  Session ID: {session['id']}")
            print(f"  Default input lang:  {session['audio']['input'].get('language')}")
            print(f"  Default output lang: {session['audio']['output'].get('language')}")
            print(f"  Default voice:       {session['audio']['output'].get('voice')}")

            session_update = {
                "type": "session.update",
                "session": {"audio": {"output": {"language": target_language}}},
            }
            await ws.send(json.dumps(session_update))
            raw2 = await asyncio.wait_for(ws.recv(), timeout=5)
            event2 = json.loads(raw2)
            if event2.get("type") == "error":
                fail(MODEL, f"session.update error: {event2['error']['message']}")
                return 1
            print(f"  session.update response: {event2.get('type')}")

            ok(MODEL, f"connected, target={target_language}")
            return 0
    except Exception as e:  # noqa: BLE001
        fail(MODEL, str(e))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Test gpt-realtime-translate (native)")
    parser.add_argument("--language", "-l", default="es", help="Target language (default: es)")
    args = parser.parse_args()
    return asyncio.run(run(target_language=args.language))


if __name__ == "__main__":
    sys.exit(main())
