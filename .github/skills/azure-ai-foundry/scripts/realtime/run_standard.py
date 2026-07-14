"""Native standard realtime — direct WebSocket to Azure OpenAI.

Connects to the Sweden Central resource at::

    wss://<endpoint>/openai/realtime?api-version=2025-04-01-preview&deployment=<model>

Authenticates with the ``api-key`` header (no Bearer/proxy wrapping).
Waits for the first server event and disconnects.

References
----------
* Azure OpenAI Realtime audio quickstart (URL shape, ``deployment`` query
  param, supported event types):
  https://learn.microsoft.com/azure/ai-services/openai/realtime-audio-quickstart
* OpenAI Realtime API reference (event names, ``session.update`` schema,
  audio input/output formats):
  https://platform.openai.com/docs/api-reference/realtime
* websockets library docs (``additional_headers=`` for the upgrade,
  ``ssl=`` for corporate TLS plumbing):
  https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html
"""

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_API_KEY,
    AZURE_COGNITIVE_ENDPOINT,
    AZURE_OPENAI_API_VERSION_REALTIME,
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


async def run(model: str) -> int:
    host = urlsplit(AZURE_COGNITIVE_ENDPOINT).netloc
    url = (
        f"wss://{host}/openai/realtime"
        f"?api-version={AZURE_OPENAI_API_VERSION_REALTIME}&deployment={model}"
    )
    header(f"realtime (native) — {model}")
    print(f"  Endpoint: {url}")
    try:
        async with websockets.connect(
            url,
            additional_headers={"api-key": AZURE_API_KEY},
            open_timeout=15,
            close_timeout=5,
            ssl=_SSL_CTX,
        ) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            event = json.loads(raw)
            ok(model, f"first event: {event.get('type', '<no type>')}")
            return 0
    except Exception as e:  # noqa: BLE001
        fail(model, str(e))
        return 1


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-realtime"
    return asyncio.run(run(model))


if __name__ == "__main__":
    sys.exit(main())
