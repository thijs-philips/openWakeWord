"""Native Responses-API sample — Azure OpenAI ``/openai/responses``.

The Responses API on Azure does NOT take a deployment in the URL.
``model`` lives in the request body and the URL is just
``<endpoint>/openai/responses?api-version=...``. The ``AzureOpenAI``
SDK client handles that automatically when you call
``client.responses.create(model="<deployment-name>", ...)``.

All five models below are deployed Responses-only on the Sweden Central
Foundry resource — calling ``chat.completions`` on them returns 404.

References
----------
* OpenAI Responses API reference (request/response shape, ``output_text``
  vs ``output[].content[].text`` extraction):
  https://platform.openai.com/docs/api-reference/responses
* Azure OpenAI — Responses API how-to (deployment behaviour, supported
  models, api-version matrix):
  https://learn.microsoft.com/azure/ai-services/openai/how-to/responses
* OpenAI Python SDK — ``client.responses.create`` source / examples:
  https://github.com/openai/openai-python#responses-api
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_OPENAI_API_VERSION_GPT5,
    fail,
    header,
    make_azure_openai_client,
    ok,
)

MODELS = [
    "gpt-5-mini",
    "gpt-5-pro",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
]

PROMPT = "Reply with exactly the two characters: OK"


def _extract_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()
    for item in getattr(resp, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            t = getattr(part, "text", None)
            if t:
                return t.strip()
    return ""


def main() -> int:
    client = make_azure_openai_client(
        api_version=AZURE_OPENAI_API_VERSION_GPT5, timeout=180
    )
    header(f"responses (native) — {len(MODELS)} models")
    failures = 0
    for m in MODELS:
        try:
            resp = client.responses.create(model=m, input=PROMPT)
            ok(m, _extract_text(resp))
        except Exception as e:  # noqa: BLE001
            fail(m, str(e))
            failures += 1
    print(f"\n{len(MODELS) - failures}/{len(MODELS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
