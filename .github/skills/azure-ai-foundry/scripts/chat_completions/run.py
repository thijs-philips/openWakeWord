"""Native chat-completions sample — talks to Azure Foundry directly.

This sample replaces the proxy version in samples/Python/chat_completions/run.py.
Because there's no LiteLLM in the middle to translate protocols, the
21 chat models split into THREE distinct call shapes:

  1. **Azure OpenAI** (gpt-4*, o-series, gpt-5*-chat)
     OpenAI Chat Completions over the Azure OpenAI deployment URL,
     reached via the ``AzureOpenAI`` SDK client.

  2. **Anthropic on Foundry** (claude-haiku-4-5, claude-opus-4-5/4-6)
     Native Anthropic Messages API at
     ``services.ai.azure.com/anthropic/v1/messages``. Different request
     and response shape — ``messages``/``content`` blocks, ``stop_reason``,
     ``usage.input_tokens``. Authenticated with ``api-key`` header (no
     Anthropic ``x-api-key`` and no ``anthropic-version`` header needed
     when going through the Azure-Anthropic surface).

  3. **Azure AI Inference** (Grok, Kimi, Phi, DeepSeek)
     Foundry's ``services.ai.azure.com/models/chat/completions`` endpoint
     — same JSON body as OpenAI Chat, but no deployment in the URL
     (``model`` lives in the body) and authentication via ``api-key``
     header.

Each branch is implemented with raw httpx where it isn't OpenAI-shaped
so the on-the-wire payload is visible without an SDK abstraction layer.

References
----------
* Azure OpenAI Chat Completions REST reference (api-versions, request shape,
  o-series ``max_completion_tokens`` parameter):
  https://learn.microsoft.com/azure/ai-services/openai/reference#chat-completions
* OpenAI Python SDK — ``client.chat.completions.create`` (same call shape
  the ``AzureOpenAI`` client exposes):
  https://github.com/openai/openai-python#chat-completions
* Anthropic Messages API (the protocol Foundry's ``/anthropic/v1/messages``
  speaks verbatim — request fields, content blocks, ``stop_reason``):
  https://docs.anthropic.com/en/api/messages
* Claude models on Azure AI Foundry (auth, deployment list, regional notes):
  https://learn.microsoft.com/azure/ai-foundry/foundry-models/concepts/models-from-anthropic
* Azure AI Model Inference — Chat Completions reference
  (the ``/models/chat/completions`` surface used by Grok / Kimi / Phi / DeepSeek):
  https://learn.microsoft.com/azure/ai-foundry/model-inference/reference/reference-model-inference-chat-completions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_AI_INFERENCE_API_VERSION,
    AZURE_API_KEY,
    AZURE_OPENAI_API_VERSION_CHAT,
    AZURE_OPENAI_API_VERSION_GPT5,
    fail,
    first_text,
    header,
    make_azure_openai_client,
    make_foundry_httpx_client,
    ok,
)

PROMPT = "Reply with exactly the two characters: OK"

# (deployment, api_version) — Azure OpenAI deployments reached via SDK.
AZURE_OPENAI_MODELS: list[tuple[str, str]] = [
    ("gpt-4o",     AZURE_OPENAI_API_VERSION_CHAT),
    ("gpt-4o-mini", AZURE_OPENAI_API_VERSION_CHAT),
    ("gpt-4.1",    AZURE_OPENAI_API_VERSION_CHAT),
    ("o1",         AZURE_OPENAI_API_VERSION_CHAT),
    ("o3",         AZURE_OPENAI_API_VERSION_CHAT),
    ("o3-mini",    AZURE_OPENAI_API_VERSION_CHAT),
    ("gpt-5",      AZURE_OPENAI_API_VERSION_GPT5),
    ("gpt-5-nano", AZURE_OPENAI_API_VERSION_GPT5),
    ("gpt-5.2",    AZURE_OPENAI_API_VERSION_GPT5),
    ("gpt-5.4",    AZURE_OPENAI_API_VERSION_GPT5),
    ("gpt-5.5",    AZURE_OPENAI_API_VERSION_GPT5),
    # gpt-5.1 is dual-surface; keep it on Responses (see responses/run.py).
]

# Anthropic deployments — native Messages protocol.
ANTHROPIC_MODELS: list[str] = [
    "claude-haiku-4-5",
    "claude-opus-4-5",
    "claude-opus-4-6",
]

# Azure AI Inference deployments — `/models/chat/completions`.
INFERENCE_MODELS: list[str] = [
    "grok-4-20-reasoning",
    "grok-4-20-non-reasoning",
    "Kimi-K2.5",
    "Kimi-K2.6",
    "Phi-4-multimodal-instruct",
    "DeepSeek-V4-Flash",
]


def _run_azure_openai(deployment: str, api_version: str) -> bool:
    client = make_azure_openai_client(api_version=api_version, timeout=120)
    try:
        resp = client.chat.completions.create(
            model=deployment,                 # = Azure deployment name
            messages=[{"role": "user", "content": PROMPT}],
            max_completion_tokens=1024,       # for o-series reasoning headroom
        )
        ok(deployment, first_text(resp.choices[0].message.content))
        return True
    except Exception as e:  # noqa: BLE001
        fail(deployment, str(e))
        return False


def _run_anthropic(model: str) -> bool:
    """Direct call to /anthropic/v1/messages.

    Foundry-hosted Anthropic preserves the *native* Anthropic auth and
    headers — not Azure's. So:
      - ``x-api-key`` header (NOT ``api-key``)
      - ``anthropic-version`` header (Anthropic protocol revision)
      - No Azure ``api-version`` query param
    The Azure key value still works as the ``x-api-key`` value.
    """
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    # The Foundry httpx client injects "api-key" by default; override
    # both auth headers explicitly here.
    headers = {
        "x-api-key": AZURE_API_KEY,
        "anthropic-version": "2023-06-01",
        "api-key": "",  # blank out the default so Azure routes via x-api-key
    }
    with make_foundry_httpx_client(timeout=120) as client:
        try:
            r = client.post("/anthropic/v1/messages", json=payload, headers=headers)
            if r.status_code >= 400:
                fail(model, f"HTTP {r.status_code}: {r.text[:200]}")
                return False
            body = r.json()
            blocks = body.get("content", [])
            text = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
            ok(model, text)
            return True
        except Exception as e:  # noqa: BLE001
            fail(model, str(e))
            return False


def _run_inference(model: str) -> bool:
    """Direct call to /models/chat/completions (Azure AI Inference)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1024,
    }
    path = f"/models/chat/completions?api-version={AZURE_AI_INFERENCE_API_VERSION}"
    with make_foundry_httpx_client(timeout=300) as client:
        try:
            r = client.post(path, json=payload)
            if r.status_code >= 400:
                fail(model, f"HTTP {r.status_code}: {r.text[:200]}")
                return False
            body = r.json()
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            ok(model, first_text(text))
            return True
        except Exception as e:  # noqa: BLE001
            fail(model, str(e))
            return False


def main() -> int:
    total = len(AZURE_OPENAI_MODELS) + len(ANTHROPIC_MODELS) + len(INFERENCE_MODELS)
    header(f"chat_completions (native) — {total} models across 3 protocols")
    failures = 0

    print("\n-- Azure OpenAI (deployment URL) --")
    for deployment, ver in AZURE_OPENAI_MODELS:
        if not _run_azure_openai(deployment, ver):
            failures += 1

    print("\n-- Anthropic on Foundry (/anthropic/v1/messages) --")
    for m in ANTHROPIC_MODELS:
        if not _run_anthropic(m):
            failures += 1

    print("\n-- Azure AI Inference (/models/chat/completions) --")
    for m in INFERENCE_MODELS:
        if not _run_inference(m):
            failures += 1

    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
