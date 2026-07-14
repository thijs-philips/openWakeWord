---
name: azure-ai-foundry
description: 'How to call Azure AI Foundry deployments directly from Python — the right key, endpoint, and protocol shape Azure expects on the wire. Covers chat completions on Azure OpenAI, Anthropic-on-Foundry, and Azure AI Inference (Grok, Kimi, Phi, DeepSeek); the Responses API; embeddings; gpt-image / MAI-Image / FLUX image generation; Mistral OCR; Azure OpenAI "On Your Data" RAG with Azure AI Search; gpt-4o-mini-tts; Azure Cognitive Services Speech (SSML); and the realtime WebSocket. Use whenever the user wants to call a Foundry deployment, build or extend a Python script against `*.cognitiveservices.azure.com`, `*.services.ai.azure.com`, `*.openai.azure.com`, or `*.tts.speech.microsoft.com`, debug a 401/404/routing error on those hosts, wire RAG with Azure AI Search, send SSML to Azure Speech, open a realtime WebSocket, or call Claude/Grok/Kimi/Phi/DeepSeek on Foundry.'
---

# Azure AI Foundry — Python calls

This skill teaches you how to call Azure AI Foundry from Python. It bundles a complete, working sample for every surface in [`scripts/`](./scripts/) so you can run them as-is, copy a snippet, or extend the suite. The code in this skill is the source of truth — every snippet inlined below comes from a script that has been verified end-to-end against the real deployments.

## When to use this skill

Use this skill whenever the task involves talking directly to an Azure Foundry deployment from Python, **or** debugging why such a call fails. Concretely:

- Adding a new Python script that calls a Foundry deployment.
- Picking the right endpoint, key, and protocol shape for a given model family.
- Diagnosing a `401`, `404`, or routing error against `*.cognitiveservices.azure.com`, `*.services.ai.azure.com`, `*.openai.azure.com`, or `*.tts.speech.microsoft.com`.
- Wiring up Azure OpenAI "On Your Data" with Azure AI Search.
- Building or extending the bundled sample suite under [`scripts/`](./scripts/).

## The two things that are easy to get wrong

Most Foundry call failures come down to these two:

1. **Wrong endpoint host for the surface.** Foundry surfaces the same Azure subscription on more than one hostname depending on the protocol family. Picking the wrong one returns a `404` (the path doesn't exist on that host) or an opaque routing error. See the [endpoint table](#endpoint--key-map) for which host to use for what.
2. **Wrong auth header for the surface.** Azure-flavoured surfaces want `api-key`. Anthropic-on-Foundry preserves the **native Anthropic** convention (`x-api-key` + `anthropic-version`) — sending `api-key` to `/anthropic/v1/messages` returns `401 Access denied due to invalid subscription key`. The Speech REST API wants `Ocp-Apim-Subscription-Key`. The realtime WebSocket wants `api-key` as a header on the upgrade request.

Both are spelled out per-surface below.

## Endpoint / key map

There are **four** distinct Azure resources in play across the suite. Use the right host for each surface, or you will get 404s.

| Surface | Host (env var) | Auth header | Key env var |
|---|---|---|---|
| Azure OpenAI: chat, Responses, embeddings, gpt-image, RAG, realtime-standard | `AZURE_COGNITIVE_ENDPOINT` (`*.cognitiveservices.azure.com`) | `api-key` | `AZURE_API_KEY` |
| Foundry services: Anthropic, Azure AI Inference, OCR, MAI/FLUX images | `AZURE_FOUNDRY_ENDPOINT` (`*.services.ai.azure.com`) | `api-key` (except Anthropic — see below) | `AZURE_API_KEY` |
| gpt-4o-mini-tts | `AZURE_OPENAI_TTS_ENDPOINT` (`*.openai.azure.com`, East US 2) | `api-key` | `AZURE_API_KEY_TTS` |
| Azure Speech REST (SSML) | `https://<region>.tts.speech.microsoft.com` | `Ocp-Apim-Subscription-Key` | `AZURE_SPEECH_KEY` (region in `AZURE_SPEECH_REGION`) |
| Realtime translate / whisper | `AZURE_REALTIME_ENDPOINT` (`*.openai.azure.com`, Canada East) | `api-key` | `AZURE_REALTIME_API_KEY` |

The full set of variables — with examples and inline notes — lives in [`scripts/.env.example`](./scripts/.env.example). Copy it to `scripts/.env` and fill in the keys before running anything.

## API versions (current, working)

These are the values that actually work today against the deployments — use them verbatim unless Azure publishes a newer LTS for the surface. They're already centralised in [`scripts/common.py`](./scripts/common.py).

```python
AZURE_OPENAI_API_VERSION_CHAT       = "2025-01-01-preview"  # gpt-4*, o-series, gpt-5*-chat
AZURE_OPENAI_API_VERSION_GPT5       = "2025-04-01-preview"  # gpt-5, gpt-5.x, Responses
AZURE_OPENAI_API_VERSION_EMBEDDINGS = "2023-05-15"
AZURE_OPENAI_API_VERSION_IMAGE      = "2025-04-01-preview"
AZURE_OPENAI_API_VERSION_TTS        = "2025-03-01-preview"
AZURE_OPENAI_API_VERSION_REALTIME   = "2025-04-01-preview"
AZURE_AI_INFERENCE_API_VERSION      = "2024-05-01-preview"
# MAI / FLUX image pass-throughs use the literal string "preview".
```

## Use the bundled common helper, don't reinvent

[`scripts/common.py`](./scripts/common.py) builds the right clients with the right keys, the right endpoint, and the corporate-TLS plumbing. Import from it instead of constructing `AzureOpenAI` or `httpx.Client` manually:

```python
from common import (
    make_azure_openai_client,        # Sweden Central, AZURE_API_KEY
    make_azure_openai_tts_client,    # East US 2, AZURE_API_KEY_TTS
    make_foundry_httpx_client,       # services.ai.azure.com base_url, api-key header pre-injected
    AZURE_API_KEY, AZURE_COGNITIVE_ENDPOINT, AZURE_FOUNDRY_ENDPOINT,
    AZURE_OPENAI_API_VERSION_CHAT, AZURE_OPENAI_API_VERSION_GPT5,
    AZURE_AI_INFERENCE_API_VERSION,
    SSL_VERIFY,                      # plumb into your own httpx / websockets
    ok, fail, skip, header, first_text,
)
```

When adding a new sample under `scripts/<domain>/run.py`, follow this exact import dance — including the `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` line — so `common` resolves the same way the existing samples do.

## Pick the protocol shape

Foundry hosts many model families and **does not** normalise them onto a single OpenAI-shaped surface. You have to pick the right one per model family.

### Chat completions — three shapes

#### 1. Azure OpenAI (deployment-in-URL) — `gpt-4*`, `o-series`, `gpt-5*-chat`

```python
client = make_azure_openai_client(api_version=AZURE_OPENAI_API_VERSION_CHAT, timeout=120)
resp = client.chat.completions.create(
    model="gpt-4o",                       # = the Azure deployment name
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=1024,           # use this name for o-series reasoning headroom
)
text = resp.choices[0].message.content
```

For `gpt-5` / `gpt-5.x` chat deployments, switch to `AZURE_OPENAI_API_VERSION_GPT5`. The Responses-only deployments (`gpt-5-mini`, `gpt-5-pro`, `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-max`) **404 on `chat.completions`** — call them via `client.responses.create(model=..., input=...)` instead.

#### 2. Anthropic on Foundry — `claude-haiku-4-5`, `claude-opus-4-5`, `claude-opus-4-6`

This is the surface most likely to bite you. It speaks **native Anthropic Messages**, not Azure OpenAI:

```python
payload = {
    "model": "claude-opus-4-6",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": prompt}],
}
headers = {
    "x-api-key": AZURE_API_KEY,           # NOT api-key
    "anthropic-version": "2023-06-01",
    "api-key": "",                        # blank out the default that the helper injects
}
with make_foundry_httpx_client(timeout=120) as client:
    r = client.post("/anthropic/v1/messages", json=payload, headers=headers)
    body = r.json()
    text = next(b["text"] for b in body["content"] if b["type"] == "text")
```

Why the empty `api-key`: `make_foundry_httpx_client` pre-sets `api-key` for the Azure-flavoured Foundry surfaces; for Anthropic you must overwrite it with an empty string so httpx sends only `x-api-key`. Sending both, or sending only `api-key`, returns `401`. There is **no** `api-version` query param on this path.

#### 3. Azure AI Inference — Grok, Kimi, Phi, DeepSeek

Same JSON shape as OpenAI Chat, but `model` lives in the body and there's no deployment in the URL:

```python
payload = {
    "model": "grok-4-20-reasoning",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1024,
}
path = f"/models/chat/completions?api-version={AZURE_AI_INFERENCE_API_VERSION}"
with make_foundry_httpx_client(timeout=300) as client:   # bump timeout — these can cold-start slowly
    body = client.post(path, json=payload).json()
    text = body["choices"][0]["message"]["content"]
```

These deployments occasionally take 60–120 s on a cold start; **always set `timeout=300`** to avoid spurious read-timeout failures.

The full working version with all 20 chat models split across the three branches lives at [`scripts/chat_completions/run.py`](./scripts/chat_completions/run.py).

### RAG — Azure OpenAI On Your Data

Native Azure RAG is a non-standard `extra_body` extension on chat completions. Azure performs the embed → search → grounding loop server-side and returns a normal completion — no client-side retrieval needed.

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer using the data_sources results when relevant."},
        {"role": "user", "content": question},
    ],
    extra_body={
        "data_sources": [{
            "type": "azure_search",
            "parameters": {
                "endpoint": AZURE_SEARCH_ENDPOINT,           # https://<svc>.search.windows.net
                "index_name": AZURE_SEARCH_INDEX,
                "authentication": {"type": "api_key", "key": AZURE_SEARCH_API_KEY},
            },
        }],
    },
)
```

The OpenAI Python SDK passes `extra_body` straight through to the Azure deployment, so the call shape stays the same as a normal chat completion. See [`scripts/rag/run.py`](./scripts/rag/run.py) for a complete grounded-answer assertion.

### Other surfaces — short pointers

These are well-covered by the bundled samples; the one-line summaries here exist so you know which file to read.

| Surface | One-liner | Sample |
|---|---|---|
| Responses (gpt-5.x) | `client.responses.create(model="gpt-5.1", input=prompt)` — model in body, no deployment in URL, SDK handles it. | [`scripts/responses/run.py`](./scripts/responses/run.py) |
| Embeddings | `client.embeddings.create(model="text-embedding-3-large", input=[...])` — Azure OpenAI SDK, deployment in URL. | [`scripts/embeddings/run.py`](./scripts/embeddings/run.py) |
| `gpt-image-2` | `client.images.generate(model="gpt-image-2", prompt=..., size="1024x1024")` — returns `b64_json`. | [`scripts/image_generation/run.py`](./scripts/image_generation/run.py) |
| MAI-Image-2 | `POST /mai/v1/images/generations?api-version=preview` (Foundry endpoint, JSON body with `prompt`/`width`/`height`/`n`/`model`). | same file |
| FLUX.2-pro | `POST /providers/blackforestlabs/v1/flux-2-pro?api-version=preview` (same body shape). | same file |
| Mistral OCR | `POST /providers/mistral/azure/ocr` on Foundry endpoint, native Mistral request/response. | [`scripts/ocr/run.py`](./scripts/ocr/run.py) |
| gpt-4o-mini-tts | `make_azure_openai_tts_client().audio.speech.create(model="gpt-4o-mini-tts-2", voice=..., input=...)`. **Different resource (East US 2), different key (`AZURE_API_KEY_TTS`).** | [`scripts/tts/run_openai.py`](./scripts/tts/run_openai.py) |
| Azure Speech (SSML) | `POST https://<region>.tts.speech.microsoft.com/cognitiveservices/v1` with `Ocp-Apim-Subscription-Key`, `Content-Type: application/ssml+xml`, raw SSML body. | [`scripts/tts/run_speech.py`](./scripts/tts/run_speech.py) |
| Realtime (standard) | `wss://<cognitive-endpoint-host>/openai/realtime?api-version=2025-04-01-preview&deployment=<model>` with `api-key` header on the upgrade. | [`scripts/realtime/run_standard.py`](./scripts/realtime/run_standard.py) |
| Realtime (translate) | `wss://<canada-east-host>/openai/v1/realtime/translations?model=gpt-realtime-translate`, `api-key` header, target language via `session.update`. | [`scripts/realtime/run_translate.py`](./scripts/realtime/run_translate.py) |
| Realtime (whisper) | Standard realtime path with `&intent=transcription`, `transcription_session.*` event protocol, model in session config. | [`scripts/realtime/run_whisper.py`](./scripts/realtime/run_whisper.py) |

## Corporate TLS (`SSL_VERIFY=False`)

If `.env` sets `SSL_VERIFY=False` (corporate TLS-interception proxy), every outbound client must respect it. `common.py` exports the parsed boolean — plumb it through:

- **httpx clients**: `httpx.Client(verify=SSL_VERIFY, ...)` — `make_foundry_httpx_client` already does this.
- **`AzureOpenAI` SDK**: must be constructed with `http_client=httpx.Client(verify=SSL_VERIFY, timeout=...)` — both helper factories already do this.
- **`websockets.connect`**: pass `ssl=_SSL_CTX` where:
  ```python
  import ssl as _ssl
  _SSL_CTX = None
  if not SSL_VERIFY:
      _SSL_CTX = _ssl.create_default_context()
      _SSL_CTX.check_hostname = False
      _SSL_CTX.verify_mode = _ssl.CERT_NONE
  ```

Forgetting any one of these on a corporate network produces an opaque `SSL: CERTIFICATE_VERIFY_FAILED` from a layer that's hard to see — always plumb all three when adding a new sample.

## Running the bundled suite

```pwsh
cd .github/skills/azure-foundry-native/scripts
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env       # then fill in real keys
.\.venv\Scripts\python.exe run_all.py
```

[`scripts/run_all.py`](./scripts/run_all.py) runs every non-interactive sample and writes a one-line summary per script. The realtime samples are interactive (they hold a WebSocket open) and are excluded — run them individually from `scripts/realtime/`.

## When you add a new sample

1. Create `scripts/<domain>/run.py`.
2. Insert the parent on `sys.path` and `from common import ...` — never re-import `dotenv` or rebuild the client config locally.
3. Pick the correct endpoint factory: `make_azure_openai_client` for `*.cognitiveservices.azure.com`, `make_foundry_httpx_client` for `*.services.ai.azure.com`, `make_azure_openai_tts_client` for the East US 2 TTS resource, or a raw `httpx.Client(verify=SSL_VERIFY, ...)` for the Speech REST endpoint.
4. Use `ok(model, snippet)` / `fail(model, err)` / `skip(model, reason)` / `header(title)` for output so it lines up with the rest of the suite.
5. Make `main()` return `0` on success and a non-zero count on failure, and add the script to [`scripts/run_all.py`](./scripts/run_all.py) if it's non-interactive.

## Common failures and what they usually mean

| Symptom | Likely cause |
|---|---|
| `401 Access denied due to invalid subscription key` on `/anthropic/v1/messages` | You sent `api-key` instead of `x-api-key`. Use the Anthropic snippet above with `"api-key": ""` to blank the default. |
| `404` on a Responses-only deployment via `chat.completions` | Use `client.responses.create(...)` — that deployment isn't exposed on the chat-completions surface. |
| `Read timed out` on Grok / Kimi / Phi / DeepSeek | Cold start. Bump the httpx timeout to 300 s. |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Corporate TLS interception. Set `SSL_VERIFY=False` in `.env` and make sure the `verify=` / `http_client=` / `ssl=` plumbing is in place for *this* client. |
| Image call returns `200` but no bytes | MAI / FLUX paths return `{"data":[{"b64_json": "..."}]}` like OpenAI; some configurations return `url` instead. Handle both. |
| Realtime `403` / `404` against the wrong region | Translate and whisper realtime are Canada East only — use `AZURE_REALTIME_ENDPOINT` + `AZURE_REALTIME_API_KEY`, not the Sweden Central key. |

## What this skill does **not** cover

- Provisioning Foundry deployments (use the Azure portal or `az` CLI).
- Wrapping these calls behind a gateway / proxy of any kind — this skill is strictly about the direct, on-the-wire call.
- Streaming responses — every snippet here is single-shot. The same clients support streaming; consult the OpenAI Python SDK / Anthropic API docs for the per-surface streaming shape.
