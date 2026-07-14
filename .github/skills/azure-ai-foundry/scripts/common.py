"""Shared helpers for the **native** Azure Foundry samples.

These samples bypass the LiteLLM proxy entirely — every call goes
straight to the relevant Azure endpoint with the real Azure key. There
is no protocol translation: Anthropic models speak the Anthropic
Messages shape, Azure AI Inference models speak the inference shape,
TTS uses raw SSML, etc.

This module just centralises endpoint/key lookups, builds an
``AzureOpenAI`` client when one is needed, and provides the same tiny
``ok``/``fail``/``skip``/``header`` printers the proxy samples use so
output stays comparable side-by-side.

References
----------
* OpenAI Python SDK (source + README, including the ``AzureOpenAI`` client):
  https://github.com/openai/openai-python
* Azure OpenAI "switching endpoints" — how ``AzureOpenAI`` differs from
  the standard ``OpenAI`` client (auth header, deployment-in-URL, api-version):
  https://learn.microsoft.com/azure/ai-services/openai/how-to/switching-endpoints
* Azure OpenAI REST reference (api-versions, endpoints, request/response shapes):
  https://learn.microsoft.com/azure/ai-services/openai/reference
* httpx (transport used everywhere here, incl. ``verify=`` and ``http_client=``):
  https://www.python-httpx.org/
* python-dotenv (loads ``.env`` next to this file):
  https://github.com/theskumar/python-dotenv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load .env from samples_native/Python/.env
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


# Honor SSL_VERIFY=False to allow running behind a corporate TLS-interception
# proxy (matches the litellm/.env convention). When false we also silence
# urllib3's noisy InsecureRequestWarning so test output stays readable.
SSL_VERIFY: bool = _env_bool("SSL_VERIFY", True)
if not SSL_VERIFY:
    import warnings
    try:
        from urllib3.exceptions import InsecureRequestWarning  # type: ignore
        warnings.simplefilter("ignore", InsecureRequestWarning)
    except Exception:  # pragma: no cover
        pass


# ── Endpoint / key configuration ───────────────────────────────────────
# Main Foundry resource (Sweden Central)
AZURE_API_KEY: str = os.environ.get("AZURE_API_KEY", "")
AZURE_COGNITIVE_ENDPOINT: str = os.environ.get(
    "AZURE_COGNITIVE_ENDPOINT",
    "https://iexigtsazurioneyeai.cognitiveservices.azure.com",
).rstrip("/")
AZURE_FOUNDRY_ENDPOINT: str = os.environ.get(
    "AZURE_FOUNDRY_ENDPOINT",
    "https://iexigtsazurioneyeai.services.ai.azure.com",
).rstrip("/")

# TTS resource (East US 2)
AZURE_API_KEY_TTS: str = os.environ.get("AZURE_API_KEY_TTS", "")
AZURE_OPENAI_TTS_ENDPOINT: str = os.environ.get(
    "AZURE_OPENAI_TTS_ENDPOINT",
    "https://iexigtsazurioneye-openai.openai.azure.com",
).rstrip("/")
AZURE_OPENAI_TTS_DEPLOYMENT: str = os.environ.get(
    "AZURE_OPENAI_TTS_DEPLOYMENT", "gpt-4o-mini-tts-2"
)

# Realtime resource (Canada East)
AZURE_REALTIME_API_KEY: str = os.environ.get("AZURE_REALTIME_API_KEY", "")
AZURE_REALTIME_ENDPOINT: str = os.environ.get(
    "AZURE_REALTIME_ENDPOINT",
    "iexigtsazurioneyeai-can-resource.openai.azure.com",
)

# Azure Cognitive Services Speech (Sweden Central)
AZURE_SPEECH_KEY: str = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION: str = os.environ.get("AZURE_SPEECH_REGION", "swedencentral")

# Azure AI Search (RAG sample only)
AZURE_SEARCH_ENDPOINT: str = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
AZURE_SEARCH_INDEX: str = os.environ.get("AZURE_SEARCH_INDEX", "")
AZURE_SEARCH_API_KEY: str = os.environ.get("AZURE_SEARCH_API_KEY", "")

# API versions
AZURE_OPENAI_API_VERSION_CHAT = "2025-01-01-preview"
AZURE_OPENAI_API_VERSION_GPT5 = "2025-04-01-preview"
AZURE_OPENAI_API_VERSION_EMBEDDINGS = "2023-05-15"
AZURE_OPENAI_API_VERSION_IMAGE = "2025-04-01-preview"
AZURE_OPENAI_API_VERSION_TTS = "2025-03-01-preview"
AZURE_OPENAI_API_VERSION_REALTIME = "2025-04-01-preview"
AZURE_AI_INFERENCE_API_VERSION = "2024-05-01-preview"


def _require_main_key() -> None:
    if not AZURE_API_KEY:
        print("ERROR: AZURE_API_KEY is not set. Copy .env.example to .env and fill it in.",
              file=sys.stderr)
        sys.exit(2)


# ── Azure OpenAI clients ───────────────────────────────────────────────
def make_azure_openai_client(api_version: str = AZURE_OPENAI_API_VERSION_CHAT,
                             timeout: float = 300.0) -> AzureOpenAI:
    """Native Azure OpenAI SDK client pointed at the Sweden Central resource.

    Use this for chat completions, embeddings, gpt-image-2, and the
    Responses API on gpt-5.x deployments.
    """
    _require_main_key()
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_COGNITIVE_ENDPOINT,
        api_version=api_version,
        timeout=timeout,
        http_client=httpx.Client(verify=SSL_VERIFY, timeout=timeout),
    )


def make_azure_openai_tts_client(timeout: float = 60.0) -> AzureOpenAI:
    """Native Azure OpenAI SDK client for the East US 2 TTS resource."""
    if not AZURE_API_KEY_TTS:
        print("ERROR: AZURE_API_KEY_TTS is not set.", file=sys.stderr)
        sys.exit(2)
    return AzureOpenAI(
        api_key=AZURE_API_KEY_TTS,
        azure_endpoint=AZURE_OPENAI_TTS_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION_TTS,
        timeout=timeout,
        http_client=httpx.Client(verify=SSL_VERIFY, timeout=timeout),
    )


# ── Raw httpx clients for non-OpenAI shapes ───────────────────────────
def make_foundry_httpx_client(timeout: float = 300.0) -> httpx.Client:
    """httpx client pre-configured for the Foundry services endpoint
    (services.ai.azure.com — Anthropic, AI Inference, OCR, MAI/FLUX images).

    Caller still has to supply the path and api-version. The api-key
    header is injected automatically.
    """
    _require_main_key()
    return httpx.Client(
        base_url=AZURE_FOUNDRY_ENDPOINT,
        headers={"api-key": AZURE_API_KEY, "Content-Type": "application/json"},
        timeout=timeout,
        verify=SSL_VERIFY,
    )


# ── pretty-printing (mirrors samples/Python/common.py) ────────────────
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _color(s: str, c: str) -> str:
    return f"{c}{s}{_RESET}"


def ok(model: str, snippet: str = "") -> None:
    print(_color("  PASS  ", _GREEN), model, "-", snippet[:60])


def fail(model: str, err: str) -> None:
    print(_color("  FAIL  ", _RED), model, "-", err[:120])


def skip(model: str, reason: str) -> None:
    print(_color("  SKIP  ", _YELLOW), model, "-", reason)


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def first_text(content) -> str:
    """Best-effort extract a short snippet from a completion response."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.replace("\n", " ").strip()
    if isinstance(content, Iterable):
        for part in content:
            text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
            if text:
                return text.replace("\n", " ").strip()
    return str(content)[:80]
