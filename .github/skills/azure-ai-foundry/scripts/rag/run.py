"""Native RAG sample — Azure OpenAI "On Your Data" + Azure AI Search.

The proxy version uses LiteLLM's ``tools=[{type:"file_search",
vector_store_ids:["eval-rag"]}]`` shim, which LiteLLM rewrites into a
provider-specific retrieval call.

The native equivalent on Azure is the **chat completions extension**
``data_sources`` (a.k.a. *Azure OpenAI On Your Data*). It's a non-standard
``extra_body`` field that the Azure deployment understands directly:

    POST https://<resource>.cognitiveservices.azure.com/openai/deployments/<dep>/chat/completions
    {
      "messages": [...],
      "data_sources": [{
        "type": "azure_search",
        "parameters": {
          "endpoint": "https://<search>.search.windows.net",
          "index_name": "<index>",
          "authentication": { "type": "api_key", "key": "<search-admin-or-query-key>" }
        }
      }]
    }

Azure handles the embed → search → grounding loop server-side and
returns a normal chat completion. The OpenAI Python SDK passes
``extra_body`` straight through, so we keep the same call shape.

Required env vars (added to .env.example):
  AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AZURE_SEARCH_API_KEY

Skips itself with SKIP if AZURE_SEARCH_* aren't configured.

References
----------
* Azure OpenAI On Your Data — concept overview (how the embed/search/
  grounding loop runs server-side):
  https://learn.microsoft.com/azure/ai-services/openai/concepts/use-your-data
* On Your Data REST reference — full ``data_sources`` schema for every
  supported source (azure_search, cosmos_db, pinecone, ...):
  https://learn.microsoft.com/azure/ai-services/openai/references/on-your-data
* Azure AI Search REST API (the underlying index this calls into):
  https://learn.microsoft.com/rest/api/searchservice/
* OpenAI Python SDK — passing Azure-only fields via ``extra_body``:
  https://github.com/openai/openai-python#extra-body
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_OPENAI_API_VERSION_CHAT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX,
    fail,
    first_text,
    header,
    make_azure_openai_client,
    ok,
    skip,
)

MODEL = "gpt-4o-mini"


def main() -> int:
    header(f"rag (native) — {MODEL} + Azure AI Search ({AZURE_SEARCH_INDEX or '?'})")

    if not (AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX and AZURE_SEARCH_API_KEY):
        skip(MODEL, "AZURE_SEARCH_* env vars not set — see .env.example")
        return 0

    client = make_azure_openai_client(
        api_version=AZURE_OPENAI_API_VERSION_CHAT, timeout=60
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": "Answer using the data_sources results when relevant."},
                {"role": "user",
                 "content": "What is the secret codename of Project Iguana, who leads it, and what is the budget?"},
            ],
            extra_body={
                "data_sources": [{
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": AZURE_SEARCH_ENDPOINT,
                        "index_name": AZURE_SEARCH_INDEX,
                        "authentication": {
                            "type": "api_key",
                            "key": AZURE_SEARCH_API_KEY,
                        },
                    },
                }],
            },
        )
        answer = first_text(resp.choices[0].message.content)
        grounded = "BLUE-OWL-42" in answer.upper().replace(" ", "")
        if grounded:
            ok(MODEL, answer)
            return 0
        fail(MODEL, f"answer not grounded in index: {answer}")
        return 1
    except Exception as e:  # noqa: BLE001
        fail(MODEL, str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
