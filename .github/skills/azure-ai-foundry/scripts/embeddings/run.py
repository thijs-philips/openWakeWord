"""Native embeddings sample — Azure OpenAI ``/embeddings`` deployments.

Both deployments live on the Sweden Central Foundry resource and are
reached via the ``AzureOpenAI`` SDK with the embeddings API version.

References
----------
* Azure OpenAI embeddings REST reference (request shape, dim sizes per model):
  https://learn.microsoft.com/azure/ai-services/openai/reference#embeddings
* OpenAI embeddings guide (model selection, normalisation, batching):
  https://platform.openai.com/docs/guides/embeddings
* OpenAI Python SDK — ``client.embeddings.create``:
  https://github.com/openai/openai-python#embeddings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_OPENAI_API_VERSION_EMBEDDINGS,
    fail,
    header,
    make_azure_openai_client,
    ok,
)

MODELS = ["text-embedding-3-large", "text-embedding-ada-002"]


def main() -> int:
    client = make_azure_openai_client(
        api_version=AZURE_OPENAI_API_VERSION_EMBEDDINGS, timeout=60
    )
    header(f"embeddings (native) — {len(MODELS)} models")
    failures = 0
    for m in MODELS:
        try:
            resp = client.embeddings.create(model=m, input="hello world")
            dim = len(resp.data[0].embedding)
            ok(m, f"dim={dim}")
        except Exception as e:  # noqa: BLE001
            fail(m, str(e))
            failures += 1
    print(f"\n{len(MODELS) - failures}/{len(MODELS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
