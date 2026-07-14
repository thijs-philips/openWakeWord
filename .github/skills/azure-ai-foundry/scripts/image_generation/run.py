"""Native image generation — gpt-image-2, MAI-Image-2, FLUX.2-pro.

Three deployments, three call shapes (only the first is OpenAI-compatible):

  • gpt-image-2  — Azure OpenAI deployment, called via the SDK's
                   ``client.images.generate(...)``.
  • MAI-Image-2  — Microsoft AI image model on Foundry, posted as JSON
                   to ``/mai/v1/images/generations``.
  • FLUX.2-pro   — Black Forest Labs on Foundry, posted as JSON to
                   ``/providers/blackforestlabs/v1/flux-2-pro``.

The two pass-through-ish models return ``{"data":[{"b64_json": "..."}]}``
just like OpenAI's image endpoint.

Run:
    python image_generation/run.py              # all three
    python image_generation/run.py gpt-image-2  # just one

References
----------
* Azure OpenAI image generation how-to (gpt-image-2 sizes, quality, n,
  ``b64_json`` vs URL):
  https://learn.microsoft.com/azure/ai-services/openai/how-to/image-generation
* OpenAI Python SDK — ``client.images.generate``:
  https://github.com/openai/openai-python#image-generation
* Azure AI Foundry model catalog (MAI-Image-2 and FLUX.2-pro deployment
  details — provider paths, supported parameters):
  https://learn.microsoft.com/azure/ai-foundry/concepts/foundry-models-overview
* Black Forest Labs FLUX API reference (the body shape the Foundry
  pass-through accepts at ``/providers/blackforestlabs/v1/flux-2-pro``):
  https://docs.bfl.ai/
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    AZURE_OPENAI_API_VERSION_IMAGE,
    fail,
    header,
    make_azure_openai_client,
    make_foundry_httpx_client,
    ok,
)

HERE = Path(__file__).resolve().parent
PROMPT = "A small orange cat sleeping on a wooden windowsill, soft morning light"


def run_gpt_image_2() -> int:
    client = make_azure_openai_client(
        api_version=AZURE_OPENAI_API_VERSION_IMAGE, timeout=300
    )
    model = "gpt-image-2"
    header(f"image_generation (native) — {model}")
    try:
        resp = client.images.generate(
            model=model, prompt=PROMPT, size="1024x1024", n=1
        )
        d = resp.data[0]
        if getattr(d, "b64_json", None):
            out = HERE / "out.png"
            out.write_bytes(base64.b64decode(d.b64_json))
            ok(model, f"saved {out.name} ({out.stat().st_size} bytes)")
        elif getattr(d, "url", None):
            ok(model, f"url={d.url[:60]}...")
        else:
            fail(model, "no b64_json or url in response")
            return 1
        return 0
    except Exception as e:  # noqa: BLE001
        fail(model, str(e))
        return 1


# Native Foundry paths (same targets the LiteLLM pass-throughs forward to).
FOUNDRY_IMAGE_MODELS = {
    "MAI-Image-2": {
        "path": "/mai/v1/images/generations?api-version=preview",
        "body": {"prompt": PROMPT, "width": 1024, "height": 1024, "n": 1,
                 "model": "MAI-Image-2"},
        "out": "out_mai.png",
    },
    "FLUX.2-pro": {
        "path": "/providers/blackforestlabs/v1/flux-2-pro?api-version=preview",
        "body": {"prompt": PROMPT, "width": 1024, "height": 1024, "n": 1,
                 "model": "FLUX.2-pro"},
        "out": "out_flux.png",
    },
}


def run_foundry(model: str) -> int:
    cfg = FOUNDRY_IMAGE_MODELS[model]
    header(f"image_generation (native) — {model}")
    with make_foundry_httpx_client(timeout=300) as client:
        try:
            r = client.post(cfg["path"], json=cfg["body"])
            if r.status_code >= 400:
                fail(model, f"HTTP {r.status_code}: {r.text[:300]}")
                return 1
            data = r.json()
            items = data.get("data", [])
            if not items:
                fail(model, "empty data array")
                return 1
            b64 = items[0].get("b64_json", "")
            if not b64:
                url = items[0].get("url", "")
                if url:
                    ok(model, f"url={url[:60]}...")
                    return 0
                fail(model, "no b64_json or url in response")
                return 1
            img = base64.b64decode(b64)
            out = HERE / cfg["out"]
            out.write_bytes(img)
            ok(model, f"wrote {out.name} ({len(img)} bytes)")
            return 0
        except Exception as e:  # noqa: BLE001
            fail(model, str(e))
        return 1


ALL_MODELS = ["gpt-image-2", "MAI-Image-2", "FLUX.2-pro"]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Test image generation models (native Azure)")
    parser.add_argument("models", nargs="*", default=ALL_MODELS,
                        help=f"Models to test (default: all). Choices: {', '.join(ALL_MODELS)}")
    args = parser.parse_args()

    rc = 0
    for m in args.models:
        if m == "gpt-image-2":
            rc |= run_gpt_image_2()
        else:
            rc |= run_foundry(m)
    return rc


if __name__ == "__main__":
    sys.exit(main())
