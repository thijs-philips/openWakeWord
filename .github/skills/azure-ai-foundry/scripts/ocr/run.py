"""Native OCR sample — Mistral Document AI on Foundry.

Posts the same Mistral-shaped JSON the proxy version sends, but goes
straight to ``/providers/mistral/azure/ocr`` on the Foundry endpoint
with the real Azure key in the ``api-key`` header. There is no
OpenAI-compatible wrapping for this surface — the request and response
are Mistral's native shape.

References
----------
* Mistral Document AI / OCR capability docs (request fields, ``document``
  ``type=image_url`` vs ``document_url``, the ``pages[].markdown`` output):
  https://docs.mistral.ai/capabilities/document_ai/basic_ocr/
* Mistral models on Azure AI Foundry (deployment names, regional availability):
  https://learn.microsoft.com/azure/ai-foundry/foundry-models/concepts/models-from-mistral
"""

import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    fail,
    header,
    make_foundry_httpx_client,
    ok,
)

MODEL = "mistral-document-ai-2505"
PATH = "/providers/mistral/azure/ocr"
HERE = Path(__file__).resolve().parent
SAMPLE_PNG = HERE / "sample_invoice.png"
EXPECTED_PHRASES = ["INVOICE", "Acme Robotics", "BLUE-OWL-42"]


def build_sample_image() -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(img)

    def _font(size: int):
        for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    big = _font(34)
    med = _font(22)
    small = _font(18)

    draw.text((30, 25), "INVOICE #2026-0001", fill="black", font=big)
    draw.text((30, 80), "Acme Robotics, Inc.", fill="black", font=med)
    draw.text((30, 110), "1 Iguana Way, Sweden", fill="black", font=small)
    draw.text((30, 135), "Project codename: BLUE-OWL-42", fill="black", font=small)

    draw.line([(30, 175), (870, 175)], fill="black", width=2)
    draw.text((30, 185), "Description", fill="black", font=med)
    draw.text((520, 185), "Qty", fill="black", font=med)
    draw.text((620, 185), "Unit", fill="black", font=med)
    draw.text((760, 185), "Total", fill="black", font=med)
    draw.line([(30, 220), (870, 220)], fill="black", width=1)

    rows = [
        ("Quad-A actuator",        "4",  "$120.00", "$480.00"),
        ("Lidar housing",          "1",  "$340.00", "$340.00"),
        ("Cable harness, 2m",      "12", "$ 18.50", "$222.00"),
        ("Calibration, on-site",   "1",  "$200.00", "$200.00"),
    ]
    y = 230
    for desc, qty, unit, total in rows:
        draw.text((30, y),  desc,  fill="black", font=small)
        draw.text((520, y), qty,   fill="black", font=small)
        draw.text((620, y), unit,  fill="black", font=small)
        draw.text((760, y), total, fill="black", font=small)
        y += 32

    draw.line([(30, y + 5), (870, y + 5)], fill="black", width=1)
    draw.text((620, y + 18), "TOTAL DUE", fill="black", font=med)
    draw.text((760, y + 18), "$1,242.00", fill="black", font=med)

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def main() -> int:
    header(f"ocr (native) — {MODEL}")

    try:
        png_bytes = build_sample_image()
    except ImportError as e:
        fail(MODEL, f"Pillow not installed: {e}")
        return 1

    SAMPLE_PNG.write_bytes(png_bytes)
    print(f"  wrote {SAMPLE_PNG.name} ({len(png_bytes)} bytes)")

    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    payload = {
        "model": MODEL,
        "document": {"type": "image_url", "image_url": data_url},
    }

    with make_foundry_httpx_client(timeout=120) as client:
        try:
            r = client.post(PATH, json=payload)
            if r.status_code >= 400:
                fail(MODEL, f"HTTP {r.status_code}: {r.text[:200]}")
                return 1
            body = r.json()
            pages = body.get("pages", [])
            md = pages[0].get("markdown", "") if pages else ""
            print("  --- recognised text (first 400 chars) ---")
            print("  " + md[:400].replace("\n", "\n  "))
            print("  --- end ---")

            missing = [p for p in EXPECTED_PHRASES if p not in md]
            if missing:
                fail(MODEL, f"missing expected phrases: {missing}")
                return 1
            ok(MODEL, f"pages={len(pages)} chars={len(md)} all expected phrases found")
            return 0
        except Exception as e:  # noqa: BLE001
            fail(MODEL, str(e))
            return 1


if __name__ == "__main__":
    sys.exit(main())
