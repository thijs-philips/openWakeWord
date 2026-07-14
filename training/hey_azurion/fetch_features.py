"""Download openWakeWord pre-computed features from HF.

Files (from davidscripka/openwakeword_features):
  - validation_set_features.npy                       ~180 MB
  - openwakeword_features_ACAV100M_2000_hrs_16bit.npy ~17.3 GB

Both end up under ./features/. The HF hub library handles resume on
interrupted downloads automatically (xet transfer protocol).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "features"
REPO = "davidscripka/openwakeword_features"
FILES = [
    "validation_set_features.npy",
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
]

# Pull HF_TOKEN from the azure-ai-foundry skill's .env.
_SKILL_ENV = (HERE.parent.parent / ".github" / "skills" /
              "azure-ai-foundry" / "scripts" / ".env")
if _SKILL_ENV.is_file() and not os.environ.get("HF_TOKEN"):
    for _line in _SKILL_ENV.read_text(encoding="utf-8").splitlines():
        if _line.startswith("HF_TOKEN="):
            _tok = _line.split("=", 1)[1].strip()
            os.environ["HF_TOKEN"] = _tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
            break

OUT_DIR.mkdir(parents=True, exist_ok=True)

from huggingface_hub import hf_hub_download  # noqa: E402

for name in FILES:
    print(f"\n=== fetching {name} ===")
    local = hf_hub_download(
        repo_id=REPO,
        filename=name,
        repo_type="dataset",
        local_dir=str(OUT_DIR),
    )
    sz = Path(local).stat().st_size / 1e9
    print(f"  -> {local}  ({sz:.2f} GB)")

print(f"\nall files under: {OUT_DIR}")
