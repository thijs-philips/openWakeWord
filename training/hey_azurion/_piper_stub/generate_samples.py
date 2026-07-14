"""Stub for piper-sample-generator's `generate_samples`.

openwakeword/train.py does `from generate_samples import generate_samples`
unconditionally, even when --generate_clips is not passed. We already
produced all clips externally (Azure Speech + gpt-4o-mini-tts), so this
stub just satisfies the import. If --generate_clips IS ever passed by
mistake, the function raises so we notice immediately.
"""

from __future__ import annotations


def generate_samples(*args, **kwargs):  # noqa: D401
    raise RuntimeError(
        "Piper generate_samples stub was called. This trainer was "
        "configured to consume pre-generated clips only — do not pass "
        "--generate_clips. If you need piper, point "
        "piper_sample_generator_path at a real clone of "
        "https://github.com/dscripka/piper-sample-generator."
    )
