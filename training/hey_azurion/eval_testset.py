"""Quick eval of hey_azurion model against the held-out test wavs.

Runs the full openwakeword pipeline (melspec + speech_embeddings + custom model)
on every wav in positive_test/ and negative_test/, prints per-file peak score,
and summary stats (recall @ threshold and false-positive rate @ threshold).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf

from openwakeword.model import Model


def load_wav_16k_mono(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="int16", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        # simple decimation/interp via scipy would be nicer; we only see 16k clips here
        raise RuntimeError(f"{path} is {sr} Hz, expected 16000")
    return data.astype(np.int16)


def peak_score(model: Model, wav: np.ndarray, wakeword_name: str) -> float:
    model.reset()
    # feed in 1280-sample (80 ms) frames; that's the openwakeword chunk size
    chunk = 1280
    peak = 0.0
    for i in range(0, len(wav) - chunk + 1, chunk):
        scores = model.predict(wav[i : i + chunk])
        s = float(scores.get(wakeword_name, 0.0))
        if s > peak:
            peak = s
    return peak


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=r"D:\Github\openWakeWord\training\hey_azurion\hey_azurion.onnx")
    p.add_argument("--positives", default=r"D:\Github\openWakeWord\training\hey_azurion\hey_azurion\positive_test")
    p.add_argument("--negatives", default=r"D:\Github\openWakeWord\training\hey_azurion\hey_azurion\negative_test")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max", type=int, default=200, help="max files per class")
    args = p.parse_args()

    wakeword_name = os.path.splitext(os.path.basename(args.model))[0]
    print(f"Loading model: {args.model}")
    model = Model(
        wakeword_models=[args.model],
        inference_framework="onnx" if args.model.endswith(".onnx") else "tflite",
    )
    print(f"Wakeword key: {wakeword_name}")

    pos_files = sorted(glob.glob(os.path.join(args.positives, "*.wav")))[: args.max]
    neg_files = sorted(glob.glob(os.path.join(args.negatives, "*.wav")))[: args.max]
    print(f"Evaluating {len(pos_files)} positives, {len(neg_files)} negatives ...")

    t0 = time.time()
    pos_scores = []
    for f in pos_files:
        wav = load_wav_16k_mono(f)
        pos_scores.append(peak_score(model, wav, wakeword_name))

    neg_scores = []
    for f in neg_files:
        wav = load_wav_16k_mono(f)
        neg_scores.append(peak_score(model, wav, wakeword_name))
    dt = time.time() - t0

    pos = np.array(pos_scores)
    neg = np.array(neg_scores)

    print(f"\n--- summary (eval time {dt:.1f}s) ---")
    print(f"Positive peak scores: n={len(pos)} mean={pos.mean():.3f} median={np.median(pos):.3f} "
          f"min={pos.min():.3f} p10={np.percentile(pos,10):.3f}")
    print(f"Negative peak scores: n={len(neg)} mean={neg.mean():.3f} median={np.median(neg):.3f} "
          f"max={neg.max():.3f} p90={np.percentile(neg,90):.3f} p99={np.percentile(neg,99):.3f}")
    for thr in [0.3, 0.5, 0.7, 0.9]:
        recall = (pos >= thr).mean()
        fpr = (neg >= thr).mean()
        print(f"thr={thr:.2f}  recall={recall:.3f}  false-trigger rate={fpr:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
