"""Run every non-interactive sample and produce a one-line summary per script.

Same harness as samples/Python/run_all.py but for the native-Azure
samples — each subprocess hits Azure directly without going through
the LiteLLM proxy.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

# (folder, timeout-seconds)
SAMPLES: list[tuple[str, int]] = [
    ("embeddings",        60),
    ("chat_completions", 600),
    ("responses",        600),
    ("rag",              120),
    ("ocr",              180),
    ("tts",               60),
    ("image_generation", 600),
]

LOG = HERE / "test-results.log"


def main() -> int:
    started = dt.datetime.now().isoformat(timespec="seconds")
    print(f"=== run_all started {started} ===")

    summary: list[tuple[str, str, float]] = []
    with LOG.open("w", encoding="utf-8") as logf:
        logf.write(f"=== run_all started {started} ===\n")
        for folder, timeout in SAMPLES:
            script = HERE / folder / "run.py"
            if not script.exists():
                summary.append((folder, "MISSING", 0.0))
                continue
            print(f"\n--- running {folder} (timeout {timeout}s) ---")
            logf.write(f"\n--- {folder} (timeout {timeout}s) ---\n")
            logf.flush()
            t0 = dt.datetime.now()
            try:
                cp = subprocess.run(
                    [str(PY), str(script)],
                    cwd=HERE,
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                )
                elapsed = (dt.datetime.now() - t0).total_seconds()
                logf.write(cp.stdout)
                if cp.stderr:
                    logf.write("STDERR:\n" + cp.stderr)
                print(cp.stdout, end="")
                status = "PASS" if cp.returncode == 0 else f"FAIL(rc={cp.returncode})"
            except subprocess.TimeoutExpired as e:
                elapsed = timeout
                logf.write(f"TIMEOUT after {timeout}s\n")
                if e.stdout:
                    logf.write(e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8", "replace"))
                status = "TIMEOUT"
                print(f"  TIMEOUT after {timeout}s")
            summary.append((folder, status, elapsed))

        print("\n=== SUMMARY ===")
        logf.write("\n=== SUMMARY ===\n")
        for folder, status, elapsed in summary:
            line = f"  {folder:20s} {status:15s} {elapsed:6.1f}s"
            print(line)
            logf.write(line + "\n")

        ended = dt.datetime.now().isoformat(timespec="seconds")
        print(f"=== run_all ended {ended} ===")
        logf.write(f"=== run_all ended {ended} ===\n")

    overall_fail = any(s != "PASS" for _, s, _ in summary)
    return 1 if overall_fail else 0


if __name__ == "__main__":
    sys.exit(main())
