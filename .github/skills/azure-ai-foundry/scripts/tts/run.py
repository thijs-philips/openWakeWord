"""Native TTS samples runner — runs both engines."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def main() -> int:
    rc = 0
    for script in ["run_speech.py", "run_openai.py"]:
        path = HERE / script
        if not path.exists():
            print(f"  SKIP  {script} (not found)")
            continue
        result = subprocess.run([PY, str(path)], cwd=str(HERE))
        rc |= result.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
