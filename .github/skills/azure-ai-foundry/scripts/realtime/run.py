"""Realtime samples runner — runs all three realtime scripts (native).

All three connect directly to Azure (no LiteLLM, no sidecar). Interactive
and NOT included in the unattended ``run_all.py`` suite.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

SCRIPTS = ["run_standard.py", "run_translate.py", "run_whisper.py"]


def main() -> int:
    rc = 0
    for script in SCRIPTS:
        path = HERE / script
        if not path.exists():
            print(f"  SKIP  {script} (not found)")
            continue
        result = subprocess.run([PY, str(path)], cwd=str(HERE))
        rc |= result.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
