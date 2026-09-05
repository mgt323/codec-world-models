"""Run Codec A parity audit (thin wrapper over shared scripts/run_parity_audit.py)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).with_name("run_parity_audit.py")
    sys.argv = [str(script), "--codec", "A", *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
