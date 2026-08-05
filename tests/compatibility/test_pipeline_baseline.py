from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("capture_pipeline_baseline.py")


def test_legacy_pipeline_contract_is_unchanged() -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--check"], check=True)

