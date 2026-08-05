"""Make the src package importable from legacy repository entry points.

This keeps the source checkout runnable without installing the project. Root-level
compatibility modules call ensure_src_path before importing the new package.
"""

from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parent / "src"


def ensure_src_path() -> Path:
    """Put this checkout's src directory first on sys.path once."""
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    return SRC_ROOT
