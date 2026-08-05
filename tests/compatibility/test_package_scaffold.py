from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def test_all_architecture_packages_are_importable() -> None:
    names = (
        "relevant_object", "relevant_object.config", "relevant_object.domain",
        "relevant_object.geometry", "relevant_object.artifacts",
        "relevant_object.data", "relevant_object.perception",
        "relevant_object.vlm", "relevant_object.fusion",
        "relevant_object.decision", "relevant_object.visualization",
        "relevant_object.pipeline", "relevant_object.cli",
    )
    for name in names:
        importlib.import_module(name)


def test_top_level_package_import_has_no_model_side_effects() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "import relevant_object; "
        "heavy = {'torch', 'numpy', 'PIL', 'transformers'} & set(sys.modules); "
        "assert not heavy, sorted(heavy)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
