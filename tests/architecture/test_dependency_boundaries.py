"""Protect dependency direction inside the new package architecture."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "relevant_object"

# A layer may depend only on the listed package layers. External imports are not
# restricted here; this test protects project architecture, not dependencies.
ALLOWED_INTERNAL_IMPORTS = {
    "artifacts": {"domain"},
    "cli": {
        "artifacts", "config", "data", "decision", "domain", "fusion",
        "geometry", "perception", "pipeline", "visualization", "vlm",
    },
    "config": {"domain"},
    "data": {"artifacts", "domain", "geometry"},
    "decision": {"artifacts", "config", "data", "domain", "fusion", "vlm"},
    "domain": set(),
    "fusion": {
        "artifacts", "config", "data", "domain", "geometry", "perception",
    },
    "geometry": {"domain"},
    "perception": {"artifacts", "config", "data", "domain", "geometry"},
    "pipeline": {
        "artifacts", "config", "data", "decision", "domain", "fusion",
        "geometry", "perception", "visualization", "vlm",
    },
    "visualization": {
        "artifacts", "config", "data", "decision", "domain", "fusion", "geometry",
    },
    "vlm": {"artifacts", "config", "data", "domain"},
}

LEGACY_MODULES = {
    "camera_geometry",
    "fusion_matching",
    "fusion_types",
    "mask_geometry",
    "multiview_candidate_fusion",
    "qwen3vl_object_role_decision",
}


def imported_modules(
    tree: ast.AST,
    relative_path: Path,
) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    module_parts = relative_path.with_suffix("").parts
    package_parts = (
        module_parts[:-1] if module_parts[-1] != "__init__" else module_parts[:-1]
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.append((node.lineno, node.module))
            elif node.level > 0:
                keep = len(package_parts) - (node.level - 1)
                anchor = package_parts[:max(0, keep)]
                suffix = tuple(node.module.split(".")) if node.module else ()
                resolved = ".".join(("relevant_object", *anchor, *suffix))
                found.append((node.lineno, resolved))
    return found


def test_package_dependency_direction() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if len(relative.parts) < 2:
            continue
        source_layer = relative.parts[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, module in imported_modules(tree, relative):
            root_module = module.split(".", 1)[0]
            if root_module in LEGACY_MODULES:
                violations.append(
                    f"{relative}:{line}: package code imports legacy module {module!r}"
                )
                continue
            prefix = "relevant_object."
            if not module.startswith(prefix):
                continue
            target_layer = module[len(prefix):].split(".", 1)[0]
            if target_layer == source_layer:
                continue
            allowed = ALLOWED_INTERNAL_IMPORTS.get(source_layer, set())
            if target_layer not in allowed:
                violations.append(
                    f"{relative}:{line}: {source_layer!r} may not import "
                    f"{target_layer!r} ({module})"
                )

    assert not violations, (
        "Architecture boundary violations:" + chr(10) + chr(10).join(violations)
    )
