"""Small dependency-free helpers shared by pipeline entrypoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def atomic_json_dump(data: Any, path: str | Path) -> None:
    """Write formatted JSON through a sibling temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(destination)


def load_json(path: str | Path) -> Any:
    """Load UTF-8 JSON from a filesystem path."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def natural_sort_key(value: str | Path) -> list[Any]:
    """Sort frame-like names numerically: ``2.png`` before ``10.png``."""
    text = Path(value).stem if isinstance(value, Path) else str(value)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def parse_csv(value: str) -> tuple[str, ...]:
    """Argparse-compatible parser for a required non-empty CSV value."""
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return items


def parse_optional_csv(value: str | None) -> tuple[str, ...] | None:
    """Parse an optional CSV value without imposing argparse validation."""
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())
