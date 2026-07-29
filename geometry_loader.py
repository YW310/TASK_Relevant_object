"""Lazy access to fused point-cloud geometry.

New fusion outputs store point arrays in a per-frame NPZ file.  The fallback
to an embedded ``points_world`` value deliberately remains here so every
consumer has one migration-compatible loading path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


class GeometryLoader:
    """Load point arrays relative to a fused JSON file, caching opened NPZs."""

    def __init__(self, fused_json_path: str | Path):
        self.base_dir = Path(fused_json_path).expanduser().resolve().parent
        self._archives: dict[Path, Any] = {}

    def load_points_world(self, record: Mapping[str, Any]) -> np.ndarray:
        """Return an ``(N, 3)`` point array from a new or legacy JSON record."""
        if "points_world" in record:
            return _as_points(record.get("points_world"))

        geometry_path = record.get("geometry_path")
        points_key = record.get("points_key")
        if not geometry_path or not points_key:
            return np.empty((0, 3), dtype=np.float32)

        path = Path(str(geometry_path)).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        path = path.resolve()
        archive = self._archives.get(path)
        if archive is None:
            archive = np.load(path, allow_pickle=False)
            self._archives[path] = archive
        if str(points_key) not in archive.files:
            raise KeyError(f"points key {points_key!r} not found in geometry archive {path}")
        points = _as_points(archive[str(points_key)])
        expected = record.get("point_count")
        if expected is not None and int(expected) != len(points):
            raise ValueError(
                f"point_count mismatch for {points_key!r}: JSON says {expected}, archive contains {len(points)}"
            )
        return points

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __enter__(self) -> "GeometryLoader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _as_points(value: Any) -> np.ndarray:
    points = np.asarray(value if value is not None else [], dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N, 3), got {points.shape}")
    return points

