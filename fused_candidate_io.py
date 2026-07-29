"""Compatibility-safe access to fused candidate manifests and geometry."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SUPPORTED_MANIFEST_SCHEMA_VERSION = 3
_SOURCE_PATH = "_fused_source_path"


def _context(frame_id: Any, object_id: Any, path: Path) -> str:
    return f"frame ID={frame_id!s}, object ID={object_id!s}, path={path}"


def _read_json(path: Path, *, frame_id: Any = "<unknown>") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read fused JSON ({_context(frame_id, '<unknown>', path)}): {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Incompatible fused schema: expected a JSON object ({_context(frame_id, '<unknown>', path)})")
    return value


def _is_legacy(manifest: Mapping[str, Any]) -> bool:
    if manifest.get("schema_version") == SUPPORTED_MANIFEST_SCHEMA_VERSION:
        return False
    frames = manifest.get("frames")
    return (
        manifest.get("schema_version") in (None, 1, 2)
        and isinstance(frames, list)
        and (not frames or all(isinstance(row, Mapping) and "objects" in row for row in frames))
    )


def load_fused_manifest(path: str | Path) -> dict[str, Any]:
    """Load and identify either a legacy monolithic JSON or a v3 manifest."""
    source = Path(path).expanduser().resolve()
    manifest = _read_json(source)
    if not isinstance(manifest.get("frames"), list):
        raise ValueError(f"Incompatible fused schema: missing frames list ({_context('<unknown>', '<unknown>', source)})")
    legacy = _is_legacy(manifest)
    if not legacy:
        if manifest.get("schema_version") != SUPPORTED_MANIFEST_SCHEMA_VERSION or not isinstance(manifest.get("generation_id"), str):
            raise ValueError(
                f"Incompatible fused schema version/generation {manifest.get('schema_version')!r}/"
                f"{manifest.get('generation_id')!r} ({_context('<unknown>', '<unknown>', source)})"
            )
        required = {"frame_id", "fused_objects_json", "status"}
        for entry in manifest["frames"]:
            if not isinstance(entry, Mapping) or not required.issubset(entry):
                frame_id = entry.get("frame_id", "<unknown>") if isinstance(entry, Mapping) else "<unknown>"
                raise ValueError(f"Incompatible fused frame entry ({_context(frame_id, '<unknown>', source)})")
    manifest[_SOURCE_PATH] = str(source)
    return manifest


def _attach_source(frame: Mapping[str, Any], source: Path, frame_ref: str | None = None) -> dict[str, Any]:
    result = dict(frame)
    result[_SOURCE_PATH] = str(source)
    if frame_ref is not None:
        result["frame_ref"] = frame_ref
    return result


def _frames(manifest: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    source_value = manifest.get(_SOURCE_PATH)
    if not source_value:
        raise ValueError("Fused manifest was not loaded by load_fused_manifest (frame ID=<unknown>, object ID=<unknown>, path=<unknown>)")
    source = Path(str(source_value))
    if _is_legacy(manifest):
        for frame in manifest["frames"]:
            yield _attach_source(frame, source)
        return

    generation_id = manifest["generation_id"]
    for entry in manifest["frames"]:
        if entry.get("status") != "complete":
            continue
        frame_id = entry.get("frame_id", "<unknown>")
        ref = str(entry["fused_objects_json"])
        frame_path = Path(ref).expanduser()
        if not frame_path.is_absolute():
            frame_path = source.parent / frame_path
        frame_path = frame_path.resolve()
        if not frame_path.is_file():
            raise FileNotFoundError(f"Missing fused frame file ({_context(frame_id, '<unknown>', frame_path)})")
        frame = _read_json(frame_path, frame_id=frame_id)
        if (frame.get("schema_version") != SUPPORTED_MANIFEST_SCHEMA_VERSION
                or frame.get("generation_id") != generation_id
                or str(frame.get("frame_id")) != str(frame_id)
                or not isinstance(frame.get("objects"), list)):
            raise ValueError(f"Incompatible fused frame schema or manifest identity ({_context(frame_id, '<unknown>', frame_path)})")
        yield _attach_source(frame, source, ref)


def iter_fused_frames(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield frames from a legacy file or lightweight manifest in file order."""
    yield from _frames(load_fused_manifest(path))


def load_fused_frame(manifest: Mapping[str, Any], frame_id: str | int) -> dict[str, Any]:
    """Load one frame by ID from a manifest returned by :func:`load_fused_manifest`."""
    for frame in _frames(manifest):
        if str(frame.get("frame_id")) == str(frame_id):
            return frame
    source = manifest.get(_SOURCE_PATH, "<unknown>")
    raise KeyError(f"Fused frame not found ({_context(frame_id, '<unknown>', Path(str(source)))})")


def load_object_points(frame: Mapping[str, Any], object_id: str | int) -> np.ndarray:
    """Load an object's ``(N, 3)`` points from embedded JSON or referenced NPZ."""
    frame_id = frame.get("frame_id", "<unknown>")
    source = Path(str(frame.get(_SOURCE_PATH, "<unknown>")))
    objects = frame.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"Incompatible fused frame schema: missing objects list ({_context(frame_id, object_id, source)})")
    obj = next((row for row in objects if isinstance(row, Mapping) and str(row.get("id")) == str(object_id)), None)
    if obj is None:
        raise KeyError(f"Fused object not found ({_context(frame_id, object_id, source)})")
    if "points_world" in obj:
        value: Any = obj["points_world"]
        geometry_path = source
    else:
        geometry_ref, points_key = obj.get("geometry_path"), obj.get("points_key")
        if not geometry_ref or not points_key:
            raise KeyError(f"Missing geometry_path or points_key ({_context(frame_id, object_id, source)})")
        geometry_path = Path(str(geometry_ref)).expanduser()
        if not geometry_path.is_absolute():
            geometry_path = source.parent / geometry_path
        geometry_path = geometry_path.resolve()
        try:
            with np.load(geometry_path, allow_pickle=False) as archive:
                _validate_archive_identity(archive, frame_id, object_id, geometry_path, frame)
                if str(points_key) not in archive.files:
                    raise KeyError(f"Missing geometry key {points_key!r} ({_context(frame_id, object_id, geometry_path)})")
                value = archive[str(points_key)]
        except KeyError:
            raise
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Cannot read geometry NPZ ({_context(frame_id, object_id, geometry_path)}): {exc}") from exc
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid points geometry ({_context(frame_id, object_id, geometry_path)}): {exc}") from exc
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid points geometry shape {points.shape} ({_context(frame_id, object_id, geometry_path)})")
    expected = obj.get("point_count")
    if expected is not None and int(expected) != len(points):
        raise ValueError(f"Geometry point_count mismatch: JSON={expected}, actual={len(points)} ({_context(frame_id, object_id, geometry_path)})")
    return points


def _validate_archive_identity(
    archive: Any,
    frame_id: Any,
    object_id: Any,
    path: Path,
    frame: Mapping[str, Any],
) -> None:
    """Validate self-describing NPZs while retaining pre-metadata compatibility."""
    metadata_keys = {"__schema_version__", "__generation_id__", "__frame_id__"}
    present = metadata_keys.intersection(archive.files)
    if not present:
        return
    if present != metadata_keys:
        raise ValueError(
            f"Incompatible geometry NPZ metadata; present keys={sorted(present)!r} "
            f"({_context(frame_id, object_id, path)})"
        )
    try:
        version = int(np.asarray(archive["__schema_version__"]).item())
        archive_generation = str(np.asarray(archive["__generation_id__"]).item())
        archive_frame_id = str(np.asarray(archive["__frame_id__"]).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid geometry NPZ metadata ({_context(frame_id, object_id, path)}): {exc}") from exc
    expected_generation = str(frame.get("generation_id", ""))
    if (
        version != SUPPORTED_MANIFEST_SCHEMA_VERSION
        or archive_generation != expected_generation
        or archive_frame_id != str(frame_id)
    ):
        raise ValueError(
            "Incompatible geometry NPZ identity: "
            f"schema={version!r}, generation={archive_generation!r}, frame={archive_frame_id!r}; "
            f"expected schema={SUPPORTED_MANIFEST_SCHEMA_VERSION!r}, generation={expected_generation!r}, "
            f"frame={str(frame_id)!r} ({_context(frame_id, object_id, path)})"
        )
