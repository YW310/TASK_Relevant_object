"""Offline role guidance interchange: JSON metadata + finite, metre-valued NPZ.

This module imports neither BridgeVLA, SAM3 nor torch and has no inference side
effects. Producers must detach tensors and provide the *actual* inverse transform
for the current observation. In particular, an arbitrary Slot XYZ is not world XYZ.
See docs/slot_guidance_interface_audit.md before writing a predicted producer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROLES = ("target", "reference")
STATUSES = ("VISIBLE", "NULL", "NOT_VISIBLE", "LOW_CONFIDENCE")
_STORAGE = {"points_file", "points_sha256"}
_TOP_REQUIRED = {
    "schema_version", "artifact_type", "episode_id", "frame_id",
    "source_frame_index", "instruction", "task_name", "coordinate_frame",
    "coordinate_units", "observed_at", "created_at", "guidance_source",
    "source_checkpoint", "source_git_sha", "transform", "target", "reference",
    "reference_null_probability", *_STORAGE,
}
_ROLE_REQUIRED = {
    "status", "semantic_valid", "confidence", "center_world", "points_key",
    "point_count",
}
_ROLE_OPTIONAL = {"spread_world", "spread_source", "semantic_label", "instance_id"}
_TRANSFORM_REQUIRED = {
    "source_frame", "target_frame", "matrix_world_from_source", "source",
    "timestamp", "source_frame_index",
}


def _fields(value: Any, required: set[str], optional: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise ValueError(f"{name}: missing={sorted(missing)}, unexpected={sorted(extra)}")


def _string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _number(value: Any, name: str, *, probability: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if probability and not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def _timestamp(value: Any, name: str) -> datetime:
    _string(value, name)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)", value):
        raise ValueError(f"{name} must be RFC3339 with uppercase T/Z and explicit timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or "T" not in value:
        raise ValueError(f"{name} must include time and timezone")
    return parsed


def _array(value: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    # NumPy otherwise silently turns [True, 0, 1] into integer coordinates.
    def has_boolean(item: Any) -> bool:
        if isinstance(item, (bool, np.bool_)):
            return True
        return isinstance(item, (list, tuple)) and any(has_boolean(part) for part in item)

    if has_boolean(value):
        raise ValueError(f"{name} cannot contain boolean coordinates")
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be detached numeric data") from exc
    if arr.dtype.kind not in "fiu" or arr.ndim != len(shape) or any(
        expected is not None and size != expected for size, expected in zip(arr.shape, shape)
    ):
        raise ValueError(f"{name} must have numeric shape {shape}, got {arr.shape}/{arr.dtype}")
    arr = np.array(arr, dtype=np.float64, copy=True)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    return arr


def _validate_transform(transform: Mapping[str, Any], frame_index: int, observed_at: str) -> None:
    _fields(transform, _TRANSFORM_REQUIRED, {"normalization"}, "transform")
    for key in ("source_frame", "source", "target_frame"):
        _string(transform[key], f"transform.{key}")
    if transform["target_frame"] != "rlbench_world":
        raise ValueError("transform.target_frame must be rlbench_world")
    _integer(transform["source_frame_index"], "transform.source_frame_index")
    if transform["source_frame_index"] != frame_index:
        raise ValueError("transform must belong to the exact source_frame_index")
    if _timestamp(transform["timestamp"], "transform.timestamp") != _timestamp(observed_at, "observed_at"):
        raise ValueError("transform timestamp must identify the current observation")
    matrix = _array(transform["matrix_world_from_source"], (4, 4), "transform matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-10):
        raise ValueError("transform must be affine with last row [0,0,0,1]")
    linear = matrix[:3, :3]
    gram = linear.T @ linear
    scale_sq = np.trace(gram) / 3
    if scale_sq <= 0 or np.linalg.det(linear) <= 0 or not np.allclose(
        gram, np.eye(3) * scale_sq, rtol=1e-7, atol=1e-12 * scale_sq
    ):
        raise ValueError("transform must be an invertible proper uniform-scale similarity")
    if transform["source_frame"] == "rlbench_world" and not np.allclose(matrix, np.eye(4), rtol=0, atol=1e-10):
        raise ValueError("rlbench_world source requires the identity transform")
    if "normalization" in transform:
        norm = transform["normalization"]
        _fields(norm, {"center_world", "cube_scale"}, set(), "transform.normalization")
        center = _array(norm["center_world"], (3,), "normalization.center_world")
        scale = _number(norm["cube_scale"], "normalization.cube_scale")
        if scale <= 0 or transform["source_frame"] != "bridgevla_cube":
            raise ValueError("cube normalization requires positive scale and bridgevla_cube source")
        expected = np.eye(4)
        expected[:3, :3] /= scale
        expected[:3, 3] = center
        if not np.allclose(matrix, expected, rtol=1e-9, atol=1e-12):
            raise ValueError("normalization disagrees with matrix_world_from_source")


def validate_guidance(metadata: Mapping[str, Any], points: Mapping[str, np.ndarray]) -> None:
    """Validate semantic, storage, temporal and geometry constraints without SAM3.

    All role coordinates are already in RLBench world metres. ``transform`` is
    provenance for the producer's conversion, and must not be applied on load.
    Visibility is an observation-level claim; PR2 must still check each camera.
    """
    _fields(metadata, _TOP_REQUIRED, set(), "guidance")
    for key, expected in (("schema_version", 1), ("artifact_type", "slot_guidance"),
                          ("coordinate_frame", "rlbench_world"), ("coordinate_units", "m")):
        if metadata[key] != expected or (key == "schema_version" and type(metadata[key]) is not int):
            raise ValueError(f"{key} must be {expected!r}")
    for key in ("episode_id", "frame_id", "instruction", "task_name"):
        _string(metadata[key], key)
    _integer(metadata["source_frame_index"], "source_frame_index")
    observed = _timestamp(metadata["observed_at"], "observed_at")
    if _timestamp(metadata["created_at"], "created_at") < observed:
        raise ValueError("created_at precedes observed_at")
    if metadata["guidance_source"] not in ("oracle_gt", "predicted_coarse_slot", "mock"):
        raise ValueError("unsupported guidance_source")
    checkpoint = metadata["source_checkpoint"]
    if checkpoint is not None:
        _string(checkpoint, "source_checkpoint")
    elif metadata["guidance_source"] == "predicted_coarse_slot":
        raise ValueError("predicted_coarse_slot requires source_checkpoint")
    if not isinstance(metadata["source_git_sha"], str) or not re.fullmatch(
        r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", metadata["source_git_sha"]
    ):
        raise ValueError("source_git_sha must be a full Git SHA")
    _number(metadata["reference_null_probability"], "reference_null_probability", probability=True)
    _validate_transform(metadata["transform"], metadata["source_frame_index"], metadata["observed_at"])
    filename = metadata["points_file"]
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.npz", filename):
        raise ValueError("points_file must be a local .npz basename")
    if not isinstance(metadata["points_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata["points_sha256"]):
        raise ValueError("points_sha256 must be a lowercase SHA-256 digest")
    if set(points) != {f"{role}_points_world" for role in ROLES}:
        raise ValueError("NPZ must contain exactly target_points_world and reference_points_world")
    for role in ROLES:
        item = metadata[role]
        _fields(item, _ROLE_REQUIRED, _ROLE_OPTIONAL, role)
        if item["status"] not in STATUSES:
            raise ValueError(f"{role}.status is unsupported")
        if type(item["semantic_valid"]) is not bool:
            raise ValueError(f"{role}.semantic_valid must be boolean")
        _number(item["confidence"], f"{role}.confidence", probability=True)
        _integer(item["point_count"], f"{role}.point_count")
        key = f"{role}_points_world"
        if item["points_key"] != key:
            raise ValueError(f"{role}.points_key must be {key}")
        cloud = _array(points[key], (None, 3), key)
        if len(cloud) != item["point_count"]:
            raise ValueError(f"{role}.point_count disagrees with NPZ")
        if item["center_world"] is not None:
            _array(item["center_world"], (3,), f"{role}.center_world")
        if ("spread_world" in item) != ("spread_source" in item):
            raise ValueError(f"{role} spread requires a documented spread_source")
        if "spread_world" in item:
            spread = _array(item["spread_world"], (3,), f"{role}.spread_world")
            if np.any(spread <= 0) or item["center_world"] is None:
                raise ValueError(f"{role} spread requires positive half-extents and a center")
        for key in ("spread_source", "semantic_label", "instance_id"):
            if key in item:
                _string(item[key], f"{role}.{key}")
        if item["status"] == "NULL":
            if role != "reference" or item["semantic_valid"] or len(cloud) or item["center_world"] is not None or any(k in item for k in _ROLE_OPTIONAL):
                raise ValueError("NULL is an absent reference, with no instance geometry or identity")
        elif not item["semantic_valid"]:
            raise ValueError(f"{role}: an existing role must be semantic_valid, even if not localized")


def identity_world_transform(*, source: str, timestamp: str, source_frame_index: int) -> dict[str, Any]:
    """Describe a producer whose coordinates are already RLBench world metres."""
    return {"source_frame": "rlbench_world", "target_frame": "rlbench_world",
            "matrix_world_from_source": np.eye(4).tolist(), "source": source,
            "timestamp": timestamp, "source_frame_index": source_frame_index}


def bridgevla_cube_transform(center_world: Any, cube_scale: float, *, source: str,
                             timestamp: str, source_frame_index: int) -> dict[str, Any]:
    """Record x_world = x_cube / cube_scale + center_world.

    Requires the exact place_pc_in_cube inputs. Does not undo renderer norm_corr,
    stage-two cropping, training augmentation, or infer a center from Slot points.
    """
    center = _array(center_world, (3,), "center_world")
    scale = _number(cube_scale, "cube_scale")
    if scale <= 0:
        raise ValueError("cube_scale must be positive")
    matrix = np.eye(4)
    matrix[:3, :3] /= scale
    matrix[:3, 3] = center
    return {"source_frame": "bridgevla_cube", "target_frame": "rlbench_world",
            "matrix_world_from_source": matrix.tolist(), "source": source,
            "timestamp": timestamp, "source_frame_index": source_frame_index,
            "normalization": {"center_world": center.tolist(), "cube_scale": scale}}


def points_to_world(points_source: Any, transform: Mapping[str, Any]) -> np.ndarray:
    """Apply a validated explicit transform once at the producer boundary."""
    _fields(transform, _TRANSFORM_REQUIRED, {"normalization"}, "transform")
    _validate_transform(transform, transform["source_frame_index"], transform["timestamp"])
    points = _array(points_source, (None, 3), "points_source")
    matrix = np.asarray(transform["matrix_world_from_source"], dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class SlotGuidance:
    """Detached artifact; points retain shape (0,3) for empty roles."""

    metadata: dict[str, Any]
    target_points_world: np.ndarray
    reference_points_world: np.ndarray

    def points(self, role: str) -> np.ndarray:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        return getattr(self, f"{role}_points_world")

    def validate(self) -> None:
        validate_guidance(self.metadata, {f"{r}_points_world": self.points(r) for r in ROLES})


def _snapshot(metadata: Mapping[str, Any], points: Mapping[str, np.ndarray]) -> SlotGuidance:
    validate_guidance(metadata, points)
    arrays = [_array(points[f"{r}_points_world"], (None, 3), r) for r in ROLES]
    for array in arrays:
        array.setflags(write=False)
    return SlotGuidance(copy.deepcopy(dict(metadata)), *arrays)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_guidance(path: str | Path, *, expected_source_frame_index: int | None = None) -> SlotGuidance:
    """Load and validate a self-contained artifact, with no observation reads."""
    path = Path(path)
    metadata = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    if not isinstance(metadata, dict):
        raise ValueError("guidance JSON must be an object")
    # Validate JSON before accessing any user-provided filesystem reference.
    empty = {f"{r}_points_world": np.empty((0, 3)) for r in ROLES}
    preflight = copy.deepcopy(metadata)
    for role in ROLES:
        if isinstance(preflight.get(role), dict):
            preflight[role]["point_count"] = 0
    validate_guidance(preflight, empty)
    if expected_source_frame_index is not None:
        _integer(expected_source_frame_index, "expected_source_frame_index")
        if metadata["source_frame_index"] != expected_source_frame_index:
            raise ValueError("guidance does not belong to the requested source_frame_index")
    points_path = path.parent / metadata["points_file"]
    payload = points_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["points_sha256"]:
        raise ValueError("NPZ checksum mismatch; geometry and metadata are not a matched pair")
    # Load exactly the bytes that were hashed, avoiding a file replacement race.
    import io
    try:
        archive = np.load(io.BytesIO(payload), allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("expected an NPZ archive, not a standalone NPY array")
        with archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("duplicate NPZ array names")
            points = {key: archive[key] for key in archive.files}
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid guidance NPZ: {exc}") from exc
    return _snapshot(metadata, points)


def save_guidance(path: str | Path, metadata: Mapping[str, Any], *,
                  target_points_world: Any, reference_points_world: Any,
                  overwrite: bool = False) -> SlotGuidance:
    """Publish NPZ first, JSON last; fill storage keys/counts on a metadata copy.

    A checksum-named sidecar preserves old readers across JSON replacements.
    Existing metadata is never overwritten unless explicitly requested.
    """
    path = Path(path)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", path.stem):
        raise ValueError("guidance filename stem must use ASCII letters, digits, _, . or -")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    points = {"target_points_world": _array(target_points_world, (None, 3), "target_points_world"),
              "reference_points_world": _array(reference_points_world, (None, 3), "reference_points_world")}
    value = copy.deepcopy(dict(metadata))
    for role in ROLES:
        if not isinstance(value.get(role), dict):
            raise ValueError(f"{role} must be an object")
        for key, generated in (("points_key", f"{role}_points_world"),
                               ("point_count", len(points[f"{role}_points_world"]))):
            if key in value[role] and value[role][key] != generated:
                raise ValueError(f"{role}.{key} disagrees with supplied points")
            value[role][key] = generated
    # Invalid inputs must fail before creating output directories or sidecars.
    value.update(points_file="pending.npz", points_sha256="0" * 64)
    validate_guidance(value, points)
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("metadata must contain only finite JSON-serializable values") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary.append(Path(handle.name))
            np.savez_compressed(handle, **points)
        digest = hashlib.sha256(temporary[-1].read_bytes()).hexdigest()
        value.update(points_file=f"{path.stem}.points.{digest}.npz", points_sha256=digest)
        sidecar = path.parent / value["points_file"]
        try:
            os.link(temporary[-1], sidecar)
        except FileExistsError:
            if hashlib.sha256(sidecar.read_bytes()).hexdigest() != digest:
                raise ValueError("existing content-addressed sidecar is corrupt")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".json", delete=False) as handle:
            temporary.append(Path(handle.name))
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        if overwrite:
            os.replace(temporary[-1], path)
        else:
            os.link(temporary[-1], path)
        return _snapshot(value, points)
    finally:
        for item in temporary:
            item.unlink(missing_ok=True)


class MockBackend:
    """Deterministic guidance producer for offline tests, requiring no SAM3.

    This mocks the Guidance boundary, not SAM3 inference or mask quality. An
    actual mask-predicting backend and prompt tests belong to PR2.
    """

    def __init__(self, guidance: SlotGuidance):
        self._guidance = _snapshot(guidance.metadata, {f"{r}_points_world": guidance.points(r) for r in ROLES})

    def predict(self, *, episode_id: str, frame_id: str, source_frame_index: int) -> SlotGuidance:
        _integer(source_frame_index, "source_frame_index")
        expected = (episode_id, frame_id, source_frame_index)
        actual = tuple(self._guidance.metadata[k] for k in ("episode_id", "frame_id", "source_frame_index"))
        if expected != actual:
            raise ValueError("MockBackend identity mismatch: episode/frame/source_frame_index must match")
        return _snapshot(self._guidance.metadata, {f"{r}_points_world": self._guidance.points(r) for r in ROLES})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="slot guidance JSON to validate")
    parser.add_argument("--source-frame-index", type=int, required=True)
    args = parser.parse_args()
    guidance = load_guidance(args.metadata, expected_source_frame_index=args.source_frame_index)
    print(json.dumps({"episode_id": guidance.metadata["episode_id"], "frame_id": guidance.metadata["frame_id"],
                      "source_frame_index": args.source_frame_index,
                      "roles": {r: {"status": guidance.metadata[r]["status"], "point_count": len(guidance.points(r))}
                                for r in ROLES}}, indent=2))


if __name__ == "__main__":
    main()
