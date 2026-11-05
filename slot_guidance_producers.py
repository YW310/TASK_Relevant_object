"""Current-frame Oracle and detached coarse-slot Guidance exporters.

These functions perform no model inference, checkpoint loading or episode reads.
A coarse caller must capture the actual rendered/cube/crop inverse and a
per-point support mask at inference time. A checkpoint path identifies the
producer; it does not turn this exporter into a checkpoint runner.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from camera_geometry import backproject_mask, transform_points
from slot_guidance import (ROLES, SlotGuidance, _timestamp, _validate_transform,
                           identity_world_transform, points_to_world)
from slot_guided_io import FrameSnapshot, calibration_digest
from slot_prompt_adapter import _validate_frame


def _numeric(value: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    if getattr(value, "requires_grad", False) or getattr(value, "grad_fn", None) is not None:
        raise ValueError(f"{name} must be detached before export")
    try:
        array = np.asarray(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be detached CPU numeric data") from exc
    if (array.dtype.kind not in "fiu" or array.ndim != len(shape)
            or any(size != expected for size, expected in zip(array.shape, shape) if expected is not None)):
        raise ValueError(f"{name} must have real numeric shape {shape}")
    return np.array(array, dtype=np.float64, copy=True)


def _boolean(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if getattr(value, "requires_grad", False) or getattr(value, "grad_fn", None) is not None:
        raise ValueError(f"{name} must be detached before export")
    try:
        array = np.asarray(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be detached boolean data") from exc
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{name} must have boolean shape {shape}")
    return np.array(array, copy=True)


def _validate_snapshot(snapshot: FrameSnapshot) -> None:
    required = {"schema_version", "artifact_type", "episode_id", "frame_id", "source_frame_index", "observed_at", "cameras"}
    metadata = snapshot.metadata
    if (set(metadata) != required or type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != 1 or metadata["artifact_type"] != "slot_rgbd_frame"):
        raise ValueError("expected a strict current-frame RGB-D snapshot")
    if type(metadata["source_frame_index"]) is not int or metadata["source_frame_index"] < 0:
        raise ValueError("snapshot requires an explicit source observation index")
    for key in ("episode_id", "frame_id"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"snapshot {key} is required")
    _timestamp(metadata["observed_at"], "snapshot.observed_at")
    if not snapshot.cameras or not isinstance(metadata["cameras"], list):
        raise ValueError("snapshot must contain current camera records")
    names = [row.get("name") for row in metadata["cameras"] if isinstance(row, Mapping)]
    if len(names) != len(metadata["cameras"]) or len(set(names)) != len(names) or set(names) != set(snapshot.cameras):
        raise ValueError("snapshot camera records do not match current arrays")
    for name, frame in snapshot.cameras.items():
        if name != frame.camera.name:
            raise ValueError("snapshot camera name mismatch")
        # Reuse exactly the prompt path's rigid/pinhole/frame/shape validation.
        # FrameSnapshot and SlotGuidance both expose current-frame metadata.
        error = _validate_frame(snapshot, frame.camera, np.asarray(frame.rgb), np.asarray(frame.depth_m))
        if error:
            raise ValueError(f"current camera {name}: {error}")
        for row in (frame.metadata, next(row for row in metadata["cameras"] if row["name"] == name)):
            if (type(row.get("source_frame_index")) is not int
                    or row.get("source_frame_index") != metadata["source_frame_index"]
                    or row.get("observed_at") != metadata["observed_at"]):
                raise ValueError("camera provenance refers to a different observation")
            if row.get("name") != name or row.get("calibration_source") != frame.camera.calibration_source:
                raise ValueError("camera calibration provenance does not match current arrays")
            for key in ("intrinsics", "extrinsics"):
                if not np.array_equal(np.asarray(row.get(key)), np.asarray(getattr(frame.camera, key))):
                    raise ValueError("camera calibration metadata does not match current arrays")


def _metadata(snapshot: FrameSnapshot, *, instruction: str, task_name: str,
              source: str, checkpoint: str | None, git_sha: str, created_at: str,
              transform: Mapping[str, Any], null_probability: float) -> dict[str, Any]:
    return {
        "schema_version": 1, "artifact_type": "slot_guidance",
        **{key: snapshot.metadata[key] for key in ("episode_id", "frame_id", "source_frame_index", "observed_at")},
        "instruction": instruction, "task_name": task_name,
        "coordinate_frame": "rlbench_world", "coordinate_units": "m",
        "created_at": created_at, "guidance_source": source,
        "source_checkpoint": checkpoint, "source_git_sha": git_sha,
        "transform": copy.deepcopy(dict(transform)), "reference_null_probability": null_probability,
        # These are in-memory placeholders; save_guidance fills a matched digest
        # and content-addressed sidecar. They are never used to load geometry.
        "points_file": "pending.npz", "points_sha256": "0" * 64,
    }


def _role(role: str, points: np.ndarray, *, status: str, confidence: float,
          center_valid: bool = True) -> dict[str, Any]:
    return {"status": status, "semantic_valid": status != "NULL", "confidence": float(confidence),
            "center_world": np.median(points, axis=0).tolist() if len(points) and center_valid and status != "NULL" else None,
            "points_key": f"{role}_points_world", "point_count": len(points)}


def _finish(metadata: dict[str, Any], points: dict[str, np.ndarray]) -> SlotGuidance:
    arrays = []
    for role in ROLES:
        array = np.array(points[role], dtype=np.float64, copy=True).reshape(-1, 3)
        array.setflags(write=False)
        arrays.append(array)
    result = SlotGuidance(copy.deepcopy(metadata), *arrays)
    result.validate()
    return result


def oracle_guidance(snapshot: FrameSnapshot, role_masks: Mapping[str, Mapping[str, np.ndarray]], *,
                    instruction: str, task_name: str, reference_is_null: bool,
                    source_git_sha: str, created_at: str,
                    role_instance_ids: Mapping[str, str] | None = None) -> SlotGuidance:
    """Backproject explicit GT role masks from every annotated current view.

    Every existing role needs an original-resolution boolean mask in every
    snapshot camera. Explicit empty GT masks establish NOT_VISIBLE. A visible
    mask with no finite positive depth establishes LOW_CONFIDENCE for geometry.
    NULL is accepted only as the supplied semantic decision, never from emptiness.
    """
    _validate_snapshot(snapshot)
    if type(reference_is_null) is not bool:
        raise ValueError("reference_is_null must be an explicit boolean semantic decision")
    if not isinstance(role_masks, Mapping) or set(role_masks) != set(snapshot.cameras):
        raise ValueError("GT role masks must explicitly annotate every current camera")
    ids = dict(role_instance_ids or {})
    if set(ids) - set(ROLES) or any(not isinstance(value, str) or not value.strip() for value in ids.values()):
        raise ValueError("role_instance_ids must map known roles to nonempty identifiers")
    if reference_is_null and "reference" in ids:
        raise ValueError("NULL reference cannot have an instance identity")
    clouds: dict[str, list[np.ndarray]] = {role: [] for role in ROLES}
    visible = {role: False for role in ROLES}
    for name, frame in snapshot.cameras.items():
        masks = role_masks[name]
        required = {"target"} if reference_is_null else set(ROLES)
        if not isinstance(masks, Mapping) or not required.issubset(masks) or set(masks) - set(ROLES):
            raise ValueError("GT masks require explicit target/reference annotations")
        for role in ROLES:
            if role not in masks:
                continue
            mask = _boolean(masks[role], frame.depth_m.shape, f"{name}/{role} mask")
            if role == "reference" and reference_is_null:
                if mask.any():
                    raise ValueError("NULL reference contradicts a nonempty GT reference mask")
                continue
            visible[role] |= bool(mask.any())
            cam_points = backproject_mask(frame.depth_m, mask, np.asarray(frame.camera.intrinsics), max_points=0)
            clouds[role].append(transform_points(cam_points, np.asarray(frame.camera.extrinsics)))
    stamp, index = snapshot.metadata["observed_at"], snapshot.metadata["source_frame_index"]
    provenance = "current GT role masks + metric depth; calibration_sha256=" + calibration_digest(snapshot)
    meta = _metadata(snapshot, instruction=instruction, task_name=task_name, source="oracle_gt", checkpoint=None,
                     git_sha=source_git_sha, created_at=created_at, null_probability=float(reference_is_null),
                     transform=identity_world_transform(source=provenance, timestamp=stamp, source_frame_index=index))
    points = {}
    for role in ROLES:
        points[role] = np.concatenate(clouds[role], axis=0) if clouds[role] else np.empty((0, 3))
        status = ("NULL" if role == "reference" and reference_is_null else
                  "VISIBLE" if len(points[role]) else "LOW_CONFIDENCE" if visible[role] else "NOT_VISIBLE")
        meta[role] = _role(role, points[role], status=status, confidence=0. if status == "LOW_CONFIDENCE" else 1.)
        if role in ids and status != "NULL":
            meta[role]["instance_id"] = ids[role]
        if len(points[role]):
            extent = np.max(np.abs(points[role] - np.asarray(meta[role]["center_world"])), axis=0)
            # Planar observed surfaces have a zero extent. Do not invent a third
            # dimension just to enable center+range fallback.
            if np.all(extent > 0):
                meta[role].update(spread_world=extent.tolist(),
                    spread_source="measured_world_aabb_half_extents:current GT-masked observed surfaces only")
    return _finish(meta, points)


def coarse_slot_guidance(snapshot: FrameSnapshot, slot_output: Mapping[str, Any], *,
                         point_valid: np.ndarray, transform: Mapping[str, Any],
                         instruction: str, task_name: str, source_checkpoint: str,
                         source_git_sha: str, created_at: str, point_support_source: str,
                         reference_null_threshold: float = .75, reference_exists: bool | None = None,
                         role_visibility: Mapping[str, str] | None = None,
                         min_confidence: float = .25) -> SlotGuidance:
    """Export one detached coarse sample with its *actual* complete inverse.

    Required model fields are points[2,P,3], confidence[2], role_token_valid[2],
    valid[2] and scalar reference_null_probability. Optional geometry[17] is
    shape checked but never supplies extents or a replacement XYZ transform.
    Explicit support, finite XYZ and renderer sentinel filtering happen BEFORE
    inverse transformation. Genuine world-frame zero coordinates are retained.

    ``reference_exists`` overrides the posterior as an explicit semantic claim.
    Without it, only p_null >= reference_null_threshold selects NULL. Invalid
    geometry alone is LOW_CONFIDENCE; NOT_VISIBLE needs current visibility evidence.
    """
    _validate_snapshot(snapshot)
    if not isinstance(slot_output, Mapping):
        raise ValueError("slot_output must be one unbatched mapping")
    required = {"points", "confidence", "role_token_valid", "valid", "reference_null_probability"}
    if not required.issubset(slot_output):
        raise ValueError(f"missing coarse fields: {sorted(required - set(slot_output))}")
    if not isinstance(point_support_source, str) or not point_support_source.strip():
        raise ValueError("point_support_source must document current per-point support")
    if reference_exists is not None and type(reference_exists) is not bool:
        raise ValueError("reference_exists must be an explicit boolean or None")
    for name, value in (("reference_null_threshold", reference_null_threshold), ("min_confidence", min_confidence)):
        number = _numeric(value, (), name).item()
        if not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"{name} must be a probability")
    visibility = dict(role_visibility or {})
    if set(visibility) - set(ROLES) or any(value not in ("VISIBLE", "NOT_VISIBLE", "UNKNOWN") for value in visibility.values()):
        raise ValueError("role_visibility requires known roles and VISIBLE/NOT_VISIBLE/UNKNOWN")
    raw = _numeric(slot_output["points"], (2, None, 3), "points")
    support = _boolean(point_valid, raw.shape[:2], "point_valid")
    confidence = _numeric(slot_output["confidence"], (2,), "confidence")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must contain finite probabilities")
    token_valid = _boolean(slot_output["role_token_valid"], (2,), "role_token_valid")
    geometry_valid = _boolean(slot_output["valid"], (2,), "valid")
    p_null = _numeric(slot_output["reference_null_probability"], (), "reference_null_probability").item()
    if not np.isfinite(p_null) or not 0 <= p_null <= 1:
        raise ValueError("reference_null_probability must be a finite probability")
    if slot_output.get("geometry") is not None:
        geometry = _numeric(slot_output["geometry"], (17,), "geometry")
        if not np.isfinite(geometry).all():
            raise ValueError("geometry must be finite when supplied")
    inverse = copy.deepcopy(dict(transform))
    _validate_transform(inverse, snapshot.metadata["source_frame_index"], snapshot.metadata["observed_at"])
    inverse["source"] += "; point_support_source=" + point_support_source + "; calibration_sha256=" + calibration_digest(snapshot)
    meta = _metadata(snapshot, instruction=instruction, task_name=task_name, source="predicted_coarse_slot",
                     checkpoint=source_checkpoint, git_sha=source_git_sha, created_at=created_at,
                     transform=inverse, null_probability=p_null)
    null_reference = reference_exists is False or (reference_exists is None and p_null >= reference_null_threshold)
    points = {}
    for index, role in enumerate(ROLES):
        keep = support[index] & np.isfinite(raw[index]).all(axis=1)
        if inverse["source_frame"] != "rlbench_world":
            keep &= ~np.all(raw[index] == 0, axis=1) & ~np.all(raw[index] == -1, axis=1)
        points[role] = points_to_world(raw[index, keep], inverse)
        reliable = bool(token_valid[index]) and confidence[index] >= min_confidence
        if role == "reference" and null_reference:
            points[role] = np.empty((0, 3))
            status = "NULL"
        elif not reliable:
            status = "LOW_CONFIDENCE"
        elif visibility.get(role, "UNKNOWN") == "NOT_VISIBLE":
            status = "NOT_VISIBLE"
        elif geometry_valid[index] and len(points[role]):
            status = "VISIBLE"
        else:
            status = "LOW_CONFIDENCE"
        meta[role] = _role(role, points[role], status=status, confidence=confidence[index],
                           center_valid=bool(geometry_valid[index]) and reliable)
    return _finish(meta, points)
