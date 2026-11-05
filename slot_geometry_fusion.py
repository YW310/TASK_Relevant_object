"""Current-frame, role-preserving RGB-D fusion of all accepted SAM3 candidates.

Only the supplied FrameSnapshot is used. This module never opens a trajectory,
an observation sequence, future camera poses or expert actions. Coordinate
conversion and robust cross-view distance reuse the existing geometry helpers.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import io
from itertools import combinations
from pathlib import Path
import re
import tempfile
import os
from typing import Any, Mapping
import zipfile

import numpy as np

from camera_geometry import backproject_mask, transform_points
from fusion_matching import symmetric_percentile_nearest_distance
from slot_guidance import ROLES, STATUSES, SlotGuidance, _array, _number, _timestamp
from slot_guided_io import (FrameSnapshot, _canonical, _read_json, atomic_json, calibration_digest,
                            guidance_input_digest, json_value, snapshot_input_digest)
from slot_prompt_adapter import _validate_frame
from slot_guided_sam3 import mask_content_digest


@dataclass(frozen=True)
class FusionConfig:
    min_support_cameras: int = 1
    min_points_per_camera: int = 3
    min_points_per_object: int = 3
    min_depth_fraction: float = 0.5
    voxel_size_m: float = 0.005
    quantile_low: float = 0.02
    quantile_high: float = 0.98
    max_points_per_camera: int = 10000
    max_points_per_object: int = 20000
    cross_view_distance_m: float = 0.04
    cross_view_center_distance_m: float = 0.10
    distance_percentile: float = 75.0
    max_joint_hypotheses: int = 100000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"fusion {key} must be a finite number")
        for key in ("min_support_cameras", "min_points_per_camera", "min_points_per_object",
                    "max_points_per_camera", "max_points_per_object", "max_joint_hypotheses"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"fusion {key} must be a positive integer")
        if not 0 < self.min_depth_fraction <= 1:
            raise ValueError("fusion min_depth_fraction must be in (0,1]")
        if not 0 <= self.quantile_low < 0.5 < self.quantile_high <= 1:
            raise ValueError("fusion quantiles must straddle the median")
        if min(self.voxel_size_m, self.cross_view_distance_m, self.cross_view_center_distance_m) <= 0:
            raise ValueError("fusion metric distances must be positive")
        if not 0 <= self.distance_percentile <= 100:
            raise ValueError("fusion distance_percentile must be in [0,100]")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trim(points: np.ndarray, config: FusionConfig) -> np.ndarray:
    if not len(points):
        return points
    low = np.quantile(points, config.quantile_low, axis=0, method="lower")
    high = np.quantile(points, config.quantile_high, axis=0, method="higher")
    return points[np.all((points >= low) & (points <= high), axis=1)]


def _voxel(points: np.ndarray, size: float, max_points: int) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    scaled = points / size
    if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) >= np.iinfo(np.int64).max:
        raise ValueError("world geometry exceeds voxel index range")
    # Camera transforms can differ by a few ULP at a voxel boundary. Snap only
    # within 1e-9 voxel widths, so physically equal samples do not gain duplicate
    # votes in neighboring bins. This is numeric tolerance, not object padding.
    nearest = np.rint(scaled)
    scaled = np.where(np.abs(scaled - nearest) <= 1e-9, nearest, scaled)
    indices = np.floor(scaled).astype(np.int64)
    _, inverse = np.unique(indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    cloud = np.column_stack([np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)])
    if len(cloud) > max_points:
        cloud = cloud[np.linspace(0, len(cloud) - 1, max_points).astype(int)]
    return cloud


def _validate_binding(snapshot: FrameSnapshot, guidance: SlotGuidance, inference: Mapping[str, Any]) -> None:
    guidance.validate()
    snapshot_fields = {"schema_version", "artifact_type", "episode_id", "frame_id", "source_frame_index", "observed_at", "cameras"}
    if (set(snapshot.metadata) != snapshot_fields or type(snapshot.metadata["schema_version"]) is not int
            or snapshot.metadata["schema_version"] != 1 or snapshot.metadata["artifact_type"] != "slot_rgbd_frame"):
        raise ValueError("fusion requires a strict single-observation snapshot")
    for key in ("episode_id", "frame_id", "source_frame_index", "observed_at"):
        if snapshot.metadata.get(key) != guidance.metadata[key] or inference.get(key) != guidance.metadata[key]:
            raise ValueError(f"snapshot/guidance/inference {key} mismatch")
    if inference.get("artifact_type") != "slot_sam3_frame" or type(inference.get("schema_version")) is not int or inference["schema_version"] != 1:
        raise ValueError("incompatible SAM3 inference artifact")
    if inference.get("coordinate_frame") != "rlbench_world" or inference.get("coordinate_units") != "m":
        raise ValueError("fusion requires RLBench world metres")
    if inference.get("guidance_metadata") != guidance.metadata:
        raise ValueError("inference guidance provenance mismatch")
    if inference.get("guidance_source") != guidance.metadata["guidance_source"]:
        raise ValueError("inference guidance source mismatch")
    if inference.get("calibration_sha256") != calibration_digest(snapshot):
        raise ValueError("stale inference calibration digest")
    if inference.get("snapshot_input_sha256") != snapshot_input_digest(snapshot):
        raise ValueError("stale inference snapshot RGB-D input digest")
    if inference.get("guidance_input_sha256") != guidance_input_digest(guidance):
        raise ValueError("stale inference guidance input digest")
    if set(inference.get("cameras", {})) != set(snapshot.cameras) or not snapshot.cameras:
        raise ValueError("inference/snapshot camera mismatch")
    declared = snapshot.metadata.get("cameras")
    if not isinstance(declared, list) or len(declared) != len(snapshot.cameras):
        raise ValueError("invalid snapshot camera manifest")
    by_name = {row.get("name"): row for row in declared}
    if set(by_name) != set(snapshot.cameras):
        raise ValueError("snapshot camera manifest mismatch")
    for name, frame in snapshot.cameras.items():
        # calibration_digest reads metadata; enforce that the actual camera used
        # for backprojection still matches that same per-observation manifest.
        required_camera = {"name", "rgb_path", "depth_path", "depth_encoding", "source_frame_index", "observed_at", "calibration_source"}
        allowed_camera = required_camera | {"intrinsics", "extrinsics", "near", "far", "instance_masks"}
        if not required_camera.issubset(frame.metadata) or set(frame.metadata) - allowed_camera:
            raise ValueError("fusion camera metadata must describe only the current observation")
        if _canonical(frame.metadata) != _canonical(by_name[name]):
            raise ValueError("snapshot camera metadata changed")
        camera = frame.camera
        if camera.name != name:
            raise ValueError("snapshot camera name mismatch")
        for key in ("source_frame_index", "observed_at", "calibration_source", "intrinsics", "extrinsics"):
            if _canonical(getattr(camera, key)) != _canonical(frame.metadata.get(key)):
                raise ValueError(f"snapshot camera {key} disagrees with calibration metadata")
        error = _validate_frame(guidance, camera, np.asarray(frame.rgb), np.asarray(frame.depth_m))
        # Missing/invalid calibration is represented as explicit failed role
        # prompts in PR2. Such cameras contribute no accepted geometry below.
        if error and any(candidate.get("accepted") is True for row in inference["cameras"][name].get("roles", {}).values()
                         for candidate in row.get("candidates", [])):
            raise ValueError(f"accepted inference candidate has invalid frame: {name}:{error}")
        camera_result = inference["cameras"][name]
        if camera_result.get("camera") != name or set(camera_result.get("roles", {})) != set(ROLES):
            raise ValueError("inference camera/role identity mismatch")
        for role in ROLES:
            if camera_result["roles"][role].get("status") != guidance.metadata[role]["status"]:
                raise ValueError("inference role status mismatch")
            for candidate in camera_result["roles"][role].get("candidates", []):
                if type(candidate.get("accepted")) is not bool:
                    raise ValueError("inference candidate accepted state must be a boolean")


def _candidate_cloud(frame: Any, candidate: Mapping[str, Any], config: FusionConfig):
    mask = np.asarray(candidate.get("mask"))
    depth = np.asarray(frame.depth_m)
    if mask.dtype != bool or mask.shape != depth.shape:
        return None, {"failure_reason": "INVALID_MASK_SHAPE_OR_TYPE"}
    if candidate.get("mask_sha256") != mask_content_digest(mask):
        raise ValueError("accepted candidate mask content digest mismatch")
    count = int(mask.sum())
    if not count:
        return None, {"failure_reason": "EMPTY_MASK"}
    depth_fraction = float((np.isfinite(depth[mask]) & (depth[mask] > 0)).mean())
    if depth_fraction < config.min_depth_fraction:
        return None, {"failure_reason": "INSUFFICIENT_DEPTH_SUPPORT", "depth_support_fraction": depth_fraction}
    quality = candidate.get("quality_score")
    if isinstance(quality, bool) or not isinstance(quality, (float, int)) or not np.isfinite(quality) or not 0 < quality <= 1:
        return None, {"failure_reason": "INVALID_CANDIDATE_QUALITY"}
    points_camera = backproject_mask(depth, mask, np.asarray(frame.camera.intrinsics), config.max_points_per_camera)
    cloud = transform_points(points_camera, np.asarray(frame.camera.extrinsics))
    if not np.isfinite(cloud).all():
        return None, {"failure_reason": "NONFINITE_WORLD_GEOMETRY"}
    original_count = len(cloud)
    trimmed = _trim(cloud, config)
    cloud = _voxel(trimmed, config.voxel_size_m, config.max_points_per_camera)
    if len(cloud) < config.min_points_per_camera:
        return None, {"failure_reason": "INSUFFICIENT_OBJECT_POINTS", "point_count": len(cloud)}
    return cloud, {"failure_reason": None, "depth_support_fraction": depth_fraction,
                   "robust_retained_fraction": len(trimmed) / max(1, original_count),
                   "mask_area_pixels": count, "point_count": len(cloud)}


class _SearchLimit(Exception):
    pass


def _joint_select(nodes: list[dict[str, Any]], expected_center: Any, config: FusionConfig):
    """Exact compatible candidate search with at most one mask per camera.

    Search prefers the number of mutually consistent cameras, then a combined
    geometry/prompt score. An explicit bound fails closed; it never truncates the
    candidate pool by SAM score or silently falls back to greedy selection.
    """
    groups: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        groups.setdefault(node["camera"], []).append(index)
    names = sorted(groups)
    pairwise = {}
    for first, second in combinations(range(len(nodes)), 2):
        a, b = nodes[first], nodes[second]
        if a["camera"] == b["camera"]:
            continue
        surface = symmetric_percentile_nearest_distance(a["cloud"], b["cloud"], config.distance_percentile)
        center = float(np.linalg.norm(a["center"] - b["center"]))
        pairwise[first, second] = (surface, center)

    def distances(first, second):
        return pairwise[min(first, second), max(first, second)]

    best: list[int] = []
    best_score = -1.0
    visited = 0

    def visit(position: int, selected: list[int]):
        nonlocal best, best_score, visited
        visited += 1
        if visited > config.max_joint_hypotheses:
            raise _SearchLimit()
        if len(selected) + len(names) - position < max(len(best), config.min_support_cameras):
            return
        if position == len(names):
            if not selected:
                return
            errors = [distances(a, b)[0] for a, b in combinations(selected, 2)]
            residual = float(np.mean(errors)) if errors else None
            agreement = float(np.exp(-residual / config.cross_view_distance_m)) if residual is not None else 0.5
            quality = float(np.mean([nodes[index]["quality"] for index in selected]))
            if expected_center is not None:
                center_error = float(np.mean([np.linalg.norm(nodes[index]["center"] - expected_center) for index in selected]))
                center_quality = float(np.exp(-center_error / config.cross_view_center_distance_m))
            else:
                center_quality = 0.0
            score = .45 * agreement + .35 * quality + .20 * center_quality
            if len(selected) > len(best) or (len(selected) == len(best) and score > best_score):
                best, best_score = list(selected), score
            return
        for index in groups[names[position]]:
            if all((distances(index, other)[0] <= config.cross_view_distance_m
                    and distances(index, other)[1] <= config.cross_view_center_distance_m) for other in selected):
                visit(position + 1, [*selected, index])
        visit(position + 1, selected)

    try:
        visit(0, [])
    except _SearchLimit:
        return [], {"failure_reason": "CANDIDATE_SEARCH_LIMIT_EXCEEDED", "hypotheses_visited": visited}
    if len(best) < config.min_support_cameras:
        return [], {"failure_reason": "INSUFFICIENT_CONSISTENT_CAMERAS", "hypotheses_visited": visited}
    residuals = [distances(a, b)[0] for a, b in combinations(best, 2)]
    residual = float(np.mean(residuals)) if residuals else None
    return best, {"failure_reason": None, "hypotheses_visited": visited,
                  "cross_view_residual_m": residual,
                  "cross_view_consistency": float(np.exp(-residual / config.cross_view_distance_m)) if residual is not None else None,
                  "joint_score": best_score}


def fuse_geometry(snapshot: FrameSnapshot, guidance: SlotGuidance, inference: dict[str, Any],
                  config: FusionConfig | None = None) -> dict[str, Any]:
    """Return canonical object metadata plus detached ``result['points']``.

    NULL / NOT_VISIBLE / LOW_CONFIDENCE remain semantic states. Empty or
    inconsistent geometry never changes a role's semantic validity or status.
    """
    config = config or FusionConfig()
    _validate_binding(snapshot, guidance, inference)
    fingerprints = inference.get("backend", {})
    if set(fingerprints) != set(snapshot.cameras):
        raise ValueError("backend camera provenance mismatch")
    kinds = {row.get("kind") for row in fingerprints.values()}
    checkpoints = {row.get("checkpoint_sha256") for row in fingerprints.values()}
    if len(kinds) != 1 or not kinds <= {"mock", "sam3"} or len(checkpoints) != 1:
        raise ValueError("fusion requires one consistent mock/SAM3 backend checkpoint")
    kind = next(iter(kinds))
    checkpoint = next(iter(checkpoints))
    if kind == "sam3" and (not isinstance(checkpoint, str) or not re.fullmatch(r"[a-f0-9]{64}", checkpoint)):
        raise ValueError("SAM3 checkpoint digest is required")
    producer_checkpoint = guidance.metadata["source_checkpoint"]
    producer_path = Path(producer_checkpoint) if producer_checkpoint else None
    result = {key: copy.deepcopy(guidance.metadata[key]) for key in
              ("episode_id", "frame_id", "source_frame_index", "observed_at", "guidance_source",
               "source_git_sha", "source_checkpoint", "reference_null_probability")}
    result.update(schema_version=1, artifact_type="slot_object_geometry", coordinate_frame="rlbench_world",
                  coordinate_units="m", geometry_path=None, geometry_sha256=None,
                  calibration_sha256=calibration_digest(snapshot), backend_kind=kind,
                  source_checkpoint_sha256=_file_digest(producer_path) if producer_path and producer_path.is_file() else None,
                  sam3_checkpoint_sha256=checkpoint if kind == "sam3" else None,
                  inference_cache_key=inference.get("cache_key"), sam3_prompt_mode=inference.get("mode"),
                  transform=copy.deepcopy(guidance.metadata["transform"]),
                  fusion_config=asdict(config), roles={}, points={})
    for role in ROLES:
        source = guidance.metadata[role]
        row = {"status": source["status"], "semantic_valid": source["semantic_valid"],
               "role_confidence": float(source["confidence"]), "geometry_confidence": 0.0,
               "segmentation_quality": 0.0, "geometric_quality": 0.0,
               "points_key": f"{role}_points_world", "point_count": 0,
               "center_world": None, "bbox3d_world": None, "support_cameras": [],
               "failure_reason": source["status"] if source["status"] != "VISIBLE" else "NO_ACCEPTED_CANDIDATES",
               "selected_candidates": [], "candidate_diagnostics": [],
               "camera_failures": {name: camera_row["roles"][role].get("failure_reason")
                                   for name, camera_row in inference["cameras"].items()
                                   if camera_row["roles"][role].get("failure_reason") is not None},
               "semantic_label": source.get("semantic_label"), "instance_id": source.get("instance_id"),
               "cross_view_consistency": None, "cross_view_residual_m": None}
        result["roles"][role] = row
        result["points"][row["points_key"]] = np.empty((0, 3), dtype=np.float64)
        if source["status"] != "VISIBLE":
            continue
        nodes = []
        for camera_name, frame in sorted(snapshot.cameras.items()):
            role_inference = inference["cameras"][camera_name]["roles"][role]
            for index, candidate in enumerate(role_inference.get("candidates", [])):
                diagnostic = {"camera": camera_name, "candidate_index": index,
                              "sam3_score": candidate.get("sam3_score"),
                              "quality_score": candidate.get("quality_score"), "selected": False}
                row["candidate_diagnostics"].append(diagnostic)
                if candidate.get("accepted") is not True:
                    diagnostic.update(failure_reason="SAM3_CANDIDATE_REJECTED", sam3_failure_reasons=candidate.get("failure_reasons", []))
                    continue
                if not role_inference.get("prompt", {}).get("prompt_valid"):
                    raise ValueError("accepted candidate requires a valid role prompt")
                cloud, geometry_metrics = _candidate_cloud(frame, candidate, config)
                diagnostic.update(geometry_metrics)
                if cloud is None:
                    continue
                nodes.append({"camera": camera_name, "candidate_index": index, "cloud": cloud,
                              "center": np.median(cloud, axis=0), "quality": float(candidate["quality_score"]),
                              "metrics": geometry_metrics, "diagnostic": diagnostic})
        if not nodes:
            continue
        selected, selection = _joint_select(nodes, source["center_world"], config)
        row["selection_diagnostics"] = selection
        if not selected:
            row["failure_reason"] = selection["failure_reason"]
            for node in nodes:
                node["diagnostic"]["failure_reason"] = selection["failure_reason"]
            continue
        selected_set = set(selected)
        for index, node in enumerate(nodes):
            node["diagnostic"]["selected"] = index in selected_set
            if index not in selected_set:
                node["diagnostic"]["failure_reason"] = "NOT_IN_CONSISTENT_JOINT_SELECTION"
        raw = np.concatenate([nodes[index]["cloud"] for index in selected])
        trimmed = _trim(raw, config)
        cloud = _voxel(trimmed, config.voxel_size_m, config.max_points_per_object)
        if len(cloud) < config.min_points_per_object:
            row["failure_reason"] = "INSUFFICIENT_FUSED_POINTS"
            continue
        center = np.median(cloud, axis=0)
        residual = selection["cross_view_residual_m"]
        cross_quality = selection["cross_view_consistency"] if residual is not None else .5
        depth_quality = float(np.mean([nodes[index]["metrics"]["depth_support_fraction"] for index in selected]))
        retained_quality = len(trimmed) / max(1, len(raw))
        center_quality = float(np.exp(-np.linalg.norm(center - source["center_world"]) / config.cross_view_center_distance_m)) if source["center_world"] is not None else 0.0
        geometric_quality = float(depth_quality * retained_quality * (.5 + .25 * cross_quality + .25 * center_quality))
        segmentation_quality = float(np.mean([nodes[index]["quality"] for index in selected]))
        row.update(point_count=len(cloud), center_world=center.tolist(),
                   bbox3d_world=np.stack((cloud.min(axis=0), cloud.max(axis=0))).tolist(),
                   support_cameras=sorted(nodes[index]["camera"] for index in selected),
                   geometry_confidence=geometric_quality * segmentation_quality,
                   geometric_quality=geometric_quality, segmentation_quality=segmentation_quality,
                   failure_reason=None, cross_view_residual_m=residual,
                   cross_view_consistency=selection["cross_view_consistency"],
                   selected_candidates=[{"camera": nodes[index]["camera"], "candidate_index": nodes[index]["candidate_index"]}
                                        for index in selected])
        result["points"][row["points_key"]] = cloud
    _validate_geometry(result, result["points"], storage=False)
    return result


def _validate_geometry(value: Mapping[str, Any], points: Mapping[str, Any], *, storage: bool) -> None:
    required = {"schema_version", "artifact_type", "episode_id", "frame_id", "source_frame_index", "observed_at",
                "coordinate_frame", "coordinate_units", "geometry_path", "geometry_sha256", "calibration_sha256",
                "guidance_source", "backend_kind", "source_git_sha", "source_checkpoint", "source_checkpoint_sha256",
                "sam3_checkpoint_sha256", "reference_null_probability", "roles"}
    if not required.issubset(value):
        raise ValueError(f"geometry metadata missing {sorted(required - value.keys())}")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["artifact_type"] != "slot_object_geometry":
        raise ValueError("incompatible object geometry schema")
    for key, expected in (("coordinate_frame", "rlbench_world"), ("coordinate_units", "m")):
        if value[key] != expected:
            raise ValueError(f"object geometry {key} must be {expected}")
    for key in ("episode_id", "frame_id"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"object geometry {key} is required")
    if type(value["source_frame_index"]) is not int or value["source_frame_index"] < 0:
        raise ValueError("object geometry needs exact source_frame_index")
    _timestamp(value["observed_at"], "geometry.observed_at")
    _number(value["reference_null_probability"], "reference_null_probability", probability=True)
    if value["guidance_source"] not in ("oracle_gt", "predicted_coarse_slot", "mock") or value["backend_kind"] not in ("mock", "sam3"):
        raise ValueError("invalid geometry producer/backend provenance")
    if not isinstance(value["source_git_sha"], str) or not re.fullmatch(r"[a-f0-9]{40}", value["source_git_sha"]):
        raise ValueError("geometry source_git_sha must be a full SHA")
    source_checkpoint = value["source_checkpoint"]
    if source_checkpoint is not None and (not isinstance(source_checkpoint, str) or not source_checkpoint.strip()):
        raise ValueError("source_checkpoint must be a path or null")
    if value["guidance_source"] == "predicted_coarse_slot" and source_checkpoint is None:
        raise ValueError("predicted guidance needs source checkpoint provenance")
    for key in ("geometry_sha256", "calibration_sha256", "source_checkpoint_sha256", "sam3_checkpoint_sha256"):
        digest = value[key]
        required_digest = key == "calibration_sha256" or (storage and key == "geometry_sha256") or (key == "sam3_checkpoint_sha256" and value["backend_kind"] == "sam3")
        if (digest is None and required_digest) or (digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest))):
            raise ValueError(f"invalid {key}")
    if value["backend_kind"] == "mock" and value["sam3_checkpoint_sha256"] is not None:
        raise ValueError("mock geometry cannot claim a SAM3 checkpoint")
    if value["source_checkpoint_sha256"] is not None and source_checkpoint is None:
        raise ValueError("checkpoint digest needs checkpoint provenance")
    if storage and (not isinstance(value["geometry_path"], str)
                    or value["geometry_path"] != f"geometry.{value['geometry_sha256']}.npz"):
        raise ValueError("geometry_path must be a local content-addressed NPZ basename")
    if set(value["roles"]) != set(ROLES) or set(points) != {f"{role}_points_world" for role in ROLES}:
        raise ValueError("object geometry must contain exactly target and reference arrays")
    for role in ROLES:
        row = value["roles"][role]
        role_required = {"status", "semantic_valid", "geometry_confidence", "segmentation_quality", "geometric_quality",
                         "role_confidence", "points_key", "point_count", "center_world", "bbox3d_world", "support_cameras", "failure_reason"}
        if not isinstance(row, Mapping) or not role_required.issubset(row):
            raise ValueError("incomplete object role geometry")
        if row["status"] not in STATUSES or (row["status"] == "NULL" and role != "reference"):
            raise ValueError("invalid object role status")
        if type(row["semantic_valid"]) is not bool or row["semantic_valid"] != (row["status"] != "NULL"):
            raise ValueError("object role semantic presence/status mismatch")
        for key in ("geometry_confidence", "segmentation_quality", "geometric_quality", "role_confidence"):
            _number(row[key], f"{role}.{key}", probability=True)
        key = f"{role}_points_world"
        if row["points_key"] != key:
            raise ValueError("object role points_key mismatch")
        cloud = _array(points[key], (None, 3), key)
        if type(row["point_count"]) is not int or row["point_count"] != len(cloud):
            raise ValueError("object role point_count mismatch")
        support = row["support_cameras"]
        if not isinstance(support, list) or any(not isinstance(item, str) or not item.strip() for item in support) or len(set(support)) != len(support):
            raise ValueError("invalid support_cameras")
        failure = row["failure_reason"]
        if failure is not None and (not isinstance(failure, str) or not failure.strip()):
            raise ValueError("failure_reason must be a string or null")
        if len(cloud):
            if row["status"] != "VISIBLE" or not support or failure is not None or row["geometry_confidence"] <= 0:
                raise ValueError("nonempty geometry needs visible valid supported role")
            center = _array(row["center_world"], (3,), "center_world")
            bbox = _array(row["bbox3d_world"], (2, 3), "bbox3d_world")
            if not np.allclose(center, np.median(cloud, axis=0), rtol=1e-7, atol=1e-9):
                raise ValueError("center_world disagrees with stored robust centroid")
            if not np.allclose(bbox, np.stack((cloud.min(axis=0), cloud.max(axis=0))), rtol=1e-7, atol=1e-9):
                raise ValueError("bbox3d_world disagrees with stored points")
        elif row["center_world"] is not None or row["bbox3d_world"] is not None or support or row["geometry_confidence"] != 0 or failure is None:
            raise ValueError("empty geometry needs explicit failure and no fabricated bounds/support")


def save_geometry(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish JSON after the checksum-named two-array NPZ."""
    path = Path(path)
    value = copy.deepcopy(dict(result))
    points = value.pop("points")
    _validate_geometry(value, points, storage=False)
    arrays = {key: _array(array, (None, 3), key) for key, array in points.items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    value.update(geometry_sha256=digest, geometry_path=f"geometry.{digest}.npz")
    _validate_geometry(value, arrays, storage=True)
    # Validate all metadata serialization before publishing either artifact.
    _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path.parent / value["geometry_path"])
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(path, value)
    return json_value(value)


def load_geometry(path: str | Path, *, expected_source_frame_index: int | None = None,
                  expected_episode_id: str | None = None, expected_frame_id: str | None = None,
                  expected_observed_at: str | None = None,
                  expected_calibration_sha256: str | None = None) -> dict[str, Any]:
    """Validate detached object geometry and optional current-frame bindings."""
    path = Path(path)
    value = _read_json(path)
    for key, expected in (("source_frame_index", expected_source_frame_index), ("episode_id", expected_episode_id),
                          ("frame_id", expected_frame_id), ("observed_at", expected_observed_at),
                          ("calibration_sha256", expected_calibration_sha256)):
        if expected is not None and value.get(key) != expected:
            raise ValueError(f"object geometry {key} mismatch")
    filename, digest = value.get("geometry_path"), value.get("geometry_sha256")
    if not isinstance(filename, str) or not re.fullmatch(r"geometry\.[a-f0-9]{64}\.npz", filename) or filename != f"geometry.{digest}.npz":
        raise ValueError("geometry_path must be a local content-addressed NPZ basename")
    payload = (path.parent / filename).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("object geometry digest mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sorted(archive.namelist()) != ["reference_points_world.npy", "target_points_world.npy"]:
            raise ValueError("geometry NPZ must contain exactly two role arrays")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        points = {key: np.array(archive[key], copy=True) for key in archive.files}
    _validate_geometry(value, points, storage=True)
    for key, cloud in points.items():
        cloud = _array(cloud, (None, 3), key)
        cloud.setflags(write=False)
        points[key] = cloud
    value["points"] = points
    return value
