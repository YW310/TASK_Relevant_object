"""Causal RLBench world/metre guidance -> original RGB pixel prompts.

No learned model, mask resize, scene loader or action is involved. A prompt is
valid only when role semantics, exact frame binding, calibration and measured
depth support agree. ``extrinsics`` always means ``T_world_camera``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np

from camera_geometry import project_points
from slot_guidance import ROLES, SlotGuidance


@dataclass(frozen=True)
class CameraFrame:
    """Current RGB-D/calibration binding; image_size is (width, height)."""

    name: str
    episode_id: str
    frame_id: str
    source_frame_index: int
    observed_at: str
    intrinsics: np.ndarray | None
    extrinsics: np.ndarray | None
    image_size: tuple[int, int] | None
    calibration_source: str | None


@dataclass(frozen=True)
class AdapterConfig:
    min_confidence: float = 0.25
    min_support: int = 3
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    padding_px: float = 4.0
    padding_fraction: float = 0.1
    depth_tolerance_m: float = 0.03
    min_visible_fraction: float = 0.5
    max_positive_points: int = 3

    def validate(self) -> None:
        values = (self.min_confidence, self.quantile_low, self.quantile_high,
                  self.padding_px, self.padding_fraction, self.depth_tolerance_m,
                  self.min_visible_fraction)
        if any(isinstance(x, (bool, np.bool_))
               or not isinstance(x, (int, float, np.integer, np.floating))
               or not np.isfinite(x) for x in values):
            raise ValueError("adapter thresholds must be finite numbers")
        if not 0 <= self.min_confidence <= 1 or not 0 < self.min_visible_fraction <= 1:
            raise ValueError("confidence/visibility thresholds must be probabilities")
        if not 0 <= self.quantile_low < 0.5 < self.quantile_high <= 1:
            raise ValueError("quantiles must straddle the median")
        if min(self.padding_px, self.padding_fraction) < 0 or self.depth_tolerance_m <= 0:
            raise ValueError("padding must be nonnegative and depth tolerance positive")
        if any(type(x) is not int or x <= 0 for x in (self.min_support, self.max_positive_points)):
            raise ValueError("support/positive point counts must be positive integers")


@dataclass
class RolePrompt:
    role: str
    camera: str
    bbox_xyxy: list[float] | None = None
    positive_points_xy: list[list[float]] = field(default_factory=list)
    negative_points_xy: list[list[float]] = field(default_factory=list)
    prompt_valid: bool = False
    confidence: float = 0.0
    failure_reason: str | None = None
    support: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("camera timestamp needs an explicit timezone")
    return parsed


def _validate_frame(guidance: SlotGuidance, camera: CameraFrame,
                    image: np.ndarray, depth: np.ndarray) -> str | None:
    meta = guidance.metadata
    if (not isinstance(camera.name, str) or not camera.name.strip()
            or camera.episode_id != meta["episode_id"]
            or camera.frame_id != meta["frame_id"]
            or type(camera.source_frame_index) is not int
            or camera.source_frame_index != meta["source_frame_index"]):
        return "FRAME_MISMATCH"
    try:
        if _time(camera.observed_at) != _time(meta["observed_at"]):
            return "FRAME_MISMATCH"
    except (AttributeError, TypeError, ValueError):
        return "FRAME_MISMATCH"
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return "INVALID_RGB"
    h, w = image.shape[:2]
    if h < 2 or w < 2 or camera.image_size != (w, h):
        return "IMAGE_SIZE_MISMATCH"
    if depth.shape != (h, w) or depth.dtype.kind not in "fiu":
        return "INVALID_DEPTH"
    if (camera.intrinsics is None or camera.extrinsics is None
            or not isinstance(camera.calibration_source, str)
            or not camera.calibration_source.strip()):
        return "MISSING_CALIBRATION"
    try:
        k, t = np.asarray(camera.intrinsics, dtype=float), np.asarray(camera.extrinsics, dtype=float)
        if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(t).all():
            return "INVALID_CALIBRATION"
        # The reused camera helper implements a pinhole camera with zero skew.
        if (k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1], rtol=0, atol=1e-10)
                or not np.allclose(t[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), rtol=0, atol=1e-7)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1, rtol=0, atol=1e-7)):
            return "INVALID_CALIBRATION"
    except (TypeError, ValueError):
        return "INVALID_CALIBRATION"
    return None


def _trim(points: np.ndarray, config: AdapterConfig) -> np.ndarray:
    # Outward-rounded empirical quantiles retain sparse clouds, but reject tails
    # once enough samples exist. A single extreme point cannot set a large box.
    low = np.quantile(points, config.quantile_low, axis=0, method="lower")
    high = np.quantile(points, config.quantile_high, axis=0, method="higher")
    return points[np.all((points >= low) & (points <= high), axis=1)]


def _camera_depth(points: np.ndarray, camera: CameraFrame) -> np.ndarray:
    inverse = np.linalg.inv(np.asarray(camera.extrinsics, dtype=float))
    return (points @ inverse[:3, :3].T + inverse[:3, 3])[:, 2]


def _box(xy: np.ndarray, image_size: tuple[int, int], config: AdapterConfig) -> list[float] | None:
    low, high = xy.min(axis=0), xy.max(axis=0)
    # A supported sparse line/point may still form a box through explicit padding.
    padding = config.padding_px + (high - low) * config.padding_fraction
    low, high = low - padding, high + padding
    w, h = image_size
    low = np.clip(low, [0, 0], [w - 1, h - 1])
    high = np.clip(high, [0, 0], [w - 1, h - 1])
    if np.any(high <= low):
        return None
    return [float(low[0]), float(low[1]), float(high[0]), float(high[1])]


def _interior_points(xy: np.ndarray, count: int) -> list[list[float]]:
    # Select only *measured* surface pixels, nearest the robust center first.
    # Further positives stay in the central region and spread across that region.
    xy = np.unique(np.rint(xy).astype(int), axis=0).astype(float)
    center = np.median(xy, axis=0)
    half_extent = np.maximum((xy.max(axis=0) - xy.min(axis=0)) / 2, 1)
    central = np.all(np.abs(xy - center) / half_extent <= 0.8, axis=1)
    candidates = xy[central] if central.any() else xy
    distance = np.sum(((candidates - center) / half_extent) ** 2, axis=1)
    selected = [int(np.argmin(distance))]
    while len(selected) < min(count, len(candidates)):
        separation = np.min(np.sum((candidates[:, None] - candidates[selected][None]) ** 2, axis=2), axis=1)
        separation[selected] = -1
        selected.append(int(np.argmax(separation)))
    return candidates[selected].tolist()


def _cloud_support(points: np.ndarray, camera: CameraFrame, depth: np.ndarray,
                   config: AdapterConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str | None]:
    xy, positive = project_points(points, np.asarray(camera.intrinsics), np.asarray(camera.extrinsics))
    h, w = depth.shape
    bounds = positive & np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
    info = {"trimmed_points": len(points), "positive_depth_points": int(positive.sum()),
            "in_bounds_points": int(bounds.sum())}
    empty_xy, empty_world = np.empty((0, 2)), np.empty((0, 3))
    if not positive.any():
        return empty_xy, empty_world, info, "BEHIND_CAMERA"
    if not bounds.any() or bounds.mean() < config.min_visible_fraction:
        return empty_xy, empty_world, info, "OUTSIDE_IMAGE"
    indices = np.flatnonzero(bounds)
    pixels = np.rint(xy[bounds]).astype(int)
    scene_z = depth[pixels[:, 1], pixels[:, 0]]
    point_z = _camera_depth(points[bounds], camera)
    measured = np.isfinite(scene_z) & (scene_z > 0)
    surface = measured & (np.abs(point_z - scene_z) <= config.depth_tolerance_m)
    info.update(valid_depth_points=int(measured.sum()), depth_supported_points=int(surface.sum()),
                visible_fraction=float(surface.sum() / len(indices)))
    if not measured.any():
        return empty_xy, empty_world, info, "INVALID_DEPTH"
    if not surface.any() and np.any(measured & (point_z > scene_z + config.depth_tolerance_m)):
        return empty_xy, empty_world, info, "OCCLUDED"
    if surface.mean() < config.min_visible_fraction:
        return empty_xy, empty_world, info, "DEPTH_UNSUPPORTED"
    supported_xy = np.rint(xy[indices[surface]])
    if len(np.unique(np.rint(supported_xy).astype(int), axis=0)) < config.min_support:
        return empty_xy, empty_world, info, "INSUFFICIENT_SUPPORT"
    return supported_xy, points[indices[surface]], info, None


_VERIFIED_SPREAD_PREFIXES = ("verified_world_aabb_half_extents:", "oracle_gt_world_aabb_half_extents:",
                             "measured_world_aabb_half_extents:")


def _spread_support(item: dict[str, Any], camera: CameraFrame, depth: np.ndarray,
                    config: AdapterConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str | None]:
    """A provenance-attested world AABB must contain measured current surfaces.

    The caller supplies positive half-extents, not Slot standard deviations.
    A verified source uses ``<kind>_world_aabb_half_extents:<provenance>``. This
    attestation is retained in diagnostics; it cannot authenticate a producer.
    """
    empty_xy, empty_world = np.empty((0, 2)), np.empty((0, 3))
    info: dict[str, Any] = {"geometry_source": "verified_spread"}
    if item.get("center_world") is None or "spread_world" not in item:
        return empty_xy, empty_world, empty_xy, info, "NO_GEOMETRY"
    source = item.get("spread_source", "")
    if not any(source.startswith(prefix) and source[len(prefix):].strip() for prefix in _VERIFIED_SPREAD_PREFIXES):
        return empty_xy, empty_world, empty_xy, info, "UNVERIFIED_SPREAD"
    center, extent = np.asarray(item["center_world"]), np.asarray(item["spread_world"])
    corners = center + np.asarray(list(product((-1, 1), repeat=3))) * extent
    projected, positive = project_points(np.vstack([center, corners]), np.asarray(camera.intrinsics), np.asarray(camera.extrinsics))
    h, w = depth.shape
    if not positive.all():
        return empty_xy, empty_world, empty_xy, info, "BEHIND_CAMERA"
    if not (0 <= projected[0, 0] <= w - 1 and 0 <= projected[0, 1] <= h - 1):
        return empty_xy, empty_world, empty_xy, info, "OUTSIDE_IMAGE"
    low = np.maximum(np.ceil(projected[1:].min(axis=0)).astype(int), 0)
    high = np.minimum(np.floor(projected[1:].max(axis=0)).astype(int), [w - 1, h - 1])
    if np.any(high < low):
        return empty_xy, empty_world, empty_xy, info, "OUTSIDE_IMAGE"
    ys, xs = np.mgrid[low[1]:high[1] + 1, low[0]:high[0] + 1]
    scene = depth[ys, xs].ravel()
    xs, ys = xs.ravel(), ys.ravel()
    measured = np.isfinite(scene) & (scene > 0)
    k, t = np.asarray(camera.intrinsics), np.asarray(camera.extrinsics)
    z = scene[measured]
    cam_points = np.stack([(xs[measured] - k[0, 2]) * z / k[0, 0],
                           (ys[measured] - k[1, 2]) * z / k[1, 1], z], axis=1)
    world = cam_points @ t[:3, :3].T + t[:3, 3]
    contained = np.all(np.abs(world - center) <= extent + 1e-9, axis=1)
    info.update(spread_source=source, projected_pixels=len(scene), valid_depth_points=int(measured.sum()),
                depth_supported_points=int(contained.sum()), visible_fraction=float(contained.sum() / max(len(scene), 1)))
    if not measured.any():
        return empty_xy, empty_world, empty_xy, info, "INVALID_DEPTH"
    if contained.sum() < config.min_support or info["visible_fraction"] < config.min_visible_fraction:
        return empty_xy, empty_world, empty_xy, info, "DEPTH_UNSUPPORTED"
    xy = np.stack([xs[measured][contained], ys[measured][contained]], axis=1).astype(float)
    return xy, world[contained], projected[1:], info, None


def build_prompts(guidance: SlotGuidance, camera: CameraFrame, image_rgb: np.ndarray,
                  depth_m: np.ndarray, config: AdapterConfig | None = None) -> dict[str, RolePrompt]:
    """Return both semantic roles, including explicit per-role failure states.

    Camera metadata binds the RGB/depth arrays to the exact observation. No file,
    future observation or expert action is read. Callers must supply metres.
    Invalid guidance/configuration raises ValueError; unavailable per-frame
    perception returns a failed RolePrompt with no box and no positive points.
    """
    guidance.validate()
    config = config or AdapterConfig()
    config.validate()
    image, depth = np.asarray(image_rgb), np.asarray(depth_m)
    frame_error = _validate_frame(guidance, camera, image, depth)
    result: dict[str, RolePrompt] = {}
    for role in ROLES:
        item = guidance.metadata[role]
        prompt = RolePrompt(role=role, camera=camera.name, confidence=float(item["confidence"]),
                            support={"role_status": item["status"], "source_frame_index": camera.source_frame_index,
                                     "calibration_source": camera.calibration_source})
        result[role] = prompt
        if item["status"] != "VISIBLE":
            prompt.failure_reason = item["status"]
            continue
        if not item["semantic_valid"] or item["confidence"] < config.min_confidence:
            prompt.failure_reason = "LOW_CONFIDENCE"
            continue
        if frame_error:
            prompt.failure_reason = frame_error
            continue
        cloud = guidance.points(role)
        if len(cloud):
            trimmed = _trim(cloud, config)
            xy, _world, info, error = _cloud_support(trimmed, camera, depth, config)
            box_xy = xy
            info.update(input_points=len(cloud), geometry_source="points_world")
        else:
            xy, _world, box_xy, info, error = _spread_support(item, camera, depth, config)
        prompt.support.update(info)
        if error:
            prompt.failure_reason = error
            continue
        bbox = _box(box_xy, camera.image_size, config)
        if bbox is None:
            prompt.failure_reason = "DEGENERATE_BOX"
            continue
        prompt.bbox_xyxy = bbox
        prompt.positive_points_xy = _interior_points(xy, config.max_positive_points)
        prompt.prompt_valid = True
    for role, prompt in result.items():
        if not prompt.prompt_valid:
            continue
        other = result["reference" if role == "target" else "target"]
        if not other.prompt_valid:
            continue
        other_role = "reference" if role == "target" else "target"
        instance = guidance.metadata[role].get("instance_id")
        if instance is not None and instance == guidance.metadata[other_role].get("instance_id"):
            prompt.support["negative_source"] = "excluded_same_instance_id"
            continue
        # Only the other role's depth-supported positives can be negatives. A
        # box overlap is ambiguous, so points within this role's box are excluded.
        x1, y1, x2, y2 = prompt.bbox_xyxy
        for point in other.positive_points_xy:
            if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2):
                prompt.negative_points_xy.append(list(point))
        prompt.support["negative_source"] = "other_role_verified_depth_support"
    return result
