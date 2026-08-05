"""Camera calibration, projection, and point-cloud geometry."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from PIL import Image


def normalize_intrinsics(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 4:
        fx, fy, cx, cy = arr.reshape(-1)
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError(f"Invalid intrinsics shape: {arr.shape}")


def normalize_extrinsics(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = arr
        return out
    raise ValueError(f"Invalid extrinsics shape: {arr.shape}")


def backproject_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    max_points: int,
) -> np.ndarray:
    if mask.shape != depth.shape:
        mask = (
            np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (depth.shape[1], depth.shape[0]),
                    Image.Resampling.NEAREST,
                )
            )
            > 127
        )
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if max_points > 0 and len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]
    z = depth[ys, xs]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    return np.stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z), axis=1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    hom = np.concatenate(
        [points, np.ones((len(points), 1), dtype=points.dtype)],
        axis=1,
    )
    return (hom @ transform.T)[:, :3]


def project_points(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into pixels; ``extrinsics`` is T_world_camera."""
    if len(points_world) == 0:
        return np.empty((0, 2)), np.empty((0,), dtype=bool)
    world_to_cam = np.linalg.inv(extrinsics)
    hom = np.concatenate(
        [points_world, np.ones((len(points_world), 1))],
        axis=1,
    )
    points_cam = (hom @ world_to_cam.T)[:, :3]
    z = points_cam[:, 2]
    valid = z > 1e-6
    safe_z = np.where(valid, z, 1.0)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = points_cam[:, 0] * fx / safe_z + cx
    v = points_cam[:, 1] * fy / safe_z + cy
    return np.stack([u, v], axis=1), valid


def cloud_observability_in_camera(
    points_world: np.ndarray,
    camera_context: Mapping[str, np.ndarray],
    depth_tolerance_m: float,
    min_visible_fraction: float,
    max_samples: int = 256,
) -> dict[str, Any]:
    """Estimate whether an existing 3D cloud should be visible in another view."""
    points = np.asarray(points_world, dtype=np.float64)
    if len(points) == 0:
        return {"observable": False, "reason": "empty_cloud"}
    if max_samples > 0 and len(points) > max_samples:
        indices = np.linspace(0, len(points) - 1, max_samples).astype(int)
        points = points[indices]

    intrinsics = np.asarray(camera_context["intrinsics"], dtype=np.float64)
    extrinsics = np.asarray(camera_context["extrinsics"], dtype=np.float64)
    depth = np.asarray(camera_context["depth"], dtype=np.float64)
    world_to_camera = np.linalg.inv(extrinsics)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    points_camera = (homogeneous @ world_to_camera.T)[:, :3]
    z = points_camera[:, 2]
    positive = z > 1e-6
    safe_z = np.where(positive, z, 1.0)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = points_camera[:, 0] * fx / safe_z + cx
    v = points_camera[:, 1] * fy / safe_z + cy
    xs = np.rint(u).astype(int)
    ys = np.rint(v).astype(int)
    in_bounds = (
        positive
        & (xs >= 0)
        & (xs < depth.shape[1])
        & (ys >= 0)
        & (ys < depth.shape[0])
    )
    projected_count = int(in_bounds.sum())
    if projected_count == 0:
        return {
            "observable": False,
            "reason": "outside_image",
            "sampled_points": int(len(points)),
            "projected_points": 0,
        }

    projected_indices = np.flatnonzero(in_bounds)
    observed_depth = depth[ys[in_bounds], xs[in_bounds]]
    valid_depth = np.isfinite(observed_depth) & (observed_depth > 0)
    valid_depth_count = int(valid_depth.sum())
    if valid_depth_count == 0:
        return {
            "observable": False,
            "reason": "missing_depth",
            "sampled_points": int(len(points)),
            "projected_points": projected_count,
            "valid_depth_points": 0,
        }

    point_depth = z[projected_indices][valid_depth]
    scene_depth = observed_depth[valid_depth]
    visible = point_depth <= scene_depth + max(0.0, float(depth_tolerance_m))
    visible_count = int(visible.sum())
    visible_fraction = float(visible_count / valid_depth_count)
    required_fraction = min(1.0, max(0.0, float(min_visible_fraction)))
    return {
        "observable": visible_fraction >= required_fraction,
        "reason": (
            "depth_visible"
            if visible_fraction >= required_fraction
            else "occluded_by_depth"
        ),
        "sampled_points": int(len(points)),
        "projected_points": projected_count,
        "valid_depth_points": valid_depth_count,
        "visible_points": visible_count,
        "visible_fraction": visible_fraction,
    }

