"""Shared RLBench camera, depth, and projection helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


# RLBench encodes normalized depth in a 24-bit RGB PNG (R<<16 | G<<8 | B).
RLBENCH_DEPTH_SCALE_FACTOR = float(2**24 - 1)


def decode_rlbench_rgb_depth(
    image_array: np.ndarray,
    near: float,
    far: float,
    scale_factor: float = RLBENCH_DEPTH_SCALE_FACTOR,
) -> np.ndarray:
    r, g, b = (image_array[..., i].astype(np.uint32) for i in range(3))
    normalized = ((r << 16) | (g << 8) | b).astype(np.float64) / scale_factor
    return near + normalized * (far - near)


def looks_like_rlbench_packed_depth(image_array: np.ndarray) -> bool:
    """Distinguish packed depth from grayscale replicated across RGB channels."""
    return not (
        np.array_equal(image_array[..., 0], image_array[..., 1])
        and np.array_equal(image_array[..., 1], image_array[..., 2])
    )


def read_depth(
    path: Path,
    depth_scale: float,
    near: float | None = None,
    far: float | None = None,
    mode: str = "auto",
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth.astype(np.float64) / float(depth_scale)

    image_array = np.asarray(Image.open(path))
    is_rgb = image_array.ndim == 3 and image_array.shape[-1] >= 3
    has_near_far = near is not None and far is not None

    use_rlbench = False
    if mode == "rlbench-rgb":
        if not (is_rgb and has_near_far):
            raise ValueError(
                "--depth-mode=rlbench-rgb requires a 3-channel PNG and near/far, "
                f"got shape={image_array.shape} near={near} far={far} ({path})."
            )
        use_rlbench = True
    elif mode == "auto":
        use_rlbench = (
            is_rgb
            and has_near_far
            and looks_like_rlbench_packed_depth(image_array[..., :3])
        )

    if use_rlbench:
        return decode_rlbench_rgb_depth(image_array[..., :3], near, far)

    depth = image_array
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float64) / float(depth_scale)


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


def load_camera_params(path: Path | None) -> dict[str, dict[str, np.ndarray]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    cameras = raw.get("cameras", raw)
    params: dict[str, dict[str, np.ndarray]] = {}
    for name, item in cameras.items():
        params[str(name)] = {
            "intrinsics": normalize_intrinsics(item.get("intrinsics", item.get("K"))),
            "extrinsics": normalize_extrinsics(
                item.get(
                    "extrinsics",
                    item.get("T_world_camera", item.get("camera_to_world")),
                )
            ),
        }
    return params


def resolve_rlbench_low_dim_path(
    episode_dir: Path,
    override_path: Path | None = None,
) -> Path:
    return override_path if override_path is not None else episode_dir / "low_dim_obs.pkl"


def load_rlbench_observations(
    episode_dir: Path,
    override_path: Path | None = None,
) -> list[Any]:
    """Load RLBench observations, including its list-like ``Demo`` wrapper."""
    path = resolve_rlbench_low_dim_path(episode_dir, override_path)
    if not path.is_file():
        return []
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, tuple):
        return list(loaded)
    observations = getattr(loaded, "_observations", None)
    if isinstance(observations, (list, tuple)):
        return list(observations)
    if hasattr(loaded, "__len__") and hasattr(loaded, "__getitem__"):
        return [loaded[i] for i in range(len(loaded))]
    raise ValueError(
        "Expected RLBench low_dim_obs.pkl to contain a sequence, "
        f"got {type(loaded).__name__}"
    )


def observation_misc(observation: Any) -> Mapping[str, Any]:
    misc = getattr(observation, "misc", None)
    if misc is None and isinstance(observation, Mapping):
        misc = observation.get("misc")
    if not isinstance(misc, Mapping):
        raise ValueError(
            "RLBench observation does not expose a misc mapping with camera parameters"
        )
    return misc


def frame_index_from_frame(frame: Mapping[str, Any]) -> int | None:
    """Resolve the source episode index, preferring a numeric frame ID."""
    for raw in (frame.get("frame_id"), frame.get("frame_index")):
        if raw is None:
            continue
        try:
            return int(str(raw))
        except (TypeError, ValueError):
            continue
    return None


def camera_param_from_rlbench_observation(
    observation: Any,
    camera: str,
    *,
    invert_extrinsics: bool = False,
) -> dict[str, np.ndarray]:
    misc = observation_misc(observation)
    intr_key = f"{camera}_camera_intrinsics"
    extr_key = f"{camera}_camera_extrinsics"
    if intr_key not in misc or extr_key not in misc:
        raise KeyError(f"Missing RLBench camera keys: {intr_key!r} / {extr_key!r}")
    intrinsics = normalize_intrinsics(misc[intr_key])
    extrinsics = normalize_extrinsics(misc[extr_key])
    if invert_extrinsics:
        extrinsics = np.linalg.inv(extrinsics)
    return {"intrinsics": intrinsics, "extrinsics": extrinsics}


def resolve_rlbench_near_far(
    camera: str,
    frame_index: int | None,
    rlbench_observations: Sequence[Any],
) -> tuple[float, float] | None:
    """Fetch the per-frame near/far planes needed for packed depth."""
    if not rlbench_observations:
        return None
    index = frame_index if frame_index is not None else 0
    if index < 0 or index >= len(rlbench_observations):
        return None
    try:
        misc = observation_misc(rlbench_observations[index])
    except ValueError:
        return None
    near_key, far_key = f"{camera}_camera_near", f"{camera}_camera_far"
    if near_key not in misc or far_key not in misc:
        return None
    return float(misc[near_key]), float(misc[far_key])


def find_first(paths: Sequence[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def resolve_depth_path(episode_dir: Path, camera: str, frame_id: str) -> Path:
    candidates = []
    for suffix in (".npy", ".png", ".tiff", ".tif", ".exr"):
        candidates.extend(
            [
                episode_dir / f"{camera}_depth" / f"{frame_id}{suffix}",
                episode_dir / f"{camera}_depths" / f"{frame_id}{suffix}",
                episode_dir / "depth" / camera / f"{frame_id}{suffix}",
                episode_dir / "depths" / camera / f"{frame_id}{suffix}",
            ]
        )
    found = find_first(candidates)
    if found is None:
        raise FileNotFoundError(
            f"Missing depth image for camera={camera} frame_id={frame_id}"
        )
    return found


def find_rgb_path(episode_dir: Path, camera: str, frame_id: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = episode_dir / f"{camera}_rgb" / f"{frame_id}{suffix}"
        if path.is_file():
            return path
    return None


def resolve_camera_param_for_frame(
    camera: str,
    frame_index: int | None,
    frame_id: str,
    explicit_camera_params: Mapping[str, dict[str, np.ndarray]],
    rlbench_observations: Sequence[Any],
    episode_dir: Path,
    *,
    invert_rlbench_extrinsics: bool = False,
) -> dict[str, np.ndarray] | None:
    """Resolve per-frame camera geometry from explicit, RLBench, or JSON data."""
    if camera in explicit_camera_params:
        return explicit_camera_params[camera]

    if rlbench_observations:
        index = frame_index if frame_index is not None else 0
        if index < 0 or index >= len(rlbench_observations):
            raise IndexError(
                f"RLBench frame_index={index} (frame_id={frame_id!r}) is outside "
                f"low_dim_obs range 0..{len(rlbench_observations) - 1}"
            )
        try:
            return camera_param_from_rlbench_observation(
                rlbench_observations[index],
                camera,
                invert_extrinsics=invert_rlbench_extrinsics,
            )
        except KeyError:
            pass

    for path in (
        episode_dir / f"{camera}_camera.json",
        episode_dir / "camera_params.json",
        episode_dir / "cameras.json",
    ):
        if path.is_file():
            loaded = load_camera_params(path)
            if camera in loaded:
                return loaded[camera]
    return None


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
