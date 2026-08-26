"""Lazy adapter for calibrated OSC trajectories stored in HDF5.

OSC camera poses come from Isaac/USD and require a separate convention from
robot/object state:

* depth is uint16 millimetres; 65535 is invalid;
* camera pose is ``[x, y, z, qw, qx, qy, qz]`` (Isaac scalar-first);
* the stored camera axes are USD (+Y up, -Z forward);
* RGB-D geometry uses OpenCV optical axes (+X right, +Y down, +Z forward).

Robot/object quaternions are not converted here and retain their dataset
``xyzw`` ordering.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

OSC_CAMERA_NAMES = (
    "GLOBAL_LEFT",
    "GLOBAL_RIGHT",
    "LOCAL_LEFT",
    "LOCAL_RIGHT",
    "WRIST",
)
OSC_DEPTH_METERS_PER_UNIT = 0.001
OSC_INVALID_DEPTH_VALUE = np.uint16(65535)
OSC_CAMERA_QUATERNION_ORDER = "wxyz"
OSC_CAMERA_AXES = "usd"

# Express OpenCV optical coordinates in a USD camera frame. Both conventions
# use +X right; OpenCV +Y/+Z are the opposites of USD +Y/+Z.
OSC_T_USD_CAMERA_FROM_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "OSC HDF5 episodes require h5py. Install project dependencies again "
            "or run `pip install h5py`."
        ) from exc
    return h5py


def find_osc_hdf5_path(episode_dir: str | Path) -> Path | None:
    episode = Path(episode_dir)
    candidates = sorted(
        {
            *episode.glob("*.hdf5"),
            *episode.glob("*.h5"),
            *(episode / "hdf5").glob("*.hdf5"),
            *(episode / "hdf5").glob("*.h5"),
        }
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path in candidates)
        raise ValueError(f"OSC episode contains multiple HDF5 files: {paths}")
    return candidates[0]


def is_osc_episode(episode_dir: str | Path) -> bool:
    return find_osc_hdf5_path(episode_dir) is not None


def available_osc_cameras(episode_dir: str | Path) -> tuple[str, ...]:
    path = find_osc_hdf5_path(episode_dir)
    if path is None:
        return ()
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        observations = handle.get("observations")
        if observations is None:
            return ()
        found = {
            name[: -len("_rgb")]
            for name in observations
            if name.endswith("_rgb")
        }
    ordered = [name for name in OSC_CAMERA_NAMES if name in found]
    ordered.extend(sorted(found.difference(ordered)))
    return tuple(ordered)


def osc_frame_count(episode_dir: str | Path, cameras: tuple[str, ...]) -> int:
    path = find_osc_hdf5_path(episode_dir)
    if path is None:
        raise FileNotFoundError(f"No OSC HDF5 trajectory found under {episode_dir}")
    if not cameras:
        raise ValueError("At least one OSC camera is required.")
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        counts = {}
        for camera in cameras:
            key = f"observations/{camera}_rgb"
            if key not in handle:
                raise KeyError(f"Missing OSC RGB dataset {key!r} in {path}")
            dataset = handle[key]
            if dataset.ndim != 4 or dataset.shape[-1] not in (3, 4):
                raise ValueError(
                    f"Expected NxHxWx3/4 OSC RGB data, got shape={dataset.shape} ({key})"
                )
            counts[camera] = int(dataset.shape[0])
    if len(set(counts.values())) != 1:
        raise ValueError(f"OSC cameras have inconsistent frame counts: {counts}")
    return next(iter(counts.values()))


def _cache_dir(episode_dir: str | Path) -> Path:
    path = find_osc_hdf5_path(episode_dir)
    if path is None:
        raise FileNotFoundError(f"No OSC HDF5 trajectory found under {episode_dir}")
    stat = path.stat()
    identity = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "relevant_object_osc_cache" / digest


def _frame_index(frame_id: str | int) -> int:
    try:
        index = int(str(frame_id))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OSC frame_id must be an integer, got {frame_id!r}") from exc
    if index < 0:
        raise IndexError(f"OSC frame index must be non-negative, got {index}")
    return index


def materialize_osc_rgb_frame(
    episode_dir: str | Path,
    camera: str,
    frame_id: str | int,
) -> Path:
    index = _frame_index(frame_id)
    output = _cache_dir(episode_dir) / f"{camera}_rgb" / f"{index}.png"
    if output.is_file():
        return output
    path = find_osc_hdf5_path(episode_dir)
    assert path is not None
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        key = f"observations/{camera}_rgb"
        if key not in handle:
            raise KeyError(f"Missing OSC RGB dataset {key!r} in {path}")
        dataset = handle[key]
        if index >= dataset.shape[0]:
            raise IndexError(
                f"OSC frame {index} is outside {key} range 0..{dataset.shape[0] - 1}"
            )
        rgb = np.asarray(dataset[index])
    if rgb.dtype != np.uint8:
        if (
            np.issubdtype(rgb.dtype, np.floating)
            and rgb.size
            and float(np.nanmax(rgb)) <= 1.0
        ):
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    Image.fromarray(rgb[..., :3], mode="RGB").save(temporary, format="PNG")
    temporary.replace(output)
    return output


def materialize_osc_depth_frame(
    episode_dir: str | Path,
    camera: str,
    frame_id: str | int,
) -> Path:
    """Materialize calibrated depth in metres; invalid samples become NaN."""
    index = _frame_index(frame_id)
    output = _cache_dir(episode_dir) / f"{camera}_depth" / f"{index}.meters.npy"
    if output.is_file():
        return output
    path = find_osc_hdf5_path(episode_dir)
    assert path is not None
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        key = f"observations/{camera}_depth"
        if key not in handle:
            raise KeyError(f"Missing OSC depth dataset {key!r} in {path}")
        dataset = handle[key]
        if index >= dataset.shape[0]:
            raise IndexError(
                f"OSC frame {index} is outside {key} range 0..{dataset.shape[0] - 1}"
            )
        raw = np.asarray(dataset[index])
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"Expected HxW OSC depth frame, got shape={raw.shape} ({key})")
    invalid = raw == OSC_INVALID_DEPTH_VALUE
    depth_m = raw.astype(np.float32) * OSC_DEPTH_METERS_PER_UNIT
    depth_m[invalid] = np.nan
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, depth_m, allow_pickle=False)
    temporary.replace(output)
    return output


class OSCFrameMap(Mapping[str, Path]):
    """Mapping that extracts an RGB frame only when its path is requested."""

    def __init__(self, episode_dir: str | Path, camera: str, frame_count: int):
        self.episode_dir = Path(episode_dir)
        self.camera = camera
        self.frame_count = int(frame_count)

    def __len__(self) -> int:
        return self.frame_count

    def __iter__(self) -> Iterator[str]:
        return (str(index) for index in range(self.frame_count))

    def __getitem__(self, frame_id: str) -> Path:
        index = _frame_index(frame_id)
        if index >= self.frame_count:
            raise KeyError(frame_id)
        return materialize_osc_rgb_frame(self.episode_dir, self.camera, index)


def collect_osc_camera_frames(
    episode_dir: str | Path,
    camera_names: tuple[str, ...],
) -> tuple[list[str], dict[str, Mapping[str, Path]]]:
    count = osc_frame_count(episode_dir, camera_names)
    ids = [str(index) for index in range(count)]
    return ids, {
        camera: OSCFrameMap(episode_dir, camera, count) for camera in camera_names
    }


def _quaternion_wxyz_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Invalid OSC camera quaternion: {quaternion!r}")
    x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def osc_camera_pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert an Isaac/USD camera pose into OpenCV ``T_world_camera``."""
    values = np.asarray(pose, dtype=np.float64)
    if values.shape == (4, 4):
        world_from_usd_camera = values
    elif values.shape == (3, 4):
        world_from_usd_camera = np.eye(4, dtype=np.float64)
        world_from_usd_camera[:3] = values
    elif values.shape == (7,):
        world_from_usd_camera = np.eye(4, dtype=np.float64)
        world_from_usd_camera[:3, :3] = _quaternion_wxyz_to_rotation(values[3:])
        world_from_usd_camera[:3, 3] = values[:3]
    else:
        raise ValueError(
            "OSC camera extrinsic must be [x,y,z,qw,qx,qy,qz], 3x4, or 4x4; "
            f"got shape={values.shape}"
        )
    return world_from_usd_camera @ OSC_T_USD_CAMERA_FROM_OPENCV_CAMERA


def osc_camera_params_for_frame(
    episode_dir: str | Path,
    camera: str,
    frame_index: int,
) -> dict[str, np.ndarray] | None:
    path = find_osc_hdf5_path(episode_dir)
    if path is None:
        return None
    index = _frame_index(frame_index)
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        intr_key = f"observations/{camera}_intrinsic"
        extr_key = f"observations/{camera}_extrinsic"
        if intr_key not in handle or extr_key not in handle:
            return None
        intr_dataset = handle[intr_key]
        extr_dataset = handle[extr_key]
        if intr_dataset.shape == (3, 3):
            intrinsics = np.asarray(intr_dataset[...], dtype=np.float64)
        elif intr_dataset.ndim == 3 and intr_dataset.shape[1:] == (3, 3):
            if index >= intr_dataset.shape[0]:
                raise IndexError(
                    f"OSC frame {index} is outside {intr_key} range 0..{intr_dataset.shape[0] - 1}"
                )
            intrinsics = np.asarray(intr_dataset[index], dtype=np.float64)
        else:
            raise ValueError(f"Invalid OSC intrinsics shape: {intr_dataset.shape} ({intr_key})")

        if extr_dataset.shape in {(7,), (3, 4), (4, 4)}:
            raw_extrinsics = np.asarray(extr_dataset[...], dtype=np.float64)
        elif extr_dataset.ndim >= 2:
            if index >= extr_dataset.shape[0]:
                raise IndexError(
                    f"OSC frame {index} is outside {extr_key} range 0..{extr_dataset.shape[0] - 1}"
                )
            raw_extrinsics = np.asarray(extr_dataset[index], dtype=np.float64)
        else:
            raise ValueError(f"Invalid OSC extrinsics shape: {extr_dataset.shape} ({extr_key})")
    return {
        "intrinsics": intrinsics,
        "extrinsics": osc_camera_pose_to_matrix(raw_extrinsics),
    }


def osc_instruction(episode_dir: str | Path) -> str | None:
    metadata_path = Path(episode_dir) / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = None
        if isinstance(metadata, Mapping):
            for key in ("task_instruction", "instruction"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    path = find_osc_hdf5_path(episode_dir)
    if path is None:
        return None
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        if "task_instruction" not in handle:
            return None
        raw = handle["task_instruction"][()]
    if isinstance(raw, np.ndarray) and raw.size == 1:
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    text = str(raw).strip()
    return text or None


def load_osc_robot_observations(episode_dir: str | Path) -> list[dict[str, Any]]:
    """Expose OSC robot state through the RLBench-like fields Stage 4 consumes."""
    path = find_osc_hdf5_path(episode_dir)
    if path is None:
        return []
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        observations = handle.get("observations")
        if observations is None:
            return []
        position_key = "ee_pos_world" if "ee_pos_world" in observations else "ee_pos"
        quaternion_key = "ee_quat_world" if "ee_quat_world" in observations else "ee_quat"
        if position_key not in observations:
            return []
        positions = np.asarray(observations[position_key], dtype=np.float64)
        quaternions = (
            np.asarray(observations[quaternion_key], dtype=np.float64)
            if quaternion_key in observations
            else None
        )
        actions = (
            np.asarray(handle["actions"], dtype=np.float64)
            if "actions" in handle
            else None
        )
    result: list[dict[str, Any]] = []
    for index, position in enumerate(positions):
        pose = np.asarray(position).reshape(-1).tolist()
        if quaternions is not None and index < len(quaternions):
            pose.extend(np.asarray(quaternions[index]).reshape(-1).tolist())
        observation: dict[str, Any] = {"gripper_pose": pose}
        if actions is not None and index < len(actions) and actions.shape[-1] >= 7:
            observation["gripper_open"] = float(
                np.clip((float(actions[index, 6]) + 1.0) / 2.0, 0.0, 1.0)
            )
        result.append(observation)
    return result


__all__ = [
    "OSC_CAMERA_AXES",
    "OSC_CAMERA_NAMES",
    "OSC_CAMERA_QUATERNION_ORDER",
    "OSC_DEPTH_METERS_PER_UNIT",
    "OSC_INVALID_DEPTH_VALUE",
    "OSC_T_USD_CAMERA_FROM_OPENCV_CAMERA",
    "OSCFrameMap",
    "available_osc_cameras",
    "collect_osc_camera_frames",
    "find_osc_hdf5_path",
    "is_osc_episode",
    "load_osc_robot_observations",
    "materialize_osc_depth_frame",
    "materialize_osc_rgb_frame",
    "osc_camera_params_for_frame",
    "osc_camera_pose_to_matrix",
    "osc_frame_count",
    "osc_instruction",
]
