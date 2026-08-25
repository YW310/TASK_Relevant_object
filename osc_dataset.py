"""Lazy RGB and language adapter for OSC episodes stored in HDF5."""

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
            for name in observations.keys()
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
        if np.issubdtype(rgb.dtype, np.floating) and rgb.size:
            if float(np.nanmax(rgb)) <= 1.0:
                rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    Image.fromarray(rgb[..., :3], mode="RGB").save(temporary, format="PNG")
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


def osc_instruction(episode_dir: str | Path) -> str | None:
    metadata_path = Path(episode_dir) / 'metadata.json'
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            metadata = None
        if isinstance(metadata, Mapping):
            for key in ('task_instruction', 'instruction'):
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


__all__ = [
    "OSC_CAMERA_NAMES",
    "OSCFrameMap",
    "available_osc_cameras",
    "collect_osc_camera_frames",
    "find_osc_hdf5_path",
    "is_osc_episode",
    "materialize_osc_rgb_frame",
    "osc_frame_count",
    "osc_instruction",
]
