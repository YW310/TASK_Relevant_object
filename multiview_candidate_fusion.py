#!/usr/bin/env python3
"""Fuse per-view SAM3 role candidates into frame-level 3D objects.

The script consumes ``episode_candidates.json`` produced by
``qwen_role_sam3_candidate_episode.py`` plus per-camera ``candidates.json`` and
mask PNG files. For each candidate, depth pixels inside the mask are
back-projected with camera intrinsics, transformed by camera extrinsics, and
clustered with same-role candidates from other views.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

ROLE_OBJECT_PREFIX = {
    "target": "target_obj",
    "reference": "reference_obj",
    "interaction_part": "part_obj",
}


@dataclass
class Observation3D:
    role: str
    camera: str
    candidate: Mapping[str, Any]
    points_world: np.ndarray
    centroid_world: np.ndarray
    bbox3d_world: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, help="RLBench/RLBench-exported episode directory.")
    parser.add_argument("--candidates-json", required=True, help="Path to episode_candidates.json.")
    parser.add_argument("--output-json", default=None, help="Default: frame_fused_candidates.json next to episode_candidates.json.")
    parser.add_argument("--cameras", default=None, help="Optional comma-separated camera subset.")
    parser.add_argument("--camera-params-json", default=None, help="Optional camera parameter JSON overriding auto-discovery.")
    parser.add_argument("--rlbench-low-dim-obs", default=None, help="Optional path to RLBench low_dim_obs.pkl. Default: <episode-dir>/low_dim_obs.pkl.")
    parser.add_argument("--invert-rlbench-extrinsics", action="store_true", help="Invert RLBench camera extrinsics before transforming camera points to world coordinates.")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Divide raw depth values by this scale.")
    parser.add_argument("--max-points-per-candidate", type=int, default=4096)
    parser.add_argument("--cluster-distance-m", type=float, default=0.03, help="Centroid threshold, e.g. 0.02-0.05 m.")
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.0, help="Optional 3D bbox IoU threshold for merging.")
    parser.add_argument("--nearest-distance-m", type=float, default=None, help="Optional point-cloud nearest-distance threshold for merging.")
    return parser


def atomic_json_dump(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def read_depth(path: Path, depth_scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = np.asarray(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float64) / float(depth_scale)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def normalize_intrinsics(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 4:
        fx, fy, cx, cy = arr.reshape(-1)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
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
            "extrinsics": normalize_extrinsics(item.get("extrinsics", item.get("T_world_camera", item.get("camera_to_world")))),
        }
    return params


class _PickleFallbackObject:
    """Minimal object used when unpickling optional RLBench classes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.__dict__.update(kwargs)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, Mapping):
            self.__dict__.update(state)
        else:
            self.state = state


class _PickleFallbackDemo(list):
    """List-compatible stand-in for rlbench.demo.Demo."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args)
        self.__dict__.update(kwargs)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, Mapping):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2:
            list_state, dict_state = state
            if list_state is not None:
                self.extend(list_state)
            if isinstance(dict_state, Mapping):
                self.__dict__.update(dict_state)
        else:
            self.state = state


class _RLBenchOptionalUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("rlbench"):
            if name == "Demo":
                return _PickleFallbackDemo
            return _PickleFallbackObject
        return super().find_class(module, name)


def load_rlbench_observations(path: Path | None) -> list[Any]:
    if path is None or not path.is_file():
        return []
    with path.open("rb") as handle:
        try:
            loaded = pickle.load(handle)
        except ModuleNotFoundError as exc:
            if exc.name != "rlbench":
                raise
            handle.seek(0)
            loaded = _RLBenchOptionalUnpickler(handle).load()
    if hasattr(loaded, "_observations"):
        observations = getattr(loaded, "_observations")
        if isinstance(observations, Sequence):
            return list(observations)
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, tuple):
        return list(loaded)
    try:
        return list(loaded)
    except TypeError as exc:
        raise ValueError(
            f"Expected RLBench low_dim_obs.pkl to contain a sequence, got {type(loaded).__name__}"
        ) from exc


def observation_misc(observation: Any) -> Mapping[str, Any]:
    misc = getattr(observation, "misc", None)
    if misc is None and isinstance(observation, Mapping):
        misc = observation.get("misc")
    if not isinstance(misc, Mapping):
        raise ValueError("RLBench observation does not expose a misc mapping with camera parameters")
    return misc


def frame_index_from_frame(frame: Mapping[str, Any]) -> int | None:
    raw = frame.get("frame_index")
    if raw is None:
        raw = frame.get("frame_id")
    try:
        return int(str(raw))
    except (TypeError, ValueError):
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


def resolve_rlbench_camera_param(
    observations: Sequence[Any],
    camera: str,
    frame_index: int | None,
    *,
    invert_extrinsics: bool = False,
) -> dict[str, np.ndarray] | None:
    if not observations:
        return None
    index = frame_index if frame_index is not None else 0
    if index < 0 or index >= len(observations):
        raise IndexError(
            f"RLBench frame_index={index} is outside low_dim_obs range 0..{len(observations) - 1}"
        )
    return camera_param_from_rlbench_observation(
        observations[index], camera, invert_extrinsics=invert_extrinsics
    )

def find_first(paths: Sequence[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def resolve_depth_path(episode_dir: Path, camera: str, frame_id: str) -> Path:
    candidates = []
    for suffix in (".npy", ".png", ".tiff", ".tif", ".exr"):
        candidates.extend([
            episode_dir / f"{camera}_depth" / f"{frame_id}{suffix}",
            episode_dir / f"{camera}_depths" / f"{frame_id}{suffix}",
            episode_dir / "depth" / camera / f"{frame_id}{suffix}",
            episode_dir / "depths" / camera / f"{frame_id}{suffix}",
        ])
    found = find_first(candidates)
    if found is None:
        raise FileNotFoundError(f"Missing depth image for camera={camera} frame_id={frame_id}")
    return found


def resolve_camera_param(
    params: Mapping[str, dict[str, np.ndarray]],
    episode_dir: Path,
    camera: str,
    rlbench_observations: Sequence[Any],
    frame_index: int | None,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    if camera in params:
        return params[camera]
    rlbench_param = resolve_rlbench_camera_param(
        rlbench_observations,
        camera,
        frame_index,
        invert_extrinsics=args.invert_rlbench_extrinsics,
    )
    if rlbench_param is not None:
        return rlbench_param
    for path in (episode_dir / f"{camera}_camera.json", episode_dir / "camera_params.json", episode_dir / "cameras.json"):
        if path.is_file():
            loaded = load_camera_params(path)
            if camera in loaded:
                return loaded[camera]
    raise FileNotFoundError(f"Missing camera intrinsics/extrinsics for camera={camera}")


def backproject_mask(depth: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray, max_points: int) -> np.ndarray:
    if mask.shape != depth.shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST)) > 127
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if max_points > 0 and len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]
    z = depth[ys, xs]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    return np.stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z), axis=1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    return (hom @ transform.T)[:, :3]


def bbox_iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    mins = np.maximum(a[0], b[0]); maxs = np.minimum(a[1], b[1])
    inter_dims = np.maximum(0.0, maxs - mins)
    inter = float(np.prod(inter_dims))
    vol_a = float(np.prod(np.maximum(0.0, a[1] - a[0]))); vol_b = float(np.prod(np.maximum(0.0, b[1] - b[0])))
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def nearest_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    step_a = max(1, len(a) // 512); step_b = max(1, len(b) // 512)
    aa, bb = a[::step_a], b[::step_b]
    d2 = ((aa[:, None, :] - bb[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1)).mean())


def should_merge(obs: Observation3D, cluster: Sequence[Observation3D], args: argparse.Namespace) -> bool:
    for other in cluster:
        if obs.role != other.role:
            continue
        centroid_ok = np.linalg.norm(obs.centroid_world - other.centroid_world) <= args.cluster_distance_m
        iou_ok = args.bbox_iou_threshold <= 0 or bbox_iou_3d(obs.bbox3d_world, other.bbox3d_world) >= args.bbox_iou_threshold
        nearest_ok = args.nearest_distance_m is None or nearest_mean_distance(obs.points_world, other.points_world) <= args.nearest_distance_m
        if centroid_ok and iou_ok and nearest_ok:
            return True
    return False


def observation_to_json(obs: Observation3D) -> dict[str, Any]:
    c = obs.candidate
    return {
        "camera": obs.camera,
        "candidate_id": c.get("id"),
        "role": obs.role,
        "mask_path": c.get("mask_path"),
        "mask_area": int(c.get("mask_area_pixels", 0)),
        "sam_score": c.get("score"),
        "mask_bbox_xyxy": c.get("mask_bbox_xyxy"),
        "points_world": obs.points_world.tolist(),
        "centroid_world": obs.centroid_world.tolist(),
        "bbox3d_world": obs.bbox3d_world.tolist(),
    }


def fuse_frame(
    frame: Mapping[str, Any],
    episode_dir: Path,
    camera_params: Mapping[str, dict[str, np.ndarray]],
    rlbench_observations: Sequence[Any],
    cameras: Sequence[str] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    observations: list[Observation3D] = []
    frame_id = str(frame["frame_id"])
    frame_index = frame_index_from_frame(frame)
    for camera, view in frame.get("views", {}).items():
        if cameras is not None and camera not in cameras:
            continue
        params = resolve_camera_param(camera_params, episode_dir, camera, rlbench_observations, frame_index, args)
        depth = read_depth(resolve_depth_path(episode_dir, camera, frame_id), args.depth_scale)
        data = json.loads(Path(view["candidates_json"]).read_text(encoding="utf-8"))
        for cand in data.get("candidates", []):
            mask = load_mask(Path(cand["mask_path"]))
            points_cam = backproject_mask(depth, mask, params["intrinsics"], args.max_points_per_candidate)
            points_world = transform_points(points_cam, params["extrinsics"])
            if len(points_world) == 0:
                continue
            centroid = points_world.mean(axis=0)
            bbox = np.stack([points_world.min(axis=0), points_world.max(axis=0)])
            observations.append(Observation3D(str(cand["role"]), camera, cand, points_world, centroid, bbox))

    clusters: list[list[Observation3D]] = []
    for obs in observations:
        for cluster in clusters:
            if should_merge(obs, cluster, args):
                cluster.append(obs)
                break
        else:
            clusters.append([obs])

    role_counts = {role: 0 for role in ROLE_OBJECT_PREFIX}
    objects = []
    for cluster in sorted(clusters, key=lambda c: (c[0].role, float(c[0].centroid_world[0]))):
        role = cluster[0].role
        prefix = ROLE_OBJECT_PREFIX.get(role, f"{role}_obj")
        index = role_counts.get(role, 0); role_counts[role] = index + 1
        all_points = np.concatenate([obs.points_world for obs in cluster], axis=0)
        objects.append({
            "id": f"{prefix}_{index:03d}",
            "role": role,
            "points_world": all_points.tolist(),
            "centroid_world": all_points.mean(axis=0).tolist(),
            "bbox3d_world": np.stack([all_points.min(axis=0), all_points.max(axis=0)]).tolist(),
            "visible_camera": sorted({obs.camera for obs in cluster}),
            "mask_area": int(sum(int(obs.candidate.get("mask_area_pixels", 0)) for obs in cluster)),
            "sam_score": float(np.mean([float(obs.candidate.get("score", 0.0)) for obs in cluster])),
            "observations": [observation_to_json(obs) for obs in cluster],
        })
    return {"frame_index": frame.get("frame_index"), "frame_id": frame_id, "objects": objects}


def main() -> None:
    args = build_parser().parse_args()
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    candidates_path = Path(args.candidates_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve() if args.output_json else candidates_path.with_name("frame_fused_candidates.json")
    summary = json.loads(candidates_path.read_text(encoding="utf-8"))
    camera_params = load_camera_params(Path(args.camera_params_json).expanduser().resolve() if args.camera_params_json else None)
    rlbench_low_dim_path = (
        Path(args.rlbench_low_dim_obs).expanduser().resolve()
        if args.rlbench_low_dim_obs
        else episode_dir / "low_dim_obs.pkl"
    )
    rlbench_observations = load_rlbench_observations(rlbench_low_dim_path)
    cameras = parse_csv(args.cameras)
    frames = [
        fuse_frame(frame, episode_dir, camera_params, rlbench_observations, cameras, args)
        for frame in summary.get("frames", [])
    ]
    result = {
        "episode_dir": str(episode_dir),
        "source_candidates_json": str(candidates_path),
        "cluster_distance_m": args.cluster_distance_m,
        "bbox_iou_threshold": args.bbox_iou_threshold,
        "nearest_distance_m": args.nearest_distance_m,
        "rlbench_low_dim_obs": str(rlbench_low_dim_path) if rlbench_observations else None,
        "invert_rlbench_extrinsics": bool(args.invert_rlbench_extrinsics),
        "frames": frames,
    }
    atomic_json_dump(result, output_path)
    print(json.dumps({"output_json": str(output_path), "frames": len(frames)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
