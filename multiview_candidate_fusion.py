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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

# Matches the per-camera candidate id prefixes (T/R/P) used by
# qwen_role_sam3_candidate_episode.py, so fused object ids (e.g. "T1", "R1")
# read consistently with the upstream per-view candidate ids.
ROLE_OBJECT_PREFIX = {
    "target": "T",
    "reference": "R",
    "interaction_part": "P",
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
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Divide raw depth values by this scale (only used for --depth-mode=raw / single-channel depth).")
    parser.add_argument(
        "--depth-mode",
        choices=("auto", "rlbench-rgb", "raw"),
        default="auto",
        help=(
            "How to decode depth PNGs. 'rlbench-rgb' forces RLBench's 24-bit R<<16|G<<8|B "
            "normalized-depth encoding (needs low_dim_obs.pkl near/far). 'raw' forces the legacy "
            "path: first channel (or .npy) divided by --depth-scale. 'auto' (default) uses "
            "rlbench-rgb only when the PNG's R/G/B channels actually differ (a grayscale depth "
            "PNG replicated across channels has R==G==B and is never RLBench-packed) and "
            "near/far are available; otherwise falls back to 'raw'."
        ),
    )
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


# RLBench encodes depth as a normalized value packed into a 24-bit RGB PNG
# (R<<16 | G<<8 | B), scaled into [0, 1], then linearly mapped into
# [near, far] meters. Naively reading only the R channel (or dividing the
# raw byte values by --depth-scale) silently produces near-random, heavily
# quantized depth and looks exactly like a camera-alignment bug even though
# the intrinsics/extrinsics math is fine. See RLBench's
# ``rlbench.backend.utils.image_to_float_array`` / ``const.DEPTH_SCALE``.
RLBENCH_DEPTH_SCALE_FACTOR = float(2 ** 24 - 1)


def decode_rlbench_rgb_depth(image_array: np.ndarray, near: float, far: float, scale_factor: float = RLBENCH_DEPTH_SCALE_FACTOR) -> np.ndarray:
    r, g, b = (image_array[..., i].astype(np.uint32) for i in range(3))
    normalized = ((r << 16) | (g << 8) | b).astype(np.float64) / scale_factor
    return near + normalized * (far - near)


def looks_like_rlbench_packed_depth(image_array: np.ndarray) -> bool:
    """Distinguish RLBench's 24-bit R<<16|G<<8|B packed depth from a plain
    grayscale depth PNG that happens to be saved with 3 replicated channels.

    A grayscale-as-RGB PNG has R == G == B for every pixel; treating it as a
    packed 24-bit value would produce nonsense depth. Genuine RLBench-packed
    depth almost never has all three channels identical everywhere.
    """
    return not (np.array_equal(image_array[..., 0], image_array[..., 1]) and np.array_equal(image_array[..., 1], image_array[..., 2]))


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
            raise ValueError(f"--depth-mode=rlbench-rgb requires a 3-channel PNG and near/far, got shape={image_array.shape} near={near} far={far} ({path}).")
        use_rlbench = True
    elif mode == "auto":
        use_rlbench = is_rgb and has_near_far and looks_like_rlbench_packed_depth(image_array[..., :3])

    if use_rlbench:
        return decode_rlbench_rgb_depth(image_array[..., :3], near, far)

    depth = image_array
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



def resolve_rlbench_low_dim_path(episode_dir: Path, override_path: Path | None = None) -> Path:
    return override_path if override_path is not None else episode_dir / "low_dim_obs.pkl"


def load_rlbench_observations(episode_dir: Path, override_path: Path | None = None) -> list[Any]:
    """Load RLBench ``low_dim_obs.pkl`` observations for an episode directory.

    Defaults to ``episode_dir / "low_dim_obs.pkl"`` unless ``override_path`` is
    given (e.g. via ``--rlbench-low-dim-obs``). Returns ``[]`` when the file is
    missing, so callers can gracefully fall back to other camera-parameter
    sources.

    RLBench episodes are pickled as a ``rlbench.demo.Demo`` object, which
    wraps a plain list in ``self._observations`` and supports ``len()``/
    indexing but is not a ``list``/``tuple`` itself. Handle that case via
    duck-typing so this works even when the ``rlbench`` package (and thus
    the ``Demo`` class) is not importable in the current environment.
    """
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
    raise ValueError(f"Expected RLBench low_dim_obs.pkl to contain a sequence, got {type(loaded).__name__}")


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


def resolve_rlbench_near_far(
    camera: str,
    frame_index: int | None,
    rlbench_observations: Sequence[Any],
) -> tuple[float, float] | None:
    """Fetch the per-frame depth near/far clip planes RLBench needs to decode its RGB-packed depth PNGs."""
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
    """Resolve intrinsics/extrinsics for one camera at one frame.

    Priority:
      1. ``explicit_camera_params`` (from ``--camera-params-json``).
      2. RLBench ``low_dim_obs.pkl`` observation at ``frame_index`` (works for
         both static cameras and the moving wrist camera, since every camera
         is read per-frame).
      3. ``{camera}_camera.json`` / ``camera_params.json`` / ``cameras.json``
         fallback files next to the episode.
      4. ``None`` when no geometry is available; callers should degrade to
         visual-only matching for this camera/frame instead of failing.
    """
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
                rlbench_observations[index], camera, invert_extrinsics=invert_rlbench_extrinsics
            )
        except KeyError:
            pass  # This camera has no entry in RLBench misc; try other sources.

    for path in (episode_dir / f"{camera}_camera.json", episode_dir / "camera_params.json", episode_dir / "cameras.json"):
        if path.is_file():
            loaded = load_camera_params(path)
            if camera in loaded:
                return loaded[camera]

    return None


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
        params = resolve_camera_param_for_frame(
            camera,
            frame_index,
            frame_id,
            camera_params,
            rlbench_observations,
            episode_dir,
            invert_rlbench_extrinsics=args.invert_rlbench_extrinsics,
        )
        if params is None:
            print(
                f"[warn] frame_id={frame_id} camera={camera}: no camera intrinsics/extrinsics found; "
                "skipping 3D fusion for this view (visual-only matching not yet implemented).",
                file=sys.stderr,
            )
            continue
        depth_near_far = resolve_rlbench_near_far(camera, frame_index, rlbench_observations)
        near, far = depth_near_far if depth_near_far is not None else (None, None)
        depth = read_depth(resolve_depth_path(episode_dir, camera, frame_id), args.depth_scale, near=near, far=far, mode=args.depth_mode)
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
        index = role_counts.get(role, 0) + 1; role_counts[role] = index
        all_points = np.concatenate([obs.points_world for obs in cluster], axis=0)
        objects.append({
            "id": f"{prefix}{index}",
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
    rlbench_low_dim_override = Path(args.rlbench_low_dim_obs).expanduser().resolve() if args.rlbench_low_dim_obs else None
    rlbench_low_dim_path = resolve_rlbench_low_dim_path(episode_dir, rlbench_low_dim_override)
    rlbench_observations = load_rlbench_observations(episode_dir, rlbench_low_dim_override)
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
