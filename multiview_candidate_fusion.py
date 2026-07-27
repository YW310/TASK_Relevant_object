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
    parser.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional 3D bbox IoU threshold. When > 0, two same-role observations also merge "
            "whenever their bbox IoU meets this threshold, even if their centroids are farther "
            "apart than --cluster-distance-m (this is an OR with the centroid check, not an AND: "
            "it is meant to catch the same physical object whose centroid estimate drifted, e.g. "
            "because of noisy depth or a partial mask, not to make merging stricter)."
        ),
    )
    parser.add_argument(
        "--nearest-distance-m",
        type=float,
        default=None,
        help=(
            "Optional point-cloud nearest-distance threshold. When set, two same-role "
            "observations also merge whenever their point clouds come within this distance, "
            "even if their centroids are farther apart than --cluster-distance-m (OR with the "
            "centroid check, same rationale as --bbox-iou-threshold)."
        ),
    )
    parser.add_argument(
        "--track-distance-m",
        type=float,
        default=0.15,
        help=(
            "Max centroid displacement (meters) between consecutive processed frames for a "
            "same-role fused object to keep its id (e.g. 'T1') across frames. Without this, ids "
            "are re-derived from scratch every frame by sorting clusters, which can silently "
            "flip which physical object is 'T1' vs 'T2' between frames whenever their sort order "
            "changes -- increase this if --frame-interval is large and objects move a lot between "
            "selected frames, decrease it if unrelated objects of the same role are close together."
        ),
    )
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


def pairwise_should_merge(a: Observation3D, b: Observation3D, args: argparse.Namespace) -> bool:
    if a.role != b.role:
        return False
    centroid_ok = np.linalg.norm(a.centroid_world - b.centroid_world) <= args.cluster_distance_m
    iou_ok = args.bbox_iou_threshold > 0 and bbox_iou_3d(a.bbox3d_world, b.bbox3d_world) >= args.bbox_iou_threshold
    nearest_ok = args.nearest_distance_m is not None and nearest_mean_distance(a.points_world, b.points_world) <= args.nearest_distance_m
    # OR, not AND: --bbox-iou-threshold / --nearest-distance-m are additional ways to detect the
    # same physical object (e.g. when its centroid estimate drifts due to noisy depth or a
    # partial mask), not extra requirements layered on top of the centroid check.
    return centroid_ok or iou_ok or nearest_ok


def cluster_observations(observations: Sequence[Observation3D], args: argparse.Namespace) -> list[list[Observation3D]]:
    """Group observations into connected components under ``pairwise_should_merge``.

    A single greedy pass (assign each new observation to the first existing
    cluster it matches, never revisiting earlier clusters) is order-dependent:
    two clusters that should ultimately be joined by a later "bridging"
    observation can be left permanently separate if that observation happens
    to match an earlier cluster first. That fragments one real-world object
    into multiple fused objects of different point counts/sizes, which show
    up as overlapping, differently-scaled boxes on the same physical object.
    Union-find over all pairwise matches guarantees a single connected
    component regardless of processing order.
    """
    n = len(observations)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for i in range(n):
        for j in range(i + 1, n):
            if pairwise_should_merge(observations[i], observations[j], args):
                union(i, j)

    groups: dict[int, list[Observation3D]] = {}
    for i, obs in enumerate(observations):
        groups.setdefault(find(i), []).append(obs)
    clusters = list(groups.values())
    warn_near_miss_unmerged_clusters(clusters, args)
    return clusters


def warn_near_miss_unmerged_clusters(clusters: list[list[Observation3D]], args: argparse.Namespace) -> None:
    """Print a diagnostic when two same-role clusters stay separate despite being
    suspiciously close, so duplicate/overlapping boxes on one physical object (a
    real fragmentation failure mode, not a rendering bug) are easy to spot from
    the fusion script's own output instead of only being noticed later in a
    visualization overlay.
    """
    diagnostic_radius_m = max(args.cluster_distance_m * 5, 0.10)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a, b = clusters[i][0], clusters[j][0]
            if a.role != b.role:
                continue
            centroid_a = np.mean([obs.centroid_world for obs in clusters[i]], axis=0)
            centroid_b = np.mean([obs.centroid_world for obs in clusters[j]], axis=0)
            dist = float(np.linalg.norm(centroid_a - centroid_b))
            if dist <= diagnostic_radius_m:
                cams_a = sorted({obs.camera for obs in clusters[i]})
                cams_b = sorted({obs.camera for obs in clusters[j]})
                print(
                    f"[warn] two '{a.role}' clusters stayed separate but are only {dist:.3f}m apart "
                    f"(cameras {cams_a} vs {cams_b}); this often means the same physical object got "
                    "fragmented into duplicate/overlapping boxes. Consider raising --cluster-distance-m "
                    "or setting --nearest-distance-m/--bbox-iou-threshold to also catch this case.",
                    file=sys.stderr,
                )


def solve_min_cost_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Solve a square min-cost bipartite assignment (Hungarian algorithm).

    O(n^3) Kuhn-Munkres with potentials; no scipy dependency so this module
    keeps working with plain ``python3`` (only numpy/PIL are required).
    Returns one (row, col) pair per row, covering every row and column
    exactly once. ``cost`` must be square.
    """
    n = cost.shape[0]
    assert cost.shape[1] == n, "solve_min_cost_assignment requires a square cost matrix"
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = 1-indexed row currently assigned to column j
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = []
    for j in range(1, n + 1):
        if p[j] != 0:
            pairs.append((p[j] - 1, j - 1))
    return pairs


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


def assign_object_ids(
    clusters: list[list[Observation3D]],
    track_state: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign fused-object ids that stay consistent across frames.

    Re-deriving ids every frame from a fresh ``role_counts`` counter combined
    with sorting clusters by x-coordinate is order-dependent across frames:
    if two same-role objects' relative x-position (or which clusters exist at
    all) changes between frames, the sort order changes and the same physical
    object can flip from e.g. "T1" to "T2". Instead, match each frame's
    clusters to the previous processed frame's tracked objects with a
    globally optimal Hungarian-algorithm assignment (minimizing total centroid
    distance, restricted to pairs within ``args.track_distance_m``; see
    ``solve_min_cost_assignment``) and inherit that id; only assign a
    brand-new id when no previous track is close enough.

    ``track_state`` is ``{"tracks": {role: [{"index": int, "centroid": [...]},
    ...]}, "next_index": {role: int}}`` from the previous frame and is not
    mutated; the updated state to carry into the next frame is returned
    alongside the objects. ``next_index`` is tracked separately from (and
    monotonically, regardless of) the currently live tracks so a role's id
    counter never gets reused after its tracks briefly disappear (e.g. one
    frame of occlusion) and then reappear.
    """
    role_clusters: dict[str, list[list[Observation3D]]] = {}
    for cluster in clusters:
        role_clusters.setdefault(cluster[0].role, []).append(cluster)

    prev_tracks_by_role: dict[str, list[dict[str, Any]]] = track_state.get("tracks", {})
    next_index_by_role: dict[str, int] = dict(track_state.get("next_index", {}))

    objects: list[dict[str, Any]] = []
    new_tracks_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in sorted(role_clusters):
        prefix = ROLE_OBJECT_PREFIX.get(role, f"{role}_obj")
        prev_tracks = prev_tracks_by_role.get(role, [])
        next_index = next_index_by_role.get(role, 0)
        role_clusters_list = role_clusters[role]
        centroids = [
            np.concatenate([obs.points_world for obs in cluster], axis=0).mean(axis=0)
            for cluster in role_clusters_list
        ]

        # Solve the globally optimal (cluster, previous track) matching with the
        # Hungarian algorithm, rather than a greedy walk that can settle for a
        # locally-available match and leave a genuinely closer pairing unmatched
        # elsewhere. Matching is framed as a square assignment problem: real
        # (cluster, track) pairs cost their centroid distance if within
        # ``args.track_distance_m`` (else a large forbidden cost), and each
        # cluster/track also gets an "abstain" dummy costing just over
        # ``args.track_distance_m`` -- strictly more than any allowed real
        # match, so the solver always prefers a real match when one exists,
        # but nothing is forced into a bad match just to complete the
        # assignment (crucially the abstain cost must NOT be 0: an all-dummy
        # "everyone stays unmatched" solution would otherwise always beat any
        # real match, since every real distance is >= 0).
        n_clusters = len(role_clusters_list)
        n_prev = len(prev_tracks)
        assigned_index_by_cluster: dict[int, int] = {}
        if n_clusters and n_prev:
            size = n_clusters + n_prev
            abstain_cost = args.track_distance_m + 1e-6
            forbidden_cost = abstain_cost * 1000.0 + 1e6
            cost = np.full((size, size), forbidden_cost, dtype=float)
            for cluster_index, centroid in enumerate(centroids):
                for prev_index, track in enumerate(prev_tracks):
                    dist = float(np.linalg.norm(centroid - np.array(track["centroid"])))
                    if dist <= args.track_distance_m:
                        cost[cluster_index, prev_index] = dist
            # Each cluster may abstain (stay unmatched) via its own dummy column.
            for cluster_index in range(n_clusters):
                cost[cluster_index, n_prev + cluster_index] = abstain_cost
            # Each previous track may abstain (stay unmatched) via its own dummy row.
            for prev_index in range(n_prev):
                cost[n_clusters + prev_index, prev_index] = abstain_cost
            # Dummy-vs-dummy padding cells only complete the permutation; they carry
            # no real meaning, so they're free.
            cost[n_clusters:, n_prev:] = 0.0

            for row, col in solve_min_cost_assignment(cost):
                if row < n_clusters and col < n_prev:
                    assigned_index_by_cluster[row] = prev_tracks[col]["index"]

        assigned_tracks: list[dict[str, Any]] = []
        for cluster_index, cluster in enumerate(role_clusters_list):
            all_points = np.concatenate([obs.points_world for obs in cluster], axis=0)
            centroid = centroids[cluster_index]
            if cluster_index in assigned_index_by_cluster:
                index = assigned_index_by_cluster[cluster_index]
            else:
                next_index += 1
                index = next_index
            objects.append({
                "id": f"{prefix}{index}",
                "role": role,
                "points_world": all_points.tolist(),
                "centroid_world": centroid.tolist(),
                "bbox3d_world": np.stack([all_points.min(axis=0), all_points.max(axis=0)]).tolist(),
                "visible_camera": sorted({obs.camera for obs in cluster}),
                "mask_area": int(sum(int(obs.candidate.get("mask_area_pixels", 0)) for obs in cluster)),
                "sam_score": float(np.mean([float(obs.candidate.get("score", 0.0)) for obs in cluster])),
                "observations": [observation_to_json(obs) for obs in cluster],
            })
            assigned_tracks.append({"index": index, "centroid": centroid.tolist()})
        new_tracks_by_role[role] = assigned_tracks
        next_index_by_role[role] = next_index
    objects.sort(key=lambda obj: (obj["role"], obj["id"].__len__(), obj["id"]))
    new_track_state = {"tracks": new_tracks_by_role, "next_index": next_index_by_role}
    return objects, new_track_state


def fuse_frame(
    frame: Mapping[str, Any],
    episode_dir: Path,
    camera_params: Mapping[str, dict[str, np.ndarray]],
    rlbench_observations: Sequence[Any],
    cameras: Sequence[str] | None,
    args: argparse.Namespace,
    track_state: dict[str, Any] | None = None,
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

    clusters = cluster_observations(observations, args)
    objects, updated_track_state = assign_object_ids(clusters, track_state or {}, args)
    if track_state is not None:
        track_state.clear()
        track_state.update(updated_track_state)
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
    track_state: dict[str, Any] = {}
    frames = [
        fuse_frame(frame, episode_dir, camera_params, rlbench_observations, cameras, args, track_state=track_state)
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
