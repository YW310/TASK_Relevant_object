#!/usr/bin/env python3
"""Fuse per-view SAM3 role candidates into frame-level 3D objects.

The script consumes ``episode_candidates.json`` produced by
``qwen_role_sam3_candidate_episode.py`` plus per-camera ``candidates.json`` and
mask PNG files. For each candidate, depth pixels inside the mask are
back-projected with camera intrinsics, transformed by camera extrinsics, and
fused by deterministic anchor-camera assignment with whole-hypothesis
geometric validation. Pairwise compatibility is only an assignment gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

ROLE_NAMES = ("target", "reference", "interaction_part")


@dataclass
class Observation3D:
    observation_id: str
    role_evidence: Mapping[str, float]
    provenance: Mapping[str, Any]
    camera: str
    candidate: Mapping[str, Any]
    points_world: np.ndarray
    centroid_world: np.ndarray
    bbox3d_world: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, help="RLBench/RLBench-exported episode directory.")
    parser.add_argument("--candidates-json", required=True, help="Path to episode_candidates.json.")
    parser.add_argument("--output-json", default=None, help="Default: <episode-dir>/frame_fused_candidates.json (a lightweight per-frame manifest).")
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
            "Optional 3D bbox IoU threshold. When > 0, two observations also merge "
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
            "Optional point-cloud nearest-distance threshold. When set, two "
            "observations also merge whenever their point clouds come within this distance, "
            "even if their centroids are farther apart than --cluster-distance-m (OR with the "
            "centroid check, same rationale as --bbox-iou-threshold)."
        ),
    )
    parser.add_argument("--max-hypothesis-diameter-m", type=float, default=0.50,
                        help="Maximum robust (1st--99th percentile) pooled point-cloud diameter after an insertion.")
    parser.add_argument("--max-size-ratio", type=float, default=4.0,
                        help="Maximum non-degenerate axis-wise 3D-box size ratio within a hypothesis.")
    parser.add_argument("--legacy-union-find", action="store_true",
                        help="DEPRECATED compatibility/debug mode: use the old pairwise transitive union-find partition.")
    parser.add_argument(
        "--track-distance-m",
        type=float,
        default=0.15,
        help=(
            "Max centroid displacement (meters) between consecutive processed frames for a "
            "fused object to keep its id (e.g. 'O1') across frames. Without this, ids "
            "are re-derived from scratch every frame by sorting clusters, which can silently "
            "flip which physical object is 'O1' vs 'O2' between frames whenever their sort order "
            "changes -- increase this if --frame-interval is large and objects move a lot between "
            "selected frames, decrease it if unrelated objects are close together."
        ),
    )
    parser.add_argument(
        "--min-fused-points",
        type=int,
        default=0,
        help=(
            "Drop a fused (post-clustering) object if its combined point cloud (summed across all "
            "its observations/cameras) has fewer than this many points. 0 (default) disables this "
            "filter. Useful for dropping small/noisy single-camera fragments (e.g. a stray partial "
            "mask that didn't merge with the real object) that would otherwise show up as spurious "
            "extra tiny boxes."
        ),
    )
    parser.add_argument(
        "--min-bbox-diagonal-m",
        type=float,
        default=0.0,
        help=(
            "Drop a fused (post-clustering) object if its 3D bounding box diagonal is smaller than "
            "this many meters. 0.0 (default) disables this filter. Also useful for dropping small/ "
            "noisy fragments; unlike --min-fused-points this is robust to candidates with many "
            "points crammed into a tiny volume (e.g. a thin sliver mask)."
        ),
    )
    parser.add_argument(
        "--save-object-summary",
        action="store_true",
        help=(
            "Save an object-level summary JSON (track/trajectory and per-frame candidate "
            "decision inputs) for downstream reasoning, e.g. Qwen3-VL target/reference "
            "selection that should use visual/geometric evidence beyond instruction text."
        ),
    )
    parser.add_argument(
        "--object-summary-json",
        default=None,
        help=(
            "Optional output path for the object-level summary JSON. Default when "
            "--save-object-summary is set: <output-json dir>/object_summary.json."
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


def canonicalize_legacy_candidates(candidates: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Defensive adapter for old one-role-per-mask candidate artifacts.

    Legacy rows are grouped only at IoU >= .80 or smaller-mask coverage >= .90;
    touching/adjacent instances therefore remain distinct. The best scoring
    mask is retained and role evidence is noisy-OR aggregated with raw audit
    values preserved.
    """
    if all(candidate.get("canonical_observation_id") for candidate in candidates):
        return [dict(candidate) for candidate in candidates], []
    groups: list[list[tuple[Mapping[str, Any], np.ndarray]]] = []
    suppressed: list[dict[str, Any]] = []
    loaded = [(candidate, load_mask(Path(str(candidate["mask_path"])))) for candidate in candidates]
    for candidate, mask in sorted(loaded, key=lambda pair: float(pair[0].get("score", 0.0)), reverse=True):
        match = None
        match_metrics = (0.0, 0.0)
        for index, group in enumerate(groups):
            other = group[0][1]
            inter = int(np.logical_and(mask, other).sum())
            union = int(np.logical_or(mask, other).sum())
            smaller = min(int(mask.sum()), int(other.sum()))
            iou = inter / union if union else 0.0
            coverage = inter / smaller if smaller else 0.0
            if iou >= 0.80 or coverage >= 0.90:
                if match is not None:  # ambiguous bridge: conservatively keep separate
                    match = None
                    break
                match, match_metrics = index, (iou, coverage)
        if match is None:
            groups.append([(candidate, mask)])
        else:
            groups[match].append((candidate, mask))
            suppressed.append({"candidate_id": candidate.get("id"), "reason": "legacy_overlap_or_containment",
                               "mask_iou": match_metrics[0], "coverage": match_metrics[1]})
    output = []
    for index, group in enumerate(groups, 1):
        representative = dict(group[0][0])
        canonical_id = f"legacy-C{index}"
        role_scores: dict[str, dict[str, Any]] = {}
        provenance = []
        for candidate, mask in group:
            role, score = str(candidate.get("role", "")), float(candidate.get("score", 0.0))
            entry = role_scores.setdefault(role, {"aggregation": "noisy_or", "raw_scores": [], "score": 0.0})
            entry["raw_scores"].append(score)
            entry["score"] = 1.0 - (1.0 - entry["score"]) * (1.0 - score)
            inter = int(np.logical_and(mask, group[0][1]).sum())
            union = int(np.logical_or(mask, group[0][1]).sum())
            provenance.append({"role": role, "source_prompt": candidate.get("source_prompt") or candidate.get("text_prompt"),
                               "prompt_index": candidate.get("prompt_index"), "sam_output_index": candidate.get("sam_output_index"),
                               "original_candidate_id": candidate.get("id"), "score": score,
                               "mask_area": int(mask.sum()), "overlap_with_canonical_mask": inter / union if union else 0.0})
        representative.update({"id": canonical_id, "canonical_observation_id": canonical_id,
                               "role_scores": role_scores, "prompt_provenance": provenance})
        output.append(representative)
    return output, suppressed


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


def symmetric_percentile_nearest_distance(a: np.ndarray, b: np.ndarray, percentile: float = 75.0) -> float:
    """Robust symmetric surface distance, bounded to keep assignment inexpensive."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    aa, bb = a[::max(1, len(a) // 256)], b[::max(1, len(b) // 256)]
    distances = np.sqrt(((aa[:, None, :] - bb[None, :, :]) ** 2).sum(axis=2))
    directed = np.concatenate((distances.min(axis=1), distances.min(axis=0)))
    return float(np.percentile(directed, percentile))


def pairwise_should_merge(a: Observation3D, b: Observation3D, args: argparse.Namespace) -> bool:
    centroid_ok = np.linalg.norm(a.centroid_world - b.centroid_world) <= args.cluster_distance_m
    iou_ok = args.bbox_iou_threshold > 0 and bbox_iou_3d(a.bbox3d_world, b.bbox3d_world) >= args.bbox_iou_threshold
    nearest_ok = args.nearest_distance_m is not None and nearest_mean_distance(a.points_world, b.points_world) <= args.nearest_distance_m
    # OR, not AND: --bbox-iou-threshold / --nearest-distance-m are additional ways to detect the
    # same physical object (e.g. when its centroid estimate drifts due to noisy depth or a
    # partial mask), not extra requirements layered on top of the centroid check.
    return centroid_ok or iou_ok or nearest_ok


def _confidence(obs: Observation3D) -> float:
    return max(obs.role_evidence.values(), default=float(obs.candidate.get("score", 0.0)))


def _hypothesis_is_valid(cluster: Sequence[Observation3D], args: argparse.Namespace) -> bool:
    """Validate the *complete* proposed hypothesis, never merely one supporting edge."""
    if len({obs.camera for obs in cluster}) != len(cluster):
        return False
    for i, a in enumerate(cluster):
        for b in cluster[i + 1:]:
            if np.linalg.norm(a.centroid_world - b.centroid_world) > args.cluster_distance_m:
                return False
            sizes_a = np.maximum(a.bbox3d_world[1] - a.bbox3d_world[0], 1e-6)
            sizes_b = np.maximum(b.bbox3d_world[1] - b.bbox3d_world[0], 1e-6)
            # Ignore sub-millimetre axes (flat/partial masks make their ratios meaningless).
            valid_axes = np.maximum(sizes_a, sizes_b) >= 1e-3
            if np.any(np.maximum(sizes_a[valid_axes] / sizes_b[valid_axes], sizes_b[valid_axes] / sizes_a[valid_axes]) > args.max_size_ratio):
                return False
            if args.bbox_iou_threshold > 0 and bbox_iou_3d(a.bbox3d_world, b.bbox3d_world) < args.bbox_iou_threshold:
                return False
            if args.nearest_distance_m is not None and symmetric_percentile_nearest_distance(a.points_world, b.points_world) > args.nearest_distance_m:
                return False
    points = np.concatenate([obs.points_world for obs in cluster])
    robust_extent = np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)
    return args.max_hypothesis_diameter_m <= 0 or np.linalg.norm(robust_extent) <= args.max_hypothesis_diameter_m


def _association_cost(obs: Observation3D, cluster: Sequence[Observation3D], args: argparse.Namespace) -> float | None:
    if not any(pairwise_should_merge(obs, other, args) for other in cluster):
        return None  # pairwise_should_merge is only a cheap compatibility gate.
    proposed = [*cluster, obs]
    if not _hypothesis_is_valid(proposed, args):
        return None
    anchor = cluster[0]
    centroid = float(np.linalg.norm(obs.centroid_world - np.median([x.centroid_world for x in cluster], axis=0)))
    sizes_a = np.maximum(obs.bbox3d_world[1] - obs.bbox3d_world[0], 1e-6)
    sizes_b = np.maximum(anchor.bbox3d_world[1] - anchor.bbox3d_world[0], 1e-6)
    size_residual = float(np.mean(np.abs(np.log(sizes_a / sizes_b))))
    overlap_penalty = 1.0 - bbox_iou_3d(obs.bbox3d_world, anchor.bbox3d_world)
    surface = symmetric_percentile_nearest_distance(obs.points_world, anchor.points_world)
    surface_term = surface / max(args.cluster_distance_m, 1e-9) if np.isfinite(surface) else 1.0
    return centroid / max(args.cluster_distance_m, 1e-9) + 0.35 * size_residual + 0.25 * overlap_penalty + 0.25 * surface_term


def legacy_union_find_clusters(observations: Sequence[Observation3D], args: argparse.Namespace) -> list[list[Observation3D]]:
    """Deprecated reproduction of pairwise transitive connected components."""
    parent = list(range(len(observations)))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            if observations[i].camera != observations[j].camera and pairwise_should_merge(observations[i], observations[j], args):
                a, b = find(i), find(j)
                if a != b: parent[b] = a
    groups: dict[int, list[Observation3D]] = {}
    for i, obs in enumerate(observations): groups.setdefault(find(i), []).append(obs)
    return list(groups.values())


def cluster_observations(observations: Sequence[Observation3D], args: argparse.Namespace) -> list[list[Observation3D]]:
    """Assign one camera at a time to confidence-seeded object hypotheses."""
    if getattr(args, "legacy_union_find", False):
        print("[warn] --legacy-union-find is deprecated and may create inconsistent hypotheses", file=sys.stderr)
        return legacy_union_find_clusters(observations, args)
    by_camera: dict[str, list[Observation3D]] = {}
    for obs in observations:
        by_camera.setdefault(obs.camera, []).append(obs)
    if not by_camera:
        return []
    # Deterministic policy: camera with the highest-confidence observation anchors;
    # lexical camera name and observation id break ties. Remaining cameras follow
    # the same descending-confidence/lexical policy.
    camera_order = sorted(by_camera, key=lambda c: (-max(_confidence(o) for o in by_camera[c]), c))
    for camera in camera_order:
        by_camera[camera].sort(key=lambda o: (-_confidence(o), o.observation_id))
    clusters: list[list[Observation3D]] = [[obs] for obs in by_camera[camera_order[0]]]
    for camera in camera_order[1:]:
        camera_obs, nh = by_camera[camera], len(clusters)
        no = len(camera_obs)
        n = no + nh
        dummy_cost, blocked = 2.0, 1e6
        cost = np.full((n, n), blocked, dtype=float)
        for i, obs in enumerate(camera_obs):
            for j, cluster in enumerate(clusters):
                value = _association_cost(obs, cluster, args)
                if value is not None:
                    cost[i, j] = value
            cost[i, nh + i] = dummy_cost  # explicit new-hypothesis assignment
        for j in range(nh):
            cost[no + j, j] = dummy_cost
        cost[no:, nh:] = 0.0
        for row, col in solve_min_cost_assignment(cost):
            if row < no:
                if col < nh and cost[row, col] < blocked:
                    clusters[col].append(camera_obs[row])
                else:
                    clusters.append([camera_obs[row]])
    warn_near_miss_unmerged_clusters(clusters, args)
    return clusters


def warn_near_miss_unmerged_clusters(clusters: list[list[Observation3D]], args: argparse.Namespace) -> None:
    """Print a diagnostic when two clusters stay separate despite being
    suspiciously close, so duplicate/overlapping boxes on one physical object (a
    real fragmentation failure mode, not a rendering bug) are easy to spot from
    the fusion script's own output instead of only being noticed later in a
    visualization overlay.
    """
    diagnostic_radius_m = max(args.cluster_distance_m * 5, 0.10)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a, b = clusters[i][0], clusters[j][0]
            centroid_a = np.mean([obs.centroid_world for obs in clusters[i]], axis=0)
            centroid_b = np.mean([obs.centroid_world for obs in clusters[j]], axis=0)
            dist = float(np.linalg.norm(centroid_a - centroid_b))
            if dist <= diagnostic_radius_m:
                cams_a = sorted({obs.camera for obs in clusters[i]})
                cams_b = sorted({obs.camera for obs in clusters[j]})
                print(
                    f"[warn] two observation clusters stayed separate but are only {dist:.3f}m apart "
                    f"(cameras {cams_a} vs {cams_b}); this often means the same physical object got "
                    "fragmented into duplicate/overlapping boxes. Consider raising --cluster-distance-m "
                    "or setting --nearest-distance-m/--bbox-iou-threshold to also catch this case.",
                    file=sys.stderr,
                )


def filter_small_clusters(clusters: list[list[Observation3D]], args: argparse.Namespace) -> list[list[Observation3D]]:
    """Drop fused (post-clustering) objects that are too small to be real.

    A common source of spurious extra boxes is a small, low-confidence,
    single-camera mask that didn't merge with the real object's cluster (its
    centroid drifted just far enough away) -- see ``warn_near_miss_unmerged_clusters``.
    Rather than only warning about it, this lets --min-fused-points and/or
    --min-bbox-diagonal-m actually remove such clusters from the output once
    clustering is done, based on the FUSED (combined multi-camera) point count
    / bbox size, not any single camera's raw candidate stats -- so a real
    object that is only barely visible from one camera but has enough total
    points/extent still survives.
    """
    if args.min_fused_points <= 0 and args.min_bbox_diagonal_m <= 0.0:
        return clusters
    kept = []
    for cluster in clusters:
        all_points = np.concatenate([obs.points_world for obs in cluster], axis=0)
        if args.min_fused_points > 0 and len(all_points) < args.min_fused_points:
            print(
                f"[info] dropping small observation cluster (cameras "
                f"{sorted({obs.camera for obs in cluster})}): {len(all_points)} points < "
                f"--min-fused-points {args.min_fused_points}.",
                file=sys.stderr,
            )
            continue
        diagonal = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
        if args.min_bbox_diagonal_m > 0.0 and diagonal < args.min_bbox_diagonal_m:
            print(
                f"[info] dropping small observation cluster (cameras "
                f"{sorted({obs.camera for obs in cluster})}): bbox diagonal {diagonal:.3f}m < "
                f"--min-bbox-diagonal-m {args.min_bbox_diagonal_m}.",
                file=sys.stderr,
            )
            continue
        kept.append(cluster)
    return kept


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
        "observation_id": obs.observation_id,
        "role_evidence": dict(obs.role_evidence),
        "provenance": dict(obs.provenance),
        "mask_path": c.get("mask_path"),
        "crop_path": c.get("crop_path"),
        "masked_crop_path": c.get("masked_crop_path"),
        "mask_area": int(c.get("mask_area_pixels", 0)),
        "sam_score": c.get("score"),
        "mask_bbox_xyxy": c.get("mask_bbox_xyxy"),
        "_points_world": obs.points_world,
        "centroid_world": obs.centroid_world.tolist(),
        "bbox3d_world": obs.bbox3d_world.tolist(),
    }


def aggregate_role_evidence(observations: Sequence[Observation3D], frame_id: str | None = None) -> dict[str, Any]:
    """Combine semantic evidence while leaving physical object identity untouched."""
    evidence: dict[str, Any] = {}
    total_mass = 0.0
    for role in ROLE_NAMES:
        supporting = [obs for obs in observations if float(obs.role_evidence.get(role, 0.0)) > 0.0]
        mass = float(sum(float(obs.role_evidence.get(role, 0.0)) for obs in supporting))
        total_mass += mass
        evidence[role] = {
            "score_mass": mass,
            "supporting_prompts": sorted({str(obs.provenance.get("prompt")) for obs in supporting if obs.provenance.get("prompt")}),
            "cameras": sorted({obs.camera for obs in supporting}),
            "frames": [frame_id] if supporting and frame_id is not None else [],
        }
    for value in evidence.values():
        value["probability"] = value["score_mass"] / total_mass if total_mass > 0 else 0.0
    return evidence


def aggregate_summary_role_evidence(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame evidence for a tracked object without selecting a role."""
    result: dict[str, Any] = {}
    total_mass = 0.0
    for role in ROLE_NAMES:
        entries = [frame.get("role_evidence", {}).get(role, {}) for frame in frames]
        mass = float(sum(float(entry.get("score_mass", 0.0)) for entry in entries))
        total_mass += mass
        result[role] = {
            "score_mass": mass,
            "supporting_prompts": sorted({prompt for entry in entries for prompt in entry.get("supporting_prompts", [])}),
            "cameras": sorted({camera for entry in entries for camera in entry.get("cameras", [])}),
            "frames": sorted({frame for entry in entries for frame in entry.get("frames", [])}),
        }
    for entry in result.values():
        entry["probability"] = entry["score_mass"] / total_mass if total_mass > 0 else 0.0
    return result


def candidate_to_observation(
    candidate: Mapping[str, Any],
    camera: str,
    frame_id: str,
    points_world: np.ndarray,
    centroid_world: np.ndarray,
    bbox3d_world: np.ndarray,
) -> Observation3D:
    """Adapt canonical and legacy scalar-role candidate records."""
    legacy_role = str(candidate.get("role", ""))
    prompt = candidate.get("source_prompt") or candidate.get("text_prompt")
    score = float(candidate.get("score", 0.0))
    candidate_id = str(candidate.get("id", "unknown"))
    canonical_id = str(candidate.get("canonical_observation_id", candidate_id))
    raw_role_scores = candidate.get("role_scores")
    if isinstance(raw_role_scores, Mapping):
        role_evidence = {role: float(raw_role_scores.get(role, {}).get("score", 0.0)) for role in ROLE_NAMES}
        provenance: Mapping[str, Any] = {"prompt_provenance": candidate.get("prompt_provenance", []),
                                        "canonical_observation_id": canonical_id}
    else:
        role_evidence = {role: score if role == legacy_role else 0.0 for role in ROLE_NAMES}
        provenance = {"role": legacy_role, "prompt": prompt, "candidate_id": candidate_id}
    return Observation3D(
        observation_id=f"{frame_id}:{camera}:{canonical_id}",
        role_evidence=role_evidence,
        provenance=provenance,
        camera=camera,
        candidate=candidate,
        points_world=points_world,
        centroid_world=centroid_world,
        bbox3d_world=bbox3d_world,
    )


def stats_from_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def relation_label(delta: np.ndarray) -> list[str]:
    labels = []
    if delta[0] > 0:
        labels.append("right_of")
    elif delta[0] < 0:
        labels.append("left_of")
    if delta[1] > 0:
        labels.append("front_of")
    elif delta[1] < 0:
        labels.append("behind")
    if delta[2] > 0:
        labels.append("above")
    elif delta[2] < 0:
        labels.append("below")
    return labels


def build_object_summary(
    frames: Iterable[Mapping[str, Any]],
    result: Mapping[str, Any],
    candidates_summary: Mapping[str, Any],
    *,
    schema_version: int,
    generation_id: str,
) -> dict[str, Any]:
    """Build compact track statistics while consuming frame records once.

    ``frames`` may be a generator backed by per-frame JSON files.  Geometry is
    never retained or copied into the summary; only scalar metadata and paths
    needed by the role-decision stage are accumulated.
    """
    by_object_id: dict[str, dict[str, Any]] = {}
    frame_decision_inputs = []

    for frame in frames:
        frame_id = str(frame.get("frame_id"))
        frame_index = frame.get("frame_index")
        for obj in frame.get("objects", []):
            object_id = str(obj["id"])
            track = by_object_id.setdefault(
                object_id,
                {"object_id": object_id, "frames": []},
            )
            track["frames"].append(
                {
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "frame_ref": frame.get("frame_ref"),
                    "centroid_world": obj["centroid_world"],
                    "bbox3d_world": obj["bbox3d_world"],
                    "point_count": int(obj.get("point_count", len(obj.get("points_world", [])))),
                    "visible_camera": obj.get("visible_camera", []),
                    "camera_count": len(obj.get("visible_camera", [])),
                    "mask_area": int(obj.get("mask_area", 0)),
                    "sam_score": float(obj.get("sam_score", 0.0)),
                    "observation_count": len(obj.get("observations", [])),
                    "role_evidence": obj.get("role_evidence", {}),
                    "observations": [
                        {
                            "camera": obs.get("camera"),
                            "candidate_id": obs.get("candidate_id"),
                            "observation_id": obs.get("observation_id"),
                            "role_evidence": obs.get("role_evidence", {}),
                            "provenance": obs.get("provenance", {}),
                            "mask_path": obs.get("mask_path"),
                            "crop_path": obs.get("crop_path"),
                            "masked_crop_path": obs.get("masked_crop_path"),
                            "mask_area": obs.get("mask_area"),
                            "sam_score": obs.get("sam_score"),
                            "mask_bbox_xyxy": obs.get("mask_bbox_xyxy"),
                        }
                        for obs in obj.get("observations", [])
                    ],
                }
            )

        objects = list(frame.get("objects", []))
        candidates = []
        for obj in objects:
            candidates.append(_summary_object_record(obj))
        pairwise_relations = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                src, dst = objects[i], objects[j]
                c_src = np.asarray(src.get("centroid_world", [0.0] * 3), dtype=np.float64)
                c_dst = np.asarray(dst.get("centroid_world", [0.0] * 3), dtype=np.float64)
                delta = c_dst - c_src
                pairwise_relations.append({
                    "source_object_id": src.get("id"), "target_object_id": dst.get("id"),
                    "distance_m": float(np.linalg.norm(delta)), "delta_world": delta.tolist(),
                    "source_to_target_labels": relation_label(delta),
                    "target_to_source_labels": relation_label(-delta),
                })
        frame_decision_inputs.append({
            "frame_id": frame_id, "frame_index": frame_index,
            "frame_ref": frame.get("frame_ref"),
            "instruction_prior": candidates_summary.get("instruction"),
            "role_spec_prior": candidates_summary.get("role_spec"),
            "candidate_objects": candidates, "pairwise_relations": pairwise_relations,
        })

    object_tracks = []
    for object_id in sorted(by_object_id):
        track = by_object_id[object_id]
        frames_sorted = sorted(track["frames"], key=lambda item: (item["frame_index"] is None, item["frame_index"], item["frame_id"]))
        centroids = [np.asarray(item["centroid_world"], dtype=np.float64) for item in frames_sorted]
        motion_path_length_m = float(
            sum(np.linalg.norm(centroids[i] - centroids[i - 1]) for i in range(1, len(centroids)))
        ) if len(centroids) >= 2 else 0.0

        bbox_diagonals = []
        bbox_sizes = []
        for item in frames_sorted:
            bbox = np.asarray(item["bbox3d_world"], dtype=np.float64)
            size = bbox[1] - bbox[0]
            bbox_sizes.append(size)
            bbox_diagonals.append(float(np.linalg.norm(size)))

        camera_histogram: dict[str, int] = {}
        for item in frames_sorted:
            for camera in item.get("visible_camera", []):
                camera_histogram[camera] = camera_histogram.get(camera, 0) + 1

        object_tracks.append(
            {
                "object_id": object_id,
                "role_evidence": aggregate_summary_role_evidence(frames_sorted),
                "first_frame_id": frames_sorted[0]["frame_id"],
                "last_frame_id": frames_sorted[-1]["frame_id"],
                "first_frame_index": frames_sorted[0]["frame_index"],
                "last_frame_index": frames_sorted[-1]["frame_index"],
                "lifespan_frames": len(frames_sorted),
                "frames_visible": [item["frame_id"] for item in frames_sorted],
                "camera_set": sorted(camera_histogram),
                "camera_histogram": camera_histogram,
                "centroid_mean_world": np.mean(np.stack(centroids, axis=0), axis=0).tolist(),
                "motion_path_length_m": motion_path_length_m,
                "point_count_stats": stats_from_values([float(item["point_count"]) for item in frames_sorted]),
                "mask_area_stats": stats_from_values([float(item["mask_area"]) for item in frames_sorted]),
                "sam_score_stats": stats_from_values([float(item["sam_score"]) for item in frames_sorted]),
                "camera_count_stats": stats_from_values([float(item["camera_count"]) for item in frames_sorted]),
                "bbox_diagonal_m_stats": stats_from_values(bbox_diagonals),
                "bbox_size_xyz_mean": np.mean(np.stack(bbox_sizes, axis=0), axis=0).tolist() if bbox_sizes else None,
                "trajectory": frames_sorted,
            }
        )

    return {
        "schema_version": schema_version,
        "generation_id": generation_id,
        "episode_dir": result.get("episode_dir"),
        "source_candidates_json": result.get("source_candidates_json"),
        "source_fused_json": None,
        "instruction_prior": candidates_summary.get("instruction"),
        "role_spec_prior": candidates_summary.get("role_spec"),
        "fusion_params": {
            "fusion_algorithm": result.get("fusion_algorithm"),
            "cluster_distance_m": result.get("cluster_distance_m"),
            "bbox_iou_threshold": result.get("bbox_iou_threshold"),
            "nearest_distance_m": result.get("nearest_distance_m"),
            "track_distance_m": result.get("track_distance_m"),
            "min_fused_points": result.get("min_fused_points"),
            "min_bbox_diagonal_m": result.get("min_bbox_diagonal_m"),
            "max_hypothesis_diameter_m": result.get("max_hypothesis_diameter_m"),
            "max_size_ratio": result.get("max_size_ratio"),
        },
        "object_tracks": object_tracks,
        "frame_decision_inputs": frame_decision_inputs,
    }


def _summary_object_record(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Strip a frame object to decision metadata (never embedded geometry)."""
    return {
        "object_id": obj.get("id"), "role_evidence": obj.get("role_evidence", {}),
        "centroid_world": obj.get("centroid_world"), "bbox3d_world": obj.get("bbox3d_world"),
        "visible_camera": obj.get("visible_camera", []),
        "camera_count": len(obj.get("visible_camera", [])),
        "point_count": int(obj.get("point_count", len(obj.get("points_world", [])))),
        "mask_area": obj.get("mask_area"), "sam_score": obj.get("sam_score"),
        "observation_count": len(obj.get("observations", [])),
        "observations": [{key: obs.get(key) for key in (
            "camera", "candidate_id", "observation_id", "role_evidence", "provenance",
            "mask_path", "crop_path", "masked_crop_path", "mask_area", "sam_score", "mask_bbox_xyxy"
        )} for obs in obj.get("observations", [])],
    }


def assign_object_ids(
    clusters: list[list[Observation3D]],
    track_state: dict[str, Any],
    args: argparse.Namespace,
    frame_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Track physical objects globally; semantic roles never partition identity."""
    prev_tracks: list[dict[str, Any]] = list(track_state.get("tracks", []))
    next_object_index = int(track_state.get("next_object_index", 0))
    centroids = [np.concatenate([o.points_world for o in c]).mean(axis=0) for c in clusters]
    assigned: dict[int, int] = {}
    if clusters and prev_tracks:
        nc, np_ = len(clusters), len(prev_tracks)
        size = nc + np_
        abstain = args.track_distance_m + 1e-6
        cost = np.full((size, size), abstain * 1000.0 + 1e6)
        for ci, centroid in enumerate(centroids):
            for pi, track in enumerate(prev_tracks):
                distance = float(np.linalg.norm(centroid - np.asarray(track["centroid"])))
                if distance <= args.track_distance_m:
                    cost[ci, pi] = distance
            cost[ci, np_ + ci] = abstain
        for pi in range(np_):
            cost[nc + pi, pi] = abstain
        cost[nc:, np_:] = 0.0
        for row, col in solve_min_cost_assignment(cost):
            if row < nc and col < np_:
                assigned[row] = int(prev_tracks[col]["index"])

    objects, tracks = [], []
    for ci, cluster in enumerate(clusters):
        all_points = np.concatenate([o.points_world for o in cluster])
        centroid = centroids[ci]
        if ci in assigned:
            index = assigned[ci]
        else:
            next_object_index += 1
            index = next_object_index
        objects.append({
            "id": f"O{index}",
            "role_evidence": aggregate_role_evidence(cluster, frame_id),
            "_points_world": all_points,
            "centroid_world": centroid.tolist(),
            "bbox3d_world": np.stack([all_points.min(axis=0), all_points.max(axis=0)]).tolist(),
            "visible_camera": sorted({o.camera for o in cluster}),
            "mask_area": int(sum(int(o.candidate.get("mask_area_pixels", 0)) for o in cluster)),
            "sam_score": float(np.mean([float(o.candidate.get("score", 0.0)) for o in cluster])),
            "observations": [observation_to_json(o) for o in cluster],
        })
        tracks.append({"index": index, "centroid": centroid.tolist()})
    objects.sort(key=lambda obj: int(str(obj["id"])[1:]))
    return objects, {"tracks": tracks, "next_object_index": next_object_index}


def _geometry_segment(value: Any) -> str:
    """Make one stable, filesystem/key-safe geometry path segment."""
    text = str(value) if value is not None else "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "unknown"


def save_frame_geometry(frame: Mapping[str, Any], output_path: Path) -> None:
    """Move transient point arrays from a fused frame into a compressed NPZ."""
    frame_key = _geometry_segment(frame.get("frame_id", frame.get("frame_index", "frame")))
    relative_path = Path("frames") / frame_key / "fused_geometry.npz"
    archive_path = output_path.parent / relative_path
    arrays: dict[str, np.ndarray] = {}

    for obj in frame.get("objects", []):
        object_id = _geometry_segment(obj.get("id"))
        object_key = f"{object_id}/points_world"
        points = np.asarray(obj.pop("_points_world"), dtype=np.float32)
        arrays[object_key] = points
        obj.update({
            "geometry_path": relative_path.as_posix(),
            "points_key": object_key,
            "point_count": int(len(points)),
        })
        used_keys: set[str] = set()
        for observation_index, obs in enumerate(obj.get("observations", []), start=1):
            camera = _geometry_segment(obs.get("camera"))
            candidate = _geometry_segment(obs.get("candidate_id") or obs.get("observation_id") or f"obs{observation_index}")
            stem = f"{object_id}/{camera}/{candidate}"
            unique_stem = stem
            suffix = 2
            while unique_stem in used_keys:
                unique_stem = f"{stem}_{suffix}"
                suffix += 1
            used_keys.add(unique_stem)
            points_key = f"{unique_stem}/points_world"
            obs_points = np.asarray(obs.pop("_points_world"), dtype=np.float32)
            arrays[points_key] = obs_points
            obs.update({
                "geometry_path": relative_path.as_posix(),
                "points_key": points_key,
                "point_count": int(len(obs_points)),
            })

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive_path, **arrays)


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
    canonicalization_suppressed: list[dict[str, Any]] = []
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
        canonical_candidates, legacy_suppressed = canonicalize_legacy_candidates(data.get("candidates", []))
        canonicalization_suppressed.extend({"camera": camera, **item} for item in legacy_suppressed)
        if legacy_suppressed:
            print(f"[info] frame_id={frame_id} camera={camera}: legacy canonicalization suppressed "
                  f"{len(legacy_suppressed)} duplicate candidates", file=sys.stderr)
        for cand in canonical_candidates:
            mask = load_mask(Path(cand["mask_path"]))
            points_cam = backproject_mask(depth, mask, params["intrinsics"], args.max_points_per_candidate)
            points_world = transform_points(points_cam, params["extrinsics"])
            if len(points_world) == 0:
                continue
            centroid = points_world.mean(axis=0)
            bbox = np.stack([points_world.min(axis=0), points_world.max(axis=0)])
            observations.append(candidate_to_observation(cand, camera, frame_id, points_world, centroid, bbox))

    same_camera_diagnostics: list[dict[str, Any]] = []
    args._same_camera_diagnostics = same_camera_diagnostics
    clusters = cluster_observations(observations, args)
    clusters = filter_small_clusters(clusters, args)
    objects, updated_track_state = assign_object_ids(clusters, track_state or {}, args, frame_id)
    if track_state is not None:
        track_state.clear()
        track_state.update(updated_track_state)
    return {"frame_index": frame.get("frame_index"), "frame_id": frame_id, "objects": objects,
            "diagnostics": {"canonicalization_suppressed": canonicalization_suppressed,
                            "attempted_same_camera_cluster_insertions": same_camera_diagnostics}}


FUSED_MANIFEST_SCHEMA_VERSION = 3


def _fusion_parameters(args: argparse.Namespace, rlbench_low_dim_path: Path, has_rlbench_observations: bool) -> dict[str, Any]:
    """Return the parameters that determine the per-frame fusion artifacts."""
    return {
        "cluster_distance_m": args.cluster_distance_m,
        "bbox_iou_threshold": args.bbox_iou_threshold,
        "nearest_distance_m": args.nearest_distance_m,
        "track_distance_m": args.track_distance_m,
        "min_fused_points": args.min_fused_points,
        "min_bbox_diagonal_m": args.min_bbox_diagonal_m,
        "max_hypothesis_diameter_m": args.max_hypothesis_diameter_m,
        "max_size_ratio": args.max_size_ratio,
        "max_points_per_candidate": args.max_points_per_candidate,
        "depth_mode": args.depth_mode,
        "depth_scale": args.depth_scale,
        "cameras": list(parse_csv(args.cameras) or []),
        "fusion_algorithm": "legacy_pairwise_union_find" if args.legacy_union_find else "anchor_gated_assignment_v1",
        "rlbench_low_dim_obs": str(rlbench_low_dim_path) if has_rlbench_observations else None,
        "invert_rlbench_extrinsics": bool(args.invert_rlbench_extrinsics),
    }


def _restore_track_state(frame: Mapping[str, Any], track_state: dict[str, Any]) -> None:
    """Advance tracking from a completed frame when resuming a partial run."""
    tracks = []
    next_object_index = int(track_state.get("next_object_index", 0))
    for obj in frame.get("objects", []):
        object_id = str(obj.get("id", ""))
        if not (object_id.startswith("O") and object_id[1:].isdigit()):
            continue
        index = int(object_id[1:])
        next_object_index = max(next_object_index, index)
        tracks.append({"index": index, "centroid": obj["centroid_world"]})
    track_state.update({"tracks": tracks, "next_object_index": next_object_index})


def _load_completed_frame(
    entry: Mapping[str, Any], output_dir: Path, generation_id: str,
) -> dict[str, Any] | None:
    """Validate and load a resumable frame artifact, or return ``None``."""
    if entry.get("status") != "complete":
        return None
    try:
        path = output_dir / str(entry["fused_objects_json"])
        frame = json.loads(path.read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError, TypeError):
        return None
    if str(frame.get("frame_id")) != str(entry.get("frame_id")):
        return None
    if frame.get("schema_version") != FUSED_MANIFEST_SCHEMA_VERSION:
        return None
    if frame.get("generation_id") != generation_id:
        return None
    return frame


def iter_manifest_frames(manifest: Mapping[str, Any], manifest_path: Path) -> Iterable[dict[str, Any]]:
    """Lazily read and validate complete frame artifacts in manifest order."""
    version = manifest.get("schema_version")
    generation_id = manifest.get("generation_id")
    if version != FUSED_MANIFEST_SCHEMA_VERSION or not isinstance(generation_id, str):
        raise ValueError(f"Unsupported fused manifest schema/generation: {version!r}/{generation_id!r}")
    for entry in manifest.get("frames", []):
        if entry.get("status") != "complete":
            continue
        frame = _load_completed_frame(entry, manifest_path.parent, generation_id)
        if frame is None:
            raise ValueError(f"Frame artifact does not match manifest generation: {entry.get('fused_objects_json')}")
        frame["frame_ref"] = entry.get("fused_objects_json")
        yield frame


def main() -> None:
    args = build_parser().parse_args()
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    candidates_path = Path(args.candidates_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve() if args.output_json else episode_dir / "frame_fused_candidates.json"
    summary = json.loads(candidates_path.read_text(encoding="utf-8"))
    camera_params = load_camera_params(Path(args.camera_params_json).expanduser().resolve() if args.camera_params_json else None)
    rlbench_low_dim_override = Path(args.rlbench_low_dim_obs).expanduser().resolve() if args.rlbench_low_dim_obs else None
    rlbench_low_dim_path = resolve_rlbench_low_dim_path(episode_dir, rlbench_low_dim_override)
    rlbench_observations = load_rlbench_observations(episode_dir, rlbench_low_dim_override)
    cameras = parse_csv(args.cameras)
    source_frames = list(summary.get("frames", []))
    fusion_parameters = _fusion_parameters(args, rlbench_low_dim_path, bool(rlbench_observations))
    old_manifest: Mapping[str, Any] = {}
    try:
        old_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    may_resume = (
        old_manifest.get("schema_version") == FUSED_MANIFEST_SCHEMA_VERSION
        and old_manifest.get("episode_metadata", {}).get("source_candidates_json") == str(candidates_path)
        and old_manifest.get("fusion_parameters") == fusion_parameters
    )
    generation_id = str(old_manifest.get("generation_id")) if may_resume else str(uuid.uuid4())
    old_entries = {
        str(entry.get("frame_id")): entry
        for entry in old_manifest.get("frames", [])
    } if may_resume else {}

    entries: list[dict[str, Any]] = []
    seen_frame_keys: set[str] = set()
    for frame in source_frames:
        frame_id = str(frame["frame_id"])
        frame_key = _geometry_segment(frame_id)
        if frame_key in seen_frame_keys:
            raise ValueError(f"Frame ids produce duplicate filesystem key: {frame_key!r}")
        seen_frame_keys.add(frame_key)
        relative_path = (Path("frames") / frame_key / "fused_objects.json").as_posix()
        old_entry = old_entries.get(frame_id, {})
        reusable = dict(old_entry) if old_entry.get("fused_objects_json") == relative_path else {}
        entries.append({
            "frame_id": frame_id,
            "frame_index": frame.get("frame_index"),
            "fused_objects_json": relative_path,
            "object_count": int(reusable.get("object_count", 0)),
            "status": reusable.get("status", "pending"),
        })

    manifest: dict[str, Any] = {
        "schema_version": FUSED_MANIFEST_SCHEMA_VERSION,
        "generation_id": generation_id,
        "episode_metadata": {
            "episode_dir": str(episode_dir),
            "source_candidates_json": str(candidates_path),
            "instruction": summary.get("instruction"),
            "role_spec": summary.get("role_spec"),
        },
        "fusion_parameters": fusion_parameters,
        "frame_order": [entry["frame_id"] for entry in entries],
        "frames": entries,
    }
    atomic_json_dump(manifest, output_path)

    track_state: dict[str, Any] = {}
    completed_count = 0
    failures = 0
    for source_frame, entry in zip(source_frames, entries):
        completed = _load_completed_frame(entry, output_path.parent, generation_id)
        if completed is not None:
            completed_count += 1
            _restore_track_state(completed, track_state)
            continue
        entry.update({"status": "pending", "object_count": 0})
        atomic_json_dump(manifest, output_path)
        previous_track_state = copy.deepcopy(track_state)
        try:
            fused_frame = fuse_frame(
                source_frame, episode_dir, camera_params, rlbench_observations,
                cameras, args, track_state=track_state,
            )
            fused_frame["schema_version"] = FUSED_MANIFEST_SCHEMA_VERSION
            fused_frame["generation_id"] = generation_id
            save_frame_geometry(fused_frame, output_path)
            atomic_json_dump(fused_frame, output_path.parent / entry["fused_objects_json"])
        except Exception as exc:
            track_state.clear()
            track_state.update(previous_track_state)
            entry["status"] = "failed"
            failures += 1
            atomic_json_dump(manifest, output_path)
            print(f"[error] frame_id={entry['frame_id']}: {exc}", file=sys.stderr)
            continue
        entry.update({"status": "complete", "object_count": len(fused_frame.get("objects", []))})
        completed_count += 1
        atomic_json_dump(manifest, output_path)

    # Keep the legacy-shaped context private to the optional summary builder;
    # the persisted manifest remains deliberately lightweight.
    result = {"episode_dir": str(episode_dir), "source_candidates_json": str(candidates_path), **fusion_parameters}

    outputs = {"output_json": str(output_path), "frames": completed_count}
    if args.save_object_summary or args.object_summary_json:
        object_summary_path = (
            Path(args.object_summary_json).expanduser().resolve()
            if args.object_summary_json
            else output_path.with_name("object_summary.json")
        )
        object_summary = build_object_summary(
            iter_manifest_frames(manifest, output_path), result, summary,
            schema_version=FUSED_MANIFEST_SCHEMA_VERSION, generation_id=generation_id,
        )
        object_summary["source_fused_json"] = str(output_path)
        atomic_json_dump(object_summary, object_summary_path)
        outputs["object_summary_json"] = str(object_summary_path)

    outputs["failed_frames"] = failures
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
