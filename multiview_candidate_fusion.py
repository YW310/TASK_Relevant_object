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
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from camera_geometry import (
    RLBENCH_DEPTH_SCALE_FACTOR,
    backproject_mask,
    camera_param_from_rlbench_observation,
    cloud_observability_in_camera,
    decode_rlbench_rgb_depth,
    find_first,
    frame_index_from_frame,
    load_camera_params,
    load_rlbench_observations,
    looks_like_rlbench_packed_depth,
    normalize_extrinsics,
    normalize_intrinsics,
    observation_misc,
    read_depth,
    resolve_camera_param_for_frame,
    resolve_depth_path,
    resolve_rlbench_low_dim_path,
    resolve_rlbench_near_far,
    transform_points,
)
from common_io import atomic_json_dump, parse_optional_csv as parse_csv
from fused_candidate_io import iter_fused_frames, load_object_points
from fusion_matching import (
    _confidence,
    bbox_iou_3d,
    camera_priority_weight,
    cluster_observations,
    legacy_union_find_clusters,
    pairwise_should_merge,
    solve_min_cost_assignment,
    suppress_same_camera_duplicates,
    symmetric_percentile_nearest_distance,
    warn_near_miss_unmerged_clusters,
    weighted_cluster_centroid,
)
from fusion_types import ROLE_NAMES, Observation3D


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
    parser.add_argument(
        "--min-candidate-mask-area-pixels",
        type=int,
        default=0,
        help=(
            "Drop 2D candidates whose mask_area_pixels is below this value before "
            "depth loading/backprojection (0 disables)."
        ),
    )
    parser.add_argument(
        "--cluster-distance-m",
        type=float,
        default=0.03,
        help=(
            "Required maximum cross-view centroid distance; optional bbox and "
            "surface checks can only make this gate stricter."
        ),
    )
    parser.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional minimum 3D bbox IoU in addition to the required centroid "
            "gate. Used as the secondary geometry check when "
            "--nearest-distance-m is not set."
        ),
    )
    parser.add_argument(
        "--nearest-distance-m",
        type=float,
        default=None,
        help=(
            "Optional maximum robust symmetric point-cloud surface distance, "
            "in addition to the required centroid gate. When set, exceeding "
            "this threshold is a hard veto even if bbox IoU passes."
        ),
    )
    parser.add_argument("--max-hypothesis-diameter-m", type=float, default=0.50,
                        help="Maximum robust (1st--99th percentile) pooled point-cloud diameter after an insertion.")
    parser.add_argument("--max-size-ratio", type=float, default=4.0,
                        help="Maximum non-degenerate axis-wise 3D-box size ratio within a hypothesis.")
    parser.add_argument(
        "--same-camera-nms-mask-iou",
        type=float,
        default=0.55,
        help=(
            "Before cross-camera fusion, suppress a lower-confidence candidate from "
            "the same camera when mask IoU reaches this value and the enabled 3D "
            "centroid/size checks also pass. Set <=0 to disable the IoU cue."
        ),
    )
    parser.add_argument(
        "--same-camera-nms-containment",
        type=float,
        default=0.85,
        help=(
            "Alternative same-camera NMS cue: minimum fraction of the smaller mask "
            "covered by the larger mask. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--same-camera-nms-centroid-distance-m",
        type=float,
        default=0.02,
        help=(
            "Maximum 3D centroid distance for same-camera duplicate suppression. "
            "Set <=0 to disable same-camera NMS entirely."
        ),
    )
    parser.add_argument(
        "--same-camera-nms-max-size-ratio",
        type=float,
        default=2.5,
        help=(
            "Maximum non-degenerate axis-wise 3D bbox size ratio for same-camera "
            "duplicate suppression."
        ),
    )
    parser.add_argument(
        "--min-fused-camera-count",
        type=int,
        default=1,
        help=(
            "Minimum supporting cameras for a fused object. A lower-support object "
            "is retained when too few missing cameras could geometrically observe it. "
            "The default 1 keeps valid single-camera objects; set >1 only for strict "
            "multi-view filtering."
        ),
    )
    parser.add_argument(
        "--preferred-camera",
        default="front",
        help=(
            "Camera trusted more during cross-view hypothesis seeding and fused "
            "centroid/score calculation. Empty disables the preference."
        ),
    )
    parser.add_argument(
        "--preferred-camera-weight",
        type=float,
        default=1.5,
        help=(
            "Reliability multiplier for --preferred-camera (default: 1.5; "
            "1.0 disables weighting)."
        ),
    )
    parser.add_argument(
        "--camera-visibility-depth-tolerance-m",
        type=float,
        default=0.03,
        help=(
            "Depth tolerance used to decide whether a fused cloud should have been "
            "visible from a missing camera."
        ),
    )
    parser.add_argument(
        "--camera-visibility-min-point-fraction",
        type=float,
        default=0.05,
        help=(
            "Minimum fraction of valid projected cloud samples that must pass the "
            "depth visibility test for a missing camera to count as observable."
        ),
    )
    parser.add_argument(
        "--single-camera-keep-score",
        type=float,
        default=0.0,
        help=(
            "Optional confidence exception for clusters below --min-fused-camera-count. "
            "0 disables this exception; values such as 0.90 keep only very high-score "
            "single-view candidates."
        ),
    )
    parser.add_argument(
        "--legacy-canonical-iou",
        type=float,
        default=0.35,
        help=(
            "Mask IoU required to collapse duplicate rows in legacy, non-canonical candidate "
            "artifacts. The default tolerates shifted masks while still requiring substantial overlap."
        ),
    )
    parser.add_argument(
        "--legacy-canonical-containment",
        type=float,
        default=0.50,
        help=(
            "Smaller-mask coverage required to collapse duplicate rows in legacy, "
            "non-canonical candidate artifacts."
        ),
    )
    parser.add_argument("--legacy-union-find", action="store_true",
                        help="DEPRECATED compatibility/debug mode: use the old pairwise transitive union-find partition.")
    parser.add_argument(
        "--track-distance-m",
        type=float,
        default=0.22,
        help=(
            "Max centroid displacement (meters) between consecutive processed frames for a "
            "fused object to keep its id (e.g. 'O1') across frames. Without this, ids "
            "are re-derived from scratch every frame by sorting clusters, which can silently "
            "flip which physical object is 'O1' vs 'O2' between frames whenever their sort order "
            "changes -- increase this if --frame-interval is large and objects move a lot between "
            "selected frames, decrease it if unrelated objects are close together. The "
            "limit is per processed frame, so FRAME_INTERVAL=10 spans ten source frames."
        ),
    )
    parser.add_argument(
        "--track-max-missed-frames",
        type=int,
        default=4,
        help=(
            "Keep an unmatched object track alive for this many processed frames so a "
            "short occlusion does not force a new object ID."
        ),
    )
    parser.add_argument(
        "--track-max-size-ratio",
        type=float,
        default=4.0,
        help=(
            "Reject a temporal ID match when non-degenerate 3D bbox axes change by more "
            "than this ratio. Set <=0 to disable the tracking size gate."
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
        "--max-centroid-to-cloud-distance-m",
        type=float,
        default=0.02,
        help=(
            "Drop a fused object when its arithmetic centroid is farther than this many "
            "meters from the nearest point in its pooled point cloud. This rejects split/"
            "contaminated clouds whose centroid falls into a large empty gap. Default: "
            "0.02 m; set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--component-voxel-size-m",
        type=float,
        default=0.008,
        help=(
            "Voxel size used to find disconnected 3D point-cloud components. "
            "Set <=0 to disable component filtering."
        ),
    )
    parser.add_argument(
        "--min-largest-component-ratio",
        type=float,
        default=0.75,
        help=(
            "Reject a multi-component cloud when its largest connected component "
            "contains less than this fraction of all points."
        ),
    )
    parser.add_argument(
        "--max-secondary-component-ratio",
        type=float,
        default=0.20,
        help=(
            "Reject a multi-component cloud when its second-largest component "
            "exceeds this fraction of all points. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--min-component-centroid-gap-m",
        type=float,
        default=0.02,
        help=(
            "Minimum centroid separation between the two largest components before "
            "the cloud is treated as two merged objects."
        ),
    )
    parser.add_argument(
        "--min-component-points",
        type=int,
        default=20,
        help="Ignore disconnected components with fewer points than this.",
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


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def canonicalize_legacy_candidates(
    candidates: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.35,
    containment_threshold: float = 0.50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Defensive adapter for old one-role-per-mask candidate artifacts.

    By default, legacy rows are grouped at IoU >= .35 or smaller-mask coverage
    >= .50. This tolerates prompt-dependent mask drift while still requiring
    substantial pixel overlap, so merely touching/adjacent instances remain
    distinct. The best scoring mask is retained and role evidence is noisy-OR
    aggregated with raw audit values preserved.
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
            if iou >= iou_threshold or coverage >= containment_threshold:
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


def filter_candidates_by_mask_area(
    candidates: Sequence[Mapping[str, Any]],
    min_area_pixels: int,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Remove raw 2D candidates below a configured mask-area threshold."""
    threshold = max(0, int(min_area_pixels))
    if threshold <= 0:
        return list(candidates), []

    kept: list[Mapping[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            area = int(candidate.get("mask_area_pixels") or 0)
        except (TypeError, ValueError):
            area = 0
        if area < threshold:
            suppressed.append(
                {
                    "candidate_id": candidate.get("id"),
                    "mask_area_pixels": area,
                    "threshold_pixels": threshold,
                    "reason": "mask_area_pixels_below_threshold",
                }
            )
            continue
        kept.append(candidate)
    return kept, suppressed


def _cluster_diagnostic_identity(
    cluster: Sequence[Observation3D],
) -> dict[str, Any]:
    """Return stable candidate/observation references for human-readable diagnostics."""
    candidate_refs = [
        {
            "camera": obs.camera,
            "candidate_id": obs.candidate.get("id"),
            "observation_id": obs.observation_id,
        }
        for obs in cluster
    ]
    return {
        "candidate_refs": candidate_refs,
        "candidate_ids": [
            ref["candidate_id"]
            for ref in candidate_refs
            if ref["candidate_id"] is not None
        ],
        "observation_ids": [ref["observation_id"] for ref in candidate_refs],
    }


def filter_clusters_by_camera_support(
    clusters: Sequence[Sequence[Observation3D]],
    args: argparse.Namespace,
    camera_contexts: Mapping[str, Mapping[str, np.ndarray]],
) -> list[list[Observation3D]]:
    """Remove unsupported clusters only when enough other views could see them."""
    min_camera_count = max(
        1, int(getattr(args, "min_fused_camera_count", 1))
    )
    if min_camera_count <= 1:
        return [list(cluster) for cluster in clusters]

    depth_tolerance = float(
        getattr(args, "camera_visibility_depth_tolerance_m", 0.03)
    )
    min_visible_fraction = float(
        getattr(args, "camera_visibility_min_point_fraction", 0.05)
    )
    single_camera_keep_score = float(
        getattr(args, "single_camera_keep_score", 0.0)
    )
    support_diagnostics = getattr(args, "_camera_support_diagnostics", None)
    filtered_diagnostics = getattr(args, "_filtered_cluster_diagnostics", None)
    kept: list[list[Observation3D]] = []

    for raw_cluster in clusters:
        cluster = list(raw_cluster)
        supporting_cameras = sorted({obs.camera for obs in cluster})
        if len(supporting_cameras) >= min_camera_count:
            kept.append(cluster)
            continue

        points = np.concatenate([obs.points_world for obs in cluster], axis=0)
        checks: dict[str, Any] = {}
        observable_missing_cameras: list[str] = []
        for camera, context in camera_contexts.items():
            if camera in supporting_cameras:
                continue
            check = cloud_observability_in_camera(
                points,
                context,
                depth_tolerance,
                min_visible_fraction,
            )
            checks[camera] = check
            if check["observable"]:
                observable_missing_cameras.append(camera)

        potential_camera_count = (
            len(supporting_cameras) + len(observable_missing_cameras)
        )
        confidence = max((_confidence(obs) for obs in cluster), default=0.0)
        high_confidence_exception = (
            single_camera_keep_score > 0.0
            and confidence >= single_camera_keep_score
        )
        visibility_exception = potential_camera_count < min_camera_count
        diagnostic = {
            **_cluster_diagnostic_identity(cluster),
            "supporting_cameras": supporting_cameras,
            "supporting_camera_count": len(supporting_cameras),
            "required_camera_count": min_camera_count,
            "observable_missing_cameras": sorted(observable_missing_cameras),
            "potential_camera_count": potential_camera_count,
            "confidence": confidence,
            "single_camera_keep_score": single_camera_keep_score,
            "visibility_checks": checks,
        }
        if visibility_exception or high_confidence_exception:
            diagnostic.update(
                {
                    "action": "kept",
                    "reason": (
                        "insufficient_observable_cameras"
                        if visibility_exception
                        else "single_camera_high_confidence"
                    ),
                }
            )
            if isinstance(support_diagnostics, list):
                support_diagnostics.append(diagnostic)
            kept.append(cluster)
            continue

        diagnostic.update(
            {
                "action": "dropped",
                "reason": "min_fused_camera_count",
            }
        )
        if isinstance(support_diagnostics, list):
            support_diagnostics.append(diagnostic)
        if isinstance(filtered_diagnostics, list):
            filtered_diagnostics.append(diagnostic)
        print(
            f"[info] dropping under-supported observation cluster "
            f"(cameras {supporting_cameras}): {len(supporting_cameras)} < "
            f"--min-fused-camera-count {min_camera_count}; observable from "
            f"{sorted(observable_missing_cameras)}.",
            file=sys.stderr,
        )
    return kept


def point_cloud_component_stats(
    points: np.ndarray,
    voxel_size_m: float,
    min_component_points: int = 1,
) -> dict[str, Any]:
    """Measure disconnected 3D regions using a dependency-free voxel graph."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or voxel_size_m <= 0.0:
        return {
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "second_component_ratio": 0.0,
            "component_centroid_gap_m": None,
        }

    voxel_coords = np.floor(points / float(voxel_size_m)).astype(np.int64)
    unique_voxels, inverse, voxel_counts = np.unique(
        voxel_coords,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    voxel_sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    np.add.at(voxel_sums, inverse, points)
    voxel_index = {
        tuple(int(value) for value in coord): index
        for index, coord in enumerate(unique_voxels)
    }
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    visited: set[int] = set()
    components: list[dict[str, Any]] = []
    for start_index, start_coord in enumerate(unique_voxels):
        if start_index in visited:
            continue
        stack = [start_index]
        visited.add(start_index)
        component_count = 0
        component_sum = np.zeros((3,), dtype=np.float64)
        while stack:
            current_index = stack.pop()
            component_count += int(voxel_counts[current_index])
            component_sum += voxel_sums[current_index]
            coord = unique_voxels[current_index]
            for dx, dy, dz in neighbor_offsets:
                neighbor = (
                    int(coord[0] + dx),
                    int(coord[1] + dy),
                    int(coord[2] + dz),
                )
                neighbor_index = voxel_index.get(neighbor)
                if neighbor_index is None or neighbor_index in visited:
                    continue
                visited.add(neighbor_index)
                stack.append(neighbor_index)
        if component_count >= max(1, int(min_component_points)):
            components.append({
                "point_count": component_count,
                "centroid_world": (component_sum / component_count),
            })

    components.sort(key=lambda item: int(item["point_count"]), reverse=True)
    total_points = max(1, len(points))
    largest_ratio = (
        float(components[0]["point_count"] / total_points) if components else 0.0
    )
    second_ratio = (
        float(components[1]["point_count"] / total_points)
        if len(components) >= 2
        else 0.0
    )
    centroid_gap = (
        float(
            np.linalg.norm(
                components[0]["centroid_world"]
                - components[1]["centroid_world"]
            )
        )
        if len(components) >= 2
        else None
    )
    return {
        "component_count": len(components),
        "largest_component_ratio": largest_ratio,
        "second_component_ratio": second_ratio,
        "component_centroid_gap_m": centroid_gap,
        "component_point_counts": [
            int(component["point_count"]) for component in components
        ],
        "voxel_size_m": float(voxel_size_m),
    }


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
    max_centroid_gap = float(
        getattr(args, "max_centroid_to_cloud_distance_m", 0.0)
    )
    component_voxel_size = float(
        getattr(args, "component_voxel_size_m", 0.0)
    )
    min_largest_component_ratio = float(
        getattr(args, "min_largest_component_ratio", 0.0)
    )
    max_secondary_component_ratio = float(
        getattr(args, "max_secondary_component_ratio", 0.0)
    )
    min_component_centroid_gap = float(
        getattr(args, "min_component_centroid_gap_m", 0.0)
    )
    min_component_points = max(
        1, int(getattr(args, "min_component_points", 1))
    )
    component_filter_enabled = (
        component_voxel_size > 0.0
        and max_secondary_component_ratio > 0.0
    )
    if (
        args.min_fused_points <= 0
        and args.min_bbox_diagonal_m <= 0.0
        and max_centroid_gap <= 0.0
        and not component_filter_enabled
    ):
        return clusters
    kept = []
    diagnostics = getattr(args, "_filtered_cluster_diagnostics", None)
    for cluster in clusters:
        all_points = np.concatenate([obs.points_world for obs in cluster], axis=0)
        if args.min_fused_points > 0 and len(all_points) < args.min_fused_points:
            if isinstance(diagnostics, list):
                diagnostics.append({
                    **_cluster_diagnostic_identity(cluster),
                    "reason": "min_fused_points",
                    "cameras": sorted({obs.camera for obs in cluster}),
                    "point_count": int(len(all_points)),
                })
            print(
                f"[info] dropping small observation cluster (cameras "
                f"{sorted({obs.camera for obs in cluster})}): {len(all_points)} points < "
                f"--min-fused-points {args.min_fused_points}.",
                file=sys.stderr,
            )
            continue
        diagonal = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
        if args.min_bbox_diagonal_m > 0.0 and diagonal < args.min_bbox_diagonal_m:
            if isinstance(diagnostics, list):
                diagnostics.append({
                    **_cluster_diagnostic_identity(cluster),
                    "reason": "min_bbox_diagonal_m",
                    "cameras": sorted({obs.camera for obs in cluster}),
                    "bbox_diagonal_m": diagonal,
                })
            print(
                f"[info] dropping small observation cluster (cameras "
                f"{sorted({obs.camera for obs in cluster})}): bbox diagonal {diagonal:.3f}m < "
                f"--min-bbox-diagonal-m {args.min_bbox_diagonal_m}.",
                file=sys.stderr,
            )
            continue
        centroid = weighted_cluster_centroid(cluster, args)
        centroid_to_cloud_distance = float(
            np.linalg.norm(all_points - centroid, axis=1).min()
        )
        if (
            max_centroid_gap > 0.0
            and centroid_to_cloud_distance > max_centroid_gap
        ):
            if isinstance(diagnostics, list):
                diagnostics.append({
                    **_cluster_diagnostic_identity(cluster),
                    "reason": "max_centroid_to_cloud_distance_m",
                    "cameras": sorted({obs.camera for obs in cluster}),
                    "centroid_world": centroid.tolist(),
                    "centroid_to_cloud_distance_m": centroid_to_cloud_distance,
                    "threshold_m": max_centroid_gap,
                })
            print(
                f"[info] dropping inconsistent observation cluster (cameras "
                f"{sorted({obs.camera for obs in cluster})}): centroid-to-cloud "
                f"distance {centroid_to_cloud_distance:.3f}m > "
                f"--max-centroid-to-cloud-distance-m {max_centroid_gap:.3f}m.",
                file=sys.stderr,
            )
            continue
        component_stats = point_cloud_component_stats(
            all_points,
            component_voxel_size,
            min_component_points,
        )
        component_gap = component_stats.get("component_centroid_gap_m")
        multiple_large_components = (
            component_filter_enabled
            and int(component_stats.get("component_count", 0)) >= 2
            and float(component_stats.get("largest_component_ratio", 0.0))
            < min_largest_component_ratio
            and float(component_stats.get("second_component_ratio", 0.0))
            > max_secondary_component_ratio
            and component_gap is not None
            and float(component_gap) >= min_component_centroid_gap
        )
        if multiple_large_components:
            detail = {
                **_cluster_diagnostic_identity(cluster),
                "reason": "multiple_large_3d_components",
                "cameras": sorted({obs.camera for obs in cluster}),
                **component_stats,
                "thresholds": {
                    "min_largest_component_ratio": min_largest_component_ratio,
                    "max_secondary_component_ratio": max_secondary_component_ratio,
                    "min_component_centroid_gap_m": min_component_centroid_gap,
                    "min_component_points": min_component_points,
                },
            }
            if isinstance(diagnostics, list):
                diagnostics.append(detail)
            print(
                f"[info] dropping merged observation cluster (cameras "
                f"{detail['cameras']}): largest component "
                f"{component_stats['largest_component_ratio']:.2%}, secondary "
                f"{component_stats['second_component_ratio']:.2%}, component "
                f"gap {float(component_gap):.3f}m.",
                file=sys.stderr,
            )
            continue
        kept.append(cluster)
    return kept


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


_LIFECYCLE_REASON_MESSAGES = {
    "mask_area_pixels_below_threshold": "Candidate mask area was below the configured minimum.",
    "legacy_overlap_or_containment": "Candidate was merged into a canonical observation with overlapping evidence.",
    "missing_camera_parameters": "Camera parameters were unavailable, so the candidate could not be lifted to 3D.",
    "empty_3d_backprojection": "No valid 3D points were produced from the candidate mask and depth image.",
    "same_camera_2d_3d_nms": "Candidate was merged into a stronger duplicate from the same camera.",
    "min_fused_points": "The fused cluster contained fewer points than required.",
    "min_bbox_diagonal_m": "The fused cluster was smaller than the configured 3D extent.",
    "max_centroid_to_cloud_distance_m": "The cluster centroid was too far from its observed point cloud.",
    "multiple_large_3d_components": "The cluster contained multiple separated large 3D components.",
    "min_fused_camera_count": "Too few cameras supported the object although other cameras could observe it.",
    "not_in_fused_output": "Candidate reached 3D processing but was not present in the final fused objects.",
}


def build_candidate_lifecycle(
    raw_candidates: Sequence[Mapping[str, Any]],
    canonical_sources: Mapping[tuple[str, str], Sequence[str]],
    backprojection_records: Sequence[Mapping[str, Any]],
    mask_area_suppressed: Sequence[Mapping[str, Any]],
    canonicalization_suppressed: Sequence[Mapping[str, Any]],
    same_camera_nms_suppressed: Sequence[Mapping[str, Any]],
    filtered_clusters: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Explain the final disposition of every Stage-1 candidate in one frame."""
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in raw_candidates:
        camera = str(candidate.get("camera", ""))
        candidate_id = str(candidate.get("candidate_id", ""))
        if not camera or not candidate_id:
            continue
        entries[(camera, candidate_id)] = {
            "camera": camera,
            "candidate_id": candidate_id,
            "canonical_observation_id": candidate.get("canonical_observation_id"),
            "role": candidate.get("role"),
            "sam_score": candidate.get("sam_score"),
            "mask_area_pixels": candidate.get("mask_area_pixels"),
            "final_status": "unresolved",
            "fused_object_id": None,
            "last_successful_stage": "candidate_input",
            "events": [
                {"stage": "candidate_input", "status": "accepted"}
            ],
        }

    def targets(camera: Any, candidate_id: Any) -> list[dict[str, Any]]:
        camera_text = str(camera or "")
        candidate_text = str(candidate_id or "")
        source_ids = canonical_sources.get(
            (camera_text, candidate_text), (candidate_text,)
        )
        return [
            entries[(camera_text, str(source_id))]
            for source_id in source_ids
            if (camera_text, str(source_id)) in entries
        ]

    def add_event(
        entry: dict[str, Any],
        stage: str,
        status: str,
        reason_code: str | None = None,
        **details: Any,
    ) -> None:
        event: dict[str, Any] = {"stage": stage, "status": status}
        if reason_code:
            event["reason_code"] = reason_code
            event["reason_message"] = _LIFECYCLE_REASON_MESSAGES.get(
                reason_code, reason_code.replace("_", " ").capitalize() + "."
            )
        event.update({key: value for key, value in details.items() if value is not None})
        entry["events"].append(event)

    for (camera, canonical_id), source_ids in canonical_sources.items():
        for source_id in source_ids:
            entry = entries.get((camera, str(source_id)))
            if entry is None:
                continue
            entry["canonical_observation_id"] = canonical_id
            if str(source_id) != canonical_id:
                entry["final_status"] = "merged"
                entry["replacement_candidate_id"] = canonical_id
                entry["last_successful_stage"] = "canonicalization"
                add_event(
                    entry,
                    "canonicalization",
                    "merged",
                    "legacy_overlap_or_containment",
                    replacement_candidate_id=canonical_id,
                )

    for detail in mask_area_suppressed:
        for entry in targets(detail.get("camera"), detail.get("candidate_id")):
            entry["final_status"] = "dropped"
            add_event(entry, "mask_area_filter", "dropped", str(detail.get("reason")))

    for detail in canonicalization_suppressed:
        for entry in targets(detail.get("camera"), detail.get("candidate_id")):
            entry["final_status"] = "merged"
            add_event(entry, "canonicalization", "merged", str(detail.get("reason")))

    for detail in backprojection_records:
        for entry in targets(detail.get("camera"), detail.get("candidate_id")):
            if detail.get("status") == "accepted":
                entry["last_successful_stage"] = "backprojection"
                add_event(
                    entry,
                    "backprojection",
                    "accepted",
                    observation_id=detail.get("observation_id"),
                    point_count=detail.get("point_count"),
                )
            else:
                entry["final_status"] = "dropped"
                add_event(
                    entry,
                    "backprojection",
                    "dropped",
                    str(detail.get("reason")),
                )

    replacement_map: dict[tuple[str, str], str] = {}
    for detail in same_camera_nms_suppressed:
        camera = str(detail.get("camera", ""))
        suppressed_id = str(detail.get("suppressed_candidate_id", ""))
        kept_id = str(detail.get("kept_candidate_id", ""))
        replacement_map[(camera, suppressed_id)] = kept_id
        for entry in targets(camera, suppressed_id):
            entry["final_status"] = "merged"
            entry["replacement_candidate_id"] = kept_id
            entry["last_successful_stage"] = "same_camera_nms"
            add_event(
                entry,
                "same_camera_nms",
                "merged",
                str(detail.get("reason")),
                replacement_candidate_id=kept_id,
            )

    for detail in filtered_clusters:
        reason = str(detail.get("reason", "not_in_fused_output"))
        for ref in detail.get("candidate_refs", []):
            for entry in targets(ref.get("camera"), ref.get("candidate_id")):
                entry["final_status"] = "dropped"
                add_event(entry, "cluster_filter", "dropped", reason)

    fused_by_candidate: dict[tuple[str, str], str] = {}
    for obj in objects:
        object_id = str(obj.get("id"))
        for observation in obj.get("observations", []):
            key = (
                str(observation.get("camera", "")),
                str(observation.get("candidate_id", "")),
            )
            fused_by_candidate[key] = object_id
            for entry in targets(*key):
                if entry["final_status"] == "unresolved":
                    entry["final_status"] = "fused"
                entry["fused_object_id"] = object_id
                entry["last_successful_stage"] = "fused_output"
                add_event(entry, "fused_output", "accepted", fused_object_id=object_id)

    for (camera, suppressed_id), kept_id in replacement_map.items():
        object_id = fused_by_candidate.get((camera, kept_id))
        if object_id is None:
            continue
        for entry in targets(camera, suppressed_id):
            entry["fused_object_id"] = object_id

    for entry in entries.values():
        if entry["final_status"] == "unresolved":
            add_event(
                entry,
                "fused_output",
                "unresolved",
                "not_in_fused_output",
            )

    lifecycle = sorted(
        entries.values(), key=lambda item: (item["camera"], item["candidate_id"])
    )
    summary = {
        "input_candidate_count": len(lifecycle),
        "directly_fused_candidate_count": sum(
            item["final_status"] == "fused" for item in lifecycle
        ),
        "merged_candidate_count": sum(
            item["final_status"] == "merged" for item in lifecycle
        ),
        "dropped_candidate_count": sum(
            item["final_status"] == "dropped" for item in lifecycle
        ),
        "unresolved_candidate_count": sum(
            item["final_status"] == "unresolved" for item in lifecycle
        ),
    }
    return lifecycle, summary


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


def compact_role_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the role values consumed by filters and Qwen prompts."""
    return {
        role: {
            key: value.get(key)
            for key in ("probability", "score_mass")
            if value.get(key) is not None
        }
        for role, value in evidence.items()
        if isinstance(value, Mapping)
    }


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
                    "centroid_world": obj["centroid_world"],
                    "bbox3d_world": obj["bbox3d_world"],
                    "point_count": int(len(load_object_points(frame, object_id))),
                    "visible_camera": obj.get("visible_camera", []),
                    "camera_count": len(obj.get("visible_camera", [])),
                    "mask_area": int(obj.get("mask_area", 0)),
                    "sam_score": float(obj.get("sam_score", 0.0)),
                    "role_evidence": obj.get("role_evidence", {}),
                }
            )

        objects = list(frame.get("objects", []))
        candidates = []
        for obj in objects:
            candidates.append(_summary_object_record(frame, obj))
        frame_decision_inputs.append({
            "frame_id": frame_id, "frame_index": frame_index,
            "frame_ref": frame.get("frame_ref"),
            "candidate_objects": candidates,
            # Pairwise geometry is deterministic and O(objects^2). Stage 4 derives
            # it from the candidate centroids/bboxes only for its active window.
            "pairwise_relations": [],
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
                "role_evidence": compact_role_evidence(
                    aggregate_summary_role_evidence(frames_sorted)
                ),
                "first_frame_id": frames_sorted[0]["frame_id"],
                "last_frame_id": frames_sorted[-1]["frame_id"],
                "first_frame_index": frames_sorted[0]["frame_index"],
                "last_frame_index": frames_sorted[-1]["frame_index"],
                "lifespan_frames": len(frames_sorted),
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
            }
        )

    return {
        "schema_version": schema_version,
        "artifact_type": "object_track_summary",
        "storage_layout": "compact_v1",
        "generation_id": generation_id,
        "coordinate_frame": "world",
        "units": {"distance": "meters", "mask_area": "pixels"},
        "summary": {
            "frame_count": len(frame_decision_inputs),
            "object_track_count": len(object_tracks),
            "visible_object_sample_count": sum(
                int(track.get("lifespan_frames", 0)) for track in object_tracks
            ),
        },
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
            "track_max_missed_frames": result.get("track_max_missed_frames"),
            "track_max_size_ratio": result.get("track_max_size_ratio"),
            "min_fused_points": result.get("min_fused_points"),
            "min_bbox_diagonal_m": result.get("min_bbox_diagonal_m"),
            "max_hypothesis_diameter_m": result.get("max_hypothesis_diameter_m"),
            "max_size_ratio": result.get("max_size_ratio"),
        },
        "object_tracks": object_tracks,
        "frame_decision_inputs": frame_decision_inputs,
    }


def _summary_object_record(frame: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    """Strip a frame object to decision metadata (never embedded geometry)."""
    return {
        "object_id": obj.get("id"),
        "role_evidence": compact_role_evidence(obj.get("role_evidence", {})),
        "centroid_world": obj.get("centroid_world"), "bbox3d_world": obj.get("bbox3d_world"),
        "primary_camera": obj.get("primary_camera"),
        "visible_camera": obj.get("visible_camera", []),
        "camera_count": len(obj.get("visible_camera", [])),
        "point_count": int(len(load_object_points(frame, obj.get("id")))),
        "mask_area": obj.get("mask_area"), "sam_score": obj.get("sam_score"),
        "observation_count": len(obj.get("observations", [])),
        "observations": [{key: obs.get(key) for key in (
            "camera", "candidate_id", "mask_path", "crop_path", "masked_crop_path",
            "mask_area", "sam_score", "mask_bbox_xyxy"
        ) if obs.get(key) is not None} for obs in obj.get("observations", [])],
    }


def assign_object_ids(
    clusters: list[list[Observation3D]],
    track_state: dict[str, Any],
    args: argparse.Namespace,
    frame_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Track physical objects globally, retaining IDs across short occlusions."""
    prev_tracks: list[dict[str, Any]] = list(track_state.get("tracks", []))
    next_object_index = int(track_state.get("next_object_index", 0))
    cluster_points = [np.concatenate([o.points_world for o in cluster]) for cluster in clusters]
    centroids = [weighted_cluster_centroid(cluster, args) for cluster in clusters]
    bbox_sizes = [points.max(axis=0) - points.min(axis=0) for points in cluster_points]
    max_missed_frames = max(0, int(getattr(args, "track_max_missed_frames", 0)))
    max_track_size_ratio = float(getattr(args, "track_max_size_ratio", 0.0))
    assigned: dict[int, int] = {}
    assigned_previous: dict[int, int] = {}
    if clusters and prev_tracks:
        nc, np_ = len(clusters), len(prev_tracks)
        size = nc + np_
        # Costs are normalized so 1.0 means "prefer a new/unmatched track".
        abstain = 1.0 + 1e-6
        cost = np.full((size, size), abstain * 1000.0 + 1e6)
        for ci, centroid in enumerate(centroids):
            for pi, track in enumerate(prev_tracks):
                missed_frames = max(0, int(track.get("missed_frames", 0)))
                distance_limit = float(args.track_distance_m) * (missed_frames + 1)
                distance = float(np.linalg.norm(centroid - np.asarray(track["centroid"])))
                if distance_limit < 0 or distance > distance_limit:
                    continue

                size_ratio = 1.0
                previous_size_value = track.get("bbox_size")
                if previous_size_value is not None:
                    previous_size = np.asarray(previous_size_value, dtype=np.float64)
                    current_size = np.asarray(bbox_sizes[ci], dtype=np.float64)
                    valid_axes = (previous_size > 1e-6) & (current_size > 1e-6)
                    if np.any(valid_axes):
                        size_ratio = float(
                            np.max(
                                np.maximum(
                                    previous_size[valid_axes] / current_size[valid_axes],
                                    current_size[valid_axes] / previous_size[valid_axes],
                                )
                            )
                        )
                    if max_track_size_ratio > 0 and size_ratio > max_track_size_ratio:
                        continue

                normalized_distance = distance / max(distance_limit, 1e-9)
                size_penalty = 0.05 * float(np.log(max(1.0, size_ratio)))
                missed_penalty = 0.03 * missed_frames
                cost[ci, pi] = normalized_distance + size_penalty + missed_penalty
            cost[ci, np_ + ci] = abstain
        for pi in range(np_):
            cost[nc + pi, pi] = abstain
        cost[nc:, np_:] = 0.0
        for row, col in solve_min_cost_assignment(cost):
            if row < nc and col < np_ and cost[row, col] < abstain:
                assigned[row] = int(prev_tracks[col]["index"])
                assigned_previous[row] = col

    objects, tracks = [], []
    for ci, cluster in enumerate(clusters):
        all_points = cluster_points[ci]
        centroid = centroids[ci]
        if ci in assigned:
            index = assigned[ci]
        else:
            next_object_index += 1
            index = next_object_index
        centroid_to_cloud_distance = float(
            np.linalg.norm(all_points - centroid, axis=1).min()
        )
        component_stats = point_cloud_component_stats(
            all_points,
            float(getattr(args, "component_voxel_size_m", 0.0)),
            max(1, int(getattr(args, "min_component_points", 1))),
        )
        ranked_observations = sorted(
            cluster,
            key=lambda observation: (
                -camera_priority_weight(observation.camera, args)
                * _confidence(observation),
                observation.observation_id,
            ),
        )
        score_weights = np.asarray(
            [camera_priority_weight(observation.camera, args) for observation in cluster],
            dtype=np.float64,
        )
        sam_scores = np.asarray(
            [float(observation.candidate.get("score", 0.0)) for observation in cluster],
            dtype=np.float64,
        )
        objects.append({
            "id": f"O{index}",
            "role_evidence": aggregate_role_evidence(cluster, frame_id),
            "_points_world": all_points,
            "centroid_world": centroid.tolist(),
            "centroid_to_cloud_distance_m": centroid_to_cloud_distance,
            "point_cloud_components": component_stats,
            "bbox3d_world": np.stack([all_points.min(axis=0), all_points.max(axis=0)]).tolist(),
            "primary_camera": ranked_observations[0].camera,
            "visible_camera": list(dict.fromkeys(o.camera for o in ranked_observations)),
            "mask_area": int(sum(int(o.candidate.get("mask_area_pixels", 0)) for o in cluster)),
            "sam_score": float(np.average(sam_scores, weights=score_weights)),
            "observations": [observation_to_json(o) for o in ranked_observations],
        })
        tracks.append(
            {
                "index": index,
                "centroid": centroid.tolist(),
                "bbox_size": bbox_sizes[ci].tolist(),
                "missed_frames": 0,
            }
        )

    matched_previous_indices = set(assigned_previous.values())
    for previous_index, previous_track in enumerate(prev_tracks):
        if previous_index in matched_previous_indices:
            continue
        missed_frames = max(0, int(previous_track.get("missed_frames", 0))) + 1
        if missed_frames > max_missed_frames:
            continue
        dormant_track = dict(previous_track)
        dormant_track["missed_frames"] = missed_frames
        tracks.append(dormant_track)

    objects.sort(key=lambda obj: int(str(obj["id"])[1:]))
    tracks.sort(key=lambda track: int(track["index"]))
    return objects, {"tracks": tracks, "next_object_index": next_object_index}


def _geometry_segment(value: Any) -> str:
    """Make one stable, filesystem/key-safe geometry path segment."""
    text = str(value) if value is not None else "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "unknown"


def _frame_artifact_key(frame_id: Any, frame_index: Any) -> str:
    """Return a stable, sortable directory key such as ``000000_0``."""
    frame_id_segment = _geometry_segment(frame_id)
    try:
        index_segment = f"{int(frame_index):06d}"
    except (TypeError, ValueError):
        index_segment = _geometry_segment(frame_index)
    return f"{index_segment}_{frame_id_segment}"


def save_frame_geometry(frame: Mapping[str, Any], output_path: Path) -> None:
    """Move transient point arrays from a fused frame into a compressed NPZ."""
    frame_key = _frame_artifact_key(frame.get("frame_id", "frame"), frame.get("frame_index", "unknown"))
    relative_path = Path("frames") / frame_key / "fused_geometry.npz"
    archive_path = output_path.parent / relative_path
    # Keep identity metadata inside the archive itself.  This makes the NPZ a
    # self-describing artifact rather than an unversioned bag of arrays, and
    # lets readers reject a stale archive copied from another frame/run.
    arrays: dict[str, np.ndarray] = {
        "__schema_version__": np.asarray(FUSED_MANIFEST_SCHEMA_VERSION, dtype=np.int64),
        "__generation_id__": np.asarray(str(frame.get("generation_id", ""))),
        "__frame_id__": np.asarray(str(frame.get("frame_id", ""))),
    }

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
    # Do not leave a half-written/corrupt archive if the process is killed
    # while replacing an existing frame output.
    temporary_path = archive_path.with_name(archive_path.name + ".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(archive_path)


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
    raw_candidate_records: list[dict[str, Any]] = []
    canonical_sources: dict[tuple[str, str], tuple[str, ...]] = {}
    backprojection_records: list[dict[str, Any]] = []
    mask_area_suppressed: list[dict[str, Any]] = []
    canonicalization_suppressed: list[dict[str, Any]] = []
    same_camera_nms_suppressed: list[dict[str, Any]] = []
    camera_contexts: dict[str, dict[str, np.ndarray]] = {}
    frame_id = str(frame["frame_id"])
    frame_index = frame_index_from_frame(frame)
    for camera, view in frame.get("views", {}).items():
        if cameras is not None and camera not in cameras:
            continue
        data = json.loads(Path(view["candidates_json"]).read_text(encoding="utf-8"))
        raw_candidates = list(data.get("candidates", []))
        for candidate in raw_candidates:
            raw_candidate_records.append(
                {
                    "camera": camera,
                    "candidate_id": candidate.get("id"),
                    "canonical_observation_id": candidate.get(
                        "canonical_observation_id"
                    ),
                    "role": candidate.get("role"),
                    "sam_score": candidate.get("score"),
                    "mask_area_pixels": candidate.get("mask_area_pixels"),
                }
            )
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
            backprojection_records.extend(
                {
                    "camera": camera,
                    "candidate_id": candidate.get("id"),
                    "status": "dropped",
                    "reason": "missing_camera_parameters",
                }
                for candidate in raw_candidates
            )
            print(
                f"[warn] frame_id={frame_id} camera={camera}: no camera intrinsics/extrinsics found; "
                "skipping 3D fusion for this view (visual-only matching not yet implemented).",
                file=sys.stderr,
            )
            continue
        depth_near_far = resolve_rlbench_near_far(camera, frame_index, rlbench_observations)
        near, far = depth_near_far if depth_near_far is not None else (None, None)
        depth = read_depth(resolve_depth_path(episode_dir, camera, frame_id), args.depth_scale, near=near, far=far, mode=args.depth_mode)
        camera_contexts[camera] = {
            "intrinsics": params["intrinsics"],
            "extrinsics": params["extrinsics"],
            "depth": depth,
        }
        area_filtered_candidates, area_suppressed = filter_candidates_by_mask_area(
            raw_candidates,
            args.min_candidate_mask_area_pixels,
        )
        mask_area_suppressed.extend({"camera": camera, **item} for item in area_suppressed)
        if area_suppressed:
            print(
                f"[info] frame_id={frame_id} camera={camera}: mask-area filter suppressed "
                f"{len(area_suppressed)} candidates below "
                f"{args.min_candidate_mask_area_pixels} pixels",
                file=sys.stderr,
            )
        canonical_candidates, legacy_suppressed = canonicalize_legacy_candidates(
            area_filtered_candidates,
            iou_threshold=args.legacy_canonical_iou,
            containment_threshold=args.legacy_canonical_containment,
        )
        canonicalization_suppressed.extend({"camera": camera, **item} for item in legacy_suppressed)
        raw_candidate_ids = {
            str(candidate.get("id"))
            for candidate in raw_candidates
            if candidate.get("id") is not None
        }
        for candidate in canonical_candidates:
            canonical_id = str(candidate.get("id"))
            source_ids = tuple(
                dict.fromkeys(
                    str(item.get("original_candidate_id"))
                    for item in candidate.get("prompt_provenance", [])
                    if item.get("original_candidate_id") is not None
                    and str(item.get("original_candidate_id")) in raw_candidate_ids
                )
            )
            if not source_ids and canonical_id in raw_candidate_ids:
                source_ids = (canonical_id,)
            canonical_sources[(camera, canonical_id)] = source_ids
        if legacy_suppressed:
            print(f"[info] frame_id={frame_id} camera={camera}: legacy canonicalization suppressed "
                  f"{len(legacy_suppressed)} duplicate candidates", file=sys.stderr)
        camera_observations: list[tuple[Observation3D, np.ndarray]] = []
        for cand in canonical_candidates:
            mask = load_mask(Path(cand["mask_path"]))
            points_cam = backproject_mask(depth, mask, params["intrinsics"], args.max_points_per_candidate)
            points_world = transform_points(points_cam, params["extrinsics"])
            if len(points_world) == 0:
                backprojection_records.append(
                    {
                        "camera": camera,
                        "candidate_id": cand.get("id"),
                        "status": "dropped",
                        "reason": "empty_3d_backprojection",
                    }
                )
                continue
            centroid = points_world.mean(axis=0)
            bbox = np.stack([points_world.min(axis=0), points_world.max(axis=0)])
            observation = candidate_to_observation(
                cand,
                camera,
                frame_id,
                points_world,
                centroid,
                bbox,
            )
            backprojection_records.append(
                {
                    "camera": camera,
                    "candidate_id": cand.get("id"),
                    "observation_id": observation.observation_id,
                    "status": "accepted",
                    "point_count": int(len(points_world)),
                }
            )
            camera_observations.append(
                (
                    observation,
                    mask,
                )
            )
        kept_camera_observations, nms_diagnostics = suppress_same_camera_duplicates(
            camera_observations,
            args,
        )
        observations.extend(kept_camera_observations)
        same_camera_nms_suppressed.extend(nms_diagnostics)
        if nms_diagnostics:
            print(
                f"[info] frame_id={frame_id} camera={camera}: same-camera 2D/3D "
                f"NMS suppressed {len(nms_diagnostics)} duplicate candidates",
                file=sys.stderr,
            )

    same_camera_diagnostics: list[dict[str, Any]] = []
    filtered_cluster_diagnostics: list[dict[str, Any]] = []
    camera_support_diagnostics: list[dict[str, Any]] = []
    args._same_camera_diagnostics = same_camera_diagnostics
    args._filtered_cluster_diagnostics = filtered_cluster_diagnostics
    args._camera_support_diagnostics = camera_support_diagnostics
    clusters = cluster_observations(observations, args)
    clusters = filter_small_clusters(clusters, args)
    clusters = filter_clusters_by_camera_support(clusters, args, camera_contexts)
    objects, updated_track_state = assign_object_ids(clusters, track_state or {}, args, frame_id)
    candidate_lifecycle, lifecycle_summary = build_candidate_lifecycle(
        raw_candidate_records,
        canonical_sources,
        backprojection_records,
        mask_area_suppressed,
        canonicalization_suppressed,
        same_camera_nms_suppressed,
        filtered_cluster_diagnostics,
        objects,
    )
    if track_state is not None:
        track_state.clear()
        track_state.update(updated_track_state)
    return {"frame_index": frame.get("frame_index"), "frame_id": frame_id,
            "summary": {**lifecycle_summary, "fused_object_count": len(objects)},
            "objects": objects, "candidate_lifecycle": candidate_lifecycle,
            "diagnostics": {"mask_area_suppressed": mask_area_suppressed,
                            "canonicalization_suppressed": canonicalization_suppressed,
                            "same_camera_nms_suppressed": same_camera_nms_suppressed,
                            "backprojection": backprojection_records,
                            "attempted_same_camera_cluster_insertions": same_camera_diagnostics,
                            "camera_support": camera_support_diagnostics,
                            "filtered_clusters": filtered_cluster_diagnostics,
                            "tracking_state": updated_track_state}}


FUSED_MANIFEST_SCHEMA_VERSION = 3


def _write_fusion_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Refresh the human-readable run summary before atomically writing it."""
    entries = list(manifest.get("frames", []))
    manifest["summary"] = {
        "frame_count": len(entries),
        "completed_frame_count": sum(
            entry.get("status") == "complete" for entry in entries
        ),
        "failed_frame_count": sum(
            entry.get("status") == "failed" for entry in entries
        ),
        "pending_frame_count": sum(
            entry.get("status") == "pending" for entry in entries
        ),
        "fused_object_count": sum(
            int(entry.get("object_count", 0))
            for entry in entries
            if entry.get("status") == "complete"
        ),
        "input_candidate_count": sum(
            int(entry.get("candidate_count", 0))
            for entry in entries
            if entry.get("status") == "complete"
        ),
        "dropped_candidate_count": sum(
            int(entry.get("dropped_candidate_count", 0))
            for entry in entries
            if entry.get("status") == "complete"
        ),
    }
    atomic_json_dump(manifest, output_path)


def _fusion_parameters(args: argparse.Namespace, rlbench_low_dim_path: Path, has_rlbench_observations: bool) -> dict[str, Any]:
    """Return the parameters that determine the per-frame fusion artifacts."""
    return {
        "cluster_distance_m": args.cluster_distance_m,
        "bbox_iou_threshold": args.bbox_iou_threshold,
        "nearest_distance_m": args.nearest_distance_m,
        "track_distance_m": args.track_distance_m,
        "track_max_missed_frames": args.track_max_missed_frames,
        "track_max_size_ratio": args.track_max_size_ratio,
        "min_fused_points": args.min_fused_points,
        "min_bbox_diagonal_m": args.min_bbox_diagonal_m,
        "max_centroid_to_cloud_distance_m": args.max_centroid_to_cloud_distance_m,
        "component_voxel_size_m": args.component_voxel_size_m,
        "min_largest_component_ratio": args.min_largest_component_ratio,
        "max_secondary_component_ratio": args.max_secondary_component_ratio,
        "min_component_centroid_gap_m": args.min_component_centroid_gap_m,
        "min_component_points": args.min_component_points,
        "max_hypothesis_diameter_m": args.max_hypothesis_diameter_m,
        "max_size_ratio": args.max_size_ratio,
        "same_camera_nms_mask_iou": args.same_camera_nms_mask_iou,
        "same_camera_nms_containment": args.same_camera_nms_containment,
        "same_camera_nms_centroid_distance_m": args.same_camera_nms_centroid_distance_m,
        "same_camera_nms_max_size_ratio": args.same_camera_nms_max_size_ratio,
        "min_fused_camera_count": args.min_fused_camera_count,
        "preferred_camera": args.preferred_camera,
        "preferred_camera_weight": args.preferred_camera_weight,
        "camera_visibility_depth_tolerance_m": args.camera_visibility_depth_tolerance_m,
        "camera_visibility_min_point_fraction": args.camera_visibility_min_point_fraction,
        "single_camera_keep_score": args.single_camera_keep_score,
        "legacy_canonical_iou": args.legacy_canonical_iou,
        "legacy_canonical_containment": args.legacy_canonical_containment,
        "max_points_per_candidate": args.max_points_per_candidate,
        "min_candidate_mask_area_pixels": args.min_candidate_mask_area_pixels,
        "depth_mode": args.depth_mode,
        "depth_scale": args.depth_scale,
        "cameras": list(parse_csv(args.cameras) or []),
        "fusion_algorithm": "legacy_pairwise_union_find" if args.legacy_union_find else "same_camera_nms_anchor_gated_assignment_v2",
        "rlbench_low_dim_obs": str(rlbench_low_dim_path) if has_rlbench_observations else None,
        "invert_rlbench_extrinsics": bool(args.invert_rlbench_extrinsics),
    }


def _restore_track_state(frame: Mapping[str, Any], track_state: dict[str, Any]) -> None:
    """Advance tracking from a completed frame when resuming a partial run."""
    persisted_state = frame.get("_resume_tracking_state")
    if not isinstance(persisted_state, Mapping):
        persisted_state = frame.get("diagnostics", {}).get("tracking_state")
    if isinstance(persisted_state, Mapping) and isinstance(persisted_state.get("tracks"), list):
        track_state.clear()
        track_state.update(copy.deepcopy(dict(persisted_state)))
        return

    tracks = []
    next_object_index = int(track_state.get("next_object_index", 0))
    for obj in frame.get("objects", []):
        object_id = str(obj.get("id", ""))
        if not (object_id.startswith("O") and object_id[1:].isdigit()):
            continue
        index = int(object_id[1:])
        next_object_index = max(next_object_index, index)
        bbox = np.asarray(obj.get("bbox3d_world", []), dtype=np.float64)
        bbox_size = (bbox[1] - bbox[0]).tolist() if bbox.shape == (2, 3) else None
        tracks.append(
            {
                "index": index,
                "centroid": obj["centroid_world"],
                "bbox_size": bbox_size,
                "missed_frames": 0,
            }
        )
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
    debug_ref = entry.get("fusion_debug_json") or frame.get("diagnostics_ref")
    if debug_ref:
        try:
            debug_path = Path(str(debug_ref)).expanduser()
            if not debug_path.is_absolute():
                debug_path = output_dir / debug_path
                if not debug_path.is_file():
                    debug_path = path.parent / Path(str(debug_ref)).name
            debug = json.loads(debug_path.read_text(encoding="utf-8"))
            if (
                debug.get("generation_id") == generation_id
                and str(debug.get("frame_id")) == str(entry.get("frame_id"))
            ):
                frame["_resume_tracking_state"] = debug.get("diagnostics", {}).get(
                    "tracking_state"
                )
        except (OSError, ValueError, TypeError):
            pass
    return frame


def compact_candidate_outcomes(
    lifecycle: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one concise, human-readable final outcome per Stage-1 candidate."""
    outcomes: list[dict[str, Any]] = []
    for entry in lifecycle:
        outcome: dict[str, Any] = {
            "camera": entry.get("camera"),
            "candidate_id": entry.get("candidate_id"),
            "status": entry.get("final_status"),
        }
        optional = {
            "object_id": entry.get("fused_object_id"),
            "replacement_candidate_id": entry.get("replacement_candidate_id"),
            "last_successful_stage": entry.get("last_successful_stage"),
        }
        reason_code = next(
            (
                event.get("reason_code")
                for event in reversed(entry.get("events", []))
                if event.get("reason_code")
            ),
            None,
        )
        optional["reason_code"] = reason_code
        outcome.update({key: value for key, value in optional.items() if value is not None})
        outcomes.append(outcome)
    return outcomes


def split_fused_frame_artifacts(
    frame: Mapping[str, Any], debug_ref: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate downstream results from verbose fusion diagnostics."""
    main_frame = dict(frame)
    main_frame.pop("_resume_tracking_state", None)
    lifecycle = list(main_frame.pop("candidate_lifecycle", []))
    diagnostics = dict(main_frame.pop("diagnostics", {}))
    main_frame["candidate_outcomes"] = compact_candidate_outcomes(lifecycle)
    main_frame["diagnostics_ref"] = debug_ref
    debug_frame = {
        "schema_version": frame.get("schema_version"),
        "artifact_type": "frame_fusion_debug",
        "generation_id": frame.get("generation_id"),
        "frame_id": frame.get("frame_id"),
        "frame_index": frame.get("frame_index"),
        "source_fused_objects_json": "fused_objects.json",
        "candidate_lifecycle": lifecycle,
        "diagnostics": diagnostics,
    }
    return main_frame, debug_frame


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
        frame_key = _frame_artifact_key(frame_id, frame.get("frame_index", "unknown"))
        if frame_key in seen_frame_keys:
            raise ValueError(f"Frame ids produce duplicate filesystem key: {frame_key!r}")
        seen_frame_keys.add(frame_key)
        relative_path = (Path("frames") / frame_key / "fused_objects.json").as_posix()
        debug_relative_path = (Path("frames") / frame_key / "fusion_debug.json").as_posix()
        old_entry = old_entries.get(frame_id, {})
        reusable = dict(old_entry) if old_entry.get("fused_objects_json") == relative_path else {}
        entries.append({
            "frame_id": frame_id,
            "frame_index": frame.get("frame_index"),
            "fused_objects_json": relative_path,
            "fusion_debug_json": debug_relative_path,
            "object_count": int(reusable.get("object_count", 0)),
            "candidate_count": int(reusable.get("candidate_count", 0)),
            "dropped_candidate_count": int(
                reusable.get("dropped_candidate_count", 0)
            ),
            "status": reusable.get("status", "pending"),
        })

    manifest: dict[str, Any] = {
        "schema_version": FUSED_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "fused_episode_manifest",
        "generation_id": generation_id,
        "coordinate_frame": "world",
        "units": {"distance": "meters", "mask_area": "pixels"},
        "episode_metadata": {
            "episode_dir": str(episode_dir),
            "source_candidates_json": str(candidates_path),
            "instruction": summary.get("instruction"),
            "role_spec": summary.get("role_spec"),
        },
        "fusion_parameters": fusion_parameters,
        "frames": entries,
    }
    _write_fusion_manifest(manifest, output_path)

    track_state: dict[str, Any] = {}
    completed_count = 0
    failures = 0
    for source_frame, entry in zip(source_frames, entries):
        completed = _load_completed_frame(entry, output_path.parent, generation_id)
        if completed is not None:
            completed_count += 1
            _restore_track_state(completed, track_state)
            # Transparently migrate resumable v3 artifacts written before the
            # debug-sidecar layout, so the refreshed manifest never points to a
            # missing fusion_debug.json.
            if "diagnostics" in completed or "candidate_lifecycle" in completed:
                completed, fusion_debug = split_fused_frame_artifacts(
                    completed, "fusion_debug.json"
                )
                atomic_json_dump(
                    fusion_debug, output_path.parent / entry["fusion_debug_json"]
                )
                atomic_json_dump(
                    completed, output_path.parent / entry["fused_objects_json"]
                )
            continue
        entry.update(
            {
                "status": "pending",
                "object_count": 0,
                "candidate_count": 0,
                "dropped_candidate_count": 0,
            }
        )
        _write_fusion_manifest(manifest, output_path)
        previous_track_state = copy.deepcopy(track_state)
        try:
            fused_frame = fuse_frame(
                source_frame, episode_dir, camera_params, rlbench_observations,
                cameras, args, track_state=track_state,
            )
            fused_frame = {
                "schema_version": FUSED_MANIFEST_SCHEMA_VERSION,
                "artifact_type": "frame_fused_objects",
                "generation_id": generation_id,
                "coordinate_frame": "world",
                "units": {"distance": "meters", "mask_area": "pixels"},
                **fused_frame,
            }
            save_frame_geometry(fused_frame, output_path)
            fused_frame, fusion_debug = split_fused_frame_artifacts(
                fused_frame, "fusion_debug.json"
            )
            atomic_json_dump(
                fusion_debug, output_path.parent / entry["fusion_debug_json"]
            )
            atomic_json_dump(fused_frame, output_path.parent / entry["fused_objects_json"])
        except Exception as exc:
            track_state.clear()
            track_state.update(previous_track_state)
            entry["status"] = "failed"
            failures += 1
            _write_fusion_manifest(manifest, output_path)
            print(f"[error] frame_id={entry['frame_id']}: {exc}", file=sys.stderr)
            continue
        frame_summary = fused_frame.get("summary", {})
        entry.update(
            {
                "status": "complete",
                "object_count": len(fused_frame.get("objects", [])),
                "candidate_count": int(
                    frame_summary.get("input_candidate_count", 0)
                ),
                "dropped_candidate_count": int(
                    frame_summary.get("dropped_candidate_count", 0)
                ),
            }
        )
        completed_count += 1
        _write_fusion_manifest(manifest, output_path)

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
            iter_fused_frames(output_path), result, summary,
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
