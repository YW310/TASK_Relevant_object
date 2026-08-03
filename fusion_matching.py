"""Pure matching and assignment algorithms for multi-view 3D fusion."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from fusion_types import Observation3D
from mask_geometry import bbox_max_axis_size_ratio, mask_overlap_metrics


def bbox_iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    mins = np.maximum(a[0], b[0])
    maxs = np.minimum(a[1], b[1])
    inter_dims = np.maximum(0.0, maxs - mins)
    inter = float(np.prod(inter_dims))
    vol_a = float(np.prod(np.maximum(0.0, a[1] - a[0])))
    vol_b = float(np.prod(np.maximum(0.0, b[1] - b[0])))
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def symmetric_percentile_nearest_distance(
    a: np.ndarray,
    b: np.ndarray,
    percentile: float = 75.0,
) -> float:
    """Robust symmetric surface distance, bounded to keep assignment inexpensive."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    aa = a[:: max(1, len(a) // 256)]
    bb = b[:: max(1, len(b) // 256)]
    distances = np.sqrt(((aa[:, None, :] - bb[None, :, :]) ** 2).sum(axis=2))
    directed = np.concatenate((distances.min(axis=1), distances.min(axis=0)))
    return float(np.percentile(directed, percentile))


def _confidence(obs: Observation3D) -> float:
    return max(
        obs.role_evidence.values(),
        default=float(obs.candidate.get("score", 0.0)),
    )


def _source_prompts(obs: Observation3D) -> set[str]:
    prompts: set[str] = set()
    provenance = obs.provenance if isinstance(obs.provenance, Mapping) else {}
    prompt = provenance.get("prompt")
    if prompt:
        prompts.add(str(prompt).strip().lower())
    for item in provenance.get("prompt_provenance", []):
        if not isinstance(item, Mapping):
            continue
        prompt = item.get("source_prompt") or item.get("prompt")
        if prompt:
            prompts.add(str(prompt).strip().lower())
    return prompts


def _is_specific_interaction_part(obs: Observation3D) -> bool:
    part_score = float(obs.role_evidence.get("interaction_part", 0.0))
    object_score = max(
        float(obs.role_evidence.get("target", 0.0)),
        float(obs.role_evidence.get("reference", 0.0)),
    )
    return part_score >= 0.25 and part_score >= object_score


def _bbox_axis_containment(small: np.ndarray, large: np.ndarray) -> float:
    raw_size = np.maximum(0.0, small[1] - small[0])
    small_size = np.maximum(raw_size, 1e-6)
    intersection = np.maximum(
        0.0,
        np.minimum(small[1], large[1]) - np.maximum(small[0], large[0]),
    )
    coverage = np.clip(intersection / small_size, 0.0, 1.0)
    degenerate = raw_size <= 1e-4
    if np.any(degenerate):
        midpoint = (small[0] + small[1]) / 2.0
        inside = (midpoint >= large[0] - 1e-4) & (midpoint <= large[1] + 1e-4)
        coverage[degenerate] = inside[degenerate].astype(np.float64)
    return float(np.min(coverage))


def _point_to_cloud_fraction(
    source: np.ndarray,
    target: np.ndarray,
    max_distance_m: float,
) -> float:
    if len(source) == 0 or len(target) == 0 or max_distance_m <= 0.0:
        return 0.0
    source_sample = source[:: max(1, len(source) // 512)][:512]
    target_sample = target[:: max(1, len(target) // 2048)][:2048]
    threshold_sq = float(max_distance_m) ** 2
    close = 0
    for start in range(0, len(source_sample), 64):
        chunk = source_sample[start : start + 64]
        squared = np.sum(
            (chunk[:, None, :] - target_sample[None, :, :]) ** 2,
            axis=2,
        )
        close += int(np.sum(np.min(squared, axis=1) <= threshold_sq))
    return float(close / len(source_sample))


def _fragment_subset_metrics(
    first: Observation3D,
    second: Observation3D,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if len(first.points_world) <= len(second.points_world):
        fragment, receiver = first, second
    else:
        fragment, receiver = second, first
    point_ratio = len(fragment.points_world) / max(1, len(receiver.points_world))
    bbox_containment = _bbox_axis_containment(
        fragment.bbox3d_world,
        receiver.bbox3d_world,
    )
    shared_prompts = sorted(_source_prompts(fragment) & _source_prompts(receiver))
    max_point_ratio = float(
        getattr(args, "same_camera_nms_fragment_max_point_ratio", 0.45)
    )
    min_bbox_containment = float(
        getattr(args, "same_camera_nms_fragment_min_bbox_containment", 0.90)
    )
    # Point-to-cloud comparison is the expensive gate. Only evaluate it for a
    # small, semantically compatible cloud already contained by the receiver.
    fragment_candidate = bool(
        max_point_ratio > 0.0
        and point_ratio <= max_point_ratio
        and bbox_containment >= min_bbox_containment
        and shared_prompts
    )
    cloud_fraction = (
        _point_to_cloud_fraction(
            fragment.points_world,
            receiver.points_world,
            float(
                getattr(args, "same_camera_nms_fragment_cloud_distance_m", 0.018)
            ),
        )
        if fragment_candidate
        else 0.0
    )
    return {
        "fragment_observation_id": fragment.observation_id,
        "receiver_observation_id": receiver.observation_id,
        "point_ratio": point_ratio,
        "bbox_axis_containment": bbox_containment,
        "point_to_cloud_fraction": cloud_fraction,
        "shared_prompts": shared_prompts,
        "is_match": bool(
            fragment_candidate
            and cloud_fraction
            >= float(
                getattr(args, "same_camera_nms_fragment_min_cloud_fraction", 0.65)
            )
            and bool(shared_prompts)
        ),
    }


def _record_same_camera_nms_rejection(
    args: argparse.Namespace,
    detail: dict[str, Any],
) -> None:
    rejected = getattr(args, "_same_camera_nms_rejected", None)
    if isinstance(rejected, list) and len(rejected) < 200:
        rejected.append(detail)


def camera_priority_weight(camera: str, args: argparse.Namespace) -> float:
    """Return the configured reliability weight for one camera."""
    preferred_camera = str(
        getattr(args, "preferred_camera", "front") or ""
    ).strip()
    if preferred_camera and camera == preferred_camera:
        return max(1e-6, float(getattr(args, "preferred_camera_weight", 1.0)))
    return 1.0


def weighted_cluster_centroid(
    cluster: Sequence[Observation3D],
    args: argparse.Namespace,
) -> np.ndarray:
    """Compute a point-weighted centroid with extra trust in the preferred view."""
    weighted_sum = np.zeros(3, dtype=np.float64)
    total_weight = 0.0
    for observation in cluster:
        point_count = len(observation.points_world)
        if point_count <= 0:
            continue
        weight = camera_priority_weight(observation.camera, args)
        weighted_sum += weight * observation.points_world.sum(axis=0)
        total_weight += weight * point_count
    if total_weight <= 0.0:
        points = np.concatenate([obs.points_world for obs in cluster], axis=0)
        return points.mean(axis=0)
    return weighted_sum / total_weight


def _merge_same_camera_role_evidence(
    primary: Observation3D,
    duplicate: Observation3D,
) -> None:
    """Retain semantic evidence from a geometry duplicate without duplicating points."""
    roles = set(primary.role_evidence).union(duplicate.role_evidence)
    primary.role_evidence = {
        role: 1.0
        - (1.0 - float(primary.role_evidence.get(role, 0.0)))
        * (1.0 - float(duplicate.role_evidence.get(role, 0.0)))
        for role in roles
    }
    provenance = dict(primary.provenance)
    suppressed = list(provenance.get("same_camera_nms_suppressed", []))
    suppressed.append(
        {
            "observation_id": duplicate.observation_id,
            "candidate_id": duplicate.candidate.get("id"),
            "role_evidence": dict(duplicate.role_evidence),
            "provenance": dict(duplicate.provenance),
        }
    )
    provenance["same_camera_nms_suppressed"] = suppressed
    primary.provenance = provenance


def suppress_same_camera_duplicates(
    observations_with_masks: Sequence[tuple[Observation3D, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[list[Observation3D], list[dict[str, Any]]]:
    """Greedily suppress strict 2D+3D duplicates within one camera."""
    centroid_threshold = float(
        getattr(args, "same_camera_nms_centroid_distance_m", 0.0)
    )
    mask_iou_threshold = float(getattr(args, "same_camera_nms_mask_iou", 0.0))
    containment_threshold = float(
        getattr(args, "same_camera_nms_containment", 0.0)
    )
    max_size_ratio = float(
        getattr(args, "same_camera_nms_max_size_ratio", 0.0)
    )
    if (
        centroid_threshold <= 0.0
        or (mask_iou_threshold <= 0.0 and containment_threshold <= 0.0)
    ):
        return [item[0] for item in observations_with_masks], []

    ranked = sorted(
        observations_with_masks,
        key=lambda item: (-_confidence(item[0]), item[0].observation_id),
    )
    kept: list[tuple[Observation3D, np.ndarray]] = []
    diagnostics: list[dict[str, Any]] = []
    for observation, mask in ranked:
        best_match: tuple[Any, ...] | None = None
        for kept_index, (primary, primary_mask) in enumerate(kept):
            iou, containment = mask_overlap_metrics(mask, primary_mask)
            overlap_ok = (
                (mask_iou_threshold > 0.0 and iou >= mask_iou_threshold)
                or (
                    containment_threshold > 0.0
                    and containment >= containment_threshold
                )
            )
            centroid_distance = float(
                np.linalg.norm(
                    observation.centroid_world - primary.centroid_world
                )
            )
            if centroid_distance > centroid_threshold:
                if overlap_ok:
                    _record_same_camera_nms_rejection(
                        args,
                        {
                            "camera": observation.camera,
                            "candidate_a": observation.candidate.get("id"),
                            "candidate_b": primary.candidate.get("id"),
                            "failed_gate": "centroid_distance",
                            "centroid_distance_m": centroid_distance,
                            "limit_m": centroid_threshold,
                            "mask_iou": iou,
                            "smaller_mask_coverage": containment,
                        },
                    )
                continue
            if _is_specific_interaction_part(observation) != _is_specific_interaction_part(primary):
                if overlap_ok:
                    _record_same_camera_nms_rejection(
                        args,
                        {
                            "camera": observation.camera,
                            "candidate_a": observation.candidate.get("id"),
                            "candidate_b": primary.candidate.get("id"),
                            "failed_gate": "interaction_part_protection",
                            "centroid_distance_m": centroid_distance,
                            "mask_iou": iou,
                            "smaller_mask_coverage": containment,
                        },
                    )
                continue
            size_ratio = bbox_max_axis_size_ratio(
                observation.bbox3d_world,
                primary.bbox3d_world,
            )
            standard_match = bool(
                overlap_ok
                and (max_size_ratio <= 0.0 or size_ratio <= max_size_ratio)
            )
            fragment_metrics = _fragment_subset_metrics(observation, primary, args)
            fragment_match = bool(fragment_metrics["is_match"])
            if not standard_match and not fragment_match:
                plausible_fragment = bool(
                    fragment_metrics["shared_prompts"]
                    and float(fragment_metrics["point_ratio"]) <= 0.60
                    and float(fragment_metrics["bbox_axis_containment"]) >= 0.50
                )
                if overlap_ok or plausible_fragment:
                    _record_same_camera_nms_rejection(
                        args,
                        {
                            "camera": observation.camera,
                            "candidate_a": observation.candidate.get("id"),
                            "candidate_b": primary.candidate.get("id"),
                            "failed_gate": "size_or_fragment_consistency",
                            "centroid_distance_m": centroid_distance,
                            "mask_iou": iou,
                            "smaller_mask_coverage": containment,
                            "max_bbox_axis_size_ratio": size_ratio,
                            "fragment_point_ratio": fragment_metrics["point_ratio"],
                            "fragment_bbox_axis_containment": fragment_metrics[
                                "bbox_axis_containment"
                            ],
                            "fragment_point_to_cloud_fraction": fragment_metrics[
                                "point_to_cloud_fraction"
                            ],
                            "shared_prompts": fragment_metrics["shared_prompts"],
                        },
                    )
                continue
            match_mode = "strict_2d_3d" if standard_match else "fragment_subset"
            strength = max(
                iou,
                containment,
                float(fragment_metrics["bbox_axis_containment"]),
                float(fragment_metrics["point_to_cloud_fraction"]),
            )
            rank = (-strength, centroid_distance, size_ratio)
            if best_match is None or rank < best_match[0]:
                best_match = (
                    rank,
                    kept_index,
                    primary,
                    iou,
                    containment,
                    centroid_distance,
                    size_ratio,
                    match_mode,
                    fragment_metrics,
                )

        if best_match is None:
            kept.append((observation, mask))
            continue

        (
            _,
            kept_index,
            primary,
            iou,
            containment,
            centroid_distance,
            size_ratio,
            match_mode,
            fragment_metrics,
        ) = best_match
        kept_observation, suppressed_observation = primary, observation
        if (
            match_mode == "fragment_subset"
            and len(observation.points_world) > len(primary.points_world)
        ):
            kept_observation, suppressed_observation = observation, primary
            kept[kept_index] = (observation, mask)
        _merge_same_camera_role_evidence(kept_observation, suppressed_observation)
        diagnostics.append(
            {
                "camera": observation.camera,
                "kept_observation_id": kept_observation.observation_id,
                "kept_candidate_id": kept_observation.candidate.get("id"),
                "suppressed_observation_id": suppressed_observation.observation_id,
                "suppressed_candidate_id": suppressed_observation.candidate.get("id"),
                "mask_iou": iou,
                "smaller_mask_coverage": containment,
                "centroid_distance_m": centroid_distance,
                "max_bbox_axis_size_ratio": size_ratio,
                "match_mode": match_mode,
                "fragment_point_ratio": fragment_metrics["point_ratio"],
                "fragment_bbox_axis_containment": fragment_metrics[
                    "bbox_axis_containment"
                ],
                "fragment_point_to_cloud_fraction": fragment_metrics[
                    "point_to_cloud_fraction"
                ],
                "shared_prompts": fragment_metrics["shared_prompts"],
                "kept_confidence": _confidence(kept_observation),
                "suppressed_confidence": _confidence(suppressed_observation),
                "reason": "same_camera_2d_3d_nms",
            }
        )
    return [item[0] for item in kept], diagnostics


def pairwise_should_merge(
    a: Observation3D,
    b: Observation3D,
    args: argparse.Namespace,
) -> bool:
    """Check strict cross-view geometric consistency.

    Centroid proximity is always the coarse gate. Optional geometry cues make
    the gate stricter instead of providing alternate ways around it. An
    enabled surface-distance check is a hard veto when point clouds disagree.
    """
    centroid_ok = bool(
        np.linalg.norm(a.centroid_world - b.centroid_world)
        <= args.cluster_distance_m
    )
    if not centroid_ok:
        return False

    iou_ok = (
        args.bbox_iou_threshold > 0
        and bbox_iou_3d(a.bbox3d_world, b.bbox3d_world)
        >= args.bbox_iou_threshold
    )
    if args.nearest_distance_m is not None:
        surface_distance = symmetric_percentile_nearest_distance(
            a.points_world,
            b.points_world,
        )
        if (
            not np.isfinite(surface_distance)
            or surface_distance > args.nearest_distance_m
        ):
            return False
        # Surface agreement is sufficient secondary geometry evidence. Bbox
        # IoU may be low for partial observations of the same physical object.
        return True

    if args.bbox_iou_threshold > 0:
        return iou_ok
    return True


def _hypothesis_is_valid(
    cluster: Sequence[Observation3D],
    args: argparse.Namespace,
) -> bool:
    """Validate the complete proposed hypothesis."""
    if len({obs.camera for obs in cluster}) != len(cluster):
        return False
    for i, a in enumerate(cluster):
        for b in cluster[i + 1 :]:
            if not pairwise_should_merge(a, b, args):
                return False
            sizes_a = np.maximum(
                a.bbox3d_world[1] - a.bbox3d_world[0],
                1e-6,
            )
            sizes_b = np.maximum(
                b.bbox3d_world[1] - b.bbox3d_world[0],
                1e-6,
            )
            valid_axes = np.maximum(sizes_a, sizes_b) >= 1e-3
            ratios = np.maximum(
                sizes_a[valid_axes] / sizes_b[valid_axes],
                sizes_b[valid_axes] / sizes_a[valid_axes],
            )
            if np.any(ratios > args.max_size_ratio):
                return False
    points = np.concatenate([obs.points_world for obs in cluster])
    robust_extent = (
        np.percentile(points, 99, axis=0)
        - np.percentile(points, 1, axis=0)
    )
    return (
        args.max_hypothesis_diameter_m <= 0
        or np.linalg.norm(robust_extent) <= args.max_hypothesis_diameter_m
    )


def _association_cost(
    obs: Observation3D,
    cluster: Sequence[Observation3D],
    args: argparse.Namespace,
) -> float | None:
    compatible = [
        other for other in cluster if pairwise_should_merge(obs, other, args)
    ]
    if not compatible:
        return None
    proposed = [*cluster, obs]
    if not _hypothesis_is_valid(proposed, args):
        return None
    anchor = cluster[0]
    cue_costs = []
    for other in compatible:
        centroid_distance = float(
            np.linalg.norm(obs.centroid_world - other.centroid_world)
        )
        if centroid_distance <= args.cluster_distance_m:
            cue_costs.append(
                centroid_distance / max(args.cluster_distance_m, 1e-9)
            )
        bbox_iou = bbox_iou_3d(obs.bbox3d_world, other.bbox3d_world)
        if (
            args.bbox_iou_threshold > 0
            and bbox_iou >= args.bbox_iou_threshold
        ):
            cue_costs.append(1.0 - bbox_iou)
        if args.nearest_distance_m is not None:
            nearest = symmetric_percentile_nearest_distance(
                obs.points_world,
                other.points_world,
            )
            if nearest <= args.nearest_distance_m:
                cue_costs.append(
                    nearest / max(args.nearest_distance_m, 1e-9)
                )
    proximity_cost = min(cue_costs)
    sizes_a = np.maximum(
        obs.bbox3d_world[1] - obs.bbox3d_world[0],
        1e-6,
    )
    sizes_b = np.maximum(
        anchor.bbox3d_world[1] - anchor.bbox3d_world[0],
        1e-6,
    )
    size_residual = float(np.mean(np.abs(np.log(sizes_a / sizes_b))))
    overlap_penalty = 1.0 - bbox_iou_3d(
        obs.bbox3d_world,
        anchor.bbox3d_world,
    )
    surface = symmetric_percentile_nearest_distance(
        obs.points_world,
        anchor.points_world,
    )
    surface_scale = (
        args.nearest_distance_m
        if args.nearest_distance_m is not None
        else args.cluster_distance_m
    )
    surface_term = (
        min(1.0, surface / max(surface_scale, 1e-9))
        if np.isfinite(surface)
        else 1.0
    )
    return (
        proximity_cost
        + 0.35 * size_residual
        + 0.25 * overlap_penalty
        + 0.25 * surface_term
    )


def legacy_union_find_clusters(
    observations: Sequence[Observation3D],
    args: argparse.Namespace,
) -> list[list[Observation3D]]:
    """Deprecated reproduction of pairwise transitive connected components."""
    parent = list(range(len(observations)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            if (
                observations[i].camera != observations[j].camera
                and pairwise_should_merge(
                    observations[i],
                    observations[j],
                    args,
                )
            ):
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    groups: dict[int, list[Observation3D]] = {}
    for i, obs in enumerate(observations):
        groups.setdefault(find(i), []).append(obs)
    return list(groups.values())


def solve_min_cost_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Solve a square min-cost bipartite assignment (Hungarian algorithm)."""
    n = cost.shape[0]
    assert (
        cost.shape[1] == n
    ), "solve_min_cost_assignment requires a square cost matrix"
    infinity = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [infinity] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = infinity
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


def warn_near_miss_unmerged_clusters(
    clusters: list[list[Observation3D]],
    args: argparse.Namespace,
) -> None:
    """Warn when suspiciously close hypotheses remain separate."""
    diagnostic_radius_m = max(args.cluster_distance_m * 5, 0.10)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            centroid_a = np.mean(
                [obs.centroid_world for obs in clusters[i]],
                axis=0,
            )
            centroid_b = np.mean(
                [obs.centroid_world for obs in clusters[j]],
                axis=0,
            )
            dist = float(np.linalg.norm(centroid_a - centroid_b))
            if dist <= diagnostic_radius_m:
                cams_a = sorted({obs.camera for obs in clusters[i]})
                cams_b = sorted({obs.camera for obs in clusters[j]})
                print(
                    "[warn] two observation clusters stayed separate but are "
                    f"only {dist:.3f}m apart (cameras {cams_a} vs {cams_b}); "
                    "inspect camera calibration, depth, partial masks, and the "
                    "pairwise geometry diagnostics before relaxing fusion "
                    "thresholds.",
                    file=sys.stderr,
                )


def cluster_observations(
    observations: Sequence[Observation3D],
    args: argparse.Namespace,
) -> list[list[Observation3D]]:
    """Assign one camera at a time to confidence-seeded object hypotheses."""
    if getattr(args, "legacy_union_find", False):
        print(
            "[warn] --legacy-union-find is deprecated and may create "
            "inconsistent hypotheses",
            file=sys.stderr,
        )
        return legacy_union_find_clusters(observations, args)
    by_camera: dict[str, list[Observation3D]] = {}
    for obs in observations:
        by_camera.setdefault(obs.camera, []).append(obs)
    if not by_camera:
        return []
    camera_order = sorted(
        by_camera,
        key=lambda camera: (
            -camera_priority_weight(camera, args)
            * max(_confidence(obs) for obs in by_camera[camera]),
            camera,
        ),
    )
    for camera in camera_order:
        by_camera[camera].sort(
            key=lambda obs: (-_confidence(obs), obs.observation_id)
        )
    clusters: list[list[Observation3D]] = [
        [obs] for obs in by_camera[camera_order[0]]
    ]
    for camera in camera_order[1:]:
        camera_obs, hypothesis_count = by_camera[camera], len(clusters)
        observation_count = len(camera_obs)
        size = observation_count + hypothesis_count
        dummy_cost, blocked = 2.0, 1e6
        cost = np.full((size, size), blocked, dtype=float)
        for i, obs in enumerate(camera_obs):
            for j, cluster in enumerate(clusters):
                value = _association_cost(obs, cluster, args)
                if value is not None:
                    cost[i, j] = value
            cost[i, hypothesis_count + i] = dummy_cost
        for j in range(hypothesis_count):
            cost[observation_count + j, j] = dummy_cost
        cost[observation_count:, hypothesis_count:] = 0.0
        for row, col in solve_min_cost_assignment(cost):
            if row < observation_count:
                if col < hypothesis_count and cost[row, col] < blocked:
                    clusters[col].append(camera_obs[row])
                else:
                    clusters.append([camera_obs[row]])
    warn_near_miss_unmerged_clusters(clusters, args)
    return clusters
