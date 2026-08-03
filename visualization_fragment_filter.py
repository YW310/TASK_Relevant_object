"""Conservative temporal filtering of duplicate fragment IDs for visualization.

The fused artifacts remain unchanged.  This module only proposes aliases when a
small cluster is repeatedly explained by a larger cluster, or when one frame has
strong multi-view receiver evidence.  Visualizers suppress the donor only while
the proposed receiver is present in the same frame.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from fused_candidate_io import load_object_points


@dataclass(frozen=True)
class FragmentFilterConfig:
    min_axis_containment: float = 0.95
    max_point_ratio: float = 0.40
    max_centroid_distance_m: float = 0.04
    max_point_to_cloud_distance_m: float = 0.018
    min_point_to_cloud_fraction: float = 0.75
    max_interaction_probability: float = 0.20
    single_camera_confirmation_frames: int = 2
    max_relative_offset_std_m: float = 0.012


def _object_id(obj: Mapping[str, Any]) -> str:
    return str(obj.get("id") or obj.get("object_id") or "")


def _bbox(obj: Mapping[str, Any]) -> np.ndarray | None:
    value = np.asarray(obj.get("bbox3d_world", []), dtype=np.float64)
    if value.shape != (2, 3) or not np.isfinite(value).all():
        return None
    return value


def _centroid(obj: Mapping[str, Any]) -> np.ndarray | None:
    value = np.asarray(obj.get("centroid_world", []), dtype=np.float64).reshape(-1)
    if value.size < 3 or not np.isfinite(value[:3]).all():
        return None
    return value[:3]


def _point_count(obj: Mapping[str, Any]) -> int:
    value = obj.get("point_count")
    if value is not None:
        return max(0, int(value))
    return sum(
        max(0, int(obs.get("point_count") or 0))
        for obs in obj.get("observations", [])
    )


def _camera_count(obj: Mapping[str, Any]) -> int:
    return len({str(camera) for camera in obj.get("visible_camera", [])})


def _role_probability(obj: Mapping[str, Any], role: str) -> float:
    evidence = obj.get("role_evidence", {})
    if not isinstance(evidence, Mapping):
        return 0.0
    value = evidence.get(role, 0.0)
    if isinstance(value, Mapping):
        value = next(
            (
                value.get(key)
                for key in ("probability", "score", "score_mass")
                if value.get(key) is not None
            ),
            0.0,
        )
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _source_prompts(obj: Mapping[str, Any]) -> set[str]:
    prompts: set[str] = set()
    for observation in obj.get("observations", []):
        provenance = observation.get("provenance", {})
        for item in provenance.get("prompt_provenance", []):
            prompt = item.get("source_prompt") or item.get("prompt")
            if prompt:
                prompts.add(str(prompt).strip().lower())
    return prompts


def _axis_containment(donor_bbox: np.ndarray, receiver_bbox: np.ndarray) -> float:
    donor_size = np.maximum(donor_bbox[1] - donor_bbox[0], 1e-6)
    intersection = np.maximum(
        0.0,
        np.minimum(donor_bbox[1], receiver_bbox[1])
        - np.maximum(donor_bbox[0], receiver_bbox[0]),
    )
    return float(np.min(np.clip(intersection / donor_size, 0.0, 1.0)))


def _sample_points(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    stride = max(1, len(points) // limit)
    return points[::stride][:limit]


def _point_to_cloud_fraction(
    donor_points: np.ndarray,
    receiver_points: np.ndarray,
    max_distance_m: float,
) -> float:
    donor = _sample_points(np.asarray(donor_points, dtype=np.float64), 512)
    receiver = _sample_points(np.asarray(receiver_points, dtype=np.float64), 2048)
    if donor.ndim != 2 or receiver.ndim != 2 or not len(donor) or not len(receiver):
        return 0.0
    threshold_sq = float(max_distance_m) ** 2
    close_count = 0
    for start in range(0, len(donor), 64):
        chunk = donor[start : start + 64]
        squared = np.sum((chunk[:, None, :] - receiver[None, :, :]) ** 2, axis=2)
        close_count += int(np.sum(np.min(squared, axis=1) <= threshold_sq))
    return float(close_count / len(donor))


def _frame_fragment_evidence(
    frame: Mapping[str, Any],
    config: FragmentFilterConfig,
    point_loader: Callable[[Mapping[str, Any], str], np.ndarray],
) -> list[dict[str, Any]]:
    objects = [obj for obj in frame.get("objects", []) if isinstance(obj, Mapping)]
    point_cache: dict[str, np.ndarray] = {}

    def points(object_id: str) -> np.ndarray:
        if object_id not in point_cache:
            point_cache[object_id] = point_loader(frame, object_id)
        return point_cache[object_id]

    evidence: list[dict[str, Any]] = []
    for donor in objects:
        donor_id = _object_id(donor)
        donor_bbox = _bbox(donor)
        donor_centroid = _centroid(donor)
        donor_count = _point_count(donor)
        if not donor_id or donor_bbox is None or donor_centroid is None or donor_count <= 0:
            continue
        if _role_probability(donor, "interaction_part") > config.max_interaction_probability:
            continue
        donor_prompts = _source_prompts(donor)
        if not donor_prompts:
            continue
        for receiver in objects:
            receiver_id = _object_id(receiver)
            if not receiver_id or receiver_id == donor_id:
                continue
            receiver_count = _point_count(receiver)
            if receiver_count <= donor_count:
                continue
            point_ratio = donor_count / max(1, receiver_count)
            if point_ratio > config.max_point_ratio:
                continue
            receiver_bbox = _bbox(receiver)
            receiver_centroid = _centroid(receiver)
            if receiver_bbox is None or receiver_centroid is None:
                continue
            containment = _axis_containment(donor_bbox, receiver_bbox)
            centroid_distance = float(np.linalg.norm(donor_centroid - receiver_centroid))
            if (
                containment < config.min_axis_containment
                or centroid_distance > config.max_centroid_distance_m
            ):
                continue
            shared_prompts = sorted(donor_prompts & _source_prompts(receiver))
            if not shared_prompts:
                continue
            try:
                cloud_fraction = _point_to_cloud_fraction(
                    points(donor_id),
                    points(receiver_id),
                    config.max_point_to_cloud_distance_m,
                )
            except (KeyError, OSError, TypeError, ValueError):
                continue
            if cloud_fraction < config.min_point_to_cloud_fraction:
                continue
            evidence.append(
                {
                    "frame_id": str(frame.get("frame_id")),
                    "donor_id": donor_id,
                    "receiver_id": receiver_id,
                    "axis_containment": round(containment, 4),
                    "point_ratio": round(point_ratio, 4),
                    "centroid_distance_m": round(centroid_distance, 6),
                    "point_to_cloud_fraction": round(cloud_fraction, 4),
                    "donor_camera_count": _camera_count(donor),
                    "receiver_camera_count": _camera_count(receiver),
                    "shared_prompts": shared_prompts,
                    "relative_offset": (donor_centroid - receiver_centroid).tolist(),
                }
            )
    return evidence


def detect_suspect_fragment_aliases(
    frames: Sequence[Mapping[str, Any]],
    config: FragmentFilterConfig | None = None,
    point_loader: Callable[[Mapping[str, Any], str], np.ndarray] = load_object_points,
) -> dict[str, Any]:
    """Return conservative donor->receiver aliases and their temporal evidence."""
    config = config or FragmentFilterConfig()
    pair_evidence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        for item in _frame_fragment_evidence(frame, config, point_loader):
            pair_evidence[(item["donor_id"], item["receiver_id"])].append(item)

    candidates: list[dict[str, Any]] = []
    for (donor_id, receiver_id), items in pair_evidence.items():
        offsets = np.asarray([item["relative_offset"] for item in items], dtype=np.float64)
        offset_std = (
            float(np.linalg.norm(np.std(offsets, axis=0)))
            if len(items) > 1
            else 0.0
        )
        has_multiview_receiver = any(item["receiver_camera_count"] >= 2 for item in items)
        confirmed = bool(
            has_multiview_receiver
            or (
                len(items) >= max(2, config.single_camera_confirmation_frames)
                and offset_std <= config.max_relative_offset_std_m
            )
        )
        if not confirmed:
            continue
        frame_ids = list(dict.fromkeys(item["frame_id"] for item in items))
        candidates.append(
            {
                "donor_id": donor_id,
                "receiver_id": receiver_id,
                "evidence_frame_count": len(items),
                "evidence_frame_ids": frame_ids[:20],
                "evidence_frame_ids_truncated": len(frame_ids) > 20,
                "has_multiview_receiver": has_multiview_receiver,
                "relative_offset_std_m": round(offset_std, 6),
                "min_axis_containment": min(item["axis_containment"] for item in items),
                "max_point_ratio": max(item["point_ratio"] for item in items),
                "max_centroid_distance_m": max(
                    item["centroid_distance_m"] for item in items
                ),
                "min_point_to_cloud_fraction": min(
                    item["point_to_cloud_fraction"] for item in items
                ),
                "max_receiver_camera_count": max(
                    item["receiver_camera_count"] for item in items
                ),
                "shared_prompts": sorted(
                    {
                        prompt
                        for item in items
                        for prompt in item["shared_prompts"]
                    }
                ),
            }
        )

    aliases: dict[str, str] = {}
    selected_evidence: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["donor_id"],
            -item["evidence_frame_count"],
            not item["has_multiview_receiver"],
            item["receiver_id"],
        ),
    ):
        donor_id = candidate["donor_id"]
        receiver_id = candidate["receiver_id"]
        if (
            donor_id in aliases
            or donor_id in aliases.values()
            or receiver_id in aliases
        ):
            continue
        aliases[donor_id] = receiver_id
        selected_evidence.append(candidate)
    return {"aliases": aliases, "evidence": selected_evidence}


def visible_suspect_aliases(
    objects: Sequence[Mapping[str, Any]], aliases: Mapping[str, str]
) -> dict[str, str]:
    """Return aliases whose donor and receiver coexist in this rendered frame."""
    visible_ids = {_object_id(obj) for obj in objects}
    return {
        str(donor): str(receiver)
        for donor, receiver in aliases.items()
        if str(donor) in visible_ids and str(receiver) in visible_ids
    }
