#!/usr/bin/env python3
"""Qwen3-VL object-level target/reference decision from fusion summary.

This stage is intentionally separate from stage-1 role-spec generation.
It consumes object-level evidence (object_summary.json) produced by
multiview_candidate_fusion.py and asks Qwen3-VL to pick current target/
reference object ids using multi-source evidence:
- instruction prior and role_spec prior
- per-object geometric/quality stats
- per-frame pairwise relations
- representative object crops/masks (when available)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from camera_geometry import frame_index_from_frame, load_rlbench_observations
from common_io import atomic_json_dump
from qwen3vl_rlbench_episode_grounding import Qwen3VLRLBenchGrounder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--object-summary-json", required=True)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Default: object_predictions.json next to --object-summary-json",
    )
    parser.add_argument(
        "--decision-scope",
        choices=("all", "single"),
        default="all",
        help=(
            "Run one rolling-window decision for every frame (all, default), or only "
            "the frame selected by --decision-frame/--decision-frame-id (single)."
        ),
    )
    parser.add_argument(
        "--decision-frame",
        choices=("last", "first"),
        default="last",
        help="Frame selector used only when --decision-scope=single.",
    )
    parser.add_argument(
        "--decision-frame-id",
        default=None,
        help="Optional explicit frame_id override. If set, --decision-frame is ignored.",
    )
    parser.add_argument("--grounding-min-side", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-candidate-images",
        type=int,
        default=8,
        help=(
            "Maximum temporal contact-sheet images attached to one decision prompt. "
            "Set to 0 to disable visual evidence."
        ),
    )
    parser.add_argument(
        "--decision-artifacts-dir",
        default=None,
        help="Directory for temporal object contact sheets. Default: decision_inputs next to --output-json.",
    )
    parser.add_argument(
        "--use-decision-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Feed recent model decisions back into the next prompt. Disabled by default "
            "to prevent an early wrong decision from propagating through the episode."
        ),
    )
    parser.add_argument(
        "--max-candidates-for-decision",
        type=int,
        default=12,
        help="Keep at most this many candidates (after filtering) for Qwen decision prompt.",
    )
    parser.add_argument(
        "--min-candidate-point-count",
        type=int,
        default=0,
        help="Drop candidates with fewer points than this before Qwen decision (0 disables).",
    )
    parser.add_argument(
        "--min-candidate-camera-count",
        type=int,
        default=1,
        help="Drop candidates seen by fewer cameras than this before Qwen decision.",
    )
    parser.add_argument(
        "--min-candidate-sam-score",
        type=float,
        default=0.0,
        help="Drop candidates whose fused sam_score is below this threshold (0 disables).",
    )
    parser.add_argument(
        "--decision-window-frames",
        type=int,
        default=3,
        help=(
            "Number of recent frames (ending at the decision frame) to include as temporal context. "
            "Set to 1 for single-frame behavior."
        ),
    )
    parser.add_argument(
        "--max-ee-distance-m",
        type=float,
        default=None,
        help=(
            "Optional end-effector distance filter: drop candidates whose minimum distance to "
            "the end-effector across the temporal decision window exceeds this threshold."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save the decision input payload without loading/running Qwen3-VL.",
    )
    return parser


def _frame_sort_key(frame: Mapping[str, Any]) -> tuple[bool, Any, str]:
    return (frame.get("frame_index") is None, frame.get("frame_index"), str(frame.get("frame_id")))


def _pick_decision_frame(
    frame_inputs: Sequence[Mapping[str, Any]],
    decision_frame: str,
    decision_frame_id: str | None,
) -> Mapping[str, Any]:
    if not frame_inputs:
        raise ValueError("object_summary contains no frame_decision_inputs")

    if decision_frame_id is not None:
        for item in frame_inputs:
            if str(item.get("frame_id")) == str(decision_frame_id):
                return item
        raise ValueError(f"frame_id={decision_frame_id!r} not found in frame_decision_inputs")

    ordered = sorted(frame_inputs, key=_frame_sort_key)
    if decision_frame == "first":
        return ordered[0]
    return ordered[-1]


def _ordered_frames(frame_inputs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(frame_inputs, key=_frame_sort_key)


def _best_observation_image(obs: Mapping[str, Any]) -> str | None:
    for key in ("masked_crop_path", "crop_path", "mask_path"):
        value = obs.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_file():
            return str(path.resolve())
    return None


def _object_id_sort_key(candidate: Mapping[str, Any]) -> tuple[int, int | str]:
    object_id = str(candidate.get("object_id") or "")
    if object_id.startswith("O") and object_id[1:].isdigit():
        return (0, int(object_id[1:]))
    return (1, object_id)


def _safe_path_segment(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "unknown"


def _contact_sheet_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _semantic_role_score(candidate: Mapping[str, Any], role_name: str) -> float:
    role_evidence = candidate.get("role_evidence", {})
    if not isinstance(role_evidence, Mapping):
        return 0.0
    value = role_evidence.get(role_name, 0.0)
    if isinstance(value, Mapping):
        for key in ("probability", "score", "score_mass"):
            if value.get(key) is not None:
                return float(value[key])
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _target_proximity_cues(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize current-distance and approach evidence used after semantic filtering."""
    context = context or {}
    explicit = context.get("target_proximity_cues", {})
    if isinstance(explicit, Mapping) and explicit:
        return dict(explicit)

    stats = context.get("end_effector_distance_m", {})
    if not isinstance(stats, Mapping):
        stats = {}
    current = stats.get("current", stats.get("last"))
    minimum = stats.get("min")
    return {
        "current_distance_m": float(current) if current is not None else None,
        "window_min_distance_m": float(minimum) if minimum is not None else None,
        "approach_delta_m": 0.0,
        "approaching_step_fraction": 0.0,
        "consistently_approaching": False,
        "distance_sample_count": 0,
    }


def _target_proximity_sort_key(
    object_id: str,
    temporal_context_by_object: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    cues = _target_proximity_cues(temporal_context_by_object.get(object_id))
    current = cues.get("current_distance_m")
    minimum = cues.get("window_min_distance_m")
    approach_delta = float(cues.get("approach_delta_m") or 0.0)
    approach_fraction = float(cues.get("approaching_step_fraction") or 0.0)
    consistently_approaching = bool(cues.get("consistently_approaching", False))
    return (
        current is None,
        float(current) if current is not None else float("inf"),
        not consistently_approaching,
        -approach_fraction,
        -approach_delta,
        float(minimum) if minimum is not None else float("inf"),
    )


def _candidate_observation_cards(
    candidate: Mapping[str, Any],
    max_views: int = 2,
) -> list[dict[str, Any]]:
    observations = sorted(
        list(candidate.get("observations", [])),
        key=lambda item: float(item.get("sam_score") or 0.0),
        reverse=True,
    )
    cards = []
    seen_cameras = set()
    for observation in observations:
        camera = str(observation.get("camera") or "unknown")
        if camera in seen_cameras:
            continue
        image_path = _best_observation_image(observation)
        if image_path is None:
            continue
        seen_cameras.add(camera)
        cards.append(
            {
                "object_id": str(candidate.get("object_id")),
                "camera": camera,
                "sam_score": float(observation.get("sam_score") or 0.0),
                "target_prior": _semantic_role_score(candidate, "target"),
                "reference_prior": _semantic_role_score(candidate, "reference"),
                "image_path": image_path,
            }
        )
        if len(cards) >= max(1, max_views):
            break
    return cards


def _build_object_contact_sheet(
    frame_input: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Render one object-ID-labelled visual summary for a fused frame."""
    cards = []
    for candidate in sorted(frame_input.get("candidate_objects", []), key=_object_id_sort_key):
        cards.extend(_candidate_observation_cards(candidate))
    if not cards:
        return None

    columns = min(4, len(cards))
    rows = (len(cards) + columns - 1) // columns
    cell_width, cell_height = 176, 184
    title_height = 30
    canvas = Image.new(
        "RGB",
        (columns * cell_width, title_height + rows * cell_height),
        (242, 242, 242),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _contact_sheet_font(14)
    label_font = _contact_sheet_font(12)
    frame_id = str(frame_input.get("frame_id"))
    frame_index = frame_input.get("frame_index")
    draw.rectangle((0, 0, canvas.width, title_height), fill=(28, 28, 28))
    draw.text(
        (8, 7),
        f"Frame {frame_id} (index={frame_index}) fused objects",
        fill=(255, 255, 255),
        font=title_font,
    )

    for card_index, card in enumerate(cards):
        column = card_index % columns
        row = card_index // columns
        x0 = column * cell_width
        y0 = title_height + row * cell_height
        draw.rectangle(
            (x0 + 3, y0 + 3, x0 + cell_width - 4, y0 + cell_height - 4),
            fill=(255, 255, 255),
            outline=(175, 175, 175),
        )
        with Image.open(card["image_path"]) as source:
            thumbnail = ImageOps.contain(
                source.convert("RGBA"),
                (cell_width - 16, 132),
                Image.Resampling.LANCZOS,
            )
        px = x0 + (cell_width - thumbnail.width) // 2
        py = y0 + 7
        canvas.paste(thumbnail.convert("RGB"), (px, py))
        draw.text(
            (x0 + 8, y0 + 143),
            f"{card['object_id']}  camera={card.get('camera')}",
            fill=(15, 15, 15),
            font=label_font,
        )
        draw.text(
            (x0 + 8, y0 + 161),
            (
                f"SAM={card.get('sam_score', 0.0):.2f} "
                f"T={card.get('target_prior', 0.0):.2f} "
                f"R={card.get('reference_prior', 0.0):.2f}"
            ),
            fill=(55, 55, 55),
            font=label_font,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_key = f"{_safe_path_segment(frame_index)}_{_safe_path_segment(frame_id)}"
    output_path = output_dir / f"{frame_key}_objects.png"
    canvas.save(output_path)
    return {
        "kind": "object_contact_sheet",
        "frame_id": frame_id,
        "frame_index": frame_index,
        "object_ids": sorted({card["object_id"] for card in cards}),
        "image_path": str(output_path.resolve()),
    }


def _collect_temporal_contact_sheets(
    temporal_frames: Sequence[Mapping[str, Any]],
    output_dir: Path | None,
    max_images: int,
    cache: dict[str, dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    if output_dir is None or max_images <= 0:
        return []
    selected_frames = list(temporal_frames)[-max_images:]
    results = []
    for frame in selected_frames:
        cache_key = f"{frame.get('frame_index')}::{frame.get('frame_id')}"
        if cache is not None and cache_key in cache:
            metadata = cache[cache_key]
        else:
            metadata = _build_object_contact_sheet(frame, output_dir / "frames")
            if cache is not None:
                cache[cache_key] = metadata
        if metadata is not None:
            results.append(metadata)
    return results


def _compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object_id": candidate.get("object_id"),
        "role_evidence": candidate.get("role_evidence", {}),
        "centroid_world": candidate.get("centroid_world"),
        "bbox3d_world": candidate.get("bbox3d_world"),
        "visible_camera": candidate.get("visible_camera", []),
        "camera_count": candidate.get("camera_count"),
        "point_count": candidate.get("point_count"),
        "mask_area": candidate.get("mask_area"),
        "sam_score": candidate.get("sam_score"),
        "observation_count": candidate.get("observation_count"),
    }


def _filter_candidates(
    candidates: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    temporal_context_by_object: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    temporal_context_by_object = temporal_context_by_object or {}
    kept = []
    dropped = []
    for candidate in candidates:
        object_id = str(candidate.get("object_id"))
        point_count = int(candidate.get("point_count") or 0)
        camera_count = int(candidate.get("camera_count") or 0)
        sam_score = float(candidate.get("sam_score") or 0.0)
        ee_stats = temporal_context_by_object.get(object_id, {}).get("end_effector_distance_m", {})
        ee_min = ee_stats.get("min")
        reasons = []
        if args.min_candidate_point_count > 0 and point_count < args.min_candidate_point_count:
            reasons.append(f"point_count<{args.min_candidate_point_count}")
        if args.min_candidate_camera_count > 1 and camera_count < args.min_candidate_camera_count:
            reasons.append(f"camera_count<{args.min_candidate_camera_count}")
        if args.min_candidate_sam_score > 0.0 and sam_score < args.min_candidate_sam_score:
            reasons.append(f"sam_score<{args.min_candidate_sam_score}")
        if args.max_ee_distance_m is not None and ee_min is not None and float(ee_min) > args.max_ee_distance_m:
            reasons.append(f"ee_distance_min>{args.max_ee_distance_m}")
        if reasons:
            dropped.append({"object_id": object_id, "reasons": reasons})
        else:
            kept.append(dict(candidate))

    if not kept:
        # Safety fallback: never send an empty candidate set into the decision prompt.
        kept = [dict(item) for item in candidates]

    best_target_score = max(
        (_semantic_role_score(item, "target") for item in kept),
        default=0.0,
    )
    target_gate_threshold = (
        max(0.05, 0.5 * best_target_score) if best_target_score > 0.0 else 0.0
    )

    def target_gate_rank(item: Mapping[str, Any]) -> int:
        # When Stage 1 has no target evidence, keep every candidate in the same
        # compatibility tier and let visual Qwen evidence perform the filtering.
        if best_target_score <= 0.0:
            return 0
        return int(_semantic_role_score(item, "target") < target_gate_threshold)

    kept.sort(
        key=lambda item: (
            target_gate_rank(item),
            *_target_proximity_sort_key(
                str(item.get("object_id")),
                temporal_context_by_object,
            ),
            -_semantic_role_score(item, "target"),
            -_semantic_role_score(item, "reference"),
            -int(item.get("camera_count") or 0),
            -float(item.get("sam_score") or 0.0),
            -int(item.get("point_count") or 0),
            str(item.get("object_id") or ""),
        )
    )
    if args.max_candidates_for_decision > 0:
        kept = kept[: args.max_candidates_for_decision]

    stats = {
        "input_candidates": len(candidates),
        "kept_candidates": len(kept),
        "dropped_candidates": dropped,
        "rules": {
            "min_candidate_point_count": args.min_candidate_point_count,
            "min_candidate_camera_count": args.min_candidate_camera_count,
            "min_candidate_sam_score": args.min_candidate_sam_score,
            "max_ee_distance_m": args.max_ee_distance_m,
            "max_candidates_for_decision": args.max_candidates_for_decision,
            "target_gate_threshold": target_gate_threshold,
            "target_gate_basis": "stage1_target_role_evidence",
            "within_target_gate_order": (
                "current_ee_distance, consistent_approach, approach_fraction, "
                "approach_delta, window_min_distance"
            ),
        },
    }
    return kept, stats


def _extract_end_effector_position(observation: Any) -> np.ndarray | None:
    pose = getattr(observation, "gripper_pose", None)
    if pose is None and isinstance(observation, Mapping):
        pose = observation.get("gripper_pose")
    if pose is None:
        misc = getattr(observation, "misc", None)
        if isinstance(misc, Mapping):
            pose = misc.get("gripper_pose")
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _resolve_temporal_frames(
    frame_inputs: Sequence[Mapping[str, Any]],
    anchor_frame: Mapping[str, Any],
    window_frames: int,
) -> list[Mapping[str, Any]]:
    ordered = sorted(frame_inputs, key=_frame_sort_key)
    anchor_id = str(anchor_frame.get("frame_id"))
    anchor_idx = next((i for i, item in enumerate(ordered) if str(item.get("frame_id")) == anchor_id), None)
    if anchor_idx is None:
        return [anchor_frame]
    w = max(1, int(window_frames))
    start = max(0, anchor_idx - w + 1)
    return ordered[start : anchor_idx + 1]


def _values_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "last": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "last": float(arr[-1]),
    }


def _build_temporal_object_context(
    summary: Mapping[str, Any],
    selected_frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_dir = summary.get("episode_dir")
    if not episode_dir:
        ee_by_frame_index: dict[int, np.ndarray] = {}
    else:
        observations = load_rlbench_observations(Path(str(episode_dir)).expanduser().resolve(), None)
        ee_by_frame_index = {}
        for frame in selected_frames:
            source_frame_index = frame_index_from_frame(frame)
            if source_frame_index is None:
                continue
            if 0 <= source_frame_index < len(observations):
                ee = _extract_end_effector_position(observations[source_frame_index])
                if ee is not None:
                    ee_by_frame_index[source_frame_index] = ee

    frame_ids = {str(frame.get("frame_id")) for frame in selected_frames}
    frame_indices = {
        int(frame.get("frame_index"))
        for frame in selected_frames
        if frame.get("frame_index") is not None and str(frame.get("frame_index")).isdigit()
    }
    decision_frame_id = str(selected_frames[-1].get("frame_id")) if selected_frames else None

    context_by_object: dict[str, Any] = {}
    for track in summary.get("object_tracks", []):
        object_id = str(track.get("object_id"))
        trajectory = list(track.get("trajectory", []))
        samples = [
            item
            for item in trajectory
            if str(item.get("frame_id")) in frame_ids
        ]
        if not samples:
            continue

        samples = sorted(samples, key=_frame_sort_key)
        ee_distance_samples = []
        for item in samples:
            source_frame_index = frame_index_from_frame(item)
            if source_frame_index is None:
                continue
            if source_frame_index not in ee_by_frame_index:
                continue
            centroid = np.asarray(item.get("centroid_world", [0.0, 0.0, 0.0]), dtype=np.float64)
            ee_distance_samples.append(
                {
                    "frame_id": str(item.get("frame_id")),
                    "frame_index": item.get("frame_index"),
                    "distance_m": float(
                        np.linalg.norm(centroid - ee_by_frame_index[source_frame_index])
                    ),
                }
            )

        ee_distances = [float(item["distance_m"]) for item in ee_distance_samples]
        approach_steps = [
            ee_distances[index - 1] - ee_distances[index]
            for index in range(1, len(ee_distances))
        ]
        # Allow 2 mm of pose/centroid jitter when deciding whether every
        # observed step moves toward the object.
        approach_tolerance_m = 0.002
        approaching_step_fraction = (
            float(
                sum(delta >= -approach_tolerance_m for delta in approach_steps)
                / len(approach_steps)
            )
            if approach_steps
            else 0.0
        )
        current_distance = next(
            (
                float(item["distance_m"])
                for item in reversed(ee_distance_samples)
                if item["frame_id"] == decision_frame_id
            ),
            None,
        )
        proximity_cues = {
            "current_distance_m": current_distance,
            "window_min_distance_m": min(ee_distances) if ee_distances else None,
            "approach_delta_m": (
                float(ee_distances[0] - ee_distances[-1])
                if len(ee_distances) >= 2
                else 0.0
            ),
            "approaching_step_fraction": approaching_step_fraction,
            "consistently_approaching": bool(
                approach_steps
                and all(delta >= -approach_tolerance_m for delta in approach_steps)
            ),
            "distance_sample_count": len(ee_distances),
        }

        centroids = [np.asarray(item.get("centroid_world", [0.0, 0.0, 0.0]), dtype=np.float64) for item in samples]
        motion_path_length_m = float(
            sum(np.linalg.norm(centroids[i] - centroids[i - 1]) for i in range(1, len(centroids)))
        )
        visible_cameras = [
            str(camera)
            for item in samples
            for camera in item.get("visible_camera", [])
        ]
        camera_histogram = {
            camera: visible_cameras.count(camera) for camera in sorted(set(visible_cameras))
        }
        bbox_diagonals = []
        for item in samples:
            bbox = np.asarray(item.get("bbox3d_world", []), dtype=np.float64)
            if bbox.shape == (2, 3):
                bbox_diagonals.append(float(np.linalg.norm(bbox[1] - bbox[0])))

        context_by_object[object_id] = {
            "frames_seen_in_window": len(samples),
            "window_camera_set": sorted(set(visible_cameras)),
            "window_camera_histogram": camera_histogram,
            "window_camera_count_stats": _values_stats([float(item.get("camera_count") or 0.0) for item in samples]),
            "window_sam_score_stats": _values_stats([float(item.get("sam_score") or 0.0) for item in samples]),
            "window_point_count_stats": _values_stats([float(item.get("point_count") or 0.0) for item in samples]),
            "window_mask_area_stats": _values_stats([float(item.get("mask_area") or 0.0) for item in samples]),
            "window_bbox_diagonal_m_stats": _values_stats(bbox_diagonals),
            "window_motion_path_length_m": motion_path_length_m,
            "end_effector_distance_m": _values_stats(ee_distances),
            "end_effector_distance_samples": ee_distance_samples,
            "target_proximity_cues": proximity_cues,
            "window_samples": [
                {
                    "frame_id": item.get("frame_id"),
                    "frame_index": item.get("frame_index"),
                    "centroid_world": item.get("centroid_world"),
                    "visible_camera": item.get("visible_camera", []),
                    "camera_count": item.get("camera_count"),
                    "sam_score": item.get("sam_score"),
                    "point_count": item.get("point_count"),
                    "mask_area": item.get("mask_area"),
                    "bbox3d_world": item.get("bbox3d_world"),
                }
                for item in samples
            ],
        }

    window_meta = {
        "decision_window_frames": len(selected_frames),
        "frame_ids": [str(frame.get("frame_id")) for frame in selected_frames],
        "frame_indices": sorted(frame_indices),
        "end_effector_available_frames": sorted(ee_by_frame_index),
    }
    return context_by_object, window_meta


def _build_prompt_payload(
    summary: Mapping[str, Any],
    frame_input: Mapping[str, Any],
    temporal_frames: Sequence[Mapping[str, Any]],
    object_track_context: Mapping[str, Any],
    temporal_window: Mapping[str, Any],
    valid_output_object_ids: set[str],
) -> str:
    decision_frame_id = str(frame_input.get("frame_id"))
    window_frames = []
    for temporal_frame in temporal_frames:
        is_decision_frame = str(temporal_frame.get("frame_id")) == decision_frame_id
        evidence_frame = frame_input if is_decision_frame else temporal_frame
        candidates = list(evidence_frame.get("candidate_objects", []))
        relations = list(evidence_frame.get("pairwise_relations", []))
        if is_decision_frame:
            relations = [
                relation
                for relation in relations
                if str(relation.get("source_object_id")) in valid_output_object_ids
                and str(relation.get("target_object_id")) in valid_output_object_ids
            ]
        window_frames.append(
            {
                "frame_id": evidence_frame.get("frame_id"),
                "frame_index": evidence_frame.get("frame_index"),
                "is_decision_frame": is_decision_frame,
                "candidate_objects": [_compact_candidate(item) for item in candidates],
                "pairwise_relations": relations,
            }
        )

    payload = {
        "instruction_prior": summary.get("instruction_prior"),
        "role_spec_prior": summary.get("role_spec_prior"),
        "decision_frame_id": frame_input.get("frame_id"),
        "decision_frame_index": frame_input.get("frame_index"),
        "temporal_window": temporal_window,
        "valid_output_object_ids": sorted(valid_output_object_ids),
        "current_candidate_objects": [_compact_candidate(item) for item in frame_input.get("candidate_objects", [])],
        "window_frames": window_frames,
        "object_track_context": object_track_context,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _decision_prompt(payload_json: str, representative_images: Sequence[Mapping[str, Any]]) -> str:
    image_list = [
        {
            "kind": item.get("kind"),
            "frame_id": item.get("frame_id"),
            "frame_index": item.get("frame_index"),
            "object_ids": item.get("object_ids", []),
            "image_path": item.get("image_path"),
        }
        for item in representative_images
    ]

    return (
        "You are deciding current RLBench object roles from fused object evidence.\n\n"
        "Goal:\n"
        "- Select the current target_object_id from valid_output_object_ids.\n"
        "- Select reference_object_id only when the instruction defines a separate object, "
        "support, container, slot, surface, or region that determines the goal relation.\n"
        "- Set reference_object_id=null when no separate reference exists. A color, base, "
        "support, or part mentioned only to identify the target is not automatically a reference.\n"
        "- Treat explicit instruction identity cues such as color, shape, and named spatial position "
        "as primary evidence. Match those cues against the labelled object crops before using proximity.\n"
        "- For push/press tasks, the target is the specifically commanded button or pressable object. "
        "Do not select the gripper, a nearby button, or a large supporting panel merely because it is closer.\n"
        "- Use not only instruction text, but also geometric relations, temporal evidence, camera visibility,"
        " mask/point quality, and the chronological object contact sheets.\n"
        "- A contact sheet can contain two camera views of the same O-id. Use the repeated O-id to compare "
        "appearance across views; it is one physical candidate, not two candidates.\n"
        "- This is an online decision: the current frame must be judged together with previous frames in the temporal window.\n"
        "- Treat window_frames as chronological evidence ending at is_decision_frame=true. "
        "Use earlier frames to resolve occlusion and appearance, but output only a current valid object ID.\n"
        "- Consider end-effector distance as a soft cue: targets are often near the active end-effector, "
        "but visual identity cues from the instruction take precedence.\n"
        "- If online_history is present, use it only as weak continuity evidence. Re-evaluate the current visual window "
        "independently and do not copy a previous choice when current visual evidence contradicts it.\n"
        "- If uncertain, set uncertain=true with explicit reason.\n\n"
        "Input evidence JSON:\n"
        f"{payload_json}\n\n"
        "Chronological object contact sheets (if any):\n"
        f"{json.dumps(image_list, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "1. Any non-null target_object_id/reference_object_id must be in valid_output_object_ids.\n"
        "2. First populate instruction_compatible_object_ids using instruction identity only. "
        "Do not include an object merely because it is close to the gripper.\n"
        "3. Select target_object_id only from instruction_compatible_object_ids. If multiple objects "
        "are compatible, prefer the smallest current_distance_m, then consistently_approaching=true, "
        "higher approaching_step_fraction, and larger approach_delta_m over t-2 to t.\n"
        "4. Prefer objects with stable multi-view support over tiny/noisy single-view fragments unless evidence strongly contradicts.\n"
        "5. reference_object_id=null can be a confident semantic decision; it does not imply uncertainty.\n"
        "6. Distinguish the manipulated object from its interaction part and from descriptive surroundings.\n"
        "7. Keep the response strictly as one JSON object.\n\n"
        "Output schema:\n"
        "{\n"
        "  \"instruction_compatible_object_ids\": [\"O1\", \"O2\"],\n"
        "  \"target_object_id\": \"O1\" or null,\n"
        "  \"reference_object_id\": \"O2\" or \"O3\" or null,\n"
        "  \"confidence\": 0.0 to 1.0,\n"
        "  \"uncertain\": false,\n"
        "  \"uncertain_reason\": null,\n"
        "  \"evidence\": [\n"
        "    {\"object_id\": \"O1\", \"reason\": \"...\"},\n"
        "    {\"object_id\": \"O2\", \"reason\": \"...\"}\n"
        "  ],\n"
        "  \"relation_reason\": \"short reasoning about target-reference relation\",\n"
        "  \"reject_object_ids\": [\"O3\"],\n"
        "  \"rejected_reason\": \"optional short reason\"\n"
        "}"
    )


def _apply_two_stage_target_selection(
    result: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    temporal_context_by_object: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select by instruction compatibility first, then gripper proximity."""
    selected = dict(result)
    valid_ids = {
        str(candidate.get("object_id"))
        for candidate in candidates
        if candidate.get("object_id") is not None
    }
    model_target = (
        str(result.get("target_object_id"))
        if result.get("target_object_id") is not None
        else None
    )
    raw_compatible = result.get("instruction_compatible_object_ids")

    if raw_compatible is None:
        compatible_ids = [model_target] if model_target in valid_ids else []
        selected["instruction_compatible_object_ids"] = compatible_ids
        selected["model_target_object_id"] = model_target
        selected["target_selection"] = {
            "strategy": "model_target_fallback_missing_compatibility_list",
            "candidate_order": compatible_ids,
        }
        return selected

    if not isinstance(raw_compatible, list):
        raise ValueError("instruction_compatible_object_ids must be a JSON list")

    compatible_ids = []
    for value in raw_compatible:
        object_id = str(value)
        if object_id not in valid_ids:
            raise ValueError(
                "instruction_compatible_object_ids contains invalid object id "
                f"{object_id!r}; valid ids are {sorted(valid_ids)}"
            )
        if object_id not in compatible_ids:
            compatible_ids.append(object_id)

    candidate_by_id = {
        str(candidate.get("object_id")): candidate for candidate in candidates
    }
    with_current_distance = [
        object_id
        for object_id in compatible_ids
        if _target_proximity_cues(
            temporal_context_by_object.get(object_id)
        ).get("current_distance_m") is not None
    ]

    if with_current_distance:
        candidate_order = sorted(
            compatible_ids,
            key=lambda object_id: (
                *_target_proximity_sort_key(
                    object_id,
                    temporal_context_by_object,
                ),
                -_semantic_role_score(candidate_by_id[object_id], "target"),
                object_id,
            ),
        )
        final_target = candidate_order[0]
        strategy = "instruction_gate_then_current_gripper_proximity"
    elif model_target in compatible_ids:
        candidate_order = [
            model_target,
            *[object_id for object_id in compatible_ids if object_id != model_target],
        ]
        final_target = model_target
        strategy = "instruction_gate_then_model_target_no_current_gripper_pose"
    else:
        candidate_order = sorted(
            compatible_ids,
            key=lambda object_id: (
                -_semantic_role_score(candidate_by_id[object_id], "target"),
                object_id,
            ),
        )
        final_target = candidate_order[0] if candidate_order else None
        strategy = "instruction_gate_then_semantic_fallback_no_current_gripper_pose"

    selected["model_target_object_id"] = model_target
    selected["instruction_compatible_object_ids"] = compatible_ids
    selected["target_object_id"] = final_target
    selected["target_selection"] = {
        "strategy": strategy,
        "candidate_order": candidate_order,
        "selected_proximity_cues": (
            _target_proximity_cues(
                temporal_context_by_object.get(str(final_target))
            )
            if final_target is not None
            else None
        ),
    }
    return selected


def _validate_decision_ids(result: dict[str, Any], valid_ids: set[str]) -> None:
    for key in ("target_object_id", "reference_object_id"):
        value = result.get(key)
        if value is None:
            continue
        if str(value) not in valid_ids:
            raise ValueError(f"{key}={value!r} is not in candidate object ids: {sorted(valid_ids)}")


def _summarize_previous_decisions(frame_decisions: Sequence[Mapping[str, Any]], max_items: int = 3) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    selected = list(frame_decisions[-max_items:])
    return [
        {
            "frame_id": item.get("frame_id"),
            "frame_index": item.get("frame_index"),
            "target_object_id": item.get("decision", {}).get("target_object_id"),
            "reference_object_id": item.get("decision", {}).get("reference_object_id"),
            "confidence": item.get("decision", {}).get("confidence"),
            "uncertain": item.get("decision", {}).get("uncertain"),
        }
        for item in selected
    ]


def _run_decision_for_frame(
    summary: Mapping[str, Any],
    frame_inputs: Sequence[Mapping[str, Any]],
    frame_input: Mapping[str, Any],
    args: argparse.Namespace,
    grounder: Qwen3VLRLBenchGrounder | None,
    previous_frame_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = _ordered_frames(frame_inputs)
    temporal_frames = _resolve_temporal_frames(ordered, frame_input, args.decision_window_frames)
    temporal_context_by_object, temporal_window_meta = _build_temporal_object_context(summary, temporal_frames)

    candidates = list(frame_input.get("candidate_objects", []))
    candidates, filter_stats = _filter_candidates(candidates, args, temporal_context_by_object)
    frame_input = dict(frame_input)
    frame_input["candidate_objects"] = candidates

    # Never expose episode-wide track aggregates here: every track feature in the
    # prompt must be derived from the resolved temporal window.
    track_map = temporal_context_by_object
    object_track_context = {
        str(item.get("object_id")): track_map[str(item.get("object_id"))]
        for item in candidates
        if str(item.get("object_id")) in track_map
    }

    candidate_ids = {str(item.get("object_id")) for item in candidates if item.get("object_id") is not None}
    artifacts_value = getattr(args, "decision_artifacts_dir", None)
    artifacts_dir = Path(artifacts_value) if artifacts_value else None
    representative_images = _collect_temporal_contact_sheets(
        temporal_frames,
        artifacts_dir,
        args.max_candidate_images,
        cache=getattr(args, "_contact_sheet_cache", None),
    )
    payload_json = _build_prompt_payload(
        summary,
        frame_input,
        temporal_frames,
        object_track_context,
        temporal_window_meta,
        candidate_ids,
    )
    previous_summary = (
        _summarize_previous_decisions(previous_frame_decisions)
        if getattr(args, "use_decision_history", False)
        else []
    )
    if previous_summary:
        payload = json.loads(payload_json)
        payload["online_history"] = previous_summary
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt_text = _decision_prompt(payload_json, representative_images)

    content: list[dict[str, Any]] = []
    for item in representative_images:
        content.append(
            {
                "type": "text",
                "text": (
                    f"TEMPORAL_FRAME={item.get('frame_id')} "
                    f"FRAME_INDEX={item.get('frame_index')} "
                    f"OBJECT_CONTACT_SHEET={','.join(item.get('object_ids', []))}"
                ),
            }
        )
        content.append({"type": "image", "image": item["image_path"]})
    content.append({"type": "text", "text": prompt_text})

    if args.dry_run:
        output = {
            "frame_id": frame_input.get("frame_id"),
            "frame_index": frame_input.get("frame_index"),
            "candidate_ids": sorted(candidate_ids),
            "candidate_filter_stats": filter_stats,
            "temporal_window": temporal_window_meta,
            "temporal_contact_sheets": representative_images,
            "representative_images": representative_images,
            "online_history": previous_summary,
            "messages": [{"role": "user", "content": content}],
            "dry_run": True,
        }
        if grounder is None:
            return output
        return output

    if grounder is None:
        raise ValueError("grounder is required for non-dry-run decisions")

    messages = [{"role": "user", "content": content}]
    result, raw_text = grounder.generate_json(messages, max_new_tokens=args.max_new_tokens)
    result = _apply_two_stage_target_selection(
        result,
        candidates,
        temporal_context_by_object,
    )
    _validate_decision_ids(result, candidate_ids)

    return {
        "frame_id": frame_input.get("frame_id"),
        "frame_index": frame_input.get("frame_index"),
        "candidate_ids": sorted(candidate_ids),
        "candidate_filter_stats": filter_stats,
        "temporal_window": temporal_window_meta,
        "temporal_contact_sheets": representative_images,
        "representative_images": representative_images,
        "online_history": previous_summary,
        "decision": {
            "instruction_compatible_object_ids": result.get(
                "instruction_compatible_object_ids", []
            ),
            "model_target_object_id": result.get("model_target_object_id"),
            "target_object_id": result.get("target_object_id"),
            "reference_object_id": result.get("reference_object_id"),
            "target_selection": result.get("target_selection"),
            "confidence": result.get("confidence"),
            "uncertain": bool(result.get("uncertain", False)),
            "uncertain_reason": result.get("uncertain_reason"),
            "evidence": result.get("evidence", []),
            "relation_reason": result.get("relation_reason"),
            "reject_object_ids": result.get("reject_object_ids", []),
            "rejected_reason": result.get("rejected_reason"),
        },
        "raw_text": raw_text,
    }


def _build_output_document(
    summary_path: Path,
    summary: Mapping[str, Any],
    frame_decisions: Sequence[Mapping[str, Any]],
    decision_scope: str,
    dry_run: bool,
) -> dict[str, Any]:
    if not frame_decisions:
        raise ValueError("Cannot build object predictions without frame decisions")
    final_entry = frame_decisions[-1]
    output = {
        "object_summary_json": str(summary_path),
        "decision_scope": decision_scope,
        "decision_frame_id": final_entry.get("frame_id"),
        "decision_frame_index": final_entry.get("frame_index"),
        "instruction_prior": summary.get("instruction_prior"),
        "role_spec_prior": summary.get("role_spec_prior"),
        "candidate_ids": final_entry.get("candidate_ids", []),
        "frame_decisions": list(frame_decisions),
    }
    if dry_run:
        output["dry_run"] = True
    else:
        output["decision"] = final_entry.get("decision")
        output["raw_text"] = final_entry.get("raw_text")
    return output


def main() -> None:
    args = build_parser().parse_args()
    summary_path = Path(args.object_summary_json).expanduser().resolve()
    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else summary_path.with_name("object_predictions.json")
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_manifest = Path(str(summary.get("source_fused_json", ""))).expanduser()
    if not source_manifest.is_absolute():
        source_manifest = (summary_path.parent / source_manifest).resolve()
    try:
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot validate object_summary against its fused manifest: {source_manifest}") from exc
    for field in ("schema_version", "generation_id"):
        if summary.get(field) != manifest.get(field):
            raise ValueError(
                f"object_summary {field}={summary.get(field)!r} does not match "
                f"manifest {field}={manifest.get(field)!r}"
            )
    frame_inputs = _ordered_frames(list(summary.get("frame_decision_inputs", [])))
    if not frame_inputs:
        raise ValueError("object_summary contains no frame_decision_inputs")

    if args.decision_scope == "single" or args.decision_frame_id is not None:
        frames_to_decide = [
            _pick_decision_frame(frame_inputs, args.decision_frame, args.decision_frame_id)
        ]
        effective_scope = "single"
    else:
        frames_to_decide = frame_inputs
        effective_scope = "all"

    frame_decisions: list[dict[str, Any]] = []
    artifacts_path = (
        Path(args.decision_artifacts_dir).expanduser().resolve()
        if args.decision_artifacts_dir
        else output_path.with_name("decision_inputs")
    )
    args.decision_artifacts_dir = str(artifacts_path)
    args._contact_sheet_cache = {}
    grounder = None if args.dry_run else Qwen3VLRLBenchGrounder(
        model_path=args.model_path,
        grounding_min_side=args.grounding_min_side,
        max_retries=args.max_retries,
    )

    for online_step, frame_input in enumerate(frames_to_decide):
        print(
            f"Decision progress {online_step + 1}/{len(frames_to_decide)}: "
            f"frame_id={frame_input.get('frame_id')}",
            flush=True,
        )
        frame_decision = _run_decision_for_frame(
            summary=summary,
            frame_inputs=frame_inputs,
            frame_input=frame_input,
            args=args,
            grounder=grounder,
            previous_frame_decisions=frame_decisions,
        )
        frame_decision["online_step"] = online_step
        frame_decisions.append(frame_decision)
        # Checkpoint after every successful model call so a long episode leaves
        # useful per-frame decisions even if a later frame fails.
        output = _build_output_document(
            summary_path,
            summary,
            frame_decisions,
            effective_scope,
            args.dry_run,
        )
        atomic_json_dump(output, output_path)
        if not args.dry_run:
            decision = frame_decision.get("decision", {})
            print(
                "Decision result "
                f"frame_id={frame_decision.get('frame_id')} "
                f"target={decision.get('target_object_id')} "
                f"reference={decision.get('reference_object_id')} "
                f"confidence={decision.get('confidence')}",
                flush=True,
            )

    final_decision_entry = frame_decisions[-1]
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "decision_scope": effective_scope,
                "decision_frame_id": final_decision_entry.get("frame_id"),
                "frame_count": len(frame_decisions),
                "decision_artifacts_dir": str(artifacts_path),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
