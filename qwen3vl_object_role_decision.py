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

from qwen3vl_rlbench_episode_grounding import Qwen3VLRLBenchGrounder, atomic_json_dump
from multiview_candidate_fusion import load_rlbench_observations


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
        "--decision-frame",
        choices=("last", "first"),
        default="last",
        help="Which frame from frame_decision_inputs to classify by default.",
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
        help="Max representative object images to attach to the decision prompt.",
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


def _collect_representative_images(candidates: Sequence[Mapping[str, Any]], max_images: int) -> list[dict[str, Any]]:
    candidates_with_obs = []
    for candidate in candidates:
        obs_list = list(candidate.get("observations", []))
        if not obs_list:
            continue
        obs_sorted = sorted(obs_list, key=lambda o: float(o.get("sam_score") or 0.0), reverse=True)
        best_obs = obs_sorted[0]
        image_path = _best_observation_image(best_obs)
        if image_path is None:
            continue
        candidates_with_obs.append(
            {
                "object_id": candidate.get("object_id"),
                "camera": best_obs.get("camera"),
                "sam_score": float(best_obs.get("sam_score") or 0.0),
                "image_path": image_path,
            }
        )

    candidates_with_obs.sort(key=lambda item: item["sam_score"], reverse=True)
    return candidates_with_obs[: max(0, max_images)]


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

    kept.sort(
        key=lambda item: (
            -int(item.get("camera_count") or 0),
            -float(item.get("sam_score") or 0.0),
            float((temporal_context_by_object.get(str(item.get("object_id")), {}).get("end_effector_distance_m", {}).get("mean") or 1e9)),
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
            frame_index = frame.get("frame_index")
            if frame_index is None:
                continue
            try:
                idx = int(frame_index)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(observations):
                ee = _extract_end_effector_position(observations[idx])
                if ee is not None:
                    ee_by_frame_index[idx] = ee

    frame_ids = {str(frame.get("frame_id")) for frame in selected_frames}
    frame_indices = {
        int(frame.get("frame_index"))
        for frame in selected_frames
        if frame.get("frame_index") is not None and str(frame.get("frame_index")).isdigit()
    }

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

        ee_distances = []
        for item in sorted(samples, key=lambda it: (it.get("frame_index") is None, it.get("frame_index"), str(it.get("frame_id")))):
            frame_index = item.get("frame_index")
            if frame_index is None:
                continue
            try:
                idx = int(frame_index)
            except (TypeError, ValueError):
                continue
            if idx not in ee_by_frame_index:
                continue
            centroid = np.asarray(item.get("centroid_world", [0.0, 0.0, 0.0]), dtype=np.float64)
            ee_distances.append(float(np.linalg.norm(centroid - ee_by_frame_index[idx])))

        context_by_object[object_id] = {
            "frames_seen_in_window": len(samples),
            "window_camera_count_stats": _values_stats([float(item.get("camera_count") or 0.0) for item in samples]),
            "window_sam_score_stats": _values_stats([float(item.get("sam_score") or 0.0) for item in samples]),
            "window_point_count_stats": _values_stats([float(item.get("point_count") or 0.0) for item in samples]),
            "end_effector_distance_m": _values_stats(ee_distances),
            "window_samples": [
                {
                    "frame_id": item.get("frame_id"),
                    "frame_index": item.get("frame_index"),
                    "centroid_world": item.get("centroid_world"),
                    "camera_count": item.get("camera_count"),
                    "sam_score": item.get("sam_score"),
                    "point_count": item.get("point_count"),
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
    object_track_context: Mapping[str, Any],
    temporal_window: Mapping[str, Any],
) -> str:
    payload = {
        "instruction_prior": summary.get("instruction_prior"),
        "role_spec_prior": summary.get("role_spec_prior"),
        "decision_frame_id": frame_input.get("frame_id"),
        "decision_frame_index": frame_input.get("frame_index"),
        "temporal_window": temporal_window,
        "candidate_objects": [_compact_candidate(item) for item in frame_input.get("candidate_objects", [])],
        "object_track_context": object_track_context,
        "pairwise_relations": frame_input.get("pairwise_relations", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _decision_prompt(payload_json: str, representative_images: Sequence[Mapping[str, Any]]) -> str:
    image_list = [
        {
            "object_id": item.get("object_id"),
            "camera": item.get("camera"),
            "sam_score": item.get("sam_score"),
            "image_path": item.get("image_path"),
        }
        for item in representative_images
    ]

    return (
        "You are deciding current RLBench object roles from fused object evidence.\n\n"
        "Goal:\n"
        "- Select one current target_object_id and one current reference_object_id from candidate_objects.\n"
        "- Use not only instruction text, but also geometric relations, temporal evidence, camera visibility,"
        " mask/point quality, and representative object images.\n"
        "- This is an online decision: the current frame must be judged together with previous frames in the temporal window.\n"
        "- Consider end-effector distance as a soft cue: targets are often near the active end-effector, "
        "but do not force this if other evidence is stronger.\n"
        "- If uncertain, set uncertain=true with explicit reason.\n\n"
        "Input evidence JSON:\n"
        f"{payload_json}\n\n"
        "Representative candidate images (if any):\n"
        f"{json.dumps(image_list, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "1. target_object_id/reference_object_id must be from candidate_objects.object_id, or null when uncertain.\n"
        "2. Prefer objects with stable multi-view support over tiny/noisy single-view fragments unless evidence strongly contradicts.\n"
        "3. Keep the response strictly as one JSON object.\n\n"
        "Output schema:\n"
        "{\n"
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

    track_map = {
        str(item.get("object_id")): {
            "role_evidence": item.get("role_evidence", {}),
            "lifespan_frames": item.get("lifespan_frames"),
            "camera_set": item.get("camera_set"),
            "camera_count_stats": item.get("camera_count_stats"),
            "point_count_stats": item.get("point_count_stats"),
            "sam_score_stats": item.get("sam_score_stats"),
            "bbox_diagonal_m_stats": item.get("bbox_diagonal_m_stats"),
            "motion_path_length_m": item.get("motion_path_length_m"),
            "window_context": temporal_context_by_object.get(str(item.get("object_id"))),
        }
        for item in summary.get("object_tracks", [])
    }
    object_track_context = {
        str(item.get("object_id")): track_map[str(item.get("object_id"))]
        for item in candidates
        if str(item.get("object_id")) in track_map
    }

    candidate_ids = {str(item.get("object_id")) for item in candidates if item.get("object_id") is not None}
    representative_images = _collect_representative_images(candidates, args.max_candidate_images)
    payload_json = _build_prompt_payload(summary, frame_input, object_track_context, temporal_window_meta)
    previous_summary = _summarize_previous_decisions(previous_frame_decisions)
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
                    f"CANDIDATE_OBJECT={item.get('object_id')} CAMERA={item.get('camera')} "
                    f"SAM_SCORE={item.get('sam_score'):.4f}"
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
    _validate_decision_ids(result, candidate_ids)

    return {
        "frame_id": frame_input.get("frame_id"),
        "frame_index": frame_input.get("frame_index"),
        "candidate_ids": sorted(candidate_ids),
        "candidate_filter_stats": filter_stats,
        "temporal_window": temporal_window_meta,
        "representative_images": representative_images,
        "online_history": previous_summary,
        "decision": {
            "target_object_id": result.get("target_object_id"),
            "reference_object_id": result.get("reference_object_id"),
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

    if args.decision_frame_id is not None or args.decision_frame != "last":
        target_frame = _pick_decision_frame(frame_inputs, args.decision_frame, args.decision_frame_id)
        selected_frames = [target_frame]
    else:
        selected_frames = frame_inputs

    frame_decisions: list[dict[str, Any]] = []
    grounder = None if args.dry_run else Qwen3VLRLBenchGrounder(
        model_path=args.model_path,
        grounding_min_side=args.grounding_min_side,
        max_retries=args.max_retries,
    )

    for index, frame_input in enumerate(selected_frames):
        frame_decision = _run_decision_for_frame(
            summary=summary,
            frame_inputs=frame_inputs,
            frame_input=frame_input,
            args=args,
            grounder=grounder,
            previous_frame_decisions=frame_decisions,
        )
        frame_decision["online_step"] = index
        frame_decisions.append(frame_decision)

    final_decision_entry = frame_decisions[-1]
    final_candidate_ids = final_decision_entry.get("candidate_ids", [])
    final_frame_id = final_decision_entry.get("frame_id")
    final_frame_index = final_decision_entry.get("frame_index")

    if args.dry_run:
        output = {
            "object_summary_json": str(summary_path),
            "decision_frame_id": final_frame_id,
            "decision_frame_index": final_frame_index,
            "instruction_prior": summary.get("instruction_prior"),
            "role_spec_prior": summary.get("role_spec_prior"),
            "candidate_ids": final_candidate_ids,
            "frame_decisions": frame_decisions,
            "dry_run": True,
        }
        atomic_json_dump(output, output_path)
        print(
            json.dumps(
                {
                    "output_json": str(output_path),
                    "decision_frame_id": final_frame_id,
                    "frame_count": len(frame_decisions),
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output = {
        "object_summary_json": str(summary_path),
        "decision_frame_id": final_frame_id,
        "decision_frame_index": final_frame_index,
        "instruction_prior": summary.get("instruction_prior"),
        "role_spec_prior": summary.get("role_spec_prior"),
        "candidate_ids": final_candidate_ids,
        "frame_decisions": frame_decisions,
        "decision": final_decision_entry.get("decision"),
        "raw_text": final_decision_entry.get("raw_text"),
    }
    atomic_json_dump(output, output_path)
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "decision_frame_id": final_frame_id,
                "frame_count": len(frame_decisions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
