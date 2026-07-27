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

from qwen3vl_rlbench_episode_grounding import Qwen3VLRLBenchGrounder, atomic_json_dump


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
    parser.add_argument("--max-new-tokens", type=int, default=384)
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
        "role_prior": candidate.get("role_prior"),
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = []
    dropped = []
    for candidate in candidates:
        object_id = str(candidate.get("object_id"))
        point_count = int(candidate.get("point_count") or 0)
        camera_count = int(candidate.get("camera_count") or 0)
        sam_score = float(candidate.get("sam_score") or 0.0)
        reasons = []
        if args.min_candidate_point_count > 0 and point_count < args.min_candidate_point_count:
            reasons.append(f"point_count<{args.min_candidate_point_count}")
        if args.min_candidate_camera_count > 1 and camera_count < args.min_candidate_camera_count:
            reasons.append(f"camera_count<{args.min_candidate_camera_count}")
        if args.min_candidate_sam_score > 0.0 and sam_score < args.min_candidate_sam_score:
            reasons.append(f"sam_score<{args.min_candidate_sam_score}")
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
            "max_candidates_for_decision": args.max_candidates_for_decision,
        },
    }
    return kept, stats


def _build_prompt_payload(
    summary: Mapping[str, Any],
    frame_input: Mapping[str, Any],
    object_track_context: Mapping[str, Any],
) -> str:
    payload = {
        "instruction_prior": summary.get("instruction_prior"),
        "role_spec_prior": summary.get("role_spec_prior"),
        "decision_frame_id": frame_input.get("frame_id"),
        "decision_frame_index": frame_input.get("frame_index"),
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
        "- Use not only instruction text, but also geometric relations, camera visibility,"
        " mask/point quality, and representative object images.\n"
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
        "  \"target_object_id\": \"T1\" or null,\n"
        "  \"reference_object_id\": \"R1\" or \"T2\" or null,\n"
        "  \"confidence\": 0.0 to 1.0,\n"
        "  \"uncertain\": false,\n"
        "  \"uncertain_reason\": null,\n"
        "  \"evidence\": [\n"
        "    {\"object_id\": \"T1\", \"reason\": \"...\"},\n"
        "    {\"object_id\": \"T2\", \"reason\": \"...\"}\n"
        "  ],\n"
        "  \"relation_reason\": \"short reasoning about target-reference relation\",\n"
        "  \"reject_object_ids\": [\"T3\"],\n"
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


def main() -> None:
    args = build_parser().parse_args()
    summary_path = Path(args.object_summary_json).expanduser().resolve()
    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else summary_path.with_name("object_predictions.json")
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame_inputs = list(summary.get("frame_decision_inputs", []))
    frame_input = _pick_decision_frame(frame_inputs, args.decision_frame, args.decision_frame_id)

    candidates = list(frame_input.get("candidate_objects", []))
    candidates, filter_stats = _filter_candidates(candidates, args)
    frame_input = dict(frame_input)
    frame_input["candidate_objects"] = candidates

    track_map = {
        str(item.get("object_id")): {
            "role": item.get("role"),
            "lifespan_frames": item.get("lifespan_frames"),
            "camera_set": item.get("camera_set"),
            "camera_count_stats": item.get("camera_count_stats"),
            "point_count_stats": item.get("point_count_stats"),
            "sam_score_stats": item.get("sam_score_stats"),
            "bbox_diagonal_m_stats": item.get("bbox_diagonal_m_stats"),
            "motion_path_length_m": item.get("motion_path_length_m"),
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
    payload_json = _build_prompt_payload(summary, frame_input, object_track_context)
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
            "object_summary_json": str(summary_path),
            "decision_frame_id": frame_input.get("frame_id"),
            "decision_frame_index": frame_input.get("frame_index"),
            "instruction_prior": summary.get("instruction_prior"),
            "role_spec_prior": summary.get("role_spec_prior"),
            "candidate_ids": sorted(candidate_ids),
            "candidate_filter_stats": filter_stats,
            "representative_images": representative_images,
            "messages": [{"role": "user", "content": content}],
            "dry_run": True,
        }
        atomic_json_dump(output, output_path)
        print(
            json.dumps(
                {
                    "output_json": str(output_path),
                    "decision_frame_id": frame_input.get("frame_id"),
                    "candidate_count": len(candidate_ids),
                    "image_count": len(representative_images),
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    grounder = Qwen3VLRLBenchGrounder(
        model_path=args.model_path,
        grounding_min_side=args.grounding_min_side,
        max_retries=args.max_retries,
    )

    messages = [{"role": "user", "content": content}]
    result, raw_text = grounder.generate_json(messages, max_new_tokens=args.max_new_tokens)
    _validate_decision_ids(result, candidate_ids)

    output = {
        "object_summary_json": str(summary_path),
        "decision_frame_id": frame_input.get("frame_id"),
        "decision_frame_index": frame_input.get("frame_index"),
        "instruction_prior": summary.get("instruction_prior"),
        "role_spec_prior": summary.get("role_spec_prior"),
        "candidate_ids": sorted(candidate_ids),
        "candidate_filter_stats": filter_stats,
        "representative_images": representative_images,
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
    atomic_json_dump(output, output_path)
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "decision_frame_id": frame_input.get("frame_id"),
                "candidate_count": len(candidate_ids),
                "image_count": len(representative_images),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
