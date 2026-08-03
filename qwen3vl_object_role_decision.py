#!/usr/bin/env python3
"""Qwen3-VL object-level target/reference decision from fusion summary.

This stage is intentionally separate from stage-1 role-spec generation.
It consumes object-level evidence (object_summary.json) produced by
multiview_candidate_fusion.py and asks Qwen3-VL to pick current target/
reference object ids using multi-source evidence:
- instruction prior and role_spec prior
- per-object geometric/quality stats
- per-frame pairwise relations
- labelled full-scene views or representative object crops/masks
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from camera_geometry import (
    find_rgb_path,
    frame_index_from_frame,
    load_rlbench_observations,
    project_points,
    resolve_camera_param_for_frame,
)
from common_io import atomic_json_dump
from dynamic_role_reasoning import (
    DynamicRoleTracker,
    ReasoningThresholds,
    apply_dynamic_role_selection,
    calibrate_decision_confidence,
    pairwise_geometry,
)
from qwen3vl_rlbench_episode_grounding import Qwen3VLRLBenchGrounder
from task_schema import compile_task_schema
from visualization_utils import load_font, object_color_for_id


DEFAULT_DECISION_SCENE_CAMERAS = (
    "front",
    "left_shoulder",
    "right_shoulder",
)


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
        "--decision-policy",
        choices=("every-frame", "adaptive"),
        default="every-frame",
        help=(
            "every-frame invokes Qwen for every selected frame; adaptive keeps "
            "per-frame outputs but refreshes Qwen only on keyframes/events."
        ),
    )
    parser.add_argument("--decision-refresh-interval", type=int, default=5)
    parser.add_argument(
        "--decision-min-propagation-confidence",
        type=float,
        default=0.70,
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
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--decision-visual-mode",
        choices=("scene", "patches"),
        default="scene",
        help=(
            "scene attaches labelled full-scene camera montages; patches preserves "
            "the legacy object contact-sheet input. The two modes are mutually exclusive."
        ),
    )
    parser.add_argument(
        "--decision-scene-cameras",
        default=",".join(DEFAULT_DECISION_SCENE_CAMERAS),
        help="Comma-separated cameras in each scene montage.",
    )
    parser.add_argument(
        "--decision-scene-window-frames",
        type=int,
        default=2,
        help="Number of latest temporal frames rendered as scene montages.",
    )
    parser.add_argument(
        "--max-candidate-images",
        type=int,
        default=3,
        help=(
            "Maximum temporal contact-sheet images attached to one decision prompt. "
            "Set to 0 to disable visual evidence."
        ),
    )
    parser.add_argument(
        "--candidate-views-per-object",
        type=int,
        default=1,
        help="Maximum best-scoring camera crops per object in a contact sheet.",
    )
    parser.add_argument(
        "--decision-max-visual-pixels",
        type=int,
        default=393216,
        help="Maximum pixels in each Stage-4 visual input; 0 disables resizing.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
    )
    parser.add_argument(
        "--decision-artifacts-dir",
        default=None,
        help="Directory for Stage-4 visual inputs. Default: decision_inputs next to --output-json.",
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
        "--dynamic-role-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Maintain task-schema, gripper-event and 3D-relation state across the "
            "episode. Disable only for legacy comparison."
        ),
    )
    parser.add_argument("--grasp-distance-m", type=float, default=0.06)
    parser.add_argument("--gripper-closed-threshold", type=float, default=0.5)
    parser.add_argument("--object-moving-distance-m", type=float, default=0.01)
    parser.add_argument("--object-stable-distance-m", type=float, default=0.008)
    parser.add_argument("--placement-stable-frames", type=int, default=2)
    parser.add_argument("--min-support-xy-overlap", type=float, default=0.35)
    parser.add_argument("--min-support-vertical-gap-m", type=float, default=-0.01)
    parser.add_argument("--max-support-vertical-gap-m", type=float, default=0.025)
    parser.add_argument("--min-containment-ratio", type=float, default=0.5)
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


def _resize_to_pixel_limit(
    image: Image.Image,
    max_visual_pixels: int,
) -> Image.Image:
    if max_visual_pixels <= 0 or image.width * image.height <= max_visual_pixels:
        return image
    scale = (max_visual_pixels / float(image.width * image.height)) ** 0.5
    return image.resize(
        (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )


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
    primary_camera = str(candidate.get("primary_camera") or "")
    observations = sorted(
        list(candidate.get("observations", [])),
        key=lambda item: (
            bool(primary_camera)
            and str(item.get("camera") or "unknown") != primary_camera,
            -float(item.get("sam_score") or 0.0),
        ),
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
    candidate_views_per_object: int = 1,
    max_visual_pixels: int = 393216,
    allowed_object_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Render one object-ID-labelled visual summary for a fused frame."""
    cards = []
    for candidate in sorted(frame_input.get("candidate_objects", []), key=_object_id_sort_key):
        object_id = str(candidate.get("object_id"))
        if allowed_object_ids is not None and object_id not in allowed_object_ids:
            continue
        cards.extend(
            _candidate_observation_cards(
                candidate,
                max_views=max(1, candidate_views_per_object),
            )
        )
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
    title_font = load_font(14)
    label_font = load_font(12)
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
    canvas = _resize_to_pixel_limit(canvas, max_visual_pixels)
    canvas.save(output_path)
    return {
        "kind": "object_contact_sheet",
        "frame_id": frame_id,
        "frame_index": frame_index,
        "object_ids": sorted({card["object_id"] for card in cards}),
        "image_path": str(output_path.resolve()),
        "width": canvas.width,
        "height": canvas.height,
        "pixel_count": canvas.width * canvas.height,
    }


def _collect_temporal_contact_sheets(
    temporal_frames: Sequence[Mapping[str, Any]],
    output_dir: Path | None,
    max_images: int,
    cache: dict[str, dict[str, Any] | None] | None = None,
    candidate_views_per_object: int = 1,
    max_visual_pixels: int = 393216,
    allowed_object_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if output_dir is None or max_images <= 0:
        return []
    selected_frames = list(temporal_frames)[-max_images:]
    results = []
    for frame in selected_frames:
        cache_key = (
            f"{frame.get('frame_index')}::{frame.get('frame_id')}::"
            f"views={candidate_views_per_object}::pixels={max_visual_pixels}::"
            f"ids={','.join(sorted(allowed_object_ids or []))}"
        )
        if cache is not None and cache_key in cache:
            metadata = cache[cache_key]
        else:
            metadata = _build_object_contact_sheet(
                frame,
                output_dir / "frames",
                candidate_views_per_object=candidate_views_per_object,
                max_visual_pixels=max_visual_pixels,
                allowed_object_ids=allowed_object_ids,
            )
            if cache is not None:
                cache[cache_key] = metadata
        if metadata is not None:
            results.append(metadata)
    return results


def _parse_scene_cameras(value: str | Sequence[str] | None) -> list[str]:
    if isinstance(value, str):
        cameras = [item.strip() for item in value.split(",") if item.strip()]
    elif value is None:
        cameras = []
    else:
        cameras = [str(item).strip() for item in value if str(item).strip()]
    return list(dict.fromkeys(cameras)) or list(DEFAULT_DECISION_SCENE_CAMERAS)


def _best_scene_observations(
    frame_input: Mapping[str, Any],
    camera: str,
    allowed_object_ids: set[str],
) -> list[tuple[str, Mapping[str, Any]]]:
    selected = []
    for candidate in sorted(
        frame_input.get("candidate_objects", []),
        key=_object_id_sort_key,
    ):
        object_id = str(candidate.get("object_id"))
        if object_id not in allowed_object_ids:
            continue
        matching = [
            observation
            for observation in candidate.get("observations", [])
            if str(observation.get("camera")) == camera
        ]
        if not matching:
            continue
        best = max(
            matching,
            key=lambda item: float(item.get("sam_score") or 0.0),
        )
        selected.append((object_id, best))
    return selected


def _scene_placeholder(camera: str, size: int = 384) -> Image.Image:
    image = Image.new("RGB", (size, size), (225, 225, 225))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(145, 145, 145), width=1)
    draw.multiline_text(
        (16, 16),
        f"{camera}\nRGB unavailable",
        fill=(65, 65, 65),
        font=load_font(16),
        spacing=4,
    )
    return image


def _observation_mask(
    observation: Mapping[str, Any],
    image_size: tuple[int, int],
) -> Image.Image | None:
    value = observation.get("mask_path")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_file():
        return None
    with Image.open(path) as source:
        mask = source.convert("L")
    if mask.size != image_size:
        mask = mask.resize(image_size, Image.Resampling.NEAREST)
    return mask


def _observation_bbox(
    observation: Mapping[str, Any],
    mask: Image.Image | None,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    raw_bbox = observation.get("mask_bbox_xyxy")
    if isinstance(raw_bbox, Sequence) and len(raw_bbox) == 4:
        try:
            values = [int(round(float(value))) for value in raw_bbox]
        except (TypeError, ValueError):
            values = []
        if values:
            x1, y1, x2, y2 = values
        else:
            return None
    elif mask is not None and mask.getbbox() is not None:
        x1, y1, x2, y2 = mask.getbbox()
    else:
        return None
    width, height = image_size
    if width < 2 or height < 2:
        return None
    x1 = max(0, min(x1, width - 2))
    y1 = max(0, min(y1, height - 2))
    x2 = max(x1 + 1, min(x2, width - 1))
    y2 = max(y1 + 1, min(y2, height - 1))
    return x1, y1, x2, y2


def _render_decision_scene_panel(
    rgb_path: Path,
    camera: str,
    frame_input: Mapping[str, Any],
    allowed_object_ids: set[str],
    episode_dir: Path,
    rlbench_observations: Sequence[Any],
) -> tuple[Image.Image, list[str], bool]:
    with Image.open(rgb_path) as source:
        image = source.convert("RGBA")
    annotations = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(annotations, "RGBA")
    rendered_ids = []
    for object_id, observation in _best_scene_observations(
        frame_input,
        camera,
        allowed_object_ids,
    ):
        color = object_color_for_id(object_id)
        mask = _observation_mask(observation, image.size)
        if mask is not None:
            tint = Image.new("RGBA", image.size, (*color, 0))
            tint.putalpha(mask.point(lambda value: 36 if value > 127 else 0))
            annotations = Image.alpha_composite(annotations, tint)
            draw = ImageDraw.Draw(annotations, "RGBA")
        bbox = _observation_bbox(observation, mask, image.size)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle(bbox, outline=(*color, 190), width=1)
        label_y = y1 - 15 if y1 >= 16 else min(image.height - 14, y1 + 2)
        draw.text(
            (x1 + 2, label_y),
            object_id,
            fill=(*color, 240),
            font=load_font(13),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
        rendered_ids.append(object_id)

    ee_rendered = False
    frame_index = frame_index_from_frame(frame_input)
    if (
        frame_index is not None
        and 0 <= frame_index < len(rlbench_observations)
    ):
        end_effector = _extract_end_effector_position(
            rlbench_observations[frame_index]
        )
        if end_effector is not None:
            try:
                params = resolve_camera_param_for_frame(
                    camera,
                    frame_index,
                    str(frame_input.get("frame_id")),
                    {},
                    rlbench_observations,
                    episode_dir,
                )
                if params is not None:
                    uv, valid = project_points(
                        np.asarray([end_effector], dtype=np.float64),
                        params["intrinsics"],
                        params["extrinsics"],
                    )
                    if valid[0]:
                        u, v = float(uv[0, 0]), float(uv[0, 1])
                        if 0 <= u < image.width and 0 <= v < image.height:
                            radius = 5
                            draw.line(
                                (u - radius, v, u + radius, v),
                                fill=(255, 255, 255, 235),
                                width=1,
                            )
                            draw.line(
                                (u, v - radius, u, v + radius),
                                fill=(255, 255, 255, 235),
                                width=1,
                            )
                            draw.text(
                                (u + 6, v - 8),
                                "EE",
                                fill=(255, 255, 255, 245),
                                font=load_font(12),
                                stroke_width=1,
                                stroke_fill=(0, 0, 0, 220),
                            )
                            ee_rendered = True
            except (IndexError, KeyError, ValueError, np.linalg.LinAlgError):
                pass

    return (
        Image.alpha_composite(image, annotations).convert("RGB"),
        sorted(set(rendered_ids)),
        ee_rendered,
    )


def _build_scene_montage(
    frame_input: Mapping[str, Any],
    episode_dir: Path,
    output_dir: Path,
    cameras: Sequence[str],
    allowed_object_ids: set[str],
    rlbench_observations: Sequence[Any],
    max_visual_pixels: int = 393216,
) -> dict[str, Any] | None:
    frame_id = str(frame_input.get("frame_id"))
    frame_index = frame_input.get("frame_index")
    cameras = list(cameras)
    if not cameras:
        return None

    panels = []
    missing_cameras = []
    object_ids = set()
    ee_cameras = []
    for camera in cameras:
        rgb_path = find_rgb_path(episode_dir, camera, frame_id)
        if rgb_path is None:
            panels.append((camera, _scene_placeholder(camera)))
            missing_cameras.append(camera)
            continue
        panel, panel_object_ids, ee_rendered = _render_decision_scene_panel(
            rgb_path,
            camera,
            frame_input,
            allowed_object_ids,
            episode_dir,
            rlbench_observations,
        )
        panels.append((camera, panel))
        object_ids.update(panel_object_ids)
        if ee_rendered:
            ee_cameras.append(camera)

    cell_width = 384
    cell_height = 384
    gap = 6
    title_height = 28
    panel_title_height = 22
    canvas_width = len(panels) * cell_width + (len(panels) + 1) * gap
    canvas_height = title_height + panel_title_height + cell_height + 2 * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas_width, title_height), fill=(25, 25, 25))
    draw.text(
        (8, 7),
        f"Frame {frame_id} (index={frame_index}) scene",
        fill=(255, 255, 255),
        font=load_font(13),
    )
    for index, (camera, panel) in enumerate(panels):
        x = gap + index * (cell_width + gap)
        y = title_height + gap
        draw.rectangle(
            (x, y, x + cell_width, y + panel_title_height),
            fill=(38, 38, 38),
        )
        draw.text(
            (x + 6, y + 5),
            camera,
            fill=(255, 255, 255),
            font=load_font(12),
        )
        fitted = ImageOps.contain(
            panel,
            (cell_width, cell_height),
            Image.Resampling.LANCZOS,
        )
        panel_x = x + (cell_width - fitted.width) // 2
        panel_y = y + panel_title_height + (cell_height - fitted.height) // 2
        canvas.paste(fitted, (panel_x, panel_y))

    canvas = _resize_to_pixel_limit(canvas, max_visual_pixels)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_key = f"{_safe_path_segment(frame_index)}_{_safe_path_segment(frame_id)}"
    output_path = output_dir / f"{frame_key}_scene.png"
    canvas.save(output_path)
    return {
        "kind": "scene_montage",
        "frame_id": frame_id,
        "frame_index": frame_index,
        "object_ids": sorted(object_ids),
        "cameras": cameras,
        "missing_cameras": missing_cameras,
        "end_effector_cameras": ee_cameras,
        "image_path": str(output_path.resolve()),
        "width": canvas.width,
        "height": canvas.height,
        "pixel_count": canvas.width * canvas.height,
    }


def _collect_temporal_scene_montages(
    temporal_frames: Sequence[Mapping[str, Any]],
    episode_dir: Path | None,
    output_dir: Path | None,
    window_frames: int,
    cameras: Sequence[str],
    allowed_object_ids: set[str],
    rlbench_observations: Sequence[Any],
    max_visual_pixels: int = 393216,
    cache: dict[str, dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    if episode_dir is None or output_dir is None or window_frames <= 0:
        return []
    results = []
    for frame in list(temporal_frames)[-window_frames:]:
        cache_key = (
            f"scene::{frame.get('frame_index')}::{frame.get('frame_id')}::"
            f"cameras={','.join(cameras)}::pixels={max_visual_pixels}::"
            f"ids={','.join(sorted(allowed_object_ids))}"
        )
        if cache is not None and cache_key in cache:
            metadata = cache[cache_key]
        else:
            metadata = _build_scene_montage(
                frame,
                episode_dir,
                output_dir / "scenes",
                cameras,
                allowed_object_ids,
                rlbench_observations,
                max_visual_pixels=max_visual_pixels,
            )
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


def _relation_labels(delta: np.ndarray) -> list[str]:
    labels = []
    for value, positive, negative in (
        (delta[0], "right_of", "left_of"),
        (delta[1], "front_of", "behind"),
        (delta[2], "above", "below"),
    ):
        if value > 0:
            labels.append(positive)
        elif value < 0:
            labels.append(negative)
    return labels


def _frame_pairwise_relations(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load legacy materialized relations or derive compact-summary relations."""
    materialized = list(frame.get("pairwise_relations", []))
    if materialized:
        return materialized

    candidates = list(frame.get("candidate_objects", []))
    relations = []
    for source_index, source in enumerate(candidates):
        for target in candidates[source_index + 1 :]:
            relation = pairwise_geometry(source, target)
            delta = np.asarray(relation.get("delta_world", []), dtype=np.float64)
            if delta.shape == (3,):
                relation["source_to_target_labels"] = _relation_labels(delta)
                relation["target_to_source_labels"] = _relation_labels(-delta)
            relations.append(relation)
    return relations


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


def _extract_gripper_open(observation: Any) -> float | None:
    value = getattr(observation, "gripper_open", None)
    if value is None and isinstance(observation, Mapping):
        value = observation.get("gripper_open")
    if value is None:
        misc = getattr(observation, "misc", None)
        if isinstance(misc, Mapping):
            value = misc.get("gripper_open")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
    observations: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_dir = summary.get("episode_dir")
    if observations is None:
        observations = (
            load_rlbench_observations(
                Path(str(episode_dir)).expanduser().resolve(), None
            )
            if episode_dir
            else []
        )
    ee_by_frame_index: dict[int, np.ndarray] = {}
    gripper_open_by_frame_index: dict[int, float] = {}
    for frame in selected_frames:
        source_frame_index = frame_index_from_frame(frame)
        if source_frame_index is None:
            continue
        if 0 <= source_frame_index < len(observations):
            observation = observations[source_frame_index]
            ee = _extract_end_effector_position(observation)
            if ee is not None:
                ee_by_frame_index[source_frame_index] = ee
            gripper_open = _extract_gripper_open(observation)
            if gripper_open is not None:
                gripper_open_by_frame_index[source_frame_index] = gripper_open

    frame_ids = {str(frame.get("frame_id")) for frame in selected_frames}
    frame_indices = {
        int(frame.get("frame_index"))
        for frame in selected_frames
        if frame.get("frame_index") is not None and str(frame.get("frame_index")).isdigit()
    }
    decision_frame_id = str(selected_frames[-1].get("frame_id")) if selected_frames else None

    context_by_object: dict[str, Any] = {}
    compact_samples_by_object: dict[str, list[dict[str, Any]]] = {}
    if summary.get("storage_layout") == "compact_v1":
        for frame in selected_frames:
            for candidate in frame.get("candidate_objects", []):
                object_id = str(candidate.get("object_id"))
                compact_samples_by_object.setdefault(object_id, []).append(
                    {
                        **dict(candidate),
                        "frame_id": frame.get("frame_id"),
                        "frame_index": frame.get("frame_index"),
                    }
                )
    for track in summary.get("object_tracks", []):
        object_id = str(track.get("object_id"))
        if summary.get("storage_layout") == "compact_v1":
            samples = compact_samples_by_object.get(object_id, [])
        else:
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
        "gripper_state_samples": [
            {
                "frame_index": frame_index,
                "gripper_open": gripper_open_by_frame_index.get(frame_index),
            }
            for frame_index in sorted(
                set(ee_by_frame_index) | set(gripper_open_by_frame_index)
            )
        ],
    }
    return context_by_object, window_meta


def _build_prompt_payload(
    summary: Mapping[str, Any],
    frame_input: Mapping[str, Any],
    temporal_frames: Sequence[Mapping[str, Any]],
    object_track_context: Mapping[str, Any],
    temporal_window: Mapping[str, Any],
    valid_output_object_ids: set[str],
    task_schema: Mapping[str, Any] | None = None,
    dynamic_role_context: Mapping[str, Any] | None = None,
) -> str:
    decision_frame_id = str(frame_input.get("frame_id"))
    window_frames = []
    for temporal_frame in temporal_frames:
        is_decision_frame = str(temporal_frame.get("frame_id")) == decision_frame_id
        evidence_frame = frame_input if is_decision_frame else temporal_frame
        relations = [
            relation
            for relation in _frame_pairwise_relations(evidence_frame)
            if str(relation.get("source_object_id")) in valid_output_object_ids
            and str(relation.get("target_object_id")) in valid_output_object_ids
        ]
        window_frames.append(
            {
                "frame_id": evidence_frame.get("frame_id"),
                "frame_index": evidence_frame.get("frame_index"),
                "is_decision_frame": is_decision_frame,
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
        "task_schema": dict(task_schema or {}),
        "dynamic_role_context": dict(dynamic_role_context or {}),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _decision_prompt(
    payload_json: str,
    representative_images: Sequence[Mapping[str, Any]],
    decision_visual_mode: str = "scene",
) -> str:
    image_list = [
        {
            "kind": item.get("kind"),
            "frame_id": item.get("frame_id"),
            "frame_index": item.get("frame_index"),
            "object_ids": item.get("object_ids", []),
            "cameras": item.get("cameras", []),
            "missing_cameras": item.get("missing_cameras", []),
            "image_path": item.get("image_path"),
        }
        for item in representative_images
    ]

    if decision_visual_mode == "patches":
        identity_guidance = (
            "Match instruction identity cues against the labelled object crops "
            "before using proximity.\n"
        )
        visual_guidance = (
            "- The chronological object contact sheets contain isolated candidate "
            "views. A repeated O-id is one physical candidate, not multiple objects.\n"
        )
        visual_heading = "Chronological object contact sheets (if any):"
    else:
        identity_guidance = (
            "Match instruction identity cues against the labelled full-scene camera "
            "views before using proximity.\n"
        )
        visual_guidance = (
            "- The chronological scene montages contain full front, left-shoulder, "
            "and right-shoulder views. The same O-id and color denote one fused object "
            "across cameras and frames. EE marks the projected end-effector when available.\n"
            "- Use the full scene to compare the gripper, objects, supports, containers, "
            "and their spatial context. A missing-camera panel is not negative evidence.\n"
        )
        visual_heading = "Chronological labelled full-scene montages (if any):"

    return (
        "You are deciding current RLBench object roles from fused object evidence.\n\n"
        "Goal:\n"
        "- Select the current target_object_id from valid_output_object_ids.\n"
        "- Select reference_object_id only when the instruction defines a separate object, "
        "support, container, slot, surface, or region that determines the goal relation.\n"
        "- Set reference_object_id=null when no separate reference exists. A color, base, "
        "support, or part mentioned only to identify the target is not automatically a reference.\n"
        "- Treat explicit instruction identity cues such as color, shape, and named spatial position "
        "as primary evidence. "
        f"{identity_guidance}"
        "- For push/press tasks, the target is the specifically commanded button or pressable object. "
        "Do not select the gripper, a nearby button, or a large supporting panel merely because it is closer.\n"
        "- Use not only instruction text, but also geometric relations, temporal evidence, camera visibility,"
        " mask/point quality, and the chronological visual evidence.\n"
        f"{visual_guidance}"
        "- This is an online decision: the current frame must be judged together with previous frames in the temporal window.\n"
        "- Treat window_frames as chronological evidence ending at is_decision_frame=true. "
        "Use earlier frames to resolve occlusion and appearance, but output only a current valid object ID.\n"
        "- Consider end-effector distance as a soft cue: targets are often near the active end-effector, "
        "but visual identity cues from the instruction take precedence.\n"
        "- If online_history is present, use it only as weak continuity evidence. Re-evaluate the current visual window "
        "independently and do not copy a previous choice when current visual evidence contradicts it.\n"
        "- task_schema expresses the instruction as a generic action and goal predicate; use it to distinguish "
        "manipulated_object, goal_anchor and interaction_part roles.\n"
        "- dynamic_role_context contains deterministic gripper events and 3D relations. Treat confirmed GRASPED, "
        "PLACED_ON and PLACED_IN events as stronger physical evidence than model decision history.\n"
        "- Object IDs are role-neutral. A previously manipulated target may become the current reference after it "
        "is released, stable, and is the top support for the next repeated ON subgoal.\n"
        "- If uncertain, set uncertain=true with explicit reason.\n\n"
        "Input evidence JSON:\n"
        f"{payload_json}\n\n"
        f"{visual_heading}\n"
        f"{json.dumps(image_list, ensure_ascii=False, separators=(',', ':'))}\n\n"
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
        "7. target_object_id and reference_object_id must never be the same object.\n"
        "8. Keep the response strictly as one JSON object.\n\n"
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
    ignored_invalid_ids = (
        [model_target]
        if model_target is not None and model_target not in valid_ids
        else []
    )

    if raw_compatible is None:
        compatible_ids = [model_target] if model_target in valid_ids else []
        final_target = compatible_ids[0] if compatible_ids else None
        selected["instruction_compatible_object_ids"] = compatible_ids
        selected["model_target_object_id"] = model_target
        selected["target_object_id"] = final_target
        selected["target_selection"] = {
            "strategy": "model_target_fallback_missing_compatibility_list",
            "candidate_order": compatible_ids,
            "ignored_invalid_object_ids": ignored_invalid_ids,
        }
        if final_target is None:
            selected["confidence"] = 0.0
            selected["uncertain"] = True
            if not selected.get("uncertain_reason"):
                selected["uncertain_reason"] = "no_valid_instruction_compatible_candidate"
        return selected

    compatibility_type_coerced = not isinstance(raw_compatible, list)
    raw_compatible_values = (
        raw_compatible if isinstance(raw_compatible, list) else [raw_compatible]
    )

    compatible_ids = []
    for value in raw_compatible_values:
        object_id = str(value)
        if object_id not in valid_ids:
            if object_id not in ignored_invalid_ids:
                ignored_invalid_ids.append(object_id)
            continue
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
        "ignored_invalid_object_ids": ignored_invalid_ids,
        "compatibility_type_coerced": compatibility_type_coerced,
        "selected_proximity_cues": (
            _target_proximity_cues(
                temporal_context_by_object.get(str(final_target))
            )
            if final_target is not None
            else None
        ),
    }
    if final_target is None:
        selected["confidence"] = 0.0
        selected["uncertain"] = True
        if not selected.get("uncertain_reason"):
            selected["uncertain_reason"] = "no_valid_instruction_compatible_candidate"
    return selected


def _sanitize_decision_ids(result: dict[str, Any], valid_ids: set[str]) -> list[dict[str, str]]:
    """Null invalid selected IDs while preserving diagnostics instead of aborting."""
    invalid: list[dict[str, str]] = []
    for key in ("target_object_id", "reference_object_id"):
        value = result.get(key)
        if value is None:
            continue
        if str(value) not in valid_ids:
            invalid.append({"field": key, "object_id": str(value)})
            result[key] = None
    if invalid:
        result["invalid_selected_object_ids"] = invalid
        if result.get("target_object_id") is None:
            result["confidence"] = 0.0
            result["uncertain"] = True
            if not result.get("uncertain_reason"):
                result["uncertain_reason"] = "model_selected_invalid_object_id"
    if (
        result.get("target_object_id") is not None
        and result.get("reference_object_id") is not None
        and str(result["target_object_id"]) == str(result["reference_object_id"])
    ):
        invalid.append(
            {
                "field": "reference_object_id",
                "object_id": str(result["reference_object_id"]),
                "reason": "same_as_target_object_id",
            }
        )
        result["reference_object_id"] = None
        result["invalid_selected_object_ids"] = invalid
    return invalid


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


def _adaptive_state(args: argparse.Namespace) -> dict[str, Any]:
    state = getattr(args, "_adaptive_decision_state", None)
    if state is None:
        state = {
            "last_model_result": None,
            "last_model_frame_id": None,
            "last_candidate_ids": set(),
            "last_final_decision": None,
            "frames_since_model": 0,
            "force_refresh": False,
            "force_refresh_reason": None,
        }
        args._adaptive_decision_state = state
    return state


def _current_dynamic_events(
    dynamic_role_context: Mapping[str, Any],
    frame_id: Any,
) -> list[str]:
    current = str(frame_id)
    events = []
    for state in dynamic_role_context.get("objects", {}).values():
        for event in state.get("events", []):
            name = str(event.get("event") or "")
            if str(event.get("frame_id")) == current and name in {
                "GRASPED",
                "RELEASED",
                "PLACED_ON",
                "PLACED_IN",
            }:
                events.append(name.lower())
    return sorted(set(events))


def _finalize_decision_result(
    base_result: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    temporal_context_by_object: Mapping[str, Mapping[str, Any]],
    dynamic_tracker: DynamicRoleTracker | None,
    dynamic_role_context: Mapping[str, Any],
    candidate_ids: set[str],
) -> dict[str, Any]:
    result = _apply_two_stage_target_selection(
        copy.deepcopy(dict(base_result)),
        candidates,
        temporal_context_by_object,
    )
    if dynamic_tracker is not None:
        result = apply_dynamic_role_selection(
            result,
            candidates,
            dynamic_role_context,
        )
        result = calibrate_decision_confidence(
            result,
            candidates,
            temporal_context_by_object,
            dynamic_role_context,
        )
    _sanitize_decision_ids(result, candidate_ids)
    return result


def _adaptive_refresh_reasons(
    args: argparse.Namespace,
    state: Mapping[str, Any],
    candidate_ids: set[str],
    provisional_result: Mapping[str, Any] | None,
    dynamic_events: Sequence[str],
    gripper_transition: bool,
) -> list[str]:
    if getattr(args, "decision_policy", "every-frame") != "adaptive":
        return ["policy_every_frame"]

    reasons: list[str] = []
    if state.get("last_model_result") is None:
        reasons.append("first_frame")
    if state.get("force_refresh"):
        reasons.append(str(state.get("force_refresh_reason") or "forced_refresh"))

    interval = max(1, int(getattr(args, "decision_refresh_interval", 5)))
    if (
        state.get("last_model_result") is not None
        and int(state.get("frames_since_model", 0)) >= interval - 1
    ):
        reasons.append("periodic_refresh")

    previous_candidate_ids = set(state.get("last_candidate_ids", set()))
    if previous_candidate_ids and candidate_ids - previous_candidate_ids:
        reasons.append("new_candidate")

    previous_decision = state.get("last_final_decision") or {}
    for role in ("target_object_id", "reference_object_id"):
        object_id = previous_decision.get(role)
        if object_id is not None and str(object_id) not in candidate_ids:
            reasons.append(f"{role.removesuffix('_object_id')}_disappeared")

    if provisional_result is not None:
        previous_target = previous_decision.get("target_object_id")
        current_target = provisional_result.get("target_object_id")
        if (
            previous_target is not None
            and current_target is not None
            and str(previous_target) != str(current_target)
        ):
            reasons.append("rule_candidate_changed")
        try:
            confidence = float(provisional_result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < float(
            getattr(args, "decision_min_propagation_confidence", 0.70)
        ) or bool(provisional_result.get("uncertain", False)):
            reasons.append("low_propagation_confidence")

    if gripper_transition:
        reasons.append("gripper_transition")
    reasons.extend(f"dynamic_event:{name}" for name in dynamic_events)
    return list(dict.fromkeys(reasons))


def _run_decision_for_frame(
    summary: Mapping[str, Any],
    frame_inputs: Sequence[Mapping[str, Any]],
    frame_input: Mapping[str, Any],
    args: argparse.Namespace,
    grounder: Qwen3VLRLBenchGrounder | None,
    previous_frame_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frame_started = time.perf_counter()
    decision_visual_mode = str(
        getattr(args, "decision_visual_mode", "scene")
    )
    adaptive_state = _adaptive_state(args)
    ordered = _ordered_frames(frame_inputs)
    temporal_frames = _resolve_temporal_frames(ordered, frame_input, args.decision_window_frames)
    observations = getattr(args, "_rlbench_observations", None)
    temporal_context_by_object, temporal_window_meta = _build_temporal_object_context(
        summary,
        temporal_frames,
        observations=observations,
    )

    dynamic_tracker = getattr(args, "_dynamic_role_tracker", None)
    task_schema = getattr(args, "_task_schema", None)
    dynamic_role_context: dict[str, Any] = {}
    gripper_transition = False
    if dynamic_tracker is not None:
        source_frame_index = frame_index_from_frame(frame_input)
        current_observation = (
            observations[source_frame_index]
            if observations is not None
            and source_frame_index is not None
            and 0 <= source_frame_index < len(observations)
            else None
        )
        gripper_position = (
            _extract_end_effector_position(current_observation)
            if current_observation is not None
            else None
        )
        gripper_open = (
            _extract_gripper_open(current_observation)
            if current_observation is not None
            else None
        )
        previous_gripper_open = dynamic_tracker.last_gripper_open
        previous_source_frame_index = getattr(
            args, "_dynamic_last_source_frame_index", None
        )
        gripper_open_history = []
        if observations is not None and source_frame_index is not None:
            history_start = (
                previous_source_frame_index + 1
                if previous_source_frame_index is not None
                else 0
            )
            for observation in observations[
                max(0, history_start) : source_frame_index + 1
            ]:
                value = _extract_gripper_open(observation)
                if value is not None:
                    gripper_open_history.append(value)
            args._dynamic_last_source_frame_index = source_frame_index
        dynamic_role_context = dynamic_tracker.update(
            frame_input,
            temporal_context_by_object,
            gripper_position,
            gripper_open,
            gripper_open_history,
        )
        threshold = float(getattr(args, "gripper_closed_threshold", 0.5))
        transition_values = (
            ([previous_gripper_open] if previous_gripper_open is not None else [])
            + [value for value in gripper_open_history if value is not None]
        )
        if gripper_open is not None and (
            not transition_values or transition_values[-1] != gripper_open
        ):
            transition_values.append(gripper_open)
        gripper_transition = any(
            (float(transition_values[index - 1]) <= threshold)
            != (float(transition_values[index]) <= threshold)
            for index in range(1, len(transition_values))
        )
        for object_id, state in dynamic_role_context.get("objects", {}).items():
            if object_id in temporal_context_by_object:
                temporal_context_by_object[object_id]["dynamic_state"] = state

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
    previous_summary = (
        _summarize_previous_decisions(previous_frame_decisions)
        if getattr(args, "use_decision_history", False)
        else []
    )
    if not candidate_ids:
        empty_decision = {
            "instruction_compatible_object_ids": [],
            "model_target_object_id": None,
            "target_object_id": None,
            "reference_object_id": None,
            "target_selection": {
                "strategy": "no_valid_candidates_skip_model",
                "candidate_order": [],
            },
            "confidence": 0.0,
            "uncertain": True,
            "uncertain_reason": "no_valid_candidates_for_frame",
            "evidence": [],
            "relation_reason": None,
            "reject_object_ids": [],
            "rejected_reason": None,
        }
        if dynamic_tracker is not None:
            dynamic_tracker.record_decision(empty_decision)
        if adaptive_state.get("last_model_result") is not None:
            adaptive_state["force_refresh"] = True
            adaptive_state["force_refresh_reason"] = "candidates_reappeared"
        adaptive_state["last_candidate_ids"] = set()
        adaptive_state["last_final_decision"] = copy.deepcopy(empty_decision)
        return {
            "frame_id": frame_input.get("frame_id"),
            "frame_index": frame_input.get("frame_index"),
            "candidate_ids": [],
            "candidate_filter_stats": filter_stats,
            "temporal_window": temporal_window_meta,
            "temporal_contact_sheets": [],
            "temporal_scene_montages": [],
            "representative_images": [],
            "decision_visual_mode": decision_visual_mode,
            "online_history": previous_summary,
            "task_schema": task_schema.to_dict() if task_schema is not None else {},
            "dynamic_role_context": dynamic_role_context,
            "model_skipped": True,
            "model_invoked": False,
            "decision_source": "no_valid_candidates",
            "source_model_frame_id": adaptive_state.get("last_model_frame_id"),
            "refresh_reasons": ["no_valid_candidates"],
            "performance": {
                "frame_total_seconds": round(time.perf_counter() - frame_started, 6),
                "contact_sheet_seconds": 0.0,
                "scene_montage_seconds": 0.0,
                "visual_preparation_seconds": 0.0,
                "model_seconds": 0.0,
                "input_image_count": 0,
                "scene_image_count": 0,
                "patch_image_count": 0,
                "input_visual_pixels": 0,
            },
            "decision": empty_decision,
            "raw_text": None,
        }
    cached_model_result = adaptive_state.get("last_model_result")
    provisional_result = (
        _finalize_decision_result(
            cached_model_result,
            candidates,
            temporal_context_by_object,
            dynamic_tracker,
            dynamic_role_context,
            candidate_ids,
        )
        if cached_model_result is not None
        else None
    )
    dynamic_events = _current_dynamic_events(
        dynamic_role_context,
        frame_input.get("frame_id"),
    )
    refresh_reasons = _adaptive_refresh_reasons(
        args,
        adaptive_state,
        candidate_ids,
        provisional_result,
        dynamic_events,
        gripper_transition,
    )
    invoke_model = bool(refresh_reasons)
    if args.dry_run and not invoke_model:
        invoke_model = True
        refresh_reasons = ["dry_run_payload"]

    representative_images: list[dict[str, Any]] = []
    contact_sheet_seconds = 0.0
    scene_montage_seconds = 0.0
    payload_json = ""
    messages: list[dict[str, Any]] = []
    if invoke_model:
        artifacts_value = getattr(args, "decision_artifacts_dir", None)
        artifacts_dir = Path(artifacts_value) if artifacts_value else None
        visual_started = time.perf_counter()
        max_visual_pixels = max(
            0,
            int(getattr(args, "decision_max_visual_pixels", 393216)),
        )
        if decision_visual_mode == "patches":
            representative_images = _collect_temporal_contact_sheets(
                temporal_frames,
                artifacts_dir,
                max(
                    0,
                    min(
                        int(getattr(args, "max_candidate_images", 3)),
                        max(1, int(getattr(args, "decision_window_frames", 3))),
                    ),
                ),
                cache=getattr(args, "_contact_sheet_cache", None),
                candidate_views_per_object=max(
                    1,
                    int(getattr(args, "candidate_views_per_object", 1)),
                ),
                max_visual_pixels=max_visual_pixels,
                allowed_object_ids=candidate_ids,
            )
            contact_sheet_seconds = time.perf_counter() - visual_started
        else:
            episode_dir_value = summary.get("episode_dir")
            episode_dir = (
                Path(str(episode_dir_value)).expanduser().resolve()
                if episode_dir_value
                else None
            )
            representative_images = _collect_temporal_scene_montages(
                temporal_frames,
                episode_dir,
                artifacts_dir,
                max(
                    0,
                    min(
                        int(getattr(args, "decision_scene_window_frames", 2)),
                        max(1, int(getattr(args, "decision_window_frames", 3))),
                    ),
                ),
                _parse_scene_cameras(
                    getattr(args, "decision_scene_cameras", None)
                ),
                candidate_ids,
                observations or [],
                max_visual_pixels=max_visual_pixels,
                cache=getattr(args, "_scene_montage_cache", None),
            )
            scene_montage_seconds = time.perf_counter() - visual_started
        payload_json = _build_prompt_payload(
            summary,
            frame_input,
            temporal_frames,
            object_track_context,
            temporal_window_meta,
            candidate_ids,
            task_schema.to_dict() if task_schema is not None else {},
            dynamic_role_context,
        )
        if previous_summary:
            payload = json.loads(payload_json)
            payload["online_history"] = previous_summary
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        prompt_text = _decision_prompt(
            payload_json,
            representative_images,
            decision_visual_mode=decision_visual_mode,
        )

        content: list[dict[str, Any]] = []
        for item in representative_images:
            if item.get("kind") == "scene_montage":
                visual_label = (
                    f"SCENE_MONTAGE_CAMERAS={','.join(item.get('cameras', []))} "
                    f"VISIBLE_OBJECT_IDS={','.join(item.get('object_ids', []))}"
                )
            else:
                visual_label = (
                    "OBJECT_CONTACT_SHEET="
                    f"{','.join(item.get('object_ids', []))}"
                )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"TEMPORAL_FRAME={item.get('frame_id')} "
                        f"FRAME_INDEX={item.get('frame_index')} "
                        f"{visual_label}"
                    ),
                }
            )
            content.append({"type": "image", "image": item["image_path"]})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

    visual_pixels = sum(
        int(item.get("pixel_count") or 0) for item in representative_images
    )
    scene_image_count = sum(
        1 for item in representative_images if item.get("kind") == "scene_montage"
    )
    patch_image_count = sum(
        1
        for item in representative_images
        if item.get("kind") == "object_contact_sheet"
    )
    base_performance = {
        "contact_sheet_seconds": round(contact_sheet_seconds, 6),
        "scene_montage_seconds": round(scene_montage_seconds, 6),
        "visual_preparation_seconds": round(
            contact_sheet_seconds + scene_montage_seconds,
            6,
        ),
        "model_seconds": 0.0,
        "input_image_count": len(representative_images),
        "scene_image_count": scene_image_count,
        "patch_image_count": patch_image_count,
        "input_visual_pixels": visual_pixels,
        "prompt_characters": len(payload_json),
        "input_tokens": None,
        "output_tokens": None,
    }

    if args.dry_run:
        base_performance["frame_total_seconds"] = round(
            time.perf_counter() - frame_started,
            6,
        )
        return {
            "frame_id": frame_input.get("frame_id"),
            "frame_index": frame_input.get("frame_index"),
            "candidate_ids": sorted(candidate_ids),
            "candidate_filter_stats": filter_stats,
            "temporal_window": temporal_window_meta,
            "temporal_contact_sheets": [
                item
                for item in representative_images
                if item.get("kind") == "object_contact_sheet"
            ],
            "temporal_scene_montages": [
                item
                for item in representative_images
                if item.get("kind") == "scene_montage"
            ],
            "representative_images": representative_images,
            "decision_visual_mode": decision_visual_mode,
            "online_history": previous_summary,
            "messages": messages,
            "model_invoked": False,
            "decision_source": "dry_run_qwen_payload",
            "source_model_frame_id": adaptive_state.get("last_model_frame_id"),
            "refresh_reasons": refresh_reasons,
            "performance": base_performance,
            "dry_run": True,
        }

    if grounder is None:
        raise ValueError("grounder is required for non-dry-run decisions")

    raw_text: str | None = None
    model_error: str | None = None
    model_seconds = 0.0
    if invoke_model:
        model_started = time.perf_counter()
        try:
            model_result, raw_text = grounder.generate_json(
                messages,
                max_new_tokens=args.max_new_tokens,
            )
        except RuntimeError as exc:
            model_error = str(exc)
            fallback = copy.deepcopy(cached_model_result or {})
            fallback.update(
                {
                    "confidence": 0.0,
                    "uncertain": True,
                    "uncertain_reason": "model_output_parse_error",
                }
            )
            result = _finalize_decision_result(
                fallback,
                candidates,
                temporal_context_by_object,
                dynamic_tracker,
                dynamic_role_context,
                candidate_ids,
            )
            adaptive_state["force_refresh"] = True
            adaptive_state["force_refresh_reason"] = "previous_model_parse_error"
            decision_source = "qwen_error"
            source_model_frame_id = adaptive_state.get("last_model_frame_id")
        else:
            adaptive_state["last_model_result"] = copy.deepcopy(model_result)
            adaptive_state["last_model_frame_id"] = frame_input.get("frame_id")
            adaptive_state["force_refresh"] = False
            adaptive_state["force_refresh_reason"] = None
            result = _finalize_decision_result(
                model_result,
                candidates,
                temporal_context_by_object,
                dynamic_tracker,
                dynamic_role_context,
                candidate_ids,
            )
            decision_source = (
                "qwen_keyframe"
                if getattr(args, "decision_policy", "every-frame") == "adaptive"
                else "qwen_every_frame"
            )
            source_model_frame_id = frame_input.get("frame_id")
        model_seconds = time.perf_counter() - model_started
        adaptive_state["frames_since_model"] = 0
    else:
        if provisional_result is None:
            raise RuntimeError("adaptive propagation requires a cached model result")
        result = provisional_result
        adaptive_state["frames_since_model"] = int(
            adaptive_state.get("frames_since_model", 0)
        ) + 1
        decision_source = "temporal_propagation"
        source_model_frame_id = adaptive_state.get("last_model_frame_id")

    if dynamic_tracker is not None:
        dynamic_tracker.record_decision(result)
    adaptive_state["last_candidate_ids"] = set(candidate_ids)
    adaptive_state["last_final_decision"] = copy.deepcopy(result)

    generation_stats = (
        dict(getattr(grounder, "last_generation_stats", {}) or {})
        if invoke_model
        else {}
    )
    base_performance.update(
        {
            "model_seconds": round(model_seconds, 6),
            "input_tokens": generation_stats.get("input_tokens"),
            "output_tokens": generation_stats.get("output_tokens"),
            "generation_attempts": generation_stats.get(
                "attempts",
                1 if invoke_model else 0,
            ),
            "processor_seconds": generation_stats.get("processor_seconds"),
            "generate_seconds": generation_stats.get("generate_seconds"),
            "decode_seconds": generation_stats.get("decode_seconds"),
        }
    )
    base_performance["frame_total_seconds"] = round(
        time.perf_counter() - frame_started,
        6,
    )

    return {
        "frame_id": frame_input.get("frame_id"),
        "frame_index": frame_input.get("frame_index"),
        "candidate_ids": sorted(candidate_ids),
        "candidate_filter_stats": filter_stats,
        "temporal_window": temporal_window_meta,
        "temporal_contact_sheets": [
            item
            for item in representative_images
            if item.get("kind") == "object_contact_sheet"
        ],
        "temporal_scene_montages": [
            item
            for item in representative_images
            if item.get("kind") == "scene_montage"
        ],
        "representative_images": representative_images,
        "decision_visual_mode": decision_visual_mode,
        "online_history": previous_summary,
        "task_schema": task_schema.to_dict() if task_schema is not None else {},
        "dynamic_role_context": dynamic_role_context,
        "model_invoked": invoke_model,
        "model_skipped": not invoke_model,
        "decision_source": decision_source,
        "source_model_frame_id": source_model_frame_id,
        "refresh_reasons": refresh_reasons,
        "performance": base_performance,
        "model_error": model_error,
        "decision": {
            "instruction_compatible_object_ids": result.get(
                "instruction_compatible_object_ids", []
            ),
            "model_target_object_id": result.get("model_target_object_id"),
            "target_object_id": result.get("target_object_id"),
            "reference_object_id": result.get("reference_object_id"),
            "target_selection": result.get("target_selection"),
            "dynamic_role_selection": result.get("dynamic_role_selection"),
            "confidence": result.get("confidence"),
            "model_confidence": result.get("model_confidence"),
            "confidence_components": result.get("confidence_components", {}),
            "uncertain": bool(result.get("uncertain", False)),
            "uncertain_reason": result.get("uncertain_reason"),
            "evidence": result.get("evidence", []),
            "relation_reason": result.get("relation_reason"),
            "reject_object_ids": result.get("reject_object_ids", []),
            "rejected_reason": result.get("rejected_reason"),
            "invalid_selected_object_ids": result.get(
                "invalid_selected_object_ids", []
            ),
        },
        "raw_text": raw_text,
    }


def _compact_persisted_frame_decision(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the stable per-frame decision contract without prompt-time snapshots."""
    decision = dict(entry.get("decision", {}))
    compact = {
        "frame_id": entry.get("frame_id"),
        "frame_index": entry.get("frame_index"),
        "online_step": entry.get("online_step"),
        "candidate_ids": entry.get("candidate_ids", []),
        "model_invoked": bool(entry.get("model_invoked", False)),
        "decision_source": entry.get("decision_source"),
        "source_model_frame_id": entry.get("source_model_frame_id"),
        "refresh_reasons": entry.get("refresh_reasons", []),
        "performance": dict(entry.get("performance", {})),
        "representative_images": list(entry.get("representative_images", [])),
        "decision": {
            key: decision.get(key)
            for key in (
                "model_target_object_id",
                "target_object_id",
                "reference_object_id",
                "confidence",
                "model_confidence",
                "uncertain",
                "uncertain_reason",
            )
        },
    }
    if entry.get("model_error") is not None:
        compact["model_error"] = entry.get("model_error")
    if entry.get("model_invoked") and entry.get("raw_text") is not None:
        compact["raw_text"] = entry.get("raw_text")
    return compact


def _build_output_document(
    summary_path: Path,
    summary: Mapping[str, Any],
    frame_decisions: Sequence[Mapping[str, Any]],
    decision_scope: str,
    dry_run: bool,
    decision_policy: str = "every-frame",
    decision_visual_mode: str = "scene",
    model_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not frame_decisions:
        raise ValueError("Cannot build object predictions without frame decisions")
    final_entry = frame_decisions[-1]
    persisted_frame_decisions = (
        [dict(item) for item in frame_decisions]
        if dry_run
        else [_compact_persisted_frame_decision(item) for item in frame_decisions]
    )
    output = {
        "schema_version": summary.get("schema_version"),
        "artifact_type": "object_role_predictions",
        "storage_layout": "debug_full_v1" if dry_run else "compact_v1",
        "generation_id": summary.get("generation_id"),
        "object_summary_json": str(summary_path),
        "decision_scope": decision_scope,
        "decision_policy": decision_policy,
        "decision_visual_mode": decision_visual_mode,
        "decision_frame_id": final_entry.get("frame_id"),
        "decision_frame_index": final_entry.get("frame_index"),
        "instruction_prior": summary.get("instruction_prior"),
        "role_spec_prior": summary.get("role_spec_prior"),
        "task_schema": final_entry.get("task_schema", {}),
        "candidate_ids": final_entry.get("candidate_ids", []),
        "frame_decisions": persisted_frame_decisions,
    }
    performance_rows = [
        dict(item.get("performance", {})) for item in frame_decisions
    ]
    output["performance"] = {
        "frame_count": len(frame_decisions),
        "model_call_count": sum(
            1 for item in frame_decisions if item.get("model_invoked")
        ),
        "propagated_frame_count": sum(
            1
            for item in frame_decisions
            if item.get("decision_source") == "temporal_propagation"
        ),
        "model_load": dict(model_runtime or {}),
        "total_frame_seconds": round(
            sum(float(item.get("frame_total_seconds") or 0.0) for item in performance_rows),
            6,
        ),
        "total_model_seconds": round(
            sum(float(item.get("model_seconds") or 0.0) for item in performance_rows),
            6,
        ),
        "total_contact_sheet_seconds": round(
            sum(float(item.get("contact_sheet_seconds") or 0.0) for item in performance_rows),
            6,
        ),
        "total_scene_montage_seconds": round(
            sum(float(item.get("scene_montage_seconds") or 0.0) for item in performance_rows),
            6,
        ),
        "total_visual_preparation_seconds": round(
            sum(float(item.get("visual_preparation_seconds") or 0.0) for item in performance_rows),
            6,
        ),
        "total_scene_image_count": sum(
            int(item.get("scene_image_count") or 0) for item in performance_rows
        ),
        "total_patch_image_count": sum(
            int(item.get("patch_image_count") or 0) for item in performance_rows
        ),
        "total_input_tokens": sum(
            int(item.get("input_tokens") or 0) for item in performance_rows
        ),
        "total_output_tokens": sum(
            int(item.get("output_tokens") or 0) for item in performance_rows
        ),
        "total_input_visual_pixels": sum(
            int(item.get("input_visual_pixels") or 0) for item in performance_rows
        ),
    }
    output["summary"] = {
        "frame_count": len(frame_decisions),
        "model_call_count": output["performance"]["model_call_count"],
        "propagated_frame_count": output["performance"]["propagated_frame_count"],
        "uncertain_frame_count": sum(
            1
            for item in frame_decisions
            if bool(item.get("decision", {}).get("uncertain", False))
        ),
        "final_candidate_count": len(final_entry.get("candidate_ids", [])),
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
    args._task_schema = compile_task_schema(
        summary.get("instruction_prior"),
        summary.get("role_spec_prior"),
    )
    episode_dir_value = summary.get("episode_dir")
    args._rlbench_observations = (
        load_rlbench_observations(
            Path(str(episode_dir_value)).expanduser().resolve(), None
        )
        if episode_dir_value
        else []
    )
    args._dynamic_role_tracker = (
        DynamicRoleTracker(
            args._task_schema,
            ReasoningThresholds(
                gripper_closed_threshold=args.gripper_closed_threshold,
                grasp_distance_m=args.grasp_distance_m,
                moving_distance_m=args.object_moving_distance_m,
                stable_distance_m=args.object_stable_distance_m,
                placement_stable_frames=max(1, args.placement_stable_frames),
                min_support_xy_overlap=args.min_support_xy_overlap,
                min_support_vertical_gap_m=args.min_support_vertical_gap_m,
                max_support_vertical_gap_m=args.max_support_vertical_gap_m,
                min_containment_ratio=args.min_containment_ratio,
            ),
        )
        if args.dynamic_role_reasoning
        else None
    )
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
    args._scene_montage_cache = {}
    grounder = None if args.dry_run else Qwen3VLRLBenchGrounder(
        model_path=args.model_path,
        grounding_min_side=args.grounding_min_side,
        max_retries=args.max_retries,
        attention_backend=args.attention_backend,
    )
    model_runtime = dict(getattr(grounder, "load_stats", {}) or {})

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
            decision_policy=args.decision_policy,
            decision_visual_mode=args.decision_visual_mode,
            model_runtime=model_runtime,
        )
        atomic_json_dump(output, output_path)
        if not args.dry_run:
            decision = frame_decision.get("decision", {})
            print(
                "Decision result "
                f"frame_id={frame_decision.get('frame_id')} "
                f"target={decision.get('target_object_id')} "
                f"reference={decision.get('reference_object_id')} "
                f"confidence={decision.get('confidence')} "
                f"source={frame_decision.get('decision_source')} "
                f"model_invoked={frame_decision.get('model_invoked')}",
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
                "decision_policy": args.decision_policy,
                "decision_visual_mode": args.decision_visual_mode,
                "model_call_count": sum(
                    1 for item in frame_decisions if item.get("model_invoked")
                ),
                "decision_artifacts_dir": str(artifacts_path),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
