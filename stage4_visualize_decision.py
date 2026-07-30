#!/usr/bin/env python3
"""Visualize stage-4 target/reference decisions on reprojection overlays.

Reads object_predictions.json + frame_fused_candidates.json and draws highlighted
TARGET / REFERENCE overlays for every frame decision onto per-camera images.

By default it writes images to:
  outputs/<episode>/viz_decision/

It re-renders fused objects on the raw RGB image so selected labels replace the
corresponding O-label instead of being stacked on an existing Stage-3 overlay.
The Stage-3 path is retained in metadata for optional comparison rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from fused_candidate_io import load_fused_frame, load_fused_manifest, load_object_points

from multiview_candidate_fusion import (
    load_camera_params,
    load_rlbench_observations,
    parse_csv,
    resolve_camera_param_for_frame,
)
from visualize_fused_candidates import OBJECT_COLORS, find_rgb_path, project_points

ROLE_COLORS = {
    "target": (255, 80, 80),
    "reference": (80, 180, 255),
}

ROLE_TAG = {
    "target": "T",
    "reference": "R",
}


def _decision_label(role_name: str, object_id: str) -> str:
    """Render a role-specific id (T1/R2), never the internal fused O-id."""
    suffix = object_id[1:] if object_id.startswith("O") and len(object_id) > 1 else object_id
    return f"{ROLE_TAG[role_name]}{suffix}"


def _display_label(object_id: str, role_name: str | None) -> str:
    """Return one label per object: O-id normally, T/R-id when selected."""
    return _decision_label(role_name, object_id) if role_name is not None else object_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-predictions-json", required=True, help="Path to object_predictions.json from stage 4.")
    parser.add_argument("--fused-json", required=True, help="Path to frame_fused_candidates.json from stage 2.")
    parser.add_argument("--episode-dir", default=None, help="Override episode dir (default: from fused JSON).")
    parser.add_argument("--output-dir", default=None, help="Default: viz_decision next to object_predictions.json.")
    parser.add_argument("--viz-dir", default=None, help="Optional Stage-3 viz directory for background overlays (default: <fused-json dir>/viz).")
    parser.add_argument("--cameras", default=None, help="Optional comma-separated camera subset.")
    parser.add_argument("--camera-params-json", default=None, help="Optional explicit camera parameter JSON.")
    parser.add_argument("--rlbench-low-dim-obs", default=None, help="Optional path to RLBench low_dim_obs.pkl.")
    parser.add_argument("--invert-rlbench-extrinsics", action="store_true", help="Invert RLBench camera extrinsics (same as fusion).")
    parser.add_argument("--point-stride", type=int, default=4, help="Subsample points_world by this stride before rendering.")
    parser.add_argument("--point-radius", type=int, default=2, help="Marker radius for highlighted decision points.")
    parser.add_argument("--mask-alpha", type=int, default=90, help="Alpha (0-255) for highlighted translucent point masks.")
    parser.add_argument("--box-width", type=int, default=1, help="Decision/object bbox line width in pixels.")
    parser.add_argument(
        "--annotation-alpha",
        type=int,
        default=150,
        help="Alpha (0-255) for bbox, centroid, and text annotations.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_decision_entries(pred: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame_decisions = pred.get("frame_decisions")
    if isinstance(frame_decisions, list) and frame_decisions:
        entries: list[dict[str, Any]] = []
        for item in frame_decisions:
            if not isinstance(item, Mapping):
                continue
            decision = item.get("decision")
            if not isinstance(decision, Mapping):
                decision = {
                    "target_object_id": pred.get("decision", {}).get("target_object_id"),
                    "reference_object_id": pred.get("decision", {}).get("reference_object_id"),
                }
            entries.append(
                {
                    "frame_id": str(item.get("frame_id")),
                    "frame_index": item.get("frame_index"),
                    "decision": dict(decision),
                }
            )
        if entries:
            return entries

    legacy_decision = pred.get("decision", {})
    frame_id = pred.get("decision_frame_id")
    if frame_id is None:
        raise ValueError("object_predictions.json missing frame_decisions and decision_frame_id")
    return [
        {
            "frame_id": str(frame_id),
            "frame_index": pred.get("decision_frame_index"),
            "decision": dict(legacy_decision) if isinstance(legacy_decision, Mapping) else {},
        }
    ]


def _draw_decision_overlay(
    background: Image.Image,
    frame: Mapping[str, Any],
    objects_by_id: Mapping[str, Mapping[str, Any]],
    decisions: Sequence[tuple[str, str]],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    point_stride: int,
    point_radius: int,
    mask_alpha: int,
    box_width: int = 1,
    annotation_alpha: int = 150,
) -> Image.Image:
    """Render objects once with thin, alpha-composited annotations."""
    image = background.convert("RGBA")
    width, height = image.size
    mask_alpha = max(0, min(255, int(mask_alpha)))
    annotation_alpha = max(0, min(255, int(annotation_alpha)))
    box_width = max(1, int(box_width))
    mask_layer = np.zeros((height, width, 4), dtype=np.uint8)
    boxes: list[tuple[tuple[int, int, int, int], tuple[int, int, int], bool]] = []
    labels: list[tuple[float, float, str, tuple[int, int, int], bool]] = []
    role_by_object_id: dict[str, str] = {}
    for role_name, object_id in decisions:
        # Target wins if malformed output assigns both roles to one object.
        if object_id not in role_by_object_id or role_name == "target":
            role_by_object_id[object_id] = role_name

    for object_index, (object_id, obj) in enumerate(objects_by_id.items()):
        role_name = role_by_object_id.get(object_id)
        selected = role_name is not None
        color = ROLE_COLORS[role_name] if role_name is not None else OBJECT_COLORS[object_index % len(OBJECT_COLORS)]
        points = load_object_points(frame, object_id)
        if len(points) == 0:
            continue
        if point_stride > 1:
            points = points[::point_stride]

        uv, valid = project_points(points, intrinsics, extrinsics)
        in_bounds = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        visible_uv = uv[in_bounds]
        if len(visible_uv) == 0:
            continue

        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        for u, v in visible_uv:
            layer_draw.ellipse(
                [u - point_radius, v - point_radius, u + point_radius, v + point_radius],
                fill=(*color, mask_alpha if selected else min(mask_alpha, 55)),
            )
        mask_layer = np.maximum(mask_layer, np.asarray(layer))

        u_min, v_min = visible_uv.min(axis=0)
        u_max, v_max = visible_uv.max(axis=0)
        label = _display_label(object_id, role_name)
        boxes.append(((int(u_min), int(v_min), int(u_max), int(v_max)), color, selected))

        centroid_uv, centroid_valid = project_points(np.asarray([obj.get("centroid_world", [0.0, 0.0, 0.0])], dtype=np.float64), intrinsics, extrinsics)
        if centroid_valid[0]:
            cu, cv = centroid_uv[0]
        else:
            cu, cv = float(u_min), float(v_min)
        labels.append((float(cu), float(cv), label, color, selected))

    image = Image.alpha_composite(image, Image.fromarray(mask_layer, mode="RGBA"))
    # Draw onto a transparent layer first. Drawing alpha-valued colors directly
    # onto an RGBA image and then converting to RGB would discard transparency.
    annotation_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(annotation_layer, "RGBA")
    for (u_min, v_min, u_max, v_max), color, selected in boxes:
        alpha = annotation_alpha if selected else min(annotation_alpha, 105)
        draw.rectangle(
            [u_min, v_min, u_max, v_max],
            outline=(*color, alpha),
            width=box_width,
        )
    for cu, cv, label, color, selected in labels:
        radius = 2.5 if selected else 1.25
        alpha = annotation_alpha if selected else min(annotation_alpha, 105)
        draw.ellipse(
            [cu - radius, cv - radius, cu + radius, cv + radius],
            fill=(*color, alpha) if selected else None,
            outline=(255, 255, 255, alpha),
            width=1,
        )
        # Draw only the replacement label. No filled label rectangle is used,
        # avoiding the large translucent block that obscured small RLBench objects.
        draw.text((cu + 5, cv - 7), label, fill=(*color, alpha))

    return Image.alpha_composite(image, annotation_layer)


def _background_image(
    viz_dir: Path,
    episode_dir: Path,
    camera: str,
    frame_id: str,
) -> tuple[Image.Image | None, str, str]:
    """Use raw RGB for re-rendering while retaining Stage-3 path for comparison."""
    viz_path = viz_dir / f"{frame_id}_{camera}_reproj.png"
    rgb_path = find_rgb_path(episode_dir, camera, frame_id)
    if rgb_path is not None:
        comparison_path = viz_path if viz_path.is_file() else rgb_path
        return Image.open(rgb_path).convert("RGBA"), str(rgb_path), str(comparison_path)
    if viz_path.is_file():
        return Image.open(viz_path).convert("RGBA"), str(viz_path), str(viz_path)
    return None, "", ""


def _blank_background_size(intrinsics: np.ndarray) -> tuple[int, int]:
    width = int(round(float(intrinsics[0, 2]) * 2))
    height = int(round(float(intrinsics[1, 2]) * 2))
    return max(1, width), max(1, height)


def main() -> None:
    args = build_parser().parse_args()

    pred_path = Path(args.object_predictions_json).expanduser().resolve()
    fused_path = Path(args.fused_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else pred_path.with_name("viz_decision")
    viz_dir = Path(args.viz_dir).expanduser().resolve() if args.viz_dir else fused_path.with_name("viz")

    pred = _load_json(pred_path)
    fused = load_fused_manifest(fused_path)

    episode_dir = Path(args.episode_dir).expanduser().resolve() if args.episode_dir else Path(str(fused.get("episode_metadata", {}).get("episode_dir"))).expanduser().resolve()
    camera_params = load_camera_params(Path(args.camera_params_json).expanduser().resolve() if args.camera_params_json else None)
    rlbench_override = Path(args.rlbench_low_dim_obs).expanduser().resolve() if args.rlbench_low_dim_obs else None
    rlbench_observations = load_rlbench_observations(episode_dir, rlbench_override)

    camera_filter = parse_csv(args.cameras)

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []

    for entry in _frame_decision_entries(pred):
        frame_id = str(entry.get("frame_id"))
        decision = entry.get("decision", {})
        target_id = decision.get("target_object_id")
        reference_id = decision.get("reference_object_id")
        decisions: list[tuple[str, str]] = []
        if target_id is not None:
            decisions.append(("target", str(target_id)))
        if reference_id is not None:
            decisions.append(("reference", str(reference_id)))
        if not decisions:
            print(f"[warn] frame_id={frame_id}: decision has neither target_object_id nor reference_object_id; skipping.", file=sys.stderr)
            continue

        frame = load_fused_frame(fused, frame_id)
        frame_index = frame.get("frame_index")
        objects = frame.get("objects", [])
        objects_by_id = {str(item.get("id")): item for item in objects}

        missing_ids = [object_id for _, object_id in decisions if object_id not in objects_by_id]
        if missing_ids:
            raise ValueError(f"Decision object ids not found in fused frame {frame_id}: {missing_ids}")

        decision_cameras = sorted({c for _, object_id in decisions for c in objects_by_id[object_id].get("visible_camera", [])})
        if not decision_cameras:
            decision_cameras = sorted({c for obj in objects for c in obj.get("visible_camera", [])})
        target_cameras = [c for c in decision_cameras if camera_filter is None or c in camera_filter]

        for camera in target_cameras:
            background, render_background_path, comparison_background_path = _background_image(
                viz_dir, episode_dir, camera, frame_id
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
                print(f"[warn] frame_id={frame_id} camera={camera}: no camera intrinsics/extrinsics found; skipping.", file=sys.stderr)
                continue
            if background is None:
                width, height = _blank_background_size(params["intrinsics"])
                background = Image.new("RGBA", (width, height), (248, 248, 248, 255))
                render_background_path = ""
                comparison_background_path = ""
                print(f"[warn] frame_id={frame_id} camera={camera}: no background image found; rendering on blank canvas.", file=sys.stderr)

            image = _draw_decision_overlay(
                background,
                frame,
                objects_by_id,
                decisions,
                params["intrinsics"],
                params["extrinsics"],
                args.point_stride,
                args.point_radius,
                args.mask_alpha,
                args.box_width,
                args.annotation_alpha,
            )
            out_path = output_dir / f"{frame_id}_{camera}_decision.png"
            image.convert("RGB").save(out_path)
            rendered.append(
                {
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "camera": camera,
                    "target_object_id": target_id,
                    "reference_object_id": reference_id,
                    "output_path": str(out_path),
                    "render_background_path": render_background_path,
                    "background_path": comparison_background_path,
                }
            )

    metadata = {
        "object_predictions_json": str(pred_path),
        "source_fused_json": str(fused_path),
        "decision_frame_id": pred.get("decision_frame_id"),
        "decision_frame_index": pred.get("decision_frame_index"),
        "target_object_id": pred.get("decision", {}).get("target_object_id"),
        "reference_object_id": pred.get("decision", {}).get("reference_object_id"),
        "frame_decisions": [
            {
                "frame_id": item.get("frame_id"),
                "frame_index": item.get("frame_index"),
                "target_object_id": item.get("decision", {}).get("target_object_id"),
                "reference_object_id": item.get("decision", {}).get("reference_object_id"),
            }
            for item in _frame_decision_entries(pred)
        ],
        "rendered": rendered,
    }
    meta_path = output_dir / "decision_visualization.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rendered_images": len(rendered),
                "metadata_json": str(meta_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
