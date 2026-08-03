#!/usr/bin/env python3
"""Visualize ``frame_fused_candidates.json`` for sanity-checking multi-view 3D fusion.

Produces, per selected frame:

1. Per-camera RGB reprojection panels. Every fused object's world points
   are re-projected back onto each requested camera view (even cameras that
   did not contribute candidates to that object). If the camera
   intrinsics/extrinsics are correct, the dots should land on the same
   physical object in every view.
2. A four-angle 3D point-cloud panel (matplotlib) of every fused object's
   points/centroid in world coordinates, for a quick bird's-eye/side sanity
   check (e.g. objects should sit near the table plane, not scattered).
3. One ``<frame_id>_montage.png`` combining all panels; individual camera and
   point-cloud panels are not written as separate files.
4. A ``sanity_report.json`` with, per fused object: point count, bbox size,
   and the max pairwise distance between the per-camera centroids that were
   merged into it (large values indicate misaligned cameras/extrinsics).

Example
-------
python visualize_fused_candidates.py \\
    --fused-json outputs/episode0/frame_fused_candidates.json \\
    --output-dir outputs/episode0/viz \\
    --frame-ids 0,10,20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from camera_geometry import (
    find_rgb_path,
    load_camera_params,
    load_rlbench_observations,
    project_points,
    resolve_camera_param_for_frame,
)
from common_io import atomic_json_dump, parse_optional_csv as parse_csv
from fused_candidate_io import (
    iter_fused_frames,
    load_fused_manifest,
    load_object_points,
)
from visualization_utils import object_color_for_id
from visualization_fragment_filter import (
    detect_suspect_fragment_aliases,
    visible_suspect_aliases,
)

DEFAULT_MONTAGE_CAMERAS = ("front", "left_shoulder", "right_shoulder")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fused-json", required=True, help="Path to frame_fused_candidates.json.")
    parser.add_argument("--episode-dir", default=None, help="Override episode dir (default: recorded in fused JSON).")
    parser.add_argument("--output-dir", default=None, help="Default: 'viz' next to --fused-json.")
    parser.add_argument("--frame-ids", default=None, help="Optional comma-separated frame_id subset to render.")
    parser.add_argument("--cameras", default=None, help="Optional comma-separated camera subset for reprojection overlays.")
    parser.add_argument("--camera-params-json", default=None, help="Optional explicit camera parameter JSON (same as fusion script).")
    parser.add_argument("--rlbench-low-dim-obs", default=None, help="Optional path to RLBench low_dim_obs.pkl.")
    parser.add_argument("--invert-rlbench-extrinsics", action="store_true", help="Invert RLBench camera extrinsics (same as fusion script).")
    parser.add_argument("--point-stride", type=int, default=4, help="Subsample points_world by this stride before rendering.")
    parser.add_argument("--point-radius", type=int, default=2, help="Reprojected point marker radius in pixels.")
    parser.add_argument("--mask-alpha", type=int, default=80, help="Alpha (0-255) for the semi-transparent reprojected point mask; lower = more transparent.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on the number of frames to render.")
    parser.add_argument("--skip-pointcloud", action="store_true", help="Skip the matplotlib 3D scatter plot (only render 2D overlays + report).")
    parser.add_argument("--montage-columns", type=int, default=2, help="Number of panel columns in each per-frame montage.")
    parser.add_argument("--montage-cell-width", type=int, default=512, help="Width and height of each montage panel in pixels.")
    parser.add_argument(
        "--hide-suspected-fragments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to hiding conservatively detected duplicate fragment IDs when their receiver is visible.",
    )
    return parser


def _rectangle_overlap_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def layout_object_labels(
    draw: ImageDraw.ImageDraw,
    labels: Sequence[tuple[tuple[float, float], str, tuple[int, int, int]]],
    width: int,
    height: int,
    min_gap: int = 2,
) -> list[
    tuple[
        tuple[float, float],
        tuple[int, int],
        tuple[int, int, int, int],
        str,
        tuple[int, int, int],
    ]
]:
    """Place dense O-ID labels without allowing their text boxes to overlap."""
    occupied: list[tuple[int, int, int, int]] = []
    placements = []
    directions = (
        (1, -1),
        (1, 1),
        (-1, -1),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, 0),
        (-1, 0),
    )
    for (cx, cy), label, color in labels:
        sample_bbox = draw.textbbox((0, 0), label)
        text_width = max(1, sample_bbox[2] - sample_bbox[0])
        text_height = max(1, sample_bbox[3] - sample_bbox[1])
        candidates: list[tuple[int, int]] = []
        for radius in (6, 12, 18, 24, 32, 40, 48):
            for dx, dy in directions:
                x = round(cx + dx * radius)
                y = round(cy + dy * radius - text_height / 2)
                x = min(max(0, x), max(0, width - text_width))
                y = min(max(0, y), max(0, height - text_height))
                position = (x, y)
                if position not in candidates:
                    candidates.append(position)

        best: tuple[tuple[int, float], tuple[int, int], tuple[int, int, int, int]] | None = None
        for x, y in candidates:
            rect = (x, y, x + text_width, y + text_height)
            padded = (
                rect[0] - min_gap,
                rect[1] - min_gap,
                rect[2] + min_gap,
                rect[3] + min_gap,
            )
            overlap = sum(
                _rectangle_overlap_area(padded, existing)
                for existing in occupied
            )
            distance = float(np.hypot(x - cx, y + text_height / 2 - cy))
            rank = (overlap, distance)
            if best is None or rank < best[0]:
                best = (rank, (x, y), rect)
            if overlap == 0:
                break
        assert best is not None
        _, position, rect = best
        occupied.append(
            (
                rect[0] - min_gap,
                rect[1] - min_gap,
                rect[2] + min_gap,
                rect[3] + min_gap,
            )
        )
        placements.append(((cx, cy), position, rect, label, color))
    return placements


def render_overlay(
    rgb_path: Path,
    frame: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    point_stride: int,
    point_radius: int,
    mask_alpha: int = 80,
) -> Image.Image:
    image = Image.open(rgb_path).convert("RGBA")
    width, height = image.size
    mask_layer = np.zeros((height, width, 4), dtype=np.uint8)
    bboxes: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]] = []
    labels: list[tuple[tuple[float, float], str, tuple[int, int, int]]] = []

    for obj in objects:
        color = object_color_for_id(obj.get("id"))
        points = load_object_points(frame, obj["id"])
        if len(points) == 0:
            continue
        if point_stride > 1:
            points = points[::point_stride]
        uv, valid = project_points(points, intrinsics, extrinsics)
        in_bounds = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        visible_uv = uv[in_bounds]
        if len(visible_uv) == 0:
            continue

        # Semi-transparent "mask": stamp a small translucent disk at every
        # reprojected point instead of a fully opaque dot. Overlapping disks
        # are combined with max() so dense point clouds read as one soft
        # translucent blob over the object rather than a cluster of solid dots.
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        for u, v in visible_uv:
            layer_draw.ellipse(
                [u - point_radius, v - point_radius, u + point_radius, v + point_radius],
                fill=(*color, mask_alpha),
            )
        mask_layer = np.maximum(mask_layer, np.asarray(layer))

        u_min, v_min = visible_uv.min(axis=0)
        u_max, v_max = visible_uv.max(axis=0)
        bboxes.append(((int(u_min), int(v_min), int(u_max), int(v_max)), color))

        centroid_uv, centroid_valid = project_points(np.asarray([obj["centroid_world"]], dtype=np.float64), intrinsics, extrinsics)
        if centroid_valid[0]:
            cu, cv = centroid_uv[0]
            labels.append(((float(cu), float(cv)), str(obj["id"]), color))

    image = Image.alpha_composite(image, Image.fromarray(mask_layer, mode="RGBA"))
    annotation_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(annotation_layer, "RGBA")
    for (u_min, v_min, u_max, v_max), color in bboxes:
        draw.rectangle(
            [u_min, v_min, u_max, v_max],
            outline=(*color, 175),
            width=1,
        )
    for (cu, cv), _, color in labels:
        r = 1.25  # 25% of the original 5px centroid marker radius.
        draw.ellipse(
            [cu - r, cv - r, cu + r, cv + r],
            outline=(255, 255, 255, 220),
            width=1,
        )
    for (cu, cv), (x, y), rect, label, color in layout_object_labels(
        draw,
        labels,
        width,
        height,
    ):
        label_center = (rect[0], (rect[1] + rect[3]) / 2)
        draw.line([(cu, cv), label_center], fill=(*color, 110), width=1)
        draw.text((x, y), label, fill=(*color, 230))

    image = Image.alpha_composite(image, annotation_layer)
    return image.convert("RGB")


def draw_overlay(
    rgb_path: Path,
    frame: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    point_stride: int,
    point_radius: int,
    out_path: Path,
    mask_alpha: int = 80,
) -> None:
    """Compatibility wrapper for callers that explicitly request one overlay."""
    image = render_overlay(
        rgb_path,
        frame,
        objects,
        intrinsics,
        extrinsics,
        point_stride,
        point_radius,
        mask_alpha,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def sanity_report_for_object(frame: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    points = load_object_points(frame, obj["id"])
    stored_centroid = np.asarray(obj["centroid_world"], dtype=np.float64)
    if len(points) > 0:
        recomputed_centroid = points.mean(axis=0)
        centroid_residual_m = float(
            np.linalg.norm(stored_centroid - recomputed_centroid)
        )
        centroid_to_cloud_distance_m = float(
            np.linalg.norm(points - stored_centroid, axis=1).min()
        )
    else:
        recomputed_centroid = np.full((3,), np.nan)
        centroid_residual_m = None
        centroid_to_cloud_distance_m = None

    per_camera_centroids: dict[str, list[np.ndarray]] = {}
    for obs in obj.get("observations", []):
        per_camera_centroids.setdefault(obs["camera"], []).append(np.asarray(obs["centroid_world"], dtype=np.float64))
    camera_centroids = {camera: np.mean(values, axis=0) for camera, values in per_camera_centroids.items()}

    max_spread = 0.0
    cameras = list(camera_centroids)
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            distance = float(np.linalg.norm(camera_centroids[cameras[i]] - camera_centroids[cameras[j]]))
            max_spread = max(max_spread, distance)

    bbox = np.asarray(obj["bbox3d_world"], dtype=np.float64)
    return {
        "id": obj["id"],
        "role_evidence": obj.get("role_evidence", {}),
        "num_points": len(points),
        "centroid_world": obj["centroid_world"],
        "recomputed_centroid_world": (
            recomputed_centroid.tolist() if len(points) > 0 else None
        ),
        "centroid_residual_m": centroid_residual_m,
        "centroid_to_cloud_distance_m": centroid_to_cloud_distance_m,
        "point_cloud_components": obj.get("point_cloud_components"),
        "bbox_size_m": (bbox[1] - bbox[0]).tolist(),
        "visible_camera": obj.get("visible_camera"),
        "cross_camera_centroid_spread_m": max_spread,
    }


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    """Force equal x/y/z scale on a 3D axes.

    matplotlib's 3D axes auto-scale each axis independently to fill the plot,
    so a physically flat/thin point cloud (small z range vs. x/y range, e.g.
    buttons on a table) gets visually stretched into a tall column even
    though the underlying world coordinates are correct. This makes the
    "height" look wrong in the rendered image despite the data being fine.
    """
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    half_range = max((maxs - mins).max() / 2.0, 1e-3)
    ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim(centers[2] - half_range, centers[2] + half_range)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass  # older matplotlib without set_box_aspect; limits above still help.


def render_pointcloud(
    frame: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]
) -> Image.Image:
    """Render the fused point cloud from several viewing angles into one PNG.

    A single default-angle 3D scatter is easy to misread (e.g. a flat object
    can look tall or vice versa depending on azimuth/elevation). Rendering a
    perspective + top/front/side view side-by-side in one image gives a
    quick, unambiguous multi-view sanity check without needing to rotate an
    interactive plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_points: list[np.ndarray] = []
    for obj in objects:
        points = load_object_points(frame, obj["id"])
        if len(points) > 0:
            all_points.append(points)
    combined_points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3))

    views: tuple[tuple[str, float, float], ...] = (
        ("perspective", 20.0, -60.0),
        # elev=89.9 (not exactly 90) avoids matplotlib's degenerate top-down
        # projection, which otherwise collapses the z-axis tick labels onto
        # a single point and renders them as an unreadable jumble.
        ("top (bird's-eye, XY)", 89.9, -90.0),
        ("front (XZ)", 0.0, -90.0),
        ("side (YZ)", 0.0, 0.0),
    )
    cols = 2
    rows = (len(views) + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 6 * rows))
    for view_index, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(rows, cols, view_index + 1, projection="3d")
        for obj in objects:
            points = load_object_points(frame, obj["id"])
            if len(points) == 0:
                continue
            color = np.array(object_color_for_id(obj.get("id"))) / 255.0
            evidence = obj.get("role_evidence", {})
            top_role = max(evidence, key=lambda role: evidence[role].get("probability", 0.0)) if evidence else None
            suffix = f" ({top_role} evidence)" if top_role else ""
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=color, label=f'{obj["id"]}{suffix}')
            centroid = np.asarray(obj["centroid_world"], dtype=np.float64)
            ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], s=90, marker="x", color=color)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        if title.startswith("top"):
            # Looking almost straight down, the z-axis is edge-on and its
            # tick labels overlap into an unreadable jumble; z isn't a
            # useful read in this view anyway, so hide the labels.
            ax.set_zticklabels([])
        if len(combined_points) > 0:
            set_axes_equal_3d(ax, combined_points)
        if view_index == 0:
            ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        return rendered.convert("RGB").copy()


def plot_pointcloud(
    frame: Mapping[str, Any], objects: Sequence[Mapping[str, Any]], out_path: Path
) -> None:
    """Compatibility wrapper for callers that explicitly request one plot."""
    image = render_pointcloud(frame, objects)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def compose_frame_montage(
    panels: Sequence[tuple[str, Image.Image]],
    frame_id: str,
    out_path: Path,
    columns: int = 2,
    cell_width: int = 512,
    label_height: int = 28,
    gap: int = 8,
) -> Image.Image:
    """Combine all selected camera overlays and the point cloud into one PNG."""
    if not panels:
        raise ValueError("Cannot compose a visualization montage without panels")
    columns = max(1, min(int(columns), len(panels)))
    rows = (len(panels) + columns - 1) // columns
    cell_width = max(64, int(cell_width))
    cell_height = cell_width
    header_height = 32
    canvas_width = columns * cell_width + (columns + 1) * gap
    canvas_height = (
        header_height
        + rows * (label_height + cell_height)
        + (rows + 1) * gap
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas_width, header_height), fill=(24, 24, 24))
    draw.text((10, 9), f"Frame {frame_id}: camera reprojections + fused point cloud", fill=(255, 255, 255))

    for index, (title, panel) in enumerate(panels):
        row, column = divmod(index, columns)
        x = gap + column * (cell_width + gap)
        y = header_height + gap + row * (label_height + cell_height + gap)
        draw.rectangle((x, y, x + cell_width, y + label_height), fill=(35, 35, 35))
        draw.text((x + 8, y + 7), title, fill=(255, 255, 255))
        fitted = ImageOps.contain(
            panel.convert("RGB"),
            (cell_width, cell_height),
            Image.Resampling.LANCZOS,
        )
        panel_x = x + (cell_width - fitted.width) // 2
        panel_y = y + label_height + (cell_height - fitted.height) // 2
        canvas.paste(fitted, (panel_x, panel_y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return canvas


def placeholder_panel(message: str, size: int = 512) -> Image.Image:
    """Create a visible panel so the fixed montage layout never collapses."""
    size = max(64, int(size))
    image = Image.new("RGB", (size, size), (224, 224, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(150, 150, 150), width=2)
    draw.multiline_text((16, 16), message, fill=(70, 70, 70), spacing=4)
    return image


def main() -> None:
    args = build_parser().parse_args()
    fused_path = Path(args.fused_json).expanduser().resolve()
    data = load_fused_manifest(fused_path)

    episode_dir = Path(args.episode_dir).expanduser().resolve() if args.episode_dir else Path(data["episode_metadata"]["episode_dir"])
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else fused_path.with_name("viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_params = load_camera_params(Path(args.camera_params_json).expanduser().resolve() if args.camera_params_json else None)
    rlbench_override = Path(args.rlbench_low_dim_obs).expanduser().resolve() if args.rlbench_low_dim_obs else None
    rlbench_observations = load_rlbench_observations(episode_dir, rlbench_override)

    frame_id_filter = parse_csv(args.frame_ids)
    cameras_filter = parse_csv(args.cameras)

    all_frames = list(iter_fused_frames(fused_path))
    fragment_result = (
        detect_suspect_fragment_aliases(all_frames)
        if args.hide_suspected_fragments
        else {"aliases": {}, "evidence": []}
    )
    suspect_aliases = dict(fragment_result["aliases"])
    frames = all_frames
    if frame_id_filter is not None:
        frames = [frame for frame in frames if str(frame["frame_id"]) in frame_id_filter]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fusion_visualization_report",
        "episode_dir": str(episode_dir),
        "source_fused_json": str(fused_path),
        "diagnostics_ref": "sanity_debug.json",
        "suspect_fragment_aliases": suspect_aliases,
        "frames": [],
    }
    debug_report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fusion_visualization_debug",
        "source_report_json": "sanity_report.json",
        "suspect_fragment_evidence": fragment_result["evidence"],
        "frames": [],
    }
    for frame in frames:
        frame_id = str(frame["frame_id"])
        frame_index = frame.get("frame_index")
        source_objects = list(frame.get("objects", []))
        frame_aliases = visible_suspect_aliases(source_objects, suspect_aliases)
        objects = [
            obj for obj in source_objects if str(obj.get("id")) not in frame_aliases
        ]
        target_cameras = (
            list(cameras_filter)
            if cameras_filter is not None
            else list(DEFAULT_MONTAGE_CAMERAS)
        )
        panels: list[tuple[str, Image.Image]] = []
        rendered_cameras: list[str] = []
        for camera in target_cameras:
            rgb_path = find_rgb_path(episode_dir, camera, frame_id)
            if rgb_path is None:
                print(f"[warn] frame_id={frame_id} camera={camera}: RGB image not found; skipping overlay.", file=sys.stderr)
                panels.append(
                    (
                        f"camera: {camera}",
                        placeholder_panel(
                            f"{camera}\nRGB image unavailable",
                            args.montage_cell_width,
                        ),
                    )
                )
                rendered_cameras.append(camera)
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
                print(f"[warn] frame_id={frame_id} camera={camera}: no camera intrinsics/extrinsics found; skipping overlay.", file=sys.stderr)
                panels.append((f"camera: {camera} (raw)", Image.open(rgb_path).convert("RGB")))
                rendered_cameras.append(camera)
                continue
            overlay = render_overlay(
                rgb_path,
                frame,
                objects,
                params["intrinsics"],
                params["extrinsics"],
                args.point_stride,
                args.point_radius,
                mask_alpha=args.mask_alpha,
            )
            panels.append((f"camera: {camera}", overlay))
            rendered_cameras.append(camera)

        includes_pointcloud = False
        if not args.skip_pointcloud:
            try:
                panels.append(("3D fused point cloud", render_pointcloud(frame, objects)))
                includes_pointcloud = True
            except ImportError:
                print("[warn] matplotlib not installed; skipping 3D point cloud plot (pip install matplotlib to enable).", file=sys.stderr)
                panels.append(
                    (
                        "3D fused point cloud",
                        placeholder_panel(
                            "point cloud\nmatplotlib unavailable",
                            args.montage_cell_width,
                        ),
                    )
                )

        montage_path = output_dir / f"{frame_id}_montage.png"
        if panels:
            compose_frame_montage(
                panels,
                frame_id,
                montage_path,
                columns=args.montage_columns,
                cell_width=args.montage_cell_width,
            )

        object_sanity = [sanity_report_for_object(frame, obj) for obj in objects]
        report["frames"].append({
            "frame_id": frame_id,
            "frame_index": frame_index,
            "montage_path": str(montage_path) if panels else None,
            "camera_panels": rendered_cameras,
            "includes_pointcloud": includes_pointcloud,
            "object_count": len(objects),
            "source_object_count": len(source_objects),
            "suppressed_suspect_ids": sorted(frame_aliases),
            "visible_suspect_aliases": frame_aliases,
        })
        debug_report["frames"].append({
            "frame_id": frame_id,
            "frame_index": frame_index,
            "objects": object_sanity,
            "suppressed_suspect_ids": sorted(frame_aliases),
        })

    report_path = output_dir / "sanity_report.json"
    report["summary"] = {
        "frame_count": len(report["frames"]),
        "montage_count": sum(
            bool(item.get("montage_path")) for item in report["frames"]
        ),
        "object_sample_count": sum(
            int(item.get("object_count", 0)) for item in report["frames"]
        ),
    }
    atomic_json_dump(debug_report, output_dir / "sanity_debug.json")
    atomic_json_dump(report, report_path)
    print(json.dumps({"output_dir": str(output_dir), "sanity_report": str(report_path), "montages_rendered": sum(bool(item.get("montage_path")) for item in report["frames"]), "frames_rendered": len(report["frames"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
