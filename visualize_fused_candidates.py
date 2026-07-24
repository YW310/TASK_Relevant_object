#!/usr/bin/env python3
"""Visualize ``frame_fused_candidates.json`` for sanity-checking multi-view 3D fusion.

Produces, per selected frame:

1. Per-camera RGB reprojection overlays. Every fused object's world points
   are re-projected back onto each requested camera view (even cameras that
   did not contribute candidates to that object). If the camera
   intrinsics/extrinsics are correct, the dots should land on the same
   physical object in every view.
2. A 3D point-cloud scatter plot (matplotlib) of every fused object's
   points/centroid in world coordinates, for a quick bird's-eye/side sanity
   check (e.g. objects should sit near the table plane, not scattered).
3. A ``sanity_report.json`` with, per fused object: point count, bbox size,
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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from multiview_candidate_fusion import (
    load_camera_params,
    load_rlbench_observations,
    parse_csv,
    resolve_camera_param_for_frame,
)

# Distinct, high-contrast colors cycled across fused objects within a frame.
OBJECT_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
)


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
    return parser


def find_rgb_path(episode_dir: Path, camera: str, frame_id: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = episode_dir / f"{camera}_rgb" / f"{frame_id}{suffix}"
        if path.is_file():
            return path
    return None


def project_points(points_world: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into pixel coordinates. ``extrinsics`` is T_world_camera."""
    if len(points_world) == 0:
        return np.empty((0, 2)), np.empty((0,), dtype=bool)
    world_to_cam = np.linalg.inv(extrinsics)
    hom = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    points_cam = (hom @ world_to_cam.T)[:, :3]
    z = points_cam[:, 2]
    valid = z > 1e-6
    safe_z = np.where(valid, z, 1.0)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u = points_cam[:, 0] * fx / safe_z + cx
    v = points_cam[:, 1] * fy / safe_z + cy
    return np.stack([u, v], axis=1), valid


def draw_overlay(
    rgb_path: Path,
    objects: Sequence[Mapping[str, Any]],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    point_stride: int,
    point_radius: int,
    out_path: Path,
    mask_alpha: int = 80,
) -> None:
    image = Image.open(rgb_path).convert("RGBA")
    width, height = image.size
    mask_layer = np.zeros((height, width, 4), dtype=np.uint8)
    bboxes: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]] = []
    labels: list[tuple[tuple[float, float], str, tuple[int, int, int]]] = []

    for index, obj in enumerate(objects):
        color = OBJECT_COLORS[index % len(OBJECT_COLORS)]
        points = np.asarray(obj["points_world"], dtype=np.float64)
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
    draw = ImageDraw.Draw(image, "RGBA")
    for (u_min, v_min, u_max, v_max), color in bboxes:
        draw.rectangle([u_min, v_min, u_max, v_max], outline=(*color, 255), width=2)
    for (cu, cv), label, color in labels:
        r = 1.25  # 25% of the original 5px centroid marker radius.
        draw.ellipse([cu - r, cv - r, cu + r, cv + r], outline=(255, 255, 255, 255), width=1)
        draw.text((cu + 6, cv - 6), label, fill=(*color, 230))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)


def sanity_report_for_object(obj: Mapping[str, Any]) -> dict[str, Any]:
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
        "role": obj["role"],
        "num_points": len(obj.get("points_world", [])),
        "centroid_world": obj["centroid_world"],
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


def plot_pointcloud(objects: Sequence[Mapping[str, Any]], out_path: Path) -> None:
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
        points = np.asarray(obj.get("points_world", []), dtype=np.float64)
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
        for index, obj in enumerate(objects):
            points = np.asarray(obj.get("points_world", []), dtype=np.float64)
            if len(points) == 0:
                continue
            color = np.array(OBJECT_COLORS[index % len(OBJECT_COLORS)]) / 255.0
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color=color, label=f'{obj["id"]} ({obj["role"]})')
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    fused_path = Path(args.fused_json).expanduser().resolve()
    data = json.loads(fused_path.read_text(encoding="utf-8"))

    episode_dir = Path(args.episode_dir).expanduser().resolve() if args.episode_dir else Path(data["episode_dir"])
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else fused_path.with_name("viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_params = load_camera_params(Path(args.camera_params_json).expanduser().resolve() if args.camera_params_json else None)
    rlbench_override = Path(args.rlbench_low_dim_obs).expanduser().resolve() if args.rlbench_low_dim_obs else None
    rlbench_observations = load_rlbench_observations(episode_dir, rlbench_override)

    frame_id_filter = parse_csv(args.frame_ids)
    cameras_filter = parse_csv(args.cameras)

    frames = data.get("frames", [])
    if frame_id_filter is not None:
        frames = [frame for frame in frames if str(frame["frame_id"]) in frame_id_filter]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    report: dict[str, Any] = {"episode_dir": str(episode_dir), "source_fused_json": str(fused_path), "frames": []}
    for frame in frames:
        frame_id = str(frame["frame_id"])
        frame_index = frame.get("frame_index")
        objects = frame.get("objects", [])
        if not objects:
            continue

        available_cameras = sorted({camera for obj in objects for camera in obj.get("visible_camera", [])})
        target_cameras = cameras_filter if cameras_filter is not None else available_cameras
        for camera in target_cameras:
            rgb_path = find_rgb_path(episode_dir, camera, frame_id)
            if rgb_path is None:
                print(f"[warn] frame_id={frame_id} camera={camera}: RGB image not found; skipping overlay.", file=sys.stderr)
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
                continue
            out_path = output_dir / f"{frame_id}_{camera}_reproj.png"
            draw_overlay(
                rgb_path,
                objects,
                params["intrinsics"],
                params["extrinsics"],
                args.point_stride,
                args.point_radius,
                out_path,
                mask_alpha=args.mask_alpha,
            )

        if not args.skip_pointcloud:
            try:
                plot_pointcloud(objects, output_dir / f"{frame_id}_pointcloud.png")
            except ImportError:
                print("[warn] matplotlib not installed; skipping 3D point cloud plot (pip install matplotlib to enable).", file=sys.stderr)

        report["frames"].append({
            "frame_id": frame_id,
            "frame_index": frame_index,
            "objects": [sanity_report_for_object(obj) for obj in objects],
        })

    report_path = output_dir / "sanity_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "sanity_report": str(report_path), "frames_rendered": len(report["frames"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
