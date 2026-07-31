import json
import sys

from PIL import Image, ImageDraw

import visualize_fused_candidates as viz_module
from visualization_utils import OBJECT_COLORS, object_color_for_id
from visualize_fused_candidates import (
    _rectangle_overlap_area,
    compose_frame_montage,
    layout_object_labels,
    render_pointcloud,
)


def test_object_id_color_is_stable_across_frame_orderings():
    first_frame = ["O2", "O7", "O3"]
    second_frame = ["O3", "O2"]

    first_colors = {object_id: object_color_for_id(object_id) for object_id in first_frame}
    second_colors = {object_id: object_color_for_id(object_id) for object_id in second_frame}

    assert first_colors["O2"] == second_colors["O2"] == OBJECT_COLORS[1]
    assert first_colors["O3"] == second_colors["O3"] == OBJECT_COLORS[2]
    assert object_color_for_id("O14") != object_color_for_id("O2")


def test_dense_object_labels_are_placed_without_overlap():
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    labels = [
        ((32.0, 32.0), f"O{index}", (255, 0, 0))
        for index in range(1, 7)
    ]

    placements = layout_object_labels(draw, labels, 64, 64, min_gap=2)
    rectangles = [placement[2] for placement in placements]

    assert len(placements) == len(labels)
    for x0, y0, x1, y1 in rectangles:
        assert 0 <= x0 < x1 <= 64
        assert 0 <= y0 < y1 <= 64
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1 :]:
            assert _rectangle_overlap_area(first, second) == 0


def test_compose_frame_montage_puts_every_panel_in_one_image(tmp_path):
    panels = [
        ("front", Image.new("RGB", (32, 24), (255, 0, 0))),
        ("left", Image.new("RGB", (32, 24), (0, 255, 0))),
        ("point cloud", Image.new("RGB", (32, 32), (0, 0, 255))),
    ]
    output_path = tmp_path / "0_montage.png"

    montage = compose_frame_montage(
        panels, "0", output_path, columns=2, cell_width=64
    )

    assert output_path.is_file()
    assert montage.width > 128
    assert montage.height > 128


def test_pointcloud_is_rendered_to_memory_for_montage():
    obj = {
        "id": "O1",
        "centroid_world": [0.0, 0.0, 0.0],
        "points_world": [
            [-0.01, -0.01, 0.0],
            [0.01, -0.01, 0.0],
            [0.0, 0.01, 0.02],
        ],
        "role_evidence": {},
    }
    frame = {"frame_id": "0", "objects": [obj]}

    image = render_pointcloud(frame, [obj])

    assert image.mode == "RGB"
    assert image.width > 0
    assert image.height > 0


def test_main_writes_one_montage_without_separate_panels(tmp_path, monkeypatch):
    fused_path = tmp_path / "frame_fused_candidates.json"
    fused_path.write_text("{}")
    output_dir = tmp_path / "viz"
    frame = {
        "frame_id": "0",
        "frame_index": 0,
        "objects": [{"id": "O1", "visible_camera": ["front", "left"]}],
    }
    monkeypatch.setattr(
        viz_module,
        "load_fused_manifest",
        lambda path: {"episode_metadata": {"episode_dir": str(tmp_path)}},
    )
    monkeypatch.setattr(viz_module, "iter_fused_frames", lambda path: iter([frame]))
    monkeypatch.setattr(viz_module, "load_camera_params", lambda path: {})
    monkeypatch.setattr(viz_module, "load_rlbench_observations", lambda *args: [])
    monkeypatch.setattr(viz_module, "find_rgb_path", lambda *args: tmp_path / "raw.png")
    monkeypatch.setattr(
        viz_module,
        "resolve_camera_param_for_frame",
        lambda *args, **kwargs: {"intrinsics": None, "extrinsics": None},
    )
    monkeypatch.setattr(
        viz_module,
        "render_overlay",
        lambda *args, **kwargs: Image.new("RGB", (32, 32), (100, 120, 140)),
    )
    monkeypatch.setattr(
        viz_module,
        "render_pointcloud",
        lambda *args, **kwargs: Image.new("RGB", (32, 32), (20, 30, 40)),
    )
    monkeypatch.setattr(
        viz_module,
        "sanity_report_for_object",
        lambda *args, **kwargs: {"id": "O1"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_fused_candidates.py",
            "--fused-json",
            str(fused_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    viz_module.main()

    assert (output_dir / "0_montage.png").is_file()
    assert list(output_dir.glob("*_reproj.png")) == []
    assert list(output_dir.glob("*_pointcloud.png")) == []
    report = json.loads((output_dir / "sanity_report.json").read_text())
    assert report["frames"][0]["camera_panels"] == [
        "front",
        "left_shoulder",
        "right_shoulder",
    ]
    assert report["frames"][0]["includes_pointcloud"] is True
    with Image.open(output_dir / "0_montage.png") as montage:
        assert montage.width == 1048
        assert montage.height == 1136
