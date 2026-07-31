import numpy as np
from PIL import Image

from stage4_visualize_decision import (
    _background_image,
    _display_label,
    _draw_decision_overlay,
)


def test_selected_object_label_replaces_internal_object_label():
    assert _display_label("O2", None) == "O2"
    assert _display_label("O2", "target") == "T2"
    assert _display_label("O2", "reference") == "R2"


def test_raw_rgb_is_used_for_rendering_even_when_stage3_overlay_exists(tmp_path):
    episode_dir = tmp_path / "episode0"
    rgb_dir = episode_dir / "front_rgb"
    rgb_dir.mkdir(parents=True)
    raw_path = rgb_dir / "0.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(raw_path)

    viz_dir = tmp_path / "viz"
    viz_dir.mkdir()
    stage3_path = viz_dir / "0_front_reproj.png"
    Image.new("RGB", (8, 8), (10, 10, 200)).save(stage3_path)

    image, render_path, comparison_path = _background_image(
        viz_dir, episode_dir, "front", "0"
    )

    assert image is not None
    assert image.getpixel((0, 0)) == (200, 10, 10, 255)
    assert render_path == str(raw_path)
    assert comparison_path == str(stage3_path)


def test_stage3_montage_is_used_as_comparison_without_separate_overlay(tmp_path):
    episode_dir = tmp_path / "episode0"
    rgb_dir = episode_dir / "front_rgb"
    rgb_dir.mkdir(parents=True)
    raw_path = rgb_dir / "0.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(raw_path)
    viz_dir = tmp_path / "viz"
    viz_dir.mkdir()
    montage_path = viz_dir / "0_montage.png"
    Image.new("RGB", (16, 16), (10, 10, 200)).save(montage_path)

    image, render_path, comparison_path = _background_image(
        viz_dir, episode_dir, "front", "0"
    )

    assert image is not None
    assert render_path == str(raw_path)
    assert comparison_path == str(montage_path)


def test_decision_label_does_not_draw_old_translucent_background_block():
    background = Image.new("RGB", (128, 128), (240, 240, 240))
    obj = {
        "id": "O2",
        "centroid_world": [0.0, 0.0, 1.0],
        "points_world": [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    }
    image = _draw_decision_overlay(
        background=background,
        frame={"frame_id": "0", "objects": [obj]},
        objects_by_id={"O2": obj},
        decisions=[("target", "O2")],
        intrinsics=np.asarray([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]),
        extrinsics=np.eye(4),
        point_stride=1,
        point_radius=1,
        mask_alpha=90,
    )

    # The old implementation filled x=20..100, y=2..20 behind the label.
    assert image.getpixel((80, 10)) == (240, 240, 240, 255)
    # The bbox is one pixel wide by default and is genuinely alpha-composited.
    assert image.getpixel((30, 39)) == (240, 240, 240, 255)
    bbox_pixel = image.getpixel((30, 40))
    assert bbox_pixel != (240, 240, 240, 255)
    assert bbox_pixel != (255, 80, 80, 255)
    assert bbox_pixel[3] == 255
