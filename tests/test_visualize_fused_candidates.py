from PIL import Image, ImageDraw

from visualize_fused_candidates import (
    _rectangle_overlap_area,
    layout_object_labels,
)
from visualization_utils import OBJECT_COLORS, object_color_for_id


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
