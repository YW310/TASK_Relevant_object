from PIL import Image, ImageDraw

from visualize_fused_candidates import (
    _rectangle_overlap_area,
    layout_object_labels,
)


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
