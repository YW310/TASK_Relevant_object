"""Small shared helpers for image annotations."""

from __future__ import annotations

from PIL import ImageFont


def color_for_index(index: int) -> tuple[int, int, int]:
    """Return a stable high-contrast color for a zero-based item index."""
    palette = (
        (255, 80, 80),
        (80, 220, 120),
        (80, 150, 255),
        (255, 190, 60),
        (190, 90, 255),
        (40, 220, 220),
        (255, 100, 190),
        (160, 210, 60),
    )
    return palette[index % len(palette)]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
