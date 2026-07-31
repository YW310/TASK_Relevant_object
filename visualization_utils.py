"""Small shared helpers for image annotations."""

from __future__ import annotations

import colorsys
import hashlib

from PIL import ImageFont


OBJECT_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
)


def object_color_for_id(object_id: object) -> tuple[int, int, int]:
    """Map an object ID to the same color across frames and processes."""
    text = str(object_id)
    if len(text) > 1 and text[0].upper() == "O" and text[1:].isdigit():
        color_index = max(0, int(text[1:]) - 1)
        if color_index < len(OBJECT_COLORS):
            return OBJECT_COLORS[color_index]

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / float(2**32)
    saturation = 0.65 + 0.20 * (digest[4] / 255.0)
    value = 0.82 + 0.15 * (digest[5] / 255.0)
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(round(channel * 255) for channel in rgb)


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
