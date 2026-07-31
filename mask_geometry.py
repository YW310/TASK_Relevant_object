"""Shared binary-mask and bounding-box geometry helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    iou, _ = mask_overlap_metrics(mask_a, mask_b)
    return iou


def mask_overlap_metrics(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[float, float]:
    """Return mask IoU and coverage of the smaller mask."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        return 0.0, 0.0
    intersection = int(np.logical_and(a, b).sum())
    area_a, area_b = int(a.sum()), int(b.sum())
    union = area_a + area_b - intersection
    smaller = min(area_a, area_b)
    return (
        intersection / union if union > 0 else 0.0,
        intersection / smaller if smaller > 0 else 0.0,
    )


def bbox_iou_2d(
    a: Sequence[float] | None,
    b: Sequence[float] | None,
) -> float:
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def split_mask_components(
    mask: np.ndarray,
    min_area: int,
    max_components: int = 0,
) -> list[np.ndarray]:
    """Return significant 8-connected regions of a binary mask."""
    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {foreground.shape}.")

    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    components: list[np.ndarray] = []
    min_area = max(1, int(min_area))

    for start_y, start_x in np.argwhere(foreground):
        y0, x0 = int(start_y), int(start_x)
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(y0, x0)]
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and foreground[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(pixels) < min_area:
            continue
        component = np.zeros_like(foreground, dtype=bool)
        ys, xs = zip(*pixels)
        component[np.asarray(ys), np.asarray(xs)] = True
        components.append(component)

    components.sort(key=lambda item: int(item.sum()), reverse=True)
    if max_components > 0:
        components = components[: int(max_components)]
    return components


def bbox_max_axis_size_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Return the largest meaningful axis-wise 3D bbox size ratio."""
    sizes_a = np.maximum(np.asarray(a[1]) - np.asarray(a[0]), 1e-6)
    sizes_b = np.maximum(np.asarray(b[1]) - np.asarray(b[0]), 1e-6)
    valid_axes = np.maximum(sizes_a, sizes_b) >= 1e-3
    if not np.any(valid_axes):
        return 1.0
    ratios = np.maximum(
        sizes_a[valid_axes] / sizes_b[valid_axes],
        sizes_b[valid_axes] / sizes_a[valid_axes],
    )
    return float(ratios.max())
