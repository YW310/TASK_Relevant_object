"""Reusable point-cloud geometry kernels."""

from __future__ import annotations

import numpy as np


def sampled_point_to_cloud_fraction(
    source_sample: np.ndarray,
    target_sample: np.ndarray,
    max_distance_m: float,
    *,
    chunk_size: int = 64,
) -> float:
    """Return the source fraction within a distance of the target cloud.

    Sampling and input-policy decisions remain with callers. This function owns
    only the shared numerical kernel so legacy edge-case behavior is preserved.
    """
    if (
        len(source_sample) == 0
        or len(target_sample) == 0
        or max_distance_m <= 0.0
    ):
        return 0.0
    threshold_sq = float(max_distance_m) ** 2
    close_count = 0
    step = max(1, int(chunk_size))
    for start in range(0, len(source_sample), step):
        chunk = source_sample[start : start + step]
        squared = np.sum(
            (chunk[:, None, :] - target_sample[None, :, :]) ** 2,
            axis=2,
        )
        close_count += int(np.sum(np.min(squared, axis=1) <= threshold_sq))
    return float(close_count / len(source_sample))


__all__ = ["sampled_point_to_cloud_fraction"]
