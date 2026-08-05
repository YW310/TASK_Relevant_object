"""Compatibility exports for legacy imports; use relevant_object.geometry."""

from _legacy_bootstrap import ensure_src_path


ensure_src_path()

from relevant_object.geometry.mask import (
    bbox_iou_2d,
    bbox_max_axis_size_ratio,
    mask_bbox,
    mask_iou,
    mask_overlap_metrics,
    split_mask_components,
)


__all__ = [
    "bbox_iou_2d",
    "bbox_max_axis_size_ratio",
    "mask_bbox",
    "mask_iou",
    "mask_overlap_metrics",
    "split_mask_components",
]
