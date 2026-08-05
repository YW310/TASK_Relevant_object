from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_is_idempotent_and_does_not_require_installation() -> None:
    from _legacy_bootstrap import SRC_ROOT, ensure_src_path

    source_path = str(SRC_ROOT)
    original = list(sys.path)
    try:
        sys.path[:] = [item for item in sys.path if item != source_path]
        assert ensure_src_path() == ROOT / "src"
        assert sys.path[0] == source_path
        ensure_src_path()
        assert sys.path.count(source_path) == 1
    finally:
        sys.path[:] = original


def test_legacy_exports_have_the_same_identity_as_package_exports() -> None:
    import camera_geometry
    import fusion_types
    import mask_geometry
    from relevant_object.domain import models
    from relevant_object.geometry import camera, mask

    assert fusion_types.Observation3D is models.Observation3D
    assert fusion_types.ROLE_NAMES is models.ROLE_NAMES
    assert camera_geometry.project_points is camera.project_points
    assert camera_geometry.backproject_mask is camera.backproject_mask
    assert mask_geometry.mask_iou is mask.mask_iou
    assert mask_geometry.split_mask_components is mask.split_mask_components
