from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fused_candidate_io import load_fused_frame, load_fused_manifest, load_object_points


FIXTURES = Path(__file__).with_name("fixtures")
GENERATION_ID = "00000000-0000-4000-8000-000000000001"


def test_minimal_artifact_fixtures_share_one_generation() -> None:
    manifest = load_fused_manifest(FIXTURES / "fusion" / "frame_fused_candidates.json")
    frame = load_fused_frame(manifest, "0")
    points = load_object_points(frame, "O1")
    assert manifest["generation_id"] == GENERATION_ID
    assert frame["generation_id"] == GENERATION_ID
    np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]])
    for name in ("object_summary.json", "object_predictions.json"):
        artifact = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        assert artifact["generation_id"] == GENERATION_ID


def test_candidate_fixture_keeps_the_stage_one_schema() -> None:
    artifact = json.loads(
        (FIXTURES / "episode_candidates.json").read_text(encoding="utf-8")
    )
    assert artifact["schema_version"] == 2
    assert artifact["artifact_type"] == "episode_candidates"


def test_manifest_relative_geometry_from_early_schema_v4_is_still_readable(tmp_path) -> None:
    frame_dir = tmp_path / "frames" / "000000_0"
    frame_dir.mkdir(parents=True)
    generation_id = "generation"
    np.savez_compressed(
        frame_dir / "fused_geometry.npz",
        **{"O1/points_world": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)},
    )
    (frame_dir / "fused_objects.json").write_text(json.dumps({
        "schema_version": 4,
        "generation_id": generation_id,
        "frame_id": "0",
        "objects": [{
            "id": "O1",
            "geometry_path": "frames/000000_0/fused_geometry.npz",
            "points_key": "O1/points_world",
            "point_count": 1,
        }],
    }), encoding="utf-8")
    manifest_path = tmp_path / "frame_fused_candidates.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 4,
        "generation_id": generation_id,
        "frames": [{
            "frame_id": "0",
            "status": "complete",
            "fused_objects_json": "frames/000000_0/fused_objects.json",
        }],
    }), encoding="utf-8")

    frame = load_fused_frame(load_fused_manifest(manifest_path), "0")
    np.testing.assert_allclose(load_object_points(frame, "O1"), [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize("legacy_version", [1, 2, 3])
def test_legacy_fused_manifests_are_explicitly_rejected(tmp_path, legacy_version) -> None:
    path = tmp_path / "frame_fused_candidates.json"
    path.write_text(
        json.dumps({"schema_version": legacy_version, "generation_id": "old", "frames": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="use a new output directory"):
        load_fused_manifest(path)
