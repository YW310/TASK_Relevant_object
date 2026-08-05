from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "episode_candidates"

