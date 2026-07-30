import argparse
import json
import sys

import qwen3vl_object_role_decision as decision_module
from qwen3vl_object_role_decision import (
    _pick_decision_frame,
    _resolve_temporal_frames,
    _run_decision_for_frame,
)


def _frames(count=7):
    return [{"frame_id": f"f{i}", "frame_index": i, "candidate_objects": []} for i in range(count)]


def test_temporal_window_at_middle_frame():
    frames = _frames()
    assert [item["frame_id"] for item in _resolve_temporal_frames(frames, frames[4], 3)] == ["f2", "f3", "f4"]


def test_temporal_window_at_last_frame():
    frames = _frames()
    assert [item["frame_id"] for item in _resolve_temporal_frames(frames, frames[-1], 3)] == ["f4", "f5", "f6"]


def test_temporal_window_truncates_at_episode_start():
    frames = _frames()
    assert [item["frame_id"] for item in _resolve_temporal_frames(frames, frames[1], 3)] == ["f0", "f1"]


def test_explicit_frame_id_anchors_window():
    frames = list(reversed(_frames()))
    anchor = _pick_decision_frame(frames, "last", "f3")
    assert anchor["frame_id"] == "f3"
    assert [item["frame_id"] for item in _resolve_temporal_frames(frames, anchor, 3)] == ["f1", "f2", "f3"]


class _MockGrounder:
    def __init__(self):
        self.calls = []

    def generate_json(self, messages, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        return {
            "target_object_id": "O1",
            "reference_object_id": None,
            "confidence": 0.8,
        }, "mock"


def test_one_model_call_and_payload_contains_only_window_frames():
    frames = _frames(9)
    for frame in frames:
        frame["candidate_objects"] = [{"object_id": "O1", "camera_count": 1, "point_count": 5, "sam_score": 0.9}]
    summary = {
        "object_tracks": [{
            "object_id": "O1",
            "lifespan_frames": 9,
            "camera_set": ["episode_wide_camera"],
            "motion_path_length_m": 99.0,
            "sam_score_stats": {"mean": 99.0},
            "trajectory": [{
                "frame_id": f"f{i}", "frame_index": i, "centroid_world": [float(i), 0, 0],
                "visible_camera": [f"camera_{i}"], "camera_count": 1, "point_count": i + 1,
                "sam_score": i / 10, "mask_area": 10 + i,
            } for i in range(9)],
        }],
    }
    args = argparse.Namespace(
        decision_window_frames=3, min_candidate_point_count=0, min_candidate_camera_count=1,
        min_candidate_sam_score=0.0, max_ee_distance_m=None, max_candidates_for_decision=12,
        max_candidate_images=0, dry_run=False, max_new_tokens=64,
    )
    grounder = _MockGrounder()

    result = _run_decision_for_frame(summary, frames, frames[6], args, grounder)

    assert len(grounder.calls) == 1
    assert result["frame_id"] == "f6"
    prompt = grounder.calls[0][0][0]["content"][-1]["text"]
    payload = json.loads(prompt.split("Input evidence JSON:\n", 1)[1].split("\n\nRepresentative candidate images", 1)[0])
    assert payload["temporal_window"]["frame_ids"] == ["f4", "f5", "f6"]
    window_samples = payload["object_track_context"]["O1"]["window_samples"]
    assert [sample["frame_id"] for sample in window_samples] == ["f4", "f5", "f6"]
    serialized = json.dumps(payload)
    assert '"f3"' not in serialized
    assert '"f7"' not in serialized
    assert '"f8"' not in serialized
    assert "lifespan_frames" not in serialized
    assert "episode_wide_camera" not in serialized
    assert payload["object_track_context"]["O1"]["window_motion_path_length_m"] == 2.0


def test_main_calls_grounder_once_for_multi_frame_episode(tmp_path, monkeypatch):
    frames = _frames(8)
    for frame in frames:
        frame["candidate_objects"] = [{"object_id": "O1", "camera_count": 1, "point_count": 5, "sam_score": 0.9}]
    manifest_path = tmp_path / "frame_fused_candidates.json"
    manifest_path.write_text(json.dumps({"schema_version": "v1", "generation_id": "g1"}))
    summary_path = tmp_path / "object_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": "v1",
        "generation_id": "g1",
        "source_fused_json": str(manifest_path),
        "frame_decision_inputs": frames,
        "object_tracks": [],
    }))
    grounder = _MockGrounder()
    monkeypatch.setattr(decision_module, "Qwen3VLRLBenchGrounder", lambda **kwargs: grounder)
    monkeypatch.setattr(sys, "argv", [
        "qwen3vl_object_role_decision.py",
        "--object-summary-json", str(summary_path),
        "--output-json", str(tmp_path / "decision.json"),
    ])

    decision_module.main()

    assert len(grounder.calls) == 1
    output = json.loads((tmp_path / "decision.json").read_text())
    assert output["decision_frame_id"] == "f7"
    assert output["decision_frame_index"] == 7
    assert len(output["frame_decisions"]) == 8
    assert [item["frame_id"] for item in output["frame_decisions"]] == [f"f{i}" for i in range(8)]
    assert all(item["decision_source_frame_id"] == "f7" for item in output["frame_decisions"])
    assert all(item["decision"]["target_object_id"] == "O1" for item in output["frame_decisions"])
