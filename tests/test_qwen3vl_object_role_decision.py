import argparse
import json
import sys

from PIL import Image

import qwen3vl_object_role_decision as decision_module
from qwen3vl_object_role_decision import (
    _collect_temporal_contact_sheets,
    _decision_prompt,
    _pick_decision_frame,
    _resolve_temporal_frames,
    _run_decision_for_frame,
)


def _frames(count=7):
    return [
        {
            "frame_id": f"f{i}",
            "frame_index": i,
            "candidate_objects": [],
            "pairwise_relations": [],
        }
        for i in range(count)
    ]


def _payload_from_call(call):
    prompt = call[0][0]["content"][-1]["text"]
    return json.loads(
        prompt.split("Input evidence JSON:\n", 1)[1].split(
            "\n\nChronological object contact sheets", 1
        )[0]
    )


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


def test_one_frame_call_payload_contains_complete_window_only():
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

    result = _run_decision_for_frame(summary, frames, frames[6], args, grounder, [])

    assert len(grounder.calls) == 1
    assert result["frame_id"] == "f6"
    payload = _payload_from_call(grounder.calls[0])
    assert payload["temporal_window"]["frame_ids"] == ["f4", "f5", "f6"]
    assert [item["frame_id"] for item in payload["window_frames"]] == ["f4", "f5", "f6"]
    assert payload["window_frames"][-1]["is_decision_frame"] is True
    assert payload["valid_output_object_ids"] == ["O1"]
    window_samples = payload["object_track_context"]["O1"]["window_samples"]
    assert [sample["frame_id"] for sample in window_samples] == ["f4", "f5", "f6"]
    serialized = json.dumps(payload)
    assert '"f3"' not in serialized
    assert '"f7"' not in serialized
    assert '"f8"' not in serialized
    assert "lifespan_frames" not in serialized
    assert "episode_wide_camera" not in serialized
    assert payload["object_track_context"]["O1"]["window_motion_path_length_m"] == 2.0


def test_main_calls_grounder_once_per_frame_with_rolling_windows(tmp_path, monkeypatch):
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

    assert len(grounder.calls) == 8
    assert [_payload_from_call(call)["temporal_window"]["frame_ids"] for call in grounder.calls[:4]] == [
        ["f0"],
        ["f0", "f1"],
        ["f0", "f1", "f2"],
        ["f1", "f2", "f3"],
    ]
    output = json.loads((tmp_path / "decision.json").read_text())
    assert output["decision_scope"] == "all"
    assert output["decision_frame_id"] == "f7"
    assert output["decision_frame_index"] == 7
    assert len(output["frame_decisions"]) == 8
    assert [item["online_step"] for item in output["frame_decisions"]] == list(range(8))
    assert output["decision"] == output["frame_decisions"][-1]["decision"]


def test_single_scope_keeps_explicit_debug_behavior(tmp_path, monkeypatch):
    frames = _frames(4)
    for frame in frames:
        frame["candidate_objects"] = [
            {"object_id": "O1", "camera_count": 1, "point_count": 5, "sam_score": 0.9}
        ]
    manifest_path = tmp_path / "frame_fused_candidates.json"
    manifest_path.write_text(json.dumps({"schema_version": 3, "generation_id": "g1"}))
    summary_path = tmp_path / "object_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 3,
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
        "--decision-scope", "single",
        "--decision-frame-id", "f2",
    ])

    decision_module.main()

    assert len(grounder.calls) == 1
    output = json.loads((tmp_path / "decision.json").read_text())
    assert output["decision_scope"] == "single"
    assert output["decision_frame_id"] == "f2"


def test_temporal_contact_sheets_include_frame_and_object_ids(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (32, 32), (180, 20, 20)).save(image_a)
    Image.new("RGB", (32, 32), (20, 180, 20)).save(image_b)
    frames = _frames(2)
    frames[0]["candidate_objects"] = [{
        "object_id": "O1",
        "observations": [{"camera": "front", "sam_score": 0.8, "masked_crop_path": str(image_a)}],
    }]
    frames[1]["candidate_objects"] = [{
        "object_id": "O2",
        "observations": [{"camera": "overhead", "sam_score": 0.9, "masked_crop_path": str(image_b)}],
    }]

    sheets = _collect_temporal_contact_sheets(frames, tmp_path / "artifacts", 8, {})

    assert [item["frame_id"] for item in sheets] == ["f0", "f1"]
    assert [item["object_ids"] for item in sheets] == [["O1"], ["O2"]]
    assert all(Image.open(item["image_path"]).size[0] > 0 for item in sheets)


def test_prompt_allows_confident_null_reference_for_unary_tasks():
    prompt = _decision_prompt("{}", [])
    assert "reference_object_id=null" in prompt
    assert "does not imply uncertainty" in prompt
