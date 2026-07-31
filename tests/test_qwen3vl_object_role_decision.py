import argparse
import json
import sys

from PIL import Image

import qwen3vl_object_role_decision as decision_module
from qwen3vl_object_role_decision import (
    _adaptive_refresh_reasons,
    _apply_two_stage_target_selection,
    _build_temporal_object_context,
    _candidate_observation_cards,
    _collect_temporal_contact_sheets,
    _current_dynamic_events,
    _decision_prompt,
    _filter_candidates,
    _pick_decision_frame,
    _resolve_temporal_frames,
    _run_decision_for_frame,
    _sanitize_decision_ids,
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


def test_current_dynamic_events_reads_tracker_event_field():
    context = {
        "objects": {
            "O1": {
                "events": [
                    {"frame_id": "f1", "event": "GRASPED"},
                    {"frame_id": "f2", "event": "RELEASED"},
                ]
            }
        }
    }

    assert _current_dynamic_events(context, "f2") == ["released"]


def test_adaptive_refresh_covers_disappearance_confidence_and_gripper_events():
    args = argparse.Namespace(
        decision_policy="adaptive",
        decision_refresh_interval=10,
        decision_min_propagation_confidence=0.70,
    )
    state = {
        "last_model_result": {"target_object_id": "O1"},
        "last_candidate_ids": {"O1", "O2"},
        "last_final_decision": {
            "target_object_id": "O1",
            "reference_object_id": "O2",
        },
        "frames_since_model": 0,
        "force_refresh": False,
    }

    reasons = _adaptive_refresh_reasons(
        args,
        state,
        {"O2"},
        {"target_object_id": "O2", "confidence": 0.5, "uncertain": False},
        ["released"],
        True,
    )

    assert "target_disappeared" in reasons
    assert "rule_candidate_changed" in reasons
    assert "low_propagation_confidence" in reasons
    assert "gripper_transition" in reasons
    assert "dynamic_event:released" in reasons


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
        self.last_generation_stats = {
            "attempts": 1,
            "input_tokens": 100,
            "output_tokens": 20,
        }
        return {
            "target_object_id": "O1",
            "reference_object_id": None,
            "confidence": 0.8,
        }, "mock"


def _run_main_with_grounder(
    tmp_path,
    monkeypatch,
    frames,
    grounder,
    *extra_args,
):
    manifest_path = tmp_path / "frame_fused_candidates.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "generation_id": "g1"})
    )
    summary_path = tmp_path / "object_summary.json"
    output_path = tmp_path / "decision.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": "g1",
                "source_fused_json": str(manifest_path),
                "frame_decision_inputs": frames,
                "object_tracks": [],
            }
        )
    )
    monkeypatch.setattr(
        decision_module,
        "Qwen3VLRLBenchGrounder",
        lambda **kwargs: grounder,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3vl_object_role_decision.py",
            "--object-summary-json",
            str(summary_path),
            "--output-json",
            str(output_path),
            *extra_args,
        ],
    )
    decision_module.main()
    return json.loads(output_path.read_text())


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
    assert all("candidate_objects" not in item for item in payload["window_frames"])


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
    assert all("online_history" not in _payload_from_call(call) for call in grounder.calls)


def test_adaptive_policy_emits_every_frame_but_refreshes_every_five(
    tmp_path, monkeypatch
):
    frames = _frames(8)
    for frame in frames:
        frame["candidate_objects"] = [
            {
                "object_id": "O1",
                "camera_count": 1,
                "point_count": 5,
                "sam_score": 0.9,
            }
        ]
    grounder = _MockGrounder()
    output = _run_main_with_grounder(
        tmp_path,
        monkeypatch,
        frames,
        grounder,
        "--decision-policy",
        "adaptive",
        "--decision-refresh-interval",
        "5",
        "--no-dynamic-role-reasoning",
        "--max-candidate-images",
        "0",
    )
    assert len(output["frame_decisions"]) == 8
    assert len(grounder.calls) == 2
    assert [
        item["frame_id"]
        for item in output["frame_decisions"]
        if item["model_invoked"]
    ] == ["f0", "f5"]
    assert output["performance"]["model_call_count"] == 2
    assert output["performance"]["propagated_frame_count"] == 6
    assert all(item["decision"] for item in output["frame_decisions"])


def test_adaptive_policy_refreshes_when_new_candidate_appears(
    tmp_path, monkeypatch
):
    frames = _frames(4)
    for frame in frames:
        frame["candidate_objects"] = [
            {"object_id": "O1", "camera_count": 1, "point_count": 5, "sam_score": 0.9}
        ]
    frames[2]["candidate_objects"].append(
        {"object_id": "O2", "camera_count": 1, "point_count": 5, "sam_score": 0.8}
    )
    grounder = _MockGrounder()
    output = _run_main_with_grounder(
        tmp_path,
        monkeypatch,
        frames,
        grounder,
        "--decision-policy",
        "adaptive",
        "--decision-refresh-interval",
        "10",
        "--no-dynamic-role-reasoning",
        "--max-candidate-images",
        "0",
    )
    assert len(grounder.calls) == 2
    assert output["frame_decisions"][2]["model_invoked"] is True
    assert "new_candidate" in output["frame_decisions"][2]["refresh_reasons"]


def test_adaptive_propagation_skips_contact_sheet_creation(tmp_path, monkeypatch):
    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (64, 64), (120, 40, 20)).save(crop_path)
    frames = _frames(2)
    for frame in frames:
        frame["candidate_objects"] = [
            {
                "object_id": "O1",
                "camera_count": 1,
                "point_count": 5,
                "sam_score": 0.9,
                "observations": [
                    {
                        "camera": "front",
                        "sam_score": 0.9,
                        "masked_crop_path": str(crop_path),
                    }
                ],
            }
        ]
    grounder = _MockGrounder()
    output = _run_main_with_grounder(
        tmp_path,
        monkeypatch,
        frames,
        grounder,
        "--decision-policy",
        "adaptive",
        "--decision-refresh-interval",
        "10",
        "--no-dynamic-role-reasoning",
    )
    assert len(output["frame_decisions"][0]["representative_images"]) == 1
    assert output["frame_decisions"][1]["decision_source"] == "temporal_propagation"
    assert output["frame_decisions"][1]["representative_images"] == []
    assert output["frame_decisions"][1]["performance"]["input_visual_pixels"] == 0


def test_adaptive_parse_error_forces_next_frame_refresh(tmp_path, monkeypatch):
    class FailOnceGrounder(_MockGrounder):
        def generate_json(self, messages, max_new_tokens):
            if not self.calls:
                self.calls.append((messages, max_new_tokens))
                self.last_generation_stats = {"attempts": 1}
                raise RuntimeError("malformed JSON")
            return super().generate_json(messages, max_new_tokens)

    frames = _frames(3)
    for frame in frames:
        frame["candidate_objects"] = [
            {"object_id": "O1", "camera_count": 1, "point_count": 5, "sam_score": 0.9}
        ]
    grounder = FailOnceGrounder()
    output = _run_main_with_grounder(
        tmp_path,
        monkeypatch,
        frames,
        grounder,
        "--decision-policy",
        "adaptive",
        "--decision-refresh-interval",
        "10",
        "--no-dynamic-role-reasoning",
        "--max-candidate-images",
        "0",
    )
    assert output["frame_decisions"][0]["decision_source"] == "qwen_error"
    assert output["frame_decisions"][0]["decision"]["uncertain"] is True
    assert output["frame_decisions"][1]["model_invoked"] is True
    assert "previous_model_parse_error" in output["frame_decisions"][1]["refresh_reasons"]


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


def test_contact_sheet_limits_views_and_visual_pixels(tmp_path):
    image_paths = []
    for index in range(3):
        path = tmp_path / f"camera_{index}.png"
        Image.new("RGB", (128, 128), (40 * index, 80, 120)).save(path)
        image_paths.append(path)
    frame = _frames(1)[0]
    frame["candidate_objects"] = [
        {
            "object_id": "O1",
            "observations": [
                {
                    "camera": f"camera_{index}",
                    "sam_score": 0.9 - index * 0.1,
                    "masked_crop_path": str(path),
                }
                for index, path in enumerate(image_paths)
            ],
        },
        {
            "object_id": "O2",
            "observations": [
                {
                    "camera": "camera_0",
                    "sam_score": 0.95,
                    "masked_crop_path": str(image_paths[0]),
                }
            ],
        },
    ]

    sheets = _collect_temporal_contact_sheets(
        [frame],
        tmp_path / "artifacts",
        3,
        {},
        candidate_views_per_object=1,
        max_visual_pixels=4096,
        allowed_object_ids={"O1"},
    )

    assert sheets[0]["object_ids"] == ["O1"]
    assert sheets[0]["pixel_count"] <= 4096
    assert Image.open(sheets[0]["image_path"]).size == (
        sheets[0]["width"],
        sheets[0]["height"],
    )


def test_candidate_contact_sheet_uses_two_distinct_best_camera_views(tmp_path):
    paths = []
    for name, color in (
        ("front.png", (180, 20, 20)),
        ("front_low.png", (120, 20, 20)),
        ("overhead.png", (20, 180, 20)),
    ):
        path = tmp_path / name
        Image.new("RGB", (24, 24), color).save(path)
        paths.append(path)
    candidate = {
        "object_id": "O2",
        "role_evidence": {
            "target": {"probability": 0.75},
            "reference": {"probability": 0.1},
        },
        "observations": [
            {"camera": "front", "sam_score": 0.9, "masked_crop_path": str(paths[0])},
            {"camera": "front", "sam_score": 0.7, "masked_crop_path": str(paths[1])},
            {"camera": "overhead", "sam_score": 0.8, "masked_crop_path": str(paths[2])},
        ],
    }

    cards = _candidate_observation_cards(candidate)

    assert [item["camera"] for item in cards] == ["front", "overhead"]
    assert [item["object_id"] for item in cards] == ["O2", "O2"]
    assert all(item["target_prior"] == 0.75 for item in cards)


def test_candidate_cap_prioritizes_semantic_target_evidence():
    candidates = [
        {
            "object_id": "O1",
            "camera_count": 4,
            "point_count": 500,
            "sam_score": 0.99,
            "role_evidence": {"target": {"probability": 0.02}},
        },
        {
            "object_id": "O2",
            "camera_count": 1,
            "point_count": 20,
            "sam_score": 0.5,
            "role_evidence": {"target": {"probability": 0.9}},
        },
    ]
    args = argparse.Namespace(
        min_candidate_point_count=0,
        min_candidate_camera_count=1,
        min_candidate_sam_score=0.0,
        max_ee_distance_m=None,
        max_candidates_for_decision=1,
    )

    kept, stats = _filter_candidates(candidates, args)

    assert [item["object_id"] for item in kept] == ["O2"]
    assert stats["kept_candidates"] == 1


def test_candidate_cap_uses_current_gripper_distance_inside_target_gate():
    candidates = [
        {
            "object_id": "O2",
            "camera_count": 2,
            "point_count": 100,
            "sam_score": 0.9,
            "role_evidence": {"target": {"probability": 0.8}},
        },
        {
            "object_id": "O3",
            "camera_count": 2,
            "point_count": 100,
            "sam_score": 0.9,
            "role_evidence": {"target": {"probability": 0.7}},
        },
    ]
    context = {
        "O2": {"target_proximity_cues": {"current_distance_m": 0.3}},
        "O3": {"target_proximity_cues": {"current_distance_m": 0.1}},
    }
    args = argparse.Namespace(
        min_candidate_point_count=0,
        min_candidate_camera_count=1,
        min_candidate_sam_score=0.0,
        max_ee_distance_m=None,
        max_candidates_for_decision=1,
    )

    kept, _ = _filter_candidates(candidates, args, context)

    assert [item["object_id"] for item in kept] == ["O3"]


def test_two_stage_selection_ignores_nearest_instruction_incompatible_object():
    candidates = [
        {"object_id": "O1", "role_evidence": {"target": {"probability": 0.95}}},
        {"object_id": "O2", "role_evidence": {"target": {"probability": 0.8}}},
        {"object_id": "O3", "role_evidence": {"target": {"probability": 0.7}}},
    ]
    context = {
        "O1": {"target_proximity_cues": {"current_distance_m": 0.02}},
        "O2": {
            "target_proximity_cues": {
                "current_distance_m": 0.25,
                "consistently_approaching": True,
                "approaching_step_fraction": 1.0,
                "approach_delta_m": 0.1,
            }
        },
        "O3": {
            "target_proximity_cues": {
                "current_distance_m": 0.15,
                "consistently_approaching": True,
                "approaching_step_fraction": 1.0,
                "approach_delta_m": 0.05,
            }
        },
    }
    model_result = {
        "instruction_compatible_object_ids": ["O2", "O3"],
        "target_object_id": "O2",
        "reference_object_id": None,
    }

    selected = _apply_two_stage_target_selection(model_result, candidates, context)

    assert selected["target_object_id"] == "O3"
    assert selected["model_target_object_id"] == "O2"
    assert "O1" not in selected["target_selection"]["candidate_order"]


def test_two_stage_selection_uses_approach_trend_when_current_distance_ties():
    candidates = [
        {"object_id": "O2", "role_evidence": {"target": {"probability": 0.8}}},
        {"object_id": "O3", "role_evidence": {"target": {"probability": 0.8}}},
    ]
    context = {
        "O2": {
            "target_proximity_cues": {
                "current_distance_m": 0.15,
                "consistently_approaching": True,
                "approaching_step_fraction": 1.0,
                "approach_delta_m": 0.08,
            }
        },
        "O3": {
            "target_proximity_cues": {
                "current_distance_m": 0.15,
                "consistently_approaching": False,
                "approaching_step_fraction": 0.5,
                "approach_delta_m": 0.02,
            }
        },
    }

    selected = _apply_two_stage_target_selection(
        {
            "instruction_compatible_object_ids": ["O3", "O2"],
            "target_object_id": "O3",
        },
        candidates,
        context,
    )

    assert selected["target_object_id"] == "O2"


def test_invalid_compatible_id_is_ignored_when_frame_has_no_candidates():
    selected = _apply_two_stage_target_selection(
        {
            "instruction_compatible_object_ids": ["O14"],
            "target_object_id": "O14",
            "reference_object_id": None,
            "confidence": 0.95,
        },
        [],
        {},
    )

    assert selected["instruction_compatible_object_ids"] == []
    assert selected["target_object_id"] is None
    assert selected["confidence"] == 0.0
    assert selected["uncertain"] is True
    assert selected["target_selection"]["ignored_invalid_object_ids"] == ["O14"]


def test_sanitizer_never_allows_same_target_and_reference():
    result = {
        "target_object_id": "O2",
        "reference_object_id": "O2",
    }

    invalid = _sanitize_decision_ids(result, {"O2"})

    assert result["target_object_id"] == "O2"
    assert result["reference_object_id"] is None
    assert invalid[-1]["reason"] == "same_as_target_object_id"


def test_empty_candidate_frame_skips_model_call():
    frames = _frames(1)
    args = argparse.Namespace(
        decision_window_frames=3,
        min_candidate_point_count=0,
        min_candidate_camera_count=1,
        min_candidate_sam_score=0.0,
        max_ee_distance_m=None,
        max_candidates_for_decision=12,
        use_decision_history=False,
        dry_run=False,
    )
    grounder = _MockGrounder()

    output = _run_decision_for_frame({}, frames, frames[0], args, grounder, [])

    assert grounder.calls == []
    assert output["model_skipped"] is True
    assert output["decision"]["target_object_id"] is None
    assert output["decision"]["confidence"] == 0.0
    assert output["decision"]["uncertain_reason"] == "no_valid_candidates_for_frame"


def test_temporal_context_computes_current_distance_and_approach_trend(
    tmp_path, monkeypatch
):
    frames = _frames(3)
    trajectory = [
        {
            "frame_id": f"f{index}",
            "frame_index": index,
            "centroid_world": [0.0, 0.0, 0.0],
            "visible_camera": ["front"],
            "camera_count": 1,
            "sam_score": 0.9,
            "point_count": 20,
            "mask_area": 10,
            "bbox3d_world": [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]],
        }
        for index in range(3)
    ]
    observations = [
        {"gripper_pose": [distance, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        for distance in (0.5, 0.3, 0.1)
    ]
    monkeypatch.setattr(
        decision_module,
        "load_rlbench_observations",
        lambda episode_dir, override: observations,
    )

    context, _ = _build_temporal_object_context(
        {
            "episode_dir": str(tmp_path),
            "object_tracks": [{"object_id": "O2", "trajectory": trajectory}],
        },
        frames,
    )

    cues = context["O2"]["target_proximity_cues"]
    assert cues["current_distance_m"] == 0.1
    assert cues["approach_delta_m"] == 0.4
    assert cues["approaching_step_fraction"] == 1.0
    assert cues["consistently_approaching"] is True


def test_decision_history_is_opt_in():
    frames = _frames(2)
    for frame in frames:
        frame["candidate_objects"] = [
            {"object_id": "O1", "camera_count": 1, "point_count": 5, "sam_score": 0.9}
        ]
    args = argparse.Namespace(
        decision_window_frames=3,
        min_candidate_point_count=0,
        min_candidate_camera_count=1,
        min_candidate_sam_score=0.0,
        max_ee_distance_m=None,
        max_candidates_for_decision=12,
        max_candidate_images=0,
        use_decision_history=True,
        dry_run=False,
        max_new_tokens=64,
    )
    previous = [{
        "frame_id": "f0",
        "frame_index": 0,
        "decision": {
            "target_object_id": "O1",
            "reference_object_id": None,
            "confidence": 0.8,
            "uncertain": False,
        },
    }]
    grounder = _MockGrounder()

    _run_decision_for_frame({}, frames, frames[1], args, grounder, previous)

    payload = _payload_from_call(grounder.calls[0])
    assert payload["online_history"][0]["frame_id"] == "f0"


def test_prompt_allows_confident_null_reference_for_unary_tasks():
    prompt = _decision_prompt("{}", [])
    assert "reference_object_id=null" in prompt
    assert "does not imply uncertainty" in prompt
    assert "visual identity cues from the instruction take precedence" in prompt
    assert "instruction_compatible_object_ids" in prompt
    assert "smallest current_distance_m" in prompt
