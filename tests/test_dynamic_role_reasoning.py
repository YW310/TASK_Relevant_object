import numpy as np

from dynamic_role_reasoning import (
    DynamicRoleTracker,
    ReasoningThresholds,
    apply_dynamic_role_selection,
    calibrate_decision_confidence,
    pairwise_geometry,
)
from task_schema import compile_task_schema


def _candidate(object_id, centroid, bbox):
    return {
        "object_id": object_id,
        "centroid_world": centroid,
        "bbox3d_world": bbox,
    }


def _context(**distances):
    return {
        object_id: {"target_proximity_cues": {"current_distance_m": distance}}
        for object_id, distance in distances.items()
    }


def test_task_schema_compiles_predicates_instead_of_task_names():
    stack = compile_task_schema("stack all of the blocks")
    drawer = compile_task_schema("open the bottom drawer using its handle")
    insert = compile_task_schema("insert the peg into the slot")
    button = compile_task_schema("push the maroon button")

    assert stack.goal_predicate == "ON"
    assert stack.repeat_policy == "repeat_until_satisfied"
    assert drawer.action_family == "articulate"
    assert drawer.interaction_part_role == "interaction_part"
    assert insert.goal_predicate == "INSERTED_IN"
    assert insert.reference_required is True
    assert button.action_family == "press"
    assert button.goal_predicate == "PRESSED"
    assert button.reference_required is False


def test_pairwise_geometry_reports_support_evidence():
    lower = _candidate(
        "O1", [0.0, 0.0, 0.025], [[-0.05, -0.05, 0.0], [0.05, 0.05, 0.05]]
    )
    upper = _candidate(
        "O2", [0.0, 0.0, 0.075], [[-0.04, -0.04, 0.05], [0.04, 0.04, 0.10]]
    )

    relation = pairwise_geometry(upper, lower)

    assert relation["xy_overlap_ratio_of_smaller"] == 1.0
    assert abs(relation["source_above_target_vertical_gap_m"]) < 1e-9


def test_placed_target_becomes_reference_for_next_repeated_on_subgoal():
    schema = compile_task_schema("stack all of the blocks")
    tracker = DynamicRoleTracker(
        schema,
        ReasoningThresholds(placement_stable_frames=1),
    )
    lower = _candidate(
        "O1", [0.0, 0.0, 0.025], [[-0.05, -0.05, 0.0], [0.05, 0.05, 0.05]]
    )
    moving = _candidate(
        "O2", [0.2, 0.0, 0.075], [[0.16, -0.04, 0.05], [0.24, 0.04, 0.10]]
    )
    placed = _candidate(
        "O2", [0.0, 0.0, 0.075], [[-0.04, -0.04, 0.05], [0.04, 0.04, 0.10]]
    )
    next_block = _candidate(
        "O3", [0.3, 0.0, 0.025], [[0.26, -0.04, 0.0], [0.34, 0.04, 0.05]]
    )

    tracker.update(
        {"frame_id": "f0", "candidate_objects": [lower, moving, next_block]},
        _context(O1=0.3, O2=0.0, O3=0.1),
        np.asarray([0.2, 0.0, 0.075]),
        1.0,
    )
    tracker.record_decision({"target_object_id": "O2"})
    tracker.update(
        {"frame_id": "f1", "candidate_objects": [lower, placed, next_block]},
        _context(O1=0.1, O2=0.0, O3=0.3),
        np.asarray([0.0, 0.0, 0.075]),
        0.0,
    )
    tracker.update(
        {"frame_id": "f2", "candidate_objects": [lower, placed, next_block]},
        _context(O1=0.1, O2=0.06, O3=0.3),
        np.asarray([0.0, 0.0, 0.135]),
        1.0,
    )
    role_context = tracker.update(
        {"frame_id": "f3", "candidate_objects": [lower, placed, next_block]},
        _context(O1=0.1, O2=0.06, O3=0.3),
        np.asarray([0.0, 0.0, 0.135]),
        1.0,
    )

    assert role_context["objects"]["O2"]["phase"] == "placed_support"
    assert role_context["objects"]["O2"]["placed_on"] == "O1"
    assert role_context["reference_candidate_ids"] == ["O2"]

    selected = apply_dynamic_role_selection(
        {
            "instruction_compatible_object_ids": ["O2", "O3"],
            "target_object_id": "O2",
            "reference_object_id": "O1",
        },
        [lower, placed, next_block],
        role_context,
    )

    assert selected["target_object_id"] == "O3"
    assert selected["reference_object_id"] == "O2"
    assert selected["dynamic_role_selection"]["target_overridden"] is True
    assert selected["dynamic_role_selection"]["reference_overridden"] is True


def test_grasp_release_between_sampled_frames_is_not_lost():
    tracker = DynamicRoleTracker(
        compile_task_schema("stack all blocks"),
        ReasoningThresholds(placement_stable_frames=1),
    )
    lower = _candidate(
        "O1", [0.0, 0.0, 0.025], [[-0.05, -0.05, 0.0], [0.05, 0.05, 0.05]]
    )
    before = _candidate(
        "O2", [0.2, 0.0, 0.075], [[0.16, -0.04, 0.05], [0.24, 0.04, 0.10]]
    )
    after = _candidate(
        "O2", [0.0, 0.0, 0.075], [[-0.04, -0.04, 0.05], [0.04, 0.04, 0.10]]
    )
    tracker.update(
        {"frame_id": "f0", "candidate_objects": [lower, before]},
        _context(O1=0.2, O2=0.0),
        np.asarray([0.2, 0.0, 0.075]),
        1.0,
        [1.0],
    )
    tracker.record_decision({"target_object_id": "O2"})
    context = tracker.update(
        {"frame_id": "f20", "candidate_objects": [lower, after]},
        _context(O1=0.2, O2=0.1),
        np.asarray([0.0, 0.0, 0.15]),
        1.0,
        [0.0, 0.0, 1.0],
    )
    assert context["objects"]["O2"]["phase"] == "released_pending_stability"
    assert [event["event"] for event in context["objects"]["O2"]["events"]] == [
        "GRASPED",
        "RELEASED",
    ]

    context = tracker.update(
        {"frame_id": "f40", "candidate_objects": [lower, after]},
        _context(O1=0.2, O2=0.1),
        np.asarray([0.0, 0.0, 0.15]),
        1.0,
        [1.0],
    )
    assert context["objects"]["O2"]["phase"] == "placed_support"


def test_confidence_is_calibrated_from_evidence_not_copied_from_model():
    candidate = {
        "object_id": "O2",
        "role_evidence": {"target": {"probability": 0.8}},
        "sam_score": 0.9,
        "camera_count": 2,
    }
    calibrated = calibrate_decision_confidence(
        {
            "target_object_id": "O2",
            "reference_object_id": None,
            "confidence": 0.95,
            "uncertain": False,
        },
        [candidate],
        {
            "O2": {
                "target_proximity_cues": {
                    "current_distance_m": 0.1,
                    "consistently_approaching": True,
                }
            }
        },
        {
            "schema": {"reference_required": False},
            "objects": {"O2": {"phase": "available"}},
            "scene_relations": [],
            "reference_candidate_ids": [],
        },
    )

    assert calibrated["model_confidence"] == 0.95
    assert 0.0 < calibrated["confidence"] < 0.95
    assert set(calibrated["confidence_components"]) == {
        "model",
        "semantic",
        "tracking",
        "interaction",
        "relation",
    }
