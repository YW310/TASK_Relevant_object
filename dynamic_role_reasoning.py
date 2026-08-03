"""Task-agnostic temporal events and dynamic object-role reasoning.

The tracker is deliberately deterministic.  It consumes tracked O-ids,
gripper state and 3D boxes, then exposes evidence to Qwen and validates the
selected roles.  It never assigns semantic identity by itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from task_schema import TaskSchema


@dataclass(frozen=True)
class ReasoningThresholds:
    gripper_closed_threshold: float = 0.5
    grasp_distance_m: float = 0.06
    moving_distance_m: float = 0.01
    stable_distance_m: float = 0.008
    follow_cosine: float = 0.75
    follow_residual_m: float = 0.025
    placement_stable_frames: int = 2
    min_support_xy_overlap: float = 0.35
    min_support_vertical_gap_m: float = -0.01
    max_support_vertical_gap_m: float = 0.025
    min_containment_ratio: float = 0.5


@dataclass
class ObjectRuntimeState:
    phase: str = "available"
    held: bool = False
    pending_release: bool = False
    placed_on: str | None = None
    placed_in: str | None = None
    stable_frames: int = 0
    last_centroid: list[float] | None = None
    last_seen_frame_id: str | None = None
    last_selected_as_target: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


def _bbox(candidate: Mapping[str, Any]) -> np.ndarray | None:
    value = np.asarray(candidate.get("bbox3d_world", []), dtype=np.float64)
    if value.shape != (2, 3) or not np.isfinite(value).all():
        return None
    return value


def _centroid(candidate: Mapping[str, Any]) -> np.ndarray | None:
    value = np.asarray(candidate.get("centroid_world", []), dtype=np.float64).reshape(-1)
    if value.size < 3 or not np.isfinite(value[:3]).all():
        return None
    return value[:3]


def _intersection_length(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def pairwise_geometry(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Return symmetric bbox metrics plus directed support/containment cues."""
    source_bbox = _bbox(source)
    target_bbox = _bbox(target)
    source_center = _centroid(source)
    target_center = _centroid(target)
    result: dict[str, Any] = {
        "source_object_id": str(source.get("object_id") or source.get("id")),
        "target_object_id": str(target.get("object_id") or target.get("id")),
    }
    if source_center is not None and target_center is not None:
        delta = target_center - source_center
        result.update(
            {
                "distance_m": float(np.linalg.norm(delta)),
                "delta_world": delta.tolist(),
            }
        )
    if source_bbox is None or target_bbox is None:
        return result

    overlap_x = _intersection_length(
        source_bbox[0, 0], source_bbox[1, 0], target_bbox[0, 0], target_bbox[1, 0]
    )
    overlap_y = _intersection_length(
        source_bbox[0, 1], source_bbox[1, 1], target_bbox[0, 1], target_bbox[1, 1]
    )
    intersection_xy = overlap_x * overlap_y
    source_xy_area = max(
        0.0,
        float(
            (source_bbox[1, 0] - source_bbox[0, 0])
            * (source_bbox[1, 1] - source_bbox[0, 1])
        ),
    )
    target_xy_area = max(
        0.0,
        float(
            (target_bbox[1, 0] - target_bbox[0, 0])
            * (target_bbox[1, 1] - target_bbox[0, 1])
        ),
    )
    smaller_xy_area = min(source_xy_area, target_xy_area)
    overlap_ratio = (
        float(intersection_xy / smaller_xy_area) if smaller_xy_area > 1e-12 else 0.0
    )
    source_above_target_gap = float(source_bbox[0, 2] - target_bbox[1, 2])
    target_above_source_gap = float(target_bbox[0, 2] - source_bbox[1, 2])

    source_size = np.maximum(source_bbox[1] - source_bbox[0], 1e-12)
    intersection = np.maximum(
        0.0,
        np.minimum(source_bbox[1], target_bbox[1])
        - np.maximum(source_bbox[0], target_bbox[0]),
    )
    source_contained_fraction = float(np.prod(intersection) / np.prod(source_size))
    result.update(
        {
            "xy_overlap_ratio_of_smaller": overlap_ratio,
            "source_above_target_vertical_gap_m": source_above_target_gap,
            "target_above_source_vertical_gap_m": target_above_source_gap,
            "source_contained_fraction_in_target_bbox": source_contained_fraction,
        }
    )
    return result


def scene_relations(
    candidates: Sequence[Mapping[str, Any]],
    thresholds: ReasoningThresholds,
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for source in candidates:
        for target in candidates:
            source_id = str(source.get("object_id") or source.get("id"))
            target_id = str(target.get("object_id") or target.get("id"))
            if not source_id or source_id == target_id:
                continue
            relation = pairwise_geometry(source, target)
            overlap = float(relation.get("xy_overlap_ratio_of_smaller", 0.0))
            gap = relation.get("source_above_target_vertical_gap_m")
            supported = bool(
                gap is not None
                and overlap >= thresholds.min_support_xy_overlap
                and thresholds.min_support_vertical_gap_m
                <= float(gap)
                <= thresholds.max_support_vertical_gap_m
            )
            contained = bool(
                float(
                    relation.get("source_contained_fraction_in_target_bbox", 0.0)
                )
                >= thresholds.min_containment_ratio
            )
            relation["source_supported_by_target"] = supported
            relation["source_inside_target_bbox"] = contained
            if supported or contained:
                relations.append(relation)
    return relations


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


class DynamicRoleTracker:
    """Maintain persistent physical predicates while roles change per subgoal."""

    def __init__(
        self,
        schema: TaskSchema,
        thresholds: ReasoningThresholds | None = None,
    ) -> None:
        self.schema = schema
        self.thresholds = thresholds or ReasoningThresholds()
        self.states: dict[str, ObjectRuntimeState] = {}
        self.last_gripper_position: np.ndarray | None = None
        self.last_gripper_open: float | None = None
        self.current_frame_id: str | None = None
        self.current_relations: list[dict[str, Any]] = []
        self.current_grasp_candidate: str | None = None

    def _event(self, object_id: str, name: str, **details: Any) -> None:
        state = self.states.setdefault(object_id, ObjectRuntimeState())
        event = {"frame_id": self.current_frame_id, "event": name, **details}
        if not state.events or state.events[-1] != event:
            state.events.append(event)
            state.events[:] = state.events[-8:]

    def update(
        self,
        frame_input: Mapping[str, Any],
        temporal_context: Mapping[str, Mapping[str, Any]],
        gripper_position: np.ndarray | None,
        gripper_open: float | None,
        gripper_open_history: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        self.current_frame_id = str(frame_input.get("frame_id"))
        candidates = list(frame_input.get("candidate_objects", []))
        candidate_by_id = {
            str(item.get("object_id")): item
            for item in candidates
            if item.get("object_id") is not None
        }
        self.current_relations = scene_relations(candidates, self.thresholds)
        gripper_delta = (
            gripper_position - self.last_gripper_position
            if gripper_position is not None and self.last_gripper_position is not None
            else None
        )
        closed = (
            gripper_open is not None
            and gripper_open <= self.thresholds.gripper_closed_threshold
        )
        sampled_gripper_values = [
            float(value)
            for value in (gripper_open_history or [])
            if value is not None
        ]
        transition_values = (
            ([self.last_gripper_open] if self.last_gripper_open is not None else [])
            + sampled_gripper_values
        )
        opened_in_interval = any(
            transition_values[index - 1]
            <= self.thresholds.gripper_closed_threshold
            < transition_values[index]
            for index in range(1, len(transition_values))
        )
        opened_now = opened_in_interval or (
            gripper_open is not None
            and self.last_gripper_open is not None
            and self.last_gripper_open <= self.thresholds.gripper_closed_threshold
            and gripper_open > self.thresholds.gripper_closed_threshold
        )

        grasp_scores: list[tuple[float, str, float, float]] = []
        displacements: dict[str, float | None] = {}
        for object_id, candidate in candidate_by_id.items():
            state = self.states.setdefault(object_id, ObjectRuntimeState())
            centroid = _centroid(candidate)
            previous = (
                np.asarray(state.last_centroid, dtype=np.float64)
                if state.last_centroid is not None
                else None
            )
            displacement = (
                float(np.linalg.norm(centroid - previous))
                if centroid is not None and previous is not None
                else None
            )
            displacements[object_id] = displacement
            if displacement is not None and displacement <= self.thresholds.stable_distance_m:
                state.stable_frames += 1
            elif displacement is not None:
                state.stable_frames = 0

            object_delta = (
                centroid - previous
                if centroid is not None and previous is not None
                else None
            )
            cues = temporal_context.get(object_id, {}).get(
                "target_proximity_cues", {}
            )
            distance = cues.get("current_distance_m")
            follow_cosine = 0.0
            follow_residual = float("inf")
            if object_delta is not None and gripper_delta is not None:
                follow_cosine = _cosine(object_delta, gripper_delta)
                follow_residual = float(np.linalg.norm(object_delta - gripper_delta))
            follows = bool(
                object_delta is not None
                and gripper_delta is not None
                and float(np.linalg.norm(object_delta))
                >= self.thresholds.moving_distance_m
                and (
                    follow_cosine >= self.thresholds.follow_cosine
                    or follow_residual <= self.thresholds.follow_residual_m
                )
            )
            if (
                closed
                and distance is not None
                and float(distance) <= self.thresholds.grasp_distance_m
                and (follows or state.last_selected_as_target)
            ):
                grasp_scores.append(
                    (
                        float(distance),
                        object_id,
                        follow_cosine,
                        follow_residual,
                    )
                )
            if centroid is not None:
                state.last_centroid = centroid.tolist()
                state.last_seen_frame_id = self.current_frame_id

        self.current_grasp_candidate = None
        if not grasp_scores and opened_in_interval:
            inferred = [
                object_id
                for object_id, state in self.states.items()
                if object_id in candidate_by_id
                and state.last_selected_as_target
                and displacements.get(object_id) is not None
                and float(displacements[object_id])
                >= self.thresholds.moving_distance_m
            ]
            if len(inferred) == 1:
                object_id = inferred[0]
                state = self.states[object_id]
                state.held = True
                state.phase = "grasped"
                self.current_grasp_candidate = object_id
                self._event(object_id, "GRASPED", inferred_between_sampled_frames=True)
        if grasp_scores:
            _, object_id, follow_cosine, follow_residual = min(grasp_scores)
            state = self.states[object_id]
            self.current_grasp_candidate = object_id
            if not state.held:
                self._event(
                    object_id,
                    "GRASPED",
                    follow_cosine=follow_cosine,
                    follow_residual_m=follow_residual,
                )
            state.held = True
            state.pending_release = False
            state.phase = "grasped"
            state.placed_on = None
            state.placed_in = None

        for object_id, state in self.states.items():
            if state.held and opened_now:
                state.held = False
                state.pending_release = True
                state.phase = "released_pending_stability"
                state.stable_frames = 0
                self._event(object_id, "RELEASED")

        supported_by: dict[str, list[dict[str, Any]]] = {}
        inside: dict[str, list[dict[str, Any]]] = {}
        for relation in self.current_relations:
            source_id = str(relation["source_object_id"])
            if relation.get("source_supported_by_target"):
                supported_by.setdefault(source_id, []).append(relation)
            if relation.get("source_inside_target_bbox"):
                inside.setdefault(source_id, []).append(relation)

        for object_id, state in self.states.items():
            if not state.pending_release:
                continue
            if state.stable_frames < max(1, self.thresholds.placement_stable_frames):
                continue
            support_options = supported_by.get(object_id, [])
            containment_options = inside.get(object_id, [])
            if self.schema.goal_predicate == "ON" and support_options:
                best = max(
                    support_options,
                    key=lambda item: (
                        float(item.get("xy_overlap_ratio_of_smaller", 0.0)),
                        -abs(
                            float(
                                item.get(
                                    "source_above_target_vertical_gap_m",
                                    float("inf"),
                                )
                            )
                        ),
                    ),
                )
                state.placed_on = str(best["target_object_id"])
                state.phase = "placed_support"
                state.pending_release = False
                self._event(
                    object_id,
                    "PLACED_ON",
                    reference_object_id=state.placed_on,
                )
            elif self.schema.goal_predicate in {"IN", "INSERTED_IN"} and containment_options:
                best = max(
                    containment_options,
                    key=lambda item: float(
                        item.get("source_contained_fraction_in_target_bbox", 0.0)
                    ),
                )
                state.placed_in = str(best["target_object_id"])
                state.phase = "placed_in_goal"
                state.pending_release = False
                self._event(
                    object_id,
                    "PLACED_IN",
                    reference_object_id=state.placed_in,
                )

        if gripper_position is not None:
            self.last_gripper_position = np.asarray(
                gripper_position, dtype=np.float64
            )
        if gripper_open is not None:
            self.last_gripper_open = float(gripper_open)
        return self.context(candidate_by_id)

    def record_decision(self, decision: Mapping[str, Any]) -> None:
        target = (
            str(decision.get("target_object_id"))
            if decision.get("target_object_id") is not None
            else None
        )
        for object_id, state in self.states.items():
            state.last_selected_as_target = object_id == target

    def context(
        self, candidate_by_id: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        visible_ids = set(candidate_by_id)
        def bbox_top(object_id: str) -> float:
            bbox = _bbox(candidate_by_id[object_id])
            return float(bbox[1, 2]) if bbox is not None else float("-inf")

        reference_candidates = [
            object_id
            for object_id in visible_ids
            if self.states.get(object_id, ObjectRuntimeState()).phase
            == "placed_support"
        ]
        reference_candidates.sort(
            key=lambda object_id: (bbox_top(object_id), object_id),
            reverse=True,
        )
        return {
            "schema": self.schema.to_dict(),
            "gripper": {
                "open": self.last_gripper_open,
                "grasp_candidate_object_id": self.current_grasp_candidate,
            },
            "objects": {
                object_id: {
                    **asdict(self.states.setdefault(object_id, ObjectRuntimeState())),
                    "events": list(
                        self.states[object_id].events[-3:]
                    ),
                }
                for object_id in sorted(visible_ids)
            },
            "scene_relations": self.current_relations,
            "reference_candidate_ids": reference_candidates,
        }
def apply_dynamic_role_selection(
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    role_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate dynamic roles while preserving the model's semantic decision."""
    selected = dict(decision)
    valid_ids = {
        str(candidate.get("object_id"))
        for candidate in candidates
        if candidate.get("object_id") is not None
    }
    states = role_context.get("objects", {})
    schema = role_context.get("schema", {})
    original_target = selected.get("target_object_id")
    original_reference = selected.get("reference_object_id")

    compatible_ids = [
        str(object_id)
        for object_id in selected.get("instruction_compatible_object_ids", [])
        if str(object_id) in valid_ids
    ]
    completed_phases = {"placed_support", "placed_in_goal"}
    eligible_targets = [
        object_id
        for object_id in compatible_ids
        if states.get(object_id, {}).get("phase") not in completed_phases
    ]
    if (
        schema.get("repeat_policy") == "repeat_until_satisfied"
        and original_target not in eligible_targets
        and eligible_targets
    ):
        selected["target_object_id"] = eligible_targets[0]

    reference_candidates = [
        str(object_id)
        for object_id in role_context.get("reference_candidate_ids", [])
        if str(object_id) in valid_ids
        and str(object_id) != str(selected.get("target_object_id"))
    ]
    if schema.get("goal_predicate") == "ON" and reference_candidates:
        selected["reference_object_id"] = reference_candidates[0]

    if selected.get("target_object_id") == selected.get("reference_object_id"):
        selected["reference_object_id"] = next(
            (
                object_id
                for object_id in reference_candidates
                if object_id != selected.get("target_object_id")
            ),
            None,
        )

    selected["dynamic_role_selection"] = {
        "strategy": "task_schema_scene_graph_events_v1",
        "model_target_object_id": original_target,
        "model_reference_object_id": original_reference,
        "eligible_target_ids": eligible_targets,
        "reference_candidate_ids": reference_candidates,
        "target_overridden": selected.get("target_object_id") != original_target,
        "reference_overridden": selected.get("reference_object_id")
        != original_reference,
    }
    return selected


def _candidate_role_score(candidate: Mapping[str, Any], role: str) -> float | None:
    evidence = candidate.get("role_evidence", {})
    if not isinstance(evidence, Mapping) or role not in evidence:
        return None
    value = evidence[role]
    if isinstance(value, Mapping):
        value = next(
            (
                value[key]
                for key in ("probability", "score", "score_mass")
                if value.get(key) is not None
            ),
            None,
        )
    try:
        return min(1.0, max(0.0, float(value))) if value is not None else None
    except (TypeError, ValueError):
        return None


def calibrate_decision_confidence(
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    temporal_context: Mapping[str, Mapping[str, Any]],
    role_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine model, semantic, tracking, interaction and relation evidence."""
    selected = dict(decision)
    try:
        model_confidence = min(
            1.0,
            max(
                0.0,
                float(
                    selected.get(
                        "model_confidence", selected.get("confidence", 0.0)
                    )
                ),
            ),
        )
    except (TypeError, ValueError):
        model_confidence = 0.0
    target_id = selected.get("target_object_id")
    candidate_by_id = {
        str(candidate.get("object_id")): candidate
        for candidate in candidates
        if candidate.get("object_id") is not None
    }
    target = candidate_by_id.get(str(target_id)) if target_id is not None else None
    if target is None:
        selected["model_confidence"] = model_confidence
        selected["confidence"] = 0.0
        selected["confidence_components"] = {
            "model": model_confidence,
            "semantic": 0.0,
            "tracking": 0.0,
            "interaction": 0.0,
            "relation": 0.0,
        }
        return selected

    semantic = _candidate_role_score(target, "target")
    if semantic is None:
        semantic = model_confidence
    sam_score = min(1.0, max(0.0, float(target.get("sam_score") or 0.0)))
    camera_support = min(1.0, max(0.0, float(target.get("camera_count") or 0.0) / 2.0))
    tracking = 0.5 * sam_score + 0.5 * camera_support

    object_state = role_context.get("objects", {}).get(str(target_id), {})
    phase = object_state.get("phase")
    if phase in {"grasped", "released_pending_stability", "placed_support", "placed_in_goal"}:
        interaction = {
            "grasped": 1.0,
            "released_pending_stability": 0.9,
            "placed_support": 0.95,
            "placed_in_goal": 0.95,
        }[phase]
    else:
        cues = temporal_context.get(str(target_id), {}).get(
            "target_proximity_cues", {}
        )
        current_distance = cues.get("current_distance_m")
        interaction = (
            float(np.exp(-max(0.0, float(current_distance)) / 0.15))
            if current_distance is not None
            else 0.5
        )
        if cues.get("consistently_approaching"):
            interaction = min(1.0, interaction + 0.15)

    reference_id = selected.get("reference_object_id")
    schema = role_context.get("schema", {})
    if reference_id is None:
        relation = 0.4 if schema.get("reference_required") else 1.0
    else:
        matching_relation = next(
            (
                item
                for item in role_context.get("scene_relations", [])
                if str(item.get("source_object_id")) == str(target_id)
                and str(item.get("target_object_id")) == str(reference_id)
            ),
            None,
        )
        if matching_relation is not None and (
            matching_relation.get("source_supported_by_target")
            or matching_relation.get("source_inside_target_bbox")
        ):
            relation = 1.0
        elif str(reference_id) in role_context.get("reference_candidate_ids", []):
            relation = 0.85
        else:
            relation = 0.7

    evidence_confidence = (
        0.35 * semantic
        + 0.20 * tracking
        + 0.25 * interaction
        + 0.20 * relation
    )
    calibrated = min(model_confidence, evidence_confidence)
    if selected.get("uncertain"):
        calibrated = min(calibrated, 0.5)
    selected["model_confidence"] = model_confidence
    selected["confidence"] = round(float(calibrated), 4)
    selected["confidence_components"] = {
        "model": round(model_confidence, 4),
        "semantic": round(float(semantic), 4),
        "tracking": round(float(tracking), 4),
        "interaction": round(float(interaction), 4),
        "relation": round(float(relation), 4),
    }
    return selected
