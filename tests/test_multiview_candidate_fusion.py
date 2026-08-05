import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from multiview_candidate_fusion import (
    Observation3D,
    assign_object_ids,
    build_candidate_lifecycle,
    build_object_summary,
    compact_candidate_outcomes,
    cluster_observations,
    filter_candidates_by_mask_area,
    filter_clusters_by_camera_support,
    filter_small_clusters,
    frame_index_from_frame,
    split_fused_frame_artifacts,
    suppress_same_camera_duplicates,
    validate_semantic_candidates,
    _load_completed_frame,
    _restore_track_state,
)


class CompactFusionArtifactTest(unittest.TestCase):
    def test_verbose_lifecycle_and_tracking_move_to_debug_artifact(self) -> None:
        frame = {
            "schema_version": 4,
            "generation_id": "generation",
            "frame_id": "0",
            "objects": [],
            "candidate_lifecycle": [{
                "camera": "front",
                "candidate_id": "T1",
                "final_status": "dropped",
                "last_successful_stage": "backprojection",
                "events": [{
                    "stage": "cluster_filter",
                    "status": "dropped",
                    "reason_code": "min_fused_points",
                    "reason_message": "verbose explanation",
                }],
            }],
            "diagnostics": {"tracking_state": {"tracks": []}},
            "_resume_tracking_state": {"tracks": []},
        }

        main, debug = split_fused_frame_artifacts(frame, "fusion_debug.json")

        self.assertNotIn("candidate_lifecycle", main)
        self.assertNotIn("diagnostics", main)
        self.assertNotIn("_resume_tracking_state", main)
        self.assertEqual("fusion_debug.json", main["diagnostics_ref"])
        self.assertEqual("min_fused_points", main["candidate_outcomes"][0]["reason_code"])
        self.assertNotIn("reason_message", main["candidate_outcomes"][0])
        self.assertEqual(frame["candidate_lifecycle"], debug["candidate_lifecycle"])
        self.assertEqual(frame["diagnostics"], debug["diagnostics"])

    def test_compact_outcome_keeps_object_mapping(self) -> None:
        outcomes = compact_candidate_outcomes([{
            "camera": "front",
            "candidate_id": "T1",
            "final_status": "fused",
            "fused_object_id": "O1",
            "last_successful_stage": "fused_output",
            "events": [],
        }])

        self.assertEqual("O1", outcomes[0]["object_id"])
        self.assertEqual("fused", outcomes[0]["status"])

    def test_resume_restores_tracking_checkpoint_from_debug_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            frame_dir = output_dir / "frames" / "000000_0"
            frame_dir.mkdir(parents=True)
            generation_id = "generation"
            (frame_dir / "fused_objects.json").write_text(json.dumps({
                "schema_version": 4,
                "generation_id": generation_id,
                "frame_id": "0",
                "objects": [],
                "diagnostics_ref": "fusion_debug.json",
            }), encoding="utf-8")
            expected_state = {
                "next_object_index": 4,
                "tracks": [{"index": 4, "centroid": [0, 0, 0]}],
            }
            (frame_dir / "fusion_debug.json").write_text(json.dumps({
                "generation_id": generation_id,
                "frame_id": "0",
                "diagnostics": {"tracking_state": expected_state},
            }), encoding="utf-8")
            entry = {
                "frame_id": "0",
                "status": "complete",
                "fused_objects_json": "frames/000000_0/fused_objects.json",
                "fusion_debug_json": "frames/000000_0/fusion_debug.json",
            }

            loaded = _load_completed_frame(entry, output_dir, generation_id)
            track_state = {}
            self.assertIsNotNone(loaded)
            _restore_track_state(loaded, track_state)

            self.assertEqual(expected_state, track_state)


class CandidateMaskAreaFilterTest(unittest.TestCase):
    def test_legacy_stage1_candidate_is_rejected_without_conversion(self) -> None:
        with self.assertRaisesRegex(ValueError, "rerun Stage1"):
            validate_semantic_candidates([{
                "id": "T1",
                "role": "target",
                "role_scores": {"target": 1.0},
            }])

    def test_candidates_below_threshold_are_removed(self) -> None:
        candidates = [
            {"id": "C-small", "mask_area_pixels": 39},
            {"id": "C-boundary", "mask_area_pixels": 40},
            {"id": "C-large", "mask_area_pixels": 100},
        ]

        kept, suppressed = filter_candidates_by_mask_area(candidates, 40)

        self.assertEqual(["C-boundary", "C-large"], [item["id"] for item in kept])
        self.assertEqual("C-small", suppressed[0]["candidate_id"])
        self.assertEqual("mask_area_pixels_below_threshold", suppressed[0]["reason"])

    def test_zero_threshold_disables_filter(self) -> None:
        candidates = [{"id": "C1", "mask_area_pixels": 1}]
        kept, suppressed = filter_candidates_by_mask_area(candidates, 0)
        self.assertEqual(candidates, kept)
        self.assertEqual([], suppressed)


class CompactObjectSummaryTest(unittest.TestCase):
    @patch(
        "multiview_candidate_fusion.load_object_points",
        return_value=np.zeros((3, 3), dtype=np.float64),
    )
    def test_summary_does_not_duplicate_verbose_frame_evidence(self, _load_points) -> None:
        semantic_evidence = [{
                "semantic_group_id": "SG1",
                "score": 0.8,
                "score_mass": 1.6,
                "support_count": 2,
                "supporting_prompts": ["red block"],
                "compatible_roles": ["target"],
                "cameras": ["front"],
                "frames": ["0"],
        }]
        objects = []
        for index, x in enumerate((0.0, 0.2), start=1):
            objects.append(
                {
                    "id": f"O{index}",
                    "centroid_world": [x, 0.0, 0.1],
                    "bbox3d_world": [[x - 0.05, -0.05, 0.05], [x + 0.05, 0.05, 0.15]],
                    "primary_camera": "front",
                    "visible_camera": ["front"],
                    "mask_area": 100,
                    "sam_score": 0.9,
                    "semantic_evidence": semantic_evidence,
                    "observations": [
                        {
                            "camera": "front",
                            "candidate_id": f"C{index}",
                            "observation_id": f"0:front:C{index}",
                            "provenance": {"very_verbose": ["unused"] * 20},
                            "semantic_evidence": semantic_evidence,
                            "mask_path": f"mask-{index}.png",
                            "sam_score": 0.9,
                        }
                    ],
                }
            )
        frame = {
            "frame_id": "0",
            "frame_index": 0,
            "frame_ref": "frames/000000_0/fused_objects.json",
            "objects": objects,
        }

        summary = build_object_summary(
            [frame],
            {},
            {"instruction": "move the red block", "role_spec": {"target": "block"}, "semantic_groups": [{"semantic_group_id": "SG1"}]},
            schema_version=4,
            generation_id="generation-id",
        )

        self.assertEqual("compact_v2", summary["storage_layout"])
        self.assertNotIn("trajectory", summary["object_tracks"][0])
        frame_input = summary["frame_decision_inputs"][0]
        self.assertNotIn("instruction_prior", frame_input)
        self.assertNotIn("role_spec_prior", frame_input)
        self.assertEqual([], frame_input["pairwise_relations"])
        candidate = frame_input["candidate_objects"][0]
        self.assertEqual(
            {"semantic_group_id": "SG1", "score": 0.8, "support_count": 2, "compatible_roles": ["target"]},
            candidate["semantic_evidence"][0],
        )
        self.assertEqual(["red block"], summary["object_tracks"][0]["semantic_evidence"][0]["supporting_prompts"])
        serialized = str(summary)
        self.assertNotIn("very_verbose", serialized)
        self.assertNotIn("observation_id", serialized)


def _observation(name: str, camera: str, bbox: list[list[float]]) -> Observation3D:
    bounds = np.asarray(bbox, dtype=np.float64)
    points = np.array([bounds[0], bounds[1], (bounds[0] + bounds[1]) / 2])
    return Observation3D(
        observation_id=name,
        semantic_evidence={"SG1": {"semantic_group_id": "SG1", "score": 0.9, "compatible_roles": ["target"], "supporting_prompts": ["target"]}},
        provenance={},
        camera=camera,
        candidate={"id": name, "score": 0.9},
        points_world=points,
        centroid_world=points.mean(axis=0),
        bbox3d_world=bounds,
    )


def _observation_from_points(
    name: str,
    camera: str,
    points: list[list[float]],
    prompt: str,
    semantic_roles: dict[str, float] | None = None,
    score: float = 0.9,
) -> Observation3D:
    cloud = np.asarray(points, dtype=np.float64)
    return Observation3D(
        observation_id=name,
        semantic_evidence={
            f"SG_{role.upper()}": {
                "semantic_group_id": f"SG_{role.upper()}",
                "score": value,
                "compatible_roles": [role],
                "supporting_prompts": [prompt],
            }
            for role, value in (semantic_roles or {"target": score}).items()
        },
        provenance={"prompt_provenance": [{"source_prompt": prompt}]},
        camera=camera,
        candidate={"id": name.split(":")[-1], "score": score},
        points_world=cloud,
        centroid_world=cloud.mean(axis=0),
        bbox3d_world=np.stack([cloud.min(axis=0), cloud.max(axis=0)]),
    )


class CrossCameraFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            cluster_distance_m=0.03,
            bbox_iou_threshold=0.30,
            nearest_distance_m=None,
            max_size_ratio=4.0,
            max_hypothesis_diameter_m=0.50,
            legacy_union_find=False,
        )

    def test_bbox_overlap_does_not_bypass_centroid_gate(self) -> None:
        front = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        left = _observation("left:C1", "left", [[0.05, 0, 0], [0.15, 0.10, 0.10]])

        clusters = cluster_observations([front, left], self.args)

        self.assertEqual([1, 1], sorted(len(cluster) for cluster in clusters))

    def test_centroid_and_bbox_agreement_merge_different_cameras(self) -> None:
        front = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        left = _observation("left:C1", "left", [[0.01, 0, 0], [0.11, 0.10, 0.10]])

        clusters = cluster_observations([front, left], self.args)

        self.assertEqual(1, len(clusters))
        self.assertEqual({"front", "left"}, {item.camera for item in clusters[0]})

    def test_surface_distance_vetoes_nearby_centroids(self) -> None:
        front = _observation("front:C1", "front", [[0, 0, 0], [0.02, 0.02, 0.02]])
        left = _observation("left:C1", "left", [[0.015, 0, 0], [0.035, 0.02, 0.02]])
        self.args.cluster_distance_m = 0.03
        self.args.bbox_iou_threshold = 0.0
        self.args.nearest_distance_m = 0.005

        clusters = cluster_observations([front, left], self.args)

        self.assertEqual([1, 1], sorted(len(cluster) for cluster in clusters))

    def test_surface_and_centroid_agreement_merge_different_cameras(self) -> None:
        front = _observation("front:C1", "front", [[0, 0, 0], [0.02, 0.02, 0.02]])
        left = _observation("left:C1", "left", [[0.001, 0, 0], [0.021, 0.02, 0.02]])
        self.args.bbox_iou_threshold = 0.0
        self.args.nearest_distance_m = 0.01

        clusters = cluster_observations([front, left], self.args)

        self.assertEqual(1, len(clusters))

    def test_same_camera_observations_remain_separate(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        second = _observation("front:C2", "front", [[0.04, 0, 0], [0.14, 0.10, 0.10]])

        clusters = cluster_observations([first, second], self.args)

        self.assertEqual([1, 1], sorted(len(cluster) for cluster in clusters))

    def test_preferred_camera_weight_seeds_hypothesis(self) -> None:
        front = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        left = _observation("left:C1", "left", [[0.01, 0, 0], [0.11, 0.10, 0.10]])
        front.semantic_evidence["SG1"]["score"] = 0.7
        left.semantic_evidence["SG1"]["score"] = 0.9
        self.args.preferred_camera = "front"
        self.args.preferred_camera_weight = 1.5

        clusters = cluster_observations([left, front], self.args)

        self.assertEqual(1, len(clusters))
        self.assertEqual("front", clusters[0][0].camera)


class SameCameraNmsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            same_camera_nms_mask_iou=0.55,
            same_camera_nms_containment=0.85,
            same_camera_nms_centroid_distance_m=0.03,
            same_camera_nms_max_size_ratio=2.5,
            same_camera_nms_fragment_max_point_ratio=0.45,
            same_camera_nms_fragment_min_bbox_containment=0.90,
            same_camera_nms_fragment_cloud_distance_m=0.018,
            same_camera_nms_fragment_min_cloud_fraction=0.65,
        )

    def test_strict_2d_3d_duplicate_is_suppressed_and_evidence_retained(self) -> None:
        primary = _observation(
            "front:C1",
            "front",
            [[0.0, 0.0, 1.0], [0.10, 0.10, 1.10]],
        )
        duplicate = _observation(
            "front:C2",
            "front",
            [[0.005, 0.0, 1.0], [0.105, 0.10, 1.10]],
        )
        duplicate.semantic_evidence = {"SG_REFERENCE": {"semantic_group_id": "SG_REFERENCE", "score": 0.8, "compatible_roles": ["reference"], "supporting_prompts": ["reference"]}}
        duplicate.candidate = {"id": "C2", "score": 0.8}
        primary.candidate = {"id": "C1", "score": 0.9}
        mask_a = np.zeros((16, 16), dtype=bool)
        mask_a[2:12, 2:12] = True
        mask_b = np.zeros_like(mask_a)
        mask_b[3:12, 3:12] = True

        kept, diagnostics = suppress_same_camera_duplicates(
            [(duplicate, mask_b), (primary, mask_a)],
            self.args,
        )

        self.assertEqual(1, len(kept))
        self.assertIs(primary, kept[0])
        self.assertEqual(1, len(diagnostics))
        self.assertEqual("same_camera_2d_3d_nms", diagnostics[0]["reason"])
        self.assertAlmostEqual(0.8, primary.semantic_evidence["SG_REFERENCE"]["score"])
        self.assertEqual(
            "front:C2",
            primary.provenance["same_camera_nms_suppressed"][0][
                "observation_id"
            ],
        )

    def test_overlapping_masks_at_different_depths_remain_separate(self) -> None:
        self.args._same_camera_nms_rejected = []
        near = _observation(
            "front:C1",
            "front",
            [[0.0, 0.0, 0.8], [0.10, 0.10, 0.9]],
        )
        far = _observation(
            "front:C2",
            "front",
            [[0.0, 0.0, 1.0], [0.10, 0.10, 1.1]],
        )
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:10, 2:10] = True

        kept, diagnostics = suppress_same_camera_duplicates(
            [(near, mask), (far, mask.copy())],
            self.args,
        )

        self.assertEqual(2, len(kept))
        self.assertEqual([], diagnostics)
        self.assertEqual(
            "centroid_distance",
            self.args._same_camera_nms_rejected[0]["failed_gate"],
        )

    def test_contained_thin_fragment_is_suppressed_without_2d_overlap(self) -> None:
        main = _observation_from_points(
            "front:C1",
            "front",
            [
                [0.00, 0.00, 1.00],
                [0.10, 0.00, 1.00],
                [0.00, 0.10, 1.00],
                [0.10, 0.10, 1.00],
                [0.00, 0.00, 1.10],
                [0.10, 0.00, 1.10],
                [0.00, 0.10, 1.10],
                [0.10, 0.10, 1.10],
                [0.04, 0.05, 1.05],
                [0.06, 0.05, 1.05],
            ],
            "a magenta block",
            score=0.85,
        )
        fragment = _observation_from_points(
            "front:C8",
            "front",
            [
                [0.04, 0.05, 1.05],
                [0.05, 0.05, 1.05],
                [0.06, 0.05, 1.05],
            ],
            "a magenta block",
            semantic_roles={"reference": 0.22},
            score=0.22,
        )
        main_mask = np.zeros((16, 16), dtype=bool)
        main_mask[2:8, 2:8] = True
        fragment_mask = np.zeros_like(main_mask)
        fragment_mask[10:12, 10:13] = True

        kept, diagnostics = suppress_same_camera_duplicates(
            [(main, main_mask), (fragment, fragment_mask)],
            self.args,
        )

        self.assertEqual([main], kept)
        self.assertEqual("fragment_subset", diagnostics[0]["match_mode"])
        self.assertEqual("C8", diagnostics[0]["suppressed_candidate_id"])
        self.assertEqual(1.0, diagnostics[0]["fragment_bbox_axis_containment"])
        self.assertEqual(1.0, diagnostics[0]["fragment_point_to_cloud_fraction"])

    def test_fragment_path_requires_shared_semantic_prompt(self) -> None:
        main = _observation_from_points(
            "front:C1",
            "front",
            [[0, 0, 1.0], [0.1, 0.1, 1.1], [0.05, 0.05, 1.05]] * 4,
            "a magenta block",
        )
        nested = _observation_from_points(
            "front:C2",
            "front",
            [[0.04, 0.04, 1.04], [0.05, 0.05, 1.05], [0.06, 0.06, 1.06]],
            "a separate handle",
            score=0.2,
        )
        mask_a = np.zeros((12, 12), dtype=bool)
        mask_b = np.zeros_like(mask_a)

        kept, diagnostics = suppress_same_camera_duplicates(
            [(main, mask_a), (nested, mask_b)],
            self.args,
        )

        self.assertEqual(2, len(kept))
        self.assertEqual([], diagnostics)

    def test_interaction_part_is_not_absorbed_by_whole_object(self) -> None:
        main = _observation_from_points(
            "front:C1",
            "front",
            [[0, 0, 1.0], [0.1, 0.1, 1.1], [0.05, 0.05, 1.05]] * 4,
            "drawer",
        )
        handle = _observation_from_points(
            "front:P1",
            "front",
            [[0.04, 0.04, 1.04], [0.05, 0.05, 1.05], [0.06, 0.06, 1.06]],
            "drawer",
            semantic_roles={"interaction_part": 0.9},
            score=0.9,
        )
        mask = np.ones((12, 12), dtype=bool)

        kept, diagnostics = suppress_same_camera_duplicates(
            [(main, mask), (handle, mask.copy())],
            self.args,
        )

        self.assertEqual(2, len(kept))
        self.assertEqual([], diagnostics)

    def test_adjacent_objects_remain_separate(self) -> None:
        left = _observation(
            "front:C1",
            "front",
            [[0.0, 0.0, 1.0], [0.04, 0.04, 1.04]],
        )
        right = _observation(
            "front:C2",
            "front",
            [[0.05, 0.0, 1.0], [0.09, 0.04, 1.04]],
        )
        mask_a = np.zeros((12, 12), dtype=bool)
        mask_a[2:6, 2:6] = True
        mask_b = np.zeros_like(mask_a)
        mask_b[2:6, 7:11] = True

        kept, diagnostics = suppress_same_camera_duplicates(
            [(left, mask_a), (right, mask_b)],
            self.args,
        )

        self.assertEqual(2, len(kept))
        self.assertEqual([], diagnostics)


class CameraSupportFilterTest(unittest.TestCase):
    @staticmethod
    def _args(
        *,
        keep_score: float = 0.0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            min_fused_camera_count=2,
            camera_visibility_depth_tolerance_m=0.03,
            camera_visibility_min_point_fraction=0.05,
            single_camera_keep_score=keep_score,
            _camera_support_diagnostics=[],
            _filtered_cluster_diagnostics=[],
        )

    @staticmethod
    def _context(depth_value: float) -> dict[str, np.ndarray]:
        return {
            "intrinsics": np.asarray(
                [[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]
            ),
            "extrinsics": np.eye(4),
            "depth": np.full((16, 16), depth_value, dtype=np.float64),
        }

    def test_single_camera_cluster_drops_when_missing_view_could_see_it(self) -> None:
        candidate = _observation(
            "front:C1",
            "front",
            [[-0.05, -0.05, 1.0], [0.05, 0.05, 1.05]],
        )
        args = self._args()

        kept = filter_clusters_by_camera_support(
            [[candidate]],
            args,
            {"front": self._context(1.0), "left": self._context(2.0)},
        )

        self.assertEqual([], kept)
        self.assertEqual(
            "min_fused_camera_count",
            args._camera_support_diagnostics[0]["reason"],
        )

    def test_camera_support_filter_is_disabled_at_one_camera(self) -> None:
        candidate = _observation(
            "front:C1",
            "front",
            [[-0.05, -0.05, 1.0], [0.05, 0.05, 1.05]],
        )
        args = self._args()
        args.min_fused_camera_count = 1

        kept = filter_clusters_by_camera_support(
            [[candidate]],
            args,
            {"front": self._context(1.0), "left": self._context(2.0)},
        )

        self.assertEqual([[candidate]], kept)
        self.assertEqual([], args._camera_support_diagnostics)

    def test_occluded_single_camera_cluster_is_kept(self) -> None:
        candidate = _observation(
            "front:C1",
            "front",
            [[-0.05, -0.05, 1.0], [0.05, 0.05, 1.05]],
        )
        args = self._args()

        kept = filter_clusters_by_camera_support(
            [[candidate]],
            args,
            {"front": self._context(1.0), "left": self._context(0.5)},
        )

        self.assertEqual(1, len(kept))
        self.assertIs(candidate, kept[0][0])
        self.assertEqual(
            "insufficient_observable_cameras",
            args._camera_support_diagnostics[0]["reason"],
        )

    def test_high_confidence_exception_is_configurable(self) -> None:
        candidate = _observation(
            "front:C1",
            "front",
            [[-0.05, -0.05, 1.0], [0.05, 0.05, 1.05]],
        )
        args = self._args(keep_score=0.85)

        kept = filter_clusters_by_camera_support(
            [[candidate]],
            args,
            {"front": self._context(1.0), "left": self._context(2.0)},
        )

        self.assertEqual(1, len(kept))
        self.assertIs(candidate, kept[0][0])
        self.assertEqual(
            "single_camera_high_confidence",
            args._camera_support_diagnostics[0]["reason"],
        )


class TemporalObjectTrackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            track_distance_m=0.22,
            track_max_missed_frames=4,
            track_max_size_ratio=4.0,
        )

    def test_track_id_survives_four_missing_frames(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        objects, state = assign_object_ids([[first]], {}, self.args, "f0")
        self.assertEqual(["O1"], [item["id"] for item in objects])

        for missed_count, frame_id in enumerate(("f1", "f2", "f3", "f4"), 1):
            objects, state = assign_object_ids([], state, self.args, frame_id)
            self.assertEqual([], objects)
            self.assertEqual(missed_count, state["tracks"][0]["missed_frames"])

        reappeared = _observation("front:C1", "front", [[0.01, 0, 0], [0.11, 0.10, 0.10]])
        objects, _ = assign_object_ids([[reappeared]], state, self.args, "f5")
        self.assertEqual(["O1"], [item["id"] for item in objects])

    def test_expired_track_gets_new_id(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        _, state = assign_object_ids([[first]], {}, self.args, "f0")
        for frame_id in ("f1", "f2", "f3", "f4", "f5"):
            _, state = assign_object_ids([], state, self.args, frame_id)

        reappeared = _observation("front:C1", "front", [[0.01, 0, 0], [0.11, 0.10, 0.10]])
        objects, _ = assign_object_ids([[reappeared]], state, self.args, "f6")
        self.assertEqual(["O2"], [item["id"] for item in objects])

    def test_relaxed_distance_gate_keeps_moderately_moving_object(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        _, state = assign_object_ids([[first]], {}, self.args, "f0")

        moved = _observation("front:C2", "front", [[0.21, 0, 0], [0.31, 0.10, 0.10]])
        objects, _ = assign_object_ids([[moved]], state, self.args, "f1")

        self.assertEqual(["O1"], [item["id"] for item in objects])

    def test_relaxed_size_gate_keeps_partial_to_full_bbox_change(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        _, state = assign_object_ids([[first]], {}, self.args, "f0")

        fuller = _observation("front:C2", "front", [[-0.10, -0.10, -0.10], [0.20, 0.20, 0.20]])
        objects, _ = assign_object_ids([[fuller]], state, self.args, "f1")

        self.assertEqual(["O1"], [item["id"] for item in objects])

    def test_size_gate_prevents_implausible_id_match(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        _, state = assign_object_ids([[first]], {}, self.args, "f0")

        much_larger = _observation("front:C2", "front", [[-0.20, -0.20, -0.20], [0.30, 0.30, 0.30]])
        objects, _ = assign_object_ids([[much_larger]], state, self.args, "f1")
        self.assertEqual(["O2"], [item["id"] for item in objects])


class SourceFrameIndexTest(unittest.TestCase):
    def test_numeric_frame_id_wins_over_sampled_frame_index(self) -> None:
        self.assertEqual(20, frame_index_from_frame({"frame_id": "20", "frame_index": 2}))

    def test_non_numeric_frame_id_falls_back_to_frame_index(self) -> None:
        self.assertEqual(2, frame_index_from_frame({"frame_id": "keyframe", "frame_index": 2}))


class CentroidCloudConsistencyTest(unittest.TestCase):
    @staticmethod
    def _component_args(diagnostics: list[dict] | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            min_fused_points=0,
            min_bbox_diagonal_m=0.0,
            max_centroid_to_cloud_distance_m=0.0,
            component_voxel_size_m=0.008,
            min_largest_component_ratio=0.75,
            max_secondary_component_ratio=0.20,
            min_component_centroid_gap_m=0.02,
            min_component_points=5,
            _filtered_cluster_diagnostics=diagnostics if diagnostics is not None else [],
        )

    def test_large_empty_gap_around_centroid_drops_cluster(self) -> None:
        bad = _observation(
            "front:bad",
            "front",
            [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
        )
        bad.points_world = np.asarray(
            [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
            dtype=np.float64,
        )
        bad.centroid_world = bad.points_world.mean(axis=0)

        good = _observation(
            "front:good",
            "front",
            [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]],
        )
        diagnostics = []
        args = SimpleNamespace(
            min_fused_points=0,
            min_bbox_diagonal_m=0.0,
            max_centroid_to_cloud_distance_m=0.02,
            _filtered_cluster_diagnostics=diagnostics,
        )

        kept = filter_small_clusters([[bad], [good]], args)

        self.assertEqual(1, len(kept))
        self.assertIs(good, kept[0][0])
        self.assertEqual("max_centroid_to_cloud_distance_m", diagnostics[0]["reason"])
        self.assertEqual(["front:bad"], diagnostics[0]["candidate_ids"])
        self.assertAlmostEqual(0.05, diagnostics[0]["centroid_to_cloud_distance_m"])

    def test_centroid_distance_filter_can_be_disabled(self) -> None:
        candidate = _observation(
            "front:split",
            "front",
            [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
        )
        candidate.points_world = np.asarray(
            [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
            dtype=np.float64,
        )
        args = SimpleNamespace(
            min_fused_points=0,
            min_bbox_diagonal_m=0.0,
            max_centroid_to_cloud_distance_m=0.0,
        )

        kept = filter_small_clusters([[candidate]], args)
        self.assertEqual(1, len(kept))
        self.assertIs(candidate, kept[0][0])

    def test_two_large_disconnected_point_regions_drop_cluster(self) -> None:
        candidate = _observation(
            "front:two-objects",
            "front",
            [[0.0, 0.0, 0.0], [0.04, 0.001, 0.001]],
        )
        candidate.points_world = np.concatenate(
            (
                np.tile([[0.0, 0.0, 0.0]], (60, 1)),
                np.tile([[0.04, 0.0, 0.0]], (40, 1)),
            ),
            axis=0,
        )
        candidate.centroid_world = candidate.points_world.mean(axis=0)
        diagnostics: list[dict] = []

        kept = filter_small_clusters(
            [[candidate]],
            self._component_args(diagnostics),
        )

        self.assertEqual([], kept)
        self.assertEqual("multiple_large_3d_components", diagnostics[0]["reason"])
        self.assertAlmostEqual(0.6, diagnostics[0]["largest_component_ratio"])
        self.assertAlmostEqual(0.4, diagnostics[0]["second_component_ratio"])
        self.assertAlmostEqual(0.04, diagnostics[0]["component_centroid_gap_m"])

    def test_small_secondary_region_is_treated_as_noise(self) -> None:
        candidate = _observation(
            "front:object-plus-noise",
            "front",
            [[0.0, 0.0, 0.0], [0.04, 0.001, 0.001]],
        )
        candidate.points_world = np.concatenate(
            (
                np.tile([[0.0, 0.0, 0.0]], (85, 1)),
                np.tile([[0.04, 0.0, 0.0]], (15, 1)),
            ),
            axis=0,
        )
        candidate.centroid_world = candidate.points_world.mean(axis=0)

        kept = filter_small_clusters(
            [[candidate]],
            self._component_args(),
        )

        self.assertEqual(1, len(kept))
        self.assertIs(candidate, kept[0][0])


class CandidateLifecycleTest(unittest.TestCase):
    def test_every_candidate_has_a_clear_final_disposition(self) -> None:
        raw_candidates = [
            {
                "camera": "front",
                "candidate_id": candidate_id,
                "sam_score": 0.9,
                "mask_area_pixels": 100,
            }
            for candidate_id in ("C1", "C2", "C3", "C4")
        ]
        canonical_sources = {
            ("front", candidate_id): (candidate_id,)
            for candidate_id in ("C1", "C2", "C3", "C4")
        }
        backprojection = [
            {
                "camera": "front",
                "candidate_id": "C1",
                "observation_id": "f:front:C1",
                "status": "accepted",
                "point_count": 40,
            },
            {
                "camera": "front",
                "candidate_id": "C2",
                "observation_id": "f:front:C2",
                "status": "accepted",
                "point_count": 30,
            },
            {
                "camera": "front",
                "candidate_id": "C3",
                "status": "dropped",
                "reason": "empty_3d_backprojection",
            },
            {
                "camera": "front",
                "candidate_id": "C4",
                "observation_id": "f:front:C4",
                "status": "accepted",
                "point_count": 10,
            },
        ]
        nms = [
            {
                "camera": "front",
                "kept_candidate_id": "C1",
                "suppressed_candidate_id": "C2",
                "reason": "same_camera_2d_3d_nms",
            }
        ]
        filtered = [
            {
                "reason": "min_fused_points",
                "candidate_refs": [
                    {
                        "camera": "front",
                        "candidate_id": "C4",
                        "observation_id": "f:front:C4",
                    }
                ],
            }
        ]
        objects = [
            {
                "id": "O1",
                "observations": [
                    {"camera": "front", "candidate_id": "C1"}
                ],
            }
        ]

        lifecycle, summary = build_candidate_lifecycle(
            raw_candidates,
            canonical_sources,
            backprojection,
            [],
            [],
            nms,
            filtered,
            objects,
        )

        by_id = {item["candidate_id"]: item for item in lifecycle}
        self.assertEqual("fused", by_id["C1"]["final_status"])
        self.assertEqual("O1", by_id["C1"]["fused_object_id"])
        self.assertEqual("merged", by_id["C2"]["final_status"])
        self.assertEqual("O1", by_id["C2"]["fused_object_id"])
        self.assertEqual("dropped", by_id["C3"]["final_status"])
        self.assertEqual("dropped", by_id["C4"]["final_status"])
        self.assertEqual(1, summary["directly_fused_candidate_count"])
        self.assertEqual(1, summary["merged_candidate_count"])
        self.assertEqual(2, summary["dropped_candidate_count"])
        self.assertEqual(0, summary["unresolved_candidate_count"])


if __name__ == "__main__":
    unittest.main()
