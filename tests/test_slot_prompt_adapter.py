"""Original-pixel prompt geometry, visibility and temporal safety without SAM3."""

from __future__ import annotations

import copy
from dataclasses import replace
import unittest

import numpy as np

from slot_guidance import SlotGuidance, identity_world_transform
from slot_prompt_adapter import AdapterConfig, CameraFrame, build_prompts


STAMP = "2026-11-05T10:00:00+08:00"


def scene() -> tuple[SlotGuidance, CameraFrame, np.ndarray, np.ndarray]:
    def cloud(x: float) -> np.ndarray:
        xx, yy = np.meshgrid(np.linspace(x - .02, x + .02, 5), np.linspace(-.02, .02, 5))
        return np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)

    target, reference = cloud(-.15), cloud(.15)
    meta = {
        "schema_version": 1, "artifact_type": "slot_guidance",
        "episode_id": "episode0", "frame_id": "sample-3", "source_frame_index": 150,
        "instruction": "put the left red block beside the right red block",
        "task_name": "move_block", "coordinate_frame": "rlbench_world", "coordinate_units": "m",
        "observed_at": STAMP, "created_at": STAMP, "guidance_source": "mock",
        "source_checkpoint": None, "source_git_sha": "a" * 40,
        "transform": identity_world_transform(source="current frame fixture", timestamp=STAMP, source_frame_index=150),
        "points_file": "fixture.npz", "points_sha256": "0" * 64,
        "reference_null_probability": .03,
    }
    for role, points, center, identity in (("target", target, [-.15, 0, 1], "left"),
                                            ("reference", reference, [.15, 0, 1], "right")):
        meta[role] = {"status": "VISIBLE", "semantic_valid": True, "confidence": .9,
                      "center_world": center, "semantic_label": "red block", "instance_id": identity,
                      "points_key": f"{role}_points_world", "point_count": len(points)}
    guidance = SlotGuidance(meta, target, reference)
    camera = CameraFrame("wrist", "episode0", "sample-3", 150, STAMP,
                         np.array([[100., 0, 60], [0, 100., 50], [0, 0, 1]]),
                         np.eye(4), (121, 101), "low_dim_obs[150].misc")
    return guidance, camera, np.zeros((101, 121, 3), np.uint8), np.ones((101, 121))


def with_cloud(guidance: SlotGuidance, role: str, cloud: np.ndarray) -> SlotGuidance:
    meta = copy.deepcopy(guidance.metadata)
    meta[role]["point_count"] = len(cloud)
    return SlotGuidance(meta, cloud if role == "target" else guidance.target_points_world,
                        cloud if role == "reference" else guidance.reference_points_world)


class SlotPromptAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guidance, self.camera, self.rgb, self.depth = scene()

    def prompts(self, guidance=None, camera=None, rgb=None, depth=None, config=None):
        return build_prompts(guidance or self.guidance, camera or self.camera,
                             self.rgb if rgb is None else rgb,
                             self.depth if depth is None else depth, config)

    def assert_failed(self, prompt, reason: str) -> None:
        self.assertFalse(prompt.prompt_valid)
        self.assertEqual(prompt.failure_reason, reason)
        self.assertIsNone(prompt.bbox_xyxy)
        self.assertEqual(prompt.positive_points_xy, [])
        self.assertEqual(prompt.negative_points_xy, [])

    def test_same_category_instances_stay_separate_and_negatives_are_supported(self):
        prompts = self.prompts()
        target, reference = prompts["target"], prompts["reference"]
        self.assertTrue(target.prompt_valid)
        self.assertTrue(reference.prompt_valid)
        self.assertLess(target.bbox_xyxy[2], reference.bbox_xyxy[0])
        self.assertEqual(target.positive_points_xy[0], [45., 50.])
        self.assertEqual(reference.positive_points_xy[0], [75., 50.])
        self.assertEqual(target.negative_points_xy, reference.positive_points_xy)
        self.assertEqual(reference.negative_points_xy, target.positive_points_xy)
        self.assertEqual(target.to_dict(), target.as_dict())
        for prompt in prompts.values():
            for x, y in prompt.positive_points_xy:
                x1, y1, x2, y2 = prompt.bbox_xyxy
                self.assertTrue(x1 <= x <= x2 and y1 <= y <= y2)
                self.assertEqual(self.depth[int(y), int(x)], 1)

    def test_empirical_quantiles_remove_extreme_world_outlier(self):
        normal = self.prompts()["target"]
        cloud = np.vstack([self.guidance.target_points_world, [50, 50, 1]])
        guidance = with_cloud(self.guidance, "target", cloud)
        actual = self.prompts(guidance=guidance)["target"]
        self.assertTrue(actual.prompt_valid)
        self.assertEqual(actual.bbox_xyxy, normal.bbox_xyxy)
        self.assertLess(actual.support["trimmed_points"], actual.support["input_points"])

    def test_behind_camera_and_outside_objects_have_no_box(self):
        for shift, reason in (([0, 0, -2], "BEHIND_CAMERA"), ([5, 0, 0], "OUTSIDE_IMAGE")):
            with self.subTest(reason=reason):
                guidance = with_cloud(self.guidance, "target", self.guidance.target_points_world + shift)
                self.assert_failed(self.prompts(guidance=guidance)["target"], reason)

    def test_occlusion_requires_actual_surface_support(self):
        self.assert_failed(self.prompts(depth=np.full_like(self.depth, .5))["target"], "OCCLUDED")
        self.assert_failed(self.prompts(depth=np.full_like(self.depth, 2.))["target"], "DEPTH_UNSUPPORTED")

    def test_invalid_measured_depth_is_not_visible(self):
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(depth=value):
                self.assert_failed(self.prompts(depth=np.full_like(self.depth, value))["target"], "INVALID_DEPTH")

    def test_null_unseen_low_confidence_remain_distinct(self):
        for status in ("NULL", "NOT_VISIBLE", "LOW_CONFIDENCE"):
            with self.subTest(status=status):
                guidance = with_cloud(self.guidance, "reference", np.empty((0, 3)))
                guidance.metadata["reference"] = {"status": status, "semantic_valid": status != "NULL",
                    "confidence": .1, "center_world": None, "points_key": "reference_points_world", "point_count": 0}
                prompts = self.prompts(guidance=guidance)
                self.assert_failed(prompts["reference"], status)
                self.assertTrue(prompts["target"].prompt_valid)
                self.assertEqual(prompts["target"].negative_points_xy, [])

    def test_visible_but_unreliable_role_is_rejected(self):
        guidance = copy.deepcopy(self.guidance)
        guidance.metadata["target"]["confidence"] = .1
        self.assert_failed(self.prompts(guidance=guidance)["target"], "LOW_CONFIDENCE")

    def test_empty_center_only_cannot_invent_box(self):
        guidance = with_cloud(self.guidance, "target", np.empty((0, 3)))
        self.assert_failed(self.prompts(guidance=guidance)["target"], "NO_GEOMETRY")

    def test_standard_deviation_or_unverified_range_is_not_a_box(self):
        for source in ("slot weighted standard deviation", "std", "verified_world_aabb_half_extents:"):
            with self.subTest(source=source):
                guidance = with_cloud(self.guidance, "target", np.empty((0, 3)))
                guidance.metadata["target"].update(spread_world=[.03, .03, .03], spread_source=source)
                self.assert_failed(self.prompts(guidance=guidance)["target"], "UNVERIFIED_SPREAD")

    def test_verified_world_range_needs_measured_depth_inside_extent(self):
        guidance = with_cloud(self.guidance, "target", np.empty((0, 3)))
        guidance.metadata["target"].update(spread_world=[.03, .03, .03],
            spread_source="oracle_gt_world_aabb_half_extents:current GT object bbox")
        prompt = self.prompts(guidance=guidance)["target"]
        self.assertTrue(prompt.prompt_valid)
        self.assertEqual(prompt.support["geometry_source"], "verified_spread")
        self.assertIn("GT object", prompt.support["spread_source"])
        self.assert_failed(self.prompts(guidance=guidance, depth=self.depth * .5)["target"], "DEPTH_UNSUPPORTED")

    def test_exact_current_frame_binding_never_uses_sample_ordinal(self):
        replacements = ({"episode_id": "episode1"}, {"frame_id": "150"}, {"source_frame_index": 3},
                        {"source_frame_index": True}, {"observed_at": "2026-11-05T10:00:01+08:00"},
                        {"observed_at": "2026-11-05T10:00:00"})
        for kwargs in replacements:
            with self.subTest(kwargs=kwargs):
                self.assert_failed(self.prompts(camera=replace(self.camera, **kwargs))["target"], "FRAME_MISMATCH")
        equivalent = replace(self.camera, observed_at="2026-11-05T02:00:00Z")
        self.assertTrue(self.prompts(camera=equivalent)["target"].prompt_valid)

    def test_wrist_camera_motion_uses_current_extrinsics(self):
        initial = self.prompts()["target"]
        moved = self.camera.extrinsics.copy()
        moved[0, 3] = .1
        actual = self.prompts(camera=replace(self.camera, extrinsics=moved))["target"]
        self.assertTrue(actual.prompt_valid)
        np.testing.assert_allclose(np.asarray(actual.bbox_xyxy)[[0, 2]], np.asarray(initial.bbox_xyxy)[[0, 2]] - 10)
        self.assertEqual(actual.positive_points_xy[0], [35., 50.])

    def test_missing_calibration_yields_failure(self):
        for kwargs in ({"intrinsics": None}, {"extrinsics": None}, {"calibration_source": None}):
            with self.subTest(kwargs=kwargs):
                self.assert_failed(self.prompts(camera=replace(self.camera, **kwargs))["target"], "MISSING_CALIBRATION")

    def test_improper_or_noncanonical_calibration_is_rejected(self):
        cases = []
        for index, value in (((0, 0), -100), ((1, 1), 0), ((0, 1), 1), ((2, 0), .1), ((2, 2), 2), ((0, 0), np.nan)):
            k = self.camera.intrinsics.copy()
            k[index] = value
            cases.append({"intrinsics": k})
        for index, value in (((0, 0), 2), ((0, 0), -1), ((3, 0), .1), ((0, 3), np.inf)):
            t = self.camera.extrinsics.copy()
            t[index] = value
            cases.append({"extrinsics": t})
        cases.extend([{"intrinsics": np.eye(4)}, {"extrinsics": np.eye(3)}])
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.assert_failed(self.prompts(camera=replace(self.camera, **kwargs))["target"], "INVALID_CALIBRATION")

    def test_plausible_but_wrong_extrinsic_is_rejected_by_depth_support(self):
        t = self.camera.extrinsics.copy()
        t[2, 3] = .2
        self.assert_failed(self.prompts(camera=replace(self.camera, extrinsics=t))["target"], "DEPTH_UNSUPPORTED")

    def test_raw_rgb_and_depth_dimensions_must_match_calibration(self):
        self.assert_failed(self.prompts(rgb=self.rgb[:, :100])["target"], "IMAGE_SIZE_MISMATCH")
        self.assert_failed(self.prompts(rgb=self.rgb.astype(float))["target"], "INVALID_RGB")
        self.assert_failed(self.prompts(depth=self.depth[:100])["target"], "INVALID_DEPTH")

    def test_repeated_world_points_do_not_fake_projection_support(self):
        cloud = np.repeat([[-.15, 0, 1]], 20, axis=0)
        self.assert_failed(self.prompts(guidance=with_cloud(self.guidance, "target", cloud))["target"], "INSUFFICIENT_SUPPORT")

    def test_overlap_does_not_create_unsupported_negative_points(self):
        guidance = with_cloud(self.guidance, "reference", self.guidance.target_points_world.copy())
        prompts = self.prompts(guidance=guidance)
        for prompt in prompts.values():
            self.assertTrue(prompt.prompt_valid)
            self.assertEqual(prompt.negative_points_xy, [])

    def test_same_declared_instance_never_becomes_a_negative(self):
        guidance = copy.deepcopy(self.guidance)
        guidance.metadata["reference"]["instance_id"] = guidance.metadata["target"]["instance_id"]
        prompts = self.prompts(guidance=guidance)
        self.assertEqual(prompts["target"].negative_points_xy, [])
        self.assertEqual(prompts["target"].support["negative_source"], "excluded_same_instance_id")

    def test_invalid_configuration_is_not_silently_clamped(self):
        for kwargs in ({"min_confidence": 2}, {"depth_tolerance_m": 0}, {"padding_px": -1},
                       {"quantile_low": .7}, {"min_visible_fraction": 0}, {"min_support": True},
                       {"max_positive_points": 0}, {"padding_fraction": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prompts(config=AdapterConfig(**kwargs))


if __name__ == "__main__":
    unittest.main()
