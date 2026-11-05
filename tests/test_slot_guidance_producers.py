"""Current-only GT/Slot export, exact inverse transforms and semantic states."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from slot_guidance import bridgevla_cube_transform, identity_world_transform
from slot_guidance_producers import coarse_slot_guidance, oracle_guidance
from slot_guided_io import FrameSnapshot, RGBDFrame
from slot_prompt_adapter import CameraFrame


STAMP = "2026-11-05T10:00:00+08:00"


def current_snapshot() -> FrameSnapshot:
    pose = np.eye(4)
    pose[0, 3] = .2
    intrinsic = np.array([[10., 0, 4], [0, 10., 3], [0, 0, 1]])
    row = {"name": "wrist", "source_frame_index": 150, "observed_at": STAMP,
           "calibration_source": "current low_dim_obs[150].misc", "intrinsics": intrinsic.tolist(),
           "extrinsics": pose.tolist(), "rgb_path": "unused.png", "depth_path": "unused.npy", "depth_encoding": "meters_npy"}
    camera = CameraFrame("wrist", "episode0", "sample-3", 150, STAMP,
                         intrinsic, pose, (9, 7), row["calibration_source"])
    frame = RGBDFrame(camera, np.zeros((7, 9, 3), np.uint8), np.ones((7, 9)), copy.deepcopy(row))
    return FrameSnapshot({"schema_version": 1, "artifact_type": "slot_rgbd_frame", "episode_id": "episode0",
                          "frame_id": "sample-3", "source_frame_index": 150, "observed_at": STAMP,
                          "cameras": [copy.deepcopy(row)]}, {"wrist": frame}, Path("not-read.json"))


class GuidanceProducerTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = current_snapshot()
        target = np.zeros((7, 9), bool)
        target[2:4, 2:4] = True
        reference = np.zeros((7, 9), bool)
        reference[3:5, 6:8] = True
        self.masks = {"wrist": {"target": target, "reference": reference}}
        raw = np.array([[.1, .2, .3], [.3, .4, .5], [0, 0, 0], [-1, -1, -1], [np.nan, 1, 2], [.9, .9, .9]])
        self.output = {"points": np.stack([raw, raw]), "confidence": np.array([.9, .8]),
                       "role_token_valid": np.array([True, True]), "valid": np.array([True, True]),
                       "reference_null_probability": .1, "geometry": np.zeros(17)}
        self.support = np.array([[True, True, True, True, True, False]] * 2)
        self.transform = bridgevla_cube_transform([1, 2, 3], 2., source="coarse place_pc_in_cube actual inverse",
                                                 timestamp=STAMP, source_frame_index=150)
        self.common = {"instruction": "move left red block beside right red block", "task_name": "move_block",
                       "source_git_sha": "a" * 40, "created_at": STAMP}

    def oracle(self, **kwargs):
        return oracle_guidance(self.snapshot, self.masks, reference_is_null=kwargs.pop("reference_is_null", False),
                               **self.common, **kwargs)

    def coarse(self, **kwargs):
        return coarse_slot_guidance(self.snapshot, self.output,
            point_valid=kwargs.pop("point_valid", self.support), transform=kwargs.pop("transform", self.transform),
            source_checkpoint="coarse.pt", point_support_source="current renderer finite XYZ + positive role support",
            **self.common, **kwargs)

    def test_oracle_backprojection_is_world_metres_and_original_pixels(self):
        result = self.oracle(role_instance_ids={"target": "left", "reference": "right"})
        result.validate()
        np.testing.assert_allclose(result.target_points_world, [[0, -.1, 1], [.1, -.1, 1], [0, 0, 1], [.1, 0, 1]], atol=1e-15)
        np.testing.assert_allclose(result.metadata["target"]["center_world"], [.05, -.05, 1], atol=1e-15)
        np.testing.assert_allclose(result.metadata["reference"]["center_world"], [.45, .05, 1], atol=1e-15)
        self.assertEqual(result.metadata["source_frame_index"], 150)
        self.assertEqual(result.metadata["frame_id"], "sample-3")
        self.assertEqual(result.metadata["guidance_source"], "oracle_gt")
        self.assertEqual(result.metadata["target"]["instance_id"], "left")
        self.assertNotIn("spread_world", result.metadata["target"])
        self.assertIn("calibration_sha256", result.metadata["transform"]["source"])

    def test_oracle_null_empty_gt_and_invalid_depth_are_distinct(self):
        self.masks["wrist"]["reference"][:] = False
        invisible = self.oracle()
        self.assertEqual(invisible.metadata["reference"]["status"], "NOT_VISIBLE")
        self.assertTrue(invisible.metadata["reference"]["semantic_valid"])
        self.assertEqual(invisible.metadata["reference_null_probability"], 0)
        absent = self.oracle(reference_is_null=True)
        self.assertEqual(absent.metadata["reference"]["status"], "NULL")
        self.assertFalse(absent.metadata["reference"]["semantic_valid"])
        self.assertEqual(absent.metadata["reference_null_probability"], 1)
        self.snapshot.cameras["wrist"].depth_m[:] = np.nan
        failed_depth = self.oracle()
        self.assertEqual(failed_depth.metadata["target"]["status"], "LOW_CONFIDENCE")
        self.assertEqual(failed_depth.metadata["target"]["point_count"], 0)
        self.assertEqual(failed_depth.metadata["reference"]["status"], "NOT_VISIBLE")

    def test_oracle_null_cannot_silently_drop_a_reference_mask_or_identity(self):
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.oracle(reference_is_null=True)
        self.masks["wrist"]["reference"][:] = False
        with self.assertRaisesRegex(ValueError, "instance identity"):
            self.oracle(reference_is_null=True, role_instance_ids={"reference": "right"})

    def test_oracle_missing_annotation_is_not_invisibility(self):
        for masks in ({}, {"wrist": {"target": self.masks["wrist"]["target"]}},
                      {"wrist": {**self.masks["wrist"], "background": self.masks["wrist"]["target"]}}):
            with self.subTest(keys=list(masks)), self.assertRaises(ValueError):
                oracle_guidance(self.snapshot, masks, reference_is_null=False, **self.common)

    def test_oracle_mask_resize_or_numeric_coercion_is_forbidden(self):
        for mask in (np.zeros((3, 4), bool), np.ones((7, 9), np.uint8)):
            masks = copy.deepcopy(self.masks)
            masks["wrist"]["target"] = mask
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaisesRegex(ValueError, "boolean shape"):
                oracle_guidance(self.snapshot, masks, reference_is_null=False, **self.common)

    def test_oracle_retains_measured_extent_only_when_all_axes_have_extent(self):
        depth = self.snapshot.cameras["wrist"].depth_m
        depth[2, 2], depth[3, 3] = .9, 1.1
        result = self.oracle()
        self.assertTrue(np.all(np.asarray(result.metadata["target"]["spread_world"]) > 0))
        self.assertTrue(result.metadata["target"]["spread_source"].startswith("measured_world_aabb_half_extents:"))

    def test_oracle_wrist_motion_uses_each_current_camera_pose(self):
        frame = copy.deepcopy(self.snapshot.cameras["wrist"])
        pose = frame.camera.extrinsics.copy()
        pose[0, 3] += .2
        frame.camera = replace(frame.camera, name="front", extrinsics=pose)
        frame.metadata.update(name="front", extrinsics=pose.tolist())
        self.snapshot.cameras["front"] = frame
        self.snapshot.metadata["cameras"].append(copy.deepcopy(frame.metadata))
        self.masks["front"] = copy.deepcopy(self.masks["wrist"])
        result = self.oracle()
        self.assertEqual(len(result.target_points_world), 8)
        np.testing.assert_allclose(result.target_points_world[4:, 0], result.target_points_world[:4, 0] + .2)

    def test_coarse_filters_padding_invalid_and_unsupported_before_inverse(self):
        original = copy.deepcopy(self.transform)
        result = self.coarse()
        result.validate()
        expected = np.array([[1.05, 2.1, 3.15], [1.15, 2.2, 3.25]])
        np.testing.assert_allclose(result.target_points_world, expected)
        np.testing.assert_allclose(result.reference_points_world, expected)
        self.assertEqual(result.metadata["target"]["point_count"], 2)
        self.assertEqual(result.metadata["guidance_source"], "predicted_coarse_slot")
        self.assertEqual(result.metadata["source_checkpoint"], "coarse.pt")
        self.assertIn("point_support_source", result.metadata["transform"]["source"])
        self.assertEqual(self.transform, original)
        self.assertFalse(result.target_points_world.flags.writeable)
        self.output["points"][0, 0] = 99
        np.testing.assert_allclose(result.target_points_world, expected)

    def test_world_origin_is_not_removed_as_renderer_padding(self):
        transform = identity_world_transform(source="actual current world XYZ", timestamp=STAMP, source_frame_index=150)
        result = self.coarse(transform=transform)
        self.assertEqual(len(result.target_points_world), 4)
        self.assertTrue(np.any(np.all(result.target_points_world == 0, axis=1)))
        self.assertTrue(np.any(np.all(result.target_points_world == -1, axis=1)))

    def test_complete_render_normalization_and_crop_inverse_is_applied_once(self):
        # world = (rendered * scene_m / crop_scale + crop_center) / cube_scale + world_center
        scene_m, crop_scale, cube_scale = .8, 2., 4.
        crop_center, world_center = np.array([.2, .3, .4]), np.array([1., 2., 3.])
        matrix = np.eye(4)
        matrix[:3, :3] *= scene_m / crop_scale / cube_scale
        matrix[:3, 3] = crop_center / cube_scale + world_center
        transform = {"source_frame": "actual_coarse_rendered_norm_corr", "target_frame": "rlbench_world",
                     "matrix_world_from_source": matrix.tolist(), "source": "captured actual render/crop/cube composition",
                     "timestamp": STAMP, "source_frame_index": 150}
        result = self.coarse(transform=transform)
        expected = (self.output["points"][0, :2] * scene_m / crop_scale + crop_center) / cube_scale + world_center
        np.testing.assert_allclose(result.target_points_world, expected, atol=1e-15)

    def test_slot_std_geometry_is_never_interpreted_as_range_or_center(self):
        self.output["geometry"][:] = 1000
        result = self.coarse()
        self.assertNotIn("spread_world", result.metadata["target"])
        self.assertNotIn("spread_source", result.metadata["target"])
        np.testing.assert_allclose(result.metadata["target"]["center_world"], [1.1, 2.15, 3.2])

    def test_model_invalidity_does_not_prove_invisibility_or_null(self):
        self.output["valid"][:] = False
        result = self.coarse()
        for role in ("target", "reference"):
            self.assertEqual(result.metadata[role]["status"], "LOW_CONFIDENCE")
            self.assertTrue(result.metadata[role]["semantic_valid"])
            self.assertIsNone(result.metadata[role]["center_world"])
        self.assertEqual(len(result.target_points_world), 2)
        invisible = self.coarse(role_visibility={"reference": "NOT_VISIBLE"})
        self.assertEqual(invisible.metadata["reference"]["status"], "NOT_VISIBLE")
        self.assertEqual(invisible.metadata["target"]["status"], "LOW_CONFIDENCE")

    def test_empty_support_unknown_and_explicit_unseen_are_distinct(self):
        empty = np.zeros_like(self.support)
        result = self.coarse(point_valid=empty)
        self.assertEqual(result.metadata["reference"]["status"], "LOW_CONFIDENCE")
        self.assertEqual(result.metadata["reference"]["point_count"], 0)
        unseen = self.coarse(point_valid=empty, role_visibility={"reference": "NOT_VISIBLE"})
        self.assertEqual(unseen.metadata["reference"]["status"], "NOT_VISIBLE")

    def test_posterior_null_and_explicit_semantics_are_preserved(self):
        self.output["reference_null_probability"] = .8
        result = self.coarse()
        self.assertEqual(result.metadata["reference"]["status"], "NULL")
        self.assertFalse(result.metadata["reference"]["semantic_valid"])
        self.assertEqual(result.reference_points_world.shape, (0, 3))
        self.assertIsNone(result.metadata["reference"]["center_world"])
        self.assertEqual(result.metadata["reference_null_probability"], .8)
        exists = self.coarse(reference_exists=True)
        self.assertEqual(exists.metadata["reference"]["status"], "VISIBLE")
        self.output["reference_null_probability"] = .1
        absent = self.coarse(reference_exists=False)
        self.assertEqual(absent.metadata["reference"]["status"], "NULL")
        self.assertEqual(absent.metadata["reference_null_probability"], .1)

    def test_low_role_confidence_or_token_invalidity_is_not_absence(self):
        self.output["confidence"][0] = .1
        self.output["role_token_valid"][1] = False
        result = self.coarse()
        for role in ("target", "reference"):
            self.assertEqual(result.metadata[role]["status"], "LOW_CONFIDENCE")
            self.assertTrue(result.metadata[role]["semantic_valid"])

    def test_explicit_support_and_unbatched_shapes_are_required(self):
        for support in (np.ones((2, 6), int), np.ones((1, 2, 6), bool), np.ones((2, 5), bool)):
            with self.subTest(shape=support.shape, dtype=support.dtype), self.assertRaises(ValueError):
                self.coarse(point_valid=support)
        self.output["points"] = self.output["points"][None]
        with self.assertRaises(ValueError):
            self.coarse()

    def test_non_detached_tensor_like_input_is_rejected_without_torch_import(self):
        class TensorRequiringGrad:
            requires_grad = True
            def __array__(self, *args, **kwargs):
                raise AssertionError("must reject before trying to convert an attached tensor")
        self.output["points"] = TensorRequiringGrad()
        with self.assertRaisesRegex(ValueError, "detached"):
            self.coarse()

    def test_current_transform_index_time_and_frame_are_required(self):
        for key, value in (("source_frame_index", 3), ("timestamp", "2026-11-05T10:00:01+08:00"),
                           ("target_frame", "camera")):
            transform = copy.deepcopy(self.transform)
            transform[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.coarse(transform=transform)

    def test_snapshot_future_source_and_calibration_metadata_mismatch_rejected(self):
        original = copy.deepcopy(self.snapshot)
        for change in ("source", "time", "calibration", "future_field"):
            self.snapshot = copy.deepcopy(original)
            if change == "source":
                self.snapshot.cameras["wrist"].camera = replace(self.snapshot.cameras["wrist"].camera, source_frame_index=151)
            elif change == "time":
                self.snapshot.metadata["cameras"][0]["observed_at"] = "2026-11-05T10:00:01+08:00"
            elif change == "calibration":
                self.snapshot.cameras["wrist"].metadata["extrinsics"][0][3] += 1
            else:
                self.snapshot.metadata["expert_future_action"] = [1, 2, 3]
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.oracle()

    def test_producers_never_open_files_observations_or_actions(self):
        with patch("builtins.open", side_effect=AssertionError("no file reads")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("no episode reads")):
            self.oracle()
            self.coarse()

    def test_import_does_not_require_torch_or_sam3(self):
        script = "import sys; sys.modules['torch']=None; sys.modules['sam3']=None; import slot_guidance_producers"
        completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
