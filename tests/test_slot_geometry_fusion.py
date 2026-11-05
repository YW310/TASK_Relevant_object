"""Role-preserving metric fusion, frame causality and standalone artifacts."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from camera_geometry import project_points
from slot_geometry_fusion import FusionConfig, fuse_geometry, load_geometry, save_geometry
from slot_guidance import SlotGuidance, identity_world_transform
from slot_guided_io import (FrameSnapshot, RGBDFrame, calibration_digest,
                            guidance_input_digest, load_inference_artifact, save_inference_artifact,
                            snapshot_input_digest)
from slot_guided_sam3 import MockBackend, mask_content_digest, run_camera
from slot_prompt_adapter import CameraFrame


STAMP = "2026-11-05T10:00:00+08:00"


def scene(*, close_roles=False):
    metadata = {"schema_version": 1, "artifact_type": "slot_guidance", "episode_id": "episode0",
                "frame_id": "sample-3", "source_frame_index": 150, "observed_at": STAMP, "created_at": STAMP,
                "instruction": "move the left red block near the right red block", "task_name": "move_block",
                "coordinate_frame": "rlbench_world", "coordinate_units": "m", "guidance_source": "mock",
                "source_checkpoint": None, "source_git_sha": "a" * 40, "reference_null_probability": .03,
                "transform": identity_world_transform(source="synthetic current frame", timestamp=STAMP, source_frame_index=150),
                "points_file": "fixture.npz", "points_sha256": "0" * 64}
    clouds = {}
    for role, sign in (("target", -1), ("reference", 1)):
        center_x = sign * (.03 if close_roles else .15)
        xx, yy = np.meshgrid(np.arange(-2, 3) / 100 + center_x, np.arange(-2, 3) / 100)
        cloud = np.column_stack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
        clouds[role] = cloud
        metadata[role] = {"status": "VISIBLE", "semantic_valid": True, "confidence": .9,
                          "center_world": [center_x, 0, 1], "semantic_label": "red block", "instance_id": role,
                          "points_key": f"{role}_points_world", "point_count": len(cloud)}
    guidance = SlotGuidance(metadata, clouds["target"], clouds["reference"])
    camera_rows, cameras, predictions, masks = [], {}, {}, {}
    k = np.array([[100., 0, 60], [0, 100., 50], [0, 0, 1]])
    for name, translation in (("front", 0), ("wrist", .10), ("left_shoulder", -.10)):
        pose = np.eye(4)
        pose[0, 3] = translation
        row = {"name": name, "intrinsics": k.tolist(), "extrinsics": pose.tolist(),
               "source_frame_index": 150, "observed_at": STAMP,
               "calibration_source": f"observation[150].{name}_camera_extrinsics",
               "rgb_path": f"{name}.png", "depth_path": f"{name}.npy", "depth_encoding": "meters_npy"}
        camera = CameraFrame(name, "episode0", "sample-3", 150, STAMP, k.copy(), pose, (121, 101), row["calibration_source"])
        frame = RGBDFrame(camera, np.zeros((101, 121, 3), dtype=np.uint8), np.ones((101, 121)), copy.deepcopy(row))
        camera_rows.append(copy.deepcopy(row))
        cameras[name] = frame
        role_results, camera_masks = {}, {}
        for role in ("target", "reference"):
            xy, valid = project_points(clouds[role], k, pose)
            assert valid.all()
            rounded = np.rint(xy).astype(int)
            mask = np.zeros((101, 121), dtype=bool)
            mask[rounded[:, 1], rounded[:, 0]] = True
            role_results[role] = {"status": "VISIBLE", "prompt": {"prompt_valid": True},
                "selected_index": 0, "failure_reason": None,
                "candidates": [{"mask": mask, "mask_sha256": mask_content_digest(mask), "sam3_score": .6, "quality_score": .8,
                                "accepted": True, "failure_reasons": [], "metrics": {}}]}
            camera_masks[role] = mask
        predictions[name] = {"camera": name, "roles": role_results}
        masks[name] = camera_masks
    snapshot_metadata = {key: metadata[key] for key in ("episode_id", "frame_id", "source_frame_index", "observed_at")}
    snapshot_metadata.update(schema_version=1, artifact_type="slot_rgbd_frame", cameras=camera_rows)
    # Deliberately nonexistent snapshot/episode paths: fusion only uses arrays.
    snapshot = FrameSnapshot(snapshot_metadata, cameras, Path("/no-episode-read/current_frame.json"))
    inference = {key: metadata[key] for key in ("episode_id", "frame_id", "source_frame_index", "observed_at",
                                               "coordinate_frame", "coordinate_units", "guidance_source")}
    inference.update(schema_version=1, artifact_type="slot_sam3_frame", cameras=predictions,
                     cache_key="b" * 64, mode="point+box", guidance_metadata=copy.deepcopy(metadata),
                     backend={name: {"kind": "mock", "frozen": True} for name in cameras},
                     calibration_sha256=calibration_digest(snapshot),
                     snapshot_input_sha256=snapshot_input_digest(snapshot),
                     guidance_input_sha256=guidance_input_digest(guidance))
    return snapshot, guidance, inference, masks


class SlotGeometryFusionTests(unittest.TestCase):
    def setUp(self):
        self.snapshot, self.guidance, self.inference, self.masks = scene()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "geometry.json"

    def fuse(self, config=None):
        return fuse_geometry(self.snapshot, self.guidance, self.inference, config)

    def test_metric_centers_bounds_and_source_index_from_three_original_cameras(self):
        result = self.fuse(FusionConfig(min_support_cameras=2))
        self.assertEqual(result["source_frame_index"], 150)
        self.assertEqual(result["frame_id"], "sample-3")
        self.assertEqual(result["coordinate_frame"], "rlbench_world")
        for role in ("target", "reference"):
            row = result["roles"][role]
            cloud = result["points"][row["points_key"]]
            self.assertIsNone(row["failure_reason"])
            self.assertEqual(row["support_cameras"], ["front", "left_shoulder", "wrist"])
            np.testing.assert_allclose(row["center_world"], self.guidance.metadata[role]["center_world"], atol=1e-12)
            np.testing.assert_allclose(row["bbox3d_world"], [cloud.min(axis=0), cloud.max(axis=0)])
            self.assertGreater(row["geometry_confidence"], 0)
            self.assertAlmostEqual(row["role_confidence"], .9)
            self.assertAlmostEqual(row["cross_view_consistency"], 1)

    def test_joint_search_uses_consistent_lower_sam_candidate_not_selected_index(self):
        row = self.inference["cameras"]["front"]["roles"]["target"]
        correct = row["candidates"][0]
        wrong = copy.deepcopy(correct)
        wrong["mask"] = np.roll(correct["mask"], 8, axis=1)
        wrong["mask_sha256"] = mask_content_digest(wrong["mask"])
        wrong.update(sam3_score=.99, quality_score=.99)
        row["candidates"] = [wrong, correct]
        row["selected_index"] = 0  # Camera-local preference is deliberately wrong.
        result = self.fuse(FusionConfig(min_support_cameras=2, cross_view_distance_m=.015))
        selected = result["roles"]["target"]["selected_candidates"]
        self.assertIn({"camera": "front", "candidate_index": 1}, selected)
        self.assertNotIn({"camera": "front", "candidate_index": 0}, selected)
        diag = result["roles"]["target"]["candidate_diagnostics"]
        self.assertEqual(len(diag), 4)
        self.assertEqual(diag[0]["failure_reason"], "NOT_IN_CONSISTENT_JOINT_SELECTION")
        np.testing.assert_allclose(result["roles"]["target"]["center_world"], [-.15, 0, 1], atol=1e-12)

    def test_spatially_inconsistent_view_is_excluded_not_concatenated(self):
        row = self.inference["cameras"]["wrist"]["roles"]["target"]
        row["candidates"][0]["mask"] = np.roll(row["candidates"][0]["mask"], 15, axis=1)
        row["candidates"][0]["mask_sha256"] = mask_content_digest(row["candidates"][0]["mask"])
        result = self.fuse(FusionConfig(min_support_cameras=2))
        self.assertEqual(result["roles"]["target"]["support_cameras"], ["front", "left_shoulder"])
        self.assertLess(np.max(result["points"]["target_points_world"][:, 0]), 0)

    def test_close_same_category_roles_never_merge(self):
        snapshot, guidance, inference, _ = scene(close_roles=True)
        result = fuse_geometry(snapshot, guidance, inference)
        self.assertEqual(result["roles"]["target"]["instance_id"], "target")
        self.assertEqual(result["roles"]["reference"]["instance_id"], "reference")
        self.assertTrue(np.all(result["points"]["target_points_world"][:, 0] < 0))
        self.assertTrue(np.all(result["points"]["reference_points_world"][:, 0] > 0))

    def test_semantic_unavailability_preserved_without_false_null(self):
        for status in ("NULL", "NOT_VISIBLE", "LOW_CONFIDENCE"):
            with self.subTest(status=status):
                snapshot, guidance, inference, _ = scene()
                metadata = guidance.metadata
                metadata["reference"] = {"status": status, "semantic_valid": status != "NULL", "confidence": .1,
                                         "center_world": None, "points_key": "reference_points_world", "point_count": 0}
                guidance = SlotGuidance(metadata, guidance.target_points_world, np.empty((0, 3)))
                inference["guidance_metadata"] = copy.deepcopy(metadata)
                inference["guidance_input_sha256"] = guidance_input_digest(guidance)
                for camera in inference["cameras"].values():
                    camera["roles"]["reference"].update(status=status, candidates=[])
                result = fuse_geometry(snapshot, guidance, inference)
                row = result["roles"]["reference"]
                self.assertEqual(row["status"], status)
                self.assertEqual(row["semantic_valid"], status != "NULL")
                self.assertEqual(row["failure_reason"], status)
                self.assertEqual(result["points"]["reference_points_world"].shape, (0, 3))
                save_geometry(self.path, result)
                self.assertEqual(load_geometry(self.path)["roles"]["reference"]["status"], status)

    def test_empty_predictions_and_all_rejected_candidates_are_explicit(self):
        for camera in self.inference["cameras"].values():
            camera["roles"]["target"]["candidates"] = []
            camera["roles"]["reference"]["candidates"][0].update(accepted=False, failure_reasons=["empty_mask"])
        result = self.fuse()
        for role in ("target", "reference"):
            row = result["roles"][role]
            self.assertEqual(row["status"], "VISIBLE")
            self.assertTrue(row["semantic_valid"])
            self.assertEqual(row["failure_reason"], "NO_ACCEPTED_CANDIDATES")
            self.assertEqual(row["geometry_confidence"], 0)
            self.assertIsNone(row["center_world"])

    def test_invalid_depth_and_accepted_empty_masks_are_rechecked(self):
        for camera in self.snapshot.cameras.values():
            camera.depth_m[:] = np.nan
        self.inference["snapshot_input_sha256"] = snapshot_input_digest(self.snapshot)
        result = self.fuse()
        self.assertEqual(result["roles"]["target"]["point_count"], 0)
        self.assertEqual(result["roles"]["target"]["candidate_diagnostics"][0]["failure_reason"], "INSUFFICIENT_DEPTH_SUPPORT")
        for camera in self.snapshot.cameras.values():
            camera.depth_m[:] = 1
        self.inference["snapshot_input_sha256"] = snapshot_input_digest(self.snapshot)
        for camera in self.inference["cameras"].values():
            camera["roles"]["target"]["candidates"][0]["mask"][:] = False
            camera["roles"]["target"]["candidates"][0]["mask_sha256"] = mask_content_digest(camera["roles"]["target"]["candidates"][0]["mask"])
        result = self.fuse()
        self.assertEqual(result["roles"]["target"]["candidate_diagnostics"][0]["failure_reason"], "EMPTY_MASK")

    def test_required_camera_count_fails_closed_without_geometry(self):
        result = self.fuse(FusionConfig(min_support_cameras=4))
        self.assertEqual(result["roles"]["target"]["failure_reason"], "INSUFFICIENT_CONSISTENT_CAMERAS")
        self.assertEqual(result["points"]["target_points_world"].shape, (0, 3))

    def test_search_limit_never_silently_truncates_by_sam_score(self):
        result = self.fuse(FusionConfig(max_joint_hypotheses=1))
        self.assertEqual(result["roles"]["target"]["failure_reason"], "CANDIDATE_SEARCH_LIMIT_EXCEEDED")
        self.assertEqual(len(result["roles"]["target"]["candidate_diagnostics"]), 3)

    def test_quantile_and_voxel_filters_bound_outlier_depth(self):
        for camera in self.snapshot.cameras.values():
            mask = self.masks[camera.camera.name]["target"]
            yy, xx = np.nonzero(mask)
            camera.depth_m[yy[-1], xx[-1]] = 100
        self.inference["snapshot_input_sha256"] = snapshot_input_digest(self.snapshot)
        result = self.fuse(FusionConfig(quantile_low=.05, quantile_high=.95))
        self.assertLess(result["roles"]["target"]["bbox3d_world"][1][2], 2)

    def test_dynamic_wrist_pose_and_frame_binding_are_exact(self):
        # The wrist translation is not shared with static cameras. Using it is
        # required for the identical world centroids tested above.
        self.assertAlmostEqual(self.snapshot.cameras["wrist"].camera.extrinsics[0, 3], .1)
        self.snapshot.cameras["wrist"].camera = replace(self.snapshot.cameras["wrist"].camera, source_frame_index=3)
        with self.assertRaisesRegex(ValueError, "source_frame_index"):
            self.fuse()

    def test_stale_camera_transform_cannot_hide_behind_old_digest(self):
        self.snapshot.cameras["wrist"].camera.extrinsics[0, 3] = .2
        with self.assertRaisesRegex(ValueError, "extrinsics"):
            self.fuse()

    def test_changed_calibration_manifest_invalidates_inference(self):
        self.snapshot.cameras["wrist"].metadata["extrinsics"][0][3] = .2
        with self.assertRaisesRegex(ValueError, "calibration digest"):
            self.fuse()

    def test_foreign_episode_future_frame_and_future_timestamp_fail(self):
        for key, value in (("episode_id", "other_world"), ("source_frame_index", 151), ("observed_at", "2026-11-05T10:00:01+08:00")):
            with self.subTest(key=key):
                inference = copy.deepcopy(self.inference)
                inference[key] = value
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    fuse_geometry(self.snapshot, self.guidance, inference)

    def test_changed_role_provenance_and_mixed_backend_rejected(self):
        inference = copy.deepcopy(self.inference)
        inference["guidance_metadata"]["reference"]["instance_id"] = "another same-category instance"
        with self.assertRaisesRegex(ValueError, "guidance provenance"):
            fuse_geometry(self.snapshot, self.guidance, inference)
        inference = copy.deepcopy(self.inference)
        inference["backend"]["wrist"] = {"kind": "sam3", "checkpoint_sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "consistent"):
            fuse_geometry(self.snapshot, self.guidance, inference)

    def test_real_checkpoint_digests_are_provenance_not_mock_metrics(self):
        checkpoint = Path(self.directory.name) / "coarse.pt"
        checkpoint.write_bytes(b"synthetic producer checkpoint digest test")
        self.guidance.metadata.update(guidance_source="predicted_coarse_slot", source_checkpoint=str(checkpoint))
        self.inference["guidance_metadata"] = copy.deepcopy(self.guidance.metadata)
        self.inference["guidance_input_sha256"] = guidance_input_digest(self.guidance)
        self.inference["guidance_source"] = "predicted_coarse_slot"
        for name in self.inference["backend"]:
            self.inference["backend"][name] = {"kind": "sam3", "checkpoint_sha256": "c" * 64}
        result = self.fuse()
        self.assertEqual(result["source_checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertEqual(result["sam3_checkpoint_sha256"], "c" * 64)

    def test_actual_mock_prompt_inference_to_fusion_path(self):
        for name, frame in self.snapshot.cameras.items():
            backend = MockBackend({role: (mask[None], [.6]) for role, mask in self.masks[name].items()})
            self.inference["cameras"][name] = run_camera(self.guidance, frame.camera, frame.rgb, frame.depth_m, backend)
        result = self.fuse(FusionConfig(min_support_cameras=2))
        self.assertGreater(result["roles"]["target"]["point_count"], 0)
        self.assertGreater(result["roles"]["reference"]["point_count"], 0)

    def test_changed_accepted_masks_cannot_reuse_old_acceptance(self):
        for camera in self.inference["cameras"].values():
            target = camera["roles"]["target"]["candidates"][0]
            target["mask"] = camera["roles"]["reference"]["candidates"][0]["mask"].copy()
        with self.assertRaisesRegex(ValueError, "mask content digest mismatch"):
            self.fuse()

    def test_missing_mask_digest_and_nonboolean_acceptance_are_rejected(self):
        candidate = self.inference["cameras"]["front"]["roles"]["target"]["candidates"][0]
        digest = candidate.pop("mask_sha256")
        with self.assertRaisesRegex(ValueError, "mask content digest mismatch"):
            self.fuse()
        candidate["mask_sha256"] = digest
        candidate["accepted"] = 1
        with self.assertRaisesRegex(ValueError, "accepted state must be a boolean"):
            self.fuse()

    def test_scored_mask_digest_survives_inference_artifact_roundtrip(self):
        for name, frame in self.snapshot.cameras.items():
            backend = MockBackend({role: (mask[None], [.6]) for role, mask in self.masks[name].items()})
            self.inference["cameras"][name] = run_camera(self.guidance, frame.camera, frame.rgb, frame.depth_m, backend)
        inference_path = self.path.parent / "inference.json"
        save_inference_artifact(inference_path, self.inference, self.snapshot)
        loaded = load_inference_artifact(inference_path)
        result = fuse_geometry(self.snapshot, self.guidance, loaded)
        self.assertGreater(result["roles"]["target"]["point_count"], 0)
        for camera in loaded["cameras"].values():
            for role in ("target", "reference"):
                candidate = camera["roles"][role]["candidates"][0]
                self.assertEqual(candidate["mask_sha256"], mask_content_digest(candidate["mask"]))

    def test_stale_depth_rgb_and_guidance_points_are_detected(self):
        for change in ("depth", "rgb", "guidance_points"):
            with self.subTest(change=change):
                snapshot, guidance, inference, _ = scene()
                if change == "depth":
                    snapshot.cameras["wrist"].depth_m[0, 0] += .01
                elif change == "rgb":
                    snapshot.cameras["wrist"].rgb[0, 0] = [12, 34, 56]
                else:
                    guidance.target_points_world[0, 0] += .01
                with self.assertRaisesRegex(ValueError, "input digest"):
                    fuse_geometry(snapshot, guidance, inference)

    def test_whole_trajectory_metadata_is_not_accepted_as_current_snapshot(self):
        self.snapshot.metadata["future_observations"] = [{"gripper_pose": [0, 0, 1]}]
        with self.assertRaisesRegex(ValueError, "single-observation"):
            self.fuse()

    def test_json_npz_roundtrip_preserves_fixed_role_arrays_and_readonly_boundary(self):
        result = self.fuse()
        metadata = save_geometry(self.path, result)
        self.assertNotIn("points", metadata)
        loaded = load_geometry(self.path, expected_source_frame_index=150, expected_episode_id="episode0",
                               expected_frame_id="sample-3", expected_observed_at=STAMP,
                               expected_calibration_sha256=calibration_digest(self.snapshot))
        self.assertEqual(metadata["geometry_path"], f"geometry.{metadata['geometry_sha256']}.npz")
        for role in ("target", "reference"):
            key = f"{role}_points_world"
            np.testing.assert_array_equal(result["points"][key], loaded["points"][key])
            self.assertFalse(loaded["points"][key].flags.writeable)
        with self.assertRaisesRegex(ValueError, "source_frame_index mismatch"):
            load_geometry(self.path, expected_source_frame_index=3)

    def test_corrupt_npz_digest_and_local_path_escape_rejected(self):
        metadata = save_geometry(self.path, self.fuse())
        sidecar = self.path.parent / metadata["geometry_path"]
        sidecar.write_bytes(sidecar.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            load_geometry(self.path)
        metadata["geometry_path"] = "../untrusted.npz"
        self.path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "basename"):
            load_geometry(self.path)

    def test_geometry_counts_centroid_bounds_semantic_flags_and_confidence_checked(self):
        original = self.fuse()
        changes = (("point_count", 999), ("center_world", [0, 0, 0]), ("bbox3d_world", [[0, 0, 0], [1, 1, 1]]),
                   ("semantic_valid", False), ("geometry_confidence", float("nan")), ("support_cameras", []))
        for key, value in changes:
            with self.subTest(key=key):
                result = copy.deepcopy(original)
                result["roles"]["target"][key] = value
                with self.assertRaises(ValueError):
                    save_geometry(self.path, result)

    def test_npz_nonfinite_or_extra_arrays_rejected_even_when_digest_matches(self):
        result = self.fuse()
        metadata = save_geometry(self.path, result)
        for change in ("nonfinite", "extra_array"):
            with self.subTest(change=change):
                points = {key: value.copy() for key, value in result["points"].items()}
                if change == "nonfinite":
                    points["target_points_world"][0, 0] = np.nan
                else:
                    points["future_action"] = np.zeros((1, 3))
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **points)
                payload = buffer.getvalue()
                digest = hashlib.sha256(payload).hexdigest()
                metadata.update(geometry_sha256=digest, geometry_path=f"geometry.{digest}.npz")
                (self.path.parent / metadata["geometry_path"]).write_bytes(payload)
                self.path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_geometry(self.path)

    def test_bad_config_rejected(self):
        for kwargs in ({"min_support_cameras": 0}, {"voxel_size_m": 0}, {"distance_percentile": 101},
                       {"min_points_per_object": True}, {"quantile_low": .7}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FusionConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
