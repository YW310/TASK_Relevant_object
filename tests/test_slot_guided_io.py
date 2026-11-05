"""Single-frame RGB-D -> mock SAM3 artifacts, cache and temporal boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from slot_guidance import identity_world_transform, save_guidance
from slot_guided_io import load_frame_snapshot, load_inference_artifact, run_offline_frame
from slot_guided_sam3 import MockBackend, SelectionConfig
from slot_prompt_adapter import AdapterConfig


STAMP = "2026-11-05T10:00:00+08:00"


class SlotGuidedIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.snapshot_path, self.guidance_path = self.root / "frame.json", self.root / "guidance.json"
        self.output_dir = self.root / "artifacts"
        self.rgb = np.zeros((101, 121, 3), np.uint8)
        self.depth = np.ones((101, 121), np.float64)
        self.snapshot = {
            "schema_version": 1, "artifact_type": "slot_rgbd_frame", "episode_id": "episode0",
            "frame_id": "sample-3", "source_frame_index": 150, "observed_at": STAMP,
            "cameras": [{"name": "wrist", "rgb_path": "rgb.png", "depth_path": "depth.npy",
                         "depth_encoding": "meters_npy", "source_frame_index": 150, "observed_at": STAMP,
                         "calibration_source": "current low_dim_obs[150].misc",
                         "intrinsics": [[100., 0, 60], [0, 100., 50], [0, 0, 1]],
                         "extrinsics": np.eye(4).tolist()}],
        }
        self.metadata = {
            "schema_version": 1, "artifact_type": "slot_guidance", "episode_id": "episode0",
            "frame_id": "sample-3", "source_frame_index": 150, "observed_at": STAMP, "created_at": STAMP,
            "instruction": "move the left red block beside the right red block", "task_name": "move_block",
            "coordinate_frame": "rlbench_world", "coordinate_units": "m", "guidance_source": "mock",
            "source_checkpoint": None, "source_git_sha": "a" * 40, "reference_null_probability": .03,
            "transform": identity_world_transform(source="synthetic current snapshot", timestamp=STAMP, source_frame_index=150),
        }
        self.clouds = {}
        for role, x, identity in (("target", -.15, "left"), ("reference", .15, "right")):
            xx, yy = np.meshgrid(np.linspace(x - .02, x + .02, 5), np.linspace(-.02, .02, 5))
            self.clouds[f"{role}_points_world"] = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
            self.metadata[role] = {"status": "VISIBLE", "semantic_valid": True, "confidence": .9,
                                   "center_world": [x, 0, 1], "semantic_label": "red block", "instance_id": identity}
        self.target_mask = np.zeros(self.depth.shape, bool)
        self.target_mask[48:53, 43:48] = True
        self.reference_mask = np.zeros(self.depth.shape, bool)
        self.reference_mask[48:53, 73:78] = True
        self.write_inputs()

    def write_snapshot(self, value=None) -> None:
        self.snapshot_path.write_text(json.dumps(self.snapshot if value is None else value), encoding="utf-8")

    def write_inputs(self) -> None:
        Image.fromarray(self.rgb).save(self.root / "rgb.png")
        np.save(self.root / "depth.npy", self.depth)
        self.write_snapshot()
        save_guidance(self.guidance_path, self.metadata, **self.clouds, overwrite=True)

    def backend(self) -> MockBackend:
        return MockBackend({"target": (np.stack([self.reference_mask, self.target_mask, np.zeros_like(self.target_mask)]),
                                        np.array([.99, .2, .98])),
                            "reference": (self.reference_mask[None], np.array([.8]))})

    def run_frame(self, backend=None, **kwargs):
        return run_offline_frame(self.snapshot_path, self.guidance_path, backend or self.backend(), self.output_dir, **kwargs)

    def test_original_resolution_multicandidate_artifact_and_diagnostics_roundtrip(self):
        backend = self.backend()
        result = self.run_frame(backend)
        self.assertFalse(result["cache_hit"])
        self.assertEqual(backend.set_image_calls, 1)
        self.assertEqual(len(backend.predict_calls), 2)
        self.assertEqual(backend.predict_calls[0]["state_id"], backend.predict_calls[1]["state_id"])
        self.assertEqual(result["source_frame_index"], 150)
        self.assertEqual(result["frame_id"], "sample-3")
        camera = result["cameras"]["wrist"]
        self.assertEqual(camera["roles"]["target"]["selected_index"], 1)
        self.assertEqual(len(camera["roles"]["target"]["candidates"]), 3)
        path = Path(result["artifact_path"])
        loaded = load_inference_artifact(path, expected_cache_key=result["cache_key"])
        row = loaded["cameras"]["wrist"]["roles"]["target"]
        np.testing.assert_array_equal(row["candidates"][0]["mask"], self.reference_mask)
        np.testing.assert_array_equal(row["candidates"][1]["mask"], self.target_mask)
        self.assertFalse(row["candidates"][0]["accepted"])
        self.assertTrue(row["candidates"][1]["accepted"])
        self.assertEqual([candidate["sam3_score"] for candidate in row["candidates"]], [.99, .2, .98])
        for candidate in row["candidates"]:
            self.assertEqual(candidate["mask"].shape, self.depth.shape)
            self.assertEqual(candidate["mask"].dtype, bool)
        diagnostic = path.parent / loaded["cameras"]["wrist"]["diagnostic_image"]
        self.assertTrue(diagnostic.is_file())
        with Image.open(diagnostic) as image:
            self.assertEqual(image.size, (121, 101))
            self.assertTrue(np.any(np.array(image) != self.rgb))

    def test_cache_hit_does_not_repeat_image_encoder_or_rewrite_artifacts(self):
        backend = self.backend()
        first = self.run_frame(backend)
        artifact = Path(first["artifact_path"])
        serialized = artifact.read_bytes()
        diagnostic = artifact.parent / "wrist.diagnostic.png"
        diagnostic_bytes = diagnostic.read_bytes()
        second = self.run_frame(backend)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["cache_key"], second["cache_key"])
        self.assertEqual(backend.set_image_calls, 1)
        self.assertEqual(len(backend.predict_calls), 2)
        self.assertEqual(artifact.read_bytes(), serialized)
        self.assertEqual(diagnostic.read_bytes(), diagnostic_bytes)
        np.testing.assert_array_equal(second["cameras"]["wrist"]["roles"]["target"]["candidates"][1]["mask"], self.target_mask)

    def test_input_calibration_backend_mode_and_config_changes_invalidate_cache(self):
        baseline = self.run_frame()
        baseline_path = Path(baseline["artifact_path"])
        baseline_files = {item: item.read_bytes() for item in baseline_path.parent.iterdir() if item.is_file()}
        cases = ("depth", "rgb", "calibration", "mode", "mock_masks", "mock_scores", "adapter_config", "selection_config")
        for case in cases:
            with self.subTest(change=case):
                self.write_inputs()
                backend, kwargs = self.backend(), {}
                if case == "depth":
                    np.save(self.root / "depth.npy", self.depth + .005)
                elif case == "rgb":
                    rgb = self.rgb.copy()
                    rgb[0, 0] = [1, 2, 3]
                    Image.fromarray(rgb).save(self.root / "rgb.png")
                elif case == "calibration":
                    snapshot = copy.deepcopy(self.snapshot)
                    snapshot["cameras"][0]["extrinsics"][0][3] = .001
                    self.write_snapshot(snapshot)
                elif case == "mode":
                    kwargs["mode"] = "box-only"
                elif case == "mock_masks":
                    backend.candidates["target"][0][2, 0, 0] = True
                elif case == "mock_scores":
                    backend.candidates["target"][1][1] = .21
                elif case == "adapter_config":
                    kwargs["adapter_config"] = AdapterConfig(padding_px=5.)
                elif case == "selection_config":
                    kwargs["selection_config"] = SelectionConfig(min_quality=.6)
                result = self.run_frame(backend, **kwargs)
                self.assertFalse(result["cache_hit"])
                self.assertNotEqual(result["cache_key"], baseline["cache_key"])
                self.assertEqual(backend.set_image_calls, 1)
                self.assertTrue(baseline_path.is_file())
                for item, content in baseline_files.items():
                    self.assertEqual(item.read_bytes(), content)

    def test_disabling_cache_explicitly_recomputes_encoder(self):
        backend = self.backend()
        first = self.run_frame(backend)
        second = self.run_frame(backend, use_cache=False)
        self.assertFalse(second["cache_hit"])
        self.assertEqual(first["cache_key"], second["cache_key"])
        self.assertEqual(backend.set_image_calls, 2)

    def test_corrupt_mask_digest_fails_instead_of_silently_reusing_cache(self):
        backend = self.backend()
        first = self.run_frame(backend)
        artifact = Path(first["artifact_path"])
        value = json.loads(artifact.read_text(encoding="utf-8"))
        mask_path = artifact.parent / value["masks_file"]
        mask_path.write_bytes(mask_path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.run_frame(backend)
        self.assertEqual(backend.set_image_calls, 1)

    def test_corrupt_guidance_digest_stops_before_encoding(self):
        metadata = json.loads(self.guidance_path.read_text(encoding="utf-8"))
        sidecar = self.guidance_path.parent / metadata["points_file"]
        sidecar.write_bytes(sidecar.read_bytes() + b"mismatched points")
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.run_frame(backend)
        self.assertEqual(backend.set_image_calls, 0)

    def test_sample_ordinal_or_future_calibration_is_not_source_frame(self):
        for change in ("sample_ordinal", "future_index", "future_timestamp", "extra_future_observation"):
            with self.subTest(change=change):
                snapshot = copy.deepcopy(self.snapshot)
                if change == "sample_ordinal":
                    snapshot["cameras"][0]["source_frame_index"] = 3
                elif change == "future_index":
                    snapshot["cameras"][0]["source_frame_index"] = 151
                elif change == "future_timestamp":
                    snapshot["cameras"][0]["observed_at"] = "2026-11-05T10:00:01+08:00"
                else:
                    snapshot["future_observation"] = {"gripper_pose": [0, 0, 1]}
                self.write_snapshot(snapshot)
                backend = self.backend()
                with self.assertRaises(ValueError):
                    self.run_frame(backend)
                self.assertEqual(backend.set_image_calls, 0)

    def test_guidance_snapshot_current_observation_binding_is_exact(self):
        for key, value in (("episode_id", "episode1"), ("frame_id", "150"),
                           ("source_frame_index", 151), ("observed_at", "2026-11-05T10:00:01+08:00")):
            with self.subTest(key=key):
                snapshot = copy.deepcopy(self.snapshot)
                snapshot[key] = value
                if key in ("source_frame_index", "observed_at"):
                    snapshot["cameras"][0][key] = value
                self.write_snapshot(snapshot)
                backend = self.backend()
                with self.assertRaises(ValueError):
                    self.run_frame(backend)
                self.assertEqual(backend.set_image_calls, 0)

    def test_snapshot_expected_source_index_is_checked_at_load(self):
        self.assertEqual(load_frame_snapshot(self.snapshot_path, expected_source_frame_index=150).cameras["wrist"].camera.image_size,
                         (121, 101))
        with self.assertRaisesRegex(ValueError, "requested observation"):
            load_frame_snapshot(self.snapshot_path, expected_source_frame_index=3)

    def test_snapshot_schema_and_acquisition_timestamp_are_strict(self):
        for key, value in (("schema_version", True), ("observed_at", "yesterday"),
                           ("observed_at", "2026-11-05T10:00:00")):
            with self.subTest(key=key, value=value):
                snapshot = copy.deepcopy(self.snapshot)
                snapshot[key] = value
                if key == "observed_at":
                    snapshot["cameras"][0][key] = value
                self.write_snapshot(snapshot)
                with self.assertRaises(ValueError):
                    load_frame_snapshot(self.snapshot_path)

    def test_packed_rlbench_depth_uses_explicit_current_near_far(self):
        near, far = .1, 2.
        encoded = round((1. - near) / (far - near) * (2 ** 24 - 1))
        packed = np.zeros(self.rgb.shape, dtype=np.uint8)
        packed[:] = [(encoded >> 16) & 255, (encoded >> 8) & 255, encoded & 255]
        Image.fromarray(packed).save(self.root / "depth24.png")
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["cameras"][0].update(depth_path="depth24.png", depth_encoding="rlbench_rgb24", near=near, far=far)
        self.write_snapshot(snapshot)
        np.testing.assert_allclose(load_frame_snapshot(self.snapshot_path).cameras["wrist"].depth_m, 1., atol=2e-7)
        snapshot["cameras"][0]["far"] = None
        self.write_snapshot(snapshot)
        with self.assertRaisesRegex(ValueError, "near < far"):
            load_frame_snapshot(self.snapshot_path)

    def test_missing_calibration_is_saved_as_explicit_per_role_failure(self):
        snapshot = copy.deepcopy(self.snapshot)
        del snapshot["cameras"][0]["extrinsics"]
        self.write_snapshot(snapshot)
        backend = self.backend()
        result = self.run_frame(backend)
        self.assertEqual(backend.set_image_calls, 0)
        for row in result["cameras"]["wrist"]["roles"].values():
            self.assertEqual(row["failure_reason"], "MISSING_CALIBRATION")
            self.assertEqual(row["candidates"], [])
            self.assertIsNone(row["selected_index"])
        self.assertTrue((Path(result["artifact_path"]).parent / "wrist.diagnostic.png").is_file())

    def test_null_reference_status_and_failure_survive_artifact_reload(self):
        self.metadata["reference"] = {"status": "NULL", "semantic_valid": False, "confidence": .99, "center_world": None}
        self.clouds["reference_points_world"] = np.empty((0, 3))
        self.metadata["reference_null_probability"] = .99
        self.write_inputs()
        backend = self.backend()
        result = self.run_frame(backend)
        loaded = load_inference_artifact(result["artifact_path"])
        row = loaded["cameras"]["wrist"]["roles"]["reference"]
        self.assertEqual(row["status"], "NULL")
        self.assertEqual(row["failure_reason"], "NULL")
        self.assertEqual(row["candidates"], [])
        self.assertEqual(backend.set_image_calls, 1)
        self.assertEqual([call["role"] for call in backend.predict_calls], ["target"])


if __name__ == "__main__":
    unittest.main()
