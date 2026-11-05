"""PR1 coordinate contracts; no SAM3, simulator, or action inference required.

Set BRIDGEVLA_REPO to a local BridgeVLA checkout to additionally compare against
its actual place_pc_in_cube and bundled VisionSensor implementations. Those
optional comparisons load source files directly and never start RLBench.
"""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from camera_geometry import (
    backproject_mask,
    cloud_observability_in_camera,
    frame_index_from_frame,
    normalize_extrinsics,
    project_points,
    resolve_camera_param_for_frame,
    transform_points,
)
from slot_guidance import (
    bridgevla_cube_transform,
    identity_world_transform,
    points_to_world,
    validate_guidance,
)


TIMESTAMP = "2026-11-05T08:00:00+08:00"


def _pose() -> np.ndarray:
    """Camera-to-world pose with rotation about two axes and translation."""
    ax, az = 0.37, -0.61
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = rz @ rx
    pose[:3, 3] = [0.42, -0.23, 0.81]
    return pose


def _camera_transform() -> dict:
    transform = identity_world_transform(
        source="synthetic current-frame camera calibration", timestamp=TIMESTAMP,
        source_frame_index=150,
    )
    transform.update(source_frame="camera_optical", matrix_world_from_source=_pose().tolist())
    return transform


def _metadata() -> dict:
    """Valid detached metadata; storage placeholders suffice for validation."""
    return {
        "schema_version": 1, "artifact_type": "slot_guidance",
        "episode_id": "synthetic-episode", "frame_id": "sample-003",
        "source_frame_index": 150, "instruction": "pick the target",
        "task_name": "synthetic", "coordinate_frame": "rlbench_world",
        "coordinate_units": "m", "observed_at": TIMESTAMP,
        "created_at": TIMESTAMP, "guidance_source": "mock",
        "source_checkpoint": None, "source_git_sha": "1" * 40,
        "transform": identity_world_transform(
            source="synthetic world points", timestamp=TIMESTAMP,
            source_frame_index=150,
        ),
        "target": {
            "status": "VISIBLE", "semantic_valid": True, "confidence": 1.0,
            "center_world": [0.1, 0.2, 0.3],
            "points_key": "target_points_world", "point_count": 0,
        },
        "reference": {
            "status": "NULL", "semantic_valid": False, "confidence": 1.0,
            "center_world": None, "points_key": "reference_points_world",
            "point_count": 0,
        },
        "reference_null_probability": 1.0,
        "points_file": "synthetic.npz", "points_sha256": "0" * 64,
    }


def _empty_clouds() -> dict[str, np.ndarray]:
    return {f"{role}_points_world": np.empty((0, 3)) for role in ("target", "reference")}


def _load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load comparison source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GuidanceCameraContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = np.array([[123.0, 0, 2.7], [0, 97.0, 1.3], [0, 0, 1]])

    def test_rotated_translated_projection_backprojection_round_trip(self) -> None:
        depth = np.linspace(0.45, 1.8, 48).reshape(6, 8)
        camera_points = backproject_mask(depth, np.ones_like(depth, dtype=bool),
                                         self.intrinsics, max_points=0)
        world = points_to_world(camera_points, _camera_transform())
        np.testing.assert_allclose(world, transform_points(camera_points, _pose()), atol=1e-12)
        pixels, valid = project_points(world, self.intrinsics, _pose())
        yy, xx = np.indices(depth.shape)
        np.testing.assert_allclose(pixels, np.stack([xx.ravel(), yy.ravel()], axis=1), atol=1e-12)
        self.assertTrue(valid.all())
        self.assertEqual(world.shape, (48, 3))

    def test_invalid_depth_does_not_become_object_geometry(self) -> None:
        depth = np.array([[np.nan, np.inf, -1.0], [0.0, -np.inf, 2.0]])
        points = backproject_mask(depth, np.ones_like(depth, dtype=bool),
                                  self.intrinsics, max_points=0)
        self.assertEqual(points.shape, (1, 3))
        np.testing.assert_allclose(points[0], [(2 - 2.7) * 2 / 123, (1 - 1.3) * 2 / 97, 2])
        empty = backproject_mask(depth, np.zeros_like(depth, dtype=bool),
                                 self.intrinsics, max_points=0)
        self.assertEqual(empty.shape, (0, 3))

    def test_projection_validity_checks_depth_but_not_image_bounds(self) -> None:
        camera_points = np.array([[0, 0, -1], [0, 0, 0], [100, 0, 1], [0, 0, 1]])
        world = transform_points(camera_points, _pose())
        pixels, valid = project_points(world, self.intrinsics, _pose())
        np.testing.assert_array_equal(valid, [False, False, True, True])
        self.assertGreater(pixels[2, 0], 8)
        # Existing projection validity alone is insufficient for a PR2 prompt.
        visible = cloud_observability_in_camera(
            world[2:3], {"intrinsics": self.intrinsics, "extrinsics": _pose(),
                         "depth": np.ones((6, 8))}, 0.03, 0.5,
        )
        self.assertFalse(visible["observable"])
        self.assertEqual(visible["reason"], "outside_image")

    def test_invalid_scene_depth_has_explicit_visibility_failure(self) -> None:
        world = transform_points(np.array([[0, 0, 1.0]]), _pose())
        context = {"intrinsics": self.intrinsics, "extrinsics": _pose(),
                   "depth": np.full((6, 8), np.nan)}
        result = cloud_observability_in_camera(world, context, 0.03, 0.5)
        self.assertFalse(result["observable"])
        self.assertEqual(result["reason"], "missing_depth")
        context["depth"] = np.full((6, 8), 0.4)
        result = cloud_observability_in_camera(world, context, 0.03, 0.5)
        self.assertFalse(result["observable"])
        self.assertEqual(result["reason"], "occluded_by_depth")

    def test_wrist_reads_only_exact_source_index_with_no_future_observation(self) -> None:
        calls = []

        class CurrentObservationOnly:
            def __init__(self, current_index, calibration):
                self.current_index = current_index
                self.calibration = calibration

            def __len__(self):
                return 200

            def __getitem__(self, index):
                calls.append(index)
                if index != self.current_index:
                    raise AssertionError("Read an observation other than the requested current source frame")
                return self.calibration

        resolved = []
        with tempfile.TemporaryDirectory() as directory:
            for source_index in (3, 150):
                with self.subTest(source_frame_index=source_index):
                    current_pose = _pose()
                    current_pose[0, 3] += source_index * 0.01
                    calibration = {"misc": {"wrist_camera_intrinsics": self.intrinsics,
                                             "wrist_camera_extrinsics": current_pose}}
                    params = resolve_camera_param_for_frame(
                        "wrist", source_index, "sample-003", {},
                        CurrentObservationOnly(source_index, calibration), Path(directory),
                    )
                    np.testing.assert_array_equal(params["extrinsics"], current_pose)
                    np.testing.assert_array_equal(params["intrinsics"], self.intrinsics)
                    resolved.append(params["extrinsics"])
        self.assertEqual(calls, [3, 150])
        self.assertFalse(np.array_equal(resolved[0], resolved[1]))

    def test_frame_id_is_not_used_as_guidance_source_index(self) -> None:
        metadata = _metadata()
        metadata["frame_id"] = "3"
        validate_guidance(metadata, _empty_clouds())
        self.assertEqual(metadata["source_frame_index"], 150)
        # The legacy helper has a different contract and ignores the new key.
        self.assertEqual(frame_index_from_frame({"frame_id": "3", "frame_index": 2,
                                                "source_frame_index": 150}), 3)
        self.assertEqual(frame_index_from_frame({"frame_id": "150", "frame_index": 3}), 150)

    def test_legacy_camera_normalization_does_not_validate_rigid_geometry(self) -> None:
        singular = np.eye(4)
        singular[2, 2] = 0
        # Existing helper checks shape only. Do not mistake it for calibration QA.
        np.testing.assert_array_equal(normalize_extrinsics(singular), singular)
        with self.assertRaises(np.linalg.LinAlgError):
            project_points(np.array([[0.0, 0.0, 1.0]]), self.intrinsics, singular)


class GuidanceTransformValidationTests(unittest.TestCase):
    def test_cube_inverse_and_world_identity_preserve_metre_values(self) -> None:
        world = np.array([[0.1, -0.2, 0.8], [0.7, 0.3, 1.2]])
        center, scale = np.array([0.4, 0.05, 1.0]), 2.5
        cube = (world - center) * scale
        transform = bridgevla_cube_transform(center, scale, source="actual normalization inputs",
                                              timestamp=TIMESTAMP, source_frame_index=150)
        np.testing.assert_allclose(points_to_world(cube, transform), world, atol=1e-12)
        identity = identity_world_transform(source="existing world points", timestamp=TIMESTAMP,
                                             source_frame_index=150)
        np.testing.assert_array_equal(points_to_world(world, identity), world)
        self.assertEqual(points_to_world(np.empty((0, 3)), identity).shape, (0, 3))

    def test_singular_reflection_nonfinite_shear_and_nonuniform_transforms_fail(self) -> None:
        matrices = {}
        for name, diagonal in (("singular", [1, 1, 0]), ("reflection", [-1, 1, 1]),
                               ("nonuniform", [1, 2, 1])):
            matrix = np.eye(4)
            matrix[:3, :3] = np.diag(diagonal)
            matrices[name] = matrix
        shear = np.eye(4)
        shear[0, 1] = 0.2
        matrices["shear"] = shear
        nonfinite = np.eye(4)
        nonfinite[0, 3] = np.nan
        matrices["nonfinite"] = nonfinite
        nonaffine = np.eye(4)
        nonaffine[3, 0] = 0.1
        matrices["nonaffine"] = nonaffine
        for name, matrix in matrices.items():
            with self.subTest(name=name):
                transform = _camera_transform()
                transform["matrix_world_from_source"] = matrix.tolist()
                with self.assertRaises(ValueError):
                    points_to_world(np.array([[0.0, 0.0, 1.0]]), transform)

    def test_source_frame_and_timestamp_provenance_must_match_observation(self) -> None:
        for field, value in (("source_frame_index", 3),
                             ("timestamp", "2026-11-05T08:00:01+08:00"),
                             ("target_frame", "unknown_world")):
            with self.subTest(field=field):
                metadata = _metadata()
                metadata["transform"][field] = value
                with self.assertRaises(ValueError):
                    validate_guidance(metadata, _empty_clouds())

    def test_world_source_cannot_hide_a_nonidentity_transform(self) -> None:
        transform = _camera_transform()
        transform["source_frame"] = "rlbench_world"
        with self.assertRaises(ValueError):
            points_to_world(np.array([[0.0, 0.0, 1.0]]), transform)

    def test_declared_cube_parameters_must_agree_with_inverse_matrix(self) -> None:
        transform = bridgevla_cube_transform([0.1, 0.2, 0.3], 2.0, source="normalization",
                                              timestamp=TIMESTAMP, source_frame_index=150)
        invalid = copy.deepcopy(transform)
        invalid["normalization"]["center_world"][0] += 0.05
        with self.assertRaises(ValueError):
            points_to_world(np.array([[0.0, 0.0, 1.0]]), invalid)


@unittest.skipUnless(os.environ.get("BRIDGEVLA_REPO"), "Set BRIDGEVLA_REPO for actual cross-repository source comparison")
class OptionalBridgeVLASourceEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(os.environ["BRIDGEVLA_REPO"])

    def test_actual_place_pc_in_cube_inverse_for_both_centering_modes(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("Optional BridgeVLA normalization comparison requires torch")
        module = _load_source(self.repo / "finetune/bridgevla/mvt/utils.py", "bridgevla_coordinate_comparison")
        cloud = torch.tensor([[0.1, -0.4, 0.3], [0.2, 0.6, 1.3], [0.7, 0.1, 0.5]], dtype=torch.float64)
        bounds = [-0.2, -0.5, 0.0, 0.8, 0.7, 1.6]
        for use_cloud_bounds in (True, False):
            with self.subTest(with_mean_or_bounds=use_cloud_bounds):
                cube, actual_inverse = module.place_pc_in_cube(
                    cloud, with_mean_or_bounds=use_cloud_bounds,
                    scene_bounds=None if use_cloud_bounds else bounds,
                )
                # Observe center/scale from this invocation's actual inverse closure.
                basis = torch.stack([torch.zeros(3, dtype=cloud.dtype),
                                     torch.tensor([1.0, 0, 0], dtype=cloud.dtype)])
                inverse_basis = actual_inverse(basis).detach().cpu().numpy()
                center = inverse_basis[0]
                scale = 1.0 / np.linalg.norm(inverse_basis[1] - center)
                transform = bridgevla_cube_transform(center, float(scale), source="actual place_pc_in_cube inverse",
                                                      timestamp=TIMESTAMP, source_frame_index=150)
                world = points_to_world(cube.detach().cpu().numpy(), transform)
                np.testing.assert_allclose(world, actual_inverse(cube).detach().cpu().numpy(), atol=1e-12)
                np.testing.assert_allclose(world, cloud.detach().cpu().numpy(), atol=1e-12)

    def test_actual_bundled_vision_sensor_matches_camera_geometry_world(self) -> None:
        module = _load_source(
            self.repo / "finetune/bridgevla/libs/peract_colab/peract_colab/pyrep/objects/vision_sensor.py",
            "bridgevla_vision_sensor_comparison",
        )
        depth = np.linspace(0.4, 1.4, 30).reshape(5, 6)
        intrinsics = np.array([[97.0, 0, 2.5], [0, 101.0, 1.8], [0, 0, 1]])
        expected = module.VisionSensor.pointcloud_from_depth_and_camera_params(depth, _pose(), intrinsics)
        camera = backproject_mask(depth, np.ones_like(depth, dtype=bool), intrinsics, max_points=0)
        world = points_to_world(camera, _camera_transform())
        np.testing.assert_allclose(world, expected.reshape(-1, 3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
