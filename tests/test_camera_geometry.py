import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from camera_geometry import (
    RLBENCH_DEPTH_SCALE_FACTOR,
    backproject_mask,
    decode_rlbench_rgb_depth,
    find_rgb_path,
    frame_index_from_frame,
    load_rlbench_observations,
    looks_like_rlbench_packed_depth,
    project_points,
    read_depth,
    resolve_camera_param_for_frame,
    resolve_depth_path,
    transform_points,
)


class CameraGeometryTests(unittest.TestCase):
    def test_rlbench_depth_endpoints(self) -> None:
        packed = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        depth = decode_rlbench_rgb_depth(packed, near=0.1, far=4.1)
        self.assertAlmostEqual(depth[0, 0], 0.1)
        self.assertAlmostEqual(depth[0, 1], 4.1)
        self.assertEqual(RLBENCH_DEPTH_SCALE_FACTOR, float(2**24 - 1))

    def test_grayscale_rgb_is_not_treated_as_packed_depth(self) -> None:
        gray = np.full((2, 3, 3), 17, dtype=np.uint8)
        self.assertFalse(looks_like_rlbench_packed_depth(gray))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            Image.fromarray(gray).save(path)
            depth = read_depth(path, depth_scale=10.0, near=0.1, far=4.0)
        np.testing.assert_allclose(depth, np.full((2, 3), 1.7))

    def test_backprojection_and_projection_are_consistent(self) -> None:
        depth = np.zeros((3, 4), dtype=np.float64)
        depth[1, 2] = 2.0
        mask = depth > 0
        intrinsics = np.array(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
        )
        points_camera = backproject_mask(depth, mask, intrinsics, max_points=10)
        np.testing.assert_allclose(points_camera, [[1.0, 0.0, 2.0]])

        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = [0.5, -0.25, 0.75]
        points_world = transform_points(points_camera, camera_to_world)
        pixels, valid = project_points(
            points_world,
            intrinsics,
            camera_to_world,
        )
        self.assertTrue(valid[0])
        np.testing.assert_allclose(pixels, [[2.0, 1.0]])

    def test_numeric_frame_id_precedes_sample_index(self) -> None:
        self.assertEqual(
            frame_index_from_frame({"frame_id": "150", "frame_index": 3}),
            150,
        )
        self.assertEqual(frame_index_from_frame({"frame_index": "3"}), 3)
        self.assertIsNone(frame_index_from_frame({"frame_id": "frame-a"}))

    def test_osc_hdf5_falls_back_to_calibrated_rgbd_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode"
            hdf5_dir = episode / "hdf5"
            hdf5_dir.mkdir(parents=True)
            with h5py.File(hdf5_dir / "trajectory.hdf5", "w") as handle:
                observations = handle.create_group("observations")
                observations.create_dataset(
                    "WRIST_rgb", data=np.zeros((1, 2, 3, 3), dtype=np.uint8)
                )
                observations.create_dataset(
                    "WRIST_depth",
                    data=np.array([[[1000, 2500, 65535], [500, 750, 1250]]], dtype=np.uint16),
                )
                observations.create_dataset("WRIST_intrinsic", data=np.eye(3))
                observations.create_dataset(
                    "WRIST_extrinsic", data=np.array([[1, 2, 3, 1, 0, 0, 0]])
                )
                observations.create_dataset("ee_pos_world", data=[[4, 5, 6]])
                observations.create_dataset("ee_quat_world", data=[[0, 0, 0, 1]])
                handle.create_dataset("actions", data=[[0, 0, 0, 0, 0, 0, 1]])

            rgb_path = find_rgb_path(episode, "WRIST", "0")
            self.assertIsNotNone(rgb_path)
            self.assertTrue(rgb_path.is_file())
            depth_path = resolve_depth_path(episode, "WRIST", "0")
            depth = read_depth(depth_path, depth_scale=999.0)
            np.testing.assert_allclose(depth[0, :2], [1.0, 2.5])
            self.assertTrue(np.isnan(depth[0, 2]))

            params = resolve_camera_param_for_frame(
                "WRIST", 0, "0", {}, [], episode
            )
            self.assertIsNotNone(params)
            np.testing.assert_allclose(params["intrinsics"], np.eye(3))
            np.testing.assert_allclose(params["extrinsics"][:3, 3], [1, 2, 3])
            points_camera = backproject_mask(
                depth,
                np.array([[True, False, False], [False, False, False]]),
                params["intrinsics"],
                max_points=10,
            )
            points_world = transform_points(points_camera, params["extrinsics"])
            np.testing.assert_allclose(points_world, [[1.0, 2.0, 2.0]])

            robot_observations = load_rlbench_observations(episode)
            self.assertEqual(robot_observations[0]["gripper_pose"][:3], [4.0, 5.0, 6.0])
            self.assertEqual(robot_observations[0]["gripper_open"], 1.0)


if __name__ == "__main__":
    unittest.main()
