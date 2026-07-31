import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from camera_geometry import (
    RLBENCH_DEPTH_SCALE_FACTOR,
    backproject_mask,
    decode_rlbench_rgb_depth,
    frame_index_from_frame,
    looks_like_rlbench_packed_depth,
    project_points,
    read_depth,
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


if __name__ == "__main__":
    unittest.main()
