import unittest
from types import SimpleNamespace

import numpy as np

from multiview_candidate_fusion import Observation3D, cluster_observations


def _observation(name: str, camera: str, bbox: list[list[float]]) -> Observation3D:
    bounds = np.asarray(bbox, dtype=np.float64)
    points = np.array([bounds[0], bounds[1], (bounds[0] + bounds[1]) / 2])
    return Observation3D(
        observation_id=name,
        role_evidence={"target": 0.9},
        provenance={},
        camera=camera,
        candidate={"score": 0.9},
        points_world=points,
        centroid_world=points.mean(axis=0),
        bbox3d_world=bounds,
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

    def test_bbox_overlap_can_merge_different_cameras_despite_centroid_drift(self) -> None:
        front = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        left = _observation("left:C1", "left", [[0.05, 0, 0], [0.15, 0.10, 0.10]])

        clusters = cluster_observations([front, left], self.args)

        self.assertEqual(1, len(clusters))
        self.assertEqual({"front", "left"}, {item.camera for item in clusters[0]})

    def test_same_camera_observations_remain_separate(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        second = _observation("front:C2", "front", [[0.04, 0, 0], [0.14, 0.10, 0.10]])

        clusters = cluster_observations([first, second], self.args)

        self.assertEqual([1, 1], sorted(len(cluster) for cluster in clusters))


if __name__ == "__main__":
    unittest.main()
