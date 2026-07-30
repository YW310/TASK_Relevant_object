import unittest
from types import SimpleNamespace

import numpy as np

from multiview_candidate_fusion import (
    Observation3D,
    assign_object_ids,
    cluster_observations,
    frame_index_from_frame,
)


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


class TemporalObjectTrackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            track_distance_m=0.15,
            track_max_missed_frames=2,
            track_max_size_ratio=2.5,
        )

    def test_track_id_survives_two_missing_frames(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        objects, state = assign_object_ids([[first]], {}, self.args, "f0")
        self.assertEqual(["O1"], [item["id"] for item in objects])

        objects, state = assign_object_ids([], state, self.args, "f1")
        self.assertEqual([], objects)
        self.assertEqual(1, state["tracks"][0]["missed_frames"])
        objects, state = assign_object_ids([], state, self.args, "f2")
        self.assertEqual([], objects)
        self.assertEqual(2, state["tracks"][0]["missed_frames"])

        reappeared = _observation("front:C1", "front", [[0.01, 0, 0], [0.11, 0.10, 0.10]])
        objects, _ = assign_object_ids([[reappeared]], state, self.args, "f3")
        self.assertEqual(["O1"], [item["id"] for item in objects])

    def test_expired_track_gets_new_id(self) -> None:
        first = _observation("front:C1", "front", [[0, 0, 0], [0.10, 0.10, 0.10]])
        _, state = assign_object_ids([[first]], {}, self.args, "f0")
        for frame_id in ("f1", "f2", "f3"):
            _, state = assign_object_ids([], state, self.args, frame_id)

        reappeared = _observation("front:C1", "front", [[0.01, 0, 0], [0.11, 0.10, 0.10]])
        objects, _ = assign_object_ids([[reappeared]], state, self.args, "f4")
        self.assertEqual(["O2"], [item["id"] for item in objects])

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


if __name__ == "__main__":
    unittest.main()
