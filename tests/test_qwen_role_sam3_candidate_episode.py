import sys
import types
import unittest

import numpy as np


if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from qwen_role_sam3_candidate_episode import (
    canonicalize_candidates,
    mask_bbox,
    split_mask_components,
)


def _candidate(candidate_id: str, role: str, score: float, mask: np.ndarray) -> dict:
    return {
        "id": candidate_id,
        "role": role,
        "score": score,
        "mask": mask,
        "mask_bbox_xyxy": mask_bbox(mask),
        "mask_area_pixels": int(mask.sum()),
    }


class SplitMaskComponentsTests(unittest.TestCase):
    def test_two_disconnected_objects_become_two_candidates(self) -> None:
        mask = np.zeros((12, 16), dtype=bool)
        mask[2:5, 1:4] = True
        mask[7:11, 10:15] = True

        components = split_mask_components(mask, min_area=4)

        self.assertEqual(2, len(components))
        self.assertEqual([10, 7, 15, 11], mask_bbox(components[0]))
        self.assertEqual([1, 2, 4, 5], mask_bbox(components[1]))

    def test_diagonal_pixels_use_eight_connectivity(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        mask[3, 3] = True

        components = split_mask_components(mask, min_area=1)

        self.assertEqual(1, len(components))
        self.assertEqual([1, 1, 4, 4], mask_bbox(components[0]))

    def test_tiny_speck_is_removed_before_bbox_generation(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 2:6] = True
        mask[9, 9] = True

        components = split_mask_components(mask, min_area=4)

        self.assertEqual(1, len(components))
        self.assertEqual([2, 2, 6, 6], mask_bbox(components[0]))

    def test_component_cap_keeps_largest_regions(self) -> None:
        mask = np.zeros((15, 15), dtype=bool)
        mask[1:3, 1:3] = True
        mask[5:8, 5:8] = True
        mask[10:14, 10:14] = True

        components = split_mask_components(mask, min_area=1, max_components=2)

        self.assertEqual([16, 9], [int(component.sum()) for component in components])


class CanonicalizationContainmentTests(unittest.TestCase):
    def test_large_mask_does_not_absorb_much_smaller_instance(self) -> None:
        broad = np.zeros((12, 12), dtype=bool)
        broad[1:11, 1:11] = True
        small = np.zeros_like(broad)
        small[3:5, 3:5] = True

        canonical, _ = canonicalize_candidates(
            [
                _candidate("T1", "target", 0.95, broad),
                _candidate("T2", "target", 0.90, small),
            ],
            iou_threshold=0.80,
            containment_threshold=0.90,
            max_containment_area_ratio=3.0,
            suppress_multi_instance_masks=False,
        )

        self.assertEqual(2, len(canonical))

    def test_similar_same_role_containment_still_deduplicates(self) -> None:
        whole = np.zeros((8, 8), dtype=bool)
        whole[2:5, 2:5] = True
        partial = np.zeros_like(whole)
        partial[2:5, 2:4] = True

        canonical, _ = canonicalize_candidates(
            [
                _candidate("T1", "target", 0.95, whole),
                _candidate("T2", "target", 0.90, partial),
            ],
            iou_threshold=0.80,
            containment_threshold=0.90,
            max_containment_area_ratio=3.0,
            suppress_multi_instance_masks=False,
        )

        self.assertEqual(1, len(canonical))

    def test_cross_role_containment_is_not_treated_as_duplicate(self) -> None:
        whole = np.zeros((8, 8), dtype=bool)
        whole[2:5, 2:5] = True
        part = np.zeros_like(whole)
        part[2:4, 2:4] = True

        canonical, _ = canonicalize_candidates(
            [
                _candidate("T1", "target", 0.95, whole),
                _candidate("P1", "interaction_part", 0.90, part),
            ],
            iou_threshold=0.80,
            containment_threshold=0.90,
            max_containment_area_ratio=3.0,
            suppress_multi_instance_masks=False,
        )

        self.assertEqual(2, len(canonical))

    def test_broad_same_role_group_mask_is_suppressed(self) -> None:
        broad = np.zeros((14, 14), dtype=bool)
        broad[1:13, 1:13] = True
        child_a = np.zeros_like(broad)
        child_a[3:5, 3:5] = True
        child_b = np.zeros_like(broad)
        child_b[9:11, 9:11] = True

        canonical, suppressed = canonicalize_candidates(
            [
                _candidate("T1", "target", 0.97, broad),
                _candidate("T2", "target", 0.91, child_a),
                _candidate("T3", "target", 0.90, child_b),
            ],
            iou_threshold=0.80,
            containment_threshold=0.90,
            max_containment_area_ratio=3.0,
            suppress_multi_instance_masks=True,
        )

        self.assertEqual(2, len(canonical))
        self.assertIn(
            "contains_multiple_same_role_instances",
            {item.get("reason") for item in suppressed},
        )


if __name__ == "__main__":
    unittest.main()
