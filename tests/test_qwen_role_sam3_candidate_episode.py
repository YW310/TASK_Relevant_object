import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from qwen_role_sam3_candidate_episode import (
    build_semantic_groups,
    canonicalize_candidates,
    mask_bbox,
    normalize_prompt,
    process_camera,
    semantic_prompt_plan,
    split_mask_components,
)


def _candidate(candidate_id: str, role: str, score: float, mask: np.ndarray) -> dict:
    group_id = {"target": "SG_TARGET", "reference": "SG_REFERENCE", "interaction_part": "SG_PART"}[role]
    return {
        "id": candidate_id,
        "score": score,
        "mask": mask,
        "mask_bbox_xyxy": mask_bbox(mask),
        "mask_area_pixels": int(mask.sum()),
        "semantic_group_ids": [group_id],
        "compatible_roles": [role],
        "roles_by_group": {group_id: [role]},
        "source_prompt": role,
    }


class SemanticGroupTests(unittest.TestCase):
    def test_identical_target_reference_specs_become_one_shared_group(self) -> None:
        spec = {
            "target": {"name": "rose block", "sam_prompts": [" Rose   Block "], "identity_cues": ["pink"]},
            "reference": {"name": "ROSE BLOCK", "sam_prompts": ["rose block"], "identity_cues": ["PINK"]},
        }

        groups = build_semantic_groups(spec, 5)

        self.assertEqual(1, len(groups))
        self.assertEqual(["target", "reference"], groups[0]["compatible_roles"])
        self.assertFalse(groups[0]["role_discriminative"])
        plan = semantic_prompt_plan(groups)
        self.assertEqual(3, len(plan))
        self.assertTrue(all(row["semantic_group_ids"] == ["SG1"] for row in plan))

    def test_partial_prompt_sharing_deduplicates_execution_but_keeps_groups(self) -> None:
        spec = {
            "target": {"name": "white cup", "sam_prompts": ["cup", "white cup"]},
            "reference": {"name": "yellow plate", "sam_prompts": [" CUP ", "yellow plate"]},
        }

        groups = build_semantic_groups(spec, 5)
        plan = semantic_prompt_plan(groups)

        self.assertEqual(2, len(groups))
        self.assertEqual(3, len(plan))
        cup = next(row for row in plan if row["normalized_prompt"] == "cup")
        self.assertEqual({"SG1", "SG2"}, set(cup["semantic_group_ids"]))

    def test_distinct_target_reference_and_part_groups_remain_discriminative(self) -> None:
        spec = {
            "target": {"name": "white cup", "sam_prompts": ["white cup"]},
            "reference": {"name": "yellow plate", "sam_prompts": ["yellow plate"]},
            "interaction_part": {"name": "handle", "sam_prompts": ["cup handle"]},
        }
        groups = build_semantic_groups(spec, 5)
        self.assertEqual(3, len(groups))
        self.assertTrue(all(group["role_discriminative"] for group in groups))
        self.assertEqual("strasse cup", normalize_prompt("  STRASSE\t cup "))

    def test_process_camera_executes_each_normalized_prompt_once(self) -> None:
        spec = {
            "target": {"name": "rose block", "sam_prompts": ["rose block"]},
            "reference": {"name": "ROSE BLOCK", "sam_prompts": [" Rose   Block "]},
        }
        groups = build_semantic_groups(spec, 5)
        args = SimpleNamespace(
            resume=False, threshold=0.25, candidate_pool_size=20,
            min_mask_area=1, split_disconnected_masks=False,
            max_mask_components=4, prompt_variants=5, mask_nms_iou=0.8,
            canonical_containment=0.9, canonical_max_area_ratio=3.0,
            canonical_bbox_iou=0.0, suppress_multi_instance_masks=False,
            top_k_per_semantic_group=8, progress=False, mask_alpha=105,
        )
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_path = root / "image.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            with patch(
                "qwen_role_sam3_candidate_episode.run_text_prompt",
                return_value=(
                    np.asarray([mask]), np.asarray([0.8]), None
                ),
            ) as run_prompt:
                result = process_camera(None, image_path, groups, root / "out", args)

        self.assertEqual(len(semantic_prompt_plan(groups)), run_prompt.call_count)
        normalized_calls = [normalize_prompt(call.args[2]) for call in run_prompt.call_args_list]
        self.assertEqual(len(normalized_calls), len(set(normalized_calls)))
        self.assertEqual(["target", "reference"], result["candidates"][0]["semantic_evidence"][0]["compatible_roles"])
        self.assertTrue(result["candidates"][0]["semantic_evidence"][0]["supporting_prompts"])


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

    def test_distinct_semantic_groups_are_not_cross_normalized_or_merged(self) -> None:
        whole = np.zeros((10, 10), dtype=bool)
        whole[1:9, 1:9] = True
        partial = np.zeros_like(whole)
        partial[2:6, 2:6] = True

        canonical, suppressed = canonicalize_candidates(
            [
                _candidate("T1", "target", 0.95, whole),
                _candidate("T2", "target", 0.80, partial),
                _candidate("R2", "reference", 0.79, partial.copy()),
            ],
            iou_threshold=0.80,
            containment_threshold=0.90,
            max_containment_area_ratio=4.0,
            suppress_multi_instance_masks=False,
        )

        self.assertEqual(2, len(canonical))
        evidence = [entry for item in canonical for entry in item["semantic_evidence"]]
        self.assertEqual({"SG_TARGET", "SG_REFERENCE"}, {entry["semantic_group_id"] for entry in evidence})
        self.assertTrue(all(0.0 <= entry["score"] <= 1.0 for entry in evidence))
        self.assertNotAlmostEqual(1.0, sum(entry["score"] for entry in evidence))
        self.assertEqual(1, len(suppressed))

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
            "contains_multiple_same_semantic_group_instances",
            {item.get("reason") for item in suppressed},
        )


if __name__ == "__main__":
    unittest.main()
