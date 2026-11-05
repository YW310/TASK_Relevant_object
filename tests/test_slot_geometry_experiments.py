from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from slot_experiment_metrics import paired_success, translation_error
from slot_guided_synthetic_demo import create_demo
from run_slot_geometry_experiments import run_experiments


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.manifest = create_demo(self.path / "inputs")

    def test_complete_mock_experiment_retains_real_metrics_unavailable(self):
        report = run_experiments(self.manifest, self.path / "results")
        self.assertEqual(report["evidence_kind"], "synthetic_mock")
        self.assertFalse(report["real_offline_acceptance_passed"])
        configurations = report["frames"][0]["configurations"]
        self.assertEqual(set(configurations), {"A_box-only", "A_point+box", "B_box-only", "B_point+box", "C"})
        for name in ("A_box-only", "A_point+box", "B_box-only", "B_point+box"):
            row = configurations[name]
            self.assertEqual(row["sam3_mask_iou"]["value"], 1.0)
            self.assertEqual(row["prompt_instance_hit_rate"]["value"], 1.0)
            self.assertEqual(row["backend_kinds"], ["mock"])
            self.assertLess(row["object_center_error_m"]["value"], 0.02)
            self.assertIsNone(row["action_translation_error_m"]["value"])
            self.assertIsNone(row["rlbench_paired_task_success_rate"]["value"])
            self.assertIsNone(row["peak_vram_bytes"]["value"])
        self.assertEqual(configurations["C"]["status"], "not_executed")
        self.assertTrue((self.path / "results" / "experiment_report.json").exists())

    def test_mock_cannot_be_labeled_real(self):
        value = json.loads(self.manifest.read_text())
        value["evidence_kind"] = "real_offline"
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Mock"):
            run_experiments(self.manifest, self.path / "results")

    def test_duplicate_or_reversed_source_frames_rejected(self):
        value = json.loads(self.manifest.read_text())
        value["frames"].append(copy.deepcopy(value["frames"][0]))
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "unique and ascending"):
            run_experiments(self.manifest, self.path / "results")

    def test_evaluation_labels_bound_to_exact_source_frame(self):
        labels = self.manifest.parent / "labels.json"
        value = json.loads(labels.read_text())
        value["source_frame_index"] += 1
        labels.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "labels source_frame_index"):
            run_experiments(self.manifest, self.path / "results")

    def test_supplied_C_report_must_be_original_path_and_same_frame(self):
        value = json.loads(self.manifest.read_text())
        value["frames"][0]["baseline_report_json"] = "baseline.json"
        self.manifest.write_text(json.dumps(value))
        baseline = {"episode_id": "synthetic_two_instances", "frame_id": "sample-2", "source_frame_index": 7,
                    "method": "original_ocrr_action_path", "evidence_kind": "synthetic_mock",
                    "metrics": {"action_translation_error_m": {"value": None, "reason": "model_not_executed"}}}
        (self.manifest.parent / "baseline.json").write_text(json.dumps(baseline))
        report = run_experiments(self.manifest, self.path / "results")
        self.assertEqual(report["frames"][0]["configurations"]["C"]["status"], "provided_baseline_report")
        baseline["source_frame_index"] = 8
        (self.manifest.parent / "baseline.json").write_text(json.dumps(baseline))
        with self.assertRaisesRegex(ValueError, "baseline report source_frame_index"):
            run_experiments(self.manifest, self.path / "results2")

    def test_pure_action_and_paired_metric_math(self):
        result = translation_error([[0, 0, 0], [1, 0, 0]], [[0.03, 0.04, 0], [1, 0, 0]])
        self.assertAlmostEqual(result["value"], 0.025)
        with self.assertRaises(ValueError):
            translation_error([[0, np.nan, 0]], [[0, 0, 0]])
        paired = paired_success([{"task_name": "t", "seed": 1, "episode_id": "e", "method_success": True, "baseline_success": False}])
        self.assertEqual(paired["method"]["value"], 1)
        self.assertEqual(paired["paired_difference"]["value"], 1)
        self.assertIsNone(paired_success([])["method"]["value"])
        with self.assertRaises(ValueError):
            paired_success([{"task_name": "t", "seed": 1, "episode_id": "e", "method_success": True}])


if __name__ == "__main__":
    unittest.main()
