from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from slot_experiment_metrics import METRIC_NAMES, METRIC_UNITS, evaluate_frame, load_baseline_report, load_current_labels, metric, paired_success, translation_error
from slot_guided_synthetic_demo import create_demo
from run_slot_geometry_experiments import run_experiments
from run_slot_guided_sam3 import mock_backends
from slot_guidance import load_guidance
from slot_guided_io import calibration_digest, load_frame_snapshot, run_offline_frame
from slot_geometry_fusion import fuse_geometry


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
        baseline = self.baseline_fixture()
        (self.manifest.parent / "baseline.json").write_text(json.dumps(baseline))
        report = run_experiments(self.manifest, self.path / "results")
        self.assertEqual(report["frames"][0]["configurations"]["C"]["status"], "provided_baseline_report")
        baseline["source_frame_index"] = 8
        (self.manifest.parent / "baseline.json").write_text(json.dumps(baseline))
        with self.assertRaisesRegex(ValueError, "baseline report source_frame_index"):
            run_experiments(self.manifest, self.path / "results2")

    def baseline_fixture(self):
        snapshot = load_frame_snapshot(self.manifest.parent / "frame.json")
        return {"schema_version": 1, "artifact_type": "ocrr_baseline_evaluation",
                "episode_id": "synthetic_two_instances", "frame_id": "sample-2", "source_frame_index": 7,
                "observed_at": snapshot.metadata["observed_at"], "calibration_sha256": calibration_digest(snapshot),
                "method": "original_ocrr_action_path", "evidence_kind": "synthetic_mock",
                "source_checkpoint": "mock://original_ocrr_fixture", "source_checkpoint_sha256": None,
                "source_git_sha": "a" * 40,
                "metrics": {name: metric(None, 0, reason="model_not_executed", unit=METRIC_UNITS[name]) for name in METRIC_NAMES}}

    def test_C_rejects_malformed_source_and_measurements(self):
        snapshot = load_frame_snapshot(self.manifest.parent / "frame.json")
        path = self.manifest.parent / "baseline.json"
        original = self.baseline_fixture()
        cases = []
        for key, value in (("schema_version", True), ("source_frame_index", True), ("source_git_sha", "short"),
                           ("calibration_sha256", "b" * 64), ("source_checkpoint", None), ("metrics", {}),
                           ("observed_at", "2026-11-05T10:00:01+08:00")):
            cases.append(dict(original, **{key: value}))
        for field, value in (("value", float("nan")), ("value", True), ("count", True), ("reason", None), ("unit", "cm")):
            candidate = copy.deepcopy(original)
            candidate["metrics"]["sam3_mask_iou"][field] = value
            cases.append(candidate)
        for value in cases:
            with self.subTest(value=value):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_baseline_report(path, snapshot, "synthetic_mock")
        real = dict(original, evidence_kind="real_offline")
        path.write_text(json.dumps(real))
        with self.assertRaisesRegex(ValueError, "real baseline"):
            load_baseline_report(path, snapshot, "real_offline")

    def test_IoU_uses_final_joint_selection_and_reports_local_selection_separately(self):
        snapshot = load_frame_snapshot(self.manifest.parent / "frame.json")
        guidance = load_guidance(self.manifest.parent / "predicted_mock.json")
        backend = mock_backends(self.manifest.parent / "mock_masks.json", snapshot)
        inference = run_offline_frame(snapshot.path, self.manifest.parent / "predicted_mock.json", backend, self.path / "local")
        geometry = fuse_geometry(snapshot, guidance, inference)
        for camera in inference["cameras"].values():
            for role in ("target", "reference"):
                # Isolate the evaluator boundary: final geometry retains index1,
                # while the camera-local diagnostic refers to another mask.
                camera["roles"][role]["selected_index"] = 0
        labels = load_current_labels(self.manifest.parent / "labels.json", snapshot)
        result = evaluate_frame(snapshot, guidance, inference, geometry, labels)
        self.assertEqual(result["sam3_mask_iou"]["value"], 1.0)
        self.assertEqual(result["camera_local_mask_iou"]["value"], 0.0)

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
