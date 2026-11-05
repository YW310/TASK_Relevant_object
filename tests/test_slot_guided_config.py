from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_slot_geometry_experiments import run_experiments
from slot_guided_config import SlotGuidedConfig, load_slot_guided_config
from slot_guided_synthetic_demo import create_demo


REPO = Path(__file__).resolve().parents[1]


class OfflineConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_config(self, value):
        path = self.root / "config.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_checked_in_configuration_preserves_existing_defaults(self):
        configured = load_slot_guided_config(REPO / "configs/slot_guided_offline.json")
        self.assertEqual(configured.to_dict(), SlotGuidedConfig().to_dict())
        self.assertEqual(configured.digest(), load_slot_guided_config().digest())

    def test_partial_overrides_are_typed_and_unknown_or_invalid_fields_fail(self):
        path = self.write_config({"schema_version": 1, "fusion": {"min_support_cameras": 2}})
        self.assertEqual(load_slot_guided_config(path).fusion.min_support_cameras, 2)
        bad = [{}, {"schema_version": True}, {"schema_version": 2},
               {"schema_version": 1, "enable_slot_guided_sam3": True},
               {"schema_version": 1, "adapter": []}, {"schema_version": 1, "adapter": {"padding": 1}},
               {"schema_version": 1, "fusion": {"min_support_cameras": 2.0}},
               {"schema_version": 1, "selection": {"min_quality": True}},
               {"schema_version": 1, "selection": {"min_quality": "0.5"}},
               {"schema_version": 1, "selection": {"min_quality": float("nan")}},
               {"schema_version": 1, "adapter": {"min_visible_fraction": 0}},
               {"schema_version": 1, "fusion": {"cross_view_distance_m": -1}}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_slot_guided_config(self.write_config(value))

    def test_duplicate_keys_rejected(self):
        path = self.root / "config.json"
        path.write_text('{"schema_version":1,"adapter":{},"adapter":{}}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_slot_guided_config(path)

    def test_config_inspection_command_needs_no_snapshot_or_model(self):
        result = subprocess.run([sys.executable, str(REPO / "slot_guided_config.py"), "--config",
                                 str(REPO / "configs/slot_guided_offline.json")], cwd=REPO, capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["effective_config"], SlotGuidedConfig().to_dict())
        self.assertEqual(value["config_sha256"], SlotGuidedConfig().digest())

    def test_experiment_config_reaches_fusion_and_is_recorded(self):
        manifest = create_demo(self.root / "inputs")
        config = load_slot_guided_config(self.write_config({"schema_version": 1, "fusion": {"min_support_cameras": 3}}))
        report = run_experiments(manifest, self.root / "results", config=config)
        self.assertEqual(report["effective_config"]["fusion"]["min_support_cameras"], 3)
        self.assertEqual(report["config_sha256"], config.digest())
        for name, row in report["frames"][0]["configurations"].items():
            if name == "C":
                continue
            self.assertEqual(row["sam3_mask_iou"]["value"], 0.0)
            self.assertEqual(row["camera_local_mask_iou"]["value"], 1.0)
            path = self.root / "results" / "synthetic_two_instances" / "7" / name / "object_geometry.json"
            geometry = json.loads(path.read_text())
            self.assertEqual(geometry["fusion_config"]["min_support_cameras"], 3)

    def test_synthetic_CLI_loads_the_supplied_config(self):
        path = self.write_config({"schema_version": 1, "fusion": {"min_support_cameras": 3}})
        output = self.root / "cli"
        result = subprocess.run([sys.executable, str(REPO / "slot_guided_synthetic_demo.py"), "--output-dir", str(output),
                                 "--config", str(path)], cwd=REPO, capture_output=True, text=True, check=True)
        report = json.loads(Path(json.loads(result.stdout)["report"]).read_text())
        self.assertEqual(report["effective_config"]["fusion"]["min_support_cameras"], 3)
        self.assertFalse(report["real_offline_acceptance_passed"])

    def test_invalid_programmatic_backend_is_not_silently_mocked(self):
        manifest = create_demo(self.root / "inputs")
        with self.assertRaisesRegex(ValueError, "backend_kind"):
            run_experiments(manifest, self.root / "results", backend_kind="typo")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            run_experiments(manifest, self.root / "results", backend_kind="sam3")

    def test_demo_refuses_existing_inputs_before_modifying_them(self):
        create_demo(self.root / "inputs")
        image = self.root / "inputs" / "front.rgb.png"
        before = image.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "new or empty"):
            create_demo(self.root / "inputs")
        self.assertEqual(image.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
