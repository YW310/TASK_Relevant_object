"""PR1 artifact boundary checks; no learned backend or episode future reads."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from slot_guidance import (
    MockBackend,
    SlotGuidance,
    bridgevla_cube_transform,
    identity_world_transform,
    load_guidance,
    points_to_world,
    save_guidance,
    validate_guidance,
)


TIMESTAMP = "2026-11-05T10:00:00+08:00"
REPO = Path(__file__).resolve().parents[1]


def fixture() -> tuple[dict, dict[str, np.ndarray]]:
    """Same category, distinct role instances; source 150 is sample ordinal 3."""
    metadata = {
        "schema_version": 1, "artifact_type": "slot_guidance",
        "episode_id": "episode0", "frame_id": "sample-3", "source_frame_index": 150,
        "instruction": "move the left red block next to the right red block",
        "task_name": "move_block", "coordinate_frame": "rlbench_world", "coordinate_units": "m",
        "observed_at": TIMESTAMP, "created_at": TIMESTAMP, "guidance_source": "mock",
        "source_checkpoint": None, "source_git_sha": "a" * 40,
        "transform": identity_world_transform(source="synthetic current-frame fixture",
                                               timestamp=TIMESTAMP, source_frame_index=150),
        "points_file": "fixture.npz", "points_sha256": "0" * 64,
        "reference_null_probability": 0.03,
        "target": {"status": "VISIBLE", "semantic_valid": True, "confidence": 0.9,
                   "center_world": [-0.1, 0, 1], "semantic_label": "red block", "instance_id": "left",
                   "points_key": "target_points_world", "point_count": 2},
        "reference": {"status": "VISIBLE", "semantic_valid": True, "confidence": 0.8,
                      "center_world": [0.1, 0, 1], "semantic_label": "red block", "instance_id": "right",
                      "points_key": "reference_points_world", "point_count": 1},
    }
    points = {"target_points_world": np.array([[-0.11, 0, 1], [-0.09, 0, 1]]),
              "reference_points_world": np.array([[0.1, 0, 1]])}
    return metadata, points


class SlotGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata, self.points = fixture()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "frame.json"

    def save(self, **kwargs) -> SlotGuidance:
        return save_guidance(self.path, self.metadata, **self.points, **kwargs)

    def test_json_npz_roundtrip_and_metadata_not_mutated(self) -> None:
        before = copy.deepcopy(self.metadata)
        written = self.save()
        loaded = load_guidance(self.path, expected_source_frame_index=150)
        self.assertEqual(before, self.metadata)
        self.assertEqual(written.metadata, loaded.metadata)
        for role in ("target", "reference"):
            np.testing.assert_array_equal(loaded.points(role), self.points[f"{role}_points_world"])
            self.assertFalse(loaded.points(role).flags.writeable)
        self.assertEqual(loaded.metadata["target"]["semantic_label"], loaded.metadata["reference"]["semantic_label"])
        self.assertNotEqual(loaded.metadata["target"]["instance_id"], loaded.metadata["reference"]["instance_id"])

    def test_missing_storage_fields_are_filled_and_counts_inferred(self) -> None:
        self.metadata.pop("points_file")
        self.metadata.pop("points_sha256")
        for role in ("target", "reference"):
            self.metadata[role].pop("points_key")
            self.metadata[role].pop("point_count")
        result = self.save()
        self.assertEqual(result.metadata["target"]["point_count"], 2)
        result.validate()

    def test_null_not_visible_and_low_confidence_survive_empty_geometry(self) -> None:
        for status in ("NULL", "NOT_VISIBLE", "LOW_CONFIDENCE"):
            with self.subTest(status=status):
                self.metadata["reference"] = {
                    "status": status, "semantic_valid": status != "NULL", "confidence": 0.1,
                    "center_world": None, "points_key": "reference_points_world", "point_count": 0,
                }
                self.metadata["reference_null_probability"] = 0.95 if status == "NULL" else 0.05
                self.points["reference_points_world"] = np.empty((0, 3))
                result = self.save(overwrite=True)
                self.assertEqual(result.metadata["reference"]["status"], status)
                self.assertEqual(result.reference_points_world.shape, (0, 3))
                self.assertEqual(load_guidance(self.path).metadata["reference"]["status"], status)

    def test_null_cannot_hide_instance_geometry(self) -> None:
        self.metadata["reference"]["status"] = "NULL"
        self.metadata["reference"]["semantic_valid"] = False
        with self.assertRaisesRegex(ValueError, "NULL"):
            self.save()

    def test_target_cannot_be_null(self) -> None:
        self.metadata["target"] = {"status": "NULL", "semantic_valid": False, "confidence": 0,
                                   "center_world": None, "points_key": "target_points_world", "point_count": 0}
        self.points["target_points_world"] = np.empty((0, 3))
        with self.assertRaisesRegex(ValueError, "NULL"):
            self.save()

    def test_existing_role_is_distinct_from_invalid_semantics(self) -> None:
        self.metadata["reference"]["status"] = "LOW_CONFIDENCE"
        self.metadata["reference"]["semantic_valid"] = False
        with self.assertRaisesRegex(ValueError, "semantic_valid"):
            self.save()

    def test_empty_visible_center_only_does_not_invent_range(self) -> None:
        self.points["target_points_world"] = np.empty((0, 3))
        self.metadata["target"]["point_count"] = 0
        result = self.save()
        self.assertNotIn("spread_world", result.metadata["target"])

    def test_spread_requires_verified_range_source_and_center(self) -> None:
        self.metadata["target"]["spread_world"] = [0.02, 0.03, 0.04]
        with self.assertRaisesRegex(ValueError, "spread_source"):
            self.save()
        self.metadata["target"]["spread_source"] = "current-frame world AABB half-extents"
        self.save()
        self.metadata["target"]["center_world"] = None
        with self.assertRaisesRegex(ValueError, "center"):
            self.save(overwrite=True)

    def test_nonfinite_malformed_or_object_points_are_rejected(self) -> None:
        for bad in (np.array([[np.nan, 0, 1]]), np.array([[0, np.inf, 1]]),
                    np.zeros((3,)), np.zeros((2, 4)), np.array([], dtype=float),
                    np.array([[object(), 0, 1]], dtype=object), np.ones((1, 3), dtype=complex)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                self.points["target_points_world"] = bad
                with self.assertRaises(ValueError):
                    self.save()
        self.assertFalse(self.path.exists())

    def test_boolean_coordinates_cannot_silently_become_integers(self) -> None:
        self.metadata["target"]["center_world"] = [True, 0, 1]
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.save()
        self.metadata, self.points = fixture()
        self.metadata["transform"]["matrix_world_from_source"][0][0] = True
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.save()
        self.metadata, self.points = fixture()
        self.points["target_points_world"] = [[True, 0, 1], [0, 0, 1]]
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.save()

    def test_non_json_metadata_fails_before_creating_any_artifact(self) -> None:
        self.metadata["target"]["center_world"] = np.array([0, 0, 1])
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            self.save()
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_bad_units_frame_confidence_and_unknown_fields_are_rejected(self) -> None:
        cases = [("coordinate_units", "cm"), ("coordinate_frame", "bridgevla_cube"),
                 ("reference_null_probability", 1.1), ("source_frame_index", True),
                 ("source_git_sha", "short"), ("expert_future_action", [0, 0, 0]),
                 ("episode_id", "   "), ("schema_version", True)]
        original = copy.deepcopy(self.metadata)
        for key, value in cases:
            with self.subTest(key=key):
                self.metadata = copy.deepcopy(original)
                self.metadata[key] = value
                with self.assertRaises(ValueError):
                    self.save()

    def test_timestamp_and_transform_frame_match_are_required(self) -> None:
        for key, value in (("timestamp", "2026-11-05T02:00:01Z"), ("source_frame_index", 151)):
            with self.subTest(key=key):
                metadata = copy.deepcopy(self.metadata)
                metadata["transform"][key] = value
                with self.assertRaises(ValueError):
                    validate_guidance(metadata, self.points)
        for bad in ("2026-11-05", "2026-11-05T10:00:00", "20261105T100000+0800", "2026-11-05t02:00:00z",
                    "2026-11-05T10:00:00+08:99", "2026-11-05T10:00:00+00:60"):
            metadata = copy.deepcopy(self.metadata)
            metadata["observed_at"] = bad
            with self.subTest(timestamp=bad), self.assertRaises(ValueError):
                validate_guidance(metadata, self.points)

    def test_prediction_requires_checkpoint(self) -> None:
        self.metadata["guidance_source"] = "predicted_coarse_slot"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.save()
        self.metadata["source_checkpoint"] = "coarse/checkpoint.pt"
        self.save()

    def test_future_or_sample_index_rejected_before_npz_read(self) -> None:
        self.save()
        for wrong in (3, 149, 151):
            with self.subTest(index=wrong), self.assertRaisesRegex(ValueError, "source_frame_index"):
                load_guidance(self.path, expected_source_frame_index=wrong)

    def test_npz_hash_and_count_prevent_mismatched_pairs(self) -> None:
        result = self.save()
        sidecar = self.path.parent / result.metadata["points_file"]
        original = sidecar.read_bytes()
        sidecar.write_bytes(original + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_guidance(self.path)
        sidecar.write_bytes(original)
        metadata = copy.deepcopy(result.metadata)
        metadata["points_sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        metadata["target"]["point_count"] += 1
        self.path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "point_count"):
            load_guidance(self.path)

    def test_unsafe_npz_references_rejected_before_file_access(self) -> None:
        self.save()
        valid = json.loads(self.path.read_text(encoding="utf-8"))
        for filename in ("../outside.npz", "C:/outside.npz", "sub\\file.npz", "/root.npz"):
            metadata = dict(valid, points_file=filename)
            self.path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "basename"):
                load_guidance(self.path)

    def test_npz_object_pickle_and_extra_keys_rejected(self) -> None:
        self.save()
        valid = json.loads(self.path.read_text(encoding="utf-8"))
        for arrays in ({**self.points, "unexpected": np.zeros(1)},
                       {**self.points, "target_points_world": np.array([[None, None, None]], dtype=object)}):
            sidecar = self.path.parent / "adversarial.npz"
            np.savez(sidecar, **arrays)
            metadata = dict(valid, points_file=sidecar.name, points_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest())
            self.path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_guidance(self.path)

    def test_standalone_npy_cannot_masquerade_as_npz(self) -> None:
        result = self.save()
        sidecar = self.path.parent / "not-an-archive.npz"
        with sidecar.open("wb") as handle:
            np.save(handle, self.points["target_points_world"])
        metadata = dict(result.metadata, points_file=sidecar.name,
                        points_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest())
        self.path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "NPZ archive"):
            load_guidance(self.path)

    def test_duplicate_json_keys_rejected(self) -> None:
        self.path.write_text('{"frame_id":"a","frame_id":"b"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_guidance(self.path)

    def test_overwrite_is_explicit_and_published_geometry_stays_readable(self) -> None:
        first = self.save()
        with self.assertRaises(FileExistsError):
            self.save()
        self.points["target_points_world"] += 0.01
        second = self.save(overwrite=True)
        self.assertNotEqual(first.metadata["points_file"], second.metadata["points_file"])
        self.assertTrue((self.path.parent / first.metadata["points_file"]).is_file())
        np.testing.assert_array_equal(load_guidance(self.path).target_points_world, self.points["target_points_world"])

    def test_mock_backend_checks_role_identity_and_current_frame(self) -> None:
        backend = MockBackend(self.save())
        result = backend.predict(episode_id="episode0", frame_id="sample-3", source_frame_index=150)
        result.metadata["target"]["instance_id"] = "modified"
        self.assertEqual(backend.predict(episode_id="episode0", frame_id="sample-3", source_frame_index=150)
                         .metadata["target"]["instance_id"], "left")
        for identity in (("episode1", "sample-3", 150), ("episode0", "sample-4", 150), ("episode0", "sample-3", 151)):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "identity mismatch"):
                backend.predict(episode_id=identity[0], frame_id=identity[1], source_frame_index=identity[2])

    def test_import_and_mock_need_neither_sam3_nor_torch(self) -> None:
        script = """
import importlib.abc, sys
class RejectLearnedBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'sam3', 'torch', 'transformers'}:
            raise AssertionError('unexpected model import: ' + fullname)
sys.meta_path.insert(0, RejectLearnedBackend())
from slot_guidance import MockBackend, load_guidance
artifact = load_guidance(sys.argv[1], expected_source_frame_index=150)
MockBackend(artifact).predict(episode_id='episode0', frame_id='sample-3', source_frame_index=150)
"""
        self.save()
        subprocess.run([sys.executable, "-c", script, str(self.path)], cwd=REPO, check=True)

    def test_cli_validates_requested_source_frame(self) -> None:
        self.save()
        result = subprocess.run([sys.executable, str(REPO / "slot_guidance.py"), str(self.path),
                                 "--source-frame-index", "150"], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["roles"]["target"]["point_count"], 2)

    def test_explicit_cube_inverse_does_not_reapply_on_load(self) -> None:
        transform = bridgevla_cube_transform([0.5, -0.2, 0.9], 4, source="exact current scene bounds",
                                            timestamp=TIMESTAMP, source_frame_index=150)
        cube = np.array([[0.4, -0.8, 0.2], [-0.4, 0.1, 0.5]])
        world = points_to_world(cube, transform)
        self.metadata["transform"] = transform
        self.points["target_points_world"] = world
        self.save()
        np.testing.assert_allclose(load_guidance(self.path).target_points_world, cube / 4 + [0.5, -0.2, 0.9])


class SlotGuidanceSchemaTests(unittest.TestCase):
    def test_schema_and_runtime_agree_on_semantic_states_and_constraints(self) -> None:
        # jsonschema is in the test extra; never a runtime/SAM3 dependency.
        from jsonschema import Draft202012Validator, FormatChecker
        schema = json.loads((REPO / "schemas" / "slot_guidance.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        metadata, points = fixture()
        validator.validate(metadata)
        validate_guidance(metadata, points)
        for status in ("NULL", "NOT_VISIBLE", "LOW_CONFIDENCE"):
            value = copy.deepcopy(metadata)
            value["reference"] = {"status": status, "semantic_valid": status != "NULL", "confidence": 0.2,
                                  "center_world": None, "points_key": "reference_points_world", "point_count": 0}
            validator.validate(value)
            validate_guidance(value, dict(points, reference_points_world=np.empty((0, 3))))
        bad_values = []
        for key, bad in (("source_frame_index", -1), ("coordinate_units", "cm"), ("points_file", "../x.npz"),
                         ("observed_at", "20261105T100000+0800"), ("source_git_sha", "short"),
                         ("future_action", [1, 2, 3]), ("reference_null_probability", -0.1)):
            bad_values.append(dict(metadata, **{key: bad}))
        value = copy.deepcopy(metadata)
        value["target"]["spread_world"] = [1, 1, 1]
        bad_values.append(value)
        value = copy.deepcopy(metadata)
        value["reference"]["status"] = "NULL"
        bad_values.append(value)
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_guidance(value, points)
            self.assertFalse(validator.is_valid(value))


if __name__ == "__main__":
    unittest.main()
