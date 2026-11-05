"""Offline role identity, candidate failures and optional frozen backend checks."""

from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from slot_guidance import SlotGuidance, identity_world_transform
from slot_guided_sam3 import (
    AUDITED_SAM3_SHA, MockBackend, Sam3Backend, SelectionConfig,
    _check_predict_signature, run_camera, verify_sam3_api_contract,
)
from slot_prompt_adapter import CameraFrame


STAMP = "2026-11-05T10:00:00+08:00"
ROOT = Path(__file__).resolve().parents[1]


def fixture():
    """Two same-category instances with current source index 150, ordinal 3."""
    masks = {}
    points = {}
    roles = {}
    for role, x0 in (("target", 6), ("reference", 20)):
        mask = np.zeros((24, 32), dtype=bool)
        mask[8:17, x0:x0 + 7] = True
        y, x = np.nonzero(mask)
        cloud = np.column_stack(((x - 16) / 40, (y - 12) / 40, np.ones(len(x))))
        masks[role], points[role] = mask, cloud
        roles[role] = {"status": "VISIBLE", "semantic_valid": True, "confidence": 0.95,
                       "center_world": np.median(cloud, axis=0).tolist(),
                       "semantic_label": "red block", "instance_id": role,
                       "points_key": f"{role}_points_world", "point_count": len(cloud)}
    metadata = {"schema_version": 1, "artifact_type": "slot_guidance",
                "episode_id": "episode0", "frame_id": "sample-3", "source_frame_index": 150,
                "instruction": "put the left block near the right block", "task_name": "move_block",
                "coordinate_frame": "rlbench_world", "coordinate_units": "m",
                "observed_at": STAMP, "created_at": STAMP, "guidance_source": "mock",
                "source_checkpoint": None, "source_git_sha": "a" * 40,
                "transform": identity_world_transform(source="synthetic current frame", timestamp=STAMP,
                                                       source_frame_index=150),
                "points_file": "fixture.npz", "points_sha256": "0" * 64,
                "reference_null_probability": 0.01, **roles}
    guidance = SlotGuidance(metadata, points["target"], points["reference"])
    camera = CameraFrame("front", "episode0", "sample-3", 150, STAMP,
                         np.array([[40, 0, 16], [0, 40, 12], [0, 0, 1.0]]), np.eye(4),
                         (32, 24), "synthetic_current_frame")
    return guidance, camera, np.zeros((24, 32, 3), dtype=np.uint8), np.ones((24, 32)), masks


class OfflineSam3Tests(unittest.TestCase):
    def setUp(self):
        self.guidance, self.camera, self.rgb, self.depth, self.masks = fixture()

    def backend(self):
        return MockBackend({role: (mask[None], [.6]) for role, mask in self.masks.items()})

    def run_backend(self, backend, **kwargs):
        return run_camera(self.guidance, self.camera, self.rgb, self.depth, backend, **kwargs)

    def test_same_image_encoded_once_and_state_shared_across_roles(self):
        backend = self.backend()
        result = self.run_backend(backend)
        self.assertEqual(backend.set_image_calls, 1)
        self.assertEqual([call["role"] for call in backend.predict_calls], ["target", "reference"])
        self.assertEqual(backend.predict_calls[0]["state_id"], backend.predict_calls[1]["state_id"])
        for role in self.masks:
            self.assertEqual(result["roles"][role]["selected_index"], 0)
            self.assertIsNone(result["roles"][role]["failure_reason"])
            self.assertAlmostEqual(result["roles"][role]["candidates"][0]["metrics"]["center_error_m"], 0)
        self.assertGreater(result["latency_ms"], 0)
        self.assertIsNone(result["peak_vram_bytes"])

    def test_wrong_same_category_highest_score_cannot_override_role(self):
        wrong, correct = self.masks["reference"], self.masks["target"]
        backend = MockBackend({"target": (np.stack([wrong, correct]), [.99, .25]),
                               "reference": (wrong[None], [.8])})
        result = self.run_backend(backend)
        target = result["roles"]["target"]
        self.assertEqual(target["selected_index"], 1)
        self.assertEqual(len(target["candidates"]), 2)
        self.assertFalse(target["candidates"][0]["accepted"])
        self.assertIn("positive_prompt_miss", target["candidates"][0]["failure_reasons"])
        self.assertIn("negative_prompt_hit", target["candidates"][0]["failure_reasons"])
        np.testing.assert_array_equal(target["candidates"][0]["mask"], wrong)

    def test_both_modes_keep_same_verified_prompt_and_mode_provenance(self):
        prompts = []
        for mode in ("box-only", "point+box"):
            backend = self.backend()
            result = self.run_backend(backend, mode=mode)
            self.assertEqual(result["mode"], mode)
            self.assertEqual(backend.predict_calls[0]["mode"], mode)
            prompts.append(result["roles"]["target"]["prompt"])
        self.assertEqual(prompts[0], prompts[1])

    def test_empty_prediction_and_empty_mask_are_distinct(self):
        backend = MockBackend({"target": (np.empty((0, 24, 32)), []),
                               "reference": (np.zeros((1, 24, 32)), [.9])})
        result = self.run_backend(backend)
        self.assertEqual(result["roles"]["target"]["failure_reason"], "empty_prediction")
        reference = result["roles"]["reference"]
        self.assertEqual(reference["failure_reason"], "all_candidates_rejected")
        self.assertIn("empty_mask", reference["candidates"][0]["failure_reasons"])
        self.assertIsNone(reference["selected_index"])

    def test_giant_mask_rejected_even_with_high_sam_score(self):
        result = self.run_backend(MockBackend({"target": (np.ones((1, 24, 32)), [1.0])}))
        item = result["roles"]["target"]
        self.assertIsNone(item["selected_index"])
        self.assertIn("abnormal_mask_area", item["candidates"][0]["failure_reasons"])

    def test_low_quality_rejected_explicitly(self):
        result = self.run_backend(self.backend(), selection_config=SelectionConfig(min_quality=1.0))
        self.assertIsNone(result["roles"]["target"]["selected_index"])
        self.assertIn("low_quality", result["roles"]["target"]["candidates"][0]["failure_reasons"])

    def test_center_gate_rejects_incorrect_depth_instance(self):
        self.guidance.metadata["target"]["center_world"][2] = 2.0
        result = self.run_backend(self.backend())
        self.assertIn("center_inconsistent", result["roles"]["target"]["candidates"][0]["failure_reasons"])

    def test_semantic_null_not_visible_lowconfidence_never_conflated(self):
        for status in ("NULL", "NOT_VISIBLE", "LOW_CONFIDENCE"):
            with self.subTest(status=status):
                guidance = copy.deepcopy(self.guidance)
                guidance.metadata["reference"] = {"status": status, "semantic_valid": status != "NULL",
                    "confidence": .1, "center_world": None, "points_key": "reference_points_world", "point_count": 0}
                guidance = SlotGuidance(guidance.metadata, guidance.target_points_world, np.empty((0, 3)))
                backend = self.backend()
                result = run_camera(guidance, self.camera, self.rgb, self.depth, backend)
                item = result["roles"]["reference"]
                self.assertEqual(item["status"], status)
                self.assertEqual(item["failure_reason"], status)
                self.assertIsNone(item["selected_index"])
                self.assertEqual([call["role"] for call in backend.predict_calls], ["target"])

    def test_missing_calibration_and_wrong_source_index_stop_encoding(self):
        for camera, reason in ((replace(self.camera, extrinsics=None), "MISSING_CALIBRATION"),
                               (replace(self.camera, source_frame_index=3), "FRAME_MISMATCH")):
            with self.subTest(reason=reason):
                backend = self.backend()
                result = run_camera(self.guidance, camera, self.rgb, self.depth, backend)
                self.assertEqual(backend.set_image_calls, 0)
                self.assertEqual(result["roles"]["target"]["failure_reason"], reason)

    def test_invalid_depth_stops_encoding_without_fabricated_mask(self):
        backend = self.backend()
        result = run_camera(self.guidance, self.camera, self.rgb, np.zeros_like(self.depth), backend)
        self.assertEqual(backend.set_image_calls, 0)
        self.assertIsNone(result["roles"]["target"]["selected_index"])
        self.assertEqual(result["roles"]["target"]["candidates"], [])

    def test_predict_exception_is_role_local(self):
        backend = MockBackend({"target": RuntimeError("mock decoder failure"),
                               "reference": (self.masks["reference"][None], [.6])})
        result = self.run_backend(backend)
        self.assertIn("sam3_exception:RuntimeError:mock decoder failure", result["roles"]["target"]["failure_reason"])
        self.assertEqual(result["roles"]["reference"]["selected_index"], 0)

    def test_encoder_exception_has_explicit_fallback_for_each_valid_role(self):
        backend = self.backend()
        with patch.object(backend, "set_image", side_effect=RuntimeError("encoder failure")):
            result = self.run_backend(backend)
        for role in self.masks:
            self.assertIn("image_encoder_exception", result["roles"][role]["failure_reason"])
            self.assertIsNone(result["roles"][role]["selected_index"])
        self.assertEqual(backend.predict_calls, [])

    def test_bad_output_shape_and_score_count_are_explicit_failures(self):
        for masks, scores in ((np.ones((1, 12, 16)), [.5]), (np.ones((2, 24, 32)), [.5]),
                              (np.full((1, 24, 32), np.nan), [.5])):
            with self.subTest(shape=masks.shape, scores=scores):
                result = self.run_backend(MockBackend({"target": (masks, scores)}))
                self.assertIn("sam3_exception:ValueError", result["roles"]["target"]["failure_reason"])

    def test_nonfinite_score_is_saved_as_null_and_rejected(self):
        result = self.run_backend(MockBackend({"target": (self.masks["target"][None], [np.nan])}))
        item = result["roles"]["target"]["candidates"][0]
        self.assertIsNone(item["sam3_score"])
        self.assertFalse(item["accepted"])
        self.assertIn("nonfinite_sam3_score", item["failure_reasons"])

    def test_mock_fingerprint_and_outputs_are_isolated(self):
        masks = self.masks["target"][None].copy()
        backend = MockBackend({"target": (masks, [.5])})
        digest = backend.fingerprint()
        masks[:] = False
        self.assertEqual(digest, backend.fingerprint())
        result = self.run_backend(backend)
        result["roles"]["target"]["candidates"][0]["mask"][:] = False
        self.assertTrue(self.run_backend(backend)["roles"]["target"]["candidates"][0]["mask"].any())

    def test_import_without_sam3_or_torch(self):
        script = "import sys; import slot_guided_sam3; assert not ({'torch','sam3','sam3_runtime'} & set(sys.modules))"
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=os.environ.copy(), check=True,
                       capture_output=True, text=True)

    def test_invalid_configuration_and_mode(self):
        for config in ({"min_quality": 1.2}, {"max_center_error_m": 0}, {"min_mask_pixels": True},
                       {"depth_tolerance_m": float("nan")}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                SelectionConfig(**config)
        with self.assertRaises(ValueError):
            self.run_backend(self.backend(), mode="unreviewed")

    def test_unreviewed_installation_rejected_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "unverified SAM3 API"):
                verify_sam3_api_contract(directory)
            path = Path(directory) / "model"
            path.mkdir()
            (path / "sam3_image.py").write_text("# incompatible implementation", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                verify_sam3_api_contract(directory)
        with self.assertRaises(FileNotFoundError):
            Sam3Backend(ROOT / "missing_sam3_checkpoint.pt")

    def test_missing_interactive_predictor_is_not_treated_as_api_support(self):
        with self.assertRaisesRegex(RuntimeError, "interactivity"):
            _check_predict_signature(SimpleNamespace(predict_inst=lambda *args: None), SimpleNamespace(set_image=lambda x: x))

    def test_optional_checked_official_source_matches_contract(self):
        path = Path(os.environ.get("SAM3_AUDIT_ROOT", str(ROOT.parent / "sam3-api-source"))) / "sam3"
        if not path.is_dir():
            self.skipTest("official source checkout not available; no model installation required")
        result = verify_sam3_api_contract(path)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(AUDITED_SAM3_SHA), 40)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional tensor gradient-boundary check")
class FrozenBackendContractTests(unittest.TestCase):
    def test_real_wrapper_freezes_and_detaches_with_reviewed_call_arguments(self):
        """A fake model checks our wrapper, not the unavailable learned SAM3."""
        import torch
        from PIL import Image

        observed = {"encode": [], "predict": []}

        class Predictor:
            def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None,
                        multimask_output=True, return_logits=False, normalize_coords=True):
                pass

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.inst_interactive_predictor = Predictor()

            def predict_inst(self, state, **kwargs):
                observed["predict"].append((torch.is_grad_enabled(), kwargs, id(state)))
                return torch.ones((1, 24, 32)), torch.tensor([.8]), None

        model = Model()

        def build_model(**kwargs):
            observed["builder"] = kwargs
            return model

        class Processor:
            def __init__(self, model, device):
                pass

            def set_image(self, image):
                observed["encode"].append((torch.is_grad_enabled(), isinstance(image, Image.Image), image.size))
                return {"original_width": image.width, "original_height": image.height}

        modules = {}
        for name in ("sam3", "sam3.model_builder", "sam3.model", "sam3.model.sam3_image_processor"):
            modules[name] = ModuleType(name)
        modules["sam3.model_builder"].build_sam3_image_model = build_model
        modules["sam3.model.sam3_image_processor"].Sam3Processor = Processor
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.pt"
            checkpoint.write_bytes(b"not a real checkpoint: fake builder only")
            spec = SimpleNamespace(submodule_search_locations=[directory])
            with patch.dict(sys.modules, modules), patch("slot_guided_sam3.importlib.util.find_spec", return_value=spec), \
                    patch("slot_guided_sam3.verify_sam3_api_contract", return_value={"test": "mock_only"}):
                backend = Sam3Backend(checkpoint, "cpu")
            guidance, camera, rgb, depth, _ = fixture()
            from slot_prompt_adapter import build_prompts
            prompt = build_prompts(guidance, camera, rgb, depth)["target"]
            state = backend.set_image(rgb)
            masks, scores = backend.predict(state, role="target", prompt=prompt, mode="point+box")
            backend.predict(state, role="target", prompt=prompt, mode="box-only")
        self.assertFalse(model.training)
        self.assertFalse(model.weight.requires_grad)
        self.assertEqual(observed["encode"], [(False, True, (32, 24))])
        self.assertFalse(observed["builder"]["load_from_HF"])
        self.assertTrue(observed["builder"]["enable_inst_interactivity"])
        for grad, kwargs, state_id in observed["predict"]:
            self.assertFalse(grad)
            self.assertEqual(state_id, id(state))
            self.assertTrue(kwargs["multimask_output"])
            self.assertTrue(kwargs["normalize_coords"])
            self.assertFalse(kwargs["return_logits"])
            np.testing.assert_array_equal(kwargs["box"], np.asarray(prompt.bbox_xyxy, dtype=np.float32))
        kwargs = observed["predict"][0][1]
        self.assertEqual(kwargs["point_coords"].shape[1], 2)
        self.assertEqual(set(kwargs["point_labels"].tolist()), {0, 1})
        self.assertIsNone(observed["predict"][1][1]["point_coords"])
        self.assertIsInstance(masks, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)


if __name__ == "__main__":
    unittest.main()
