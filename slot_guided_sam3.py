"""Detached, offline role-guided SAM3 instance inference.

Importing this module and using MockBackend requires neither torch nor SAM3.
No role identity is inferred from a SAM3 category/quality score. All candidates
are retained, including rejected masks, for offline diagnosis and fusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import inspect
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from camera_geometry import backproject_mask, project_points, transform_points
from slot_guidance import SlotGuidance
from slot_prompt_adapter import AdapterConfig, CameraFrame, RolePrompt, build_prompts


AUDITED_SAM3_SHA = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
# LF-normalized full-source hashes, checked before importing installed SAM3.
SAM3_CONTRACT_HASHES = {
    "model/sam3_image.py": "2c20de90fb0b76b01dd437385eb7226f8c3cd4757a4c80cc8dec943dcf46a4a0",
    "model/sam3_image_processor.py": "d8738a0efb6138b01c0dc5deceffd29de9e675860a9a1ed3822b08766373333b",
    "model/sam1_task_predictor.py": "f8902ea6b997495875302cda47dc858c4178bd5d4f8497f40bc8e8531b0ca48a",
    "model/utils/sam1_utils.py": "3eb335d6eb447e117eb079136cb897475a0210d41b3990edb830c5866b587c2a",
}


def verify_sam3_api_contract(package_root: str | Path) -> dict[str, str]:
    """Reject unreviewed installed implementations, allowing CRLF checkouts.

    A compatible new SAM3 revision needs a deliberate source audit and updated
    hashes; parameter names alone cannot prove the coordinate convention.
    """
    root = Path(package_root)
    for name, expected in SAM3_CONTRACT_HASHES.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"unverified SAM3 API: missing {name}")
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"unverified SAM3 API: source mismatch in {name}; "
                               f"use audited revision {AUDITED_SAM3_SHA}")
    return dict(SAM3_CONTRACT_HASHES)


def _check_predict_signature(model: Any, processor: Any) -> None:
    if not callable(getattr(model, "predict_inst", None)):
        raise RuntimeError("SAM3 model has no predict_inst")
    predictor = getattr(model, "inst_interactive_predictor", None)
    if predictor is None or not callable(getattr(predictor, "predict", None)):
        raise RuntimeError("SAM3 instance interactivity is required")
    parameters = inspect.signature(predictor.predict).parameters
    required = {"point_coords", "point_labels", "box", "mask_input", "multimask_output",
                "return_logits", "normalize_coords"}
    if not required.issubset(parameters):
        raise RuntimeError(f"SAM3 predictor signature lacks {sorted(required - parameters.keys())}")
    if not callable(getattr(processor, "set_image", None)):
        raise RuntimeError("SAM3 processor has no set_image")


class Sam3Backend:
    """Frozen real backend with explicit checkpoint and reviewed pixel API.

    Construction never downloads a checkpoint. SAM3 calls are serialized by
    run_camera; the official predictor temporarily mutates its internal state.
    Do not share one instance between concurrent threads.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu", no_bf16: bool = False):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
        spec = importlib.util.find_spec("sam3")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("SAM3 is not installed; use MockBackend or install the audited source")
        self.contract_hashes = verify_sam3_api_contract(next(iter(spec.submodule_search_locations)))
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3_runtime import autocast_context, normalize_masks, normalize_scores

        if device != "cpu" and not (device == "cuda" or device.startswith("cuda:")):
            raise ValueError("device must be cpu, cuda or cuda:<index>")
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.torch = torch
        self.device = device
        self.no_bf16 = no_bf16
        self.autocast_context = autocast_context
        self.normalize_masks = normalize_masks
        self.normalize_scores = normalize_scores
        # The reviewed builder handles only the literal 'cuda'; .to covers indices.
        builder_device = "cpu" if device.startswith("cuda:") else device
        self.model = build_sam3_image_model(checkpoint_path=str(checkpoint), device=builder_device,
                                           eval_mode=True, load_from_HF=False,
                                           enable_inst_interactivity=True)
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.processor = Sam3Processor(self.model, device=device)
        _check_predict_signature(self.model, self.processor)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self._fingerprint = {"kind": "sam3", "api_source_git_sha": AUDITED_SAM3_SHA,
                             "api_source_hashes": self.contract_hashes,
                             "checkpoint_sha256": digest.hexdigest(), "device": device,
                             "no_bf16": bool(no_bf16), "torch_version": torch.__version__,
                             "frozen": True}

    def fingerprint(self) -> dict[str, Any]:
        return dict(self._fingerprint)

    def _autocast(self):
        # Reuse the existing autocast implementation for all CUDA index forms.
        return self.autocast_context("cuda" if self.device.startswith("cuda") else "cpu", self.no_bf16)

    def start_measurement(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def finish_measurement(self) -> int | None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
            return int(self.torch.cuda.max_memory_allocated(self.device))
        return None

    def set_image(self, image_rgb: np.ndarray) -> Any:
        from PIL import Image
        # Official processor's ndarray branch reads shape[-2:], which is wrong
        # for HWC. PIL preserves the original (width,height) without resizing.
        with self.torch.inference_mode(), self._autocast():
            state = self.processor.set_image(Image.fromarray(image_rgb, mode="RGB"))
        if (state.get("original_height"), state.get("original_width")) != image_rgb.shape[:2]:
            raise RuntimeError("SAM3 image state has incorrect original image dimensions")
        return state

    def predict(self, state: Any, *, role: str, prompt: RolePrompt, mode: str):
        positive = np.asarray(prompt.positive_points_xy, dtype=np.float32).reshape(-1, 2)
        negative = np.asarray(prompt.negative_points_xy, dtype=np.float32).reshape(-1, 2)
        coords = labels = None
        if mode == "point+box":
            coords = np.concatenate((positive, negative))
            labels = np.concatenate((np.ones(len(positive), dtype=np.int32),
                                     np.zeros(len(negative), dtype=np.int32)))
            if not len(positive):
                raise ValueError("point+box requires verified positive points")
        try:
            with self.torch.inference_mode(), self._autocast():
                masks, scores, _ = self.model.predict_inst(
                    state, point_coords=coords, point_labels=labels,
                    box=np.asarray(prompt.bbox_xyxy, dtype=np.float32), mask_input=None,
                    multimask_output=True, return_logits=False, normalize_coords=True,
                )
        finally:
            # The reviewed official wrapper clears these only on success.
            self.model.inst_interactive_predictor._features = None
            self.model.inst_interactive_predictor._is_image_set = False
        masks = self.normalize_masks(masks)
        return masks, self.normalize_scores(scores, len(masks))


class MockBackend:
    """Caller-specified deterministic masks; this is never a learned predictor.

    ``candidates`` maps each role to ``(masks[C,H,W], scores[C])`` or an exception
    to raise for that role. Missing roles produce an empty candidate sequence.
    """

    def __init__(self, candidates: Mapping[str, Any]):
        self.candidates = {}
        for role, value in candidates.items():
            if role not in ("target", "reference"):
                raise ValueError(f"unknown role {role}")
            self.candidates[role] = value if isinstance(value, Exception) else (
                np.array(value[0], copy=True), np.array(value[1], copy=True))
        self.set_image_calls = 0
        self.predict_calls: list[dict[str, Any]] = []

    def fingerprint(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for role in sorted(self.candidates):
            digest.update(role.encode())
            value = self.candidates[role]
            if isinstance(value, Exception):
                digest.update(f"{type(value).__name__}:{value}".encode())
            else:
                for array in value:
                    digest.update(str((array.shape, array.dtype.str)).encode())
                    digest.update(array.tobytes())
        return {"kind": "mock", "candidates_sha256": digest.hexdigest(), "frozen": True}

    def set_image(self, image_rgb: np.ndarray) -> dict[str, Any]:
        self.set_image_calls += 1
        return {"shape": image_rgb.shape, "image_call": self.set_image_calls}

    def predict(self, state: Any, *, role: str, prompt: RolePrompt, mode: str):
        self.predict_calls.append({"state_id": id(state), "role": role,
                                   "prompt": prompt.as_dict(), "mode": mode})
        value = self.candidates.get(role)
        if isinstance(value, Exception):
            raise value
        if value is None:
            return np.empty((0, *state["shape"][:2]), dtype=bool), np.empty(0)
        return np.array(value[0], copy=True), np.array(value[1], copy=True)


@dataclass(frozen=True)
class SelectionConfig:
    min_mask_pixels: int = 4
    max_mask_fraction: float = 0.6
    min_positive_coverage: float = 1.0
    max_negative_coverage: float = 0.0
    min_depth_fraction: float = 0.5
    min_box_fraction: float = 0.5
    min_geometry_support: float = 0.5
    max_center_error_m: float = 0.15
    depth_tolerance_m: float = 0.03
    min_quality: float = 0.5
    max_center_samples: int = 2048

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"nonfinite selection setting {key}")
        for key in ("max_mask_fraction", "min_positive_coverage", "max_negative_coverage",
                    "min_depth_fraction", "min_box_fraction", "min_geometry_support", "min_quality"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"selection {key} must be in [0,1]")
        for key in ("min_mask_pixels", "max_center_samples"):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"selection {key} must be a positive integer")
        if self.max_center_error_m <= 0 or self.depth_tolerance_m <= 0:
            raise ValueError("selection metric tolerances must be positive")


def _coverage(mask: np.ndarray, points: Any, *, empty: float) -> float:
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(xy):
        return empty
    rounded = np.floor(xy + 0.5).astype(int)
    valid = (rounded[:, 0] >= 0) & (rounded[:, 0] < mask.shape[1]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < mask.shape[0])
    hit = np.zeros(len(xy), dtype=bool)
    hit[valid] = mask[rounded[valid, 1], rounded[valid, 0]]
    return float(hit.mean())


def score_candidate(mask: np.ndarray, sam3_score: float, *, prompt: RolePrompt,
                    guidance: SlotGuidance, role: str, camera: CameraFrame,
                    depth_m: np.ndarray, config: SelectionConfig) -> dict[str, Any]:
    """Score prompt agreement plus measured metric geometry, keeping failures."""
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    reasons = []
    area_fraction = count / mask.size
    if count < config.min_mask_pixels:
        reasons.append("empty_mask" if count == 0 else "mask_too_small")
    if area_fraction > config.max_mask_fraction:
        reasons.append("abnormal_mask_area")
    positive = _coverage(mask, prompt.positive_points_xy, empty=0.0)
    negative = _coverage(mask, prompt.negative_points_xy, empty=0.0)
    if positive < config.min_positive_coverage:
        reasons.append("positive_prompt_miss")
    if negative > config.max_negative_coverage:
        reasons.append("negative_prompt_hit")
    yy, xx = np.nonzero(mask)
    box = prompt.bbox_xyxy
    in_box = (xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])
    box_fraction = float(in_box.mean()) if count else 0.0
    if box_fraction < config.min_box_fraction:
        reasons.append("box_inconsistent")
    depth_valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_fraction = float(depth_valid[mask].mean()) if count else 0.0
    if depth_fraction < config.min_depth_fraction:
        reasons.append("insufficient_depth_support")
    world = transform_points(backproject_mask(depth_m, mask, np.asarray(camera.intrinsics),
                                              config.max_center_samples), np.asarray(camera.extrinsics))
    center = np.median(world, axis=0) if len(world) else None
    expected_center = guidance.metadata[role]["center_world"]
    center_error = float(np.linalg.norm(center - expected_center)) if center is not None and expected_center is not None else None
    if center_error is None:
        reasons.append("no_verified_center")
    elif center_error > config.max_center_error_m:
        reasons.append("center_inconsistent")
    points = guidance.points(role)
    support_fraction = None
    if len(points):
        pixels, positive_depth = project_points(points, np.asarray(camera.intrinsics), np.asarray(camera.extrinsics))
        finite = np.isfinite(pixels).all(axis=1)
        h, w = depth_m.shape
        safe_pixels = np.clip(np.where(finite[:, None], pixels, -1), [-1, -1], [w, h])
        rounded = np.floor(safe_pixels + 0.5).astype(int)
        eligible = positive_depth & finite & (rounded[:, 0] >= 0) & (rounded[:, 0] < w) & (rounded[:, 1] >= 0) & (rounded[:, 1] < h)
        inds = np.flatnonzero(eligible)
        if len(inds):
            sampled_depth = depth_m[rounded[inds, 1], rounded[inds, 0]]
            cam_points = transform_points(points[inds], np.linalg.inv(np.asarray(camera.extrinsics)))
            agreement = np.isfinite(sampled_depth) & (sampled_depth > 0) & (np.abs(sampled_depth - cam_points[:, 2]) <= config.depth_tolerance_m)
            observed_inds = inds[agreement]
            support_fraction = _coverage(mask, pixels[observed_inds], empty=0.0)
        else:
            support_fraction = 0.0
        if support_fraction < config.min_geometry_support:
            reasons.append("guidance_geometry_miss")
    center_quality = float(np.exp(-center_error / config.max_center_error_m)) if center_error is not None else 0.0
    # SAM quality is a small secondary contribution; it cannot override gates.
    quality = (0.25 * positive + 0.15 * (1 - negative) + 0.15 * box_fraction
               + 0.15 * depth_fraction + 0.15 * center_quality
               + 0.10 * (support_fraction if support_fraction is not None else depth_fraction)
               + 0.05 * float(np.clip(sam3_score, 0, 1)))
    if not np.isfinite(sam3_score):
        reasons.append("nonfinite_sam3_score")
        quality = 0.0
    if quality < config.min_quality:
        reasons.append("low_quality")
    return {"mask": mask.copy(), "sam3_score": float(sam3_score) if np.isfinite(sam3_score) else None,
            "quality_score": float(quality), "accepted": not reasons, "failure_reasons": reasons,
            "metrics": {"area_pixels": count, "area_fraction": area_fraction,
                        "positive_coverage": positive, "negative_coverage": negative,
                        "box_fraction": box_fraction, "depth_support_fraction": depth_fraction,
                        "guidance_geometry_support": support_fraction,
                        "center_world": center.tolist() if center is not None else None,
                        "center_error_m": center_error}}


def _normalize_prediction(masks: Any, scores: Any, shape: tuple[int, int]):
    # Mock path deliberately does not import sam3_runtime (which imports torch).
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[1:] != shape or masks.dtype.kind not in "biuf":
        raise ValueError(f"SAM3 masks must be [C,{shape[0]},{shape[1]}] numeric, got {masks.shape}")
    if not np.isfinite(masks).all():
        raise ValueError("nonfinite SAM3 masks")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(masks):
        raise ValueError("SAM3 mask/score count mismatch")
    return masks > 0.5, scores


def run_camera(guidance: SlotGuidance, camera: CameraFrame, image_rgb: np.ndarray,
               depth_m: np.ndarray, backend: Any, mode: str = "point+box",
               adapter_config: AdapterConfig | None = None,
               selection_config: SelectionConfig | None = None) -> dict[str, Any]:
    """Run one current-frame RGB encoder, sharing its state between both roles.

    On failure the explicit fallback is no selected SAM mask. It never invents
    a mask/box, retries with another role identity, or reads other observations.
    """
    if mode not in ("box-only", "point+box"):
        raise ValueError("mode must be box-only or point+box")
    config = selection_config or SelectionConfig()
    start_measurement = getattr(backend, "start_measurement", None)
    if start_measurement:
        start_measurement()
    started = time.perf_counter()
    prompts = build_prompts(guidance, camera, image_rgb, depth_m, adapter_config)
    result = {"camera": camera.name, "mode": mode, "roles": {},
              "backend": backend.fingerprint(), "selection_config": asdict(config),
              "fallback": "no_selected_mask_preserve_guidance_status"}
    state = None
    encoding_error = None
    if any(prompt.prompt_valid for prompt in prompts.values()):
        try:
            state = backend.set_image(image_rgb)
        except Exception as exc:
            encoding_error = f"image_encoder_exception:{type(exc).__name__}:{exc}"
    for role in ("target", "reference"):
        prompt = prompts[role]
        role_result = {"prompt": prompt.as_dict(), "candidates": [], "selected_index": None,
                       "status": guidance.metadata[role]["status"], "failure_reason": prompt.failure_reason}
        result["roles"][role] = role_result
        if not prompt.prompt_valid:
            continue
        if encoding_error:
            role_result["failure_reason"] = encoding_error
            continue
        try:
            masks, scores = backend.predict(state, role=role, prompt=prompt, mode=mode)
            masks, scores = _normalize_prediction(masks, scores, tuple(depth_m.shape))
            for mask, score in zip(masks, scores):
                role_result["candidates"].append(score_candidate(
                    mask, float(score), prompt=prompt, guidance=guidance, role=role,
                    camera=camera, depth_m=depth_m, config=config))
            accepted = [(index, item) for index, item in enumerate(role_result["candidates"]) if item["accepted"]]
            if accepted:
                role_result["selected_index"] = max(accepted, key=lambda pair: pair[1]["quality_score"])[0]
                role_result["failure_reason"] = None
            else:
                role_result["failure_reason"] = "empty_prediction" if not len(masks) else "all_candidates_rejected"
        except Exception as exc:
            role_result["failure_reason"] = f"sam3_exception:{type(exc).__name__}:{exc}"
    finish_measurement = getattr(backend, "finish_measurement", None)
    result["peak_vram_bytes"] = finish_measurement() if finish_measurement else None
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    return result
