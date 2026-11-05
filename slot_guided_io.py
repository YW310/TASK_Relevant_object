"""Strict single-observation RGB-D snapshots and offline SAM3 artifacts.

No episode pickle, observation iterator or expert action is opened here. Snapshot
producers must export only the current observation and attest its frame/time.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from camera_geometry import decode_rlbench_rgb_depth
from slot_guidance import SlotGuidance, load_guidance, _timestamp
from slot_prompt_adapter import CameraFrame


@dataclass
class RGBDFrame:
    camera: CameraFrame
    rgb: np.ndarray
    depth_m: np.ndarray
    metadata: dict[str, Any]


@dataclass
class FrameSnapshot:
    metadata: dict[str, Any]
    cameras: dict[str, RGBDFrame]
    path: Path


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key}")
            out[key] = value
        return out
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def load_frame_snapshot(path: str | Path, *, expected_source_frame_index: int | None = None) -> FrameSnapshot:
    """Load explicit current-frame files; never infer calibration from frame_id."""
    path = Path(path).resolve()
    metadata = _read_json(path)
    required = {"schema_version", "artifact_type", "episode_id", "frame_id", "source_frame_index", "observed_at", "cameras"}
    if set(metadata) != required or type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1 or metadata["artifact_type"] != "slot_rgbd_frame":
        raise ValueError("invalid single-frame snapshot fields/schema")
    index = metadata["source_frame_index"]
    if type(index) is not int or index < 0:
        raise ValueError("source_frame_index must be an explicit nonnegative integer")
    if expected_source_frame_index is not None and index != expected_source_frame_index:
        raise ValueError("snapshot source_frame_index does not match requested observation")
    for key in ("episode_id", "frame_id", "observed_at"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"snapshot {key} is required")
    _timestamp(metadata["observed_at"], "snapshot.observed_at")
    rows = metadata["cameras"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("snapshot cameras must be a nonempty list")
    cameras = {}
    for row in rows:
        needed = {"name", "rgb_path", "depth_path", "depth_encoding", "source_frame_index", "observed_at", "calibration_source"}
        allowed = needed | {"intrinsics", "extrinsics", "near", "far", "instance_masks"}
        if not isinstance(row, dict) or not needed.issubset(row) or set(row) - allowed:
            raise ValueError("invalid camera snapshot fields")
        name = row["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in cameras:
            raise ValueError("camera names must be unique safe identifiers")
        if type(row["source_frame_index"]) is not int or row["source_frame_index"] != index or row["observed_at"] != metadata["observed_at"]:
            raise ValueError("camera calibration must belong to the exact source frame/time")
        if not isinstance(row["calibration_source"], str) or not row["calibration_source"].strip():
            raise ValueError("calibration_source is required")
        rgb_path = path.parent / row["rgb_path"]
        with Image.open(rgb_path) as image:
            rgb = np.array(image.convert("RGB"))
        depth_path = path.parent / row["depth_path"]
        if row["depth_encoding"] == "meters_npy":
            depth = np.load(depth_path, allow_pickle=False)
            if not isinstance(depth, np.ndarray) or depth.dtype.kind not in "fiu":
                raise ValueError("metric depth must be a real numeric NPY array")
            depth = depth.astype(np.float64)
        elif row["depth_encoding"] == "rlbench_rgb24":
            near, far = row.get("near"), row.get("far")
            if isinstance(near, bool) or isinstance(far, bool) or not isinstance(near, (int, float)) or not isinstance(far, (int, float)) or not np.isfinite([near, far]).all() or not 0 < near < far:
                raise ValueError("packed depth requires current-frame finite 0 < near < far")
            with Image.open(depth_path) as image:
                packed = np.array(image)
            if packed.ndim != 3 or packed.shape[-1] != 3 or packed.dtype != np.uint8:
                raise ValueError("RLBench packed depth must be RGB uint8")
            depth = decode_rlbench_rgb_depth(packed, near, far)
        else:
            raise ValueError("unsupported depth_encoding; use meters_npy or rlbench_rgb24")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("original RGB and depth must be registered at identical resolution")
        camera = CameraFrame(name=name, episode_id=metadata["episode_id"], frame_id=metadata["frame_id"],
                             source_frame_index=index, observed_at=metadata["observed_at"],
                             intrinsics=row.get("intrinsics"), extrinsics=row.get("extrinsics"),
                             image_size=(rgb.shape[1], rgb.shape[0]), calibration_source=row["calibration_source"])
        cameras[name] = RGBDFrame(camera, rgb, depth, copy.deepcopy(row))
    return FrameSnapshot(metadata, cameras, path)


def _array_digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def calibration_digest(snapshot: FrameSnapshot) -> str:
    """Identity + each current camera's K/T/time/provenance, with stable ordering."""
    value = {key: snapshot.metadata[key] for key in ("episode_id", "frame_id", "source_frame_index", "observed_at")}
    value["cameras"] = [{key: frame.metadata.get(key) for key in
                         ("name", "intrinsics", "extrinsics", "source_frame_index", "observed_at", "calibration_source")}
                        for _, frame in sorted(snapshot.cameras.items())]
    return hashlib.sha256(_canonical(value)).hexdigest()


def snapshot_input_digest(snapshot: FrameSnapshot) -> str:
    value = {"calibration_sha256": calibration_digest(snapshot),
             "images": {name: {"rgb": _array_digest(frame.rgb), "depth": _array_digest(frame.depth_m)}
                        for name, frame in sorted(snapshot.cameras.items())}}
    return hashlib.sha256(_canonical(value)).hexdigest()


def guidance_input_digest(guidance: SlotGuidance) -> str:
    value = {"metadata": guidance.metadata,
             "points": {role: _array_digest(guidance.points(role)) for role in ("target", "reference")}}
    return hashlib.sha256(_canonical(value)).hexdigest()


def inference_cache_key(snapshot: FrameSnapshot, guidance: SlotGuidance, backend: Mapping[str, Any],
                        *, mode: str, adapter_config: Any = None, selection_config: Any = None) -> str:
    """Bind cache to actual inputs, settings, checkpoint/backend and inference code."""
    from dataclasses import is_dataclass
    config = lambda value: asdict(value) if is_dataclass(value) else value
    code = {}
    for name in ("slot_guidance.py", "slot_prompt_adapter.py", "slot_guided_sam3.py", "slot_guided_io.py",
                 "camera_geometry.py", "src/relevant_object/geometry/camera.py"):
        code[name] = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    value = {"snapshot": snapshot.metadata, "guidance": guidance.metadata, "backend": backend,
             "mode": mode, "adapter_config": config(adapter_config), "selection_config": config(selection_config),
             "code": code, "points": {role: _array_digest(guidance.points(role)) for role in ("target", "reference")},
             "images": {name: {"rgb": _array_digest(frame.rgb), "depth": _array_digest(frame.depth_m)}
                        for name, frame in snapshot.cameras.items()}}
    return hashlib.sha256(_canonical(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload + b"\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_inference_artifact(path: str | Path, report: Mapping[str, Any], snapshot: FrameSnapshot) -> dict[str, Any]:
    """Keep every candidate mask and score; render selected masks and prompt boxes."""
    path = Path(path)
    value = copy.deepcopy(dict(report))
    arrays = {}
    for name, result in value["cameras"].items():
        if name not in snapshot.cameras:
            raise ValueError("result camera does not match snapshot")
        rgb = snapshot.cameras[name].rgb
        overlay = Image.fromarray(rgb)
        legend = []
        for role, color in (("target", (255, 80, 40)), ("reference", (40, 150, 255))):
            row = result["roles"][role]
            for index, candidate in enumerate(row["candidates"]):
                mask = np.asarray(candidate.pop("mask"))
                if mask.shape != rgb.shape[:2] or mask.dtype != bool:
                    raise ValueError("candidate masks must be original-resolution bool arrays")
                key = f"{name}/{role}/{index}"
                arrays[key] = mask
                candidate["mask_key"] = key
                if index == row["selected_index"]:
                    tint = np.array(overlay).astype(float)
                    tint[mask] = tint[mask] * 0.55 + np.array(color) * 0.45
                    overlay = Image.fromarray(np.clip(tint, 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(overlay)
            prompt = row["prompt"]
            if prompt.get("bbox_xyxy") is not None:
                draw.rectangle(prompt["bbox_xyxy"], outline=color, width=1)
            for x, y in prompt.get("positive_points_xy", []):
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
            for x, y in prompt.get("negative_points_xy", []):
                draw.line([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255))
            legend.append((role, row["status"], row["selected_index"], row["failure_reason"], color))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep original pixels and prompt coordinates intact; put readable
        # diagnostics beside even tiny RGB images instead of covering objects.
        canvas = Image.new("RGB", (overlay.width + 280, max(overlay.height, 110)), (25, 25, 25))
        canvas.paste(overlay, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((overlay.width + 8, 4), name, fill=(235, 235, 235))
        for index, (role, status, chosen, failure, color) in enumerate(legend):
            y = 22 + index * 42
            draw.text((overlay.width + 8, y), f"{role}: {status}; candidate={chosen}", fill=color)
            if failure:
                draw.text((overlay.width + 8, y + 14), str(failure)[:45], fill=(235, 235, 235))
        canvas.save(path.parent / f"{name}.diagnostic.png")
        result["diagnostic_image"] = f"{name}.diagnostic.png"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    mask_name = f"masks.{digest}.npz"
    mask_path = path.parent / mask_name
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, mask_path)
    finally:
        temporary.unlink(missing_ok=True)
    value.update(masks_file=mask_name, masks_sha256=digest)
    atomic_json(path, value)
    return value


def load_inference_artifact(path: str | Path, *, expected_cache_key: str | None = None) -> dict[str, Any]:
    path = Path(path)
    value = _read_json(path)
    if value.get("schema_version") != 1 or value.get("artifact_type") != "slot_sam3_frame":
        raise ValueError("incompatible inference artifact")
    if expected_cache_key is not None and value.get("cache_key") != expected_cache_key:
        raise ValueError("stale inference cache key")
    filename = value.get("masks_file", "")
    if not isinstance(filename, str) or not re.fullmatch(r"masks\.[a-f0-9]{64}\.npz", filename):
        raise ValueError("invalid inference mask basename")
    payload = (path.parent / filename).read_bytes()
    if hashlib.sha256(payload).hexdigest() != value.get("masks_sha256"):
        raise ValueError("inference mask digest mismatch")
    archive = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("inference masks must be an NPZ archive")
    with archive:
        used = set()
        for result in value["cameras"].values():
            for row in result["roles"].values():
                selected = row.get("selected_index")
                candidates = row["candidates"]
                if selected is not None and (type(selected) is not int or not 0 <= selected < len(candidates)
                                              or candidates[selected].get("accepted") is not True):
                    raise ValueError("cached selection must name an accepted candidate")
                for candidate in row["candidates"]:
                    key = candidate["mask_key"]
                    mask = archive[key]
                    if mask.ndim != 2 or mask.dtype != bool:
                        raise ValueError("invalid cached candidate mask")
                    candidate["mask"] = mask.copy()
                    used.add(key)
        if set(archive.files) != used:
            raise ValueError("unreferenced or missing mask arrays")
    return value


def run_offline_frame(snapshot_path: str | Path, guidance_path: str | Path, backend: Any,
                      output_dir: str | Path, *, mode: str = "point+box", adapter_config: Any = None,
                      selection_config: Any = None, use_cache: bool = True) -> dict[str, Any]:
    """A single source observation only; backend may be per-camera mapping."""
    from slot_guided_sam3 import SelectionConfig, run_camera
    from slot_prompt_adapter import AdapterConfig
    adapter_config = adapter_config if adapter_config is not None else AdapterConfig()
    selection_config = selection_config if selection_config is not None else SelectionConfig()
    snapshot = load_frame_snapshot(snapshot_path)
    guidance = load_guidance(guidance_path, expected_source_frame_index=snapshot.metadata["source_frame_index"])
    for key in ("episode_id", "frame_id", "observed_at"):
        if guidance.metadata[key] != snapshot.metadata[key]:
            raise ValueError(f"guidance/snapshot {key} mismatch")
    fingerprint = {name: (backend[name] if isinstance(backend, Mapping) else backend).fingerprint()
                   for name in snapshot.cameras}
    cache_key = inference_cache_key(snapshot, guidance, fingerprint, mode=mode,
                                    adapter_config=adapter_config, selection_config=selection_config)
    output = Path(output_dir) / cache_key / "inference.json"
    if use_cache and output.exists():
        result = load_inference_artifact(output, expected_cache_key=cache_key)
        for key in ("episode_id", "frame_id", "source_frame_index", "observed_at", "mode"):
            expected = mode if key == "mode" else snapshot.metadata[key]
            if result.get(key) != expected:
                raise ValueError(f"cached inference {key} mismatch")
        if set(result["cameras"]) != set(snapshot.cameras):
            raise ValueError("cached inference cameras mismatch")
        for name, camera_result in result["cameras"].items():
            for row in camera_result["roles"].values():
                if any(candidate["mask"].shape != snapshot.cameras[name].rgb.shape[:2] for candidate in row["candidates"]):
                    raise ValueError("cached inference mask size mismatch")
        result.update(cache_hit=True, artifact_path=str(output))
        return result
    cameras = {}
    for name, frame in snapshot.cameras.items():
        cameras[name] = run_camera(guidance, frame.camera, frame.rgb, frame.depth_m,
                                   backend[name] if isinstance(backend, Mapping) else backend,
                                   mode=mode, adapter_config=adapter_config, selection_config=selection_config)
    report = {"schema_version": 1, "artifact_type": "slot_sam3_frame", "cache_key": cache_key,
              "episode_id": snapshot.metadata["episode_id"], "frame_id": snapshot.metadata["frame_id"],
              "source_frame_index": snapshot.metadata["source_frame_index"], "observed_at": snapshot.metadata["observed_at"],
              "coordinate_frame": "rlbench_world", "coordinate_units": "m", "mode": mode,
              "guidance_source": guidance.metadata["guidance_source"], "backend": fingerprint, "cameras": cameras,
              "guidance_metadata": guidance.metadata, "reference_null_probability": guidance.metadata["reference_null_probability"],
              "source_git_sha": guidance.metadata["source_git_sha"], "source_checkpoint": guidance.metadata["source_checkpoint"],
              "calibration_sha256": calibration_digest(snapshot),
              "snapshot_input_sha256": snapshot_input_digest(snapshot),
              "guidance_input_sha256": guidance_input_digest(guidance)}
    report["effective_config"] = {"adapter": asdict(adapter_config), "selection": asdict(selection_config)}
    save_inference_artifact(output, report, snapshot)
    report.update(cache_hit=False, artifact_path=str(output))
    return report
