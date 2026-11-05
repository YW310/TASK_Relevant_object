"""Evaluator-only metrics; labels never enter predicted perception.

Missing evidence stays null with a reason. Synthetic/Mock reports cannot grant
the real-offline acceptance required by BridgeVLA action conditioning.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from camera_geometry import backproject_mask, transform_points
from slot_guided_io import FrameSnapshot, _read_json, calibration_digest
from slot_guidance import SlotGuidance, _timestamp


METRIC_NAMES = ("role_selection_accuracy", "prompt_instance_hit_rate", "sam3_mask_iou", "multiview_surface_distance_m",
                "object_center_error_m", "action_translation_error_m", "rlbench_paired_task_success_rate",
                "per_frame_backend_latency_ms", "peak_vram_bytes")
METRIC_UNITS = {name: ("m" if name.endswith("_m") else "ms" if name.endswith("_ms") else "bytes" if name == "peak_vram_bytes" else None)
                for name in METRIC_NAMES}


def metric(value: float | None, count: int, *, reason: str | None = None, unit: str | None = None) -> dict:
    return {"value": value, "count": count, "reason": reason if value is None else None, "unit": unit}


def _mean(values: list[float], reason: str, unit: str | None = None) -> dict:
    return metric(float(np.mean(values)) if values else None, len(values), reason=reason, unit=unit)


def load_current_labels(path: str | Path, snapshot: FrameSnapshot) -> dict:
    """Read role/instance evaluation annotations for exactly the current frame."""
    path = Path(path)
    labels = _read_json(path)
    if set(labels) != {"episode_id", "frame_id", "source_frame_index", "roles"}:
        raise ValueError("invalid current-frame role labels")
    for key in ("episode_id", "frame_id", "source_frame_index"):
        if labels[key] != snapshot.metadata[key] or (key == "source_frame_index" and type(labels[key]) is not int):
            raise ValueError(f"evaluation labels {key} mismatch")
    if set(labels["roles"]) != {"target", "reference"}:
        raise ValueError("evaluation labels require both semantic roles")
    for role, row in labels["roles"].items():
        if row.get("status") not in ("VISIBLE", "NOT_VISIBLE", "NULL") or (role == "target" and row["status"] == "NULL"):
            raise ValueError("invalid annotated role status")
        if set(row.get("masks", {})) != set(snapshot.cameras):
            raise ValueError("labels must explicitly annotate every current camera")
        loaded = {}
        for name, filename in row["masks"].items():
            with Image.open(path.parent / filename) as image:
                mask = np.asarray(image.convert("L")) > 0
            if mask.shape != snapshot.cameras[name].rgb.shape[:2]:
                raise ValueError("evaluation masks must match original RGB dimensions")
            loaded[name] = mask
        if row["status"] in ("NULL", "NOT_VISIBLE") and any(mask.any() for mask in loaded.values()):
            raise ValueError("absent/unseen annotation contradicts visible mask")
        row["loaded_masks"] = loaded
    return labels


def translation_error(predicted: Any, reference: Any) -> dict:
    """Pure evaluator helper for separately supplied post-inference annotations."""
    predicted, reference = np.asarray(predicted, float), np.asarray(reference, float)
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 3 or not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("action evaluation requires matched finite [N,3] translations")
    values = np.linalg.norm(predicted - reference, axis=1).tolist()
    return _mean(values, "no_action_predictions", "m")


def paired_success(records: list[dict]) -> dict:
    seen = set()
    for row in records:
        identity = (row.get("task_name"), row.get("seed"), row.get("episode_id"))
        if any(item is None for item in identity) or identity in seen:
            raise ValueError("paired task results need unique task/seed/episode identities")
        seen.add(identity)
        if type(row.get("method_success")) is not bool or type(row.get("baseline_success")) is not bool:
            raise ValueError("paired task results must contain both measured booleans")
    return {"method": _mean([float(row["method_success"]) for row in records], "paired_rollouts_not_executed"),
            "baseline": _mean([float(row["baseline_success"]) for row in records], "paired_rollouts_not_executed"),
            "paired_difference": _mean([float(row["method_success"]) - float(row["baseline_success"]) for row in records],
                                       "paired_rollouts_not_executed")}


def load_baseline_report(path: str | Path, snapshot: FrameSnapshot, evidence_kind: str) -> dict:
    """Validate an external C report without treating it as real acceptance."""
    report = _read_json(Path(path))
    required = {"schema_version", "artifact_type", "episode_id", "frame_id", "source_frame_index", "observed_at",
                "calibration_sha256", "method", "evidence_kind", "source_checkpoint", "source_checkpoint_sha256",
                "source_git_sha", "metrics"}
    if set(report) != required or type(report["schema_version"]) is not int or report["schema_version"] != 1 or report["artifact_type"] != "ocrr_baseline_evaluation":
        raise ValueError("baseline report requires the complete ocrr_baseline_evaluation schema 1")
    if type(report["source_frame_index"]) is not int:
        raise ValueError("baseline report source_frame_index must be an integer")
    for key in ("episode_id", "frame_id", "source_frame_index", "observed_at"):
        if report[key] != snapshot.metadata[key]:
            raise ValueError(f"baseline report {key} mismatch")
    _timestamp(report["observed_at"], "baseline.observed_at")
    if report["calibration_sha256"] != calibration_digest(snapshot):
        raise ValueError("baseline report calibration digest mismatch")
    if report["method"] != "original_ocrr_action_path" or evidence_kind not in ("synthetic_mock", "real_offline") or report["evidence_kind"] != evidence_kind:
        raise ValueError("baseline must identify the original OCRR/action path and matching evidence kind")
    if not isinstance(report["source_git_sha"], str) or not re.fullmatch(r"(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})", report["source_git_sha"]):
        raise ValueError("baseline report requires the full producer Git SHA")
    if not isinstance(report["source_checkpoint"], str) or not report["source_checkpoint"].strip():
        raise ValueError("baseline report source_checkpoint is required")
    digest = report["source_checkpoint_sha256"]
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)):
        raise ValueError("baseline checkpoint digest must be lowercase SHA256")
    if evidence_kind == "real_offline" and (digest is None or report["source_checkpoint"].startswith("mock:")):
        raise ValueError("real baseline requires a real checkpoint identifier and digest")
    metrics = report["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
        raise ValueError("baseline must report every metric, using explicit null reasons when unavailable")
    for name, item in metrics.items():
        if not isinstance(item, dict) or set(item) != {"value", "count", "reason", "unit"}:
            raise ValueError(f"baseline metric {name} requires value/count/reason/unit")
        value, count, reason = item["value"], item["count"], item["reason"]
        if type(count) is not int or count < 0 or item["unit"] != METRIC_UNITS[name]:
            raise ValueError(f"invalid baseline metric {name} count/unit")
        if value is None:
            if count != 0 or not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"unavailable baseline metric {name} needs count 0 and a reason")
        elif type(value) not in (int, float) or not np.isfinite(value) or value < 0 or count < 1 or reason is not None:
            raise ValueError(f"invalid measured baseline metric {name}")
        elif name in {"role_selection_accuracy", "prompt_instance_hit_rate", "sam3_mask_iou", "rlbench_paired_task_success_rate"} and value > 1:
            raise ValueError(f"baseline probability metric {name} must be in [0,1]")
    return report


def evaluate_frame(snapshot: FrameSnapshot, guidance: SlotGuidance, inference: dict,
                   geometry: dict, labels: dict) -> dict:
    role_correct, prompt_hits, ious, local_ious, center_errors, consistency = [], [], [], [], [], []
    per_role = {}
    for role in ("target", "reference"):
        truth = labels["roles"][role]
        identity = guidance.metadata[role].get("instance_id")
        expected = truth.get("instance_id")
        role_accuracy = []
        if truth["status"] == "NULL":
            role_accuracy.append(float(guidance.metadata[role]["status"] == "NULL"))
        elif expected is not None and identity is not None:
            role_accuracy.append(float(identity == expected and guidance.metadata[role]["status"] != "NULL"))
        role_correct.extend(role_accuracy)
        role_ious, role_local_ious, role_hits, ground_points = [], [], [], []
        final_indices = {}
        for item in geometry["roles"][role].get("selected_candidates", []):
            name, index = item["camera"], item["candidate_index"]
            if name not in snapshot.cameras or name in final_indices or type(index) is not int:
                raise ValueError("invalid final geometry candidate selection")
            candidates = inference["cameras"][name]["roles"][role]["candidates"]
            if not 0 <= index < len(candidates) or candidates[index]["accepted"] is not True:
                raise ValueError("final geometry must select an accepted candidate")
            final_indices[name] = index
        for name, frame in snapshot.cameras.items():
            mask = truth["loaded_masks"][name]
            if not mask.any():
                continue
            ground_points.append(transform_points(backproject_mask(frame.depth_m, mask, np.asarray(frame.camera.intrinsics), 0),
                                                  np.asarray(frame.camera.extrinsics)))
            row = inference["cameras"][name]["roles"][role]
            prompt = row["prompt"]
            hit = False
            if prompt["prompt_valid"]:
                if inference["mode"] == "point+box":
                    points = np.floor(np.asarray(prompt["positive_points_xy"]) + 0.5).astype(int).reshape(-1, 2)
                else:
                    box = prompt["bbox_xyxy"]
                    points = np.floor(np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]) + 0.5).astype(int)
                bounded = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
                hit = bool(len(points) and bounded.all() and mask[points[:, 1], points[:, 0]].all()) if bounded.all() else False
            role_hits.append(float(hit))
            local_index = row["selected_index"]
            local = np.zeros_like(mask) if local_index is None else row["candidates"][local_index]["mask"]
            local_union = (local | mask).sum()
            role_local_ious.append(float((local & mask).sum() / local_union) if local_union else 1.0)
            selected = final_indices.get(name)
            predicted = np.zeros_like(mask) if selected is None else row["candidates"][selected]["mask"]
            union = (predicted | mask).sum()
            role_ious.append(float((predicted & mask).sum() / union) if union else 1.0)
        points = np.concatenate(ground_points) if ground_points else np.empty((0, 3))
        center = geometry["roles"][role]["center_world"]
        if len(points) and center is not None:
            center_errors.append(float(np.linalg.norm(np.asarray(center) - np.median(points, axis=0))))
        # Distinct cameras only; a single supported view does not prove agreement.
        selected_clouds = []
        for selected in geometry["roles"][role].get("selected_candidates", []):
            frame = snapshot.cameras[selected["camera"]]
            candidate = inference["cameras"][selected["camera"]]["roles"][role]["candidates"][selected["candidate_index"]]
            selected_clouds.append(transform_points(backproject_mask(frame.depth_m, candidate["mask"], np.asarray(frame.camera.intrinsics), 512),
                                                    np.asarray(frame.camera.extrinsics)))
        if len(selected_clouds) >= 2:
            from fusion_matching import symmetric_percentile_nearest_distance
            for a, b in combinations(selected_clouds, 2):
                distance = symmetric_percentile_nearest_distance(a, b, percentile=75)
                consistency.append(float(distance))
        prompt_hits.extend(role_hits)
        ious.extend(role_ious)
        local_ious.extend(role_local_ious)
        per_role[role] = {"role_selection_accuracy": _mean(role_accuracy, "comparable_instance_identities_unavailable"),
                          "prompt_hit_rate": _mean(role_hits, "role_not_visible"),
                          "mask_iou": _mean(role_ious, "role_not_visible"),
                          "camera_local_mask_iou": _mean(role_local_ious, "role_not_visible")}
    backend_kinds = {item["kind"] for item in inference["backend"].values()}
    latency = sum(result["latency_ms"] for result in inference["cameras"].values())
    vram = [result["peak_vram_bytes"] for result in inference["cameras"].values() if result["peak_vram_bytes"] is not None]
    return {"role_selection_accuracy": _mean(role_correct, "comparable_instance_identities_unavailable"),
            "prompt_instance_hit_rate": _mean(prompt_hits, "no_visible_annotated_instances"),
            "sam3_mask_iou": _mean(ious, "no_visible_annotated_instances"),
            "camera_local_mask_iou": _mean(local_ious, "no_visible_annotated_instances"),
            "multiview_surface_distance_m": _mean(consistency, "fewer_than_two_supporting_views", "m"),
            "object_center_error_m": _mean(center_errors, "no_reconstructed_annotated_centers", "m"),
            "action_translation_error_m": metric(None, 0, reason="bridgevla_action_predictions_unavailable", unit="m"),
            "rlbench_paired_task_success_rate": metric(None, 0, reason="paired_rollouts_not_executed"),
            "per_frame_backend_latency_ms": metric(float(latency), 1, unit="ms"),
            "peak_vram_bytes": metric(float(max(vram)) if vram else None, len(vram), reason="cuda_runtime_unavailable", unit="bytes"),
            "backend_kinds": sorted(backend_kinds), "cache_hit": bool(inference["cache_hit"]),
            "per_role": per_role}
