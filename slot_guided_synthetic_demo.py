"""Reproducible software experiment, explicitly not real model evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from camera_geometry import backproject_mask, transform_points
from slot_guidance import save_guidance, identity_world_transform
from slot_guidance_producers import oracle_guidance, coarse_slot_guidance
from slot_guided_io import atomic_json, load_frame_snapshot


def create_demo(directory: str | Path) -> Path:
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("synthetic demo requires a new or empty input directory")
    directory.mkdir(parents=True, exist_ok=True)
    timestamp, source_index = "2026-11-05T10:00:00+08:00", 7
    identity = {"episode_id": "synthetic_two_instances", "frame_id": "sample-2", "source_frame_index": source_index}
    k = np.array([[100., 0, 32.], [0, 100., 24.], [0, 0, 1.]])
    cameras, role_masks, mock_cameras, label_roles = [], {}, {}, {}
    for name, pixel_shift in (("front", 0), ("wrist", 2)):
        pose = np.eye(4)
        pose[0, 3] = pixel_shift / 100
        rgb = np.full((48, 64, 3), 35, dtype=np.uint8)
        masks = {}
        for role, x_start in (("target", 10 - pixel_shift), ("reference", 42 - pixel_shift)):
            mask = np.zeros((48, 64), dtype=bool)
            mask[17:30, x_start:x_start + 12] = True
            rgb[mask] = [205, 90, 90]  # Two visually identical category instances.
            masks[role] = mask
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"{name}.{role}.png")
        role_masks[name] = masks
        Image.fromarray(rgb).save(directory / f"{name}.rgb.png")
        np.save(directory / f"{name}.depth.npy", np.ones((48, 64)))
        cameras.append({"name": name, "rgb_path": f"{name}.rgb.png", "depth_path": f"{name}.depth.npy",
                        "depth_encoding": "meters_npy", "source_frame_index": source_index, "observed_at": timestamp,
                        "calibration_source": "synthetic analytic camera, wrist moved at source index 7",
                        "intrinsics": k.tolist(), "extrinsics": pose.tolist()})
        mock_cameras[name] = {role: {"masks": [f"{name}.{other}.png", f"{name}.{role}.png"], "scores": [0.99, 0.65]}
                              for role, other in (("target", "reference"), ("reference", "target"))}
    snapshot_path = directory / "frame.json"
    atomic_json(snapshot_path, {"schema_version": 1, "artifact_type": "slot_rgbd_frame", **identity,
                                "observed_at": timestamp, "cameras": cameras})
    snapshot = load_frame_snapshot(snapshot_path)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    oracle = oracle_guidance(snapshot, role_masks, instruction="move the left red block next to the right red block",
                              task_name="synthetic_same_class", reference_is_null=False, source_git_sha=sha,
                              created_at=timestamp, role_instance_ids={"target": "left", "reference": "right"})
    save_guidance(directory / "oracle.json", oracle.metadata, target_points_world=oracle.target_points_world,
                   reference_points_world=oracle.reference_points_world)
    # A mocked coarse output, never a BridgeVLA checkpoint prediction.
    sample_count = 64
    coarse_points = np.stack([oracle.points(role)[np.linspace(0, len(oracle.points(role)) - 1, sample_count).astype(int)]
                              for role in ("target", "reference")])
    coarse_points += np.array([0.002, 0., 0.])
    coarse = {"points": coarse_points, "confidence": np.array([0.9, 0.9]),
              "role_token_valid": np.array([True, True]), "valid": np.array([True, True]),
              "reference_null_probability": 0.02, "geometry": None}
    predicted = coarse_slot_guidance(snapshot, coarse, point_valid=np.ones((2, sample_count), dtype=bool),
                                     transform=identity_world_transform(source="mock coarse points in known world frame",
                                                                        timestamp=timestamp, source_frame_index=source_index),
                                     instruction=oracle.metadata["instruction"], task_name="synthetic_same_class",
                                     source_checkpoint="mock://analytic_coarse_fixture", source_git_sha=sha, created_at=timestamp,
                                     point_support_source="synthetic known foreground samples, not real learned slots")
    predicted.metadata["guidance_source"] = "mock"
    for role, instance in (("target", "left"), ("reference", "right")):
        predicted.metadata[role]["instance_id"] = instance
    save_guidance(directory / "predicted_mock.json", predicted.metadata, target_points_world=predicted.target_points_world,
                   reference_points_world=predicted.reference_points_world)
    for role, instance in (("target", "left"), ("reference", "right")):
        label_roles[role] = {"status": "VISIBLE", "instance_id": instance,
                             "masks": {name: f"{name}.{role}.png" for name in role_masks}}
    atomic_json(directory / "labels.json", {**identity, "roles": label_roles})
    atomic_json(directory / "mock_masks.json", {**identity, "cameras": mock_cameras})
    manifest = directory / "experiment.json"
    atomic_json(manifest, {"schema_version": 1, "artifact_type": "slot_geometry_experiment", "evidence_kind": "synthetic_mock",
                           "frames": [{"snapshot_json": "frame.json", "source_frame_index": source_index,
                                       "oracle_guidance_json": "oracle.json", "predicted_guidance_json": "predicted_mock.json",
                                       "mock_candidates_json": "mock_masks.json", "ground_truth_json": "labels.json"}]})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="validated offline threshold JSON")
    args = parser.parse_args()
    from run_slot_geometry_experiments import run_experiments
    from slot_guided_config import load_slot_guided_config
    config = load_slot_guided_config(args.config)
    manifest = create_demo(args.output_dir / "inputs")
    report = run_experiments(manifest, args.output_dir / "results", config=config)
    print(json.dumps({"manifest": str(manifest), "report": str(args.output_dir / "results" / "experiment_report.json"),
                      "evidence_kind": report["evidence_kind"], "real_offline_acceptance_passed": False}, indent=2))


if __name__ == "__main__":
    main()
