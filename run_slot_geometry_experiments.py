"""A/B box-only vs point+box and C baseline report harness, strictly offline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_slot_guided_sam3 import mock_backends
from slot_guidance import load_guidance
from slot_guided_io import atomic_json, load_frame_snapshot, run_offline_frame, _read_json
from slot_guided_sam3 import Sam3Backend
from slot_experiment_metrics import evaluate_frame, load_current_labels, metric
from slot_geometry_fusion import fuse_geometry, save_geometry


def run_experiments(manifest_path, output_dir, *, backend_kind="mock", checkpoint=None, device="cpu"):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = _read_json(manifest_path)
    if set(manifest) != {"schema_version", "artifact_type", "evidence_kind", "frames"} or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "slot_geometry_experiment":
        raise ValueError("invalid experiment manifest")
    if manifest.get("evidence_kind") not in ("synthetic_mock", "real_offline") or not isinstance(manifest.get("frames"), list) or not manifest["frames"]:
        raise ValueError("experiment must declare its evidence kind and nonempty frames")
    if manifest["evidence_kind"] == "real_offline" and backend_kind != "sam3":
        raise ValueError("Mock predictions cannot be reported as real offline evidence")
    backend = Sam3Backend(checkpoint, device) if backend_kind == "sam3" else None
    reports, seen = [], set()
    last_indices = {}
    for entry in manifest["frames"]:
        required = {"snapshot_json", "source_frame_index", "oracle_guidance_json", "predicted_guidance_json", "ground_truth_json"}
        if not isinstance(entry, dict) or not required.issubset(entry) or set(entry) - required - {"mock_candidates_json", "baseline_report_json"}:
            raise ValueError("experiment frames must contain only current-frame artifact references")
        if type(entry["source_frame_index"]) is not int or entry["source_frame_index"] < 0:
            raise ValueError("experiment frame needs an exact integer source index")
        snapshot_path = manifest_path.parent / entry["snapshot_json"]
        snapshot = load_frame_snapshot(snapshot_path, expected_source_frame_index=entry["source_frame_index"])
        identity = (snapshot.metadata["episode_id"], snapshot.metadata["source_frame_index"])
        if identity in seen or identity[1] <= last_indices.get(identity[0], -1):
            raise ValueError("experiment source frames must be unique and ascending within each episode")
        seen.add(identity)
        last_indices[identity[0]] = identity[1]
        row = {key: snapshot.metadata[key] for key in ("episode_id", "frame_id", "source_frame_index")}
        row["configurations"] = {}
        pending = []
        for variant, guidance_key in (("A", "oracle_guidance_json"), ("B", "predicted_guidance_json")):
            guidance_path = manifest_path.parent / entry[guidance_key]
            guidance = load_guidance(guidance_path, expected_source_frame_index=identity[1])
            expected_source = "oracle_gt" if variant == "A" else "predicted_coarse_slot"
            if manifest["evidence_kind"] == "real_offline" and guidance.metadata["guidance_source"] != expected_source:
                raise ValueError(f"real {variant} requires {expected_source} guidance")
            for mode in ("box-only", "point+box"):
                active_backend = backend or mock_backends(manifest_path.parent / entry["mock_candidates_json"], snapshot)
                configuration = f"{variant}_{mode}"
                directory = output_dir / snapshot.metadata["episode_id"] / str(identity[1]) / configuration
                inference = run_offline_frame(snapshot_path, guidance_path, active_backend, directory, mode=mode, use_cache=False)
                geometry = fuse_geometry(snapshot, guidance, inference)
                save_geometry(directory / "object_geometry.json", geometry)
                pending.append((configuration, guidance, inference, geometry))
        # Evaluator labels are loaded only after A/B perception has finished.
        labels = load_current_labels(manifest_path.parent / entry["ground_truth_json"], snapshot)
        for configuration, guidance, inference, geometry in pending:
            row["configurations"][configuration] = evaluate_frame(snapshot, guidance, inference, geometry, labels)
        baseline_path = entry.get("baseline_report_json")
        if baseline_path:
            baseline = _read_json(manifest_path.parent / baseline_path)
            for key in ("episode_id", "frame_id", "source_frame_index"):
                if baseline.get(key) != row[key]:
                    raise ValueError(f"baseline report {key} mismatch")
            if baseline.get("method") != "original_ocrr_action_path" or baseline.get("evidence_kind") != manifest["evidence_kind"]:
                raise ValueError("baseline must identify the original OCRR/action path and matching evidence kind")
            row["configurations"]["C"] = {"status": "provided_baseline_report", "metrics": baseline.get("metrics", {}),
                                              "source_checkpoint": baseline.get("source_checkpoint"),
                                              "source_git_sha": baseline.get("source_git_sha")}
        else:
            row["configurations"]["C"] = {"status": "not_executed", "reason": "original_ocrr_checkpoint_runtime_unavailable"}
        reports.append(row)
    report = {"schema_version": 1, "artifact_type": "slot_geometry_experiment_report",
              "evidence_kind": manifest["evidence_kind"], "backend_kind": backend_kind, "frames": reports,
              "real_offline_acceptance_passed": False,
              "acceptance_reason": "requires_separate_reviewed_real_model_and_paired_rollout_evidence",
              "missing_runtime_metrics": {
                  "action_translation_error_m": metric(None, 0, reason="bridgevla_action_predictions_unavailable", unit="m"),
                  "rlbench_paired_task_success_rate": metric(None, 0, reason="paired_rollouts_not_executed"),
              }}
    atomic_json(output_dir / "experiment_report.json", report)
    # Flat, shareable metrics including explicit missing C/action/rollout values.
    names = ("role_selection_accuracy", "prompt_instance_hit_rate", "sam3_mask_iou", "multiview_surface_distance_m",
             "object_center_error_m", "action_translation_error_m", "rlbench_paired_task_success_rate",
             "per_frame_backend_latency_ms", "peak_vram_bytes")
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_id", "source_frame_index", "configuration", "evidence_kind", "metric", "value", "count", "reason"])
        for frame in reports:
            for configuration, result in frame["configurations"].items():
                measured = result.get("metrics", result)
                for name in names:
                    item = measured.get(name, metric(None, 0, reason=result.get("reason", "baseline_metric_unavailable")))
                    writer.writerow([frame["episode_id"], frame["source_frame_index"], configuration,
                                     manifest["evidence_kind"], name, item.get("value"), item.get("count", 0), item.get("reason")])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("mock", "sam3"), default="mock")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_experiments(args.manifest, args.output_dir, backend_kind=args.backend, checkpoint=args.checkpoint, device=args.device)
    print(json.dumps({"report": str(args.output_dir / "experiment_report.json"), "evidence_kind": report["evidence_kind"],
                      "frames": len(report["frames"]), "real_offline_acceptance_passed": False}, indent=2))


if __name__ == "__main__":
    main()
