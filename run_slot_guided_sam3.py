"""Run one strictly bound current-frame snapshot offline; never a control loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from slot_guided_io import load_frame_snapshot, run_offline_frame
from slot_guided_sam3 import MockBackend, Sam3Backend


def mock_backends(path: str | Path, snapshot) -> dict[str, MockBackend]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"episode_id", "frame_id", "source_frame_index", "cameras"}:
        raise ValueError("invalid mock candidates manifest")
    for key in ("episode_id", "frame_id", "source_frame_index"):
        if value[key] != snapshot.metadata[key]:
            raise ValueError(f"mock candidates {key} mismatch")
    if set(value["cameras"]) != set(snapshot.cameras):
        raise ValueError("mock candidates cameras mismatch")
    backends = {}
    for name, frame in snapshot.cameras.items():
        candidates = {}
        for role in ("target", "reference"):
            row = value["cameras"][name].get(role, {"masks": [], "scores": []})
            masks = []
            for filename in row["masks"]:
                with Image.open(path.parent / filename) as image:
                    mask = np.asarray(image.convert("L")) > 0
                if mask.shape != frame.rgb.shape[:2]:
                    raise ValueError("mock masks must have the original RGB resolution")
                masks.append(mask)
            stack = np.stack(masks) if masks else np.empty((0, *frame.rgb.shape[:2]), dtype=bool)
            candidates[role] = (stack, np.asarray(row["scores"], dtype=float))
        backends[name] = MockBackend(candidates)
    return backends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True, help="single current-frame RGB-D/calibration JSON")
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("box-only", "point+box"), default="point+box")
    parser.add_argument("--backend", choices=("mock", "sam3"), required=True)
    parser.add_argument("--mock-candidates", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    snapshot = load_frame_snapshot(args.frame)
    if args.backend == "mock":
        if args.mock_candidates is None:
            parser.error("--mock-candidates is required for --backend mock")
        backend = mock_backends(args.mock_candidates, snapshot)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --backend sam3")
        backend = Sam3Backend(args.checkpoint, device=args.device)
    report = run_offline_frame(args.frame, args.guidance, backend, args.output_dir,
                               mode=args.mode, use_cache=not args.no_cache)
    print(json.dumps({"artifact_path": report["artifact_path"], "cache_hit": report["cache_hit"],
                      "source_frame_index": report["source_frame_index"],
                      "roles": {name: {role: {"status": row["status"], "selected_index": row["selected_index"],
                                               "failure_reason": row["failure_reason"]}
                                       for role, row in result["roles"].items()}
                                for name, result in report["cameras"].items()}}, indent=2))


if __name__ == "__main__":
    main()
