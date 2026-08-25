from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np


if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from qwen_role_sam3_candidate_episode import main


def test_stage1_dry_run_auto_discovers_osc_inputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    episode = tmp_path / "disassemble_left_0"
    hdf5_dir = episode / "hdf5"
    hdf5_dir.mkdir(parents=True)
    with h5py.File(hdf5_dir / "disassemble_left.hdf5", "w") as handle:
        observations = handle.create_group("observations")
        for camera in (
            "GLOBAL_LEFT",
            "GLOBAL_RIGHT",
            "LOCAL_LEFT",
            "LOCAL_RIGHT",
            "WRIST",
        ):
            observations.create_dataset(
                f"{camera}_rgb",
                data=np.zeros((4, 3, 5, 3), dtype=np.uint8),
            )
        handle.create_dataset("task_instruction", data="HDF5 instruction")
    (episode / "metadata.json").write_text(
        json.dumps({"task_instruction": "Metadata instruction"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen_role_sam3_candidate_episode.py",
            "--episode-dir",
            str(episode),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--sam-model-dir",
            str(tmp_path / "sam3"),
            "--frame-interval",
            "2",
            "--max-frames",
            "2",
            "--dry-run",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "Instruction (OSC metadata/HDF5 task_instruction): Metadata instruction" in output
    assert '"selected_frame_ids": [\n    "0",\n    "2"' in output
    for camera in (
        "GLOBAL_LEFT",
        "GLOBAL_RIGHT",
        "LOCAL_LEFT",
        "LOCAL_RIGHT",
        "WRIST",
    ):
        assert camera in output

