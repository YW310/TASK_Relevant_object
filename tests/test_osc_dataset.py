from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

h5py = pytest.importorskip("h5py")

from osc_dataset import (
    available_osc_cameras,
    collect_osc_camera_frames,
    find_osc_hdf5_path,
    is_osc_episode,
    osc_instruction,
)


def _write_episode(root: Path) -> Path:
    episode = root / "disassemble_left_0"
    hdf5_dir = episode / "hdf5"
    hdf5_dir.mkdir(parents=True)
    path = hdf5_dir / "disassemble_left.hdf5"
    with h5py.File(path, "w") as handle:
        observations = handle.create_group("observations")
        for offset, camera in enumerate(("GLOBAL_LEFT", "WRIST")):
            rgb = np.zeros((3, 4, 5, 3), dtype=np.uint8)
            rgb[..., 0] = np.arange(3, dtype=np.uint8)[:, None, None] + offset * 10
            observations.create_dataset(f"{camera}_rgb", data=rgb)
        handle.create_dataset(
            "task_instruction",
            data="Disassemble the left wheel and place it on the workbench",
        )
    return episode


def test_discovers_osc_hdf5_cameras_and_instruction(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path)

    assert is_osc_episode(episode)
    assert find_osc_hdf5_path(episode) == episode / "hdf5" / "disassemble_left.hdf5"
    assert available_osc_cameras(episode) == ("GLOBAL_LEFT", "WRIST")
    assert osc_instruction(episode) == (
        "Disassemble the left wheel and place it on the workbench"
    )


def test_camera_frame_mapping_extracts_only_requested_rgb(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path)
    frame_ids, camera_frames = collect_osc_camera_frames(
        episode, ("GLOBAL_LEFT", "WRIST")
    )

    assert frame_ids == ["0", "1", "2"]
    first_path = camera_frames["WRIST"]["1"]
    assert first_path.is_file()
    assert not first_path.with_name("2.png").exists()
    np.testing.assert_array_equal(
        np.asarray(Image.open(first_path))[..., 0],
        np.full((4, 5), 11, dtype=np.uint8),
    )

    second_path = camera_frames["WRIST"]["2"]
    assert second_path.is_file()


def test_metadata_task_instruction_overrides_hdf5_instruction(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path)
    (episode / 'metadata.json').write_text(
        json.dumps({'task_instruction': 'Instruction from metadata'}),
        encoding='utf-8',
    )

    assert osc_instruction(episode) == 'Instruction from metadata'


def test_missing_requested_camera_fails_clearly(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path)

    with pytest.raises(KeyError, match="observations/LOCAL_LEFT_rgb"):
        collect_osc_camera_frames(episode, ("LOCAL_LEFT",))
