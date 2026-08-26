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
    load_osc_robot_observations,
    materialize_osc_depth_frame,
    osc_camera_params_for_frame,
    osc_camera_pose_to_matrix,
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
            depth = np.full((3, 4, 5), 1000, dtype=np.uint16)
            depth[1, 0, 1] = 2500
            depth[1, 0, 2] = 65535
            observations.create_dataset(f"{camera}_depth", data=depth)
            observations.create_dataset(
                f"{camera}_intrinsic",
                data=np.array(
                    [[100.0, 0.0, 2.0], [0.0, 101.0, 1.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            )
            extrinsics = np.zeros((3, 7), dtype=np.float32)
            extrinsics[:, 3] = 1.0
            extrinsics[1] = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
            observations.create_dataset(f"{camera}_extrinsic", data=extrinsics)
        observations.create_dataset(
            "ee_pos_world",
            data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        )
        observations.create_dataset(
            "ee_quat_world",
            data=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (3, 1)),
        )
        actions = np.zeros((3, 7), dtype=np.float32)
        actions[:, 6] = [-1.0, 1.0, 0.0]
        handle.create_dataset("actions", data=actions)
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


def test_depth_camera_pose_and_robot_state_use_verified_conventions(
    tmp_path: Path,
) -> None:
    episode = _write_episode(tmp_path)

    depth_path = materialize_osc_depth_frame(episode, "WRIST", 1)
    depth = np.load(depth_path)
    assert depth_path.name == "1.meters.npy"
    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(1.0)
    assert depth[0, 1] == pytest.approx(2.5)
    assert np.isnan(depth[0, 2])

    params = osc_camera_params_for_frame(episode, "WRIST", 1)
    assert params is not None
    np.testing.assert_allclose(
        params["intrinsics"],
        [[100.0, 0.0, 2.0], [0.0, 101.0, 1.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(params["extrinsics"][:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        params["extrinsics"][:3, :3],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        atol=1e-6,
    )

    observations = load_osc_robot_observations(episode)
    assert observations[1]["gripper_pose"] == [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]
    assert [item["gripper_open"] for item in observations] == [0.0, 1.0, 0.5]


def test_wxyz_usd_camera_pose_converts_to_opencv_optical_axes() -> None:
    matrix = osc_camera_pose_to_matrix(
        np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        matrix[:3, :3],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        atol=1e-12,
    )


def test_reported_osc_camera_poses_face_their_work_areas_after_conversion() -> None:
    left_wheel = np.array([-0.6780, 0.2260, 1.3625])
    right_wheel = np.array([-0.6779, -0.1390, 1.3625])
    poses_and_targets = {
        "GLOBAL_LEFT": (
            [-1.9012605, 0.5651362, 1.7400469, -0.44614732, -0.18106404, 0.434604, 0.76110953],
            left_wheel,
        ),
        "GLOBAL_RIGHT": (
            [-1.944286, -0.3868556, 1.7633, 0.6472904, 0.2846316, -0.28463158, -0.6472904],
            right_wheel,
        ),
        "LOCAL_LEFT": (
            [-0.6571691, 0.1196582, 1.5177046, 0.96592581, 0.25881905, 0.0, 0.0],
            left_wheel,
        ),
        "LOCAL_RIGHT": (
            [-0.6571691, -0.0345857, 1.5177046, 0.0, 0.0, 0.25881904, 0.96592581],
            right_wheel,
        ),
        "WRIST": (
            [-1.0383601, 0.34647477, 1.4581852, 0.79558516, 0.21701482, -0.564716, 0.03232206],
            left_wheel,
        ),
    }

    for camera, (pose, target) in poses_and_targets.items():
        matrix = osc_camera_pose_to_matrix(np.asarray(pose))
        direction = target - matrix[:3, 3]
        direction /= np.linalg.norm(direction)
        optical_axis_alignment = float(direction @ matrix[:3, 2])
        assert optical_axis_alignment > 0.7, (camera, optical_axis_alignment)


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
