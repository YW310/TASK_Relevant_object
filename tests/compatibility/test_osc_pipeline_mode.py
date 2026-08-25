from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .capture_pipeline_baseline import PIPELINE_SCRIPT, REPO_ROOT, _find_bash


def test_osc_hdf5_episode_defaults_to_stage1_only(tmp_path: Path) -> None:
    episode = tmp_path / "episode with spaces"
    hdf5_dir = episode / "hdf5"
    hdf5_dir.mkdir(parents=True)
    (hdf5_dir / "trajectory.hdf5").touch()

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": "echo",
            "EPISODE_DIR": episode.as_posix(),
            "SAM_MODEL_DIR": "/tmp/sam",
            "MODEL_PATH": "/tmp/qwen",
            "CAMERAS": "GLOBAL_LEFT,WRIST",
            "CANDIDATE_DRY_RUN": "1",
        }
    )
    completed = subprocess.run(
        [_find_bash(), str(PIPELINE_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert '  Fused:' not in completed.stdout
    assert '  Viz:' not in completed.stdout
    assert "OSC HDF5 detected: running Qwen3 + SAM3 recognition only" in completed.stdout
    assert "--cameras GLOBAL_LEFT,WRIST" in completed.stdout
    assert "Skipping multiview fusion (SKIP_FUSION=1)" in completed.stdout
    assert "Skipping visualization (SKIP_VIZ=1)" in completed.stdout
