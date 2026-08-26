from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .capture_pipeline_baseline import PIPELINE_SCRIPT, REPO_ROOT, _find_bash


def test_osc_hdf5_episode_runs_full_rgbd_pipeline(tmp_path: Path) -> None:
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
            "CANDIDATE_ROLES": "target",
            "PROMPT_VARIANTS": "1",
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
    assert "  Fused:" in completed.stdout
    assert "  Viz:" in completed.stdout
    assert "OSC HDF5 detected: running Qwen3 + SAM3 recognition only" not in completed.stdout
    assert "--cameras GLOBAL_LEFT,WRIST" in completed.stdout
    assert "--candidate-roles target" in completed.stdout
    assert "--prompt-variants 1" in completed.stdout
    assert "qwen_role_sam3_candidate_episode.py" in completed.stdout
    assert "multiview_candidate_fusion.py" in completed.stdout
    assert "visualize_fused_candidates.py" in completed.stdout
    assert "Skipping multiview fusion" not in completed.stdout
    assert "Skipping visualization" not in completed.stdout
