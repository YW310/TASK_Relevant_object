"""Capture and check the legacy shell invocation/environment contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "run_full_pipeline.sh"
INVOCATION_SNAPSHOT = Path(__file__).with_name("pipeline_invocations.json")
ENVIRONMENT_SNAPSHOT = Path(__file__).with_name("pipeline_environment_contract.json")

SCENARIOS: dict[str, dict[str, str]] = {
    "default": {"EPISODE_DIR": "/tmp/episode0", "SAM_MODEL_DIR": "/tmp/sam"},
    "all_stages": {
        "EPISODE_DIR": "/tmp/episode0",
        "SAM_MODEL_DIR": "/tmp/sam",
        "SKIP_DECISION": "0",
        "SKIP_STAGE_COMPARE": "0",
    },
    "reuse_artifacts": {
        "EPISODE_DIR": "/tmp/episode0",
        "SAM_MODEL_DIR": "/tmp/sam",
        "SKIP_CANDIDATES": "1",
        "SKIP_FUSION": "1",
        "SKIP_VIZ": "1",
        "SKIP_DECISION": "0",
        "SKIP_DECISION_VIZ": "1",
        "SKIP_STAGE_COMPARE": "1",
        "OBJECT_SUMMARY_JSON": "/tmp/existing/summary.json",
    },
    "optional_values": {
        "EPISODE_DIR": "/tmp/episode0",
        "SAM_MODEL_DIR": "/tmp/sam",
        "MODEL_PATH": "/tmp/qwen",
        "INSTRUCTION": "pick_red_block",
        "INSTRUCTION_FILE": "/tmp/instruction.txt",
        "ROLE_SPEC_JSON": "/tmp/role.json",
        "END_FRAME": "20",
        "SAM_CHECKPOINT": "/tmp/sam.pt",
        "CAMERAS": "front,left_shoulder",
        "MAX_FRAMES": "3",
        "CAMERA_THRESHOLD_OVERRIDES": "front=0.2",
        "RLBENCH_LOW_DIM_OBS": "/tmp/low.pkl",
        "NEAREST_DISTANCE_M": "0.01",
        "CAMERA_PARAMS_JSON": "/tmp/cameras.json",
        "INVERT_RLBENCH_EXTRINSICS": "1",
        "SAVE_OBJECT_SUMMARY": "1",
        "OBJECT_SUMMARY_JSON": "/tmp/summary.json",
        "SKIP_DECISION": "0",
        "MAX_EE_DISTANCE_M": "0.15",
        "DECISION_MODEL_PATH": "/tmp/qwen2",
        "DECISION_FRAME_ID": "10",
        "DECISION_OUTPUT_JSON": "/tmp/pred.json",
        "DECISION_ARTIFACTS_DIR": "/tmp/artifacts",
        "USE_DECISION_HISTORY": "1",
        "DYNAMIC_ROLE_REASONING": "0",
        "DECISION_DRY_RUN": "1",
        "VIZ_FRAME_IDS": "0,10",
        "VIZ_MAX_FRAMES": "2",
        "VIZ_SKIP_POINTCLOUD": "1",
        "VIZ_HIDE_SUSPECTED_FRAGMENTS": "1",
        "DECISION_VIZ_OUTPUT_DIR": "/tmp/decision_viz",
        "DECISION_VIZ_HIDE_SUSPECTED_FRAGMENTS": "1",
        "SKIP_STAGE_COMPARE": "0",
        "STAGE1_CANDIDATES_JSON": "/tmp/stage1.json",
        "STAGE_COMPARE_OUTPUT_DIR": "/tmp/compare",
    },
    "space_paths": {
        "EPISODE_DIR": "/tmp/episode space/episode0",
        "SAM_MODEL_DIR": "/tmp/sam model",
        "OUTPUT_DIR": "/tmp/output dir",
        "SKIP_FUSION": "1",
        "SKIP_VIZ": "1",
    },
}

BOOLEAN_VARIABLES = {
    "CANDIDATE_DRY_RUN", "CANDIDATE_PROGRESS", "CANDIDATE_RESUME",
    "COMPILE_MODEL", "DECISION_DRY_RUN",
    "DECISION_VIZ_HIDE_SUSPECTED_FRAGMENTS", "DYNAMIC_ROLE_REASONING",
    "INVERT_RLBENCH_EXTRINSICS", "LEGACY_UNION_FIND",
    "SAVE_FRAME_CONTACT_SHEET", "SAVE_OBJECT_SUMMARY", "SKIP_CANDIDATES",
    "SKIP_DECISION", "SKIP_DECISION_VIZ", "SKIP_FUSION",
    "SKIP_STAGE_COMPARE", "SKIP_VIZ", "SPLIT_DISCONNECTED_MASKS",
    "SUPPRESS_MULTI_INSTANCE_MASKS", "USE_BF16", "USE_DECISION_HISTORY",
    "VIZ_HIDE_SUSPECTED_FRAGMENTS", "VIZ_SKIP_POINTCLOUD",
}
OPTIONAL_NUMERIC_VARIABLES = {
    "END_FRAME", "MAX_EE_DISTANCE_M", "MAX_FRAMES", "NEAREST_DISTANCE_M",
    "VIZ_MAX_FRAMES",
}
STAGE_NAMES = {
    "1": "candidates",
    "2": "fusion",
    "3": "fusion_visualization",
    "4": "decision",
    "5": "decision_visualization",
    "6": "comparison",
}


def _find_bash() -> str:
    candidates = [os.environ.get("PIPELINE_TEST_BASH"), shutil.which("bash")]
    if sys.platform == "win32":
        candidates += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("Bash was not found; set PIPELINE_TEST_BASH.")


def _value_kind(name: str, default: str) -> str:
    if name in BOOLEAN_VARIABLES:
        return "boolean_0_or_1"
    if name in OPTIONAL_NUMERIC_VARIABLES:
        return "optional_number"
    if re.fullmatch(r"-?\d+", default):
        return "integer"
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", default):
        return "number"
    if any(token in name for token in ("DIR", "PATH", "JSON", "CHECKPOINT", "OBS")):
        return "path_or_empty" if default == "" else "path"
    return "string"


def build_environment_contract() -> list[dict[str, Any]]:
    lines = PIPELINE_SCRIPT.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(r'^([A-Z][A-Z0-9_]*)="\$\{\1:-(.*)\}"$')
    declarations: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, 1):
        match = pattern.match(line)
        if not match:
            continue
        name, default = match.groups()
        declarations[name] = {
            "line": line_number,
            "default_expression": default,
            "value_kind": _value_kind(name, default),
            "empty_semantics": (
                "optional value; related CLI flag is omitted when empty"
                if default == "" else
                "derived from another setting when empty"
                if default.startswith("${") else "use declared default"
            ),
            "used_by": [],
            "cli_flags": [],
        }

    flag_pattern = re.compile(r"--[a-z][a-z0-9-]*")
    variable_pattern = re.compile(r"\$\{([A-Z][A-Z0-9_]*)")
    current_stage: str | None = None
    stage_header = re.compile(r"^# Stage ([1-6]):")
    for line_number, line in enumerate(lines, 1):
        header = stage_header.match(line)
        if header:
            current_stage = STAGE_NAMES[header.group(1)]
        variables = set(variable_pattern.findall(line)) & declarations.keys()
        flags = set(flag_pattern.findall(line))
        stages = {current_stage} if current_stage is not None else set()
        for variable in variables:
            declarations[variable]["cli_flags"] = sorted(
                set(declarations[variable]["cli_flags"]) | flags
            )
            declarations[variable]["used_by"] = sorted(
                set(declarations[variable]["used_by"]) | stages
            )
    return [{"name": name, **data} for name, data in declarations.items()]


def _normalize_command(line: str) -> str:
    scripts = (
        "qwen_role_sam3_candidate_episode.py", "multiview_candidate_fusion.py",
        "qwen3vl_object_role_decision.py", "visualize_fused_candidates.py",
        "stage4_visualize_decision.py", "stage6_visualize_stage_montage.py",
    )
    for script_name in scripts:
        index = line.find(script_name)
        if index >= 0:
            return f"<SCRIPT_DIR>/{line[index:]}"
    return line


def capture_invocations(contract: list[dict[str, Any]]) -> dict[str, list[str]]:
    clean_names = {item["name"] for item in contract}
    baseline: dict[str, list[str]] = {}
    for name, overrides in SCENARIOS.items():
        environment = {k: v for k, v in os.environ.items() if k not in clean_names}
        environment.update({"PYTHON": "echo", **overrides})
        completed = subprocess.run(
            [_find_bash(), str(PIPELINE_SCRIPT)], cwd=REPO_ROOT, env=environment,
            text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Scenario {name!r} failed with exit {completed.returncode}:\n"
                f"{completed.stdout}"
            )
        baseline[name] = [
            _normalize_command(line) for line in completed.stdout.splitlines()
            if re.search(r"\.py(?: |$)", line)
        ]
    return baseline


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    contract = build_environment_contract()
    expected = {
        INVOCATION_SNAPSHOT: _json_text(capture_invocations(contract)),
        ENVIRONMENT_SNAPSHOT: _json_text(contract),
    }
    if args.write:
        for path, content in expected.items():
            path.write_text(content, encoding="utf-8")
        return 0
    mismatches = [
        path for path, content in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if mismatches:
        rendered = ", ".join(str(path.relative_to(REPO_ROOT)) for path in mismatches)
        print(f"Compatibility baseline changed: {rendered}", file=sys.stderr)
        print("Review the change, then rerun with --write.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
