# Qwen role spec + SAM3 episode candidates

`qwen_role_sam3_candidate_episode.py` is an episode-level entrypoint for RLBench/RLBench-exported episode folders. It expects the common RLBench saved-image convention (`front_rgb`, `left_shoulder_rgb`, `right_shoulder_rgb`, optionally `wrist_rgb` / `overhead_rgb`) and resolves the task language from standard RLBench variation description files when `--instruction` is not supplied. It first resolves bbox-free semantic roles with Qwen3-VL, then runs SAM3 text prompts on every selected frame and camera to generate segmentation candidates.

## What it produces

By default, outputs are written under:

```text
outputs/<episode>/
├── role_spec.json
├── raw_role_spec_output.json
├── episode_candidates.json
└── frames/
    └── <frame_key>/
        ├── qwen_candidates_contact_sheet.png
        └── <camera>/
            └── qwen_candidates/
                ├── candidates.json
                ├── numbered_candidates.png
                ├── candidate_grid.png
                ├── masks/
                ├── crops/
                └── masked_crops/
```

Candidate IDs use role-specific prefixes:

- `T*`: target candidates
- `R*`: reference candidates
- `P*`: interaction-part candidates

The first-stage `role_spec.json` intentionally contains only:

- `instruction`
- `target`
- `reference`
- `interaction_part`
- `relation`

It does not contain Qwen bounding boxes.

## Install dependencies

Run the project dependency setup before using the script:

```bash
set -euxo pipefail
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You also need a working environment that can import SAM3 and Transformers/Qwen3-VL model classes.

## Required input layout

The episode folder should contain camera RGB directories named `<camera>_rgb`, for example:

```text
/path/to/episode/
├── variation_descriptions.pkl  # or instruction.txt / descriptions.json / etc.
├── front_rgb/
│   ├── 0.png
│   └── 1.png
├── left_shoulder_rgb/
│   ├── 0.png
│   └── 1.png
└── right_shoulder_rgb/
    ├── 0.png
    └── 1.png
```

Camera frame stems must overlap across all selected cameras.

## RLBench-specific notes

This workflow is designed around RLBench episodes exported with RGB folders:

- Default cameras are inherited from the existing grounding script: `front,left_shoulder,right_shoulder`.
- If your RLBench export also has `wrist_rgb` or `overhead_rgb`, include them explicitly with `--cameras front,left_shoulder,right_shoulder,wrist,overhead`.
- Task language can be read from common RLBench files such as `variation_descriptions.pkl`, `variation_description.pkl`, `descriptions.pkl`, `instruction.txt`, or corresponding JSON/TXT variants. You can always override discovery with `--instruction` or `--instruction-file`.
- RLBench frames are often small, so Qwen role identification keeps the existing `--grounding-min-side` upscaling path from `qwen3vl_rlbench_episode_grounding.py`; SAM3 still receives the original RGB frame for mask generation.
- The script assumes synchronized camera folders and selects only frame IDs present in every requested camera.

Example with all common RLBench cameras:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/rlbench/task/variation0/episodes/episode0 \
  --output-root outputs \
  --model-path /new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --sam-checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --cameras front,left_shoulder,right_shoulder,wrist,overhead \
  --frame-interval 5
```

## Recommended dry run

Use `--dry-run` first to validate episode discovery, selected frame IDs, camera names, instruction discovery, and the planned output directory without loading Qwen or SAM3:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --dry-run
```

If you already have a role spec and only want to validate the SAM3-side plan, pass it explicitly:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --role-spec-json outputs/<episode>/role_spec.json \
  --dry-run
```

## Full run

Run Qwen once to create `role_spec.json`, then generate SAM3 candidates for all selected frames/cameras:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --output-root outputs \
  --model-path /new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --sam-checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --device cuda \
  --cameras front,left_shoulder,right_shoulder \
  --frame-interval 1 \
  --top-k-per-role 8 \
  --threshold 0.5
```

## Reuse an existing role spec

To skip Qwen and only run SAM3 candidate generation:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --output-root outputs \
  --role-spec-json outputs/<episode>/role_spec.json \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --sam-checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --device cuda
```

## Process a subset of frames

Use `--start`, `--end`, `--frame-interval`, and `--max-frames` to select frames:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --output-root outputs \
  --role-spec-json outputs/<episode>/role_spec.json \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --sam-checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --start 0 \
  --end 100 \
  --frame-interval 5 \
  --max-frames 10
```

## Visualization options

Per camera, the script writes:

- `numbered_candidates.png`: source image with mask overlays and candidate IDs.
- `candidate_grid.png`: masked crop grid with candidate IDs and scores.

Per frame, the script writes `qwen_candidates_contact_sheet.png` by default, combining all camera `numbered_candidates.png` images.

Disable contact sheets:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --role-spec-json outputs/<episode>/role_spec.json \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --no-save-frame-contact-sheet
```

Change contact-sheet cell width:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --role-spec-json outputs/<episode>/role_spec.json \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --visualization-cell-width 512
```

## Resume

Use `--resume` to reuse per-camera outputs when `candidates.json`, `numbered_candidates.png`, and `candidate_grid.png` already exist:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --output-root outputs \
  --role-spec-json outputs/<episode>/role_spec.json \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --sam-checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --resume
```

## CPU smoke check

For environments without CUDA, validate CLI and dry-run behavior with:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py --help
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --sam-model-dir /path/to/sam3 \
  --role-spec-json /path/to/role_spec.json \
  --dry-run
```
