# Qwen role spec + SAM3 episode candidates

`qwen_role_sam3_candidate_episode.py` is an episode-level entrypoint for RLBench/RLBench-exported episode folders. It expects the common RLBench saved-image convention (`front_rgb`, `left_shoulder_rgb`, `right_shoulder_rgb`, optionally `wrist_rgb` / `overhead_rgb`) and resolves the task language from standard RLBench variation description files when `--instruction` is not supplied. It first resolves bbox-free semantic roles with Qwen3-VL, then runs SAM3 text prompts on every selected frame and camera to generate segmentation candidates.

## End-to-end pipeline summary (run_full_pipeline.sh)

Use [run_full_pipeline.sh](run_full_pipeline.sh) when you want one command for the full episode workflow.

### Stage overview

1. Stage 1 (Qwen role spec + SAM3 candidates)
- Script: [qwen_role_sam3_candidate_episode.py](qwen_role_sam3_candidate_episode.py)
- Input: RLBench episode directory with RGB folders and task instruction.
- Output: role_spec.json, raw_role_spec_output.json, episode_candidates.json, and per-frame/per-camera candidate artifacts.

2. Stage 2 (multi-view 3D fusion)
- Script: [multiview_candidate_fusion.py](multiview_candidate_fusion.py)
- Input: episode_candidates.json, depth maps, camera intrinsics/extrinsics.
- Output: frame_fused_candidates.json containing fused object ids, point clouds, centroids, 3D boxes, visibility, and per-observation evidence.
- Optional: object_summary.json (enable with SAVE_OBJECT_SUMMARY=1, or auto-enabled when Stage 4 runs).

3. Stage 3 (visual sanity check)
- Script: [visualize_fused_candidates.py](visualize_fused_candidates.py)
- Input: frame_fused_candidates.json.
- Output: reprojection visualization PNGs under outputs/<episode>/viz.

4. Stage 4 (optional object-level target/reference decision)
- Script: [qwen3vl_object_role_decision.py](qwen3vl_object_role_decision.py)
- Input: object_summary.json from Stage 2.
- Output: object_predictions.json with one rolling-window target/reference decision per frame, plus the last decision at top level for compatibility.
- Visual evidence: chronological per-frame object contact sheets under decision_inputs/.
- Each candidate can contribute two distinct camera views; semantic target/reference priors are shown on the sheet and used before the candidate cap.
- Previous model outputs are not fed into later prompts by default, preventing an early wrong target from becoming self-reinforcing. Set USE_DECISION_HISTORY=1 only when that continuity prior is desired.
- Target selection is two-stage: Qwen emits instruction-compatible IDs first, then code chooses within that set by current gripper distance and the t-2 to t approach trend. Both the raw model target and final ranking diagnostics are retained.
- Default is disabled (SKIP_DECISION=1).

5. Stage 5 (optional decision visualization)
- Script: [stage4_visualize_decision.py](stage4_visualize_decision.py)
- Input: object_predictions.json + frame_fused_candidates.json.
- Output: per-frame decision overlays under outputs/<episode>/viz_decision. Selected labels replace the corresponding O-label (for example O2 -> T2) and no filled label background is drawn.

6. Stage 6 (optional compact stage comparison)
- Script: [stage6_visualize_stage_montage.py](stage6_visualize_stage_montage.py)
- Input: Stage 3 reprojection overlays + Stage 5 decision overlays.
- Output: side-by-side per-frame comparison panels under outputs/<episode>/viz_compare.

### Default execution behavior

- By default, Stages 1-3 run and Stage 4 is skipped.
- If SKIP_DECISION=0, Stage 4 runs and Stage 2 automatically exports object_summary.json.
- Stage 4 defaults to DECISION_SCOPE=all; use DECISION_SCOPE=single for the legacy one-frame debug behavior.

### Typical command

```bash
EPISODE_DIR=/path/to/rlbench/.../episode0 \
SAM_MODEL_DIR=/common-data-32t/.cache/facebook/sam3 \
MODEL_PATH=/new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct \
SKIP_DECISION=0 \
SAVE_OBJECT_SUMMARY=1 \
./run_full_pipeline.sh
```

### Key knobs

- Sampling and coverage: FRAME_INTERVAL, MAX_FRAMES, CAMERAS
- SAM3 recall/precision: THRESHOLD, CAMERA_THRESHOLD_OVERRIDES
- Per-view canonicalization: MASK_NMS_IOU, CANONICAL_CONTAINMENT, CANONICAL_BBOX_IOU
- Legacy canonicalization during fusion: LEGACY_CANONICAL_IOU, LEGACY_CANONICAL_CONTAINMENT
- 3D fusion/tracking: CLUSTER_DISTANCE_M, TRACK_MAX_MISSED_FRAMES, TRACK_MAX_SIZE_RATIO, MIN_FUSED_POINTS, MIN_BBOX_DIAGONAL_M, MAX_CENTROID_TO_CLOUD_DISTANCE_M, COMPONENT_VOXEL_SIZE_M, MIN_LARGEST_COMPONENT_RATIO, MAX_SECONDARY_COMPONENT_RATIO
- Decision stage: SKIP_DECISION, DECISION_SCOPE, DECISION_FRAME, DECISION_FRAME_ID, DECISION_WINDOW_FRAMES, MAX_CANDIDATE_IMAGES

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
  --threshold 0.25
```


## Avoiding empty `candidate_grid.png` outputs

If `candidate_grid.png` says `No SAM3 candidates`, the most common cause is
that SAM3 filtered out every text-prompt result. This entrypoint now defaults to
`--threshold 0.25` and tries concise role-name prompts before longer
cue-heavy descriptions, because SAM3 concept prompting is usually more reliable
with short object names such as `red mug` than with full relational sentences.

Useful knobs for small RLBench objects:

```bash
set -euxo pipefail
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode \
  --role-spec-json outputs/<episode>/role_spec.json \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --threshold 0.20 \
  --candidate-pool-size 20 \
  --prompt-variants 5 \
  --top-k-per-role 8 \
  --min-mask-area 4 \
  --split-disconnected-masks \
  --max-mask-components 4 \
  --canonical-max-area-ratio 3.0 \
  --suppress-multi-instance-masks
```

Use `--prompt-variants 1` to only try the shortest role name, or increase it to
include more Qwen-provided visual cues as fallbacks. `candidates.json` also records
`prompt_attempts`, `mask_area_pixels`, and the exact `text_prompt` that produced
each candidate, so you can confirm whether SAM3 returned masks and whether tiny
objects were filtered by area.

Disconnected regions in one SAM3 mask are split before bbox generation by
default. This prevents two separated object instances from becoming one large
min/max bbox. Each resulting candidate records its source-mask component index,
component count, and area ratio. Set `--no-split-disconnected-masks` only for
comparison/debugging.

Containment canonicalization also has an area-ratio guard: a high-score broad
mask cannot absorb a much smaller same-role instance merely because it covers
that instance. After canonicalization, a broad mask that contains at least two
independent same-role candidates is treated as a group prediction and
suppressed. This targets cases where SAM3 proposes an entire button panel in
addition to individual button masks.

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


## SAM3 progress output

SAM3 candidate generation can be slow across many frames and cameras. Progress
logging is enabled by default and prints one line when each camera starts, one
line per role prompt with raw/non-empty mask counts, and one completion line with
per-role saved candidate totals. Disable it with `--no-progress` if you need
quieter logs.

Example progress lines:

```text
SAM3 progress frame 1/10 (000000_0) camera 1/3 (front): start /path/front_rgb/0.png
SAM3 progress frame 1/10 (000000_0) camera 1/3 (front): role=target prompt=1 raw_masks=20 non_empty=6 saved_so_far=0
SAM3 progress frame 1/10 (000000_0) camera 1/3 (front): done total_candidates=8 role_counts={'target': 4, 'reference': 3, 'interaction_part': 1}
```

## Visualization options

Per camera, the script writes:

- `numbered_candidates.png`: source image with mask overlays and candidate IDs.
- `candidate_grid.png`: masked crop grid with candidate IDs and scores. Very small masks are enlarged in this grid so tiny RLBench objects remain visible.

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

## Fuse multiview 2D candidates into 3D objects

`multiview_candidate_fusion.py` reads the SAM3 per-camera candidate masks,
matching per-frame depth images, and camera intrinsics/extrinsics to create
frame-level 3D object candidates. It writes `frame_fused_candidates.json` plus
one compressed `frames/<frame_key>/fused_geometry.npz` archive per frame. The
key is `<six-digit frame_index>_<frame_id>` (for example `000000_0`).
Objects and contributing observations reference their arrays with
`geometry_path`, `points_key`, and `point_count`, while compact geometry fields
such as `centroid_world` and `bbox3d_world` remain in JSON alongside camera,
score, and role evidence. The shared loader also accepts legacy outputs with an
embedded `points_world` array.

Expected depth files are searched in common layouts such as
`<camera>_depth/<frame_id>.npy`, `<camera>_depth/<frame_id>.png`,
`depth/<camera>/<frame_id>.npy`, or `depths/<camera>/<frame_id>.png`.
Camera parameters can be supplied with `--camera-params-json` using either
`intrinsics`/`extrinsics` or `K`/`T_world_camera` keys. Intrinsics may be a 3x3
matrix or `[fx, fy, cx, cy]`; extrinsics must transform camera-frame points into
the world or robot base frame.

```bash
set -euxo pipefail
python multiview_candidate_fusion.py \
  --episode-dir /path/to/episode \
  --candidates-json outputs/<episode>/episode_candidates.json \
  --camera-params-json /path/to/camera_params.json \
  --cluster-distance-m 0.03 \
  --bbox-iou-threshold 0.0
```

Objects are clustered by physical geometry across all role hypotheses. Stable,
role-neutral IDs use one global monotonic sequence: `O1`, `O2`, `O3`, and so on.
Each object reports `role_evidence` separately, with probabilities, score mass,
supporting prompts, cameras, and frames for target, reference, and interaction-part
evidence. The original candidate role and prompt remain available in observation
provenance for backward compatibility.

### Canonical per-view observations

SAM outputs are still generated independently for every role and prompt
variant, but are canonicalized before `candidates.json` is written. Masks are
associated across roles using strong mask IoU (`--mask-nms-iou`), smaller-mask
coverage (`--canonical-containment`), and optionally bbox overlap
(`--canonical-bbox-iou`). Bbox overlap is never sufficient without at least
50% smaller-mask coverage, so adjacent instances are not merged. Ambiguous masks that would
bridge two observations remain separate.

Each output row has one `canonical_observation_id`, one representative mask,
complete `prompt_provenance`, and per-role `role_scores`. Role scores use
**noisy-OR** (`1 - product(1 - score)`) and retain `raw_scores` for auditing.
The `canonicalization.suppressed_candidates` diagnostic records collapsed and
top-k-suppressed inputs. Multi-view fusion defensively canonicalizes legacy
role-prefixed JSON and enforces a single observation per camera in every fused
object; rejected same-camera insertions are recorded in frame diagnostics. The
legacy adapter defaults to 0.35 IoU or 0.50 smaller-mask coverage so shifted
prompt masks for the same object are normally collapsed. Both thresholds remain
configurable with `--legacy-canonical-iou` and
`--legacy-canonical-containment`; inspect the suppressed-candidate diagnostics
when tuning them further to avoid merging adjacent instances.

Current canonical artifacts also receive a strict same-camera 2D+3D NMS pass
before cross-camera assignment. This closes the anchor-camera duplication case:
two masks are treated as duplicates only when mask overlap, world-centroid
distance, and 3D bbox size all agree. Frame diagnostics expose every removal in
`same_camera_nms_suppressed`. Low-support fused clusters are then checked for
visibility in missing cameras using reprojection plus depth; the default
`MIN_FUSED_CAMERA_COUNT=2` drops them only when another camera should have seen
the cloud, while genuinely out-of-view or occluded single-camera objects remain.
