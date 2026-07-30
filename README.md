# Relevant Object Grounding for RLBench

This repository provides an episode-level pipeline for finding task-relevant
objects in RLBench demonstrations. It combines Qwen3-VL semantic role parsing,
SAM3 instance proposals, multi-view 3D fusion, and optional object-level role
selection and visualizations.

The pipeline keeps perception and task reasoning separate: SAM3 proposes
instances for each camera view, fusion assigns role-neutral object IDs, and
Qwen3-VL can subsequently select the current target and reference objects.

## Pipeline

| Stage | Script | Result |
| --- | --- | --- |
| 1. Role parsing and proposals | `qwen_role_sam3_candidate_episode.py` | `role_spec.json`, per-view masks/crops, and `episode_candidates.json` |
| 2. Multi-view fusion | `multiview_candidate_fusion.py` | Lightweight `frame_fused_candidates.json`, per-frame `fused_objects.json` / `fused_geometry.npz`, and optional `object_summary.json` |
| 3. Fusion visualization | `visualize_fused_candidates.py` | Per-camera reprojection overlays in `viz/` |
| 4. Object role decision (optional) | `qwen3vl_object_role_decision.py` | `object_predictions.json` |
| 5. Decision visualization (optional) | `stage4_visualize_decision.py` | Target/reference overlays in `viz_decision/` |
| 6. Stage comparison (optional) | `stage6_visualize_stage_montage.py` | Side-by-side panels in `viz_compare/` |

Stages 1–3 run by default. Stage 4 and its visualizations are opt-in.

## Stage details and parameters

The shell pipeline exposes the most common settings as environment variables.
Each stage can also be run independently with the Python arguments below.

### Stage 1: semantic roles and per-view SAM3 candidates

`qwen_role_sam3_candidate_episode.py` first asks Qwen3-VL to turn the task
instruction into a bbox-free role specification (`target`, `reference`,
`interaction_part`, and `relation`). It then prompts SAM3 for each role in each
selected camera/frame, filters and deduplicates the masks, and saves stable
role-prefixed IDs (`T*`, `R*`, and `P*`). Qwen runs once for the role frame;
SAM3 runs across the selected episode frames.

Important input and sampling arguments:

| Python argument | Pipeline variable | Default | Purpose |
| --- | --- | --- | --- |
| `--episode-dir` | `EPISODE_DIR` | required | Episode containing `<camera>_rgb` folders. |
| `--model-path` | `MODEL_PATH` | script model path | Local Qwen3-VL checkpoint. |
| `--sam-model-dir` | `SAM_MODEL_DIR` | required | SAM3 model/config directory. |
| `--sam-checkpoint` | — | `<sam-model-dir>/sam3.pt` | Explicit SAM3 weights override. |
| `--instruction` | `INSTRUCTION` | auto-discovered | Override the episode instruction. |
| `--instruction-file`, `--instruction-index` | — | unset, `0` | Read a selected instruction from a file. |
| `--role-spec-json` | `ROLE_SPEC_JSON` | unset | Reuse role semantics and avoid rerunning Qwen. |
| `--cameras` | `CAMERAS` | front and shoulder cameras | Comma-separated camera folder prefixes. |
| `--start`, `--end` | — | `0`, unset | Restrict the source-frame index range. |
| `--frame-interval` | `FRAME_INTERVAL` | `1` | Keep one source frame every N frames. |
| `--max-frames` | `MAX_FRAMES` | unset | Cap selected frames after sampling. |
| `--role-frame` | — | `first` | Frame used to infer the semantic role specification. |

Candidate generation and runtime arguments:

| Python argument | Pipeline variable | Default | Purpose |
| --- | --- | --- | --- |
| `--threshold` | `THRESHOLD` | `0.25` | Global SAM3 confidence cutoff. Lower it to improve recall for small objects. |
| `--camera-threshold-overrides` | `CAMERA_THRESHOLD_OVERRIDES` | unset | Per-camera cutoffs such as `left_shoulder=0.15`. |
| `--candidate-pool-size` | — | `20` | Raw masks inspected per prompt before final selection. |
| `--top-k-per-role` | — | `8` | Maximum candidates retained for each semantic role. |
| `--prompt-variants` | — | `5` | Maximum Qwen/SAM text variants attempted per role. |
| `--min-mask-area` | — | `4` px | Remove tiny mask fragments. |
| `--mask-nms-iou` | — | `0.80` | Suppress same-role masks whose IoU exceeds this value. |
| `--device` | `DEVICE` | `cuda` | Run models on `cuda` or `cpu`. |
| `--no-bf16` | inverse of `USE_BF16` | enabled by pipeline | Disable BF16 autocast for compatibility/debugging. |
| `--compile` | — | off | Enable PyTorch model compilation. |
| `--grounding-min-side` | — | `512` | Upscale the Qwen role image so its shorter side reaches this size. |
| `--max-retries` | — | `1` | Retry malformed Qwen structured output. |
| `--resume` | always passed | off in direct CLI | Reuse completed per-camera outputs. |
| `--dry-run` | — | off | Validate discovery and print the plan without loading models. |

Visualization controls include `--[no-]save-frame-contact-sheet` (enabled),
`--visualization-cell-width` (`384` pixels), `--mask-alpha` (`105`), and
`--[no-]progress` (enabled). The main manifest is
`episode_candidates.json`; individual masks, crops, masked crops, candidate
grids, and numbered overlays are stored below `frames/<frame>/<camera>/`.

### Stage 2: depth lifting, multi-view fusion, and tracking

`multiview_candidate_fusion.py` intersects every candidate mask with its depth
image, uses camera intrinsics/extrinsics to lift pixels into world-space point
clouds, then performs deterministic anchor-camera assignment and tracks the
resulting role-neutral IDs (`O1`, `O2`, …) across frames. The anchor camera is
the camera containing the highest-confidence observation (camera name breaks
ties); remaining cameras use the same confidence-first, name-tiebroken order.

| Python argument | Pipeline variable | Default | Purpose |
| --- | --- | --- | --- |
| `--candidates-json` | derived from `OUTPUT_DIR` | required | Stage 1 manifest. |
| `--output-json` | derived from `OUTPUT_DIR` | beside input | Fused frame/object output. |
| `--cameras` | `CAMERAS` | manifest cameras | Restrict cameras participating in fusion. |
| `--camera-params-json` | `CAMERA_PARAMS_JSON` | auto-discovered | Override camera intrinsics/extrinsics. |
| `--rlbench-low-dim-obs` | — | episode `low_dim_obs.pkl` | RLBench camera metadata and depth near/far planes. |
| `--invert-rlbench-extrinsics` | `INVERT_RLBENCH_EXTRINSICS` | off | Invert extrinsics before camera-to-world transformation. |
| `--depth-mode` | — | `auto` | Decode `auto`, packed `rlbench-rgb`, or single-channel `raw` depth. |
| `--depth-scale` | — | `1.0` | Divisor for raw/single-channel depth values. |
| `--max-points-per-candidate` | — | `4096` | Bound point-cloud size and memory use per observation. |
| `--cluster-distance-m` | `CLUSTER_DISTANCE_M` | `0.03` m | Maximum pairwise centroid distance in a completed hypothesis. |
| `--bbox-iou-threshold` | — | `0.0` (off) | Optional minimum pairwise 3D bbox IoU during full validation. |
| `--nearest-distance-m` | — | unset | Optional maximum robust symmetric surface distance during validation. |
| `--max-hypothesis-diameter-m` | — | `0.50` m | Maximum robust pooled point-cloud diameter. |
| `--max-size-ratio` | — | `4.0` | Maximum axis-wise box-size ratio. |
| `--legacy-union-find` | — | off | Deprecated one-release debug path for old transitive pairwise clustering. |
| `--track-distance-m` | — | `0.15` m | Maximum inter-frame centroid movement for retaining an object ID. |
| `--min-fused-points` | `MIN_FUSED_POINTS` | `0` (off) | Drop fused objects with too few total points. |
| `--min-bbox-diagonal-m` | `MIN_BBOX_DIAGONAL_M` | `0.0` (off) | Drop spatially tiny 3D boxes. |
| `--save-object-summary` | `SAVE_OBJECT_SUMMARY` | off | Export trajectories and decision-ready object evidence. |
| `--object-summary-json` | `OBJECT_SUMMARY_JSON` | `object_summary.json` | Override the summary path. |

Pairwise centroid, bbox-IoU, and nearest-point tests remain alternative cheap
compatibility gates only. They never define the final transitive partition.
Every insertion is jointly validated against the complete hypothesis for
centroid spread, robust diameter, box size, and enabled IoU/surface limits. A
per-camera Hungarian assignment includes explicit dummy columns, so an
incompatible observation creates a new hypothesis instead of being forced into
one; consequently every hypothesis has at most one observation per camera.
Appearance embeddings are intentionally not part of this geometry-only MVP.
`object_summary.json` is generated
automatically when Stage 4 is enabled.

#### Three-layer Stage 2 output contract

Stage 2 deliberately separates episode indexing, frame evidence, and temporal
reasoning. The schemas live in `schemas/` and all three JSON layers carry the
same integer `schema_version` (currently `3`) and UUID `generation_id`:

1. **`frame_fused_candidates.json`** is a lightweight episode manifest. Its
   frame entries contain status, object count, and a relative
   `fused_objects_json` reference; it never stores objects or point clouds.
2. **`frames/<frame_key>/fused_objects.json`** owns the current frame's
   `objects`, their `observations`, and fusion `diagnostics`. Point arrays are
   still external in the sibling `fused_geometry.npz` and referenced by key.
   The key is `<six-digit frame_index>_<frame_id>` (for example `000000_0`).
3. **`object_summary.json`** contains cross-frame tracks, aggregate statistics,
   decision-ready scalar metadata, and `frame_ref` links back to the per-frame
   files. It does not duplicate complete point clouds or other full geometry.

The summary builder consumes the referenced frame JSON files one at a time, so
all frame artifacts need not coexist in memory. Readers validate both identity
fields before joining artifacts; a schema mismatch or a `generation_id` from a
different fusion run is an error rather than a silent mixed-run result. A
compatible resumed run retains its generation ID, while a new run creates one.
See `schemas/frame_fused_candidates.schema.json`,
`schemas/fused_objects.schema.json`, and `schemas/object_summary.schema.json`.

### Stage 3: fused-object sanity visualization

`visualize_fused_candidates.py` projects fused world points back into each
camera image and creates colored 2D overlays, an object report, and a 3D
point-cloud view. This stage does not change fusion results; use it to diagnose
depth decoding, camera transforms, bad masks, or incorrect clustering.

| Python argument | Pipeline variable | Default | Purpose |
| --- | --- | --- | --- |
| `--fused-json` | derived from `OUTPUT_DIR` | required | Stage 2 output. |
| `--output-dir` | derived | sibling `viz/` | Visualization destination. |
| `--frame-ids`, `--max-frames` | — | all | Render a frame subset or cap. |
| `--cameras` | `CAMERAS` | all available | Select overlay cameras. |
| `--camera-params-json` | `CAMERA_PARAMS_JSON` | auto | Use the same camera override as fusion. |
| `--invert-rlbench-extrinsics` | `INVERT_RLBENCH_EXTRINSICS` | off | Keep projection convention aligned with Stage 2. |
| `--point-stride` | — | `4` | Render every Nth world point. |
| `--point-radius` | — | `2` px | Reprojected marker radius. |
| `--mask-alpha` | — | `80` | Overlay opacity in the range 0–255. |
| `--skip-pointcloud` | — | off | Omit the matplotlib 3D scatter plot. |

Set `SKIP_VIZ=1` in the shell pipeline to omit this stage.

### Stage 4: Qwen3-VL object-level role decision (optional)

`qwen3vl_object_role_decision.py` consumes Stage 2's object summary rather than
raw SAM role IDs. It filters fused objects, constructs a temporal evidence
window with representative crops and geometry, and asks Qwen3-VL to select the
current target/reference object IDs with confidence, evidence, and uncertainty.
The single window-based result is retained for every episode frame in
`frame_decisions`, so downstream visualization still saves every frame.

| Python argument | Pipeline variable | Default | Purpose |
| --- | --- | --- | --- |
| `--object-summary-json` | `OBJECT_SUMMARY_JSON` or derived | required | Decision-ready Stage 2 summary. |
| `--output-json` | `DECISION_OUTPUT_JSON` | `object_predictions.json` | Selection result path. |
| `--model-path` | `DECISION_MODEL_PATH` / `MODEL_PATH` | script model path | Qwen3-VL checkpoint for selection. |
| `--decision-frame` | `DECISION_FRAME` | `last` | Decide at the `first` or `last` available frame. |
| `--decision-frame-id` | `DECISION_FRAME_ID` | unset | Explicit frame, overriding `--decision-frame`. |
| `--decision-window-frames` | `DECISION_WINDOW_FRAMES` | `3` | Current frame `t` plus its two preceding frames `[t-2, t-1, t]` (when available), evaluated in one model call; `1` is single-frame mode. |
| `--max-candidate-images` | `MAX_CANDIDATE_IMAGES` | `8` | Maximum representative images attached to the prompt. |
| `--max-candidates-for-decision` | `MAX_CANDIDATES_FOR_DECISION` | `12` | Candidate cap after filtering. |
| `--min-candidate-point-count` | `MIN_CANDIDATE_POINT_COUNT` | `0` (off) | Remove sparse fused objects. |
| `--min-candidate-camera-count` | `MIN_CANDIDATE_CAMERA_COUNT` | `1` | Require support from this many cameras. |
| `--min-candidate-sam-score` | `MIN_CANDIDATE_SAM_SCORE` | `0.0` (off) | Remove candidates below a fused SAM score. |
| `--max-ee-distance-m` | `MAX_EE_DISTANCE_M` | unset | Remove objects never close enough to the end effector in the window. |
| `--grounding-min-side` | — | `512` | Minimum short side for Qwen input images. |
| `--max-new-tokens` | `DECISION_MAX_NEW_TOKENS` | `1024` | Structured response generation budget. |
| `--max-retries` | — | `1` | Retry malformed model output. |
| `--dry-run` | — | off | Save the exact decision payload without running Qwen. |

Enable this stage with `SKIP_DECISION=0`. Doing so also enables object-summary
generation in Stage 2. Filters apply before the candidate cap and are useful
for excluding fragments, but aggressive thresholds can discard the true task
object.

### Stage 5: decision overlays (optional)

`stage4_visualize_decision.py` joins `object_predictions.json` with
`frame_fused_candidates.json`, highlights selected target/reference points on
the camera views, and writes `decision_visualization.json` plus images under
`viz_decision/`. Labels use role-specific IDs (`T1`, `T2`, … for targets and
`R1`, `R2`, … for references) rather than internal fused IDs (`O1`, `O2`, …),
and their boxes, backgrounds, text, and centroid marks are translucently
composited so scene content remains visible.

Its required inputs are `--object-predictions-json` and `--fused-json`.
`--episode-dir`, `--output-dir`, `--viz-dir`, `--cameras`,
`--camera-params-json`, and `--rlbench-low-dim-obs` override discovered paths
or camera selection. Rendering is controlled by `--point-stride` (default
`4`), `--point-radius` (`2` pixels), and `--mask-alpha` (`90`); use
`--invert-rlbench-extrinsics` when Stage 2 used the same transform option.
The pipeline exposes `DECISION_VIZ_OUTPUT_DIR` and
`SKIP_DECISION_VIZ` for this stage.

Stage 2 stores point clouds outside the JSON in compressed per-frame archives
at `frames/<frame_key>/fused_geometry.npz`, using a key such as `000000_0`
(`<six-digit frame_index>_<frame_id>`). Object and observation records
contain `geometry_path`, `points_key`, and `point_count`; the visualization
commands load those arrays lazily and remain compatible with older JSON files
that embed `points_world` directly.

### Stage 6: compact comparison montage (optional)

`stage6_visualize_stage_montage.py` combines Stage 1 candidate context, Stage 3
fusion overlays, and Stage 5 decision overlays into compact panels for rapid
visual comparison. It requires Stage 5's `--decision-meta-json`; optionally
set `--stage1-candidates-json` and `--output-dir`. Layout controls are
`--panel-gap` (`8` pixels), `--label-height` (`26`), `--summary-width` (`360`),
and `--background` (`white`). Enable it with `SKIP_STAGE_COMPARE=0` and set
`STAGE_COMPARE_OUTPUT_DIR` to override `viz_compare/`.

## Requirements

- Python 3.10 or newer
- A CUDA-capable environment is recommended
- A local SAM3 model directory and checkpoint
- A local Qwen3-VL checkpoint for role parsing and optional object selection
- RLBench episodes exported with RGB, depth, and camera metadata

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The environment must also be able to import the SAM3 and Qwen3-VL model
implementations used by the scripts. Model weights are not included in this
repository and are not downloaded automatically.

## Input layout

An episode should use the standard RLBench camera-folder convention:

```text
/path/to/episode0/
├── variation_descriptions.pkl
├── front_rgb/
│   ├── 0.png
│   └── 1.png
├── front_depth/
├── left_shoulder_rgb/
├── left_shoulder_depth/
├── right_shoulder_rgb/
├── right_shoulder_depth/
└── low_dim_obs.pkl
```

Frame stems must overlap across the selected cameras. The instruction may be
discovered from common RLBench description files or supplied explicitly with
the `INSTRUCTION` environment variable.

## Quick start

First validate episode discovery without loading either model:

```bash
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode0 \
  --sam-model-dir /path/to/sam3 \
  --dry-run
```

Then run the default candidate, fusion, and visualization stages:

```bash
EPISODE_DIR=/path/to/episode0 \
SAM_MODEL_DIR=/path/to/sam3 \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
./run_full_pipeline.sh
```

To include the object-level decision and its overlay:

```bash
EPISODE_DIR=/path/to/episode0 \
SAM_MODEL_DIR=/path/to/sam3 \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
SKIP_DECISION=0 \
./run_full_pipeline.sh
```

Outputs are written to `outputs/<episode-name>/` unless `OUTPUT_ROOT` or
`OUTPUT_DIR` is set.

## Common configuration

`run_full_pipeline.sh` is configured with environment variables. Frequently
used options include:

| Variable | Default | Description |
| --- | --- | --- |
| `CAMERAS` | script default | Comma-separated cameras such as `front,left_shoulder,right_shoulder` |
| `FRAME_INTERVAL` | `1` | Process every Nth source frame |
| `MAX_FRAMES` | unset | Limit the number of processed frames |
| `DEVICE` | `cuda` | Model device |
| `THRESHOLD` | `0.25` | SAM3 proposal threshold |
| `MASK_NMS_IOU` | `0.80` | Stage-1 same-role NMS and cross-role canonical mask IoU |
| `CANONICAL_CONTAINMENT` | `0.90` | Stage-1 smaller-mask coverage for canonical observations |
| `CANONICAL_BBOX_IOU` | `0.0` | Optional Stage-1 bbox IoU support; `0` disables it |
| `CLUSTER_DISTANCE_M` | `0.03` | Maximum centroid distance for 3D fusion |
| `LEGACY_CANONICAL_IOU` | `0.35` | Stage-2 mask IoU for canonicalizing legacy candidate JSON |
| `LEGACY_CANONICAL_CONTAINMENT` | `0.50` | Stage-2 smaller-mask coverage for legacy candidate JSON |
| `SKIP_CANDIDATES` | `0` | Reuse an existing `episode_candidates.json` |
| `SKIP_FUSION` | `0` | Reuse an existing `frame_fused_candidates.json` |
| `SKIP_VIZ` | `0` | Disable fusion visualizations |
| `SKIP_DECISION` | `1` | Set to `0` to run object-level role selection |
| `SKIP_DECISION_VIZ` | `0` | Disable decision overlays when selection runs |
| `SKIP_STAGE_COMPARE` | `1` | Set to `0` to create comparison montages |

See the documented variables at the top of `run_full_pipeline.sh` for the full
set of filtering, camera, model, and output options.

Every tunable option of the six downstream stage CLIs has an environment-variable
counterpart in `run_full_pipeline.sh`, including less-common model, sampling,
canonicalization, depth, geometry, rendering, and montage controls. Paths that
the pipeline normally derives can also be overridden with `CANDIDATES_JSON`,
`FUSED_JSON`, `VIZ_DIR`, `OBJECT_SUMMARY_JSON`, `DECISION_OUTPUT_JSON`, and the
stage-specific output variables documented in the script header.

## Reusing intermediate results

Expensive stages can be skipped while iterating on downstream processing:

```bash
EPISODE_DIR=/path/to/episode0 \
SAM_MODEL_DIR=/path/to/sam3 \
OUTPUT_DIR=outputs/episode0 \
SKIP_CANDIDATES=1 \
SKIP_FUSION=1 \
./run_full_pipeline.sh
```

Set `ROLE_SPEC_JSON` to reuse an existing semantic role specification while
regenerating candidates. Set `SAVE_OBJECT_SUMMARY=1` to export the input used
by the optional object decision stage; this is enabled automatically when
`SKIP_DECISION=0`.

## Additional documentation

- [`README_qwen_role_sam3_candidates.md`](README_qwen_role_sam3_candidates.md)
  contains detailed stage-one, tuning, and troubleshooting guidance.
- [`README_CN.md`](README_CN.md) contains Chinese notes for local SAM3
  single-image testing and candidate export.

## Utility entry points

- `run_quick_test.sh` runs a local SAM3 smoke test.
- `run_qwen_s1.sh` launches the first-stage Qwen/SAM workflow.
- `demo_sam3.py` demonstrates direct SAM3 usage.
- `qwen3_bbox_guided_sam3_demo.py` demonstrates Qwen-guided box prompting.

Run any Python entry point with `--help` to inspect its complete command-line
interface.
