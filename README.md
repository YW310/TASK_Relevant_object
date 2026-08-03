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
| 1. Role parsing and proposals | `qwen_role_sam3_candidate_episode.py` | `role_spec.json`, compact per-view `candidates.json`, separate `candidate_debug.json`, and `episode_candidates.json` |
| 2. Multi-view fusion | `multiview_candidate_fusion.py` | Lightweight manifest, per-frame `fused_objects.json` / `fused_geometry.npz`, separate `fusion_debug.json`, and optional `object_summary.json` |
| 3. Fusion visualization | `visualize_fused_candidates.py` | One all-camera + point-cloud montage per frame plus compact report in `viz/` |
| 4. Object role decision (optional) | `qwen3vl_object_role_decision.py` | `object_predictions.json` |
| 5. Decision visualization (optional) | `stage4_visualize_decision.py` | Target/reference overlays in `viz_decision/` |
| 6. Stage comparison (optional) | `stage6_visualize_stage_montage.py` | Side-by-side panels in `viz_compare/` |

Stages 1–3 run by default. Stage 4 and its visualizations are opt-in.

## Code organization

The command-line scripts remain stable entrypoints, while reusable logic lives
in small dependency-oriented modules:

| Module | Responsibility |
| --- | --- |
| `common_io.py` | Atomic JSON output, CSV parsing, and natural filename sorting. |
| `sam3_runtime.py` | SAM3 checkpoint discovery, autocast, and tensor normalization. |
| `mask_geometry.py` | Binary-mask, component, and 2D bbox geometry. |
| `camera_geometry.py` | RLBench depth decoding, camera metadata, backprojection, and reprojection. |
| `fusion_types.py` | Shared fusion data structures and role constants. |
| `fusion_matching.py` | Same-camera NMS, cross-camera compatibility, and Hungarian assignment. |
| `fused_candidate_io.py` | Versioned fusion artifact readers and point-cloud loading. |
| `visualization_utils.py` | Annotation primitives shared by generation and visualization stages. |

For backward compatibility, the main fusion entrypoint still re-exports its
previously public geometry and matching helpers. New library-style callers
should import the focused modules directly.

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
| `--[no-]split-disconnected-masks` | `SPLIT_DISCONNECTED_MASKS` | enabled | Split disconnected regions from one SAM3 mask before computing candidate bboxes. |
| `--max-mask-components` | `MAX_MASK_COMPONENTS` | `4` | Maximum significant regions retained from one SAM3 mask (`0` keeps all). |
| `--mask-nms-iou` | — | `0.80` | Suppress same-role masks whose IoU exceeds this value. |
| `--canonical-max-area-ratio` | `CANONICAL_MAX_AREA_RATIO` | `3.0` | Prevent containment-based merging when two masks have very different areas. |
| `--[no-]suppress-multi-instance-masks` | `SUPPRESS_MULTI_INSTANCE_MASKS` | enabled | Drop a broad same-role mask that contains at least two independent candidates. |
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
Each camera/frame `candidates.json` contains only the image, counts, candidates,
and a `diagnostics_ref`. Prompt attempts, generation parameters, and suppressed
canonicalization details live in the sibling `candidate_debug.json`; resume mode
uses that debug file when checking whether cached parameters still match.

### Stage 2: depth lifting, multi-view fusion, and tracking

`multiview_candidate_fusion.py` intersects every candidate mask with its depth
image, uses camera intrinsics/extrinsics to lift pixels into world-space point
clouds, then performs deterministic anchor-camera assignment and tracks the
resulting role-neutral IDs (`O1`, `O2`, …) across frames. Cameras are ordered by
confidence multiplied by their configured reliability weight, so the default
`front` weight of `1.5` normally makes it the anchor while still allowing a much
stronger shoulder-camera observation to lead.

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
| `--min-candidate-mask-area-pixels` | `MIN_CANDIDATE_MASK_AREA_PIXELS` | follows `MIN_MASK_AREA` (`4`) | Drop cached/raw candidates whose `mask_area_pixels` is below this value before backprojection; `0` disables it. |
| `--cluster-distance-m` | `CLUSTER_DISTANCE_M` | `0.03` m | Required maximum pairwise centroid distance in a completed hypothesis. |
| `--bbox-iou-threshold` | — | `0.0` (off) | Optional secondary 3D bbox IoU requirement; it cannot bypass the centroid gate. |
| `--nearest-distance-m` | — | unset | Optional robust symmetric surface-distance requirement; exceeding it vetoes fusion even when bbox IoU passes. |
| `--max-hypothesis-diameter-m` | — | `0.50` m | Maximum robust pooled point-cloud diameter. |
| `--max-size-ratio` | — | `4.0` | Maximum axis-wise box-size ratio. |
| `--same-camera-nms-mask-iou` | `SAME_CAMERA_NMS_MASK_IOU` | `0.55` | Same-view mask IoU cue for strict 2D+3D duplicate suppression. |
| `--same-camera-nms-containment` | `SAME_CAMERA_NMS_CONTAINMENT` | `0.85` | Alternative smaller-mask coverage cue for same-view NMS. |
| `--same-camera-nms-centroid-distance-m` | `SAME_CAMERA_NMS_CENTROID_DISTANCE_M` | `0.02` m | Required 3D centroid proximity for same-view NMS; `<=0` disables it. |
| `--same-camera-nms-max-size-ratio` | `SAME_CAMERA_NMS_MAX_SIZE_RATIO` | `2.5` | Required 3D bbox size consistency for same-view NMS. |
| `--min-fused-camera-count` | `MIN_FUSED_CAMERA_COUNT` | `1` | Keep valid single-camera objects by default; values above `1` enable strict multi-view support filtering. |
| `--preferred-camera` | `PREFERRED_CAMERA` | `front` | Camera trusted more for hypothesis seeding, fused centroid, score, and primary-view ordering. |
| `--preferred-camera-weight` | `PREFERRED_CAMERA_WEIGHT` | `1.5` | Reliability multiplier for the preferred camera; `1.0` makes all cameras equal. |
| `--camera-visibility-depth-tolerance-m` | `CAMERA_VISIBILITY_DEPTH_TOLERANCE_M` | `0.03` m | Depth tolerance for missing-view visibility/occlusion checks. |
| `--camera-visibility-min-point-fraction` | `CAMERA_VISIBILITY_MIN_POINT_FRACTION` | `0.05` | Minimum projected cloud fraction required to call a missing view observable. |
| `--single-camera-keep-score` | `SINGLE_CAMERA_KEEP_SCORE` | `0.0` (off) | Optional confidence exception to the camera-support filter. |
| `--legacy-union-find` | — | off | Deprecated one-release debug path for old transitive pairwise clustering. |
| `--track-distance-m` | — | `0.15` m | Maximum inter-frame centroid movement for retaining an object ID. |
| `--track-max-missed-frames` | `TRACK_MAX_MISSED_FRAMES` | `2` | Preserve dormant tracks through short processed-frame occlusions. |
| `--track-max-size-ratio` | `TRACK_MAX_SIZE_RATIO` | `2.5` | Reject implausible temporal matches whose 3D bbox size changes too much. |
| `--min-fused-points` | `MIN_FUSED_POINTS` | `0` (off) | Drop fused objects with too few total points. |
| `--min-bbox-diagonal-m` | `MIN_BBOX_DIAGONAL_M` | `0.0` (off) | Drop spatially tiny 3D boxes. |
| `--max-centroid-to-cloud-distance-m` | `MAX_CENTROID_TO_CLOUD_DISTANCE_M` | `0.02` m | Drop a fused object whose centroid is farther than this from every cloud point, indicating a split or contaminated cloud; `<=0` disables it. |
| `--component-voxel-size-m` | `COMPONENT_VOXEL_SIZE_M` | `0.008` m | Voxel size for dependency-free 3D connected-component detection; `<=0` disables it. |
| `--min-largest-component-ratio` | `MIN_LARGEST_COMPONENT_RATIO` | `0.75` | Minimum dominant-component point fraction expected from a coherent candidate. |
| `--max-secondary-component-ratio` | `MAX_SECONDARY_COMPONENT_RATIO` | `0.20` | Reject when a separated second component exceeds this point fraction; `<=0` disables it. |
| `--min-component-centroid-gap-m` | `MIN_COMPONENT_CENTROID_GAP_M` | `0.02` m | Require this separation between the two main components before rejection. |
| `--min-component-points` | `MIN_COMPONENT_POINTS` | `20` | Ignore smaller disconnected point groups as noise. |
| `--save-object-summary` | `SAVE_OBJECT_SUMMARY` | off | Export trajectories and decision-ready object evidence. |
| `--object-summary-json` | `OBJECT_SUMMARY_JSON` | `object_summary.json` | Override the summary path. |

Before anchor assignment, each camera independently runs strict duplicate NMS:
a lower-confidence mask is suppressed only when mask IoU/containment, 3D
centroid distance, and bbox size consistency all agree. Suppressed semantic
role evidence is retained on the winning observation and the frame diagnostic
records every decision. This prevents multiple masks for one physical object
in the anchor camera from seeding multiple `O*` hypotheses.

Pairwise centroid, bbox-IoU, and nearest-point tests remain alternative cheap
compatibility gates only. They never define the final transitive partition.
Every insertion is jointly validated against the complete hypothesis for
centroid spread, robust diameter, box size, and enabled IoU/surface limits. A
per-camera Hungarian assignment includes explicit dummy columns, so an
incompatible observation creates a new hypothesis instead of being forced into
one; consequently every hypothesis has at most one observation per camera.
By default, a valid observation from any single camera is retained. Setting
`--min-fused-camera-count` above `1` restores the optional strict support filter,
which reprojects low-support clouds into missing views and checks scene depth.
The preferred camera (default `front`) receives a modest reliability weight and
therefore tends to seed ambiguous hypotheses and dominate fused centroids and
scores without discarding evidence from the shoulder cameras.
Appearance embeddings are intentionally not part of this geometry-only MVP.
`object_summary.json` is generated
automatically when Stage 4 is enabled.

#### Three-layer Stage 2 output contract

Stage 2 deliberately separates episode indexing, frame evidence, and temporal
reasoning. The schemas live in `schemas/` and all three JSON layers carry the
same integer `schema_version` (currently `3`) and UUID `generation_id`:

1. **`frame_fused_candidates.json`** is a lightweight episode manifest. Its
   frame entries contain status, object count, and a relative
   `fused_objects_json` reference plus an optional `fusion_debug_json` reference;
   it never stores objects or point clouds. Frame order is the order of the
   `frames` array and is not duplicated in a second list.
2. **`frames/<frame_key>/fused_objects.json`** owns the current frame's
   `objects`, their observations, counts, and compact `candidate_outcomes`.
   Point arrays are
   still external in the sibling `fused_geometry.npz` and referenced by key.
   The key is `<six-digit frame_index>_<frame_id>` (for example `000000_0`).
   `candidate_outcomes` gives each Stage-1 candidate's final status, stable
   reason code, and resulting object/replacement ID without repeating every
   intermediate event.
   The sibling **`fusion_debug.json`** retains the complete candidate lifecycle,
   readable reason messages, backprojection/filter diagnostics, and tracking
   checkpoint. Resume mode reads that checkpoint, while older outputs with
   inline `diagnostics.tracking_state` remain supported.
3. **`object_summary.json`** contains cross-frame tracks, aggregate statistics,
   decision-ready scalar metadata, and `frame_ref` links back to the per-frame
   files. It does not duplicate complete point clouds or other full geometry.

`object_summary.json` uses the `compact_v1` storage layout. Per-frame candidates
retain only Qwen filtering and visualization fields; verbose observation
provenance remains in the referenced `fused_objects.json`. Track statistics are
stored once without duplicating trajectories, and pairwise geometry is derived
by Stage 4 only for its active decision window. This keeps summary growth linear
in visible object samples instead of quadratic in the number of objects.

The summary builder consumes the referenced frame JSON files one at a time, so
all frame artifacts need not coexist in memory. Readers validate both identity
fields before joining artifacts; a schema mismatch or a `generation_id` from a
different fusion run is an error rather than a silent mixed-run result. A
compatible resumed run retains its generation ID, while a new run creates one.
See `schemas/frame_fused_candidates.schema.json`,
`schemas/fused_objects.schema.json`, `schemas/fusion_debug.schema.json`,
`schemas/candidate_debug.schema.json`, `schemas/object_summary.schema.json`,
`schemas/episode_candidates.schema.json`, and
`schemas/object_predictions.schema.json`.

#### JSON readability conventions

Persisted JSON artifacts are UTF-8, two-space indented, newline-terminated,
and written atomically where the shared I/O helper is used. Major artifacts
start with `schema_version` and `artifact_type`, and expose a compact top-level
`summary` before their detailed records. Existing fields remain available for
backward compatibility in readers where legacy artifacts are supported. Stable
machine-readable failure codes use
`reason_code`; `reason_message` provides a concise human-readable explanation.
Large point arrays remain in NPZ archives rather than being embedded in JSON.

### Stage 3: fused-object sanity visualization

`visualize_fused_candidates.py` projects fused world points back into each
camera image and combines fixed `front`, `left_shoulder`, and `right_shoulder`
camera panels plus the four-angle 3D point-cloud panel into one 2x2
`<frame_id>_montage.png`. A missing RGB/calibration dependency produces a labelled
placeholder/raw-camera panel instead of changing the layout. It no longer writes separate
`*_reproj.png` or `*_pointcloud.png` panels. This stage does not change fusion
results; use it to diagnose
depth decoding, camera transforms, bad masks, or incorrect clustering. Its
`sanity_report.json` contains only frame/image references and counts. Detailed
stored/recomputed centroid residuals, centroid-to-cloud distances, voxel
component ratios, and main-component gaps live in `sanity_debug.json`.
Dense `O*` labels are placed with overlap avoidance and connected to their
centroids by translucent leader lines; boxes and annotations are rendered on a
transparent layer.

| Python argument | Pipeline variable | Default | Purpose |
| --- | --- | --- | --- |
| `--fused-json` | derived from `OUTPUT_DIR` | required | Stage 2 output. |
| `--output-dir` | derived | sibling `viz/` | Visualization destination. |
| `--frame-ids`, `--max-frames` | — | all | Render a frame subset or cap. |
| `--cameras` | `CAMERAS` | `front,left_shoulder,right_shoulder` | Override the fixed overlay-camera set. |
| `--camera-params-json` | `CAMERA_PARAMS_JSON` | auto | Use the same camera override as fusion. |
| `--invert-rlbench-extrinsics` | `INVERT_RLBENCH_EXTRINSICS` | off | Keep projection convention aligned with Stage 2. |
| `--point-stride` | — | `4` | Render every Nth world point. |
| `--point-radius` | — | `2` px | Reprojected marker radius. |
| `--mask-alpha` | — | `80` | Overlay opacity in the range 0–255. |
| `--skip-pointcloud` | — | off | Omit the matplotlib 3D scatter plot. |
| `--montage-columns` | `VIZ_MONTAGE_COLUMNS` | `2` | Number of panel columns in the per-frame montage. |
| `--montage-cell-width` | `VIZ_MONTAGE_CELL_WIDTH` | `512` px | Square cell size used for camera and point-cloud panels. |

Set `SKIP_VIZ=1` in the shell pipeline to omit this stage.

### Stage 4: Qwen3-VL object-level role decision (optional)

`qwen3vl_object_role_decision.py` consumes Stage 2's object summary rather than
raw SAM role IDs. Every sampled frame receives a decision using a rolling
temporal window ending at the current frame. The pipeline defaults to adaptive
Qwen keyframes: stable frames reuse semantic compatibility from the latest
model call while recalculating current `t-2 -> t` geometry, gripper state, and
dynamic roles. Direct CLI use defaults to `every-frame` for compatibility. The
top-level `decision` remains the last frame's result, while `frame_decisions`
contains the complete episode and per-frame performance/source metadata.
Visual input defaults to labelled full-scene montages. Each Qwen keyframe sees
the latest two available frames, with fixed `front`, `left_shoulder`, and
`right_shoulder` panels. Set `DECISION_VISUAL_MODE=patches` to restore the
previous object contact-sheet input; the two modes are never mixed.

| Python argument | Pipeline variable | Default | Purpose |
| --- | --- | --- | --- |
| `--object-summary-json` | `OBJECT_SUMMARY_JSON` or derived | required | Decision-ready Stage 2 summary. |
| `--output-json` | `DECISION_OUTPUT_JSON` | `object_predictions.json` | Selection result path. |
| `--model-path` | `DECISION_MODEL_PATH` / `MODEL_PATH` | script model path | Qwen3-VL checkpoint for selection. |
| `--decision-scope` | `DECISION_SCOPE` | `all` | `all` emits every frame; `single` selects one frame. |
| `--decision-policy` | `DECISION_POLICY` | CLI `every-frame`; pipeline `adaptive` | Select per-frame Qwen calls or adaptive keyframe refresh plus temporal propagation. |
| `--decision-refresh-interval` | `DECISION_REFRESH_INTERVAL` | `5` | Maximum stable sampled-frame interval between adaptive Qwen refreshes. |
| `--decision-min-propagation-confidence` | `DECISION_MIN_PROPAGATION_CONFIDENCE` | `0.70` | Refresh Qwen when propagated confidence falls below this value. |
| `--reference-switch-confirmation-frames` | `REFERENCE_SWITCH_CONFIRMATION_FRAMES` | `2` | Require the same proposed Reference for consecutive sampled frames before changing the locked ID. |
| `--decision-frame` | `DECISION_FRAME` | `last` | Select the `first` or `last` frame when scope is `single`. |
| `--decision-frame-id` | `DECISION_FRAME_ID` | unset | Explicit frame ID; when set it forces a single-frame decision. |
| `--decision-window-frames` | `DECISION_WINDOW_FRAMES` | `3` | Current frame `t` plus its two preceding frames `[t-2, t-1, t]` (when available), evaluated in one model call; `1` is single-frame mode. |
| `--[no-]use-decision-history` | `USE_DECISION_HISTORY` | off / `0` | Optionally feed recent model outputs back into the prompt. Off by default so an early wrong target does not propagate. |
| `--decision-visual-mode` | `DECISION_VISUAL_MODE` | `scene` | `scene` uses labelled full RGB montages; `patches` restores legacy object contact sheets. |
| `--decision-scene-cameras` | `DECISION_SCENE_CAMERAS` | `front,left_shoulder,right_shoulder` | Fixed camera panels in each scene montage. |
| `--decision-scene-window-frames` | `DECISION_SCENE_WINDOW_FRAMES` | `2` | Latest temporal frames rendered in scene mode (`t-1`, `t` when available). |
| `--max-candidate-images` | `MAX_CANDIDATE_IMAGES` | `3` | Maximum chronological contact sheets in `patches` mode. |
| `--candidate-views-per-object` | `CANDIDATE_VIEWS_PER_OBJECT` | `1` | Best-scoring camera crop count per object in `patches` mode. |
| `--decision-max-visual-pixels` | `DECISION_MAX_VISUAL_PIXELS` | `393216` | Maximum pixels per scene montage or contact sheet. |
| `--decision-artifacts-dir` | `DECISION_ARTIFACTS_DIR` | `decision_inputs/` | Scene montage/contact-sheet output directory. |
| `--max-candidates-for-decision` | `MAX_CANDIDATES_FOR_DECISION` | `12` | Candidate cap after filtering. |
| `--min-candidate-point-count` | `MIN_CANDIDATE_POINT_COUNT` | `0` (off) | Remove sparse fused objects. |
| `--min-candidate-camera-count` | `MIN_CANDIDATE_CAMERA_COUNT` | `1` | Require support from this many cameras. |
| `--min-candidate-sam-score` | `MIN_CANDIDATE_SAM_SCORE` | `0.0` (off) | Remove candidates below a fused SAM score. |
| `--max-ee-distance-m` | `MAX_EE_DISTANCE_M` | unset | Remove objects never close enough to the end effector in the window. |
| `--[no-]dynamic-role-reasoning` | `DYNAMIC_ROLE_REASONING` | on / `1` | Maintain task predicates, gripper events, 3D relations, and persistent object state across the episode. |
| `--grasp-distance-m` | `GRASP_DISTANCE_M` | `0.06` m | Maximum gripper/object distance used to confirm a grasp candidate. |
| `--object-moving-distance-m` | `OBJECT_MOVING_DISTANCE_M` | `0.01` m | Minimum sampled-frame displacement treated as object motion. |
| `--object-stable-distance-m` | `OBJECT_STABLE_DISTANCE_M` | `0.008` m | Maximum sampled-frame displacement treated as stable. |
| `--placement-stable-frames` | `PLACEMENT_STABLE_FRAMES` | `2` | Stable sampled frames required after release before confirming placement. |
| `--min-support-xy-overlap` | `MIN_SUPPORT_XY_OVERLAP` | `0.35` | Minimum smaller-bbox XY overlap for `ON`/support evidence. |
| `--min-support-vertical-gap-m`, `--max-support-vertical-gap-m` | matching pipeline variables | `-0.01`, `0.025` m | Accepted vertical gap between an upper object and support. |
| `--min-containment-ratio` | `MIN_CONTAINMENT_RATIO` | `0.5` | Minimum source bbox fraction inside a goal container bbox. |
| `--grounding-min-side` | `DECISION_GROUNDING_MIN_SIDE` | `512` | Compatibility setting for shared grounder operations; Stage 4 sheets use the pixel cap above. |
| `--attention-backend` | `DECISION_ATTENTION_BACKEND` | `auto` | Prefer FlashAttention 2 when installed, otherwise SDPA/eager. |
| `--max-new-tokens` | `DECISION_MAX_NEW_TOKENS` | `384` | Structured response generation budget per Qwen keyframe. |
| `--max-retries` | `DECISION_MAX_RETRIES` | `0` | Optional malformed-output retry; adaptive mode instead marks uncertainty and refreshes next frame. |
| `--dry-run` | — | off | Save the exact decision payload without running Qwen. |

Enable this stage with `SKIP_DECISION=0`. Doing so also enables object-summary
generation in Stage 2. Filters apply before the candidate cap and are useful
for excluding fragments, but aggressive thresholds can discard the true task
object. A non-relational instruction is allowed to produce a confident
`reference_object_id=null`; descriptive colors, bases, or parts do not by
themselves create a separate reference object.

Reference selection uses a separate `reference_compatible_object_ids` semantic
gate and Stage-1 reference-role evidence; it never reuses the target-compatible
set as an implicit goal anchor. A selected Reference is temporally locked. A new
ID needs consecutive confirmation (two sampled frames by default), while a
briefly missing locked ID yields `reference_object_id=null` instead of rebinding
to the first visible distractor. Confirmed physical placement events may switch
the lock immediately. Each compact frame decision keeps `model_reference_object_id`
and `reference_stability` so proposed, retained, missing, and switched states are
distinguishable from underlying O-ID tracking failures.

Each frame records `model_invoked`, `decision_source`,
`source_model_frame_id`, `refresh_reasons`, and preprocessing/generation token
statistics. The top-level `performance` block aggregates model calls, propagated
frames, scene/patch image counts, visual preparation time, model time, visual
pixels, and input/output tokens.

Normal `object_predictions.json` output uses `compact_v1`: each frame keeps IDs,
model-call provenance, refresh reasons, performance, one non-duplicated image
metadata list, and the final target/reference/confidence fields. Prompt-time
snapshots such as `dynamic_role_context`, `online_history`, repeated montage
lists, and detailed intermediate selection structures are not persisted per
frame. `--dry-run` uses `debug_full_v1` and intentionally retains complete
messages so the exact Qwen request remains inspectable.
Malformed JSON is marked uncertain and schedules another adaptive refresh on
the next frame instead of immediately repeating an expensive call.

Target selection is explicitly two-stage. Qwen first returns
`instruction_compatible_object_ids` from visual and instruction identity cues.
Only within that set does the pipeline prefer the smallest current
end-effector distance, followed by a consistent approach over `[t-2, t-1, t]`.
The JSON keeps `model_target_object_id` and `target_selection` diagnostics so
distance-based adjustments remain inspectable.

Dynamic role reasoning compiles the instruction into an action family and goal
predicate such as `PRESSED`, `OPEN`, `ON`, `IN`, or `INSERTED_IN`; it does not
branch on RLBench task names. A short rolling window estimates approach and
motion, while episode-level deterministic state retains confirmed `GRASPED`,
`RELEASED`, `PLACED_ON`, and `PLACED_IN` events. Object IDs remain role-neutral:
for repeated `ON` goals, a released and stable previous target can become the
top support/reference of the next subgoal. Qwen still supplies semantic identity;
the deterministic layer validates physical state and prevents target/reference
from resolving to the same object. Final confidence is calibrated from semantic,
tracking, gripper-interaction, and relation evidence; the original Qwen value is
preserved as `model_confidence` with inspectable `confidence_components`.

### Stage 5: decision overlays (optional)

`stage4_visualize_decision.py` joins `object_predictions.json` with
`frame_fused_candidates.json`, re-renders fused objects once on each raw camera
view, and writes `decision_visualization.json` plus images under
`viz_decision/`. A selected internal label is replaced in place (`O2` becomes
`T2` or `R2`) instead of drawing both labels. Labels have no filled translucent
background block, so small objects remain visible. The visualization report
stores only source references and rendered-image records; decisions themselves
remain canonical in `object_predictions.json` instead of being copied again.

Its required inputs are `--object-predictions-json` and `--fused-json`.
`--episode-dir`, `--output-dir`, `--viz-dir`, `--cameras`,
`--camera-params-json`, and `--rlbench-low-dim-obs` override discovered paths
or camera selection. Rendering is controlled by `--point-stride` (default
`4`), `--point-radius` (`2` pixels), `--mask-alpha` (`90`), `--box-width`
(`1` pixel), and `--annotation-alpha` (`150`); use
`--invert-rlbench-extrinsics` when Stage 2 used the same transform option.
The pipeline exposes `DECISION_VIZ_OUTPUT_DIR`, `DECISION_VIZ_BOX_WIDTH`,
`DECISION_VIZ_ANNOTATION_ALPHA`, and `SKIP_DECISION_VIZ` for this stage.

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
| `TRACK_MAX_MISSED_FRAMES` | `2` | Preserve object IDs through brief processed-frame occlusions |
| `TRACK_MAX_SIZE_RATIO` | `2.5` | Reject temporal ID matches with implausible bbox-size changes |
| `SKIP_CANDIDATES` | `0` | Reuse an existing `episode_candidates.json` |
| `SKIP_FUSION` | `0` | Reuse an existing `frame_fused_candidates.json` |
| `SKIP_VIZ` | `0` | Disable fusion visualizations |
| `SKIP_DECISION` | `1` | Set to `0` to run object-level role selection |
| `DECISION_SCOPE` | `all` | Run one rolling-window decision per frame; use `single` for debugging |
| `DECISION_POLICY` | `adaptive` | Emit every-frame decisions while limiting Qwen calls to keyframes/events |
| `DECISION_VISUAL_MODE` | `scene` | Use full-scene montages; set `patches` for the previous contact-sheet input |
| `DECISION_SCENE_WINDOW_FRAMES` | `2` | Render `t-1` and `t` as scene montages on Qwen keyframes |
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
