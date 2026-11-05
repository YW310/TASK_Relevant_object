# PR1 initial interface and coordinate audit

Audit date: 2026-11-05 (Asia/Shanghai). This is the initial audit of the existing
interfaces, recorded before the Slot Guidance implementation is committed. It is
not a SAM3 integration or a claim of improved robot performance.

| Repository | Audited baseline SHA | Local source |
| --- | --- | --- |
| `YW310/BridgeVLA` | `20c0e67aeb71c64c307d1f7c1097a41598a90181` | `C:\Users\yiwei.chen\BridgeVLA` |
| `YW310/TASK_Relevant_object` | `5f8f1485c6a99c17d3fed4d2272c8e1988c6aab2` | isolated checkout `C:\Users\yiwei.chen\BridgeVLA\tmp\slot-guidance-pr1` |

Line references below refer to these baseline versions. Prefix `BridgeVLA/`
means a file in the first repository; `TASK/` means a file in the second. New
Guidance files are deliberately excluded from the description of existing
behavior. BridgeVLA model, configuration, training and evaluation files are not
changed by PR1.

## 1. Existing TASK interfaces

The reviewed root files are `qwen3_bbox_guided_sam3_demo.py`, `sam3_runtime.py`,
`camera_geometry.py`, `multiview_candidate_fusion.py`, `fused_candidate_io.py`,
`qwen_role_sam3_candidate_episode.py`, and `tests/test_camera_geometry.py`.
`camera_geometry.py:21-33` is already a compatibility entry point: it calls
`_legacy_bootstrap.ensure_src_path()` and reexports the implementation in
`src/relevant_object/geometry/camera.py`. New shared logic should follow the
repository's `src/relevant_object` package plus root compatibility shim pattern,
rather than introducing a second projection implementation.

| Interface | Actual data and behavior | Baseline evidence |
| --- | --- | --- |
| `backproject_mask(depth, mask, K, max_points)` | depth `[H,W]`; mask is resized with nearest interpolation if needed; finite depth strictly greater than zero only; returns camera points `[N,3]` | `TASK/src/relevant_object/geometry/camera.py:35-60` |
| `transform_points(points, T)` | homogeneous row-vector implementation `(hom @ T.T)[:,:3]`; point arrays `[N,3]` | same file `:63-70` |
| `project_points(points_world, K, T_world_camera)` | returns pixel XY `[N,2]`, validity `[N]`; internally inverts camera pose; existing validity checks `z>1e-6` only | same file `:73-94` |
| camera parameters | intrinsics `[3,3]` or `[fx,fy,cx,cy]`; extrinsics `[4,4]` or promoted `[3,4]` | same file `:11-32` |
| `run_box_prompt(...)` | original-image pixel XYXY `[4]`, float32; returns one bool `[H,W]` mask and float score | `TASK/qwen3_bbox_guided_sam3_demo.py:196-229` |
| SAM normalization | accepts `[N,1,H,W]`, `[N,H,W]`, or `[H,W]`; thresholds masks at `.5`; scores flattened to `[N]`; tensor conversion detaches | `TASK/sam3_runtime.py:62-87` |
| mask to object observation | mask/depth backprojection, camera-to-world transform, arithmetic centroid `[3]`, min/max bbox `[2,3]` | `TASK/multiview_candidate_fusion.py:1710-1731` |
| fused artifact reader | strict manifest schema v4 with generation ID; points inline or NPZ referenced by geometry path and key; `allow_pickle=False`, `[N,3]` and point count checks | `TASK/fused_candidate_io.py:13,32-48,108-160` |
| fused artifact writer | per-frame NPZ; object keys `<object_id>/points_world`; JSON references include geometry path/key and count | `TASK/multiview_candidate_fusion.py:1543-1585` |

Existing normalization functions check matrix shape, not camera calibration
correctness. The new adapter must additionally reject non-finite or singular
matrices and malformed camera rotations, and validate image size and focal
lengths. A merely well-shaped but incorrect extrinsic may remain mathematically
valid; detecting it requires scene/depth or independent calibration evidence.

`project_points` does not establish image bounds, occlusion, finite pixel output,
or support by observed depth. Its boolean output alone is insufficient to mark a
SAM prompt valid. `cloud_observability_in_camera` supplies existing depth-based
visibility diagnostics (`TASK/src/relevant_object/geometry/camera.py:97-186`),
but its one-sided occlusion test is not proof that projected points lie on the
observed object surface. Surface agreement needs a separate depth residual.

### Coordinate convention

For a camera pixel `(u,v)` with metric depth `z`, the existing code uses:

```text
p_camera = [(u-cx)*z/fx, (v-cy)*z/fy, z]
p_world  = T_world_camera @ [p_camera, 1]
p_camera = inverse(T_world_camera) @ [p_world, 1]
u = fx*X_camera/Z_camera + cx
v = fy*Y_camera/Z_camera + cy
```

Thus depth is optical-axis Z in meters, +Z is forward, and increasing camera X/Y
maps to increasing image u/v for positive fx/fy. Pixel values are coordinates in
the original RGB/depth resolution. `world` means the RLBench simulator world
associated with the saved camera pose, not the BridgeVLA cube, a virtual view,
robot base, or a normalized image frame. Record a dataset/world-frame identity
in addition to the word `world` when combining artifacts from separate runs.

Packed RLBench RGB depth is decoded using 24-bit scale `2**24-1` and per-frame
near/far values (`TASK/camera_geometry.py:36-48,59-99`). Plain image/array depth
uses the configured scale; the scale and depth mode must be explicit provenance.
Never assume a depth PNG already contains meters.

### Existing source-frame hazards

`TASK/camera_geometry.py:166-175` prioritizes numeric `frame_id`, then
`frame_index`; it does not inspect `source_frame_index`. The old test at
`tests/test_camera_geometry.py:63-69` intentionally preserves that precedence.
An enumerate index in the candidate pipeline (`qwen_role_sam3_candidate_episode.py:
1436-1438`) is the sampled-record index and can differ from the source episode
index. The new path must use an explicit source episode index without changing
this old default.

`resolve_camera_param_for_frame` prioritizes an explicit per-camera override
before dynamic observation metadata (`camera_geometry.py:265-267`). Missing
RLBench frame index defaults to frame 0 (`:278-290`); nonnumeric OSC IDs can also
fall back to frame 0 (`:269-276`); static JSON is another fallback (`:294-303`).
These behaviors are incompatible with silently resolving a moving wrist camera
for the new path. A new strict resolver must require the source index, matching
camera/frame provenance and per-frame extrinsics, and fail when unavailable.

`load_rlbench_observations` unpickles/materializes the complete observation
sequence (`camera_geometry.py:128-151`). Even if later code only indexes the
current frame, this reads future observations. It cannot be reused under the
new path's strict no-future-read contract. A current-frame snapshot or previously
exported per-frame calibration file is needed. Trusted offline dataset
preparation and online/causal inference must be described separately; a full
episode loader must not be presented as causal.

### Existing SAM3 call and unresolved installed API

The reviewed box wrapper calls:

```python
model.predict_inst(
    inference_state,
    point_coords=None,
    point_labels=None,
    box=np.asarray(box_xyxy, dtype=np.float32),
    mask_input=None,
    multimask_output=False,
    return_logits=False,
)
```

The image state is created once per image before the role loop with
`processor.set_image(image)` (`qwen3_bbox_guided_sam3_demo.py:873-878`). Preserve
this useful state-sharing contract. The old wrapper selects `argmax(scores)` and
discards other masks (`:221-229`); it must not be reused unchanged as the new
instance-selection policy. New code must retain candidates and score their
geometric/prompt consistency while preserving the selected role identity.

The model builder requests `eval_mode=True` and interactivity
(`:793-800`). `sam3_runtime.autocast_context` is autocast only (`:50-59`);
neither that context nor eval mode guarantees disabled autograd. New offline
inference must explicitly use `torch.inference_mode()` or `torch.no_grad()`.

SAM3 is not installed in the checked local test environment. The repository call
is source evidence only: the installed version's point+box parameter shapes,
point-label conventions, multimask behavior, coordinate transformation and
cache mutation are **not verified**. PR2 must inspect the actual installed
implementation/signature and run a real smoke test before enabling a real
backend. Do not infer that point+box support is proven merely because the old
wrapper passes `point_coords=None`.

The text/candidate pipeline is a different interface: `run_text_prompt` sets a
new image state per call (`qwen_role_sam3_candidate_episode.py:857-880`), with
semantic candidate aggregation later. It is not the shared interactive backend
for the new path. Existing Qwen semantic selection/defaults remain unchanged.

## 2. Existing BridgeVLA tensors and semantic boundaries

The reviewed model files are `finetune/bridgevla/models/oracle_prior.py`,
`refine_role_masks.py`, `object_conditioning.py`, `cross_scale_roles.py`,
`finetune/bridgevla/mvt/mvt.py`, `mvt_single.py`, and their agent/geometry callers.

`V` below is `renderer.num_img`, not the number of original cameras
(`mvt.py:251-258`). `rvt2.yaml:3` requests three virtual views; runtime renderer
execution has not been performed in this audit. The default RLBench input has
four original cameras: front/left_shoulder/right_shoulder/wrist
(`RLBench/utils/peract_utils_rlbench.py:8-19`). These camera namespaces must not
be interchanged.

| Tensor / value | Actual shape, meaning and qualifications | Evidence in `BridgeVLA/finetune/` |
| --- | --- | --- |
| replay RGB / world point cloud | camera `[B,T,3,128,128]` with optional replay time dimension; preprocessing stacks time on channels | `RLBench/utils/dataset.py:52-105`; `RLBench/utils/peract_utils_rlbench.py:31-43` |
| original depth / calibration | depth `[B,T,1,H,W]`, K `[B,T,3,3]`, camera pose `[B,T,4,4]` when replay time dimension exists | `RLBench/utils/dataset.py:65-105` |
| rendered XYZ | `[B,V,3,Hr,Wr]`, usually Hr=Wr=224 with rvt2; frame depends on flags/stage | `bridgevla/mvt/mvt_single.py:550-567`; `bridgevla/mvt/configs/rvt2.yaml:8` |
| object decoder features | `[B*V,2048,16,16]`; grid and C are hardcoded in current VLM implementation | `bridgevla/mvt/mvt_single.py:108,132,636-643` |
| slot masks / logits | `[B,V,K,16,16]`, K configurable | `bridgevla/models/oracle_prior.py:993-1029` |
| `role_prior` / dictionary `prior` | coarse `[B,V,2,16,16]`, order Target then Reference; mixtures for unordered slots, independent role maps for fixed queries | `bridgevla/models/oracle_prior.py:1035-1050,1129-1133` |
| exported `object_slot_prior` | same prior under the network output name; refined mask head can instead produce `[B,V,2,224,224]` | `bridgevla/mvt/mvt_single.py:794-809`; `bridgevla/models/refine_role_masks.py:250-281` |
| `points` | `[B,2,P,3]`, P=`min(point_samples,V*Hr*Wr)`; default/configured point_samples commonly 128; current rendered frame | `bridgevla/models/oracle_prior.py:911-943`; `RLBench/configs/rlbench_o2_internal_slots.yaml:43-49` |
| `role_tokens` | `[B,2,D]`, commonly D=128; semantic features, not XYZ | `bridgevla/models/oracle_prior.py:1079-1082,1148`; same YAML `:46` |
| `geometry` | soft path `[B,17]`: Target/Reference centers 6, weighted standard deviations 6, Reference-minus-Target displacement 3, geometry-available 2; None in legacy non-soft mode | `bridgevla/models/object_conditioning.py:404-440`; `bridgevla/models/oracle_prior.py:1075-1077` |
| `reference_null_probability` | `[B]`; categorical posterior competing with Reference slots for unordered slots; sigmoid for fixed role queries | `bridgevla/models/oracle_prior.py:1040-1045,1125-1128` |
| `confidence` | `[B,2]`; weighted objectness for unordered slots, prior max-support for role queries; not calibrated identity probability | `bridgevla/models/oracle_prior.py:1052-1061,1103-1109,1134` |
| `role_token_valid` | `[B,2]`; thresholded semantic/map reliability before rendered-geometry availability | `bridgevla/models/oracle_prior.py:1062-1077,1134-1136` |
| `valid` | `[B,2]`; additionally requires weighted finite non-background rendered geometry in soft mode | `bridgevla/models/object_conditioning.py:420-440` |
| Oracle pair selection | input `[B,O,P,3]` plus valid/role `[B,O]`; output `[B,2,P,3]`, validity `[B,2]`, source slot `[B,2]`; ambiguous/missing roles invalidated in non-strict mode | `bridgevla/models/oracle_prior.py:189-250` |

`soft_role_geometry` excludes non-finite XYZ and all-zero/all-minus-one renderer
background. It forms weighted means and weighted standard deviations, not the
object's full extent (`object_conditioning.py:405-408,420-439`). Geometry validity
needs role validity plus weighted mass greater than 1e-6 (`:432`). The spread is
therefore not a validated box half-size. Center-plus-range fallback must require
independent, recorded range evidence; it must not manufacture object dimensions
from these numbers.

Legacy `_extract_points` masks background scores to -1 then selects top-k; it
does not export a per-point validity mask or enforce positive role-support mass
for each selected point (`oracle_prior.py:926-943`). Fewer valid XYZ pixels than
P can produce zero padding. The new export boundary must explicitly filter
padding/non-finite points before any inverse transformation. Otherwise zero
padding becomes a false point at the world-space cube center.

`reference_is_null` uses `p_null >= 1-confidence_threshold`; at threshold .25,
this is .75 (`oracle_prior.py:1062-1065`). A low-confidence Reference is explicitly
not automatically NULL (`:1067-1068`). In legacy non-soft conditioning, an
uncertain Reference can also gate Target validity (`:1069-1070`). This older
gating must be reported as model behavior, not reinterpreted as ground-truth
absence or visibility.

The new contract must retain these distinct outcomes:

- `NULL`: semantic Reference absence; a role-presence label or explicit semantic
  posterior decision supplies evidence, never an empty local point set alone.
- `NOT_VISIBLE`: role exists and is semantically trusted, but the specified
  current camera has no verified projected/depth support.
- `LOW_CONFIDENCE`: model cannot reliably choose/localize the role; semantic
  task-role presence is distinct from reliable instance selection. Do not infer
  visibility or absence from this status. The new Guidance contract uses
  `semantic_valid=True` for an existing task role; the original model's
  thresholded `role_token_valid` must not be copied into it without interpreting
  task semantics.

`cross_scale_roles.py:27-102` already separates inherited token validity/NULL
from local visible point hints and retains crop-external coarse geometry. Its
local `valid=False` must not erase the semantic role or become NULL.

## 3. BridgeVLA world/cube/crop transforms

### Global cube

`bridgevla/mvt/utils.py:9-63` implements:

```text
u_cube = s * (p_world - c_world)
s = 2 / max(x_length, y_length, z_length)
p_world = u_cube / s + c_world
```

For `with_mean_or_bounds=True`, `c_world=(pc.max+pc.min)/2` and lengths are the
scene-cloud AABB ranges (`:40-42`). Despite the docstring's word "mean", the code
does not use an arithmetic mean. For False it uses the configured scene bounds'
midpoint and lengths (`:44-52`). RLBench scene bounds are metric
`[-.3,-.5,.6,.7,.5,1.6]` by the current utility default
(`RLBench/utils/peract_utils_rlbench.py:9-16`); run/checkpoint configuration is
the authority, not this default.

`bridgevla_agent.py:2619-2643` uses this same transform for scene and Oracle
points during act. It stores inverse transforms as local Python closures in
`rev_trans`; they are not a portable JSON provenance record. Export must record
the actual center, scale, source scene/bounds and configuration, not reconstruct
an approximate transform from object centers. Bounds filtering happens first
(`:2569-2571`; `bridgevla/utils/rvt_utils.py:44-67`).

### Refinement crop

`bridgevla/mvt/utils.py:66-97` and `mvt.py:757-788` implement:

```text
q_crop = a * (u_cube - c_crop)
u_cube = q_crop / a + c_crop
p_world = (q_crop / a + c_crop) / s + c_world
```

`a=st_sca` (default 4 in `mvt/config.py:30`). At evaluation the crop center is the
current coarse predicted waypoint (`mvt.py:779-788`); training can instead use a
noisy GT waypoint (`:759-776`). Coarse Guidance must explicitly name its stage;
refine points require the additional crop inverse. `out['rev_trans']` in MVT is
the crop inverse (`:801-802`), while the agent's `rev_trans` list is the global
cube inverse. The same name refers to different coordinate boundaries.

`cross_scale_roles.transform_role_geometry` applies centers, spreads and
displacement to the actual crop frame without fabricating geometry
(`cross_scale_roles.py:8-24`). For the positive scalar cube/crop scales, metric
spread recovery divides by the corresponding scale, with no translation.

### Additional legacy correlation normalization and augmentation

When `norm_corr=True` and clean role XYZ mode is disabled, rendered XYZ features
are `u_cube / max(abs(scene_u_cube))`, or equivalently the current crop points
divided by their own maximum absolute value (`mvt.py:469-487`). The rvt2 default
sets `norm_corr=True` (`mvt/configs/rvt2.yaml:2`). The role decoder reads those
features directly (`mvt_single.py:564`). Therefore legacy Slot points can have
an additional per-scene/per-stage factor `m`:

```text
u_render = u_cube / m
p_world = (m * u_render) / s + c_world
```

For crop points, restore m before the crop inverse. Capture m at the source; it
cannot safely be guessed from the selected object points. Clean role XYZ mode
is enabled by `object_conditioning_inherit_coarse_roles` or predictor type
`role_queries` (`mvt.py:438-445`), which bypasses this extra division. Audit
actual checkpoint flags before choosing the inverse.

Image augmentation can add noise to XYZ features in legacy mode; the clean role
path restores XYZ after augmentation (`mvt.py:530-541`). Offline export must run
eval with augmentation disabled. Unsupported/missing transform provenance must
produce a failure state rather than declaring normalized coordinates to be
meters.

### Synthetic proof of consistency

A read-only numerical check imported the actual TASK geometry implementation,
the vendored BridgeVLA PyRep `VisionSensor` implementation and the actual
`bridgevla/mvt/utils.py` without constructing a model. It used float64, a 3x4
depth array `arange(12).reshape(3,4)/30+.7` meters,
`K=[[80,0,1.4],[0,85,.8],[0,0,1]]`, a camera pose with a .37-radian Y rotation and
translation `[.2,-.1,.6]`, both cube-centering modes, and crop center
`[.1,-.2,.3]` with scale 4. Results:

| Check | Maximum absolute error |
| --- | --- |
| TASK backproject+camera-to-world versus vendored PyRep world point cloud | `2.220446049250313e-16 m` |
| TASK projection of those world points versus source integer pixel grid | `5.329070518200751e-15 px` |
| Actual global cube inverse, scene-bounds mode | `0.0 m` |
| Actual global cube inverse, cloud-AABB mode | `0.0 m` |
| Actual crop inverse composed with cube inverse, either mode | `0.0 m` |

All projected depths were positive. The vendored PyRep method documents meters
and constructs world-to-camera from `R.T` and `-R.T@C`
(`bridgevla/libs/peract_colab/peract_colab/pyrep/objects/vision_sensor.py:8-28`).
Stored RLBench loader uses each `obs[i].misc` pose and converts packed depth with
that frame's near/far before backprojection
(`bridgevla/libs/peract_colab/peract_colab/rlbench/utils.py:50-88`). Together with
the nonidentity numerical check, this verifies the code convention and inverse
algebra. It does not establish that a real episode has correct calibration,
depth registration or timestamps, or that model-derived points hit the desired
object. No actual SAM3, GPU renderer or BridgeVLA checkpoint was run.

## 4. Gradient and action-conditioning boundaries

- `BridgeVLAAgent.act` is decorated `@torch.no_grad()`
  (`bridgevla_agent.py:2548`). This is the appropriate existing inference
  boundary for frozen, offline extraction, subject to source-frame restrictions.
- Scene preparation and rendering are under no_grad in the existing paths
  (`bridgevla_agent.py:1346-1435`; `mvt.py:462,653-663,757-799`). Top-k indices
  do not transfer gradients to mask scores; selected XYZ comes from this fixed
  rendered observation.
- During training, role prior, role tokens, NULL posterior and soft geometry can
  receive action gradients. The action adapter is called without detach
  (`mvt_single.py:645-697`), and inherited coarse tokens/geometry deliberately
  remain outside no_grad (`mvt.py:804-810`). Current behavior must not be
  described as a frozen Slot path.
- `refine_mask_detach` detaches only the local action prior. Tokens, including
  local updates, can still reach the anchor; auxiliary logits remain trainable
  (`refine_role_masks.py:282-299`). Projected hints and heatmap spatial inputs
  have their own detach boundaries (`:63,113,224-225`). This flag does not
  establish a complete SAM3/action gradient stop.
- `--train_object_adapter_only` / legacy `--train_oracle_adapter_only` invokes
  a helper that freezes the backbone but re-enables object slot predictors,
  Oracle adapters and refine mask heads (`RLBench/training_utils.py:136-150`;
  `RLBench/train.py:1133-1147`). It is not the requested "freeze Slot" switch.
- New JSON/NPZ Guidance is an explicit detached process/artifact boundary. PR2
  must freeze SAM3 and use inference mode; PR4 must detach all imported SAM3
  geometry and separately verify that Slot/main backbone parameter gradients
  remain disabled in its first experiment.

The existing `OracleRelationAnchorFeatureAdapter.forward_with_anchor` accepts
features, prior, role validity, relation points, current state, context, role
tokens, NULL posterior, soft geometry and role-token validity
(`oracle_prior.py:681-686`). `mvt_single.py:677-697` supplies them directly. It
can preserve an unknown Reference separately from semantic NULL and combine the
NULL posterior with role features (`oracle_prior.py:739-766`). This is a
candidate PR4 interface, not authorization to overwrite the existing
`role_prior` in PR1. The point-only adapter expects full `[B,2,P,3]` tensors and
does not currently consume a per-point validity mask; padded SAM3 points will
need an explicit, tested conversion in PR4.

## 5. Current versus future observations and entry points

Training replay contains `sample_frame`, `keypoint_frame`,
`next_keypoint_frame` and `keypoint_idx` as different fields
(`RLBench/utils/dataset.py:131-137,325-331`). Current RGB/calibration comes from
current `obs` through `extract_obs` (`:304-310`), but replay `gripper_pose` is
the future keypoint pose `obs_tp1.gripper_pose` (`:283-299,333-337`). Guidance
must never use this training label as the measured current object/camera state.

Training update can apply SE3 augmentation centered on that future action and
does not return the full augmentation provenance
(`bridgevla_agent.py:1370-1384`). It is unsuitable as an offline world Guidance
export hook without an explicit inverse. Use unaugmented current inference.

Actual `low_dim_state` order is `[gripper_open,left_joint,right_joint,time]`
(`bridgevla/libs/peract/helpers/utils.py:342-344,369-372`), despite stale comments
elsewhere; the role state takes the first three values
(`bridgevla_agent.py:2560-2562`). Camera K/poses are copied from the current
observation's misc (`peract/helpers/utils.py:365-367`). The default live eval
camera config has RGB/point cloud enabled but `depth=False` (`:387-393`). PR3
must obtain original metric depth explicitly rather than assuming eval's
observation dictionary contains it.

BridgeVLA's stored-demo loader eagerly reads all frames
(`peract_colab/rlbench/utils.py:50-88`). Some existing Oracle/expert-replay paths
also build whole-episode event/ground-truth schedules
(`RLBench/utils/custom_rlbench_env.py:141-163`; `o2_oracle_provider.py:2428-2481`).
These paths cannot be assumed to satisfy strict no-future observation/action
reads. An Oracle experiment may use declared current-frame role annotations,
but must not derive current guidance from future expert motion/events. The new
path must use a source-frame-bounded provider and log read provenance.

| Entry point | Configuration/interface relevance |
| --- | --- |
| `BridgeVLA/finetune/RLBench/train.py` | MVT config merge `:1084-1091`, construction `:1098-1131`, freezing `:1133-1147`; default CLI config paths `:1446-1447` |
| `BridgeVLA/finetune/RLBench/eval.py` | loads saved MVT config `:159-165`, constructs network `:172-200`, observation config `:413` |
| `BridgeVLA/finetune/RLBench/configs/rlbench_o2_internal_slots.yaml` | unordered Slot experiment; point_samples 128, D128; place_with_mean False; auxiliary Oracle labels |
| `BridgeVLA/finetune/RLBench/configs/rlbench_o2_role_queries.yaml` | explicit fixed-role, inherited-role/token-preservation flags `:33-46`; separate experimental behavior |
| `TASK/qwen3_bbox_guided_sam3_demo.py` | existing Qwen interactive box pipeline; its CLI/default selection remain unchanged |
| `TASK/qwen_role_sam3_candidate_episode.py` | existing semantic text candidate pipeline; sampled enumerate index is distinct from source frame |
| `TASK/multiview_candidate_fusion.py` | reuse geometry/matching helpers with new strict frame/calibration boundary, rather than inheriting old fallback behavior |

## 6. PR boundaries and acceptance gates

PR1 delivers this audit, a detached JSON/NPZ Guidance contract, strict validation
and mock/synthetic tests. It must not enable SAM3, change Qwen defaults, change
BridgeVLA model outputs, or reinterpret old checkpoint coordinates. The schema
must specify meter/world coordinates, semantic role state, independent geometric
support, source episode index, transform provenance and timestamps. Camera
calibration provenance is a separate per-view contract in PR2/PR3. Missing
or contradictory provenance must be reviewable as a validation failure.

PR2 is gated on passing PR1 tests and the installed SAM3 API check. It adds the
strict Slot prompt adapter, shared image state, box-only and point+box offline
inference, cached artifacts and diagnostics. A mock backend verifies failure
states and candidate handling without claiming actual SAM3 accuracy. SAM3 score
alone cannot decide which same-class instance is Target or Reference.

PR3 is gated on verified prompt inference and current-frame RGB-D/calibration
access. It must handle dynamic wrist extrinsics, Oracle/current-predicted
Guidance, cross-camera consistency, invalid/missing depth and explicit NULL versus
unseen Reference. Geometry and mask metrics need real labeled episodes.

PR4 is deferred until offline experiments pass. Only then may a separate
BridgeVLA change add optional object geometry to existing Relation/Anchor
conditioning, with `enable_slot_guided_sam3=False`, preserved checkpoint loading,
frozen Slot/SAM3/main backbone and explicit SAM3 action-gradient stops. A disabled
feature comparison on the same checkpoint/inputs must establish unchanged
baseline outputs. No new coarse/refine cross-attention or fusion backbone is
part of this first integration.

Required later measurements remain unreported until actually executed: semantic
role selection accuracy, prompt instance-hit rate, SAM3 mask IoU, cross-view
consistency, metric 3D center error, action translation error, paired RLBench
success, per-frame latency and peak GPU memory. The synthetic audit above and
mock tests do not measure robot manipulation performance.
