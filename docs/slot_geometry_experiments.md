# PR3: role-preserving RGB-D geometry and offline experiments

This PR adds `slot_guidance_producers.py`, `slot_geometry_fusion.py`,
`slot_experiment_metrics.py`, `run_slot_geometry_experiments.py`,
`slot_guided_synthetic_demo.py`, three focused test modules and this document.
`slot_guided_io.py` additionally binds inference to exact RGB-D/Guidance digests
and puts diagnostic legends beside the original image so small objects remain
visible; its artifact test was updated for that separate legend area. Existing
Qwen/SAM3, fusion and model entry points are unchanged.

## Current-frame producers

`oracle_guidance(snapshot, role_masks, ...)` uses explicit task-role annotations
and original-resolution masks from every current camera. It backprojects only
finite positive measured depth with that frame's K and `T_world_camera`.
Reference NULL must be explicitly annotated. Empty masks for an existing role
mean NOT_VISIBLE; visible masks without usable depth mean LOW_CONFIDENCE.

`coarse_slot_guidance(snapshot, slot_output, point_valid=..., transform=..., ...)`
accepts detached **unbatched** points `[2,P,3]`, confidence/role_token_valid/valid
`[2]`, a scalar NULL probability and optional 17D geometry. It requires an explicit
point-support mask and its source because the existing Slot top-k output lacks
guaranteed point validity. Renderer sentinels are removed before the explicit
inverse transform; genuine world-origin points remain valid. World conversion
must include actual cube/crop/legacy norm_corr factors. Slot standard deviations
are never interpreted as full object extents. Missing support defaults to
LOW_CONFIDENCE; NOT_VISIBLE needs separate current-frame visibility evidence.

These are executable interchange producers, not an automatic checkpoint hook.
An actual frozen Coarse run must supply the audited transform and point-support
evidence. No transformer/torch import or model execution is needed by the producers.

## Fusion and geometry artifact

`fuse_geometry(snapshot, guidance, inference, config=FusionConfig(...))`:

1. Checks exact episode/frame/source-index/time, calibration, actual RGB-D and
   Guidance hashes, including the actual camera object versus its manifest.
2. Backprojects every **accepted** candidate using the existing camera helpers.
3. Associates candidates separately for Target and Reference, using the existing
   `fusion_matching.symmetric_percentile_nearest_distance` metric. A bounded
   joint hypothesis search chooses a mutually consistent set with at most one
   candidate per camera. A lower SAM score can win through cross-view agreement.
4. Applies explicit robust quantile filtering and metric voxel downsampling;
   records observed point center/bbox, supporting cameras, role confidence,
   segmentation quality, geometric quality and composite geometry confidence.

No Target/Reference cross-role merging or temporal identity extrapolation occurs.
Default minimum support is one camera; `min_support_cameras=2` requests stricter
multi-view evidence. One supported camera cannot establish cross-view accuracy.
Search-limit exhaustion, inconsistent cameras, empty masks and invalid depths
produce explicit failures rather than a guessed object. These are observed
surface clouds/bboxes, not complete hidden object shapes.

`save_geometry(path, result)` emits `slot_object_geometry` schema 1 JSON plus
checksum-linked `geometry.<sha256>.npz`. `load_geometry` restores finite
`target_points_world:[Nt,3]` / `reference_points_world:[Nr,3]` and validates counts,
centers, bbox, semantic states and optional expected frame/time/calibration.
NULL and unavailable roles use explicit `[0,3]` arrays. The JSON contains:

- episode/frame/source index/time, `coordinate_frame=rlbench_world`, units m;
- geometry and calibration SHA256, Guidance Git/checkpoint source, available
  checkpoint digests, backend kind and preserved NULL posterior;
- role status, semantic validity, role/geometric/segmentation confidence, center,
  world bbox, point key/count, supporting cameras and failure diagnostics.

No episode pickle or future observation/action is opened. Original packed depth
requires the same frame's near/far; wrist calibration has no static fallback.

## A/B/C and ablation harness

```powershell
python run_slot_geometry_experiments.py --manifest experiment.json --output-dir artifacts/real_experiment --backend sam3 --checkpoint /path/to/sam3.pt --device cuda
```

`experiment.json` is a strict `slot_geometry_experiment` schema-1 object with
`evidence_kind` (`real_offline` or `synthetic_mock`) and a `frames` list. Each
entry provides `snapshot_json`, `source_frame_index`, `oracle_guidance_json`,
`predicted_guidance_json`, `ground_truth_json`, and optionally
`baseline_report_json` / `mock_candidates_json`. Frame indices must be unique
and ascending per episode. Paths are relative to the manifest. Referenced files
must describe the exact current observation; whole-episode files are unsupported.

The runner executes A (Oracle) and B (predicted Coarse) in both box-only and
point+box modes. Real B requires `predicted_coarse_slot` provenance; Mock cannot
be labeled real. C consumes a separately executed original OCRR/action-path
baseline report, preserving its method/checkpoint/evidence identity. It does not
silently substitute a new pipeline for C. Missing baseline runtime is recorded
as `not_executed`. Ground-truth evaluation masks are loaded after A/B inference
and never passed into B inference. Oracle Guidance is intentionally GT-based.

Current labels contain episode_id/frame_id/source_frame_index and
`roles:{target/reference:{status,instance_id,masks:{camera:path}}}`. Every camera
must be explicitly annotated. A C report must have the same identity,
`method=original_ocrr_action_path`, matching evidence_kind, checkpoint/Git source
and externally measured `metrics`; importing it does not authenticate those
measurements or establish real acceptance.

Outputs are `experiment_report.json`, flat `metrics.csv`, all PR2 masks/prompts/
scores/diagnostics and PR3 object geometry. Metrics include separate per-role
selection accuracy (only comparable instance IDs), prompt hit (all positive
points in the annotated instance, or box center for box-only), selected mask IoU,
75th-percentile cross-view surface distance, robust-center 3D error, backend
latency and peak CUDA allocated memory. Missing/failed selections count as IoU
zero for an annotated visible instance. Missing role identity remains unavailable.
Single-view consistency, Action error, paired rollouts and unavailable GPU
measurements remain null with an explicit reason. Evaluator-only pure helpers
provide translation-error and paired-success arithmetic without feeding labels
into perception. No future expert action is loaded by this runner.

## Reproducible synthetic software experiment

```powershell
python slot_guided_synthetic_demo.py --output-dir artifacts/synthetic_slot_demo
```

Use a new output directory: Guidance writers intentionally refuse implicit
overwrite. The demo generates two identical-category planar instances in front
and moving wrist cameras at source frame 7 (`frame_id=sample-2`), known original
RGB-D/calibration, Oracle guidance and a **mocked** Coarse point output. Supplied
Mock candidates include a wrong-instance mask with SAM score .99 and the correct
mask with score .65. Geometry/prompt scoring rejects the wrong instance.

In the executed demo, all four A/B prompt-mode configurations obtained fixture
IoU 1, prompt-hit 1, identity accuracy 1, 3D center error 0 and approximately
`1.7e-17 m` cross-view surface distance. The masks are supplied by the fixture:
these numbers verify deterministic software behavior, **not SAM3 or Coarse
model accuracy**. Mock per-frame latency is recorded separately from any real
model benchmark. C, real Action error, paired RLBench success and CUDA VRAM
remain unavailable. `real_offline_acceptance_passed` is always false; a separate
reviewed real-model report is required for PR4 activation.

## Validation and remaining limits

The cumulative new path, original camera tests and default pipeline contract
passed **151 tests and 167 subtests**. Existing fusion tests separately passed
37 tests. The expanded unchanged Qwen candidate suite has 16 passing tests and
one pre-existing wheel-prompt expectation failure
(`test_osc_wheel_prompts_drop_color_noun_and_exclusion_text`), reproduced on the
unchanged PR1 baseline. That suite also installs a process-global torch stub
unless real torch is imported first, so run it in a separate process. Its default
prompt behavior was intentionally not changed.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:BRIDGEVLA_REPO='C:\Users\yiwei.chen\BridgeVLA'
python -m pytest tests/test_slot_guidance.py tests/test_slot_guidance_geometry.py tests/test_slot_prompt_adapter.py tests/test_slot_guided_sam3.py tests/test_slot_guided_io.py tests/test_slot_guidance_producers.py tests/test_slot_geometry_fusion.py tests/test_slot_geometry_experiments.py tests/test_camera_geometry.py tests/compatibility/test_pipeline_baseline.py -q
python -m pytest tests/test_multiview_candidate_fusion.py -q
```

The local real episode has RGB/depth/instance-handle PNGs but lacks exported
current-frame near/far/calibration and role-to-handle annotations. The available
runtime has CPU torch, no SAM3 installation/weights or BridgeVLA model checkpoint,
and no CUDA renderer. Actual A/B/C predictions, mask quality, action outputs and
paired RLBench rollouts were not executed. These data/runtime gaps must be filled
before real acceptance; the default-disabled PR4 code remains a software draft.
