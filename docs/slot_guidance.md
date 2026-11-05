# Slot Guidance v1 (PR1)

PR1 provides a detached interchange contract and mock validation. It does not
extract a checkpoint's slots, produce SAM3 prompts, run SAM3 or condition actions.
Read [the initial interface audit](slot_guidance_interface_audit.md) before
implementing a real producer. BridgeVLA remains at its audited baseline.

## Files and data flow

```text
current-frame producer (oracle_gt / predicted_coarse_slot / mock)
  -> filter renderer padding and invalid support BEFORE inverse normalization
  -> explicitly convert coordinates with the verified current-frame transform
  -> save_guidance(JSON path, metadata, target_points_world, reference_points_world)
  -> JSON + checksum-named NPZ sidecar
  -> load_guidance(JSON path, expected_source_frame_index=current source index)
  -> detached SlotGuidance for a future PR2 prompt adapter
```

The standalone `slot_guidance.py` needs only NumPy and the Python standard
library. It never imports SAM3, torch or Qwen, and no existing entry point imports
it. The new JSON Schema is `schemas/slot_guidance.schema.json`. Runtime validation
also checks constraints unavailable to JSON Schema: finite NPZ `[N,3]`, counts,
digest, similarity matrix, exact timestamp/frame match and cube inverse algebra.

All **stored** centers, point sets and optional ranges are already in
`coordinate_frame="rlbench_world"`, `coordinate_units="m"`. The transform is
conversion provenance. Loading does **not** transform stored world points again.

## Identity, semantics and validity

`episode_id` identifies the episode/world instance. `frame_id` is an opaque RGB
filename or record identifier. `source_frame_index` is the exact zero-based
episode observation index, never an enumerate index of sampled frames. For
example, `frame_id="sample-3"` can legitimately have `source_frame_index=150`.
Every consumer with a current frame must supply that expected index when loading.
The mock also checks episode ID and frame ID. PR2/PR3 must bind RGB, depth and
per-frame camera calibration to this same episode and source index.

| Role status | Meaning | `semantic_valid` | Empty geometry |
| --- | --- | --- | --- |
| `VISIBLE` | Role exists with an observation-level visibility claim; each camera still needs verification | true | allowed; does not imply prompt readiness |
| `NOT_VISIBLE` | Role exists but has no verified current visibility | true | allowed; current-frame geometry from another view may be retained |
| `LOW_CONFIDENCE` | Task role exists, but the model cannot reliably assign/localize its instance | true | allowed |
| `NULL` | Reference does not exist in task semantics | false | required; center null, no range, label or instance ID |

Target cannot be NULL. Here `semantic_valid` denotes task-role presence; it is
not a copy of BridgeVLA's thresholded `role_token_valid` or geometry `valid`.
A producer must explicitly make/preserve the semantic decision. Empty points or
a failed camera projection cannot establish NULL.

`confidence` is a declared producer role/localization score in [0,1], not a SAM3
class score. `reference_null_probability` is recorded independently in [0,1];
NULL does not force it to exactly 1. Optional `semantic_label` and `instance_id`
preserve the producer's identity, including two same-category instances. They
do not prove that geometric instance selection is correct.

`spread_world`, when present, means verified positive **world-axis half-extents**
in metres around `center_world`, with required `spread_source`. It is not the
weighted standard deviation in BridgeVLA's 17D `geometry`. Center-only guidance
without a verified range is preserved as incomplete; PR2 must not invent a box.

## Coordinate and time provenance

Required `transform` fields are `source_frame`, `target_frame=rlbench_world`,
`matrix_world_from_source:[4,4]`, a nonempty `source` description, acquisition
`timestamp`, and `source_frame_index`. Timestamp and index must identify the
same observation as top-level `observed_at` and `source_frame_index`.
`created_at` records artifact creation and cannot precede acquisition. Timestamps
use RFC3339 with uppercase `T`/`Z` and explicit UTC/offset; simulation data needs
an explicitly documented timestamp mapping, never an invented acquisition time.

`identity_world_transform(...)` is for coordinates already known to be RLBench
world metres. `bridgevla_cube_transform(center_world, cube_scale, ...)` records
the exact inverse `world=cube/cube_scale+center_world`. `points_to_world(...)`
applies an explicit validated matrix once, before saving. Generic proper
uniform-scale similarity matrices can record a composed crop/render inverse.
Singular, reflected, nonuniform, nonfinite or nonaffine matrices are rejected.

Neither helper discovers normalization. The BridgeVLA producer still needs the
actual scene center/scale, stage, checkpoint flags, any `norm_corr` factor and
crop transform. Missing or unverified factors must stop export. Runtime checks
cannot detect a plausible but wrong calibration, a fabricated provenance record
or unreported future reads by an upstream producer.

`source_git_sha` is the full producer Git SHA, not the consumer's SHA.
`source_checkpoint` is mandatory for `predicted_coarse_slot`; oracle/mock sources
may use null. A path identifies the checkpoint but does not authenticate its
contents. A real experiment should additionally retain its checkpoint digest
and complete configuration alongside the artifact.

PR1 reads only the specified JSON and its adjacent NPZ; it does not read any
observation, camera sequence, action or expert trajectory. This bounds PR1 I/O,
but does not prove an upstream producer is causal. PR2/PR3 must use strict
current-frame providers instead of whole-episode loaders. Do not use replay
future `gripper_pose` labels, future keypoints or action-centered augmentation.

## Python and CLI interface

```python
from slot_guidance import save_guidance, load_guidance, MockBackend

# metadata follows schemas/slot_guidance.schema.json; all points already world/m.
# The writer fills points_file, points_sha256, role points_key and point_count.
artifact = save_guidance(
    "artifacts/frame.json", metadata,
    target_points_world=target_xyz,       # [Nt,3], including explicit [0,3]
    reference_points_world=reference_xyz, # [Nr,3]
)
current = load_guidance("artifacts/frame.json", expected_source_frame_index=150)
result = MockBackend(current).predict(
    episode_id=current.metadata["episode_id"],
    frame_id=current.metadata["frame_id"], source_frame_index=150,
)
```

Arrays must be finite, real numeric `[N,3]`. Empty arrays must have shape `[0,3]`,
not `[0]`. The NPZ contains exactly `target_points_world` and
`reference_points_world`. Returned arrays are copied, read-only float64. Metadata
is copied, and each mock result is isolated from other results. The MockBackend
mocks this boundary; it does not predict masks or demonstrate SAM3 quality.

```powershell
python slot_guidance.py artifacts/frame.json --source-frame-index 150
```

Validation raises `ValueError`; missing files raise filesystem exceptions.
The writer rejects an existing JSON unless `overwrite=True`. It publishes a
checksum-named sidecar before atomically publishing JSON, using local filesystem
hard links (same-directory link support required). Overwrite keeps old sidecars
so concurrent readers of old JSON remain valid. A failed publication may leave
an unreferenced content-addressed sidecar; garbage collection is not part of PR1.
Do not manually remove sidecars while readers use them. Duplicate JSON keys,
pickle arrays, unrecognized fields and nonlocal sidecar paths are rejected.

## Verification

In a prepared repository environment, install the test extra if needed and run:

```powershell
python -m pip install -e ".[test]"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:BRIDGEVLA_REPO='C:\Users\yiwei.chen\BridgeVLA'
python -m pytest tests/test_slot_guidance.py tests/test_slot_guidance_geometry.py tests/test_camera_geometry.py tests/compatibility/test_pipeline_baseline.py -q
```

`BRIDGEVLA_REPO` enables two optional direct source comparisons: actual
`place_pc_in_cube` inverse (requires torch) and bundled PyRep world backprojection.
Without it these comparisons skip. All remaining tests run without SAM3.
Test schema validation uses the `jsonschema` test extra, not a runtime dependency.

The PR1 run on 2026-11-05 passed **46 tests and 58 subtests**, including the two
cross-repository comparisons, existing camera tests and the default shell
pipeline invocation/environment snapshot regression. Unrelated environment
pytest plugin autoload was disabled after an initial plugin startup MemoryError.
This verifies the tested software contracts; it is not a BridgeVLA checkpoint
output comparison, a SAM3 experiment or a robot success-rate measurement.

Remaining gates: verified real coarse-slot export/provenance; installed SAM3
signature and coordinate checks; prompt/mask candidates and cache diagnostics
(PR2); strict original RGB-D/calibration snapshots and labeled Oracle/Predicted
experiments (PR3); disabled-feature checkpoint equality and frozen optional
action conditioning (PR4). No robot performance metric is reported in PR1.
