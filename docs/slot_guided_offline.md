# PR2: current-frame Slot prompts and offline SAM3

The existing Qwen3 → SAM3 commands are untouched. This is a separate offline
entry point using the PR1 role/world/metre artifact. Real SAM3 requires the
[interactive runtime API](sam3_api_contract.md), a local checkpoint and its dependencies.
There is no fixed source-hash or commit gate. Runtime signature and image/mask
shape checks remain enabled; source hashes are provenance only. No weights are downloaded.

Start with the [bilingual executable quickstart](slot_guided_quickstart.md) for
the prepared Windows environment, a model-free run and cache replay.

## Reproducible configuration

All offline CLIs accept `--config configs/slot_guided_offline.json`. This checked-in
file matches the existing AdapterConfig/SelectionConfig/FusionConfig defaults.
Unknown fields, numeric booleans, invalid thresholds and duplicate keys fail
before model construction. Omitted sections inherit those same defaults.

```powershell
python slot_guided_config.py --config configs/slot_guided_offline.json
```

This prints effective settings and a stable configuration SHA256 without loading
SAM3 or opening data. Single-frame inference uses adapter/selection settings;
the experiment runner also uses fusion settings. Effective settings are retained
in inference and experiment reports. Configuration cannot activate BridgeVLA.

## Strict snapshot input

Provide one JSON file for exactly one acquired observation. It must not contain
actions, expert futures or episode observations. Camera K and `T_world_camera`
must describe the indicated **source episode frame**, including wrist motion.
No static-JSON, sample-ordinal or frame-zero fallback is used. Example structure:

```json
{
  "schema_version": 1,
  "artifact_type": "slot_rgbd_frame",
  "episode_id": "episode0",
  "frame_id": "sample-3",
  "source_frame_index": 150,
  "observed_at": "2026-11-05T10:00:00+08:00",
  "cameras": [{
    "name": "wrist",
    "rgb_path": "wrist_rgb/150.png",
    "depth_path": "wrist_depth/150.meters.npy",
    "depth_encoding": "meters_npy",
    "source_frame_index": 150,
    "observed_at": "2026-11-05T10:00:00+08:00",
    "calibration_source": "current observation.misc wrist camera pose",
    "intrinsics": [[100, 0, 64], [0, 100, 64], [0, 0, 1]],
    "extrinsics": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  }]
}
```

The displayed calibration is illustrative, not episode0 calibration. Never copy
it into real data. Alternative depth encoding `rlbench_rgb24` requires explicit
current-frame `near` and `far`. RGB/depth must be registered at original size.
Missing/invalid K or pose yields failed role prompts; missing files/ambiguous
depth encoding fail snapshot loading. Optional `instance_masks` is reserved for
the PR3 Oracle/evaluator, and is not opened by predicted inference.

## Prompt and inference behavior

`slot_prompt_adapter.build_prompts` validates exact episode/frame/time identity,
calibration, positive camera depth, bounds, measured surface agreement and
projected support. It uses robust point quantiles plus explicit padding and
deterministic interior measured positive points. Negative points come only from
the other reliable role's measured, spatially separated geometry. Empty role
points can only fall back to a verified world AABB half-extent with measured
current-frame support; standard deviations are not accepted as object extents.

The adapter preserves NULL/unseen/low-confidence failure states. SAM3 never
assigns roles. `slot_guided_sam3.run_camera` encodes the original RGB once, shares
that state between valid roles and retains all candidates. Explicit prompt,
depth and center gates reject wrong same-class instances even with high SAM
score. Failures select no mask; no box-shaped mask or unvalidated geometry is
substituted. Cross-view association is PR3.

```powershell
python run_slot_guided_sam3.py --frame snapshots/current.json --guidance guidance/current.json --backend sam3 --checkpoint /path/to/sam3.pt --device cuda --mode point+box --config configs/slot_guided_offline.json --output-dir artifacts/slot_sam3
python run_slot_guided_sam3.py --frame snapshots/current.json --guidance guidance/current.json --backend sam3 --checkpoint /path/to/sam3.pt --device cuda --mode box-only --config configs/slot_guided_offline.json --output-dir artifacts/slot_sam3
```

For model-free software checks, `--backend mock --mock-candidates masks.json`
accepts a manifest with exact episode_id/frame_id/source_frame_index and
`cameras:{camera:{target:{masks:["mask0.png"],scores:[0.8]},reference:...}}`.
Mocks return supplied masks; they are never reported as learned predictions.

## Artifacts and cache

`slot_guided_io.run_offline_frame` writes `<output>/<cache_key>/inference.json`,
checksum-linked NPZ containing **all** bool candidate masks at original RGB
resolution, and one role/mask/prompt diagnostic PNG per camera. Original RGB
pixels retain their scale in the left region of the diagnostic, with a readable
legend beside them. JSON contains every SAM score, composite score, gate
diagnostic, chosen index, failure and per-candidate mask content digest. Fusion
checks that digest before accepting a scored mask, so later mask mutation cannot
silently reuse its earlier acceptance decision.

The cache key hashes actual RGB, metric depth, guidance arrays and provenance,
per-frame calibration, prompt/selection settings, mode, model/checkpoint/backend
fingerprint and relevant code. A hit avoids encoding. Corrupt NPZ or stale key
fails explicitly. Cache is local and sequential; distributed locking and online
control are outside scope. Latency/peak CUDA allocated bytes describe the actual
backend. Mock CPU latency does not measure SAM3 performance.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/test_slot_prompt_adapter.py tests/test_slot_guided_sam3.py tests/test_slot_guided_io.py tests/test_slot_guided_config.py tests/compatibility/test_pipeline_baseline.py -q
```

The inspected local episode has packed RGB depth and instance masks, but no
current-frame calibration snapshot/near/far or handle-to-role annotation. Its
whole-episode pickle is deliberately not opened by this path. This machine also
lacks SAM3 weights and CUDA. Real mask IoU, role-instance accuracy, model latency
and VRAM remain unavailable; synthetic/Mock checks establish software behavior.

PR2 validation on 2026-11-05: cumulative PR1/PR2, existing camera and default
pipeline regression suite passed **100 tests and 130 subtests**. The real wrapper
was additionally checked with an injected lightweight model for frozen eval,
inference-mode, original PIL dimensions and exact reviewed prompt kwargs; that
test does not load SAM3 weights.
