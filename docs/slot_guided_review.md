# Slot-guided offline path: post-implementation review

Review date: 2026-11-05 (Asia/Shanghai). This follow-up reviews software,
configuration, reproducible commands and report consistency. Real SAM3,
BridgeVLA checkpoint and paired RLBench experiments remain unexecuted.

## Changes from review

- Added `configs/slot_guided_offline.json` and `slot_guided_config.py`. The
  checked-in values exactly match previous dataclass defaults. All three offline
  CLIs accept `--config`; effective settings and digests are retained in reports.
  Invalid fields/types/thresholds, duplicate keys and model-activation settings
  fail validation. Configuration inspection opens no observations or models.
- Final `sam3_mask_iou` now evaluates masks chosen by joint multi-view geometry.
  `camera_local_mask_iou` separately records the earlier per-camera selection.
  This fixes disagreement between reported mask quality and saved object geometry.
- Each candidate stores a mask-content SHA256 when scored. Fusion checks it
  before using an accepted mask, rejecting mutation after prompt/geometry scoring.
  This is a consistency check, not authentication of an untrusted producer.
- C baseline reports now require exact frame/time/calibration, producer Git and
  checkpoint provenance and nine explicit, typed metric records with checked
  units. Unavailable values retain null/count-zero/reason. Importing external
  measurements still does not authorize real action conditioning.
- Invalid programmatic backend names and missing SAM3 checkpoint arguments fail
  explicitly. Re-running the demo on nonempty inputs fails before changing them.
- Added README entry points, bilingual quickstart and current config/command
  examples. The independent legacy Qwen/SAM entry points remain unchanged.

The related BridgeVLA branch retains `enable_slot_guided_sam3=False`. Its
dedicated draft YAML and documentation identify the supported existing Anchor
architecture; no current-frame importer or REAL acceptance is implied by loading
that YAML. See the [action draft guide](https://github.com/YW310/BridgeVLA/blob/codex/slot-guided-sam3-action-draft/docs/slot_guided_sam3_action_draft.md).

## Reproduction

Use the prepared-environment block in the [quickstart](slot_guided_quickstart.md).
From this repository root:

```powershell
& $Python slot_guided_config.py --config configs/slot_guided_offline.json
& $Python slot_guided_synthetic_demo.py --config configs/slot_guided_offline.json --output-dir $RunDir
& $Python -m pytest tests/test_slot_guidance.py tests/test_slot_guidance_geometry.py tests/test_slot_prompt_adapter.py tests/test_slot_guided_sam3.py tests/test_slot_guided_io.py tests/test_slot_guidance_producers.py tests/test_slot_geometry_fusion.py tests/test_slot_geometry_experiments.py tests/test_slot_guided_config.py tests/test_camera_geometry.py tests/compatibility/test_pipeline_baseline.py -q
& $Python -m pytest tests/test_multiview_candidate_fusion.py -q
```

The quickstart also exercises standalone frame inference twice (second run hits
cache), an A/B manifest rerun and CLI help. Outputs include configuration SHA256,
source/input/candidate digests, JSON/CSV metrics, masks, object geometry and
diagnostic PNGs. Source HEAD alone does not capture dirty workspace changes.

## Evidence boundary

Final software validation for this review:

| Check | Result |
| --- | --- |
| Full perception/config/candidate/cache/camera/default-contract command above, with optional local source and Bridge comparisons enabled | 166 passed, 198 subtests passed |
| Existing fusion suite in a separate process | 37 passed |
| Related BridgeVLA draft/config/Anchor/receipt/fingerprint core suite | 130 passed |
| Additional BridgeVLA docs/training/config checks | 39 passed, 3 subtests passed |
| Windows quickstart | Config inspection, synthetic demo, standalone cache miss/hit, A/B rerun and four CLI help commands executed successfully |

The checked-in default config resolves to SHA256
`9506d3699551be7a14f755861fd3ddaf842be5638288c884f8618b6b385fdbbb`.
The optional official SAM3 source-contract check uses `SAM3_AUDIT_ROOT`; it
does not require a model installation or establish successful weight inference.

Software tests and supplied Mock masks do not measure model quality or robot
performance. The synthetic report remains `synthetic_mock` with
`real_offline_acceptance_passed=false`. Current runtime verification still finds
SAM3 absent from the prepared Python environment. Real calibrated snapshots,
role annotations, weights/runtime, actual Action predictions and paired rollouts
are needed before real acceptance and runtime activation.

The previous wider legacy checks found one Qwen wheel-prompt expectation failure
and two BridgeVLA reduced-mode expectation failures, reproduced on unchanged
baselines. Those defaults were not changed. Passing the targeted commands does
not mean the entire repository test suite passes.
