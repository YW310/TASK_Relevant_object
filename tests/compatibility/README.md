# Compatibility baseline

This directory freezes externally observable behavior before implementation is
moved into `src/relevant_object/`.

- `capture_pipeline_baseline.py` runs `run_full_pipeline.sh` with
  `PYTHON=echo`, so no model is loaded and no pipeline artifact is written.
- `pipeline_invocations.json` records normalized Python commands for default,
  all-stage, reuse, optional-value, and path-with-spaces scenarios.
- `pipeline_environment_contract.json` inventories shell defaults, stage use,
  CLI flags, and empty-value behavior.
- `fixtures/` contains minimal versioned artifacts joined by one fixed
  `generation_id`.

Refresh snapshots only after reviewing a deliberate compatibility change:

```bash
python tests/compatibility/capture_pipeline_baseline.py --write
python tests/compatibility/generate_geometry_fixture.py
```

