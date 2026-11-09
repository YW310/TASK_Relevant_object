# SAM3 interactive API and runtime compatibility

On 2026-11-05 the official `facebookresearch/sam3` source was cloned read-only
for API inspection, without installing SAM3 or downloading model weights.
The reviewed revision is
[`f1647e8d0c1d280991f4d1b7d974c2704537fe66`](https://github.com/facebookresearch/sam3/tree/f1647e8d0c1d280991f4d1b7d974c2704537fe66).
This verifies source contracts; it is not an installed-model smoke test.

The inspected commit is a reference for the table below, not an installation
requirement. The fixed source-hash/revision gate was removed on 2026-11-09 at the
user's request. An installed implementation is no longer rejected because its
source differs from that reference, including harmless edits.

The backend still requires instance interactivity and inspects the actual
predictor signature for point/box arguments and `normalize_coords`. It checks
the encoded original image dimensions and validates original-resolution mask
shapes, finite outputs and candidate geometry. Model/checkpoint loading errors
remain explicit. These checks do not certify every SAM3 release; verify a
representative calibrated frame when changing the implementation.

LF-normalized hashes of available source files are recorded only as cache
provenance, without comparison to expected hashes. Missing/unreadable source
files do not block startup. The fingerprint records the actual package root,
package version and `source_hash_enforced=false`; it no longer reports the
reference commit as the installed implementation. Checkpoint hashes, mask
content hashes and cache-integrity checks remain unchanged.

The builder uses an explicit local checkpoint, `load_from_HF=False`,
`eval_mode=True`, `enable_inst_interactivity=True`.

| Source | Verified behavior |
| --- | --- |
| `sam3/model/sam3_image_processor.py:42-73` | `set_image(PIL.Image)` records original height/width and computes the image backbone once; supplies SAM2 features for instance interactivity. The ndarray branch uses `image.shape[-2:]`, so passing HWC ndarray would record incorrect dimensions. The new backend always converts the original uint8 RGB to PIL without resizing. |
| `sam3/model/sam3_image.py:628-665` | `predict_inst(inference_state, **kwargs)` reuses cached backbone features, reconstructs the interactive predictor state, calls `inst_interactive_predictor.predict(**kwargs)`, and clears temporary features. Sequential target/reference calls share the same image state. |
| `sam3/model/sam1_task_predictor.py:235-300` | `predict(point_coords, point_labels, box, mask_input, multimask_output, return_logits, normalize_coords)` accepts original-image XY points `[N,2]`, labels `[N]` (1 positive/0 negative), and pixel XYXY box `[4]`. Output masks `[C,H,W]` use the original image size. |
| `sam3/model/sam1_task_predictor.py:302-334,385-404` | `_prep_prompts` transforms points and boxes; `_predict` concatenates box-corner labels 2/3 and point prompts, proving combined box+point support. |
| `sam3/model/utils/sam1_utils.py:49-78` | With explicit `normalize_coords=True`, original pixel X is divided by original width and Y by original height, then scaled to model resolution. The adapter must not pre-normalize or pre-resize prompts. |

Both modes request `multimask_output=True`, `return_logits=False`. Box-only
passes no point prompts; point+box concatenates verified positive and negative
points. All returned masks/scores are retained. Candidate acceptance gates on
prompt coverage, box agreement, depth coverage, role center agreement and
projected measured guidance support. SAM quality contributes only 5% of the
composite score and cannot bypass geometry gates. Cross-view agreement is a
separate PR3 stage. Thresholds are explicit experiment configuration, not
calibrated quality probabilities.

`eval()` and `requires_grad_(False)` freeze model parameters. Both image encoding
and mask prediction run in `torch.inference_mode()` and the existing
`sam3_runtime.autocast_context`. Output masks/scores use the existing runtime
normalizers and are detached CPU arrays. Importing or testing the Mock backend
does not import `torch`, `sam3` or `sam3_runtime`.

Failure falls back to **no selected SAM mask**, preserving the guidance status
and every returned candidate with rejection reasons. It never substitutes an
unverified box/mask, confuses invisibility with reference NULL, or changes the
existing Qwen/SAM3 entry points.

Real checkpoint inference and GPU latency/VRAM measurements remain unmeasured
until a compatible SAM3 installation, accessible weights and representative
current-frame RGB-D inputs are supplied. CUDA timings synchronize and peak VRAM
is `torch.cuda.max_memory_allocated` (allocated bytes, not process reservation).
