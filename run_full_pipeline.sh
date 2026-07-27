#!/usr/bin/env bash
# Run the full episode pipeline end-to-end:
#   1) qwen_role_sam3_candidate_episode.py  -> outputs/<episode>/episode_candidates.json
#   2) multiview_candidate_fusion.py        -> outputs/<episode>/frame_fused_candidates.json
#   3) visualize_fused_candidates.py        -> outputs/<episode>/viz/*
#   4) qwen3vl_object_role_decision.py      -> outputs/<episode>/object_predictions.json (optional)
#
# Configure via environment variables (required ones must be set):
#   EPISODE_DIR (required)        RLBench episode directory, e.g. data/.../episode0
#   SAM_MODEL_DIR (required)      SAM3 checkpoint directory (unless SKIP_CANDIDATES=1)
#
#   OUTPUT_ROOT (default: outputs)          Root for outputs/<episode name>/...
#   OUTPUT_DIR                              Explicit output dir override.
#   MODEL_PATH                              Qwen3-VL model path override.
#   INSTRUCTION                             Explicit instruction override.
#   ROLE_SPEC_JSON                          Reuse an existing role_spec.json (skips Qwen stage 1).
#   CAMERAS                                 Comma-separated camera subset, e.g. "front,left_shoulder".
#   FRAME_INTERVAL (default: 1)             Process one frame every N source frames.
#   MAX_FRAMES                              Optional cap on number of frames.
#   DEVICE (default: cuda)                  cuda|cpu for the SAM3/Qwen stage.
#   USE_BF16 (default: 0)                   Set to 1 to enable bf16 autocast.
#   THRESHOLD (default: 0.25)               SAM3 confidence threshold.
#   CAMERA_THRESHOLD_OVERRIDES              e.g. "left_shoulder=0.15,right_shoulder=0.15".
#   CLUSTER_DISTANCE_M (default: 0.03)      Fusion clustering centroid distance threshold.
#   MIN_FUSED_POINTS (default: 0)           Drop fused objects with fewer combined points than this (0=off).
#   MIN_BBOX_DIAGONAL_M (default: 0.0)      Drop fused objects with a smaller 3D bbox diagonal than this (0=off).
#   SAVE_OBJECT_SUMMARY (default: 0)        Set to 1 to export object_summary.json for downstream Qwen3-VL role decisions.
#   OBJECT_SUMMARY_JSON                      Optional explicit object summary output path.
#   SKIP_DECISION (default: 1)              Set to 0 to run stage 4 object-level target/reference decision.
#   DECISION_MODEL_PATH                      Qwen3-VL model path for stage 4 (default: MODEL_PATH or script default).
#   DECISION_FRAME (default: last)          first|last decision frame from object_summary.
#   DECISION_FRAME_ID                        Optional explicit frame_id for decision.
#   DECISION_OUTPUT_JSON                     Optional explicit output path for object_predictions.json.
#   MAX_CANDIDATE_IMAGES (default: 8)       Max representative object images attached to decision prompt.
#   CAMERA_PARAMS_JSON                      Optional explicit camera params (fusion + viz stages).
#   INVERT_RLBENCH_EXTRINSICS (default: 0)  Set to 1 to pass --invert-rlbench-extrinsics.
#   SKIP_CANDIDATES (default: 0)            Set to 1 to skip stage 1 (reuse existing episode_candidates.json).
#   SKIP_FUSION (default: 0)                Set to 1 to skip stage 2 (reuse existing frame_fused_candidates.json).
#   SKIP_VIZ (default: 0)                   Set to 1 to skip stage 3.
#   PYTHON (default: python)                Python interpreter to use.
#
# Example:
  # EPISODE_DIR=data/BridgeVLA_RLBench_EVAL_DATA/push_buttons/all_variations/episodes/episode0 \
  # SAM_MODEL_DIR=/common-data-32t/.cache/facebook/sam3 \
  # MODEL_PATH=/new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct \
  # CAMERA_THRESHOLD_OVERRIDES="left_shoulder=0.15,right_shoulder=0.15" \
  # ./run_full_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

EPISODE_DIR="${EPISODE_DIR:-}"
SAM_MODEL_DIR="${SAM_MODEL_DIR:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
INSTRUCTION="${INSTRUCTION:-}"
ROLE_SPEC_JSON="${ROLE_SPEC_JSON:-}"
CAMERAS="${CAMERAS:-}"
FRAME_INTERVAL="${FRAME_INTERVAL:-1}"
MAX_FRAMES="${MAX_FRAMES:-}"
DEVICE="${DEVICE:-cuda}"
USE_BF16="${USE_BF16:-0}"
THRESHOLD="${THRESHOLD:-0.25}"
CAMERA_THRESHOLD_OVERRIDES="${CAMERA_THRESHOLD_OVERRIDES:-}"
CLUSTER_DISTANCE_M="${CLUSTER_DISTANCE_M:-0.03}"
MIN_FUSED_POINTS="${MIN_FUSED_POINTS:-0}"
MIN_BBOX_DIAGONAL_M="${MIN_BBOX_DIAGONAL_M:-0.0}"
SAVE_OBJECT_SUMMARY="${SAVE_OBJECT_SUMMARY:-0}"
OBJECT_SUMMARY_JSON="${OBJECT_SUMMARY_JSON:-}"
SKIP_DECISION="${SKIP_DECISION:-1}"
DECISION_MODEL_PATH="${DECISION_MODEL_PATH:-${MODEL_PATH:-}}"
DECISION_FRAME="${DECISION_FRAME:-last}"
DECISION_FRAME_ID="${DECISION_FRAME_ID:-}"
DECISION_OUTPUT_JSON="${DECISION_OUTPUT_JSON:-}"
MAX_CANDIDATE_IMAGES="${MAX_CANDIDATE_IMAGES:-8}"
CAMERA_PARAMS_JSON="${CAMERA_PARAMS_JSON:-}"
INVERT_RLBENCH_EXTRINSICS="${INVERT_RLBENCH_EXTRINSICS:-0}"
SKIP_CANDIDATES="${SKIP_CANDIDATES:-0}"
SKIP_FUSION="${SKIP_FUSION:-0}"
SKIP_VIZ="${SKIP_VIZ:-0}"

if [[ -z "${EPISODE_DIR}" ]]; then
  echo "Set EPISODE_DIR to the RLBench episode folder." >&2
  exit 2
fi
if [[ -z "${SAM_MODEL_DIR}" && "${SKIP_CANDIDATES}" != "1" ]]; then
  echo "Set SAM_MODEL_DIR to the SAM3 checkpoint directory (or SKIP_CANDIDATES=1)." >&2
  exit 2
fi

EPISODE_NAME="$(basename "${EPISODE_DIR}")"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/${EPISODE_NAME}"
fi

CANDIDATES_JSON="${OUTPUT_DIR}/episode_candidates.json"
FUSED_JSON="${OUTPUT_DIR}/frame_fused_candidates.json"
VIZ_DIR="${OUTPUT_DIR}/viz"
OBJECT_SUMMARY_PATH_DEFAULT="${OUTPUT_DIR}/object_summary.json"
OBJECT_PREDICTIONS_PATH_DEFAULT="${OUTPUT_DIR}/object_predictions.json"

echo "=========================================="
echo "Episode:    ${EPISODE_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Stage 1: Qwen role identification + SAM3 per-view candidate generation
# ---------------------------------------------------------------------------
if [[ "${SKIP_CANDIDATES}" == "1" ]]; then
  echo "[stage 1/3] Skipping candidate generation (SKIP_CANDIDATES=1)."
else
  echo "[stage 1/3] Generating role spec + SAM3 candidates..."
  STAGE1_ARGS=(
    --episode-dir "${EPISODE_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --sam-model-dir "${SAM_MODEL_DIR}"
    --device "${DEVICE}"
    --threshold "${THRESHOLD}"
    --frame-interval "${FRAME_INTERVAL}"
    --resume
  )
  [[ "${USE_BF16}" != "1" ]] && STAGE1_ARGS+=(--no-bf16)
  [[ -n "${MODEL_PATH}" ]] && STAGE1_ARGS+=(--model-path "${MODEL_PATH}")
  [[ -n "${INSTRUCTION}" ]] && STAGE1_ARGS+=(--instruction "${INSTRUCTION}")
  [[ -n "${ROLE_SPEC_JSON}" ]] && STAGE1_ARGS+=(--role-spec-json "${ROLE_SPEC_JSON}")
  [[ -n "${CAMERAS}" ]] && STAGE1_ARGS+=(--cameras "${CAMERAS}")
  [[ -n "${MAX_FRAMES}" ]] && STAGE1_ARGS+=(--max-frames "${MAX_FRAMES}")
  [[ -n "${CAMERA_THRESHOLD_OVERRIDES}" ]] && STAGE1_ARGS+=(--camera-threshold-overrides "${CAMERA_THRESHOLD_OVERRIDES}")
  "${PYTHON}" "${SCRIPT_DIR}/qwen_role_sam3_candidate_episode.py" "${STAGE1_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 2: Multi-view fusion into per-frame 3D objects
# ---------------------------------------------------------------------------
if [[ "${SKIP_FUSION}" == "1" ]]; then
  echo "[stage 2/3] Skipping multiview fusion (SKIP_FUSION=1)."
else
  echo "[stage 2/3] Fusing multi-view candidates into 3D objects..."
  STAGE2_ARGS=(
    --episode-dir "${EPISODE_DIR}"
    --candidates-json "${CANDIDATES_JSON}"
    --output-json "${FUSED_JSON}"
    --cluster-distance-m "${CLUSTER_DISTANCE_M}"
    --min-fused-points "${MIN_FUSED_POINTS}"
    --min-bbox-diagonal-m "${MIN_BBOX_DIAGONAL_M}"
  )
  # Stage 4 depends on object_summary; auto-enable summary export when decision is requested.
  [[ "${SAVE_OBJECT_SUMMARY}" == "1" || "${SKIP_DECISION}" == "0" ]] && STAGE2_ARGS+=(--save-object-summary)
  [[ -n "${OBJECT_SUMMARY_JSON}" ]] && STAGE2_ARGS+=(--object-summary-json "${OBJECT_SUMMARY_JSON}")
  [[ -n "${CAMERAS}" ]] && STAGE2_ARGS+=(--cameras "${CAMERAS}")
  [[ -n "${CAMERA_PARAMS_JSON}" ]] && STAGE2_ARGS+=(--camera-params-json "${CAMERA_PARAMS_JSON}")
  [[ "${INVERT_RLBENCH_EXTRINSICS}" == "1" ]] && STAGE2_ARGS+=(--invert-rlbench-extrinsics)
  "${PYTHON}" "${SCRIPT_DIR}/multiview_candidate_fusion.py" "${STAGE2_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 4: Qwen3-VL object-level role decision (optional)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DECISION}" == "1" ]]; then
  echo "[stage 4/4] Skipping object-level decision (SKIP_DECISION=1)."
else
  echo "[stage 4/4] Running Qwen3-VL object-level target/reference decision..."
  SUMMARY_INPUT="${OBJECT_SUMMARY_JSON:-${OBJECT_SUMMARY_PATH_DEFAULT}}"
  STAGE4_ARGS=(
    --object-summary-json "${SUMMARY_INPUT}"
    --decision-frame "${DECISION_FRAME}"
    --max-candidate-images "${MAX_CANDIDATE_IMAGES}"
  )
  [[ -n "${DECISION_MODEL_PATH}" ]] && STAGE4_ARGS+=(--model-path "${DECISION_MODEL_PATH}")
  [[ -n "${DECISION_FRAME_ID}" ]] && STAGE4_ARGS+=(--decision-frame-id "${DECISION_FRAME_ID}")
  [[ -n "${DECISION_OUTPUT_JSON}" ]] && STAGE4_ARGS+=(--output-json "${DECISION_OUTPUT_JSON}")
  "${PYTHON}" "${SCRIPT_DIR}/qwen3vl_object_role_decision.py" "${STAGE4_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 3: Sanity-check visualization
# ---------------------------------------------------------------------------
if [[ "${SKIP_VIZ}" == "1" ]]; then
  echo "[stage 3/3] Skipping visualization (SKIP_VIZ=1)."
else
  echo "[stage 3/3] Rendering sanity-check visualizations..."
  STAGE3_ARGS=(
    --fused-json "${FUSED_JSON}"
    --output-dir "${VIZ_DIR}"
  )
  [[ -n "${CAMERAS}" ]] && STAGE3_ARGS+=(--cameras "${CAMERAS}")
  [[ -n "${CAMERA_PARAMS_JSON}" ]] && STAGE3_ARGS+=(--camera-params-json "${CAMERA_PARAMS_JSON}")
  [[ "${INVERT_RLBENCH_EXTRINSICS}" == "1" ]] && STAGE3_ARGS+=(--invert-rlbench-extrinsics)
  "${PYTHON}" "${SCRIPT_DIR}/visualize_fused_candidates.py" "${STAGE3_ARGS[@]}"
fi

echo "=========================================="
echo "Done. Outputs under: ${OUTPUT_DIR}"
echo "  Candidates: ${CANDIDATES_JSON}"
echo "  Fused:      ${FUSED_JSON}"
echo "  Viz:        ${VIZ_DIR}"
if [[ "${SAVE_OBJECT_SUMMARY}" == "1" || "${SKIP_DECISION}" == "0" ]]; then
  echo "  ObjSummary: ${OBJECT_SUMMARY_JSON:-${OBJECT_SUMMARY_PATH_DEFAULT}}"
fi
if [[ "${SKIP_DECISION}" == "0" ]]; then
  echo "  Decision:   ${DECISION_OUTPUT_JSON:-${OBJECT_PREDICTIONS_PATH_DEFAULT}}"
fi
echo "=========================================="
