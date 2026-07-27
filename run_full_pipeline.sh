#!/usr/bin/env bash
# Run the full episode pipeline end-to-end:
#   1) qwen_role_sam3_candidate_episode.py  -> outputs/<episode>/episode_candidates.json
#   2) multiview_candidate_fusion.py        -> outputs/<episode>/frame_fused_candidates.json
#   3) visualize_fused_candidates.py        -> outputs/<episode>/viz/*
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
#   CAMERA_PARAMS_JSON                      Optional explicit camera params (fusion + viz stages).
#   INVERT_RLBENCH_EXTRINSICS (default: 0)  Set to 1 to pass --invert-rlbench-extrinsics.
#   SKIP_CANDIDATES (default: 0)            Set to 1 to skip stage 1 (reuse existing episode_candidates.json).
#   SKIP_FUSION (default: 0)                Set to 1 to skip stage 2 (reuse existing frame_fused_candidates.json).
#   SKIP_VIZ (default: 0)                   Set to 1 to skip stage 3.
#   PYTHON (default: python)                Python interpreter to use.
#
# Example:
#   EPISODE_DIR=data/BridgeVLA_RLBench_EVAL_DATA/close_jar/all_variations/episodes/episode0 \
#   SAM_MODEL_DIR=/path/to/sam3_checkpoint \
#   CAMERA_THRESHOLD_OVERRIDES="left_shoulder=0.15,right_shoulder=0.15" \
#   ./run_full_pipeline.sh

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
  )
  [[ -n "${CAMERAS}" ]] && STAGE2_ARGS+=(--cameras "${CAMERAS}")
  [[ -n "${CAMERA_PARAMS_JSON}" ]] && STAGE2_ARGS+=(--camera-params-json "${CAMERA_PARAMS_JSON}")
  [[ "${INVERT_RLBENCH_EXTRINSICS}" == "1" ]] && STAGE2_ARGS+=(--invert-rlbench-extrinsics)
  "${PYTHON}" "${SCRIPT_DIR}/multiview_candidate_fusion.py" "${STAGE2_ARGS[@]}"
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
echo "=========================================="
