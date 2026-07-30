#!/usr/bin/env bash
# Run the full episode pipeline end-to-end:
#   1) qwen_role_sam3_candidate_episode.py  -> outputs/<episode>/episode_candidates.json
#   2) multiview_candidate_fusion.py        -> outputs/<episode>/frame_fused_candidates.json
#   3) visualize_fused_candidates.py        -> outputs/<episode>/viz/*
#   4) qwen3vl_object_role_decision.py      -> outputs/<episode>/object_predictions.json (optional)
#   5) stage4_visualize_decision.py         -> outputs/<episode>/viz_decision/* (optional)
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
#   MASK_NMS_IOU (default: 0.80)             Stage-1 same-role NMS and cross-role canonical mask IoU.
#   CANONICAL_CONTAINMENT (default: 0.90)    Stage-1 smaller-mask coverage for canonical observations.
#   CANONICAL_BBOX_IOU (default: 0.0)        Optional Stage-1 bbox IoU support (0=off).
#   CLUSTER_DISTANCE_M (default: 0.03)      Fusion clustering centroid distance threshold.
#   LEGACY_CANONICAL_IOU (default: 0.35)     Stage-2 IoU for deduplicating legacy candidate JSON.
#   LEGACY_CANONICAL_CONTAINMENT (default: 0.50) Stage-2 smaller-mask coverage for legacy JSON.
#   MIN_FUSED_POINTS (default: 0)           Drop fused objects with fewer combined points than this (0=off).
#   MIN_BBOX_DIAGONAL_M (default: 0.0)      Drop fused objects with a smaller 3D bbox diagonal than this (0=off).
#   SAVE_OBJECT_SUMMARY (default: 0)        Set to 1 to export object_summary.json for downstream Qwen3-VL role decisions.
#   OBJECT_SUMMARY_JSON                      Optional explicit object summary output path.
#   SKIP_DECISION (default: 1)              Set to 0 to run stage 4 object-level target/reference decision.
#   DECISION_MODEL_PATH                      Qwen3-VL model path for stage 4 (default: MODEL_PATH or script default).
#   DECISION_FRAME (default: last)          first|last decision frame from object_summary.
#   DECISION_FRAME_ID                        Optional explicit frame_id for decision.
#   DECISION_WINDOW_FRAMES (default: 3)     Current t plus [t-2,t-1] ([t-2,t-1,t]), in one model call.
#   DECISION_MAX_NEW_TOKENS (default: 1024) Stage-4 JSON generation budget per frame.
#   DECISION_OUTPUT_JSON                     Optional explicit output path for object_predictions.json.
#   MAX_CANDIDATE_IMAGES (default: 8)       Max representative object images attached to decision prompt.
#   MAX_CANDIDATES_FOR_DECISION (default: 12) Max candidates kept for stage-4 prompt after filtering.
#   MIN_CANDIDATE_POINT_COUNT (default: 0)   Drop tiny candidates by point count before stage-4 decision.
#   MIN_CANDIDATE_CAMERA_COUNT (default: 1)  Require at least this many supporting cameras.
#   MIN_CANDIDATE_SAM_SCORE (default: 0.0)   Drop low-score candidates before stage-4 decision.
#   MAX_EE_DISTANCE_M                         Optional filter by min end-effector distance across the temporal window.
#   SKIP_DECISION_VIZ (default: 0)          Set to 1 to skip Stage 5 decision visualization.
#   DECISION_VIZ_OUTPUT_DIR                  Optional explicit output dir for decision overlays.
#   SKIP_STAGE_COMPARE (default: 1)         Set to 0 to render compact Stage 3 vs Stage 5 montages.
#   STAGE_COMPARE_OUTPUT_DIR                Optional explicit output dir for stage comparison montages.
#   CAMERA_PARAMS_JSON                      Optional explicit camera params (fusion + viz stages).
#   INVERT_RLBENCH_EXTRINSICS (default: 0)  Set to 1 to pass --invert-rlbench-extrinsics.
#   SKIP_CANDIDATES (default: 0)            Set to 1 to skip stage 1 (reuse existing episode_candidates.json).
#   SKIP_FUSION (default: 0)                Set to 1 to skip stage 2 (reuse existing frame_fused_candidates.json).
#   SKIP_VIZ (default: 0)                   Set to 1 to skip stage 3.
#   PYTHON (default: python)                Python interpreter to use.
#
# Every tunable downstream CLI option is exposed. Less-common variables:
#   Stage 1: INSTRUCTION_FILE, INSTRUCTION_INDEX, START_FRAME, END_FRAME,
#     ROLE_FRAME, GROUNDING_MIN_SIDE, MAX_RETRIES, SAM_CHECKPOINT,
#     COMPILE_MODEL, TOP_K_PER_ROLE, CANDIDATE_POOL_SIZE, MIN_MASK_AREA,
#     PROMPT_VARIANTS, CANDIDATE_MASK_ALPHA, SAVE_FRAME_CONTACT_SHEET,
#     VISUALIZATION_CELL_WIDTH, CANDIDATE_RESUME, CANDIDATE_PROGRESS,
#     CANDIDATE_DRY_RUN.
#   Stage 2: CANDIDATES_JSON, FUSED_JSON, RLBENCH_LOW_DIM_OBS, DEPTH_SCALE,
#     DEPTH_MODE, MAX_POINTS_PER_CANDIDATE, BBOX_IOU_THRESHOLD,
#     NEAREST_DISTANCE_M, MAX_HYPOTHESIS_DIAMETER_M, MAX_SIZE_RATIO,
#     LEGACY_UNION_FIND, TRACK_DISTANCE_M.
#   Stage 3: VIZ_DIR, VIZ_FRAME_IDS, VIZ_POINT_STRIDE, VIZ_POINT_RADIUS,
#     VIZ_MASK_ALPHA, VIZ_MAX_FRAMES, VIZ_SKIP_POINTCLOUD.
#   Stage 4: DECISION_GROUNDING_MIN_SIDE, DECISION_MAX_RETRIES,
#     DECISION_DRY_RUN (plus the DECISION_* variables above).
#   Stage 5: DECISION_VIZ_POINT_STRIDE, DECISION_VIZ_POINT_RADIUS,
#     DECISION_VIZ_MASK_ALPHA.
#   Stage 6: STAGE1_CANDIDATES_JSON, STAGE_COMPARE_PANEL_GAP,
#     STAGE_COMPARE_LABEL_HEIGHT, STAGE_COMPARE_SUMMARY_WIDTH,
#     STAGE_COMPARE_BACKGROUND.
#
# Example:
  # EPISODE_DIR=data/BridgeVLA_RLBench_EVAL_DATA/push_buttons/all_variations/episodes/episode0 \
  # SAM_MODEL_DIR=/common-data-32t/.cache/facebook/sam3 \
  # FRAME_INTERVAL=10 \
  # MODEL_PATH=/new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct \
  # CAMERA_THRESHOLD_OVERRIDES="left_shoulder=0.15,right_shoulder=0.15" \
  # SKIP_DECISION=0 \
  # SAVE_OBJECT_SUMMARY=1 \
  # DECISION_MODEL_PATH=/new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct \
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
INSTRUCTION_FILE="${INSTRUCTION_FILE:-}"
INSTRUCTION_INDEX="${INSTRUCTION_INDEX:-0}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:-}"
ROLE_FRAME="${ROLE_FRAME:-first}"
GROUNDING_MIN_SIDE="${GROUNDING_MIN_SIDE:-512}"
MAX_RETRIES="${MAX_RETRIES:-1}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
TOP_K_PER_ROLE="${TOP_K_PER_ROLE:-8}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-20}"
MIN_MASK_AREA="${MIN_MASK_AREA:-4}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-5}"
MASK_NMS_IOU="${MASK_NMS_IOU:-0.80}"
CANONICAL_CONTAINMENT="${CANONICAL_CONTAINMENT:-0.90}"
CANONICAL_BBOX_IOU="${CANONICAL_BBOX_IOU:-0.0}"
CANDIDATE_MASK_ALPHA="${CANDIDATE_MASK_ALPHA:-105}"
SAVE_FRAME_CONTACT_SHEET="${SAVE_FRAME_CONTACT_SHEET:-1}"
VISUALIZATION_CELL_WIDTH="${VISUALIZATION_CELL_WIDTH:-384}"
CANDIDATE_RESUME="${CANDIDATE_RESUME:-1}"
CANDIDATE_PROGRESS="${CANDIDATE_PROGRESS:-1}"
CANDIDATE_DRY_RUN="${CANDIDATE_DRY_RUN:-0}"
CLUSTER_DISTANCE_M="${CLUSTER_DISTANCE_M:-0.03}"
LEGACY_CANONICAL_IOU="${LEGACY_CANONICAL_IOU:-0.35}"
LEGACY_CANONICAL_CONTAINMENT="${LEGACY_CANONICAL_CONTAINMENT:-0.50}"
RLBENCH_LOW_DIM_OBS="${RLBENCH_LOW_DIM_OBS:-}"
DEPTH_SCALE="${DEPTH_SCALE:-1.0}"
DEPTH_MODE="${DEPTH_MODE:-auto}"
MAX_POINTS_PER_CANDIDATE="${MAX_POINTS_PER_CANDIDATE:-4096}"
BBOX_IOU_THRESHOLD="${BBOX_IOU_THRESHOLD:-0.0}"
NEAREST_DISTANCE_M="${NEAREST_DISTANCE_M:-}"
MAX_HYPOTHESIS_DIAMETER_M="${MAX_HYPOTHESIS_DIAMETER_M:-0.50}"
MAX_SIZE_RATIO="${MAX_SIZE_RATIO:-4.0}"
LEGACY_UNION_FIND="${LEGACY_UNION_FIND:-0}"
TRACK_DISTANCE_M="${TRACK_DISTANCE_M:-0.15}"
MIN_FUSED_POINTS="${MIN_FUSED_POINTS:-0}"
MIN_BBOX_DIAGONAL_M="${MIN_BBOX_DIAGONAL_M:-0.0}"
SAVE_OBJECT_SUMMARY="${SAVE_OBJECT_SUMMARY:-0}"
OBJECT_SUMMARY_JSON="${OBJECT_SUMMARY_JSON:-}"
SKIP_DECISION="${SKIP_DECISION:-1}"
DECISION_MODEL_PATH="${DECISION_MODEL_PATH:-${MODEL_PATH:-}}"
DECISION_FRAME="${DECISION_FRAME:-last}"
DECISION_FRAME_ID="${DECISION_FRAME_ID:-}"
DECISION_WINDOW_FRAMES="${DECISION_WINDOW_FRAMES:-3}"
DECISION_MAX_NEW_TOKENS="${DECISION_MAX_NEW_TOKENS:-1024}"
DECISION_GROUNDING_MIN_SIDE="${DECISION_GROUNDING_MIN_SIDE:-512}"
DECISION_MAX_RETRIES="${DECISION_MAX_RETRIES:-1}"
DECISION_DRY_RUN="${DECISION_DRY_RUN:-0}"
DECISION_OUTPUT_JSON="${DECISION_OUTPUT_JSON:-}"
MAX_CANDIDATE_IMAGES="${MAX_CANDIDATE_IMAGES:-8}"
MAX_CANDIDATES_FOR_DECISION="${MAX_CANDIDATES_FOR_DECISION:-12}"
MIN_CANDIDATE_POINT_COUNT="${MIN_CANDIDATE_POINT_COUNT:-0}"
MIN_CANDIDATE_CAMERA_COUNT="${MIN_CANDIDATE_CAMERA_COUNT:-1}"
MIN_CANDIDATE_SAM_SCORE="${MIN_CANDIDATE_SAM_SCORE:-0.0}"
MAX_EE_DISTANCE_M="${MAX_EE_DISTANCE_M:-}"
SKIP_DECISION_VIZ="${SKIP_DECISION_VIZ:-0}"
DECISION_VIZ_OUTPUT_DIR="${DECISION_VIZ_OUTPUT_DIR:-}"
SKIP_STAGE_COMPARE="${SKIP_STAGE_COMPARE:-1}"
STAGE_COMPARE_OUTPUT_DIR="${STAGE_COMPARE_OUTPUT_DIR:-}"
CAMERA_PARAMS_JSON="${CAMERA_PARAMS_JSON:-}"
INVERT_RLBENCH_EXTRINSICS="${INVERT_RLBENCH_EXTRINSICS:-0}"
SKIP_CANDIDATES="${SKIP_CANDIDATES:-0}"
SKIP_FUSION="${SKIP_FUSION:-0}"
SKIP_VIZ="${SKIP_VIZ:-0}"
VIZ_FRAME_IDS="${VIZ_FRAME_IDS:-}"
VIZ_POINT_STRIDE="${VIZ_POINT_STRIDE:-4}"
VIZ_POINT_RADIUS="${VIZ_POINT_RADIUS:-2}"
VIZ_MASK_ALPHA="${VIZ_MASK_ALPHA:-80}"
VIZ_MAX_FRAMES="${VIZ_MAX_FRAMES:-}"
VIZ_SKIP_POINTCLOUD="${VIZ_SKIP_POINTCLOUD:-0}"
DECISION_VIZ_POINT_STRIDE="${DECISION_VIZ_POINT_STRIDE:-4}"
DECISION_VIZ_POINT_RADIUS="${DECISION_VIZ_POINT_RADIUS:-2}"
DECISION_VIZ_MASK_ALPHA="${DECISION_VIZ_MASK_ALPHA:-90}"
STAGE1_CANDIDATES_JSON="${STAGE1_CANDIDATES_JSON:-}"
STAGE_COMPARE_PANEL_GAP="${STAGE_COMPARE_PANEL_GAP:-8}"
STAGE_COMPARE_LABEL_HEIGHT="${STAGE_COMPARE_LABEL_HEIGHT:-26}"
STAGE_COMPARE_SUMMARY_WIDTH="${STAGE_COMPARE_SUMMARY_WIDTH:-360}"
STAGE_COMPARE_BACKGROUND="${STAGE_COMPARE_BACKGROUND:-white}"

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

CANDIDATES_JSON="${CANDIDATES_JSON:-${OUTPUT_DIR}/episode_candidates.json}"
FUSED_JSON="${FUSED_JSON:-${OUTPUT_DIR}/frame_fused_candidates.json}"
VIZ_DIR="${VIZ_DIR:-${OUTPUT_DIR}/viz}"
OBJECT_SUMMARY_PATH_DEFAULT="${OUTPUT_DIR}/object_summary.json"
OBJECT_PREDICTIONS_PATH_DEFAULT="${OUTPUT_DIR}/object_predictions.json"
DECISION_VIZ_DIR_DEFAULT="${OUTPUT_DIR}/viz_decision"

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
    --instruction-index "${INSTRUCTION_INDEX}"
    --start "${START_FRAME}"
    --role-frame "${ROLE_FRAME}"
    --grounding-min-side "${GROUNDING_MIN_SIDE}"
    --max-retries "${MAX_RETRIES}"
    --top-k-per-role "${TOP_K_PER_ROLE}"
    --candidate-pool-size "${CANDIDATE_POOL_SIZE}"
    --min-mask-area "${MIN_MASK_AREA}"
    --prompt-variants "${PROMPT_VARIANTS}"
    --mask-nms-iou "${MASK_NMS_IOU}"
    --canonical-containment "${CANONICAL_CONTAINMENT}"
    --canonical-bbox-iou "${CANONICAL_BBOX_IOU}"
    --mask-alpha "${CANDIDATE_MASK_ALPHA}"
    --visualization-cell-width "${VISUALIZATION_CELL_WIDTH}"
    --frame-interval "${FRAME_INTERVAL}"
  )
  [[ "${USE_BF16}" != "1" ]] && STAGE1_ARGS+=(--no-bf16)
  [[ "${COMPILE_MODEL}" == "1" ]] && STAGE1_ARGS+=(--compile)
  [[ "${SAVE_FRAME_CONTACT_SHEET}" == "1" ]] && STAGE1_ARGS+=(--save-frame-contact-sheet) || STAGE1_ARGS+=(--no-save-frame-contact-sheet)
  [[ "${CANDIDATE_RESUME}" == "1" ]] && STAGE1_ARGS+=(--resume)
  [[ "${CANDIDATE_PROGRESS}" == "1" ]] && STAGE1_ARGS+=(--progress) || STAGE1_ARGS+=(--no-progress)
  [[ "${CANDIDATE_DRY_RUN}" == "1" ]] && STAGE1_ARGS+=(--dry-run)
  [[ -n "${MODEL_PATH}" ]] && STAGE1_ARGS+=(--model-path "${MODEL_PATH}")
  [[ -n "${INSTRUCTION}" ]] && STAGE1_ARGS+=(--instruction "${INSTRUCTION}")
  [[ -n "${INSTRUCTION_FILE}" ]] && STAGE1_ARGS+=(--instruction-file "${INSTRUCTION_FILE}")
  [[ -n "${ROLE_SPEC_JSON}" ]] && STAGE1_ARGS+=(--role-spec-json "${ROLE_SPEC_JSON}")
  [[ -n "${END_FRAME}" ]] && STAGE1_ARGS+=(--end "${END_FRAME}")
  [[ -n "${SAM_CHECKPOINT}" ]] && STAGE1_ARGS+=(--sam-checkpoint "${SAM_CHECKPOINT}")
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
    --depth-scale "${DEPTH_SCALE}"
    --depth-mode "${DEPTH_MODE}"
    --max-points-per-candidate "${MAX_POINTS_PER_CANDIDATE}"
    --cluster-distance-m "${CLUSTER_DISTANCE_M}"
    --bbox-iou-threshold "${BBOX_IOU_THRESHOLD}"
    --max-hypothesis-diameter-m "${MAX_HYPOTHESIS_DIAMETER_M}"
    --max-size-ratio "${MAX_SIZE_RATIO}"
    --legacy-canonical-iou "${LEGACY_CANONICAL_IOU}"
    --legacy-canonical-containment "${LEGACY_CANONICAL_CONTAINMENT}"
    --track-distance-m "${TRACK_DISTANCE_M}"
    --min-fused-points "${MIN_FUSED_POINTS}"
    --min-bbox-diagonal-m "${MIN_BBOX_DIAGONAL_M}"
  )
  # Stage 4 depends on object_summary; auto-enable summary export when decision is requested.
  [[ "${SAVE_OBJECT_SUMMARY}" == "1" || "${SKIP_DECISION}" == "0" ]] && STAGE2_ARGS+=(--save-object-summary)
  [[ -n "${OBJECT_SUMMARY_JSON}" ]] && STAGE2_ARGS+=(--object-summary-json "${OBJECT_SUMMARY_JSON}")
  [[ -n "${RLBENCH_LOW_DIM_OBS}" ]] && STAGE2_ARGS+=(--rlbench-low-dim-obs "${RLBENCH_LOW_DIM_OBS}")
  [[ -n "${NEAREST_DISTANCE_M}" ]] && STAGE2_ARGS+=(--nearest-distance-m "${NEAREST_DISTANCE_M}")
  [[ "${LEGACY_UNION_FIND}" == "1" ]] && STAGE2_ARGS+=(--legacy-union-find)
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
    --decision-window-frames "${DECISION_WINDOW_FRAMES}"
    --grounding-min-side "${DECISION_GROUNDING_MIN_SIDE}"
    --max-retries "${DECISION_MAX_RETRIES}"
    --max-new-tokens "${DECISION_MAX_NEW_TOKENS}"
    --max-candidate-images "${MAX_CANDIDATE_IMAGES}"
    --max-candidates-for-decision "${MAX_CANDIDATES_FOR_DECISION}"
    --min-candidate-point-count "${MIN_CANDIDATE_POINT_COUNT}"
    --min-candidate-camera-count "${MIN_CANDIDATE_CAMERA_COUNT}"
    --min-candidate-sam-score "${MIN_CANDIDATE_SAM_SCORE}"
  )
  [[ -n "${MAX_EE_DISTANCE_M}" ]] && STAGE4_ARGS+=(--max-ee-distance-m "${MAX_EE_DISTANCE_M}")
  [[ -n "${DECISION_MODEL_PATH}" ]] && STAGE4_ARGS+=(--model-path "${DECISION_MODEL_PATH}")
  [[ -n "${DECISION_FRAME_ID}" ]] && STAGE4_ARGS+=(--decision-frame-id "${DECISION_FRAME_ID}")
  [[ -n "${DECISION_OUTPUT_JSON}" ]] && STAGE4_ARGS+=(--output-json "${DECISION_OUTPUT_JSON}")
  [[ "${DECISION_DRY_RUN}" == "1" ]] && STAGE4_ARGS+=(--dry-run)
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
    --episode-dir "${EPISODE_DIR}"
    --point-stride "${VIZ_POINT_STRIDE}"
    --point-radius "${VIZ_POINT_RADIUS}"
    --mask-alpha "${VIZ_MASK_ALPHA}"
  )
  [[ -n "${VIZ_FRAME_IDS}" ]] && STAGE3_ARGS+=(--frame-ids "${VIZ_FRAME_IDS}")
  [[ -n "${RLBENCH_LOW_DIM_OBS}" ]] && STAGE3_ARGS+=(--rlbench-low-dim-obs "${RLBENCH_LOW_DIM_OBS}")
  [[ -n "${VIZ_MAX_FRAMES}" ]] && STAGE3_ARGS+=(--max-frames "${VIZ_MAX_FRAMES}")
  [[ "${VIZ_SKIP_POINTCLOUD}" == "1" ]] && STAGE3_ARGS+=(--skip-pointcloud)
  [[ -n "${CAMERAS}" ]] && STAGE3_ARGS+=(--cameras "${CAMERAS}")
  [[ -n "${CAMERA_PARAMS_JSON}" ]] && STAGE3_ARGS+=(--camera-params-json "${CAMERA_PARAMS_JSON}")
  [[ "${INVERT_RLBENCH_EXTRINSICS}" == "1" ]] && STAGE3_ARGS+=(--invert-rlbench-extrinsics)
  "${PYTHON}" "${SCRIPT_DIR}/visualize_fused_candidates.py" "${STAGE3_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 5: Visualize stage-4 decision on reprojection overlays (optional)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DECISION}" == "1" ]]; then
  echo "[stage 5/5] Skipping decision visualization (Stage 4 disabled)."
elif [[ "${SKIP_DECISION_VIZ}" == "1" ]]; then
  echo "[stage 5/5] Skipping decision visualization (SKIP_DECISION_VIZ=1)."
else
  echo "[stage 5/5] Rendering decision overlays (target/reference highlights)..."
  STAGE5_ARGS=(
    --object-predictions-json "${DECISION_OUTPUT_JSON:-${OBJECT_PREDICTIONS_PATH_DEFAULT}}"
    --fused-json "${FUSED_JSON}"
    --viz-dir "${VIZ_DIR}"
    --episode-dir "${EPISODE_DIR}"
    --point-stride "${DECISION_VIZ_POINT_STRIDE}"
    --point-radius "${DECISION_VIZ_POINT_RADIUS}"
    --mask-alpha "${DECISION_VIZ_MASK_ALPHA}"
  )
  [[ -n "${RLBENCH_LOW_DIM_OBS}" ]] && STAGE5_ARGS+=(--rlbench-low-dim-obs "${RLBENCH_LOW_DIM_OBS}")
  [[ -n "${DECISION_VIZ_OUTPUT_DIR}" ]] && STAGE5_ARGS+=(--output-dir "${DECISION_VIZ_OUTPUT_DIR}")
  [[ -n "${CAMERAS}" ]] && STAGE5_ARGS+=(--cameras "${CAMERAS}")
  [[ -n "${CAMERA_PARAMS_JSON}" ]] && STAGE5_ARGS+=(--camera-params-json "${CAMERA_PARAMS_JSON}")
  [[ "${INVERT_RLBENCH_EXTRINSICS}" == "1" ]] && STAGE5_ARGS+=(--invert-rlbench-extrinsics)
  "${PYTHON}" "${SCRIPT_DIR}/stage4_visualize_decision.py" "${STAGE5_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 6: Compact stage comparison montage (optional)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DECISION}" == "1" ]]; then
  echo "[stage 6/6] Skipping stage comparison (Stage 4 disabled)."
elif [[ "${SKIP_STAGE_COMPARE}" == "1" ]]; then
  echo "[stage 6/6] Skipping stage comparison (SKIP_STAGE_COMPARE=1)."
else
  echo "[stage 6/6] Rendering compact Stage 3 vs Stage 5 montages..."
  STAGE_COMPARE_META="${DECISION_VIZ_OUTPUT_DIR:-${DECISION_VIZ_DIR_DEFAULT}}/decision_visualization.json"
  STAGE6_ARGS=(
    --decision-meta-json "${STAGE_COMPARE_META}"
    --panel-gap "${STAGE_COMPARE_PANEL_GAP}"
    --label-height "${STAGE_COMPARE_LABEL_HEIGHT}"
    --summary-width "${STAGE_COMPARE_SUMMARY_WIDTH}"
    --background "${STAGE_COMPARE_BACKGROUND}"
  )
  [[ -n "${STAGE1_CANDIDATES_JSON}" ]] && STAGE6_ARGS+=(--stage1-candidates-json "${STAGE1_CANDIDATES_JSON}")
  [[ -n "${STAGE_COMPARE_OUTPUT_DIR}" ]] && STAGE6_ARGS+=(--output-dir "${STAGE_COMPARE_OUTPUT_DIR}")
  "${PYTHON}" "${SCRIPT_DIR}/stage6_visualize_stage_montage.py" "${STAGE6_ARGS[@]}"
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
if [[ "${SKIP_DECISION}" == "0" && "${SKIP_DECISION_VIZ}" != "1" ]]; then
  echo "  DecisionViz:${DECISION_VIZ_OUTPUT_DIR:-${DECISION_VIZ_DIR_DEFAULT}}"
fi
if [[ "${SKIP_DECISION}" == "0" && "${SKIP_STAGE_COMPARE}" != "1" ]]; then
  echo "  StageCmp:   ${STAGE_COMPARE_OUTPUT_DIR:-${OUTPUT_DIR}/viz_compare}"
fi
echo "=========================================="
