#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR="${MODEL_DIR:-}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
USE_BF16="${USE_BF16:-0}"
BF16_ARGS=()
if [[ "${USE_BF16}" != "1" ]]; then
  BF16_ARGS+=(--no-bf16)
fi
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs}"

if [[ -z "${MODEL_DIR}" ]]; then
  echo "ERROR: MODEL_DIR is required."
  echo "Example:"
  echo "  MODEL_DIR=/path/to/modelscope/sam3.1 bash ${SCRIPT_DIR}/run_sam31_examples.sh"
  exit 2
fi

SCRIPT="${SCRIPT_DIR}/test_sam3_modelscope.py"
COFFEE_IMAGE="${SCRIPT_DIR}/inputs/coffee_scene.png"
COFFEE_MASK="${SCRIPT_DIR}/inputs/coffee_mask_prompt.png"
SHAPES_IMAGE="${SCRIPT_DIR}/inputs/shapes_scene.png"
SHAPES_MASK="${SCRIPT_DIR}/inputs/shapes_green_rectangle_mask.png"

common_coffee=(
  "${PYTHON}" "${SCRIPT}"
  --model-dir "${MODEL_DIR}"
  --image "${COFFEE_IMAGE}"
  --device "${DEVICE}"
  "${BF16_ARGS[@]}"
)

echo "[1/9] Coffee image: open-vocabulary text prompt"
"${common_coffee[@]}" \
  --mode text \
  --text "cup" \
  --threshold 0.30 \
  --output-dir "${OUT_ROOT}/coffee_text"

echo "[2/9] Coffee image: positive point prompt"
"${common_coffee[@]}" \
  --mode point \
  --point 292 206 1 \
  --output-dir "${OUT_ROOT}/coffee_point"

echo "[3/9] Coffee image: box prompt"
"${common_coffee[@]}" \
  --mode box \
  --box 164 16 430 322 \
  --output-dir "${OUT_ROOT}/coffee_box"

echo "[4/9] Coffee image: point + box prompt"
"${common_coffee[@]}" \
  --mode point_box \
  --point 292 206 1 \
  --point 392 286 0 \
  --box 164 16 430 322 \
  --output-dir "${OUT_ROOT}/coffee_point_box"

echo "[5/9] Coffee image: iterative mask refinement"
"${common_coffee[@]}" \
  --mode mask_refine \
  --point 292 206 1 \
  --point 392 286 0 \
  --point 85 80 0 \
  --output-dir "${OUT_ROOT}/coffee_mask_refine"

echo "[6/9] Coffee image: external mask prompt"
"${common_coffee[@]}" \
  --mode mask \
  --mask-input "${COFFEE_MASK}" \
  --point 292 206 1 \
  --point 392 286 0 \
  --output-dir "${OUT_ROOT}/coffee_external_mask"

echo "[7/9] Coffee image: text + positive/negative exemplar boxes"
"${common_coffee[@]}" \
  --mode text_box \
  --text "cup" \
  --box 164 16 430 322 \
  --negative-box 326 202 455 346 \
  --threshold 0.30 \
  --output-dir "${OUT_ROOT}/coffee_text_box"

echo "[8/9] Coffee image: exemplar boxes only"
"${common_coffee[@]}" \
  --mode exemplar_box \
  --box 164 16 430 322 \
  --negative-box 326 202 455 346 \
  --threshold 0.30 \
  --output-dir "${OUT_ROOT}/coffee_exemplar_box"

echo "[9/9] Synthetic image: exact external mask prompt"
"${PYTHON}" "${SCRIPT}" \
  --model-dir "${MODEL_DIR}" \
  --image "${SHAPES_IMAGE}" \
  --device "${DEVICE}" \
  --mode mask \
  --mask-input "${SHAPES_MASK}" \
  --point 372 245 1 \
  --point 160 255 0 \
  --point 655 260 0 \
  --output-dir "${OUT_ROOT}/shapes_external_mask"

echo
echo "Completed. Results: ${OUT_ROOT}"
