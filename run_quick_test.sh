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

if [[ -z "${MODEL_DIR}" ]]; then
  echo "Set MODEL_DIR to the downloaded ModelScope snapshot directory."
  exit 2
fi

"${PYTHON}" "${SCRIPT_DIR}/demo_sam3.py" \
  --model-dir "${MODEL_DIR}" \
  --image "${SCRIPT_DIR}/inputs/coffee_scene.png" \
  --device "${DEVICE}" \
  --mode point_box \
  --point 292 206 1 \
  --point 392 286 0 \
  --box 164 16 430 322 \
  --output-dir "${SCRIPT_DIR}/outputs/quick_test"
