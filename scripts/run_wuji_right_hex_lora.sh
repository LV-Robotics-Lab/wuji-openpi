#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TRAIN_ROOT="$(realpath ../../data/franka_wuji/training/hexagonal_peg_insertion)"
RUN_NAME="${1:-right_hex_lora_20260906}"
shift "$(( $# > 0 ? 1 : 0 ))"
RUN_DIR="${TRAIN_ROOT}/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
if [[ -f "${RUN_DIR}/pid" ]] && kill -0 "$(cat "${RUN_DIR}/pid")" 2>/dev/null; then
  echo "Training is already running for ${RUN_NAME}" >&2
  exit 1
fi
echo "$$" > "${RUN_DIR}/pid"
date -Iseconds > "${RUN_DIR}/started_at"
trap 'result=$?; echo "$result" > "${RUN_DIR}/exit_code"; date -Iseconds > "${RUN_DIR}/finished_at"' EXIT
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export HF_HOME="${TRAIN_ROOT}/hf-cache"
export OPENPI_DATA_HOME="${TRAIN_ROOT}/../openpi-cache"
export WANDB_MODE=disabled
.venv/bin/python -m pip freeze > "${RUN_DIR}/environment.txt" 2>/dev/null || ./uv pip freeze > "${RUN_DIR}/environment.txt"
git rev-parse HEAD > "${RUN_DIR}/source_commit.txt"
git diff > "${RUN_DIR}/source.patch"
.venv/bin/python scripts/train.py pi05_wuji_right_hex_lora \
  --exp-name="${RUN_NAME}" --num-workers=0 --save-interval=500 "$@"
