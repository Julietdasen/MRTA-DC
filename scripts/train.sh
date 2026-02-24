#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -----------------------------
# User editable settings
# -----------------------------
ALGO_NAME="${ALGO_NAME:-REINFORCE}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs}"
RUN_TAG="${RUN_TAG:-baseline}"
RUN_COMMENT="${RUN_COMMENT:-manual_run_with_v0_3}" # 这里可以添加一些备注信息，比如实验的目的、使用的数据集等，方便后续回顾和分析
PYTHON_BIN="${PYTHON_BIN:-python}"
DRIVER_PATH="${DRIVER_PATH:-${PROJECT_ROOT}/driver.py}"

# Optional resume checkpoint path.
RESUME_CKPT="${RESUME_CKPT:-}"

# -----------------------------
# Build run name and directories
# -----------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)" # 时间戳
SAFE_TAG="$(echo "${RUN_TAG}" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"
SAFE_COMMENT="$(echo "${RUN_COMMENT}" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"

RUN_NAME="${TIMESTAMP}_${ALGO_NAME}"
if [[ -n "${SAFE_TAG}" ]]; then
  RUN_NAME="${RUN_NAME}_${SAFE_TAG}"
fi
if [[ -n "${SAFE_COMMENT}" ]]; then
  RUN_NAME="${RUN_NAME}_${SAFE_COMMENT}"
fi

RUN_DIR="${RUN_ROOT}/${RUN_NAME}" # 训练结果的根目录
MODEL_DIR="${RUN_DIR}/model" #ckpt模型的保存目录
TB_DIR="${RUN_DIR}/train/tb" # TensorBoard日志的保存目录
GIF_DIR="${RUN_DIR}/artifacts/gifs" # 训练过程中生成的GIF动画的保存目录
LOG_DIR="${RUN_DIR}/logs" # 训练日志的保存目录
CONFIG_DIR="${RUN_DIR}/config" # 训练配置的保存目录

mkdir -p "${MODEL_DIR}" "${TB_DIR}" "${GIF_DIR}" "${LOG_DIR}" "${CONFIG_DIR}"

echo "Run name: ${RUN_NAME}"
echo "Run dir:  ${RUN_DIR}"
echo "Model:    ${MODEL_DIR}"
echo "TensorBoard: ${TB_DIR}"
echo "Log file: ${LOG_DIR}/train.log"

if [[ -n "${RESUME_CKPT}" ]]; then
  echo "Resume checkpoint: ${RESUME_CKPT}"
fi

# Export runtime paths so parameters.py picks them up.
export DCMRTA_RUN_DIR="${RUN_DIR}"
export DCMRTA_RUN_TAG="${RUN_TAG}"
export DCMRTA_RUN_COMMENT="${RUN_COMMENT}"
export DCMRTA_MODEL_PATH="${MODEL_DIR}"
export DCMRTA_TRAIN_PATH="${TB_DIR}"
export DCMRTA_GIFS_PATH="${GIF_DIR}"
# train.sh uses tee to persist logs, so disable logger file handler to avoid duplicate lines.
export DCMRTA_DISABLE_FILE_LOG="${DCMRTA_DISABLE_FILE_LOG:-1}"
if [[ -n "${RESUME_CKPT}" ]]; then
  export DCMRTA_RESUME_CKPT="${RESUME_CKPT}"
fi

# Save launch context.
{
  echo "timestamp=${TIMESTAMP}"
  echo "algo_name=${ALGO_NAME}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "run_tag=${RUN_TAG}"
  echo "run_comment=${RUN_COMMENT}"
  echo "python_bin=${PYTHON_BIN}"
  echo "driver_path=${DRIVER_PATH}"
  echo "resume_ckpt=${RESUME_CKPT}"
} > "${CONFIG_DIR}/launch.env"

# Launch training.
"${PYTHON_BIN}" "${DRIVER_PATH}" --tag "${RUN_TAG}" --comment "${RUN_COMMENT}" 2>&1 | tee -a "${LOG_DIR}/train.log"
