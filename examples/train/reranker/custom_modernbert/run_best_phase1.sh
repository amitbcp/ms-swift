#!/usr/bin/env bash
# Run Phase-1 ModernBERT fine-tunes on each dataset using the best Optuna params.
# Assumes the usual environment exports (CUDA_VISIBLE_DEVICES, NPROC_PER_NODE, etc.)
# are already in place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PHASE_SCRIPT="${SCRIPT_DIR}/phase1_train.py"
OUTPUT_ROOT="${MS_SWIFT_ROOT}/output/custom_modernbert/best_optuna_params"
mkdir -p "${OUTPUT_ROOT}"

BEST_LR="6.795952228240486e-08"
BEST_BATCH="8"
BEST_EPOCHS="1"
BEST_MAX_NEG="4"
BEST_LOSS="listwise_reranker"
BEST_MAX_LEN="8100"

run_dataset() {
  local dataset="$1"
  local run_dir="${OUTPUT_ROOT}/${dataset}_ga2"
  mkdir -p "${run_dir}"
  echo "[best-phase1] dataset=${dataset} -> ${run_dir}"
  python "${PHASE_SCRIPT}" \
    --dataset "${dataset}" \
    --output-dir "${run_dir}" \
    --learning-rate "${BEST_LR}" \
    --per-device-train-batch-size "${BEST_BATCH}" \
    --per-device-eval-batch-size "${BEST_BATCH}" \
    --num-train-epochs "${BEST_EPOCHS}" \
    --max-negative-samples "${BEST_MAX_NEG}" \
    --loss-type "${BEST_LOSS}" \
    --max-length "${BEST_MAX_LEN}"
}

DATASETS=(rlhn reasonir_hq hn_mine)

for dataset in "${DATASETS[@]}"; do
  run_dataset "${dataset}"
done
