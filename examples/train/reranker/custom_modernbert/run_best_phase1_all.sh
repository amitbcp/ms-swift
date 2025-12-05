#!/usr/bin/env bash
# Run a single Phase-1 ModernBERT fine-tune that mixes all curated datasets
# using the best RLHN Optuna hyperparameters.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PHASE_SCRIPT="${SCRIPT_DIR}/phase1_train_all.py"
OUTPUT_DIR="${MS_SWIFT_ROOT}/output/custom_modernbert/best_optuna_params/all_datasets_p2_n8_bs1_ga2"
mkdir -p "${OUTPUT_DIR}"

DATASETS=(rlhn reasonir_hq hn_mine)
BEST_LR="6.795952228240486e-08"
BEST_BATCH="1" #4
BEST_EPOCHS="1"
BEST_MAX_NEG="8" #4
BEST_LOSS="listwise_reranker"
BEST_MAX_LEN="8100"
BEST_GRAD_ACCUM="2"

python "${PHASE_SCRIPT}" \
  --datasets "${DATASETS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --learning-rate "${BEST_LR}" \
  --per-device-train-batch-size "${BEST_BATCH}" \
  --per-device-eval-batch-size "${BEST_BATCH}" \
  --num-train-epochs "${BEST_EPOCHS}" \
  --max-negative-samples "${BEST_MAX_NEG}" \
  --loss-type "${BEST_LOSS}" \
  --max-length "${BEST_MAX_LEN}" \
  --gradient-accumulation-steps "${BEST_GRAD_ACCUM}"
