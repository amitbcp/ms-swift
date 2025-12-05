#!/usr/bin/env bash
# ModernBERT baseline that mirrors train_reranker.sh but evaluates against the
# curated reranker_eval JSONL suite to verify multi-dataset eval wiring.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
EVAL_ROOT="${REPO_ROOT}/data/reranker_eval"

# Default to a single GPU unless the caller overrides CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
swift sft \
    --model iic/gte-reranker-modernbert-base \
    --task_type reranker \
    --loss_type reranker \
    --train_type full \
    --dataset MTEB/scidocs-reranking \
    --load_from_cache_file true \
    --split_dataset_ratio 0.0 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --dataset_num_proc 8 \
    --learning_rate 6e-6 \
    --label_names labels \
    --dataloader_drop_last true \
    --val_dataset \
        "${EVAL_ROOT}/scidocs/validation.jsonl" \
        "${EVAL_ROOT}/scidocs/test.jsonl" \
        "${EVAL_ROOT}/syntec/test.jsonl" \
        "${EVAL_ROOT}/askubuntu/test.jsonl"
