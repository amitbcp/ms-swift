#!/usr/bin/env python3
"""Launch Phase-1 ModernBERT reranker fine-tunes across multiple curated datasets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import DATASET_CONFIGS, DatasetConfig, get_dataset_config
from phase1_train import _project_root, _default_val_datasets

DEFAULT_LR = 6.795952228240486e-08
DEFAULT_BATCH = 8
DEFAULT_EPOCHS = 1.0
DEFAULT_MAX_NEG = 4
DEFAULT_MAX_LEN = 8100
DEFAULT_LOSS = "listwise_reranker"
DEFAULT_EVAL_STEPS = 100
DEFAULT_GRAD_ACCUM = 2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a ModernBERT reranker fine-tune on a combination of curated datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=sorted(DATASET_CONFIGS),
        help="Dataset keys defined in config.py (e.g., rlhn reasonir_hq reasonir_vl hn_mine).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory. Defaults to ms-swift/output/custom_modernbert/phase1/all/<timestamp>.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-negative-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument(
        "--loss-type",
        type=str,
        default=DEFAULT_LOSS,
        choices=["reranker", "listwise_reranker"],
    )
    parser.add_argument(
        "--val-dataset",
        nargs="+",
        default=None,
        help="Optional validation dataset paths. Defaults to the ReasonIR-HQ eval shards.",
    )
    parser.add_argument(
        "--eval-dataset",
        nargs="+",
        default=None,
        help="Optional evaluation-only dataset paths forwarded via --eval_dataset.",
    )
    parser.add_argument(
        "--additional-swift-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Raw arguments appended verbatim after -- to the swift command.",
    )
    return parser.parse_args()

def _default_output_dir(ms_swift_root: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return ms_swift_root / "output" / "custom_modernbert" / "phase1" / "all" / timestamp


def _collect_dataset_shards(dataset_keys: List[str], project_root: Path) -> List[str]:
    shards: List[str] = []
    for key in dataset_keys:
        config: DatasetConfig = get_dataset_config(key)
        shards.extend(str(path) for path in config.shards(project_root))
    return shards

def main() -> None:
    args = parse_args()
    script_path = Path(__file__)
    project_root = _project_root(script_path)
    ms_swift_root = script_path.resolve().parents[4]

    dataset_entries = _collect_dataset_shards(args.datasets, project_root)
    output_dir = args.output_dir or _default_output_dir(ms_swift_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    lr = args.learning_rate or DEFAULT_LR
    train_bs = args.per_device_train_batch_size or DEFAULT_BATCH
    eval_bs = args.per_device_eval_batch_size or DEFAULT_BATCH
    grad_accum = args.gradient_accumulation_steps
    epochs = args.num_train_epochs or DEFAULT_EPOCHS
    eval_steps = args.eval_steps or DEFAULT_EVAL_STEPS
    split_ratio = 0.0
    max_neg = args.max_negative_samples or DEFAULT_MAX_NEG
    max_length = args.max_length
    loss_type = args.loss_type
    val_datasets: Optional[List[str]] = args.val_dataset or _default_val_datasets(project_root)
    eval_datasets: Optional[List[str]] = args.eval_dataset

    env = os.environ.copy()
    env.setdefault("MAX_NEGATIVE_SAMPLES", str(max_neg))
    env.setdefault("MAX_POSITIVE_SAMPLES", "2") #1
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

    swift_cli = ms_swift_root / "swift" / "cli" / "sft.py"
    swift_args = [
        "--model",
        "iic/gte-reranker-modernbert-base",
        "--task_type",
        "reranker",
        "--loss_type",
        loss_type,
        "--train_type",
        "full",
        "--dataset",
        *dataset_entries,
        "--load_from_cache_file",
        "true",
        "--split_dataset_ratio",
        str(split_ratio),
        "--eval_strategy",
        "steps",
        "--output_dir",
        str(output_dir),
        "--eval_steps",
        str(eval_steps),
        "--num_train_epochs",
        str(epochs),
        "--save_steps",
        str(eval_steps * 2),
        "--per_device_train_batch_size",
        str(train_bs),
        "--per_device_eval_batch_size",
        str(eval_bs),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--dataset_num_proc",
        "8",
        "--learning_rate",
        f"{lr:.8g}",
        "--label_names",
        "labels",
        "--dataloader_drop_last",
        "true",
        "--report_to",
        "tensorboard",
        "--max_length",
        str(max_length),
        "--add_version",
        "false",
        "--eval_on_start",
        "true",
        "--load_best_model_at_end",
        "true",
        "--metric_for_best_model",
        "eval_mrr",
        "--greater_is_better",
        "true",
    ]
    if val_datasets:
        swift_args.append("--val_dataset")
        swift_args.extend(val_datasets)
    if eval_datasets:
        swift_args.append("--eval_dataset")
        swift_args.extend(eval_datasets)
    if args.additional_swift_args:
        swift_args.extend(args.additional_swift_args)

    nproc_env = os.environ.get("NPROC_PER_NODE", "1")
    try:
        nproc = max(1, int(nproc_env))
    except ValueError:
        nproc = 1
    use_torchrun = nproc > 1

    if use_torchrun:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(nproc),
            str(swift_cli),
        ] + swift_args
    else:
        cmd = ["swift", "sft"] + swift_args

    print("Running command:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=ms_swift_root, env=env)


if __name__ == "__main__":
    sys.exit(main())
