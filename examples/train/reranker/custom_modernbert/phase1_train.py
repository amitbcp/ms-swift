#!/usr/bin/env python3
"""Wrapper to launch Phase-1 ModernBERT reranker fine-tunes on curated datasets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import DATASET_CONFIGS, DatasetConfig, get_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a ModernBERT reranker fine-tune on a curated dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_CONFIGS),
        help="Dataset key defined in config.py (e.g., rlhn, reasonir_vl, reasonir_hq, hn_mine).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory. Defaults to ms-swift/output/custom_modernbert/phase1/<dataset>/<timestamp>.",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        help="Override train batch size per device.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Override eval batch size per device.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help="Override number of train epochs.",
    )
    parser.add_argument(
        "--max-negative-samples",
        type=int,
        default=None,
        help="Override MAX_NEGATIVE_SAMPLES env value (defaults depend on dataset).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8100,
        help="Override maximum sequence length (defaults to the config’s max_length, e.g., 8192).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Override evaluation interval in steps.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Override gradient accumulation steps (defaults to 2 for better GPU utilization).",
    )
    parser.add_argument(
        "--split-dataset-ratio",
        type=float,
        default=None,
        help="Override fraction of training data reserved for validation.",
    )
    parser.add_argument(
        "--additional-swift-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Optional raw arguments appended verbatim after '--' to the swift command.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default=None,
        choices=["reranker", "listwise_reranker"],
        help="Override the reranker loss type (defaults to reranker).",
    )
    parser.add_argument(
        "--val-dataset",
        nargs="+",
        default=None,
        help=(
            "Optional validation dataset paths. Defaults to the ReasonIR-HQ shards so every run "
            "shares the same held-out set."
        ),
    )
    parser.add_argument(
        "--eval-dataset",
        nargs="+",
        default=None,
        help=(
            "Optional evaluation-only dataset paths forwarded via --eval_dataset. "
            "Use this to compute metrics on extra corpora without affecting the validation split."
        ),
    )
    return parser.parse_args()


def _project_root(script_path: Path) -> Path:
    """Infer repo roots relative to this file."""
    # script .../ms-swift/examples/train/reranker/custom_modernbert/phase1_train.py
    ms_swift_root = script_path.resolve().parents[4]
    return ms_swift_root.parent


def _build_output_dir(ms_swift_root: Path, dataset_key: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return ms_swift_root / "output" / "custom_modernbert" / "phase1" / dataset_key / timestamp


def _dataset_args(config: DatasetConfig, project_root: Path) -> List[str]:
    """Return all shard paths so Swift ingests them as separate datasets."""
    return [str(path) for path in config.shards(project_root)]


def _default_val_datasets(project_root: Path) -> List[str]:
    root = project_root / "data" / "reranker_training_data" / "reasonir" / "eval_hq"
    shards = sorted(root.glob("train_shard*.jsonl"))
    if not shards:
        raise FileNotFoundError(
            f"No ReasonIR-HQ shards found under {root}. Generate them via "
            "notebooks/reranker_datacuration/reasonir_data.py before running Phase-1."
        )
    return [str(path) for path in shards]


def main() -> None:
    args = parse_args()
    script_path = Path(__file__)
    project_root = _project_root(script_path)
    ms_swift_root = script_path.resolve().parents[4]
    config = get_dataset_config(args.dataset)

    dataset_entries = _dataset_args(config, project_root)
    output_dir = args.output_dir or _build_output_dir(ms_swift_root, config.key)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    lr = args.learning_rate or config.default_learning_rate
    train_bs = args.per_device_train_batch_size or config.default_train_batch_size
    eval_bs = args.per_device_eval_batch_size or 8
    grad_accum = args.gradient_accumulation_steps
    epochs = args.num_train_epochs or config.default_num_train_epochs
    eval_steps = args.eval_steps or config.eval_steps
    split_ratio = 0.0
    max_neg = args.max_negative_samples or config.default_max_negative_samples
    max_length = args.max_length or config.default_max_length
    loss_type = args.loss_type or "reranker"
    val_datasets: Optional[List[str]] = args.val_dataset
    if val_datasets is None:
        val_datasets = _default_val_datasets(project_root)
    eval_datasets: Optional[List[str]] = args.eval_dataset

    env = os.environ.copy()
    env.setdefault("MAX_NEGATIVE_SAMPLES", str(max_neg))
    env.setdefault("MAX_POSITIVE_SAMPLES", "1")
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

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ms_swift_root, env=env)


if __name__ == "__main__":
    sys.exit(main())
