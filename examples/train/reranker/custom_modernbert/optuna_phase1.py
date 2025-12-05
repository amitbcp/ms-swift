#!/usr/bin/env python3
"""
Optuna-based hyperparameter sweep driver for Phase-1 ModernBERT experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import optuna

from config import DATASET_CONFIGS, DatasetConfig, get_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_CONFIGS),
        help="Dataset key defined in config.py.",
    )
    parser.add_argument("--trials", type=int, default=60, help="Number of Optuna trials to run.")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name. Defaults to dataset+timestamp if storage is provided.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db). If omitted, study lives in-memory.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval_mrr",
        help="Metric to maximize from logging.jsonl (e.g., eval_mrr, eval_ndcg).",
    )
    parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Optimization direction for the chosen metric.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed forwarded to Optuna samplers/pruners.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to place per-trial swift outputs. Defaults to ms-swift/output/custom_modernbert/optuna.",
    )
    parser.add_argument(
        "--max-trial-time",
        type=int,
        default=None,
        help="Optional per-trial time limit in seconds (enforced via timeout on swift).",
    )
    parser.add_argument(
        "--pruner",
        choices=["median", "none"],
        default="median",
        help="Optuna pruner strategy.",
    )
    parser.add_argument(
        "--additional-swift-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Optional raw arguments appended after '--' to phase1_train.py.",
    )
    return parser.parse_args()


def _ms_swift_root(script_path: Path) -> Path:
    return script_path.resolve().parents[4]


def _project_root(script_path: Path) -> Path:
    return _ms_swift_root(script_path).parent


def _default_output_root(script_path: Path) -> Path:
    return _ms_swift_root(script_path) / "output" / "custom_modernbert" / "optuna"


def _build_cmd(
    phase1_script: Path,
    dataset_key: str,
    output_dir: Path,
    trial_params: Dict[str, float],
    additional_args: Optional[list[str]],
) -> list[str]:
    per_device_train = trial_params["per_device_train_batch_size"]
    per_device_eval = trial_params["per_device_eval_batch_size"]
    cmd = [
        sys.executable,
        str(phase1_script),
        "--dataset",
        dataset_key,
        "--output-dir",
        str(output_dir),
        "--learning-rate",
        f"{trial_params['learning_rate']:.8g}",
        "--per-device-train-batch-size",
        str(per_device_train),
        "--per-device-eval-batch-size",
        str(per_device_eval),
        "--num-train-epochs",
        f"{trial_params['num_train_epochs']:.4g}",
        "--max-negative-samples",
        str(trial_params["max_negative_samples"]),
        "--loss-type",
        trial_params["loss_type"],
        "--max-length",
        str(trial_params["max_length"]),
    ]
    if additional_args:
        cmd.extend(additional_args)
    return cmd


def _parse_metric(log_file: Path, metric: str) -> float:
    value: Optional[float] = None
    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric in record:
                value = float(record[metric])
    if value is None:
        raise RuntimeError(f"Metric '{metric}' not found in {log_file}.")
    return value


def _trial_summary_path(output_dir: Path) -> Path:
    return output_dir / "optuna_summary.json"


def _suggest_params(trial: optuna.Trial, config: DatasetConfig) -> Dict[str, float]:
    """Define the search space for Phase-1 sweeps."""
    params: Dict[str, float] = {}
    params["learning_rate"] = trial.suggest_float("learning_rate", 4e-8, 2e-6, log=True)
    params["per_device_train_batch_size"] = trial.suggest_categorical(
        # "per_device_train_batch_size", [1, 2, 4, 8]  # Corresponds to 8, 16, 32 with 8 GPUs
        "per_device_train_batch_size", [8]  # Corresponds to 8, 16, 32 with 8 GPUs
    )
    params["per_device_eval_batch_size"] = params["per_device_train_batch_size"]
    # params["num_train_epochs"] = trial.suggest_float("num_train_epochs", 1, 2, log=True)
    params["num_train_epochs"] = trial.suggest_categorical("num_train_epochs", [1] )
    params["max_negative_samples"] = trial.suggest_categorical(
        "max_negative_samples", [2, 4, 6, 8]
    )
    params["loss_type"] = trial.suggest_categorical(
        "loss_type", ["reranker", "listwise_reranker"]
    )
    params["max_length"] = 8100
    return params


def main() -> None:
    args = parse_args()
    script_path = Path(__file__)
    ms_swift_root = _ms_swift_root(script_path)
    output_root = args.output_root or _default_output_root(script_path)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_config = get_dataset_config(args.dataset)
    phase1_script = script_path.parent / "phase1_train.py"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.pruner == "median":
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(n_startup_trials=2)
    else:
        pruner = optuna.pruners.NopPruner()

    default_study_name = args.study_name or f"{args.dataset}-{datetime.utcnow():%Y%m%d-%H%M%S}"
    study = optuna.create_study(
        study_name=default_study_name,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, dataset_config)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        trial_dir = output_root / f"{args.dataset}_trial{trial.number:03d}_{timestamp}"
        run_dir = trial_dir # No nested versioning for easier parsing
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = _build_cmd(phase1_script, args.dataset, run_dir, params, args.additional_swift_args)

        env = os.environ.copy()
        env.setdefault("WANDB_SILENT", "true")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        print(f"[trial {trial.number}] Running: {' '.join(cmd)}")
        try:
            subprocess.run(
                cmd,
                cwd=ms_swift_root,
                env=env,
                check=True,
                timeout=args.max_trial_time,
            )
        except subprocess.TimeoutExpired as exc:
            raise optuna.TrialPruned(f"Trial exceeded {args.max_trial_time}s limit.") from exc
        except subprocess.CalledProcessError as exc:
            raise optuna.TrialPruned(f"swift failed with exit code {exc.returncode}.") from exc

        log_file = run_dir / "logging.jsonl"
        value = _parse_metric(log_file, args.metric)
        summary_payload = {
            "dataset": args.dataset,
            "trial_number": trial.number,
            "metric": args.metric,
            "metric_value": value,
            "params": params,
            "output_dir": str(run_dir),
        }
        _trial_summary_path(run_dir).write_text(
            json.dumps(summary_payload, indent=2), encoding="utf-8"
        )
        trial.set_user_attr("output_dir", str(run_dir))
        return value

    study.optimize(objective, n_trials=args.trials)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
