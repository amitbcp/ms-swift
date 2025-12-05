"""
Dataset metadata and utilities for the custom ModernBERT reranker experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class DatasetConfig:
    """Small helper struct capturing per-dataset defaults."""

    key: str
    dataset_dirname: str
    description: str
    default_train_batch_size: int
    default_eval_batch_size: int
    default_learning_rate: float
    default_num_train_epochs: float
    default_max_negative_samples: int
    default_max_length: int = 8192
    eval_steps: int = 100
    split_ratio: float = 0.05

    def dataset_dir(self, project_root: Path) -> Path:
        """Resolve the absolute dataset path using the repo project root."""
        return project_root / "data" / "reranker_training_data" / self.dataset_dirname

    def shards(self, project_root: Path) -> List[Path]:
        """Return sorted list of train shard files for swift ingestion."""
        dataset_dir = self.dataset_dir(project_root)
        shards = sorted(dataset_dir.glob("train_shard*.jsonl"))
        if not shards:
            raise FileNotFoundError(
                f"No train_shard*.jsonl files found under {dataset_dir}. "
                "Generate shards via notebooks/reranker_datacuration/* scripts first."
            )
        return shards

    def dataset_glob(self, project_root: Path) -> str:
        """Return a glob path (directory + wildcard) so Swift ingests multiple shards as one split."""
        dataset_dir = self.dataset_dir(project_root)
        pattern = dataset_dir / "train_shard*.jsonl"
        if not list(dataset_dir.glob("train_shard*.jsonl")):
            raise FileNotFoundError(
                f"No train_shard*.jsonl files found under {dataset_dir}. "
                "Generate shards via notebooks/reranker_datacuration/* scripts first."
            )
        return str(pattern)


def _build_config_table() -> Dict[str, DatasetConfig]:
    """Return all supported dataset configurations."""
    return {
        "rlhn": DatasetConfig(
            key="rlhn",
            dataset_dirname="rlhn-680k",
            description="rlhn/rlhn-680K curated to the reranker schema (conversational queries).",
            default_train_batch_size=48,
            default_eval_batch_size=48,
            default_learning_rate=6e-6,
            default_num_train_epochs=1.5,
            default_max_negative_samples=4,
            default_max_length=8192,
        ),
        "rlhn_mle_50": DatasetConfig(
            key="rlhn_mle_50",
            dataset_dirname="rlhn-680k_scored_gte_modernbert_filtered_mle_0.50",
            description="rlhn/rlhn-680K curated to the reranker schema (conversational queries). With margin le 0.5",
            default_train_batch_size=48,
            default_eval_batch_size=48,
            default_learning_rate=6e-6,
            default_num_train_epochs=1.5,
            default_max_negative_samples=4,
            default_max_length=8192,
        ),
        "reasonir_vl": DatasetConfig(
            key="reasonir_vl",
            dataset_dirname="reasonir/vl",
            description="ReasonIR Varied-Length configuration (long-form QA + retrieval).",
            default_train_batch_size=32,
            default_eval_batch_size=32,
            default_learning_rate=5e-6,
            default_num_train_epochs=1.0,
            default_max_negative_samples=6,
            default_max_length=8192,
        ),
        "reasonir_hq": DatasetConfig(
            key="reasonir_hq",
            dataset_dirname="reasonir/hq",
            description="ReasonIR Hard-Query split with BRIGHT documents resolved.",
            default_train_batch_size=24,
            default_eval_batch_size=24,
            default_learning_rate=5e-6,
            default_num_train_epochs=1.0,
            default_max_negative_samples=6,
            default_max_length=8192,
        ),
        "reasonir_hq_mle_50": DatasetConfig(
            key="reasonir_hq_mle_50",
            dataset_dirname="resonir_hq_scored_gte_modernbert_filtered_mle_0.50",
            description="ReasonIR Hard-Query split with BRIGHT documents resolved.. With margin le 0.5",
            default_train_batch_size=24,
            default_eval_batch_size=24,
            default_learning_rate=5e-6,
            default_num_train_epochs=1.0,
            default_max_negative_samples=6,
            default_max_length=8192,
        ),
        "hn_mine": DatasetConfig(
            key="hn_mine",
            dataset_dirname="embeddings_supervised_hn_mine",
            description="Merged hard-negative mining splits from lightonai/embeddings_supervised.",
            default_train_batch_size=48,
            default_eval_batch_size=48,
            default_learning_rate=6e-6,
            default_num_train_epochs=1.0,
            default_max_negative_samples=4,
            default_max_length=8192,
        ),
        "hn_mine_mle_50": DatasetConfig(
            key="hn_mine_mle_50",
            dataset_dirname="embeddings_supervised_hn_mine_scored_gte_modernbert_filtered_mle_0.50",
            description="Merged hard-negative mining splits from lightonai/embeddings_supervised.. With margin le 0.5",
            default_train_batch_size=48,
            default_eval_batch_size=48,
            default_learning_rate=6e-6,
            default_num_train_epochs=1.0,
            default_max_negative_samples=4,
            default_max_length=8192,
        ),
    }


DATASET_CONFIGS: Dict[str, DatasetConfig] = _build_config_table()


def get_dataset_config(key: str) -> DatasetConfig:
    """Fetch the dataset configuration, raising a helpful error on unknown keys."""
    try:
        return DATASET_CONFIGS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset '{key}'. Valid options: {', '.join(sorted(DATASET_CONFIGS))}."
        ) from exc
