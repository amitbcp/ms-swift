# Custom ModernBERT Reranker Experiments

This directory hosts all Phase‑1 assets for training `iic/gte-reranker-modernbert-base`
on the curated in-house datasets described in `changelog/0008`. The scripts live
inside `ms-swift/examples/train/reranker` so they can reuse the same tooling,
conventions, and Conda environment as the stock reranker examples.

## Prerequisites

```bash
conda activate rerank_swift
cd /mnt/shared/aamita/project/rerank/ms-swift
pip install -e .  # already done during repo setup
pip install optuna  # one-time for the sweep driver
```

Datasets must already exist under `data/reranker_training_data/<dataset_name>`
(see Task 0007 conversion scripts). The Phase‑1 plan intentionally excludes the
`bge-reranker-data` and multimodal VL sources, so only the following keys are enabled:

| Key          | Description                                                   | Path fragment                               |
|--------------|---------------------------------------------------------------|---------------------------------------------|
| `rlhn`       | `rlhn/rlhn-680K` conversational data (curated schema)         | `data/reranker_training_data/rlhn-680k`     |
| `reasonir_vl`| ReasonIR Varied-Length split                                  | `data/reranker_training_data/reasonir/vl`   |
| `reasonir_hq`| ReasonIR Hard-Query split (BRIGHT doc text resolved)          | `data/reranker_training_data/reasonir/hq`   |
| `hn_mine`    | Merged hard-negative mining shards from `embeddings_supervised`| `data/reranker_training_data/embeddings_supervised_hn_mine` |

### Recommended runtime exports (multi-GPU)

Always set the following before launching `phase1_train.py` or `optuna_phase1.py`
so Swift runs under torch.distributed across all 8 GPUs and avoids TorchDynamo
instabilities seen on ModernBERT:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export TORCHDYNAMO_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
# Optional (only after the ModernBERT weights are cached locally):

```

`phase1_train.py` automatically wraps the Swift CLI in `torchrun` when it detects
`NPROC_PER_NODE>1`, and defaults the master endpoint to `127.0.0.1:29500` if you
do not override `MASTER_ADDR/MASTER_PORT`. Once the ModelScope download has
completed at least once, enabling `TRANSFORMERS_OFFLINE=1` avoids per-rank
re-downloads entirely.

## Single-dataset runs (`phase1_train.py`)

```bash
cd /mnt/shared/aamita/project/rerank/ms-swift
python examples/train/reranker/custom_modernbert/phase1_train.py \
  --dataset rlhn \
  --output-dir output/custom_modernbert/manual/rlhn_run1
```

Key behaviors:

- Automatically loads every `train_shard*.jsonl` file via a single `--dataset`
  flag so Swift ingests all shards in one shot.
- Keeps the Task‑006 ModernBERT hyperparameters intact so performance shifts
  reflect data/model differences (batch sizes + eval cadence stay fixed).
- Applies the ReasonIR-HQ shards as the default validation set via
  `--val_dataset ...`; override with `--val-dataset path1 path2` if you need a
  different validation bundle. Internally `split_dataset_ratio` stays at `0`.
- Exposes overrides such as `--learning-rate`, `--num-train-epochs`,
  `--max-negative-samples`, `--loss-type`, `--max-length`, and pass-through arguments via
  `--additional-swift-args`.
- Writes outputs under `ms-swift/output/custom_modernbert/phase1/<dataset>/<timestamp>` by default.
- Forces `--eval_on_start true` plus `--load_best_model_at_end true` with
  `--metric_for_best_model eval_mrr` so Optuna and manual runs share a consistent
  baseline metric, and the best checkpoint is always tracked/saved.
- Uses `--per_device_eval_batch_size 8` by default unless you override the flag.
- Defaults `--gradient_accumulation_steps 2` for better GPU utilization; override with
  `--gradient-accumulation-steps N` if you want to change the global batch size.
- Accepts extra evaluation-only corpora via `--eval-dataset path1 path2 ...`, which
  get forwarded to Swift as `--eval_dataset` (distinct from the validation split).

Example with overrides and extra Swift flags:

```bash
python examples/train/reranker/custom_modernbert/phase1_train.py \
  --dataset reasonir_hq \
  --learning-rate 8e-6 \
  --loss-type reranker \
  --max-length 8100 \
  --additional-swift-args -- --gradient_checkpointing false --save_strategy epoch
```

## Baseline sanity check with external eval data

To mirror the stock `train_reranker.sh` run while exercising the curated eval
suite, use:

```bash
cd /mnt/shared/aamita/project/rerank/ms-swift
CUDA_VISIBLE_DEVICES=0 examples/train/reranker/custom_modernbert/train_reranker_evalsuite.sh
```

The command keeps the original training recipe (ModernBERT on
`MTEB/scidocs-reranking`) but disables the automatic train/val split
(`--split_dataset_ratio 0`) and injects four evaluation datasets from
`data/reranker_eval`. This is the quickest way to verify that the evaluation
JSONL files are wired correctly before running larger sweeps.

## Best-param batch runner

After you identify a promising hyperparameter set (e.g., from Optuna), run all
datasets sequentially via:

```bash
cd /mnt/shared/aamita/project/rerank/ms-swift
bash examples/train/reranker/custom_modernbert/run_best_phase1.sh
```

The helper script currently applies the best RLHN params (learning rate
`6.795952e-08`, per-device batch `8`, epochs `1`, `max_negative_samples 4`,
`listwise_reranker` loss, `max_length 8100`, gradient accumulation `2`) to each
Phase‑1 dataset (`rlhn`, `reasonir_vl`, `reasonir_hq`, `hn_mine`). Outputs land
in `output/custom_modernbert/best_optuna_params/<dataset>` using the same
ReasonIR-HQ validation split as the sweeps. Tweak the script if future sweeps
produce different best settings.

## Optuna sweep driver (`optuna_phase1.py`)

```bash
cd /mnt/shared/aamita/project/rerank/ms-swift
python examples/train/reranker/custom_modernbert/optuna_phase1.py \
  --dataset rlhn \
  --trials 9 \
  --metric eval_mrr \
  --pruner median \
  --output-root output/custom_modernbert/optuna/rlhn_sweep2
```

What it does:

- Samples learning rate, epochs, `MAX_NEGATIVE_SAMPLES`, loss type, and
  per-device batch size (256/512/1024). Eval cadence and split ratio remain
  fixed to isolate modeling effects.
- Launches `phase1_train.py` (which now forwards `--add_version false` to
  `swift sft`, so outputs go directly into the provided folder) for each trial,
  then parses `logging.jsonl` from that folder to obtain the metric (default `eval_mrr`).
- Automatically inherits the `torchrun` launcher via `phase1_train.py`, so every
  trial fans out to the 8 visible GPUs when `NPROC_PER_NODE` is exported.
- Stores per-trial summaries in `optuna_summary.json` alongside the Swift outputs
  plus Optuna’s best-value report at the end.
- Supports persistent studies via `--storage sqlite:///optuna.db` and alternative
  pruners via `--pruner none`.

To append extra Swift arguments for every trial, add them after `--additional-swift-args -- ...`.

## External evaluation harness (Phase‑1 requirement)

All Phase‑1 checkpoints must be evaluated on a consistent external suite for
apples-to-apples comparisons: the MTEB Reranking tasks (e.g., Scidocs,
Touche-2020) plus a small BEIR slice. The evaluation wrapper will live alongside
these scripts in the next update; until then, log the checkpoint path produced by
`phase1_train.py`/Optuna and run the MTEB CLI manually so that results can be
copied into `changelog/0008`.

## Next steps

- Wire in the formal evaluation wrapper + result aggregation.
- Add Phase‑2/Phase‑3 launchers once dataset mixing and loss hybrids are ready.

### All-dataset run

To replay the best RLHN parameters across every Phase-1 dataset in a single job,
run:

```bash
bash examples/train/reranker/custom_modernbert/run_best_phase1_all.sh
```

This script calls `phase1_train_all.py` with `rlhn`, `reasonir_vl`, `reasonir_hq`,
and `hn_mine`, storing the checkpoint under
`output/custom_modernbert/best_optuna_params/all_datasets`.
