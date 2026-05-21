# Shadow Training Runner

This directory owns the target and shadow model training workflow for the
shadow-attack split. It reuses `centralized/train_centralized_dp.py::train` for
all model, optimizer, DP, transform, dataloader, checkpoint, and metrics logic.

Shadow experiments use MONAI `PersistentDataset` with a shared disk cache for
deterministic preprocessing. The manifest CSV and split JSON flow is unchanged;
the cache directory can be deleted safely and will rebuild on the next run.

## Commands

Run the full queue with the default sequential one-GPU worker:

```bash
./venv/bin/python shadow/train/run.py -c shadow/experiments/full_shadow_training.yml
```

Create or refresh the split files without training:

```bash
./venv/bin/python shadow/train/run.py -c shadow/experiments/full_shadow_training.yml --generate-split-only
```

Validate the split and print record summaries for the target model and first
selected shadow jobs:

```bash
./venv/bin/python shadow/train/run.py -c shadow/experiments/full_shadow_training.yml --check-data-only
```

Run the 3-epoch smoke queue with one target model and four shadow models
(`target_x_01`, `IN 1..2`, `OUT 1..2`):

```bash
./venv/bin/python shadow/train/run.py -c shadow/experiments/smoke_3epoch_4shadow.yml
```

## Artifact Layout

Generated files live under `shadow_runs/<experiment_name>/`:

```text
shadow_runs/full_shadow_training/
shadow_runs/smoke_3epoch_4shadow/
split/attack_manifest.csv
split/split_plan.json
target_model/
shadow_models/target_x_01/IN/model_in_01/
shadow_models/target_x_01/OUT/model_out_01/
run_manifest.json
run_summary.jsonl
```

Each model workdir contains the centralized DP trainer outputs, including
`resolved_config.yml`, `metrics.json`, `best_model.pth`, and `last_model.pth`.
The runner also writes `shadow_resolved_config.yml` before starting each job.

## Resume Behavior

By default, `shadow_training.skip_completed: true` skips a job only when its
workdir contains every file listed in `shadow_training.completed_files`.
Use `--force` to rerun selected jobs even if completion files exist.

The queue order is deterministic:

1. `target_model`, trained on `D_mem` and validated/tested on `D_non`.
2. For each target in `split_plan["targets"]`, all `IN` models in order.
3. For the same target, all `OUT` models in order.

`shadow_training.gpu_nodes` controls local worker count. `gpu_nodes: 1` runs
in-process sequentially. Values greater than one launch bounded subprocess
workers and assign one logical device per worker with `CUDA_VISIBLE_DEVICES`.
