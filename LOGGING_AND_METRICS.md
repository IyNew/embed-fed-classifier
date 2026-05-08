# Logging and Metrics Output

This document describes how round information, training progress, and metrics are recorded for the centralized and federated EMBED classifier workflows.

## Federated Output Locations

Federated outputs are written under the configured `workdir` in `federated/config.yml`:

```yaml
workdir: "./federated_runs/workdir_exp_manifest_fedavg"
```

The active output layout is:

```text
federated_runs/workdir_exp_manifest_fedavg/
  metrics.json
  server/
    log.txt
    log_error.txt
    log_fl.txt
    log.json
  site-1/
    metrics.json
    log.txt
    log_error.txt
    log_fl.txt
    log.json
  site-2/
    metrics.json
    ...
  site-5/
    metrics.json
    ...
  site-6/
    metrics.json
    ...
```

If the job is launched with shell redirection, for example:

```bash
python federated/job.py -c federated/config.yml >> federated_runs/workdir_exp_manifest_fedavg.log 2>&1
```

then `federated_runs/workdir_exp_manifest_fedavg.log` captures launcher stdout/stderr, model weight downloads, and console log output in addition to the NVFlare workspace logs.

## Federated Server Logging

`federated/job.py` creates the NVFlare `BaseFedJob` and `FedAvg` controller. NVFlare writes server-side lifecycle events to:

```text
<workdir>/server/log.txt
<workdir>/server/log_error.txt
<workdir>/server/log_fl.txt
<workdir>/server/log.json
```

Typical server log events include:

- FedAvg start and completion.
- Round start banners.
- Sampled clients for each round.
- Task dispatch to client sites.
- Aggregation progress, such as how many client updates have arrived.
- Fatal errors, task aborts, or min-response failures.

When resume mode is enabled, `federated/job.py` also logs the checkpoint path, completed round, `round_offset`, remaining rounds, and total target rounds.

## Federated Round Numbering

The server passes the resume offset to NVFlare:

```python
FedAvg(..., start_round=resume_state.get("round_offset", 0))
```

Each client receives the current round through `input_model.current_round` in `federated/client.py`:

```python
nvflare_round = int(input_model.current_round)
effective_round = nvflare_round
effective_total_rounds = total_target_rounds or (round_offset + int(input_model.total_rounds))
```

Because NVFlare includes `start_round` in `current_round`, `effective_round` is the absolute round number. For a fresh run it starts at `0`. For a resumed run, it starts at the configured `round_offset`.

Each client logs round identity at the start of training:

```text
(site-1) current_round=0, total_rounds=50 (nvflare_round=0, resumed_rounds=50)
```

The same round fields are also stored in each site metrics record:

```json
{
  "round": 0,
  "nvflare_round": 0,
  "total_rounds": 50,
  "resumed_total_rounds": 50,
  "local_epochs": 5,
  "epochs": []
}
```

## Federated Client Logging

Each site writes logs under its site directory:

```text
<workdir>/site-1/log.txt
<workdir>/site-1/log_error.txt
<workdir>/site-1/log_fl.txt
<workdir>/site-1/log.json
```

Client logs include:

- Subprocess startup and NVFlare initialization.
- RAM and GPU state from `log_system_state(...)`.
- Dataset counts for train, validation, and test splits.
- Class weights.
- Loss function, optimizer, and metric selection.
- Round start and end markers.
- Local epoch progress.
- Validation metrics at configured validation intervals.
- Model checkpoint save paths.
- Global test metrics from `site-1` before local training for rounds greater than `0`.

The system-state lines are useful for diagnosing memory failures:

```text
[sysinfo round_0_start (site-1)] RAM ... | GPU[0] ... | VRAM ...
[sysinfo round_0_end (site-1)] RAM ... | GPU[0] ... | VRAM ...
```

## Federated Metric Calculation

`federated/client.py` computes metrics with `calculate_metrics(...)`.

For labels `[0, 1]`, where `0 = benign` and `1 = malignant`, it records:

| Key | Meaning |
| --- | --- |
| `balanced_accuracy` | `sklearn.metrics.balanced_accuracy_score(labels, predictions)` |
| `specificity` | `TN / (TN + FP)` |
| `sensitivity` | `TP / (TP + FN)` |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]` |
| `loss` | Average loss, when available |
| `auc` | ROC AUC, when probabilities are available during evaluation |

Training epochs record loss, balanced accuracy, specificity, sensitivity, confusion matrix, epoch number, and duration. Validation/test evaluation additionally includes AUC.

## Per-Site Metrics JSON

Each client initializes and updates:

```text
<workdir>/site-<N>/metrics.json
```

The top-level structure is:

```json
{
  "site": "site-1",
  "client_site": "1",
  "device": "cuda:0",
  "data": {
    "train": {"total": 1681, "benign": 1130, "malignant": 551},
    "validation": {"total": 223, "benign": 159, "malignant": 64},
    "test": {"total": 1167, "benign": 843, "malignant": 324}
  },
  "rounds": [],
  "global_test": []
}
```

For each completed local round, the client appends one record to `rounds`:

```json
{
  "round": 0,
  "nvflare_round": 0,
  "total_rounds": 50,
  "resumed_total_rounds": 50,
  "local_epochs": 5,
  "epochs": [
    {
      "epoch": 1,
      "loss": 0.7687,
      "balanced_accuracy": 0.5108,
      "specificity": 0.2956,
      "sensitivity": 0.7260,
      "confusion_matrix": [[334, 796], [151, 400]],
      "duration_seconds": 36.857,
      "duration": "00:00:37"
    }
  ],
  "duration_seconds": 180.0,
  "duration": "00:03:00"
}
```

When an epoch aligns with `val_interval`, that epoch also contains:

```json
{
  "validation": {
    "balanced_accuracy": 0.35,
    "specificity": 0.44,
    "sensitivity": 0.26,
    "confusion_matrix": [[...], [...]],
    "loss": 0.8,
    "auc": 0.35
  }
}
```

The site metrics file is saved at startup, after global test evaluation, and after each completed round. The code also calls `save_site_metrics(...)` inside the epoch loop, but the current `round_metrics` object is appended to `site_metrics["rounds"]` only after the round finishes.

## Global Test Metrics

During federated training, only `site-1` evaluates the current global model on the test loader before local training, and only for rounds greater than `0`:

```python
if effective_round > 0 and client_id == "site-1":
    test_metrics = evaluate(...)
    site_metrics["global_test"].append(test_metrics)
```

Each global test record includes the same evaluation metrics plus:

```json
{
  "round": 1,
  "nvflare_round": 1
}
```

In manifest mode, the test loader can be global or local depending on `federated/client_config.yml`:

```yaml
test_scope: "global"
```

With `global`, each client test set contains all manifest test rows. With `local`, it contains only rows for that client site.

## Metrics Sent Back to NVFlare

After each local training task, the client sends an `FLModel` back to NVFlare. Its `metrics` field contains the latest validation metrics from that round:

```python
{
    "validation": balanced_accuracy,
    "accuracy": balanced_accuracy,
    "balanced_accuracy": balanced_accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "auc": auc,
}
```

These metrics are available to NVFlare during task result handling and aggregation. The full local epoch history remains in the per-site `metrics.json` files.

## Aggregated Federated Metrics JSON

After the simulator finishes, `federated/job.py` calls `write_run_metrics(...)` and writes:

```text
<workdir>/metrics.json
```

This file combines run-level metadata, all site metrics, and global test records:

```json
{
  "summary": {
    "recipe": "fedavg",
    "num_rounds": 50,
    "executed_rounds": 50,
    "client_list": [1, 2, 5, 6],
    "data_csv": "...",
    "cleaned_data_root": "...",
    "workdir": "...",
    "simulator": {
      "threads": 1,
      "gpu": "0",
      "executor_mode": "external_per_task",
      "memory_gc_rounds": 1,
      "cuda_empty_cache": true
    },
    "resume": {
      "enabled": false,
      "round_offset": 0,
      "executed_rounds": 50,
      "total_target_rounds": 50
    },
    "split_counts_by_site": {
      "site-1": {"train": 1681, "validation": 223, "test": 552}
    }
  },
  "duration_seconds": 3866.24,
  "duration": "01:04:26",
  "average_round_seconds": 77.32,
  "average_round_duration": "00:01:17",
  "sites": {
    "site-1": {"...": "..."}
  },
  "global_test": []
}
```

`sites` is a copy of each `<workdir>/site-<N>/metrics.json`. `global_test` is a flattened list of all site `global_test` records with an added `site` field.

## Resume Behavior

When resuming from a checkpoint, `round_offset` prevents duplicate metrics for rounds that are being rerun. `load_or_create_site_metrics(...)` loads the existing site metrics and calls `filter_resume_metrics(...)`, which keeps only records with:

```python
record["round"] < round_offset
```

New resumed rounds are appended after that point.

## Centralized Logging and Metrics

Centralized training writes outputs under the configured `workdir` in `centralized/centralized_config.yml`, for example:

```text
centralized_runs/convnext_tiny/
  resolved_config.yml
  best_model.pth
  last_model.pth
  metrics.json
```

`centralized/train_centralized.py` prints one progress line per epoch:

```text
Epoch 001/050 train_loss=... train_bal_acc=... val_bal_acc=... epoch_time=...
```

The script itself writes `metrics.json` at the end of training. If centralized training is launched with stdout/stderr redirection, those printed progress lines can also be captured in a file such as `train.log`.

The centralized `metrics.json` contains:

```json
{
  "summary": {
    "train": {"total": 0, "benign": 0, "malignant": 0},
    "validation": {"total": 0, "benign": 0, "malignant": 0},
    "test": {"total": 0, "benign": 0, "malignant": 0}
  },
  "class_weights": [],
  "device": "cuda:0",
  "epochs": [
    {
      "epoch": 1,
      "train": {"loss": 0.0, "balanced_accuracy": 0.0},
      "validation": {"loss": 0.0, "balanced_accuracy": 0.0},
      "duration_seconds": 0.0
    }
  ],
  "best_epoch": 1,
  "best_validation": {},
  "test": {},
  "best_model_test": {},
  "duration_seconds": 0.0,
  "duration": "00:00:00",
  "average_epoch_seconds": 0.0,
  "average_epoch_duration": "00:00:00"
}
```

The centralized metrics use the same core metric definitions as federated training: balanced accuracy, specificity, sensitivity, confusion matrix, loss, and AUC when probabilities are available.

## Quick Inspection Commands

Inspect the active federated server log:

```bash
tail -n 100 federated_runs/workdir_exp_manifest_fedavg/server/log.txt
```

Inspect one site metrics file:

```bash
python -m json.tool federated_runs/workdir_exp_manifest_fedavg/site-1/metrics.json | less
```

Inspect the aggregate federated metrics file:

```bash
python -m json.tool federated_runs/workdir_exp_manifest_fedavg/metrics.json | less
```

Inspect centralized metrics:

```bash
python -m json.tool centralized_runs/convnext_tiny/metrics.json | less
```
