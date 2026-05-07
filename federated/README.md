# Federated Training Pipeline

This directory contains the NVFlare local simulation pipeline for EMBED federated classification.

## Entry Points

- `job.py`: builds the FedAvg job, resolves manifest paths, configures simulator execution, launches NVFlare local simulation, and writes run-level `metrics.json`.
- `client.py`: runs one site training task, loads manifest rows for that site, evaluates validation/test metrics, saves per-site metrics, and returns model weights to NVFlare.
- `config.yml`: server, simulator, model, manifest, output, and optional soft-resume settings.
- `client_config.yml`: client-side data loading, transforms, loss, optimizer, metric, and local epoch settings.

## Data Input

The default input is the prepared manifest:

`/home/wenytang/FL_projects/embed_data_full/data_handling_auto/model_data/first_model_trainig.csv`

Images are resolved under:

`/mnt/c/Data/EMBED/embed_cleaned`

The manifest path columns are preferred in this order:

1. `image_path_suffix`
2. `output_relpath`

Splits come from `model_split`, not physical folders. Labels use `binary_label` with `benign -> 0` and `malignant -> 1`. Client ownership uses `loc_num`; the global test set uses all manifest test rows unless `test_scope` is changed in `client_config.yml`.

## Simulator And Cleanup

The default simulator mode is now:

`simulator.executor_mode: external_per_task`

This starts each client task in a fresh external Python process and exits it after `flare.send(...)`. That gives every FL task a process-level cleanup boundary for Python imports, MONAI dataloaders, CUDA contexts, and PyTorch allocator state. The client also runs `gc.collect()` and `torch.cuda.empty_cache()` before the per-task process exits.

The previous `in_process` mode can still be selected in `config.yml`, but it has shown instability after several rounds with this workload.

## Outputs

Default outputs are under:

`federated_runs/workdir_exp_manifest_fedavg`

The run writes:

- server and site NVFlare logs
- per-site checkpoints named `embed_net_round_<round>_epoch_<epoch>.pth`
- per-site `metrics.json`
- final run-level `metrics.json` with duration, split counts, simulator settings, resume metadata, and global test records

Progressive stdout/stderr should be redirected to:

`federated_runs/workdir_exp_manifest_fedavg.log`
