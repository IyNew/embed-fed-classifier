# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python workflows for EMBED image classification with NVFlare, MONAI, and PyTorch. Federated and centralized training entry points are kept in separate directories, with shared model/config utilities at the repository root.

- `federated/job.py` is the federated job launcher. It reads server configuration, builds the initial model, assigns client scripts, and runs the simulator.
- `federated/client.py` contains client training, evaluation, manifest/path data loading, transforms, and FL send/receive logic.
- `federated/config.yml` is the server/job configuration. `federated/client_config.yml` controls datasets, optimization, transforms, and personalization.
- `centralized/train_centralized.py` is the standalone centralized trainer/evaluator.
- `centralized/centralized_config.yml` controls the centralized manifest input, model, optimizer, transforms, epochs, and output directory.
- `model.py` defines selectable architectures: ConvNeXt variants, ViT, and `SimpleNetwork`.
- `utils.py` contains configuration, seeding, transforms, and collation helpers.
- `environment.txt` is a conda environment export. There is currently no `tests/` directory or checked-in sample data.

## Data Input

The current primary input format is the prepared manifest CSV:

- `data_csv`: `/home/wenytang/FL_projects/embed_data_full/data_handling_auto/model_data/first_model_trainig.csv`
- `cleaned_data_root`: `/mnt/c/Data/EMBED/embed_cleaned`

Both pipelines expect manifest columns:

- `model_split`: split assignment. Use the configured split names, currently `train`, `validation`, and `test`.
- `binary_label`: `benign` or `malignant`, mapped to labels `0` and `1`.
- `image_path_suffix`: preferred relative path under `cleaned_data_root`.
- `output_relpath`: fallback relative path when `image_path_suffix` is unavailable.
- `loc_num`: site/client identifier for federated training.

Use `model_split` for split assignment rather than physical folder names, because rows with `model_split=test` can still point to a path containing `train`. The prepared NIfTI slices are grayscale 2D images stored with a singleton depth dimension; training transforms squeeze that singleton axis before 2D resize/augmentation.

For federated manifest input, `federated/job.py` assigns clients by `loc_num` from `client_list`. Each client trains and validates on its own site rows. `federated/client_config.yml` controls whether test evaluation is `global` across all test rows or `local` to the client site.

## Build, Test, and Development Commands

- `conda create --name embed-fed --file environment.txt`: recreate the pinned environment.
- `conda activate embed-fed`: activate the environment before running scripts.
- `python federated/job.py -c federated/config.yml`: run the configured NVFlare federated simulation.
- `python federated/client.py --data_csv DATA_CSV --cleaned_data_root ROOT --client_site SITE --client_config_path federated/client_config.yml`: run a client script directly with manifest input.
- `python centralized/train_centralized.py -c centralized/centralized_config.yml`: run centralized training.
- `python centralized/train_centralized.py -c centralized/centralized_config.yml --check-data-only`: validate the centralized manifest/path compatibility without training.
- `python -m py_compile federated/client.py federated/job.py centralized/train_centralized.py model.py utils.py`: perform a lightweight syntax check without data access.

Update absolute paths in `federated/config.yml`, `federated/client_config.yml`, and `centralized/centralized_config.yml` before running on a new machine.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Keep functions and variables in `snake_case`, classes in `PascalCase`, and constants such as `DEVICE` in `UPPER_CASE`. Prefer YAML configuration. Keep model additions inside `model.py` and expose them through `get_model(model_args)`.

## Testing Guidelines

No automated test suite is currently present. For changes that do not require data, run `python -m py_compile federated/client.py federated/job.py centralized/train_centralized.py model.py utils.py`. For training or data-path changes, run a small simulation with reduced `num_rounds`, `local_epochs`, and `client_list`, or run centralized training with `--epochs 1 --num-workers 0 --batch-size 4`. New tests should live under `tests/` and use `test_*.py` naming.

## Experiment Archive Protocol

When starting any experiment run or sweep, keep a global log following the centralized training process pattern. Create the log before launching jobs, append the experiment name, timestamp, config path(s), command, output/work directories, and each run's start/completion/failure status as it happens. Keep this log outside the individual model workdirs so it can be archived with the completed results.

When archiving experiment results, use a date-name path under `centralized_runs/archive/YYYY-MM-DD/<descriptive_name>_<timestamp>/`. Move the completed run directories and the global sweep log into that folder. Include a `configs/` subdirectory containing exact YAML snapshots for every config used in the run, even if the active config files will also be moved elsewhere.

Write an `insights.md` file in the archive root before closing the task. It should include the run timestamp, which configs completed or failed, the key validation/test metrics, privacy values when applicable, and a short interpretation of what the results imply for the next sweep.

When the user asks to archive results, update `centralized_runs/results_summary.csv` for every successfully finished run being archived. Include the archive path, run name, status, key training settings, DP settings, validation/test metrics, privacy values, and leave the `Note` field available for manual annotations.

After archiving, clean the active workspace: remove or move the run artifacts from the top level of `centralized_runs`, and move completed active YAMLs from `centralized/experiment_configs/` into `centralized/experiment_configs/archive/` unless the user asks to keep them active. Verify no training or watchdog processes still point at moved logs before or after moving files.

## Commit & Pull Request Guidelines

The Git history uses short summaries such as `env file` and `initial commit`. Keep commits concise and focused, for example `add convnext dropout config` or `fix client evaluation metrics`.

Pull requests should include experiment or bug context, changed configuration keys, commands run, and metric impact. Include paths to generated models or logs when relevant, but do not commit datasets, checkpoints, credentials, or machine-specific work directories.

## Security & Configuration Tips

Treat EMBED data paths, metadata files, and trained weights as local artifacts unless explicitly approved for sharing. Keep machine-specific absolute paths in YAML files documented, and avoid adding PHI, credentials, or large generated outputs to the repository.
