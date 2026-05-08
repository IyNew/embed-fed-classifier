# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python workflow for federated EMBED image classification with NVFlare, MONAI, and PyTorch.

- `job.py` is the federated job launcher. It reads server configuration, builds the initial model, assigns client scripts, and runs the simulator.
- `client.py` contains client training, evaluation, data loading, transforms, and FL send/receive logic.
- `model.py` defines selectable architectures: ConvNeXt variants, ViT, and `SimpleNetwork`.
- `utils.py` contains configuration, seeding, transforms, and collation helpers.
- `config.yml` is the server/job configuration. `client_config.yml` controls datasets, optimization, transforms, and personalization.
- `/home/wenytang/nvflare_example/venv` is the project virtual environment. `environment.txt` is a conda environment export kept for reference. There is currently no `tests/` directory or checked-in sample data.

## Build, Test, and Development Commands

- `source /home/wenytang/nvflare_example/venv/bin/activate`: activate the project virtual environment before running scripts.
- `/home/wenytang/nvflare_example/venv/bin/python job.py -c config.yml`: run the configured NVFlare federated simulation.
- `/home/wenytang/nvflare_example/venv/bin/python client.py --client_cases CASE_ID --client_config_path client_config.yml`: run a client script directly; provide comma-separated case IDs for multiple cases.
- `/home/wenytang/nvflare_example/venv/bin/python -m py_compile client.py job.py model.py utils.py`: perform a lightweight syntax check without data access.

Update absolute paths in `config.yml` and `client_config.yml` before running on a new machine.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Keep functions and variables in `snake_case`, classes in `PascalCase`, and constants such as `DEVICE` in `UPPER_CASE`. Prefer YAML configuration. Keep model additions inside `model.py` and expose them through `get_model(model_args)`.

## Testing Guidelines

No automated test suite is currently present. For changes that do not require data, run `/home/wenytang/nvflare_example/venv/bin/python -m py_compile client.py job.py model.py utils.py`. For training or data-path changes, run a small simulation with reduced `num_rounds`, `local_epochs`, and `client_list`. New tests should live under `tests/` and use `test_*.py` naming.

## Commit & Pull Request Guidelines

The Git history uses short summaries such as `env file` and `initial commit`. Keep commits concise and focused, for example `add convnext dropout config` or `fix client evaluation metrics`.

Pull requests should include experiment or bug context, changed configuration keys, commands run, and metric impact. Include paths to generated models or logs when relevant, but do not commit datasets, checkpoints, credentials, or machine-specific work directories.

## Security & Configuration Tips

Treat EMBED data paths, metadata files, and trained weights as local artifacts unless explicitly approved for sharing. Keep machine-specific absolute paths in YAML files documented, and avoid adding PHI, credentials, or large generated outputs to the repository.
