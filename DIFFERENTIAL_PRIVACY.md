# Differential Privacy for Centralized Training

This repository supports optional Opacus DP-SGD for the centralized training workflow in `centralized/train_centralized.py`. Federated training does not currently use these settings.

DP is disabled by default. When enabled, Opacus wraps the centralized model, optimizer, and training dataloader. Validation, test evaluation, class summaries, and logged aggregate metrics are still written normally; the DP mechanism protects training updates, not every aggregate in the output files.

## Dependency

Opacus is pinned in `environment.txt`:

```text
opacus=1.5.4=pypi_0
```

Recreate the conda environment or install the package into the active environment:

```bash
python -m pip install opacus==1.5.4
```

If `differential_privacy.enabled: true` and Opacus is missing, the trainer fails with an install message.

## Configuration

The DP block lives in `centralized/centralized_config.yml`:

```yaml
differential_privacy:
  enabled: false
  accountant: "prv"
  secure_mode: false
  mode: "noise_multiplier"
  noise_multiplier: 1.0
  target_epsilon: null
  target_delta: null
  max_grad_norm: 1.0
  poisson_sampling: true
  clipping: "flat"
  grad_sample_mode: "hooks"
  fix_modules: true
```

| Setting | Meaning |
| --- | --- |
| `enabled` | Turns centralized DP-SGD on or off. Default is `false`, which keeps existing non-DP behavior. |
| `accountant` | Opacus privacy accountant. Default is `"prv"`. |
| `secure_mode` | Uses Opacus secure random number generation when `true`. Keep `false` for faster experiments; use `true` for production DP runs. |
| `mode` | DP setup mode. Use `"noise_multiplier"` to provide noise directly, or `"target_epsilon"` to let Opacus choose noise for a target privacy budget. |
| `noise_multiplier` | Gaussian noise multiplier used when `mode: "noise_multiplier"`. |
| `target_epsilon` | Target epsilon used when `mode: "target_epsilon"`. Must be non-null in that mode. |
| `target_delta` | Delta for privacy accounting. If null, the trainer uses `1 / len(train_dataset)`. |
| `max_grad_norm` | Per-sample gradient clipping norm. |
| `poisson_sampling` | Enables Opacus Poisson sampling for the private training dataloader. |
| `clipping` | Opacus clipping strategy. Default is `"flat"`. |
| `grad_sample_mode` | Opacus gradient sampling backend. Default is `"hooks"`. |
| `fix_modules` | Runs Opacus `ModuleValidator.fix_and_validate(...)` before wrapping. Keep this enabled for current ConvNeXt heads because they include `BatchNorm1d`, which Opacus treats as incompatible. |

## Noise Multiplier Mode

Use this mode when you want to set the noise multiplier directly and inspect the resulting epsilon.

```yaml
differential_privacy:
  enabled: true
  accountant: "prv"
  secure_mode: false
  mode: "noise_multiplier"
  noise_multiplier: 1.0
  target_epsilon: null
  target_delta: null
  max_grad_norm: 1.0
  poisson_sampling: true
  clipping: "flat"
  grad_sample_mode: "hooks"
  fix_modules: true
```

Run centralized training:

```bash
python centralized/train_centralized.py -c centralized/centralized_config.yml
```

## Target Epsilon Mode

Use this mode when you want Opacus to estimate a noise multiplier for a target privacy budget.

```yaml
differential_privacy:
  enabled: true
  accountant: "prv"
  secure_mode: false
  mode: "target_epsilon"
  noise_multiplier: 1.0
  target_epsilon: 8.0
  target_delta: null
  max_grad_norm: 1.0
  poisson_sampling: true
  clipping: "flat"
  grad_sample_mode: "hooks"
  fix_modules: true
```

`noise_multiplier` is ignored by the configuration path in this mode; Opacus computes the private optimizer noise internally from `target_epsilon`, `target_delta`, dataloader sampling, and `epochs`.

## Recommended Run Process

1. Validate data paths without training:

   ```bash
   python centralized/train_centralized.py -c centralized/centralized_config.yml --check-data-only
   ```

2. Run a short DP smoke test by reducing the config or using CLI overrides:

   ```bash
   python centralized/train_centralized.py \
     -c centralized/centralized_config.yml \
     --epochs 1 \
     --batch-size 4 \
     --num-workers 0 \
     --workdir ./centralized_runs/dp_smoke
   ```

   Make sure `differential_privacy.enabled: true` in the config used for the smoke run.

3. Inspect the DP metadata:

   ```bash
   python -m json.tool centralized_runs/dp_smoke/metrics.json
   ```

4. Evaluate the saved checkpoint:

   ```bash
   python centralized/train_centralized.py \
     -c centralized/centralized_config.yml \
     --eval-checkpoint ./centralized_runs/dp_smoke/best_model.pth \
     --eval-split test
   ```

5. For a full run, set a dedicated `workdir`, choose either `noise_multiplier` or `target_epsilon`, and consider `secure_mode: true` for the final privacy-relevant training run.

## Outputs

Checkpoint filenames are unchanged:

```text
<workdir>/best_model.pth
<workdir>/last_model.pth
```

When Opacus wraps the model, the trainer saves the underlying module state dict. This keeps `--eval-checkpoint` compatible with both DP and non-DP checkpoints.

`metrics.json` includes a top-level `differential_privacy` object:

```json
{
  "enabled": true,
  "accountant": "prv",
  "secure_mode": false,
  "mode": "noise_multiplier",
  "max_grad_norm": 1.0,
  "poisson_sampling": true,
  "clipping": "flat",
  "grad_sample_mode": "hooks",
  "fix_modules": true,
  "module_fixed": true,
  "target_delta": 0.00025886616515661404,
  "noise_multiplier": 1.0,
  "final_epsilon": 3.2,
  "per_epoch_epsilon": [
    {"epoch": 1, "epsilon": 1.1}
  ]
}
```

The exact epsilon depends on dataset size, batch size, epochs, sampling, accountant, delta, clipping, and noise settings.

