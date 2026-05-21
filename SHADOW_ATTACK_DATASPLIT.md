# Shadow Attack CSV Input Protocol

This shadow attack workflow uses the prepared manifest CSV as the source of truth. It does not infer labels, splits, or membership from physical folders.

## Source Inputs

Use the same manifest columns as centralized training:

| Column | Purpose |
| --- | --- |
| `binary_label` | `benign` or `malignant`, mapped to labels `0` and `1`. |
| `image_path_suffix` | Preferred relative NIfTI path under `cleaned_data_root`. |
| `output_relpath` | Fallback relative path when `image_path_suffix` is blank. |
| `uid` | Preferred stable sample identifier for derived attack split metadata. |

The active manifest currently contains 5,608 rows, so the original 6,000-row pool sizes are scaled for the real CSV.

## Pool Definition

The derived shadow attack manifest partitions all rows into mutually exclusive pools:

| Pool | Count | Purpose |
| --- | ---: | --- |
| `D_mem` | 3,364 | Target model training data and source of member targets. |
| `D_non` | 1,122 | Samples unseen by the target model; source of non-member targets and centralized eval. |
| `D_aux` | 1,122 | Adversary auxiliary data used for shadow model training. |

Sampling is deterministic from the YAML `seed` and stratified by `binary_label`.

## Target and Shadow Sets

The split generator selects:

| Item | Count | Source |
| --- | ---: | --- |
| `T_mem` | 10 | Stratified sample from `D_mem`. |
| `T_non` | 10 | Stratified sample from `D_non`. |
| `Pool_shadow` | 800 | Stratified sample from `D_aux`. |
| `S_1 ... S_32` | 500 each | Sampled with replacement from `Pool_shadow`. |

For each target sample `x_i`, the shadow training sets are:

| Condition | Subsets | Train composition |
| --- | --- | --- |
| `IN` | `S_1 ... S_16` | `S_j + {x_i}` for 501 records. |
| `OUT` | `S_17 ... S_32` | `S_j` for 500 records. |

Target samples never overlap with `Pool_shadow` because `D_mem`, `D_non`, and `D_aux` are disjoint.

## Config

Use `centralized/shadow_attack_config.yml` for this workflow. The important additions are:

```yaml
input:
  mode: "shadow_attack"
  attack_manifest: "./centralized_runs/shadow_attack/attack_manifest.csv"
  split_plan: "./centralized_runs/shadow_attack/split_plan.json"
  regenerate: false
  validate_paths: true

shadow_attack:
  pool_counts:
    D_mem: 3364
    D_non: 1122
    D_aux: 1122
  target_counts:
    member: 10
    nonmember: 10
  shadow_pool_size: 800
  shadow_subset_count: 32
  shadow_subset_size: 500
  in_subset_count: 16
  nonmember_validation_fraction: 0.3333333333
```

Existing model, optimizer, DP, transform, dataloader, and training settings keep the same meaning as in centralized training.

## Loader Mapping

The new input module returns the same structure expected by centralized training:

```python
{
    "train": [{"image": "...", "label": 0 or 1}, ...],
    "validation": [{"image": "...", "label": 0 or 1}, ...],
    "test": [{"image": "...", "label": 0 or 1}, ...],
}
```

For target model training:

| Centralized split | Source |
| --- | --- |
| `train` | all `D_mem` rows |
| `validation` | stratified first third of `D_non` |
| `test` | remaining `D_non` rows |

For a shadow model:

| Centralized split | Source |
| --- | --- |
| `train` | selected `IN` or `OUT` shadow training set |
| `validation` | stratified first third of `D_non` |
| `test` | remaining `D_non` rows |

## Commands

Generate or validate the split without training:

```bash
python centralized/train_shadow_attack.py \
  -c centralized/shadow_attack_config.yml \
  --generate-split-only
```

Check data compatibility for the target model and first shadow model:

```bash
python centralized/train_shadow_attack.py \
  -c centralized/shadow_attack_config.yml \
  --check-data-only
```

Train a single shadow model:

```bash
python centralized/train_shadow_attack.py \
  -c centralized/shadow_attack_config.yml \
  --target-id target_x_01 \
  --condition IN \
  --subset-index 1
```

Run a small smoke train:

```bash
python centralized/train_shadow_attack.py \
  -c centralized/shadow_attack_config.yml \
  --target-id target_x_01 \
  --condition IN \
  --subset-index 1 \
  --epochs 1 \
  --num-workers 0 \
  --batch-size 4
```

Running `train_shadow_attack.py` without `--target-id`, `--condition`, or `--subset-index` trains the target model and then all configured shadow models.

## Verification Checklist

- `D_mem`, `D_non`, and `D_aux` are mutually exclusive.
- Pool counts sum to the manifest row count.
- `T_mem` has 10 samples and comes only from `D_mem`.
- `T_non` has 10 samples and comes only from `D_non`.
- `Pool_shadow` has 800 samples and comes only from `D_aux`.
- `T_all` does not overlap with `Pool_shadow`.
- Every generated centralized record resolves through `cleaned_data_root / image_path_suffix` or `cleaned_data_root / output_relpath`.
