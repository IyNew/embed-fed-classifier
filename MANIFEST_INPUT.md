# Manifest CSV Input Workflow

This repository can train the EMBED classifier from a prepared manifest CSV instead of scanning physical train/test folders. The manifest is the source of truth for split assignment, labels, client/site assignment, and relative image paths.

## Configuration

Both training entry points read the manifest path and image root from YAML:

```yaml
data_csv: "/home/wenytang/FL_projects/embed_data_full/data_handling_auto/model_data/first_model_trainig.csv"
cleaned_data_root: "/mnt/c/Data/EMBED/embed_cleaned"
splits:
  train: "train"
  validation: "validation"
  test: "test"
```

For centralized training these keys live in `centralized/centralized_config.yml`. For federated training they live in `federated/config.yml`, while per-client dataset settings such as `test_scope`, `dataset`, `cache_rate`, transforms, and dataloader settings live in `federated/client_config.yml`.

When `data_csv` is set, the manifest workflow is used. The legacy folder-scanning paths are used only when `data_csv` is empty.

## Required Columns

The manifest must contain these columns:

| Column | Purpose |
| --- | --- |
| `model_split` | Logical split assignment. Current configured values are `train`, `validation`, and `test`. |
| `binary_label` | Class label. Supported values are `benign` and `malignant`. |
| `image_path_suffix` | Preferred relative image path under `cleaned_data_root`. |
| `output_relpath` | Fallback relative image path when `image_path_suffix` is blank or unavailable. |
| `loc_num` | Site/client identifier used by federated training. |

Centralized training requires `model_split`, `binary_label`, and at least one path column. Federated training additionally requires `loc_num`.

Use `model_split` for split assignment. Do not infer the split from physical folder names, because a row with `model_split=test` can still point to a path containing `train`.

## Path Resolution

Each row is converted into an absolute image path with this precedence:

1. Use `image_path_suffix` if present and non-empty.
2. Otherwise use `output_relpath` if present and non-empty.
3. Join the selected relative path to `cleaned_data_root`.

For example:

```text
cleaned_data_root / image_path_suffix
cleaned_data_root / output_relpath
```

The loaders verify that the manifest exists and that every referenced image file exists before training starts. Missing path columns, missing image files, unexpected labels, or empty splits raise an error instead of silently dropping data.

## Label Mapping

Labels are normalized by lowercasing and stripping whitespace, then mapped to integer class IDs:

```text
benign -> 0
malignant -> 1
```

These integer labels are stored in the MONAI record dictionaries and passed to `CrossEntropyLoss` and the metric code.

## Centralized Flow

`centralized/train_centralized.py` loads the full manifest in `load_manifest(config)`:

1. Read all rows with `csv.DictReader`.
2. Validate required columns and configured split values.
3. Validate `binary_label` values against `benign` and `malignant`.
4. Convert each row into:

   ```python
   {"image": "/absolute/path/to/image.nii.gz", "label": 0 or 1}
   ```

5. Group records by the configured logical split names: `train`, `validation`, and `test`.
6. Build MONAI `Dataset` or `CacheDataset` instances from those record lists.
7. Apply training transforms to train records and evaluation transforms to validation/test records.
8. Feed the datasets into PyTorch/MONAI dataloaders for model training and evaluation.

The centralized loader uses all rows for their declared split and does not filter by `loc_num`.

## Federated Flow

The federated workflow uses the same manifest, but the server and clients use it for different purposes.

### Server Assignment

`federated/job.py` reads `federated/config.yml` and uses `client_list` to define participating sites:

```yaml
client_list: [1, 2, 5, 6]
```

When `data_csv` is configured, `build_manifest_client_cases(config, logger)`:

1. Loads the manifest.
2. Validates `loc_num`, `model_split`, `binary_label`, and path columns.
3. Filters rows to sites listed in `client_list`.
4. Logs train/validation/test counts for each site.
5. Passes `--data_csv`, `--cleaned_data_root`, and `--client_site <site>` to each client script.

The server does not pass physical case paths to clients in manifest mode. Each client re-reads the manifest and filters its own rows.

### Client Loading

`federated/client.py` loads records in `load_manifest_records(data_csv, cleaned_data_root, client_site, local_config)`:

1. Read all manifest rows.
2. For `train` rows, keep only rows where `loc_num == client_site`.
3. For `validation` rows, keep only rows where `loc_num == client_site`.
4. For `test` rows, behavior depends on `federated/client_config.yml`:

   ```yaml
   test_scope: "global"
   ```

   `global` keeps all manifest test rows for every client. `local` restricts test rows to the current `client_site`.

5. Convert each kept row into the MONAI record format:

   ```python
   {"image": "/absolute/path/to/image.nii.gz", "label": 0 or 1}
   ```

6. Validate that train, validation, and test records are non-empty for the client.
7. Create MONAI `Dataset` or `CacheDataset` objects and dataloaders.

Class weights for the local loss are computed from that client's training records.

## Image Shape Handling

The prepared EMBED NIfTI slices are grayscale 2D images stored with a singleton depth dimension. The transform pipeline loads images with MONAI `LoadImaged`, applies intensity scaling and orientation, then uses `SqueezeSingletonDepthd` to remove the singleton depth axis before 2D resizing and augmentation.

The final dataloader batches contain dictionaries with:

```python
{
    "image": tensor,  # transformed image batch
    "label": tensor,  # class IDs: 0 benign, 1 malignant
}
```

## Useful Validation Commands

Centralized manifest/path validation without training:

```bash
python centralized/train_centralized.py -c centralized/centralized_config.yml --check-data-only
```

Federated syntax check without data access:

```bash
python -m py_compile federated/client.py federated/job.py centralized/train_centralized.py model.py utils.py
```

For a new machine, update `data_csv`, `cleaned_data_root`, and any output workdirs in the YAML files before running training.
