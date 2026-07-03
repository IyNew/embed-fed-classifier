# Shadow Model Loss Collection

## Purpose

For membership inference attacks (MIA), collect per-sample cross-entropy losses on the 20 target samples from:
- The target model (trained on D_mem, 3364 samples)
- 16 IN shadow models per target (trained on S_XX + target sample, size 501)
- 16 OUT shadow models per target (trained on S_XX+16, target excluded, size 500)

## Script

```
python shadow/collect_losses.py [-c config.yml] [--model-type last|best]
```

- Default: `shadow/experiments/full_shadow_training.yml`, uses `last_model.pth`
- Resumable: re-running skips rows already present in the output CSV

## Output

**File:** `shadow_runs/full_shadow_training/loss_collection.csv`

**Columns:**

| Column | Description |
|---|---|
| `target_id` | `target_x_01`..`target_x_20` |
| `membership` | `member` or `nonmember` (ground truth from split_plan) |
| `model_category` | `target`, `in`, or `out` |
| `model_index` | `01`..`16` for shadow models, empty for target |
| `loss` | CrossEntropyLoss (label_smoothing=0.1, no class weights) |
| `logit_benign` | Raw logit for class 0 |
| `logit_malignant` | Raw logit for class 1 |
| `confidence_benign` | Softmax(0) |
| `confidence_malignant` | Softmax(1) |
| `label` | Ground truth (0=benign, 1=malignant) |
| `predicted` | Argmax prediction |
| `correct` | `True` if `predicted == label` |

**Total:** 660 rows (20 targets × 1 target model + 320 IN + 320 OUT)

## Loss Function

- `CrossEntropyLoss(label_smoothing=0.1)` matching the training config
- No class weights applied (for consistent loss comparison across models with different train sets)

## Evaluation Transforms

Same eval pipeline as training (no augmentation):
1. LoadImaged (ensure_channel_first=True)
2. ScaleIntensityd
3. Orientationd (RAS)
4. SqueezeSingletonDepthd
5. ResizeWithPadOrCropd (256×256)

## Usage for MIA

The CSV is structured for membership inference:

```python
import pandas as pd
df = pd.read_csv("shadow_runs/full_shadow_training/loss_collection.csv")

# Per-target aggregation for MIA analysis
target_losses = df[df["model_category"] == "target"]
shadow_losses = df[df["model_category"] != "target"]

# Mean IN/OUT loss per target
mia_stats = shadow_losses.groupby(["target_id", "membership", "model_category"])["loss"].mean()
```

Run log: `shadow_runs/full_shadow_training/loss_collection.log`
