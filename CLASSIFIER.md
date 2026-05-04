# Classifier and Training Notes

## Classifier

The project currently uses a binary image classifier configured as `ConvNeXtTiny` in `client_config.yml`:

```yaml
model:
  name: 'ConvNeXtTiny'
  num_classes: 2
  pretrained: true
  dropout: 0.5
```

The implementation in `model.py` starts from `torchvision.models.convnext_tiny` with ImageNet weights. The first convolution is adapted from RGB to one input channel by averaging the pretrained RGB weights. The original classifier is replaced with `Flatten`, `BatchNorm1d`, `ReLU`, `Dropout`, and a final linear head with 2 output logits.

The output classes are inferred as binary labels: `0` for paths containing `benign`, and `1` for all other paths, which are treated as malignant or positive.

## Training Settings

Local training settings come from `client_config.yml` and are applied in `client.py`.

```yaml
dataset: Dataset
dataloader:
  batch_size: 128
  num_workers: 10
  shuffle: true
  pin_memory: true

local_epochs: 5
val_interval: 5

optimizer:
  name: 'Adam'
  lr_backbone: 0.00001
  lr_finetune: 0.0001
  weight_decay: 0.00005

loss:
  name: 'CrossEntropyLoss'
  label_smoothing: 0.1
  class_weights: true
```

The optimizer uses two parameter groups: backbone parameters use `lr_backbone`, while the classifier head and first convolution use `lr_finetune`. Class weights are computed from the local benign and malignant case counts. Reported metrics include balanced accuracy, specificity, sensitivity, and AUC.

## Expected Input Data

Training and test roots are configured in YAML:

```yaml
train_dataset_path: "/path/to/train/original"
test_dataset_path: "/path/to/test"
```

`client.py` recursively scans files under these roots and builds MONAI records:

```python
{"image": image_path, "label": 0 if "benign" in str(image_path) else 1}
```

Images must be readable by MONAI `LoadImaged`. The active transform pipeline loads channel-first images, scales intensity, and orients images to `RAS`. Training also applies random flips, rotations, and affine augmentation.

The classifier expects grayscale image tensors shaped like:

```text
[batch, 1, height, width]
```

The configured data paths indicate the intended input resolution is 256x256. `client_config.yml` contains crop and resize settings, but those transforms are not currently included in the active `client.py` transform list.
