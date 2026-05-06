import argparse
import csv
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABEL_MAP = {"benign": 0, "malignant": 1}
REQUIRED_COLUMNS = {"model_split", "binary_label"}
PATH_COLUMNS = ("image_path_suffix", "output_relpath")


class SqueezeSingletonDepthd:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        import torch

        output = dict(data)
        for key in self.keys:
            image = output[key]
            if image.ndim == 4 and image.shape[-1] == 1:
                output[key] = torch.squeeze(image, dim=-1)
            elif image.ndim == 3 and image.shape[0] != 1 and image.shape[-1] == 1:
                output[key] = torch.movedim(torch.squeeze(image, dim=-1), -1, 0)
        return output


def parse_args():
    parser = argparse.ArgumentParser(description="Train a centralized EMBED ConvNeXt classifier.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(Path(__file__).with_name("centralized_config.yml")),
    )
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained", choices=["true", "false"], default=None)
    parser.add_argument("--eval-checkpoint", type=str, default=None)
    parser.add_argument("--eval-split", choices=["validation", "test"], default="test")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, config_path):
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def apply_overrides(config, args):
    effective = deepcopy(config)
    if args.epochs is not None:
        effective["epochs"] = args.epochs
    if args.batch_size is not None:
        effective.setdefault("dataloader", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        effective.setdefault("dataloader", {})["num_workers"] = args.num_workers
    if args.workdir is not None:
        effective["workdir"] = args.workdir
    if args.seed is not None:
        effective["seed"] = args.seed
    if args.pretrained is not None:
        effective.setdefault("model", {})["pretrained"] = args.pretrained == "true"
    return effective


def set_seed(seed):
    import torch
    from monai.utils import set_determinism
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    set_determinism(seed=seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_path_value(row):
    for column in PATH_COLUMNS:
        if column in row and row[column] is not None and str(row[column]).strip():
            return str(row[column]).strip()
    return None


def load_manifest(config):
    data_csv = Path(config["data_csv"])
    cleaned_root = Path(config["cleaned_data_root"])
    if not data_csv.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {data_csv}")

    with open(data_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    missing_columns = REQUIRED_COLUMNS - fieldnames
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")
    if not any(column in fieldnames for column in PATH_COLUMNS):
        raise ValueError("Manifest must include image_path_suffix or output_relpath.")

    split_values = set(config.get("splits", {}).values())
    found_splits = {str(row["model_split"]).strip() for row in rows if row.get("model_split")}
    invalid_splits = found_splits - split_values
    if invalid_splits:
        raise ValueError(f"Unexpected model_split values: {sorted(invalid_splits)}")

    labels = {str(row["binary_label"]).strip().lower() for row in rows if row.get("binary_label")}
    invalid_labels = labels - set(LABEL_MAP)
    if invalid_labels:
        raise ValueError(f"Unexpected binary_label values: {sorted(invalid_labels)}")

    records_by_split = {name: [] for name in config.get("splits", {})}
    missing_files = []
    missing_path_rows = []

    for row_index, row in enumerate(rows, start=2):
        split_value = str(row["model_split"]).strip()
        split_name = next(
            (name for name, value in config["splits"].items() if value == split_value),
            None,
        )
        if split_name is None:
            continue

        relpath = select_path_value(row)
        if relpath is None:
            missing_path_rows.append(row_index)
            continue

        image_path = cleaned_root / relpath
        if not image_path.exists():
            missing_files.append(str(image_path))

        label = LABEL_MAP[str(row["binary_label"]).strip().lower()]
        records_by_split[split_name].append({"image": str(image_path), "label": label})

    if missing_path_rows:
        preview = missing_path_rows[:10]
        raise ValueError(f"Rows missing usable image path columns: {preview}")
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} manifest image files are missing. First missing paths:\n{preview}"
        )

    for split_name, records in records_by_split.items():
        if not records:
            raise ValueError(f"No records found for split '{split_name}'.")

    return records_by_split


def summarize_records(records_by_split):
    summary = {}
    for split_name, records in records_by_split.items():
        labels = [record["label"] for record in records]
        summary[split_name] = {
            "total": len(records),
            "benign": labels.count(0),
            "malignant": labels.count(1),
        }
    return summary


def build_transforms(config, train):
    from monai.transforms import (
        Compose,
        LoadImaged,
        Orientationd,
        RandAffined,
        RandFlipd,
        RandRotated,
        ResizeWithPadOrCropd,
        ScaleIntensityd,
    )

    key = "train_transforms" if train else "eval_transforms"
    transform_config = config[key]
    transforms = [
        LoadImaged(
            keys=transform_config["LoadImaged_im"]["keys"],
            ensure_channel_first=transform_config["LoadImaged_im"]["ensure_channel_first"],
        ),
        ScaleIntensityd(keys=transform_config["ScaleIntensityd"]["keys"]),
        Orientationd(
            keys=transform_config["Orientationd"]["keys"],
            axcodes=transform_config["Orientationd"]["axcodes"],
            as_closest_canonical=transform_config["Orientationd"]["as_closest_canonical"],
        ),
        SqueezeSingletonDepthd(keys=["image"]),
        ResizeWithPadOrCropd(
            keys=transform_config["ResizeWithPadOrCropd"]["keys"],
            spatial_size=transform_config["ResizeWithPadOrCropd"]["spatial_size"],
        ),
    ]

    if train:
        transforms.extend(
            [
                RandFlipd(
                    keys=transform_config["RandFlipd_x"]["keys"],
                    prob=transform_config["RandFlipd_x"]["prob"],
                    spatial_axis=transform_config["RandFlipd_x"]["spatial_axis"],
                ),
                RandFlipd(
                    keys=transform_config["RandFlipd_y"]["keys"],
                    prob=transform_config["RandFlipd_y"]["prob"],
                    spatial_axis=transform_config["RandFlipd_y"]["spatial_axis"],
                ),
                RandRotated(
                    keys=transform_config["RandRotated"]["keys"],
                    prob=transform_config["RandRotated"]["prob"],
                    range_x=transform_config["RandRotated"]["range_x"],
                    range_y=transform_config["RandRotated"]["range_y"],
                    mode=transform_config["RandRotated"]["mode"],
                ),
                RandAffined(
                    keys=transform_config["RandAffined"]["keys"],
                    prob=transform_config["RandAffined"]["prob"],
                    rotate_range=transform_config["RandAffined"]["rotate_range"],
                    scale_range=transform_config["RandAffined"]["scale_range"],
                    mode=transform_config["RandAffined"]["mode"],
                ),
            ]
        )

    return Compose(transforms)


def build_datasets(config, records_by_split):
    from monai.data import CacheDataset, Dataset

    dataset_cls = CacheDataset if config.get("dataset") == "CacheDataset" else Dataset
    train_transforms = build_transforms(config, train=True)
    eval_transforms = build_transforms(config, train=False)

    kwargs = {}
    if dataset_cls is CacheDataset:
        kwargs["cache_rate"] = config.get("cache_rate", 1)

    return {
        "train": dataset_cls(data=records_by_split["train"], transform=train_transforms, **kwargs),
        "validation": dataset_cls(data=records_by_split["validation"], transform=eval_transforms, **kwargs),
        "test": dataset_cls(data=records_by_split["test"], transform=eval_transforms, **kwargs),
    }


def build_loaders(config, datasets):
    from monai.data import DataLoader

    loader_config = config["dataloader"]
    batch_size = loader_config.get("batch_size", 1)
    num_workers = loader_config.get("num_workers", 0)
    pin_memory = loader_config.get("pin_memory", True)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=loader_config.get("shuffle", True),
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def prepare_inputs(batch, device):
    import torch

    inputs = torch.cat([batch["image"]], dim=1).float().to(device)
    while inputs.ndim > 4 and inputs.shape[-1] == 1:
        inputs = torch.squeeze(inputs, dim=-1)
    labels = batch["label"].long().to(device)
    return inputs, labels


def compute_class_weights(train_records, device):
    import torch

    labels = [record["label"] for record in train_records]
    benign = labels.count(0)
    malignant = labels.count(1)
    if benign == 0 or malignant == 0:
        raise ValueError("Both benign and malignant samples are required for class weights.")
    total = len(labels)
    return torch.tensor(
        [total / (2 * benign), total / (2 * malignant)],
        dtype=torch.float32,
        device=device,
    )


def split_parameters(model):
    backbone_params = []
    finetune_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "features.0.0" in name or "head" in name:
            finetune_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, finetune_params


def build_optimizer(config, model):
    import torch.optim as optim

    backbone_params, finetune_params = split_parameters(model)
    optimizer_config = config["optimizer"]
    param_groups = [
        {"params": backbone_params, "lr": optimizer_config.get("lr_backbone")},
        {"params": finetune_params, "lr": optimizer_config.get("lr_finetune")},
    ]

    if optimizer_config["name"] == "Adam":
        return optim.Adam(
            param_groups,
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            eps=optimizer_config.get("epsilon", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 0),
        )
    if optimizer_config["name"] == "SGD":
        return optim.SGD(
            param_groups,
            momentum=optimizer_config.get("momentum", 0),
            weight_decay=optimizer_config.get("weight_decay", 0),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")


def build_criterion(config, class_weights, device):
    import torch.nn as nn

    loss_config = config["loss"]
    if loss_config["name"] != "CrossEntropyLoss":
        raise ValueError(f"Unsupported loss for centralized training: {loss_config['name']}")
    return nn.CrossEntropyLoss(
        label_smoothing=loss_config.get("label_smoothing", 0),
        weight=class_weights if loss_config.get("class_weights", False) else None,
    ).to(device)


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else math.nan


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_metrics(labels, predictions, probabilities, average_loss):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    cm = confusion_matrix(labels_array, predictions_array, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(labels_array, np.asarray(probabilities))
    except ValueError:
        auc = math.nan

    return {
        "loss": float(average_loss),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions_array)),
        "specificity": safe_divide(tn, tn + fp),
        "sensitivity": safe_divide(tp, tp + fn),
        "auc": float(auc),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    import torch

    model.train()
    running_loss = 0.0
    labels_total = []
    predictions_total = []
    probabilities_total = []

    for batch in loader:
        inputs, labels = prepare_inputs(batch, device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probabilities = torch.softmax(outputs.detach(), dim=1)[:, 1]
        predictions = torch.argmax(outputs.detach(), dim=1)
        labels_total.extend(labels.detach().cpu().numpy().tolist())
        predictions_total.extend(predictions.cpu().numpy().tolist())
        probabilities_total.extend(probabilities.cpu().numpy().tolist())

    return calculate_metrics(
        labels_total,
        predictions_total,
        probabilities_total,
        running_loss / max(len(loader), 1),
    )


def evaluate(model, loader, criterion, device):
    import torch

    model.eval()
    running_loss = 0.0
    labels_total = []
    predictions_total = []
    probabilities_total = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = prepare_inputs(batch, device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = torch.argmax(outputs, dim=1)
            labels_total.extend(labels.detach().cpu().numpy().tolist())
            predictions_total.extend(predictions.detach().cpu().numpy().tolist())
            probabilities_total.extend(probabilities.detach().cpu().numpy().tolist())

    return calculate_metrics(
        labels_total,
        predictions_total,
        probabilities_total,
        running_loss / max(len(loader), 1),
    )


def train(config, records_by_split):
    workdir = Path(config["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(workdir / "torch_cache"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import torch

    from model import get_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 0))

    save_config(config, workdir / "resolved_config.yml")

    datasets = build_datasets(config, records_by_split)
    loaders = build_loaders(config, datasets)
    model = get_model(config["model"]).to(device)
    class_weights = compute_class_weights(records_by_split["train"], device)
    criterion = build_criterion(config, class_weights, device)
    optimizer = build_optimizer(config, model)

    best_metric_name = config.get("save_metric", "balanced_accuracy")
    best_metric = -math.inf
    metrics_history = {
        "summary": summarize_records(records_by_split),
        "class_weights": [float(v) for v in class_weights.detach().cpu().numpy().tolist()],
        "device": str(device),
        "epochs": [],
    }

    print(f"Training on {device}. Outputs will be saved to {workdir}")
    print(f"Split summary: {json.dumps(metrics_history['summary'], sort_keys=True)}")

    training_start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        epoch_metrics = {"epoch": epoch, "train": train_metrics}

        if epoch % config.get("val_interval", 1) == 0:
            validation_metrics = evaluate(model, loaders["validation"], criterion, device)
            epoch_metrics["validation"] = validation_metrics
            metric_value = validation_metrics[best_metric_name]
            if not math.isnan(metric_value) and metric_value > best_metric:
                best_metric = metric_value
                torch.save(model.state_dict(), workdir / "best_model.pth")
                metrics_history["best_epoch"] = epoch
                metrics_history["best_validation"] = validation_metrics

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_metrics["duration_seconds"] = epoch_seconds
        metrics_history["epochs"].append(epoch_metrics)
        print(
            f"Epoch {epoch:03d}/{config['epochs']:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} "
            f"val_bal_acc={epoch_metrics.get('validation', {}).get('balanced_accuracy', math.nan):.4f} "
            f"epoch_time={format_duration(epoch_seconds)}"
        )

    torch.save(model.state_dict(), workdir / "last_model.pth")
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    metrics_history["test"] = test_metrics
    best_model_path = workdir / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_test_metrics = evaluate(model, loaders["test"], criterion, device)
        metrics_history["best_model_test"] = best_test_metrics
    total_seconds = time.perf_counter() - training_start
    metrics_history["duration_seconds"] = total_seconds
    metrics_history["duration"] = format_duration(total_seconds)
    metrics_history["average_epoch_seconds"] = total_seconds / max(config["epochs"], 1)
    metrics_history["average_epoch_duration"] = format_duration(
        metrics_history["average_epoch_seconds"]
    )

    with open(workdir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2, allow_nan=True)

    print(f"Final test metrics: {json.dumps(test_metrics, sort_keys=True)}")
    if "best_model_test" in metrics_history:
        print(f"Best model test metrics: {json.dumps(metrics_history['best_model_test'], sort_keys=True)}")
    print(
        f"Total training time: {metrics_history['duration']} "
        f"(avg_epoch={metrics_history['average_epoch_duration']})"
    )
    return metrics_history


def evaluate_checkpoint(config, records_by_split, checkpoint_path, split):
    workdir = Path(config["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(workdir / "torch_cache"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import torch

    from model import get_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 0))

    datasets = build_datasets(config, records_by_split)
    loaders = build_loaders(config, datasets)
    model = get_model(config["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    class_weights = compute_class_weights(records_by_split["train"], device)
    criterion = build_criterion(config, class_weights, device)
    metrics = evaluate(model, loaders[split], criterion, device)

    print(f"Evaluated checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    records_by_split = load_manifest(config)
    summary = summarize_records(records_by_split)

    if args.check_data_only:
        print("Data compatibility check passed.")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.eval_checkpoint:
        evaluate_checkpoint(config, records_by_split, args.eval_checkpoint, args.eval_split)
        return

    train(config, records_by_split)


if __name__ == "__main__":
    main()
