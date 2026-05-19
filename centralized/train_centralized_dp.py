import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_centralized import (  # noqa: E402
    build_criterion,
    build_datasets,
    calculate_metrics,
    compute_class_weights,
    disable_stochastic_depth,
    format_duration,
    get_checkpoint_module,
    get_dp_config,
    get_epsilon,
    load_config,
    load_manifest,
    load_opacus,
    resolve_dp_delta,
    save_config,
    set_seed,
    summarize_records,
    classify_parameters,
    count_parameters,
    split_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a centralized EMBED classifier, optionally using Opacus DP."
    )
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
    parser.add_argument(
        "--grad-output",
        action="store_true",
        help="Enable per-batch gradient debug printing.",
    )
    return parser.parse_args()


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
    if args.grad_output:
        effective.setdefault("debug", {})["grad_output"] = True
    return effective


def make_model_dp_compatible(config, model):
    dp_config = get_dp_config(config)
    if not dp_config["enabled"]:
        return model, {"module_fixed": False, "stochastic_depth_disabled": 0}

    _, module_validator = load_opacus()
    module_fixed = False
    stochastic_depth_disabled = disable_stochastic_depth(model)

    if dp_config.get("fix_modules", True):
        fix_and_validate = getattr(module_validator, "fix_and_validate", None)
        if fix_and_validate is not None:
            model = fix_and_validate(model)
        else:
            model = module_validator.fix(model)
            module_validator.validate(model, strict=True)
        module_fixed = True
    else:
        module_validator.validate(model, strict=True)

    return model, {
        "module_fixed": module_fixed,
        "stochastic_depth_disabled": stochastic_depth_disabled,
    }


def normalize_optimizer_name(name):
    normalized = str(name or "SGD").strip().lower()
    if normalized == "sgd":
        return "SGD"
    if normalized == "adam":
        return "Adam"
    raise ValueError(f"Unsupported optimizer: {name}")


def configure_trainable_parameters(config, model):
    optimizer_config = config.get("optimizer", {})
    optimizer_name = normalize_optimizer_name(optimizer_config.get("name", "SGD"))
    freeze_backbone = bool(optimizer_config.get("freeze_backbone", False))
    initial_groups = classify_parameters(model)

    if freeze_backbone:
        for _, param in initial_groups["backbone"]:
            param.requires_grad = False

    final_groups = classify_parameters(model)
    frozen_backbone = initial_groups["backbone"] if freeze_backbone else []

    return {
        "configured_name": optimizer_config.get("name", "SGD"),
        "effective_name": optimizer_name,
        "name_overridden_for_dp": False,
        "freeze_backbone": freeze_backbone,
        "frozen_parameter_count": count_parameters(frozen_backbone),
        "trainable_parameter_count": count_parameters(
            final_groups["backbone"] + final_groups["finetune"]
        ),
        "frozen_parameter_names": [name for name, _ in frozen_backbone],
        "trainable_parameter_names": [
            name for name, _ in final_groups["backbone"] + final_groups["finetune"]
        ],
        "parameter_group_lrs": {
            "backbone": optimizer_config.get("lr_backbone"),
            "finetune": optimizer_config.get("lr_finetune"),
        },
        "momentum": float(optimizer_config.get("momentum", 0.0)),
        "betas": [float(v) for v in optimizer_config.get("betas", [0.9, 0.999])],
        "epsilon": float(optimizer_config.get("epsilon", 1e-8)),
        "weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
        "groups": {
            "backbone": {
                "lr": optimizer_config.get("lr_backbone"),
                "trainable": bool(final_groups["backbone"]),
                "parameter_count": count_parameters(final_groups["backbone"]),
                "parameter_names": [name for name, _ in final_groups["backbone"]],
            },
            "finetune": {
                "lr": optimizer_config.get("lr_finetune"),
                "trainable": bool(final_groups["finetune"]),
                "parameter_count": count_parameters(final_groups["finetune"]),
                "parameter_names": [name for name, _ in final_groups["finetune"]],
            },
        },
    }


def build_optimizer(config, model):
    import torch.optim as optim

    backbone_params, finetune_params = split_parameters(model)
    optimizer_config = config["optimizer"]
    optimizer_name = normalize_optimizer_name(optimizer_config.get("name", "SGD"))
    param_groups = []
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": optimizer_config.get("lr_backbone")}
        )
    if finetune_params:
        param_groups.append(
            {"params": finetune_params, "lr": optimizer_config.get("lr_finetune")}
        )

    if not param_groups:
        raise ValueError("No trainable parameters available for optimizer.")

    if optimizer_name == "Adam":
        return optim.Adam(
            param_groups,
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            eps=optimizer_config.get("epsilon", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    if optimizer_name == "SGD":
        return optim.SGD(
            param_groups,
            momentum=optimizer_config.get("momentum", 0.0),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class TransformedTupleDataset:
    def __init__(self, dataset, split_name, read_retries=3, retry_delay_seconds=0.25):
        self.dataset = dataset
        self.split_name = split_name
        self.read_retries = read_retries
        self.retry_delay_seconds = retry_delay_seconds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch
        import time

        for attempt in range(1, self.read_retries + 1):
            try:
                sample = self.dataset[index]
                break
            except Exception as exc:
                if attempt == self.read_retries:
                    raise RuntimeError(
                        f"Failed to load transformed {self.split_name} sample at index {index} "
                        f"after {self.read_retries} attempts."
                    ) from exc
                time.sleep(self.retry_delay_seconds)
        return sample["image"].float(), torch.as_tensor(sample["label"], dtype=torch.long)


def build_torch_loader(config, dataset, split_name, shuffle):
    from torch.utils.data import DataLoader

    loader_config = config["dataloader"]
    tuple_dataset = TransformedTupleDataset(
        dataset,
        split_name=split_name,
        read_retries=loader_config.get("read_retries", 3),
        retry_delay_seconds=loader_config.get("retry_delay_seconds", 0.25),
    )
    return DataLoader(
        tuple_dataset,
        batch_size=loader_config.get("batch_size", 1),
        shuffle=shuffle,
        num_workers=loader_config.get("num_workers", 0),
        pin_memory=loader_config.get("pin_memory", True),
    )


def build_torch_loaders(config, datasets):
    loader_config = config["dataloader"]
    return {
        "train": build_torch_loader(
            config,
            datasets["train"],
            split_name="train",
            shuffle=loader_config.get("shuffle", True),
        ),
        "validation": build_torch_loader(
            config,
            datasets["validation"],
            split_name="validation",
            shuffle=False,
        ),
        "test": build_torch_loader(
            config,
            datasets["test"],
            split_name="test",
            shuffle=False,
        ),
    }


def make_dp_metadata(dp_config, train_dataset_size, dp_model_info):
    if not dp_config["enabled"]:
        return {
            "enabled": False,
            "module_fixed": bool(dp_model_info["module_fixed"]),
            "stochastic_depth_disabled": int(dp_model_info["stochastic_depth_disabled"]),
        }

    target_delta = resolve_dp_delta(dp_config, train_dataset_size)
    metadata = {
        "enabled": True,
        "accountant": dp_config["accountant"],
        "secure_mode": bool(dp_config.get("secure_mode", False)),
        "mode": dp_config["mode"],
        "max_grad_norm": float(dp_config["max_grad_norm"]),
        "poisson_sampling": bool(dp_config.get("poisson_sampling", True)),
        "clipping": dp_config.get("clipping", "flat"),
        "grad_sample_mode": dp_config.get("grad_sample_mode", "hooks"),
        "fix_modules": bool(dp_config.get("fix_modules", True)),
        "module_fixed": bool(dp_model_info["module_fixed"]),
        "stochastic_depth_disabled": int(dp_model_info["stochastic_depth_disabled"]),
        "target_delta": target_delta,
        "final_epsilon": None,
        "per_epoch_epsilon": [],
    }
    if dp_config["mode"] == "noise_multiplier":
        metadata["noise_multiplier"] = float(dp_config["noise_multiplier"])
    elif dp_config["mode"] == "target_epsilon":
        metadata["target_epsilon"] = float(dp_config["target_epsilon"])
    return metadata


def prepare_private_training(config, model, optimizer, train_loader):
    dp_config = get_dp_config(config)
    if not dp_config["enabled"]:
        return model, optimizer, train_loader, None

    if dp_config["mode"] not in {"noise_multiplier", "target_epsilon"}:
        raise ValueError(
            "Unsupported differential_privacy.mode: "
            f"{dp_config['mode']}. Expected 'noise_multiplier' or 'target_epsilon'."
        )

    privacy_engine_cls, _ = load_opacus()
    privacy_engine = privacy_engine_cls(
        accountant=dp_config["accountant"],
        secure_mode=dp_config.get("secure_mode", False),
    )
    common_kwargs = {
        "module": model,
        "optimizer": optimizer,
        "data_loader": train_loader,
        "max_grad_norm": float(dp_config["max_grad_norm"]),
        "poisson_sampling": bool(dp_config.get("poisson_sampling", True)),
        "clipping": dp_config.get("clipping", "flat"),
        "grad_sample_mode": dp_config.get("grad_sample_mode", "hooks"),
    }
    if dp_config["mode"] == "noise_multiplier":
        model, optimizer, train_loader = privacy_engine.make_private(
            noise_multiplier=float(dp_config["noise_multiplier"]),
            **common_kwargs,
        )
    else:
        if dp_config.get("target_epsilon") is None:
            raise ValueError(
                "differential_privacy.target_epsilon is required when mode is 'target_epsilon'."
            )
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            epochs=int(config["epochs"]),
            target_epsilon=float(dp_config["target_epsilon"]),
            target_delta=resolve_dp_delta(dp_config, len(train_loader.dataset)),
            **common_kwargs,
        )
    return model, optimizer, train_loader, privacy_engine


def prepare_tuple_batch(batch, device):
    inputs, labels = batch
    inputs = inputs.float().to(device)
    while inputs.ndim > 4 and inputs.shape[-1] == 1:
        inputs = inputs.squeeze(dim=-1)
    labels = labels.long().to(device)
    return inputs, labels


def train_one_epoch(model, loader, criterion, optimizer, device, grad_output=False):
    import torch

    model.train()
    running_loss = 0.0
    labels_total = []
    predictions_total = []
    probabilities_total = []

    for batch in loader:
        inputs, labels = prepare_tuple_batch(batch, device)
        if labels.numel() == 0:
            continue
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient debug: pass --grad-output to enable this block.
        if grad_output:
            grad_sample_norms = []
            for param in model.parameters():
                grad_sample = getattr(param, "grad_sample", None)
                if grad_sample is None:
                    continue
                grad_sample_norms.append(grad_sample.flatten(start_dim=1).norm(2, dim=1))
            if grad_sample_norms:
                per_sample_norms = torch.stack(grad_sample_norms, dim=1).norm(2, dim=1)
                print(
                    "DP per-sample grad norms: "
                    f"mean={per_sample_norms.mean().item():.4f} "
                    f"max={per_sample_norms.max().item():.4f} "
                    f"p95={torch.quantile(per_sample_norms, 0.95).item():.4f}"
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=False
                )
                print(f"Gradient norm: {grad_norm:.4f}")

        optimizer.step()

        running_loss += loss.item()
        probabilities = torch.softmax(outputs.detach(), dim=1)[:, 1]
        predictions = torch.argmax(outputs.detach(), dim=1)
        labels_total.extend(labels.detach().cpu().numpy().tolist())
        predictions_total.extend(predictions.cpu().numpy().tolist())
        probabilities_total.extend(probabilities.cpu().numpy().tolist())

    if not labels_total:
        return {
            "loss": math.nan,
            "balanced_accuracy": math.nan,
            "specificity": math.nan,
            "sensitivity": math.nan,
            "auc": math.nan,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

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
            inputs, labels = prepare_tuple_batch(batch, device)
            if labels.numel() == 0:
                continue
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = torch.argmax(outputs, dim=1)
            labels_total.extend(labels.cpu().numpy().tolist())
            predictions_total.extend(predictions.cpu().numpy().tolist())
            probabilities_total.extend(probabilities.cpu().numpy().tolist())

    if not labels_total:
        return {
            "loss": math.nan,
            "balanced_accuracy": math.nan,
            "specificity": math.nan,
            "sensitivity": math.nan,
            "auc": math.nan,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

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

    dp_config = get_dp_config(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 0))

    save_config(config, workdir / "resolved_config.yml")

    datasets = build_datasets(config, records_by_split)
    loaders = build_torch_loaders(config, datasets)
    model = get_model(config["model"])
    model, dp_model_info = make_model_dp_compatible(config, model)
    model = model.to(device)
    class_weights = compute_class_weights(records_by_split["train"], device)
    criterion = build_criterion(config, class_weights, device)
    optimizer_metadata = configure_trainable_parameters(config, model)
    optimizer = build_optimizer(config, model)
    model, optimizer, loaders["train"], privacy_engine = prepare_private_training(
        config,
        model,
        optimizer,
        loaders["train"],
    )

    best_metric_name = config.get("save_metric", "balanced_accuracy")
    best_metric = -math.inf
    metrics_history = {
        "summary": summarize_records(records_by_split),
        "class_weights": [float(v) for v in class_weights.detach().cpu().numpy().tolist()],
        "device": str(device),
        "optimizer": optimizer_metadata,
        "differential_privacy": make_dp_metadata(
            dp_config,
            len(datasets["train"]),
            dp_model_info,
        ),
        "epochs": [],
    }

    print(f"Training on {device}. Outputs will be saved to {workdir}")
    print(f"Split summary: {json.dumps(metrics_history['summary'], sort_keys=True)}")
    print("Data loaders: torch.utils.data.DataLoader for train/validation/test")
    print(f"Differential privacy enabled: {dp_config['enabled']}")
    print(
        "Optimizer: "
        f"configured={optimizer_metadata['configured_name']} "
        f"effective={optimizer_metadata['effective_name']} "
        f"overridden_for_dp={optimizer_metadata['name_overridden_for_dp']}"
    )
    print(
        "Parameter freezing: "
        f"freeze_backbone={optimizer_metadata['freeze_backbone']} "
        f"frozen={optimizer_metadata['frozen_parameter_count']:,} "
        f"trainable={optimizer_metadata['trainable_parameter_count']:,}"
    )

    training_start = time.perf_counter()
    grad_output = bool(config.get("debug", {}).get("grad_output", False))
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            grad_output=grad_output,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
        }
        epsilon = None
        if dp_config["enabled"]:
            epsilon = get_epsilon(
                privacy_engine,
                metrics_history["differential_privacy"]["target_delta"],
            )
            epoch_metrics["differential_privacy"] = {
                "epsilon": epsilon,
                "delta": metrics_history["differential_privacy"]["target_delta"],
            }
            metrics_history["differential_privacy"]["per_epoch_epsilon"].append(
                {"epoch": epoch, "epsilon": epsilon}
            )
            metrics_history["differential_privacy"]["final_epsilon"] = epsilon
            if hasattr(optimizer, "noise_multiplier"):
                metrics_history["differential_privacy"]["noise_multiplier"] = float(
                    optimizer.noise_multiplier
                )

        if epoch % config.get("val_interval", 1) == 0:
            validation_metrics = evaluate(model, loaders["validation"], criterion, device)
            epoch_metrics["validation"] = validation_metrics
            metric_value = validation_metrics[best_metric_name]
            if not math.isnan(metric_value) and metric_value > best_metric:
                best_metric = metric_value
                torch.save(get_checkpoint_module(model).state_dict(), workdir / "best_model.pth")
                metrics_history["best_epoch"] = epoch
                metrics_history["best_validation"] = validation_metrics

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_metrics["duration_seconds"] = epoch_seconds
        metrics_history["epochs"].append(epoch_metrics)
        progress = (
            f"Epoch {epoch:03d}/{config['epochs']:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} "
            f"val_bal_acc={epoch_metrics.get('validation', {}).get('balanced_accuracy', math.nan):.4f} "
        )
        if epsilon is not None:
            progress += f"epsilon={epsilon:.4f} "
        progress += f"epoch_time={format_duration(epoch_seconds)}"
        print(progress)

    torch.save(get_checkpoint_module(model).state_dict(), workdir / "last_model.pth")
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    metrics_history["test"] = test_metrics
    best_model_path = workdir / "best_model.pth"
    if best_model_path.exists():
        get_checkpoint_module(model).load_state_dict(
            torch.load(best_model_path, map_location=device)
        )
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
        print(
            "Best model test metrics: "
            f"{json.dumps(metrics_history['best_model_test'], sort_keys=True)}"
        )
    print(
        f"Total training time: {metrics_history['duration']} "
        f"(avg_epoch={metrics_history['average_epoch_duration']})"
    )
    return metrics_history


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    records_by_split = load_manifest(config)
    summary = summarize_records(records_by_split)

    if args.check_data_only:
        print("Data compatibility check passed.")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    train(config, records_by_split)


if __name__ == "__main__":
    main()
