import argparse
import csv
import json
import math
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import nvflare.client as flare

from pathlib import Path
from nvflare.fuel.utils.log_utils import get_script_logger
from monai.transforms import (Compose, LoadImaged, ScaleIntensityd, RandAdjustContrastd, Orientationd, RandGaussianSmoothd, RandFlipd, RandRotated,
                             RandShiftIntensityd, RandGaussianNoised, ThresholdIntensityd, RandAffined, ResizeWithPadOrCropd)
from monai.data import DataLoader, CacheDataset, Dataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import *
from utils import *


EMBED_ROOT = "/mnt/d/Users/kokouk/Projects/Data/EMBED"
Training_ROOT = "/mnt/d/Users/kokouk/Projects/EMBEDFedClassifier"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {"benign": 0, "malignant": 1}
PATH_COLUMNS = ("image_path_suffix", "output_relpath")

def error_raise(ex): raise Exception(ex)


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else math.nan


def calculate_metrics(labels, predictions, probabilities=None, average_loss=None):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "specificity": safe_divide(tn, tn + fp),
        "sensitivity": safe_divide(tp, tp + fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if average_loss is not None:
        metrics["loss"] = float(average_loss)
    if probabilities is not None:
        try:
            metrics["auc"] = float(roc_auc_score(labels, probabilities))
        except ValueError:
            metrics["auc"] = math.nan
    return metrics


def save_site_metrics(metrics_path, metrics):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)


class SqueezeSingletonDepthd:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            if image.ndim == 4 and image.shape[-1] == 1:
                d[key] = torch.squeeze(image, dim=-1)
            elif image.ndim == 3 and image.shape[0] != 1 and image.shape[-1] == 1:
                d[key] = torch.movedim(torch.squeeze(image, dim=-1), -1, 0)
        return d

def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default=EMBED_ROOT, nargs="?")
    parser.add_argument("--test_dataset_path", type=str, default=EMBED_ROOT, nargs="?")
    parser.add_argument("--data_csv", type=str, default="", nargs="?")
    parser.add_argument("--cleaned_data_root", type=str, default="", nargs="?")
    parser.add_argument("--client_site", type=str, default="", nargs="?")
    parser.add_argument("--batch_size", type=int, default=4, nargs="?")
    parser.add_argument("--learning_rate", type=float, default=0.0003, nargs="?")
    parser.add_argument("--local_epochs", type=int, default=5, nargs="?")
    parser.add_argument("--client_model_path", type=str, default=f"{Training_ROOT}/embed_net.pth", nargs="?")
    parser.add_argument("--global_model_path", type=str, default=f"{Training_ROOT}/embed_net_global.pth", nargs="?")
    parser.add_argument("--client_config_path", type=str, default="./client_config.yml", nargs="?")
    parser.add_argument("--client_cases", type=str, default="", nargs="?")
    parser.add_argument("--workdir", type=str, default=Training_ROOT, nargs="?")

    return parser.parse_args()


def select_path_value(row):
    for column in PATH_COLUMNS:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def load_manifest_records(data_csv, cleaned_data_root, client_site, local_config):
    data_csv = Path(data_csv)
    cleaned_data_root = Path(cleaned_data_root)
    if not data_csv.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {data_csv}")
    if not cleaned_data_root.exists():
        raise FileNotFoundError(f"Cleaned data root does not exist: {cleaned_data_root}")

    with open(data_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    required = {"loc_num", "model_split", "binary_label"}
    missing = required - fieldnames
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if not any(column in fieldnames for column in PATH_COLUMNS):
        raise ValueError("Manifest must include image_path_suffix or output_relpath.")

    split_config = local_config.get("splits", {})
    train_split = split_config.get("train", "train")
    validation_split = split_config.get("validation", "validation")
    test_split = split_config.get("test", "test")
    test_scope = local_config.get("test_scope", "global")

    records = {"train": [], "validation": [], "test": []}
    missing_files = []
    invalid_labels = set()

    for row in rows:
        split_value = str(row["model_split"]).strip()
        row_site = str(row["loc_num"]).strip()
        if split_value == train_split:
            split_name = "train"
            if row_site != str(client_site):
                continue
        elif split_value == validation_split:
            split_name = "validation"
            if row_site != str(client_site):
                continue
        elif split_value == test_split:
            split_name = "test"
            if test_scope == "local" and row_site != str(client_site):
                continue
        else:
            continue

        label_name = str(row["binary_label"]).strip().lower()
        if label_name not in LABEL_MAP:
            invalid_labels.add(label_name)
            continue

        relpath = select_path_value(row)
        if relpath is None:
            raise ValueError("Manifest row is missing image_path_suffix and output_relpath.")
        image_path = cleaned_data_root / relpath
        if not image_path.exists():
            missing_files.append(str(image_path))
        records[split_name].append({"image": str(image_path), "label": LABEL_MAP[label_name]})

    if invalid_labels:
        raise ValueError(f"Unexpected binary_label values: {sorted(invalid_labels)}")
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} manifest image files are missing. First missing paths:\n{preview}"
        )
    for split_name, split_records in records.items():
        if not split_records:
            raise ValueError(f"No {split_name} records found for client site {client_site}.")

    return records


def load_legacy_records(train_dataset_path, test_dataset_path, local_cases):
    train_cases = [case for case in Path(train_dataset_path).rglob("*") if case.is_file()]
    train_cases = [case for case in train_cases if case.parents[0].name in local_cases]
    val_cases = []
    test_cases = [case for case in Path(test_dataset_path).rglob("*") if case.is_file()]

    benign_cases = [c for c in train_cases if 'benign' in str(c)]
    malignant_cases = [c for c in train_cases if 'malignant' in str(c)]

    val_num_benign = int(0.1 * len(benign_cases))
    val_num_malignant = int(0.1 * len(malignant_cases))
    val_cases.extend(random.sample(benign_cases, val_num_benign))
    val_cases.extend(random.sample(malignant_cases, val_num_malignant))

    train_cases = [c for c in train_cases if c not in val_cases]

    return {
        "train": [{"image": im, "label": 0 if "benign" in str(im) else 1} for im in train_cases],
        "validation": [{"image": im, "label": 0 if "benign" in str(im) else 1} for im in val_cases],
        "test": [{"image": im, "label": 0 if "benign" in str(im) else 1} for im in test_cases],
    }


def class_weights_from_records(train_records):
    labels = [record["label"] for record in train_records]
    benign = labels.count(0)
    malignant = labels.count(1)
    if benign == 0 or malignant == 0:
        raise ValueError("Both benign and malignant samples are required for class weights.")
    total = len(labels)
    return torch.tensor([total / (2 * benign), total / (2 * malignant)])


def add_resize_if_configured(transforms, transform_config):
    if "ResizeWithPadOrCropd" in transform_config:
        transforms.append(SqueezeSingletonDepthd(keys=["image"]))
        transforms.append(
            ResizeWithPadOrCropd(
                keys=transform_config["ResizeWithPadOrCropd"]["keys"],
                spatial_size=transform_config["ResizeWithPadOrCropd"]["spatial_size"],
            )
        )
    return transforms

def evaluate(model_args, input_weights, val_loader, criterion=None):
    logger.info("Evaluating model...")
    net = get_model(model_args)
    net.load_state_dict(input_weights)
    net.to(DEVICE)
    net.eval()

    val_predicted_total = []
    val_ground_total = []
    all_probs = []  
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs = torch.cat([data['image']], dim=1).to(DEVICE)
            while inputs.ndim > 4 and inputs.shape[-1] == 1:
                inputs = torch.squeeze(inputs, dim=-1)
            labels = data['label'].to(DEVICE) 

            try:
                outputs = net(inputs) # [0] needed only if using ViT model to get logits
                if criterion is not None:
                    running_loss += criterion(outputs, labels).item()

                _, predicted_classes = torch.max(outputs.detach().cpu(), 1)
                val_predicted_total.extend(predicted_classes.numpy().tolist())
                val_ground_total.extend(labels.detach().cpu().numpy().tolist())

                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())

            except Exception as e:
                print(f"Error during evaluation: {e}")

    average_loss = running_loss / max(len(val_loader), 1) if criterion is not None else None
    return calculate_metrics(
        val_ground_total,
        val_predicted_total,
        probabilities=all_probs,
        average_loss=average_loss,
    )

def main():
    # # Define local parameters
    args = define_parser()

    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    batch_size = args.batch_size
    lr = args.learning_rate
    local_config_path = args.client_config_path
    client_model_path = args.client_model_path
    global_model_path = args.global_model_path
    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = REPO_ROOT / workdir
    workdir = str(workdir)

    local_cases = args.client_cases
    local_cases = local_cases.split(',') # list of case IDs
    local_config = load_config(local_config_path)

    train_transforms_config = local_config.get('train_transforms')
    eval_transforms_config = local_config.get('eval_transforms')

    train_transform_list = [
                            # # Load images and labels
                            LoadImaged(keys=train_transforms_config['LoadImaged_im']['keys'], ensure_channel_first=train_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            # # Intensity normalization
                            ScaleIntensityd(keys=train_transforms_config['ScaleIntensityd']['keys']),
                            # # Image orientation
                            Orientationd(keys=train_transforms_config['Orientationd']['keys'], axcodes=train_transforms_config['Orientationd']['axcodes'], 
                                         as_closest_canonical=train_transforms_config['Orientationd']['as_closest_canonical']),
    ]
    train_transform_list = add_resize_if_configured(train_transform_list, train_transforms_config)
    train_transform_list.extend([
                            # # Augmentations
                            RandFlipd(keys=train_transforms_config['RandFlipd_x']['keys'], prob=train_transforms_config['RandFlipd_x']['prob'], spatial_axis=train_transforms_config['RandFlipd_x']['spatial_axis']),
                            RandFlipd(keys=train_transforms_config['RandFlipd_y']['keys'], prob=train_transforms_config['RandFlipd_y']['prob'], spatial_axis=train_transforms_config['RandFlipd_y']['spatial_axis']),
                            RandRotated(keys=train_transforms_config['RandRotated']['keys'], prob=train_transforms_config['RandRotated']['prob'], 
                                        range_x=train_transforms_config['RandRotated']['range_x'], range_y=train_transforms_config['RandRotated']['range_y'], mode=train_transforms_config['RandRotated']['mode']),
                            RandAffined(keys=train_transforms_config['RandAffined']['keys'], prob=train_transforms_config['RandAffined']['prob'], 
                                        rotate_range=train_transforms_config['RandAffined']['rotate_range'], scale_range=train_transforms_config['RandAffined']['scale_range'], mode=train_transforms_config['RandAffined']['mode']),
                            # RandAdjustContrastd(keys=train_transforms_config['RandAdjustContrastd']['keys'], prob=train_transforms_config['RandAdjustContrastd']['prob'], gamma=train_transforms_config['RandAdjustContrastd']['gamma']),
                            # RandGaussianSmoothd(keys=train_transforms_config['RandGaussianSmoothd']['keys'], prob=train_transforms_config['RandGaussianSmoothd']['prob'], sigma_x=train_transforms_config['RandGaussianSmoothd']['sigma_x'], sigma_y=train_transforms_config['RandGaussianSmoothd']['sigma_y']),
                            # RandShiftIntensityd(keys=train_transforms_config['RandShiftIntensityd']['keys'], prob=train_transforms_config['RandShiftIntensityd']['prob'], offsets=train_transforms_config['RandShiftIntensityd']['offsets']),
                            # RandGaussianNoised(keys=train_transforms_config['RandGaussianNoised']['keys'], prob=train_transforms_config['RandGaussianNoised']['prob'], mean=train_transforms_config['RandGaussianNoised']['mean'], std=train_transforms_config['RandGaussianNoised']['std']),
                            # ThresholdIntensityd(keys=train_transforms_config['ThresholdIntensityd_clip_upper']['keys'], threshold=train_transforms_config['ThresholdIntensityd_clip_upper']['threshold'], above=train_transforms_config['ThresholdIntensityd_clip_upper']['above'], cval=train_transforms_config['ThresholdIntensityd_clip_upper']['cval']),
                            # ThresholdIntensityd(keys=train_transforms_config['ThresholdIntensityd_clip_lower']['keys'], threshold=train_transforms_config['ThresholdIntensityd_clip_lower']['threshold'], above=train_transforms_config['ThresholdIntensityd_clip_lower']['above'], cval=train_transforms_config['ThresholdIntensityd_clip_lower']['cval']),

    ])
    train_transforms = Compose(train_transform_list)

    val_transform_list = [
                            # # Load images and labels
                            LoadImaged(keys=eval_transforms_config['LoadImaged_im']['keys'], ensure_channel_first=eval_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            # # Intensity normalization
                            ScaleIntensityd(keys=eval_transforms_config['ScaleIntensityd']['keys']),
                            # # Image orientation
                            Orientationd(keys=eval_transforms_config['Orientationd']['keys'], axcodes=eval_transforms_config['Orientationd']['axcodes'], 
                                         as_closest_canonical=eval_transforms_config['Orientationd']['as_closest_canonical']),
    ]
    val_transform_list = add_resize_if_configured(val_transform_list, eval_transforms_config)
    val_transforms = Compose(val_transform_list)

    print(f"Loading data from {train_dataset_path}...")
    print(f"Using batch size: {batch_size}")
    print(f"Using learning rate: {lr}")

    # print("Before hitting breakpoint")
    # import pdb;
    # pdb.set_trace()
    # or use breakpoint() 

    if args.data_csv:
        records = load_manifest_records(
            data_csv=args.data_csv,
            cleaned_data_root=args.cleaned_data_root,
            client_site=args.client_site,
            local_config=local_config,
        )
    else:
        records = load_legacy_records(train_dataset_path, test_dataset_path, local_cases)

    train_dict = records["train"]
    val_dict = records["validation"]
    test_dict = records["test"]
    w_class = class_weights_from_records(train_dict)
    train_labels = [record["label"] for record in train_dict]
    logger.info(
        f"Number of training cases: {len(train_dict)} "
        f"(Benign: {train_labels.count(0)}, Malignant: {train_labels.count(1)})"
    )
    logger.info(f"Class weights - Benign: {w_class[0]:.4f}, Malignant: {w_class[1]:.4f}")

    # # Define dataset
    logger.info("Preparing datasets...")
    if local_config['dataset'] == 'Dataset':
        train_dataset = Dataset(data=train_dict, transform=train_transforms)
        val_dataset = Dataset(data=val_dict, transform=val_transforms)
        test_dataset = Dataset(data=test_dict, transform=val_transforms)
    elif local_config['dataset'] == 'CacheDataset':
        train_dataset = CacheDataset(data=train_dict, transform=train_transforms, cache_rate=local_config['cache_rate'])
        val_dataset = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=local_config['cache_rate'])
        test_dataset = CacheDataset(data=test_dict, transform=val_transforms, cache_rate=local_config['cache_rate'])

    # # Initialize NVFlare client
    flare.init() 
    client_id = flare.get_site_name()
    site_workdir = Path(workdir) / client_id
    site_workdir.mkdir(parents=True, exist_ok=True)
    metrics_path = site_workdir / "metrics.json"
    site_metrics = {
        "site": client_id,
        "client_site": args.client_site,
        "device": str(DEVICE),
        "data": {
            "train": {
                "total": len(train_dict),
                "benign": sum(1 for record in train_dict if record["label"] == 0),
                "malignant": sum(1 for record in train_dict if record["label"] == 1),
            },
            "validation": {
                "total": len(val_dict),
                "benign": sum(1 for record in val_dict if record["label"] == 0),
                "malignant": sum(1 for record in val_dict if record["label"] == 1),
            },
            "test": {
                "total": len(test_dict),
                "benign": sum(1 for record in test_dict if record["label"] == 0),
                "malignant": sum(1 for record in test_dict if record["label"] == 1),
            },
        },
        "rounds": [],
        "global_test": [],
    }
    save_site_metrics(metrics_path, site_metrics)

    logger.info(f"({flare.get_site_name()}) Number of training samples: {len(train_dataset)}")
    logger.info(f"({flare.get_site_name()}) Number of validation samples: {len(val_dataset)}")
    logger.info(f"({flare.get_site_name()}) Number of test samples: {len(test_dataset)}")

    # # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=local_config.get('dataloader').get('batch_size', 1),
                              shuffle=local_config.get('dataloader').get('shuffle', False), 
                              num_workers=local_config.get('dataloader').get('num_workers', 1),
                              pin_memory=local_config.get('dataloader').get('pin_memory', True))
    val_loader = DataLoader(val_dataset, batch_size=local_config.get('dataloader').get('batch_size', 1), 
                            shuffle=local_config.get('dataloader').get('shuffle', False),
                            num_workers=local_config.get('dataloader').get('num_workers', 1),
                            pin_memory=local_config.get('dataloader').get('pin_memory', True))
    test_loader = DataLoader(test_dataset, batch_size=local_config.get('dataloader').get('batch_size', 1), 
                             shuffle=local_config.get('dataloader').get('shuffle', False),
                             num_workers=local_config.get('dataloader').get('num_workers', 1),
                             pin_memory=local_config.get('dataloader').get('pin_memory', True))

    # # Get model from config and initialize model object
    model_args = local_config.get('model') if local_config.get('model') else error_raise("model_args must be provided")
    net = get_model(model_args)

    local_epochs = local_config['local_epochs']
    best_metric = local_config['save_model_when']

    while flare.is_running():

        input_model = flare.receive()
        client_id = flare.get_site_name()

        if flare.is_train():
            round_start = time.perf_counter()
            logger.info(f"({client_id}) current_round={input_model.current_round}, total_rounds={input_model.total_rounds}")
            net.load_state_dict(input_model.params) # Load received global model weights

            if input_model.current_round > 0 and client_id == 'site-1':
                logger.info(f"({client_id}) Performing testing with current global model before local training...")
                test_metrics = evaluate(model_args, net.state_dict(), test_loader)
                test_metrics["round"] = int(input_model.current_round)
                site_metrics["global_test"].append(test_metrics)
                save_site_metrics(metrics_path, site_metrics)
                logger.info(f"({client_id}) -- Balanced accuracy on test set: {test_metrics['balanced_accuracy']:.4f}")
                logger.info(f"({client_id}) -- Specificity on test set: {test_metrics['specificity']:.4f}")
                logger.info(f"({client_id}) -- Sensitivity on test set: {test_metrics['sensitivity']:.4f}")
                logger.info(f"({client_id}) -- AUC on test set: {test_metrics['auc']:.4f}")
                logger.info(f"Saving global model to {global_model_path}_{input_model.current_round}...")
                # torch.save(net.state_dict(), f"{global_model_path}_{input_model.current_round}")
                best_metric = test_metrics["balanced_accuracy"]  # Update best metric based on test set performance
                                                                               
            # # Define loss function
            if local_config.get('loss'):
                if local_config.get('loss').get('name') == 'CrossEntropyLoss':
                    criterion = nn.CrossEntropyLoss(label_smoothing=local_config.get('loss').get('label_smoothing'),
                                                    weight=w_class.to(DEVICE) if local_config.get('loss').get('class_weights', False) else None
                                                    )
                elif local_config.get('loss').get('name') == 'BCEWithLogitsLoss':
                    criterion = nn.BCEWithLogitsLoss()
                elif local_config.get('loss').get('name') == 'BCELoss':
                    criterion = nn.BCELoss()
                logger.info(f"({client_id}) Using loss function: {local_config.get('loss').get('name')}")
            else:
                error_raise("Loss function must be provided")

            # # Use different learning rate for backbone and classifier head if specified
            # # Separate parameters
            backbone_params = []
            finetune_params = []
            for name, param in net.named_parameters():
                if 'classifier' in name or 'features.0.0' in name or 'head' in name:
                    finetune_params.append(param)
                else:
                    backbone_params.append(param)

            # # Define optimizer
            if local_config.get('optimizer'):
                if local_config.get('optimizer').get('name') == 'SGD':
                    optimizer = optim.SGD([
                        {'params': backbone_params, 'lr': local_config.get('optimizer').get('lr_backbone', lr)},
                        {'params': finetune_params, 'lr': local_config.get('optimizer').get('lr_finetune', lr)}
                    ], momentum=local_config.get('optimizer').get('momentum'), weight_decay=local_config.get('optimizer').get('weight_decay'))
                elif local_config.get('optimizer').get('name') == 'Adam':
                    optimizer = optim.Adam([
                        {'params': backbone_params, 'lr': local_config.get('optimizer').get('lr_backbone', lr)},
                        {'params': finetune_params, 'lr': local_config.get('optimizer').get('lr_finetune', lr)}
                    ], weight_decay=local_config.get('optimizer').get('weight_decay'))
                logger.info(f"({client_id}) Using optimizer: {local_config.get('optimizer').get('name')}")
            else:
                error_raise("Optimizer must be provided")
            
            # # Define metric
            if local_config.get('metric'):
                if local_config.get('metric').get('name') == 'Accuracy':
                    logger.info(f"({client_id}) Using metric: {local_config.get('metric').get('name')}")
            else:
                error_raise("Metric must be provided")

            # # Send model to device
            net.to(DEVICE)
            steps = local_epochs * len(train_loader)

            training_loss = []
            logger.info(f"({client_id}) Starting Training for {local_epochs} epochs...")

            local_epochs = 40 if local_config.get('personalized', False) and input_model.current_round == 19 else local_epochs  # If personalized and best metric is high, train for more epochs
            round_metrics = {
                "round": int(input_model.current_round),
                "total_rounds": int(input_model.total_rounds),
                "local_epochs": int(local_epochs),
                "epochs": [],
            }
            for epoch in range(local_epochs):  # loop over the dataset multiple times
                epoch_start = time.perf_counter()
                logger.info(f"-------------------------------------------")
                logger.info(f"({client_id}) Epoch {epoch + 1}...")
                running_loss = 0.0
                predicted_total = []
                ground_total = []

                for i, data in enumerate(train_loader, 0):
                    inputs = torch.cat([data['image']], dim=1).to(DEVICE)
                    while inputs.ndim > 4 and inputs.shape[-1] == 1:
                        inputs = torch.squeeze(inputs, dim=-1)
                    labels = data['label'].to(DEVICE) 

                    # # For debugging: visualize input images
                    # # IMAGES LOOK QUITE DIFFERENT IN TERMS OF INTENSITY DISTRIBUTION 
                    # breakpoint()
                    # import matplotlib.pyplot as plt
                    # stacked_imgs = np.hstack([inputs[j,0,:,:].cpu() for j in range(4)])
                    # plt.imshow(stacked_imgs, cmap='gray')
                    # plt.show()
                    # continue

                    try:
                        optimizer.zero_grad()

                        outputs = net(inputs) # [0] needed only if using ViT model to get logits

                        # print(f'Out: {outputs}, Labels: {labels}') # check outputs and labels size and values for debugging
                        loss = criterion(outputs, labels)
                        loss.backward()

                        optimizer.step()

                        # Accuracy calculation
                        _, predicted_classes = torch.max(outputs.detach().cpu(), 1)
                        labels = labels.detach().cpu()
                        predicted_total.extend(predicted_classes.numpy().tolist())
                        ground_total.extend(labels.numpy().tolist())

                    except Exception as e:
                        print(f"({client_id}) Error during training: {e}")

                    running_loss += loss.item()
                epoch_loss = running_loss / len(train_loader)
                training_loss.append(epoch_loss)
                epoch_metrics = calculate_metrics(
                    ground_total,
                    predicted_total,
                    average_loss=epoch_loss,
                )
                epoch_metrics["epoch"] = int(epoch + 1)
                epoch_metrics["duration_seconds"] = time.perf_counter() - epoch_start
                epoch_metrics["duration"] = format_duration(epoch_metrics["duration_seconds"])

                logger.info(f"Epoch loss: {epoch_loss:.4f}")
                logger.info(f"Epoch balanced accuracy: {epoch_metrics['balanced_accuracy']:.4f}")
                logger.info(f"Epoch specificity: {epoch_metrics['specificity']:.4f}")
                logger.info(f"Epoch sensitivity: {epoch_metrics['sensitivity']:.4f}")
                logger.info(f"-------------------------------------------")

                if (epoch+1) % local_config.get('val_interval', 10) == 0:
                    local_metrics = evaluate(model_args, net.state_dict(), val_loader, criterion=criterion)
                    epoch_metrics["validation"] = local_metrics
                    logger.info(f"({client_id}) -- Balanced accuracy on validation set: {local_metrics['balanced_accuracy']:.4f}")
                    logger.info(f"({client_id}) -- Specificity on validation set: {local_metrics['specificity']:.4f}")
                    logger.info(f"({client_id}) -- Sensitivity on validation set: {local_metrics['sensitivity']:.4f}")
                    logger.info(f"({client_id}) -- AUC on validation set: {local_metrics['auc']:.4f}")
                    if local_metrics["balanced_accuracy"] > best_metric:
                        logger.info(f"({client_id}) New best model found {local_metrics['balanced_accuracy']:.4f} at round {input_model.current_round} (previous best was {best_metric:.4f})")
                        best_metric = local_metrics["balanced_accuracy"]
                    logger.info(f"Saving model to {client_model_path}_{input_model.current_round}...")

                if (epoch+1) % 5 == 0:
                    logger.info(f"({client_id}) Saving model to {workdir}/{client_id}/embed_net_round_{input_model.current_round}_epoch_{epoch+1}.pth...")
                    torch.save(net.state_dict(), f"{workdir}/{client_id}/embed_net_round_{input_model.current_round}_epoch_{epoch+1}.pth")
                round_metrics["epochs"].append(epoch_metrics)
                save_site_metrics(metrics_path, site_metrics)

            logger.info(f"({client_id}) Finished Training")
            round_metrics["duration_seconds"] = time.perf_counter() - round_start
            round_metrics["duration"] = format_duration(round_metrics["duration_seconds"])
            site_metrics["rounds"].append(round_metrics)
            save_site_metrics(metrics_path, site_metrics)

            output_model = flare.FLModel(
                params=net.cpu().state_dict(),
                # metrics={"metric": global_model_metric},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )

            flare.send(output_model)

        # elif flare.is_evaluate():
        #     global_model_metric = evaluate(net, input_model.params)
        #     print(f"({client_id}) Metric: {global_model_metric}")
        #     flare.send(flare.FLModel(metrics={"metric": global_model_metric}))

        # elif flare.is_submit_model():
        #     model_name = input_model.meta["submit_model_name"]
        #     if model_name == ModelName.BEST_MODEL:
        #         try:
        #             weights = torch.load(model_path)
        #             net = get_model(local_config.get('model'))
        #             net.load_state_dict(weights)
        #             flare.send(flare.FLModel(params=net.cpu().state_dict()))
        #         except Exception as e:
        #             error_raise("Unable to load best model")
        #     else:
        #         error_raise(f"Unknown submit model name: {model_name}")


if __name__ == "__main__":
    logger = get_script_logger()
    main()
