import argparse
import json
import os
import time
import pandas as pd
import torch
import torchvision
import torchvision.models

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, Orientationd, ScaleIntensityd
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.fuel.utils.log_utils import get_script_logger
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from model import *
from utils import *


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate_torchvision_import():
    if not hasattr(torchvision, "extension"):
        raise RuntimeError(
            "torchvision did not initialize its extension module. "
            f"torch={torch.__version__}, torchvision={torchvision.__version__}"
        )


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


def json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def metric_text(value):
    return f"{value:.4f}" if value is not None else "n/a"


def calculate_metrics(labels, predictions, probs=None, loss=None):
    if not labels:
        return {
            "loss": loss,
            "balanced_accuracy": None,
            "specificity": None,
            "sensitivity": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "auc": None,
        }

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "loss": loss,
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "specificity": safe_divide(tn, tn + fp),
        "sensitivity": safe_divide(tp, tp + fn),
        "confusion_matrix": matrix.astype(int).tolist(),
        "auc": None,
    }
    if probs and len(set(labels)) == 2:
        metrics["auc"] = roc_auc_score(labels, probs)
    return metrics


def build_eval_transforms(client_config):
    eval_transforms_config = client_config.get("eval_transforms")
    return Compose([
        LoadImaged(
            keys=eval_transforms_config["LoadImaged_im"]["keys"],
            ensure_channel_first=eval_transforms_config["LoadImaged_im"]["ensure_channel_first"],
        ),
        ScaleIntensityd(keys=eval_transforms_config["ScaleIntensityd"]["keys"]),
        Orientationd(
            keys=eval_transforms_config["Orientationd"]["keys"],
            axcodes=eval_transforms_config["Orientationd"]["axcodes"],
            as_closest_canonical=eval_transforms_config["Orientationd"]["as_closest_canonical"],
        ),
    ])


def label_counts(cases):
    benign = len([case for case in cases if "benign" in str(case)])
    malignant = len([case for case in cases if "malignant" in str(case)])
    return {
        "total": len(cases),
        "benign": benign,
        "malignant": malignant,
    }


def build_test_loader(config, client_config):
    test_dataset_path = Path(config.get("test_dataset_path") or client_config.get("test_dataset_path"))
    test_cases = sorted([case for case in test_dataset_path.rglob("*") if case.is_file()])
    test_dict = [{"image": case, "label": 0 if "benign" in str(case) else 1} for case in test_cases]
    eval_transforms = build_eval_transforms(client_config)

    if client_config["dataset"] == "CacheDataset":
        test_dataset = CacheDataset(
            data=test_dict,
            transform=eval_transforms,
            cache_rate=client_config.get("cache_rate", 1),
        )
    else:
        test_dataset = Dataset(data=test_dict, transform=eval_transforms)

    dataloader_config = client_config.get("dataloader", {})
    test_loader = DataLoader(
        test_dataset,
        batch_size=dataloader_config.get("batch_size", config.get("batch_size", 1)),
        shuffle=False,
        num_workers=0,
        pin_memory=dataloader_config.get("pin_memory", True) and torch.cuda.is_available(),
    )
    return test_loader, test_dataset_path, label_counts(test_cases)


def load_final_model_weights(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def evaluate_model(model_args, input_weights, data_loader, logger):
    logger.info("Evaluating final global model on %s...", DEVICE)
    net = get_model(model_args)
    net.load_state_dict(input_weights)
    net.to(DEVICE)
    net.eval()

    predicted_total = []
    ground_total = []
    all_probs = []

    with torch.no_grad():
        for data in data_loader:
            inputs = torch.cat([data["image"]], dim=1).to(DEVICE)
            inputs = torch.squeeze(inputs, dim=len(inputs.shape) - 1)
            labels = data["label"].to(DEVICE)

            outputs = net(inputs)
            _, predicted_classes = torch.max(outputs.detach().cpu(), 1)
            predicted_total.extend(predicted_classes.numpy().tolist())
            ground_total.extend(labels.detach().cpu().numpy().tolist())

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    return calculate_metrics(ground_total, predicted_total, all_probs)


def record_final_global_metrics(config, client_config, logger):
    workdir = Path(config.get("workdir"))
    checkpoint_path = workdir / "server" / "simulate_job" / "app_server" / "FL_global_model.pt"
    metrics_path = workdir / "server" / "final_global_metrics.json"

    if not checkpoint_path.exists():
        logger.warning("Final global model checkpoint not found at %s; final metrics were not recorded.", checkpoint_path)
        return None

    start = time.time()
    test_loader, test_dataset_path, test_counts = build_test_loader(config, client_config)
    model_args = dict(config.get("model") or client_config.get("model"))
    model_args["load_pretrained_weights"] = False
    final_weights = load_final_model_weights(checkpoint_path)
    metrics = evaluate_model(model_args, final_weights, test_loader, logger)
    duration = time.time() - start

    record = {
        "checkpoint_path": checkpoint_path.resolve(),
        "completed_rounds": config.get("num_rounds"),
        "final_round": config.get("num_rounds", 1) - 1,
        "dataset": {
            "test_dataset_path": test_dataset_path.resolve(),
            "test": test_counts,
        },
        "device": str(DEVICE),
        "model": model_args,
        "duration_seconds": duration,
        "duration": time.strftime("%H:%M:%S", time.gmtime(duration)),
        "metrics": metrics,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(json_safe(record), f, indent=2)

    logger.info("Final global metrics written to %s", metrics_path)
    logger.info("Final global balanced accuracy: %s", metric_text(metrics["balanced_accuracy"]))
    logger.info("Final global specificity: %s", metric_text(metrics["specificity"]))
    logger.info("Final global sensitivity: %s", metric_text(metrics["sensitivity"]))
    if metrics.get("auc") is not None:
        logger.info("Final global AUC: %s", metric_text(metrics["auc"]))
    return record


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Job Runner")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()


if __name__ == "__main__":
    logger = get_script_logger()
    validate_torchvision_import()
    logger.info(
        "Validated torchvision import: torch=%s, torchvision=%s, extension=%s",
        torch.__version__,
        torchvision.__version__,
        hasattr(torchvision, "extension"),
    )
    working_dir = os.getcwd()

    args = parse_args()
    config = load_config(args.config)

    experiment_config_dir = Path(f"{config.get('workdir')}_config")
    if not os.path.exists(experiment_config_dir):
        os.makedirs(experiment_config_dir)

    # To run multiple experiments, save a copy of the config file in the experiment directory
    save_config(config, os.path.join(experiment_config_dir, 'server_config.yml'))
    client_config = load_config(config.get('client_config_path'))
    save_config(client_config, os.path.join(experiment_config_dir, 'client_config.yml'))
    
    n_clients = config.get('n_clients')
    num_rounds = config.get('num_rounds')   
    train_script = config.get('client_script')
    
    seed = config.get('seed', 0)
    set_seed(manual_seed=seed)

    model_args = config.get('model')
    model = get_model(model_args)

    if config.get('pretrained', False):
        pretrained_model_path = config.get('pretrained_model_path')
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            model.load_state_dict(torch.load(pretrained_model_path))
            logger.info(f"Loaded pretrained model from {pretrained_model_path}")
        else:
            logger.warning(f"Pretrained model path {pretrained_model_path} does not exist. Proceeding without loading pretrained weights.")

    if config.get('recipe') == 'fedavg':
        job = FedAvgJob(name='fedavg', n_clients=n_clients, num_rounds=num_rounds, initial_model=model)

    train_data_path = Path(config.get('train_dataset_path'))
    test_data_path = Path(config.get('test_dataset_path'))
    meta = pd.read_csv(config.get('meta_data_path'))

    cases = [c for c in train_data_path.rglob('*') if c.is_file()]
    series_instance_uids = [c.parents[0].name for c in cases] # there are 4 duplicates. all is 7351

    # get duplicated
    import collections
    duplicated = [item for item, count in collections.Counter(series_instance_uids).items() if count > 1]

    #######################################################################################################
    #######################################################################################################
    # Match metadata rows to image files present in the configured training path.
    meta['anon_dicom_path'] = meta['anon_dicom_path'].astype(str).str.split('/').str[-1]
    meta['anon_dicom_path'] = meta['anon_dicom_path'].astype(str).str.replace('.dcm', '', regex=False)
    meta = meta[meta['anon_dicom_path'].isin(series_instance_uids)] # this returns 7647 (duplicates are out)

    meta = meta[meta['loc_num'].isin(config.get('client_list'))] # client_list = [1,2,3,5,6,7,8,9] # if only 4 largest sites: [1,2,5,6] - losing around 200 cases
    meta = meta[meta['asses'].isin(['N', 'B', 'M', 'K'])]  # binary classification - losing around 200 cases

    meta_site_dicom = meta[['loc_num', 'anon_dicom_path']]
    meta_site_dicom = meta_site_dicom[meta_site_dicom['anon_dicom_path'].isin(series_instance_uids)]
    meta_site_dicom.drop_duplicates(subset=['anon_dicom_path'], inplace=True)
    #######################################################################################################
    #######################################################################################################

    client_list = config.get('client_list')
    cases_dir = experiment_config_dir / "case_lists"
    cases_dir.mkdir(parents=True, exist_ok=True)
    
    client_python_command = config.get(
        'client_python_command',
        '/home/wenytang/nvflare_example/venv/bin/python -u',
    )

    for i, site in enumerate(client_list):
        client_model_path = config.get('workdir') + f'/EMBED_net_client_{site}.pth'
        global_model_path = config.get('workdir') + f'/EMBED_net_global.pth'
        # client_cases = sites[i]
        client_cases = meta_site_dicom[meta_site_dicom['loc_num'] == site]['anon_dicom_path'].tolist()
        client_cases_file = cases_dir / f"site-{site}_cases.txt"
        client_cases_file.write_text("\n".join(client_cases) + "\n")
        script_args = f"--train_dataset_path {config.get('train_dataset_path')} \
                        --test_dataset_path {config.get('test_dataset_path')} \
                        --batch_size {config.get('batch_size')} \
                        --learning_rate {config.get('learning_rate')} \
                        --client_model_path {client_model_path} \
                        --global_model_path {global_model_path} \
                        --client_config_path {config.get('client_config_path')} \
                        --client_cases_file {client_cases_file.resolve()} \
                        --workdir {Path(config.get('workdir'))}" 
        executor = ScriptRunner(
            script=train_script,
            script_args=script_args,
            launch_external_process=True,
            command=client_python_command,
        )
        job.to(executor, f"site-{site}")

    # # threads=1 or 0 to avoid deadlocks from GPUs and to be able for matplotlib to work properly
    job.simulator_run(workspace=config.get('workdir'), threads=0)
    record_final_global_metrics(config, client_config, logger)
