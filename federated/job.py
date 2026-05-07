import os
import argparse
import json
import math
import pandas as pd
import shlex
import shutil
import sys
import time
import torch

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.fuel.utils.log_utils import get_script_logger
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Job Runner")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def resolve_path(path_value, base_dir):
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists() or path.parts[:1] in {("federated_runs",), ("federated",), ("centralized",)}:
        return repo_path
    return base_dir / path


def preserve_resume_checkpoint(checkpoint_path, workdir, completed_round, logger):
    preserve_dir = Path(workdir).parent / "resume_checkpoints"
    preserve_dir.mkdir(parents=True, exist_ok=True)
    preserved_path = preserve_dir / f"round_{completed_round}_FL_global_model.pt"
    if checkpoint_path.resolve() != preserved_path.resolve():
        shutil.copy2(checkpoint_path, preserved_path)
        logger.info(f"Copied resume checkpoint to stable path: {preserved_path}")
    return preserved_path


def normalize_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_resume_state(model, config, logger):
    resume_config = config.get("resume") or {}
    if not normalize_bool(resume_config.get("enabled"), default=False):
        return {
            "enabled": False,
            "round_offset": 0,
            "executed_rounds": int(config.get("num_rounds") or 0),
            "total_target_rounds": int(config.get("num_rounds") or 0),
        }

    checkpoint_path = resolve_path(resume_config.get("checkpoint_path"), Path(config.get("workdir")).parent)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        meta_props = checkpoint.get("meta_props", {})
    else:
        state_dict = checkpoint
        meta_props = {}

    completed_round = int(resume_config.get("completed_round", meta_props.get("current_round", -1)))
    total_target_rounds = int(resume_config.get("total_target_rounds", config.get("num_rounds")))
    expected_aggregated = int(config.get("n_clients") or 0)
    nr_aggregated = meta_props.get("nr_aggregated")
    if nr_aggregated is not None and expected_aggregated and int(nr_aggregated) != expected_aggregated:
        raise ValueError(
            f"Resume checkpoint is not fully aggregated: nr_aggregated={nr_aggregated}, "
            f"expected={expected_aggregated}"
        )

    model.load_state_dict(state_dict)
    preserved_checkpoint_path = preserve_resume_checkpoint(
        checkpoint_path=checkpoint_path,
        workdir=config.get("workdir"),
        completed_round=completed_round,
        logger=logger,
    )
    round_offset = completed_round + 1
    executed_rounds = total_target_rounds - round_offset
    if executed_rounds <= 0:
        raise ValueError(
            f"Resume target is already complete: completed_round={completed_round}, "
            f"total_target_rounds={total_target_rounds}"
        )
    config["num_rounds"] = executed_rounds
    logger.info(
        f"Soft-resuming from {checkpoint_path}: completed_round={completed_round}, "
        f"round_offset={round_offset}, remaining_rounds={executed_rounds}, "
        f"total_target_rounds={total_target_rounds}"
    )
    return {
        "enabled": True,
        "checkpoint_path": str(checkpoint_path),
        "preserved_checkpoint_path": str(preserved_checkpoint_path),
        "checkpoint_meta_props": meta_props,
        "completed_round": completed_round,
        "round_offset": round_offset,
        "executed_rounds": executed_rounds,
        "total_target_rounds": total_target_rounds,
        "effective_round_start": round_offset,
        "effective_round_end": total_target_rounds - 1,
    }


def build_manifest_client_cases(config, logger):
    data_csv = Path(config.get("data_csv"))
    if not data_csv.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {data_csv}")

    meta = pd.read_csv(data_csv)
    required = {"loc_num", "model_split", "binary_label"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns for federated split: {sorted(missing)}")
    if "image_path_suffix" not in meta.columns and "output_relpath" not in meta.columns:
        raise ValueError("Manifest must include image_path_suffix or output_relpath.")

    split_config = config.get("splits", {})
    train_split = split_config.get("train", "train")
    validation_split = split_config.get("validation", "validation")
    test_split = split_config.get("test", "test")
    allowed_splits = {train_split, validation_split, test_split}

    meta["model_split"] = meta["model_split"].astype(str).str.strip()
    invalid_splits = set(meta["model_split"].unique()) - allowed_splits
    if invalid_splits:
        raise ValueError(f"Unexpected model_split values: {sorted(invalid_splits)}")

    client_list = config.get("client_list")
    meta = meta[meta["loc_num"].isin(client_list)]
    logger.info(f"Loaded manifest rows for configured clients: {len(meta)}")

    client_cases = {}
    for site in client_list:
        site_rows = meta[meta["loc_num"] == site]
        client_cases[site] = site_rows.index.astype(str).tolist()
        train_count = int((site_rows["model_split"] == train_split).sum())
        val_count = int((site_rows["model_split"] == validation_split).sum())
        test_count = int((site_rows["model_split"] == test_split).sum())
        logger.info(
            f"site-{site}: train={train_count}, validation={val_count}, test={test_count}"
        )

    return client_cases


def summarize_manifest_by_client(config):
    data_csv = Path(config.get("data_csv"))
    meta = pd.read_csv(data_csv)
    split_config = config.get("splits", {})
    train_split = split_config.get("train", "train")
    validation_split = split_config.get("validation", "validation")
    test_split = split_config.get("test", "test")

    summary = {}
    for site in config.get("client_list"):
        site_rows = meta[meta["loc_num"] == site]
        summary[f"site-{site}"] = {
            "train": int((site_rows["model_split"] == train_split).sum()),
            "validation": int((site_rows["model_split"] == validation_split).sum()),
            "test": int((site_rows["model_split"] == test_split).sum()),
        }
    return summary


def write_run_metrics(config, client_list, duration_seconds, simulator_settings, resume_state):
    workdir = Path(config.get("workdir"))
    sites = {}
    global_test = []
    for site in client_list:
        site_name = f"site-{site}"
        site_metrics_path = workdir / site_name / "metrics.json"
        if not site_metrics_path.exists():
            sites[site_name] = {"error": f"missing site metrics file: {site_metrics_path}"}
            continue
        with open(site_metrics_path, "r") as f:
            site_metrics = json.load(f)
        sites[site_name] = site_metrics
        for test_record in site_metrics.get("global_test", []):
            enriched = dict(test_record)
            enriched["site"] = site_name
            global_test.append(enriched)

    executed_rounds = int(config.get("num_rounds", 0) or 0)
    total_target_rounds = int(resume_state.get("total_target_rounds", executed_rounds) or executed_rounds)
    metrics = {
        "summary": {
            "recipe": config.get("recipe"),
            "num_rounds": total_target_rounds,
            "executed_rounds": executed_rounds,
            "client_list": client_list,
            "data_csv": config.get("data_csv"),
            "cleaned_data_root": config.get("cleaned_data_root"),
            "workdir": config.get("workdir"),
            "simulator": simulator_settings,
            "resume": resume_state,
            "split_counts_by_site": summarize_manifest_by_client(config) if config.get("data_csv") else {},
        },
        "duration_seconds": float(duration_seconds),
        "duration": format_duration(duration_seconds),
        "average_round_seconds": float(duration_seconds / executed_rounds) if executed_rounds else math.nan,
        "average_round_duration": format_duration(duration_seconds / executed_rounds) if executed_rounds else None,
        "sites": sites,
        "global_test": global_test,
    }
    with open(workdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)
    return metrics


def build_legacy_client_cases(config):
    train_data_path = Path(config.get('train_dataset_path'))
    meta = pd.read_csv(config.get('meta_data_path'))

    cases = [c for c in train_data_path.rglob('*') if c.is_file()]
    series_instance_uids = [c.parents[0].name for c in cases]

    meta['anon_dicom_path'] = meta['anon_dicom_path'].astype(str).str.split('/').str[-1]
    meta['anon_dicom_path'] = meta['anon_dicom_path'].astype(str).str.replace('.dcm', '', regex=False)
    meta = meta[meta['anon_dicom_path'].isin(series_instance_uids)]
    meta = meta[meta['loc_num'].isin(config.get('client_list'))]
    meta = meta[meta['asses'].isin(['N', 'B', 'M', 'K'])]

    meta_site_dicom = meta[['loc_num', 'anon_dicom_path']]
    meta_site_dicom = meta_site_dicom[meta_site_dicom['anon_dicom_path'].isin(series_instance_uids)]
    meta_site_dicom.drop_duplicates(subset=['anon_dicom_path'], inplace=True)

    return {
        site: meta_site_dicom[meta_site_dicom['loc_num'] == site]['anon_dicom_path'].tolist()
        for site in config.get('client_list')
    }


if __name__ == "__main__":
    logger = get_script_logger()

    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    workdir = Path(config.get("workdir"))
    config["workdir"] = str(workdir if workdir.is_absolute() else REPO_ROOT / workdir)

    client_config_path = resolve_path(config.get('client_config_path'), config_dir)
    train_script = resolve_path(config.get('client_script'), REPO_ROOT)
    try:
        train_script_for_runner = str(train_script.relative_to(REPO_ROOT))
    except ValueError:
        train_script_for_runner = str(train_script)
    config['client_config_path'] = str(client_config_path)
    config['client_script'] = str(train_script)
    if config.get("data_csv"):
        config["data_csv"] = str(resolve_path(config.get("data_csv"), config_dir))
    torch_home = Path(os.environ.get("TORCH_HOME", Path(config.get("workdir")) / "torch_cache"))
    if not torch_home.is_absolute():
        torch_home = REPO_ROOT / torch_home
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    # Limit glibc memory arenas to reduce RSS fragmentation across FL rounds.
    # NVFlare's own memory_utils.py recommends MALLOC_ARENA_MAX=2 for client processes.
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not os.environ.get("PYTHONPATH")
        else f"{REPO_ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
    )

    experiment_config_dir = Path(f"{config.get('workdir')}_config")
    if not os.path.exists(experiment_config_dir):
        os.makedirs(experiment_config_dir)

    # To run multiple experiments, save a copy of the config file in the experiment directory
    save_config(config, os.path.join(experiment_config_dir, 'server_config.yml'))
    client_config = load_config(client_config_path)
    save_config(client_config, os.path.join(experiment_config_dir, 'client_config.yml'))
    
    n_clients = config.get('n_clients')
    train_script = train_script_for_runner
    
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

    resume_state = load_resume_state(model, config, logger)
    num_rounds = config.get('num_rounds')

    if config.get('recipe') == 'fedavg':
        job = BaseFedJob(name='fedavg', initial_model=model, min_clients=n_clients)
        controller = FedAvg(
            num_clients=n_clients,
            num_rounds=num_rounds,
            start_round=resume_state.get("round_offset", 0),
            persistor_id=job.comp_ids["persistor_id"],
        )
        job.to_server(controller)

    client_list = config.get('client_list')
    if config.get("data_csv"):
        cases_by_client = build_manifest_client_cases(config, logger)
    else:
        cases_by_client = build_legacy_client_cases(config)
    
    for i, site in enumerate(client_list):
        client_model_path = config.get('workdir') + f'/EMBED_net_client_{site}.pth'
        global_model_path = config.get('workdir') + f'/EMBED_net_global.pth'
        script_args = [
            "--batch_size", str(config.get('batch_size')),
            "--learning_rate", str(config.get('learning_rate')),
            "--client_model_path", client_model_path,
            "--global_model_path", global_model_path,
            "--client_config_path", config.get('client_config_path'),
            "--workdir", str(Path(config.get('workdir'))),
            "--client_site", str(site),
            "--round_offset", str(resume_state.get("round_offset", 0)),
            "--total_target_rounds", str(resume_state.get("total_target_rounds", num_rounds)),
        ]
        if config.get("data_csv"):
            script_args.extend([
                "--data_csv", config.get("data_csv"),
                "--cleaned_data_root", config.get("cleaned_data_root"),
            ])
        else:
            script_args.extend(["--client_cases", ','.join(cases_by_client[site])])
            script_args.extend([
                "--train_dataset_path", config.get('train_dataset_path'),
                "--test_dataset_path", config.get('test_dataset_path'),
            ])
        simulator_config = config.get("simulator") or {}
        executor_mode = simulator_config.get("executor_mode", "in_process")
        if executor_mode not in {"in_process", "external_per_task"}:
            raise ValueError("simulator.executor_mode must be 'in_process' or 'external_per_task'")
        if executor_mode == "external_per_task":
            script_args.append("--single_task_exit")

        script_args = shlex.join(script_args)
        executor = ScriptRunner(
            script=train_script,
            script_args=script_args,
            launch_external_process=executor_mode == "external_per_task",
            command=f"{sys.executable} -u",
            launch_once=False,
            shutdown_timeout=float(simulator_config.get("shutdown_timeout", 30.0)),
            memory_gc_rounds=int(simulator_config.get("memory_gc_rounds", 1)),
            cuda_empty_cache=normalize_bool(simulator_config.get("cuda_empty_cache"), default=True),
        )
        job.to(executor, f"site-{site}")

    simulator_config = config.get("simulator") or {}
    simulator_settings = {
        "threads": int(simulator_config.get("threads", 1)),
        "gpu": str(simulator_config.get("gpu", "0")),
        "executor_mode": simulator_config.get("executor_mode", "in_process"),
        "memory_gc_rounds": int(simulator_config.get("memory_gc_rounds", 1)),
        "cuda_empty_cache": normalize_bool(simulator_config.get("cuda_empty_cache"), default=True),
    }
    start_time = time.perf_counter()
    job.simulator_run(
        workspace=config.get('workdir'),
        threads=simulator_settings["threads"],
        gpu=simulator_settings["gpu"],
    )
    duration_seconds = time.perf_counter() - start_time
    run_metrics = write_run_metrics(config, client_list, duration_seconds, simulator_settings, resume_state)
    logger.info(
        f"Federated run completed in {run_metrics['duration']} "
        f"(avg_round={run_metrics['average_round_duration']})"
    )
