import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CENTRALIZED_DIR = REPO_ROOT / "centralized"
for path in (REPO_ROOT, CENTRALIZED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from centralized.train_centralized import load_config, save_config  # noqa: E402
from centralized.train_centralized_dp import apply_overrides, train  # noqa: E402
from shadow.train.input_data import (  # noqa: E402
    describe_attack_split,
    ensure_shadow_attack_split,
    load_shadow_attack_records,
    summarize_records,
)


DEFAULT_COMPLETED_FILES = ["metrics.json", "best_model.pth", "last_model.pth"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run target and shadow model training from a shadow split plan."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(REPO_ROOT / "shadow" / "experiments" / "full_shadow_training.yml"),
    )
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--generate-split-only", action="store_true")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-in", type=int, default=None)
    parser.add_argument("--max-out", type=int, default=None)
    parser.add_argument("--gpu-nodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained", choices=["true", "false"], default=None)
    parser.add_argument("--grad-output", action="store_true")
    parser.add_argument("--job-index", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["_config_path"] = args.config
    config = apply_overrides(config, args)
    config.setdefault("input", {})["mode"] = "shadow_attack"
    if args.gpu_nodes is not None:
        config.setdefault("shadow_training", {})["gpu_nodes"] = args.gpu_nodes
    apply_limit_overrides(config, args)

    rows, split_plan = ensure_shadow_attack_split(config)
    split_summary = describe_attack_split(config)
    print("Shadow split ready.")
    print(json.dumps(split_summary, indent=2, sort_keys=True))

    if args.generate_split_only:
        return

    jobs = build_jobs(config, split_plan)
    if args.job_index is None:
        write_run_manifest(config, jobs, split_summary)

    if args.check_data_only:
        print_check_data_summary(config, jobs)
        return

    if args.job_index is not None:
        job = jobs[args.job_index]
        result = run_job(config, job, force=args.force)
        if result["status"] == "failed":
            raise RuntimeError(result["error"])
        return

    run_jobs(config, args, jobs)


def apply_limit_overrides(config, args):
    shadow_training = config.setdefault("shadow_training", {})
    for attr, key in (
        ("max_targets", "max_targets"),
        ("max_in", "max_in"),
        ("max_out", "max_out"),
    ):
        value = getattr(args, attr)
        if value is not None:
            shadow_training[key] = value


def build_jobs(config, split_plan):
    shadow_training = config.get("shadow_training", {})
    jobs = []
    if shadow_training.get("run_target", True):
        jobs.append(
            {
                "index": len(jobs),
                "kind": "target",
                "name": "target_model",
                "workdir": str(Path(config["workdir"]) / "target_model"),
                "train_spec": {"mode": "target"},
            }
        )

    if shadow_training.get("run_shadows", True):
        targets = split_plan["targets"]
        max_targets = shadow_training.get("max_targets")
        if max_targets is not None:
            targets = targets[: int(max_targets)]
        in_count = limited_count(split_plan["in_subset_count"], shadow_training.get("max_in"))
        out_count = limited_count(split_plan["in_subset_count"], shadow_training.get("max_out"))
        for target in targets:
            target_id = target["target_id"]
            for subset_index in range(1, in_count + 1):
                jobs.append(make_shadow_job(config, len(jobs), target_id, "IN", subset_index))
            for subset_index in range(1, out_count + 1):
                jobs.append(make_shadow_job(config, len(jobs), target_id, "OUT", subset_index))
    return jobs


def limited_count(default_count, limit):
    default_count = int(default_count)
    if limit is None:
        return default_count
    return min(int(limit), default_count)


def make_shadow_job(config, index, target_id, condition, subset_index):
    return {
        "index": index,
        "kind": "shadow",
        "name": f"{target_id}_{condition}_{subset_index:02d}",
        "target_id": target_id,
        "condition": condition,
        "subset_index": subset_index,
        "workdir": str(
            Path(config["workdir"])
            / "shadow_models"
            / target_id
            / condition
            / f"model_{condition.lower()}_{subset_index:02d}"
        ),
        "train_spec": {
            "mode": "shadow",
            "target_id": target_id,
            "condition": condition,
            "subset_index": subset_index,
        },
    }


def print_check_data_summary(config, jobs):
    print(f"Training jobs selected: {len(jobs)}")
    for job in jobs[: min(3, len(jobs))]:
        records = load_records_for_job(config, job)
        print(f"{job['index']:03d} {job['name']}:")
        print(json.dumps(summarize_records(records), indent=2, sort_keys=True))
    if len(jobs) > 3:
        print(f"... {len(jobs) - 3} additional jobs omitted from check-data output.")


def run_jobs(config, args, jobs):
    shadow_training = config.get("shadow_training", {})
    force = args.force
    gpu_nodes = int(shadow_training.get("gpu_nodes", 1))
    stop_on_failure = bool(shadow_training.get("stop_on_failure", True))
    summary_path = run_summary_path(config)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("")

    if gpu_nodes <= 1:
        for job in jobs:
            result = run_job(config, job, force=force)
            append_jsonl(summary_path, result)
            if result["status"] == "failed" and stop_on_failure:
                raise RuntimeError(result["error"])
        return

    validate_gpu_nodes(gpu_nodes)
    run_jobs_subprocess(config, args, jobs, gpu_nodes, summary_path, stop_on_failure)


def run_jobs_subprocess(config, args, jobs, gpu_nodes, summary_path, stop_on_failure):
    pending = list(jobs)
    with ThreadPoolExecutor(max_workers=gpu_nodes) as executor:
        futures = {}
        for gpu_index in range(min(gpu_nodes, len(pending))):
            job = next_pending_job(config, pending, args.force, summary_path)
            if job is None:
                break
            futures[executor.submit(run_job_subprocess, args, job, gpu_index)] = job

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                result = future.result()
                append_jsonl(summary_path, result)
                if result["status"] == "failed" and stop_on_failure:
                    for remaining in futures:
                        remaining.cancel()
                    raise RuntimeError(result["error"])
                gpu_index = result.get("gpu_index", 0)
                next_job = next_pending_job(config, pending, args.force, summary_path)
                if next_job is not None:
                    futures[executor.submit(run_job_subprocess, args, next_job, gpu_index)] = next_job


def next_pending_job(config, pending, force, summary_path):
    while pending:
        job = pending.pop(0)
        if should_skip_job(config, job, force):
            started_at = time.time()
            result = base_result(job, started_at, started_at)
            result["status"] = "skipped"
            append_jsonl(summary_path, result)
            continue
        return job
    return None


def run_job_subprocess(args, job, gpu_index):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "-c",
        args.config,
        "--job-index",
        str(job["index"]),
    ]
    pass_through_args(args, command)
    if args.force:
        command.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    started_at = time.time()
    completed = subprocess.run(command, env=env, text=True)
    ended_at = time.time()
    result = base_result(job, started_at, ended_at)
    result["gpu_index"] = gpu_index
    result["returncode"] = completed.returncode
    if completed.returncode == 0:
        result["status"] = "completed"
    else:
        result["status"] = "failed"
        result["error"] = f"Job subprocess exited with code {completed.returncode}"
    return result


def pass_through_args(args, command):
    for name, flag in (
        ("epochs", "--epochs"),
        ("batch_size", "--batch-size"),
        ("num_workers", "--num-workers"),
        ("workdir", "--workdir"),
        ("seed", "--seed"),
        ("pretrained", "--pretrained"),
        ("max_targets", "--max-targets"),
        ("max_in", "--max-in"),
        ("max_out", "--max-out"),
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend([flag, str(value)])
    if args.grad_output:
        command.append("--grad-output")


def validate_gpu_nodes(gpu_nodes):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("gpu_nodes > 1 requires PyTorch to validate CUDA devices.") from exc
    visible_count = torch.cuda.device_count()
    if visible_count < gpu_nodes:
        raise ValueError(
            f"shadow_training.gpu_nodes={gpu_nodes} exceeds visible CUDA devices={visible_count}."
        )


def run_job(config, job, force=False):
    started_at = time.time()
    try:
        if should_skip_job(config, job, force):
            ended_at = time.time()
            result = base_result(job, started_at, ended_at)
            result["status"] = "skipped"
            return result

        job_config = make_job_config(config, job)
        records = load_records_for_job(job_config, job)
        train(job_config, records)
        ended_at = time.time()
        result = base_result(job, started_at, ended_at)
        result["status"] = "completed"
        return result
    except Exception as exc:
        ended_at = time.time()
        result = base_result(job, started_at, ended_at)
        result["status"] = "failed"
        result["error"] = str(exc)
        return result


def should_skip_job(config, job, force):
    if force:
        return False
    shadow_training = config.get("shadow_training", {})
    if not shadow_training.get("skip_completed", True):
        return False
    completed_files = shadow_training.get("completed_files", DEFAULT_COMPLETED_FILES)
    workdir = Path(job["workdir"])
    return all((workdir / filename).exists() for filename in completed_files)


def make_job_config(config, job):
    job_config = deepcopy(config)
    job_config["workdir"] = job["workdir"]
    Path(job_config["workdir"]).mkdir(parents=True, exist_ok=True)
    save_config(job_config, Path(job_config["workdir"]) / "shadow_resolved_config.yml")
    return job_config


def load_records_for_job(config, job):
    train_spec = job.get("train_spec")
    if train_spec and train_spec.get("mode") == "target":
        train_spec = None
    return load_shadow_attack_records(config, train_spec=train_spec)


def base_result(job, started_at, ended_at):
    return {
        "job_index": job["index"],
        "job_name": job["name"],
        "kind": job["kind"],
        "workdir": job["workdir"],
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": ended_at - started_at,
    }


def write_run_manifest(config, jobs, split_summary):
    path = run_manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(
            Path(
                config.get(
                    "_config_path",
                    "shadow/experiments/full_shadow_training.yml",
                )
            )
        ),
        "split_summary": split_summary,
        "job_count": len(jobs),
        "jobs": jobs,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path, payload):
    with Path(path).open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def run_manifest_path(config):
    return Path(config["workdir"]) / "run_manifest.json"


def run_summary_path(config):
    return Path(config["workdir"]) / "run_summary.jsonl"


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    main()
