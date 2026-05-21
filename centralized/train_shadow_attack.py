import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from input_data import (  # noqa: E402
    describe_attack_split,
    ensure_shadow_attack_split,
    load_shadow_attack_records,
    summarize_records,
)
from train_centralized import load_config, save_config  # noqa: E402
from train_centralized_dp import apply_overrides, train  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train target and shadow centralized models from a shadow-attack CSV split."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(Path(__file__).with_name("centralized_config.yml")),
    )
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--generate-split-only", action="store_true")
    parser.add_argument("--skip-target", action="store_true")
    parser.add_argument("--target-id", type=str, default=None)
    parser.add_argument("--condition", choices=["IN", "OUT"], default=None)
    parser.add_argument("--subset-index", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--max-in", type=int, default=None)
    parser.add_argument("--max-out", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained", choices=["true", "false"], default=None)
    parser.add_argument("--grad-output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    config.setdefault("input", {})["mode"] = "shadow_attack"

    rows, split_plan = ensure_shadow_attack_split(config)
    summary = describe_attack_split(config)
    print("Shadow attack split ready.")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.generate_split_only:
        return

    if args.target_id or args.condition or args.subset_index:
        validate_single_shadow_args(args)
        if args.check_data_only:
            shadow_config = deepcopy(config)
        else:
            shadow_config = make_shadow_model_config(
                config,
                target_id=args.target_id,
                condition=args.condition,
                subset_index=args.subset_index,
            )
        records = load_shadow_attack_records(
            shadow_config,
            train_spec={
                "mode": "shadow",
                "target_id": args.target_id,
                "condition": args.condition,
                "subset_index": args.subset_index,
            },
        )
        print(json.dumps(summarize_records(records), indent=2, sort_keys=True))
        if not args.check_data_only:
            train(shadow_config, records)
        return

    if args.check_data_only:
        target_records = load_shadow_attack_records(config)
        first_target = split_plan["targets"][0]["target_id"]
        first_shadow_records = load_shadow_attack_records(
            config,
            train_spec={
                "mode": "shadow",
                "target_id": first_target,
                "condition": "IN",
                "subset_index": 1,
            },
        )
        print("Target model record summary:")
        print(json.dumps(summarize_records(target_records), indent=2, sort_keys=True))
        print(f"First shadow model record summary ({first_target} IN 1):")
        print(json.dumps(summarize_records(first_shadow_records), indent=2, sort_keys=True))
        return

    if not args.skip_target:
        target_config = make_target_model_config(config)
        target_records = load_shadow_attack_records(target_config)
        train(target_config, target_records)

    for target in selected_targets(split_plan, args.max_targets):
        target_id = target["target_id"]
        for subset_index in selected_subset_indexes(split_plan, "IN", args.max_in):
            shadow_config = make_shadow_model_config(config, target_id, "IN", subset_index)
            records = load_shadow_attack_records(
                shadow_config,
                train_spec={
                    "mode": "shadow",
                    "target_id": target_id,
                    "condition": "IN",
                    "subset_index": subset_index,
                },
            )
            train(shadow_config, records)
        for subset_index in selected_subset_indexes(split_plan, "OUT", args.max_out):
            shadow_config = make_shadow_model_config(config, target_id, "OUT", subset_index)
            records = load_shadow_attack_records(
                shadow_config,
                train_spec={
                    "mode": "shadow",
                    "target_id": target_id,
                    "condition": "OUT",
                    "subset_index": subset_index,
                },
            )
            train(shadow_config, records)


def validate_single_shadow_args(args):
    missing = [
        name
        for name, value in {
            "--target-id": args.target_id,
            "--condition": args.condition,
            "--subset-index": args.subset_index,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Single shadow training requires all of: " + ", ".join(missing)
        )


def make_target_model_config(config):
    target_config = deepcopy(config)
    target_config["workdir"] = str(Path(config["workdir"]) / "target_model")
    save_resolved_shadow_config(target_config)
    return target_config


def make_shadow_model_config(config, target_id, condition, subset_index):
    shadow_config = deepcopy(config)
    shadow_config["workdir"] = str(
        Path(config["workdir"])
        / "shadow_models"
        / target_id
        / condition
        / f"model_{condition.lower()}_{subset_index:02d}"
    )
    save_resolved_shadow_config(shadow_config)
    return shadow_config


def save_resolved_shadow_config(config):
    workdir = Path(config["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    save_config(config, workdir / "shadow_resolved_config.yml")


def selected_targets(split_plan, max_targets):
    targets = split_plan["targets"]
    if max_targets is not None:
        return targets[:max_targets]
    return targets


def selected_subset_indexes(split_plan, condition, max_count):
    in_subset_count = int(split_plan["in_subset_count"])
    count = in_subset_count if max_count is None else min(max_count, in_subset_count)
    return range(1, count + 1)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    main()
