"""
Collect per-sample losses for all 20 target samples from:
  1. The target model (trained on D_mem)
  2. All 16 IN shadow models per target (trained with target included)
  3. All 16 OUT shadow models per target (trained with target excluded)

Output CSV: shadow_runs/full_shadow_training/loss_collection.csv

Usage:
  python shadow/collect_losses.py
  python shadow/collect_losses.py -c shadow/experiments/full_shadow_training.yml
  python shadow/collect_losses.py --model-type last  (default: last_model.pth)
  python shadow/collect_losses.py --model-type best  (use best_model.pth)
  python shadow/collect_losses.py --output shadow_runs/full_shadow_training/loss_collection_latest.csv
  python shadow/collect_losses.py --checkpoint-root /mnt/hpc/run --attack-manifest /mnt/hpc/run/split/attack_manifest.csv --split-plan /mnt/hpc/run/split/split_plan.json --cleaned-data-root /mnt/c/Data/EMBED/embed_cleaned
"""

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import get_model
from centralized.train_centralized import (
    LABEL_MAP,
    PATH_COLUMNS,
    SqueezeSingletonDepthd,
    build_transforms,
    get_checkpoint_module,
    load_config,
    make_model_dp_compatible,
    select_path_value,
    set_seed,
)
from centralized.input_data import read_csv_rows, read_json, row_id

SHADOW_BASE = Path("shadow_runs/full_shadow_training")
TARGET_MODEL_DIR = SHADOW_BASE / "target_model"
SHADOW_MODELS_DIR = SHADOW_BASE / "shadow_models"
SPLIT_DIR = SHADOW_BASE / "split"
OUTPUT_CSV = SHADOW_BASE / "loss_collection.csv"

CSV_FIELDNAMES = [
    "target_id",
    "sample_id",
    "image_path",
    "membership",
    "model_category",
    "model_index",
    "model_name",
    "model_checkpoint",
    "loss",
    "logit_benign",
    "logit_malignant",
    "confidence_benign",
    "confidence_malignant",
    "label",
    "predicted",
    "correct",
]


def resolve_target_records(rows, split_plan, cleaned_data_root):
    row_by_uid = {}
    for i, row in enumerate(rows):
        uid = str(row.get("uid", "")).strip()
        if uid:
            row_by_uid[uid] = row

    targets = []
    for target in split_plan["targets"]:
        uid = target["sample_id"]
        row = row_by_uid.get(uid)
        if row is None:
            print(f"WARNING: target {target['target_id']} uid {uid} not in manifest")
            continue
        relpath = select_path_value(row)
        if relpath is None:
            print(f"WARNING: no path for target {target['target_id']}")
            continue
        image_path = Path(cleaned_data_root) / relpath
        if not image_path.exists():
            print(f"WARNING: image not found for {target['target_id']}: {image_path}")
            continue
        label = LABEL_MAP[str(row.get("binary_label", "")).strip().lower()]
        targets.append(
            {
                "target_id": target["target_id"],
                "membership": target["membership"],
                "sample_id": uid,
                "image": str(image_path),
                "image_path": str(image_path),
                "label": label,
            }
        )
    return targets


def build_criterion(config, device):
    loss_config = config.get("loss", {})
    label_smoothing = loss_config.get("label_smoothing", 0)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)


def build_model_for_checkpoints(config, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    model = get_model(config.get("model", {}))
    model, _ = make_model_dp_compatible(config, model)
    model = model.to(device)
    model.eval()
    return model


def load_checkpoint_into_model(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_from_checkpoint(config, checkpoint_path, device):
    model = build_model_for_checkpoints(config, device)
    load_checkpoint_into_model(model, checkpoint_path, device)
    return model


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def infer_single_sample(model, image_tensor, label, criterion, device):
    with torch.no_grad():
        inputs = image_tensor.unsqueeze(0).float().to(device)
        label_tensor = torch.tensor([label], dtype=torch.long).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, label_tensor)
        logits = outputs.cpu().numpy()[0]
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])

    return {
        "loss": float(loss.item()),
        "logit_benign": float(logits[0]),
        "logit_malignant": float(logits[1]),
        "confidence_benign": float(probs[0]),
        "confidence_malignant": float(probs[1]),
        "predicted": pred,
        "correct": pred == label,
    }


def preprocess_targets(targets, eval_transform):
    transformed = {}
    for t in targets:
        data_dict = {"image": t["image"], "label": t["label"]}
        transformed[t["target_id"]] = eval_transform(data_dict)
    return transformed


def load_existing_set(output_path):
    if not output_path.exists():
        return set()
    existing = set()
    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["target_id"], row["model_category"], str(row.get("model_index", "")))
            existing.add(key)
    return existing


def non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def limited_targets(targets, max_targets):
    if max_targets is None:
        return targets
    return targets[:max_targets]


def limited_model_indexes(count, max_models):
    limit = count if max_models is None else min(count, max_models)
    return range(1, limit + 1)


def write_row(writer, target, model_category, model_index, model_name,
              checkpoint_path, result):
    writer.writerow(
        {
            "target_id": target["target_id"],
            "sample_id": target["sample_id"],
            "image_path": target["image_path"],
            "membership": target["membership"],
            "model_category": model_category,
            "model_index": model_index,
            "model_name": model_name,
            "model_checkpoint": str(checkpoint_path),
            "loss": result["loss"],
            "logit_benign": result["logit_benign"],
            "logit_malignant": result["logit_malignant"],
            "confidence_benign": result["confidence_benign"],
            "confidence_malignant": result["confidence_malignant"],
            "label": target["label"],
            "predicted": result["predicted"],
            "correct": result["correct"],
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-sample losses from target and shadow models."
    )
    parser.add_argument(
        "-c", "--config", type=str, default="shadow/experiments/full_shadow_training.yml",
    )
    parser.add_argument(
        "--model-type", type=str, default="last", choices=["last", "best"],
        help="Use last_model.pth (last epoch) or best_model.pth (best val metric). Default: last"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: <config workdir>/loss_collection.csv",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help=(
            "Root containing target_model/ and shadow_models/. "
            "Default: config workdir."
        ),
    )
    parser.add_argument(
        "--attack-manifest",
        type=str,
        default=None,
        help="Attack manifest CSV path. Default: input.attack_manifest from config.",
    )
    parser.add_argument(
        "--split-plan",
        type=str,
        default=None,
        help="Split plan JSON path. Default: input.split_plan from config.",
    )
    parser.add_argument(
        "--cleaned-data-root",
        type=str,
        default=None,
        help="Image root override for resolving manifest image paths.",
    )
    parser.add_argument(
        "--max-targets",
        type=non_negative_int,
        default=None,
        help="Limit number of target samples for smoke testing. Default: all.",
    )
    parser.add_argument(
        "--max-in",
        type=non_negative_int,
        default=None,
        help="Limit number of IN shadow models per target. Default: all.",
    )
    parser.add_argument(
        "--max-out",
        type=non_negative_int,
        default=None,
        help="Limit number of OUT shadow models per target. Default: all.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 0))

    workdir = Path(config.get("workdir", SHADOW_BASE))
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else workdir
    target_model_dir = checkpoint_root / "target_model"
    shadow_models_dir = checkpoint_root / "shadow_models"
    split_dir = workdir / "split"
    os.environ.setdefault("TORCH_HOME", str(workdir / "torch_cache"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    model_filename = f"{args.model_type}_model.pth"

    input_config = config.get("input", {})
    split_plan_path = args.split_plan or input_config.get(
        "split_plan", str(split_dir / "split_plan.json")
    )
    attack_manifest_path = args.attack_manifest or input_config.get(
        "attack_manifest", str(split_dir / "attack_manifest.csv")
    )
    cleaned_data_root = args.cleaned_data_root or config["cleaned_data_root"]

    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Attack manifest: {attack_manifest_path}")
    print(f"Split plan: {split_plan_path}")
    print(f"Cleaned data root: {cleaned_data_root}")

    split_plan = read_json(split_plan_path)
    manifest_rows = read_csv_rows(attack_manifest_path)

    targets = resolve_target_records(manifest_rows, split_plan, cleaned_data_root)
    targets = limited_targets(targets, args.max_targets)
    if not targets:
        raise SystemExit("No valid target samples found. Exiting.")

    print(f"Resolved {len(targets)} target samples on {device}")
    for t in targets:
        print(f"  {t['target_id']} ({t['membership']}): label={t['label']}")

    print("Building eval transforms...")
    eval_transform = build_transforms(config, train=False)
    eval_transform.set_random_state(seed=config.get("seed", 0))

    print("Pre-processing target images...")
    transformed_images = preprocess_targets(targets, eval_transform)
    print(f"Pre-processed {len(transformed_images)} images.")

    criterion = build_criterion(config, device)

    output_path = Path(args.output) if args.output else workdir / "loss_collection.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_set(output_path)
    write_header = not output_path.exists()

    print(f"Output file: {output_path}")
    print(f"Existing rows: {len(existing)}")

    total_inferences = 0
    inference_model = build_model_for_checkpoints(config, device)

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writerow({k: k for k in CSV_FIELDNAMES})

        target_ckpt = target_model_dir / model_filename
        if not target_ckpt.exists():
            print(f"ERROR: target model not found: {target_ckpt}")
        else:
            print(f"\n=== Target model ({target_ckpt}) ===")
            print("Loading target model...")
            load_checkpoint_into_model(inference_model, target_ckpt, device)

            for t in targets:
                key = (t["target_id"], "target", "")
                if key in existing:
                    print(f"  SKIP {t['target_id']} (already collected)")
                    continue
                image_tensor = transformed_images[t["target_id"]]["image"]
                result = infer_single_sample(inference_model, image_tensor,
                                             t["label"], criterion, device)
                write_row(
                    writer,
                    t,
                    "target",
                    "",
                    "target_model",
                    target_ckpt,
                    result,
                )
                print(
                    f"  {t['target_id']} ({t['membership']}): "
                    f"loss={result['loss']:.4f} "
                    f"prob_mal={result['confidence_malignant']:.4f} "
                    f"correct={result['correct']}"
                )
                total_inferences += 1

            print("Target model completed.")

        in_subset_count = int(split_plan.get("in_subset_count", 16))

        for t in targets:
            target_id = t["target_id"]
            membership = t["membership"]
            image_tensor = transformed_images[target_id]["image"]
            label = t["label"]

            print(f"\n=== {target_id} ({membership}) shadow models ===")

            in_model_dir = shadow_models_dir / target_id / "IN"
            out_model_dir = shadow_models_dir / target_id / "OUT"

            for model_idx in limited_model_indexes(in_subset_count, args.max_in):
                model_index_str = f"{model_idx:02d}"

                ckpt_in = in_model_dir / f"model_in_{model_index_str}" / model_filename
                key_in = (target_id, "in", model_index_str)
                if not ckpt_in.exists():
                    print(f"  IN_{model_index_str} SKIP (no checkpoint: {ckpt_in})")
                elif key_in in existing:
                    print(f"  IN_{model_index_str} SKIP (already collected)")
                else:
                    load_checkpoint_into_model(inference_model, ckpt_in, device)
                    result = infer_single_sample(inference_model, image_tensor,
                                                 label, criterion, device)
                    write_row(
                        writer,
                        t,
                        "in",
                        model_index_str,
                        f"{target_id}_IN_{model_index_str}",
                        ckpt_in,
                        result,
                    )
                    print(
                        f"  IN_{model_index_str}: "
                        f"loss={result['loss']:.4f} "
                        f"prob_mal={result['confidence_malignant']:.4f} "
                        f"correct={result['correct']}"
                    )
                    total_inferences += 1

            for model_idx in limited_model_indexes(in_subset_count, args.max_out):
                model_index_str = f"{model_idx:02d}"
                ckpt_out = out_model_dir / f"model_out_{model_index_str}" / model_filename
                key_out = (target_id, "out", model_index_str)
                if not ckpt_out.exists():
                    print(f"  OUT_{model_index_str} SKIP (no checkpoint: {ckpt_out})")
                elif key_out in existing:
                    print(f"  OUT_{model_index_str} SKIP (already collected)")
                else:
                    load_checkpoint_into_model(inference_model, ckpt_out, device)
                    result = infer_single_sample(inference_model, image_tensor,
                                                 label, criterion, device)
                    write_row(
                        writer,
                        t,
                        "out",
                        model_index_str,
                        f"{target_id}_OUT_{model_index_str}",
                        ckpt_out,
                        result,
                    )
                    print(
                        f"  OUT_{model_index_str}: "
                        f"loss={result['loss']:.4f} "
                        f"prob_mal={result['confidence_malignant']:.4f} "
                        f"correct={result['correct']}"
                    )
                    total_inferences += 1

    unload_model(inference_model)
    print(f"\nDone. Total new inferences: {total_inferences}")
    existing_after = load_existing_set(output_path)
    print(f"Total rows in CSV: {len(existing_after)}")


if __name__ == "__main__":
    main()
