import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from centralized.train_centralized import LABEL_MAP, PATH_COLUMNS, load_manifest
except ModuleNotFoundError:
    from train_centralized import LABEL_MAP, PATH_COLUMNS, load_manifest


REQUIRED_MANIFEST_COLUMNS = {"binary_label"}
ATTACK_POOL_COLUMN = "attack_pool"
ATTACK_TARGET_ROLE_COLUMN = "attack_target_role"
ATTACK_TARGET_ID_COLUMN = "attack_target_id"
SHADOW_POOL_COLUMN = "shadow_pool"
ATTACK_MANIFEST_DROP_COLUMNS = {
    "empi_anon",
    "acc_anon",
    "asses",
    "path_severity",
    "source",
}


DEFAULT_POOL_COUNTS = {
    "D_mem": 3364,
    "D_non": 1122,
    "D_aux": 1122,
}
DEFAULT_TARGET_COUNTS = {
    "member": 10,
    "nonmember": 10,
}


def load_records(config):
    input_config = config.get("input", {})
    mode = input_config.get("mode", "manifest")
    if mode == "manifest":
        return load_manifest(config)
    if mode == "shadow_attack":
        return load_shadow_attack_records(config)
    raise ValueError(f"Unsupported input.mode: {mode}")


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


def load_shadow_attack_records(config, train_spec=None):
    rows, split_plan = ensure_shadow_attack_split(config)
    train_records, validation_records, test_records = select_shadow_attack_rows(
        rows,
        split_plan,
        train_spec=train_spec,
    )
    records_by_split = {
        "train": rows_to_records(train_records, config),
        "validation": rows_to_records(validation_records, config),
        "test": rows_to_records(test_records, config),
    }
    validate_nonempty_splits(records_by_split)
    return records_by_split


def ensure_shadow_attack_split(config):
    input_config = config.get("input", {})
    attack_manifest_path = resolve_output_path(
        input_config.get("attack_manifest"),
        config,
        "attack_manifest.csv",
    )
    split_plan_path = resolve_output_path(
        input_config.get("split_plan"),
        config,
        "split_plan.json",
    )
    regenerate = bool(input_config.get("regenerate", False))

    if not regenerate and attack_manifest_path.exists() and split_plan_path.exists():
        return read_csv_rows(attack_manifest_path), read_json(split_plan_path)

    rows = read_source_manifest(config)
    split_plan = build_shadow_attack_split_plan(rows, config)
    apply_shadow_attack_plan(rows, split_plan)
    write_csv_rows(attack_manifest_path, rows)
    write_json(split_plan_path, split_plan)
    return rows, split_plan


def read_source_manifest(config):
    data_csv = Path(config["data_csv"])
    if not data_csv.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {data_csv}")
    rows = read_csv_rows(data_csv)
    if not rows:
        raise ValueError(f"Manifest CSV is empty: {data_csv}")

    fieldnames = set(rows[0].keys())
    missing_columns = REQUIRED_MANIFEST_COLUMNS - fieldnames
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")
    if not any(column in fieldnames for column in PATH_COLUMNS):
        raise ValueError("Manifest must include image_path_suffix or output_relpath.")

    invalid_labels = {
        str(row.get("binary_label", "")).strip().lower()
        for row in rows
        if str(row.get("binary_label", "")).strip().lower() not in LABEL_MAP
    }
    if invalid_labels:
        raise ValueError(f"Unexpected binary_label values: {sorted(invalid_labels)}")
    return rows


def build_shadow_attack_split_plan(rows, config):
    shadow_config = config.get("shadow_attack", {})
    seed = int(config.get("seed", 0))
    rng = random.Random(seed)

    pool_counts = dict(DEFAULT_POOL_COUNTS)
    pool_counts.update(shadow_config.get("pool_counts", {}))
    target_counts = dict(DEFAULT_TARGET_COUNTS)
    target_counts.update(shadow_config.get("target_counts", {}))

    total_requested = sum(int(v) for v in pool_counts.values())
    if total_requested != len(rows):
        raise ValueError(
            "shadow_attack.pool_counts must sum to the source manifest row count: "
            f"{total_requested} != {len(rows)}"
        )

    row_ids = [row_id(row, index) for index, row in enumerate(rows)]
    ids_by_label = defaultdict(list)
    for index, row in enumerate(rows):
        ids_by_label[normalized_label(row)].append(row_ids[index])
    for ids in ids_by_label.values():
        rng.shuffle(ids)

    pool_ids = stratified_partition(ids_by_label, pool_counts, rows)
    target_ids = {
        "T_mem": stratified_take(pool_ids["D_mem"], rows, int(target_counts["member"]), rng),
        "T_non": stratified_take(pool_ids["D_non"], rows, int(target_counts["nonmember"]), rng),
    }

    shadow_pool_size = int(shadow_config.get("shadow_pool_size", 800))
    if shadow_pool_size > len(pool_ids["D_aux"]):
        raise ValueError(
            f"shadow_pool_size={shadow_pool_size} exceeds D_aux size={len(pool_ids['D_aux'])}"
        )
    shadow_pool_ids = stratified_take(pool_ids["D_aux"], rows, shadow_pool_size, rng)

    subset_count = int(shadow_config.get("shadow_subset_count", 32))
    subset_size = int(shadow_config.get("shadow_subset_size", 500))
    shadow_subsets = {}
    for subset_index in range(1, subset_count + 1):
        shadow_subsets[f"S_{subset_index:02d}"] = [
            rng.choice(shadow_pool_ids) for _ in range(subset_size)
        ]

    d_non_eval = split_eval_ids(
        pool_ids["D_non"],
        rows,
        float(shadow_config.get("nonmember_validation_fraction", 1 / 3)),
        rng,
    )

    plan = {
        "seed": seed,
        "source_manifest": str(Path(config["data_csv"])),
        "pool_counts": {key: len(value) for key, value in pool_ids.items()},
        "target_counts": {key: len(value) for key, value in target_ids.items()},
        "shadow_pool_size": len(shadow_pool_ids),
        "shadow_subset_count": subset_count,
        "shadow_subset_size": subset_size,
        "in_subset_count": int(shadow_config.get("in_subset_count", 16)),
        "pools": pool_ids,
        "targets": make_targets(target_ids),
        "shadow_pool": shadow_pool_ids,
        "shadow_subsets": shadow_subsets,
        "validation_ids": d_non_eval["validation"],
        "test_ids": d_non_eval["test"],
    }
    validate_shadow_attack_plan(plan)
    return plan


def stratified_partition(ids_by_label, pool_counts, rows):
    pools = {pool_name: [] for pool_name in pool_counts}
    remaining_by_label = {label: list(ids) for label, ids in ids_by_label.items()}

    pool_items = list(pool_counts.items())
    for pool_index, (pool_name, requested_count) in enumerate(pool_items):
        remaining_label_totals = {
            label: len(ids) for label, ids in remaining_by_label.items()
        }
        remaining_total = sum(remaining_label_totals.values())
        if pool_index == len(pool_items) - 1:
            allocations = remaining_label_totals
        else:
            allocations = proportional_allocations(
                remaining_label_totals,
                remaining_total,
                int(requested_count),
            )
        for label, count in allocations.items():
            available = remaining_by_label[label]
            if count > len(available):
                raise ValueError(
                    f"Not enough {label} rows for {pool_name}: need {count}, have {len(available)}"
                )
            pools[pool_name].extend(available[:count])
            del available[:count]

    for pool_ids in pools.values():
        pool_ids.sort(key=lambda sample_id: id_sort_key(sample_id, rows))
    return pools


def proportional_allocations(label_totals, total, requested_count):
    raw = {
        label: (count / total) * requested_count
        for label, count in label_totals.items()
    }
    allocations = {label: int(value) for label, value in raw.items()}
    remaining = requested_count - sum(allocations.values())
    remainders = sorted(
        raw,
        key=lambda label: (raw[label] - allocations[label], label),
        reverse=True,
    )
    for label in remainders[:remaining]:
        allocations[label] += 1
    return allocations


def stratified_take(candidate_ids, rows, count, rng):
    ids_by_label = defaultdict(list)
    for sample_id in candidate_ids:
        ids_by_label[label_by_id(sample_id, rows)].append(sample_id)
    for ids in ids_by_label.values():
        rng.shuffle(ids)
    allocations = proportional_allocations(
        {label: len(ids) for label, ids in ids_by_label.items()},
        len(candidate_ids),
        count,
    )
    selected = []
    for label, label_count in allocations.items():
        selected.extend(ids_by_label[label][:label_count])
    rng.shuffle(selected)
    return selected


def split_eval_ids(candidate_ids, rows, validation_fraction, rng):
    validation_count = round(len(candidate_ids) * validation_fraction)
    validation_ids = set(stratified_take(candidate_ids, rows, validation_count, rng))
    return {
        "validation": [sample_id for sample_id in candidate_ids if sample_id in validation_ids],
        "test": [sample_id for sample_id in candidate_ids if sample_id not in validation_ids],
    }


def make_targets(target_ids):
    targets = []
    for index, sample_id in enumerate(target_ids["T_mem"], start=1):
        targets.append(
            {"target_id": f"target_x_{index:02d}", "membership": "member", "sample_id": sample_id}
        )
    start = len(target_ids["T_mem"]) + 1
    for offset, sample_id in enumerate(target_ids["T_non"], start=start):
        targets.append(
            {
                "target_id": f"target_x_{offset:02d}",
                "membership": "nonmember",
                "sample_id": sample_id,
            }
        )
    return targets


def apply_shadow_attack_plan(rows, plan):
    row_lookup = {row_id(row, index): row for index, row in enumerate(rows)}
    for row in rows:
        row[ATTACK_POOL_COLUMN] = ""
        row[ATTACK_TARGET_ROLE_COLUMN] = ""
        row[ATTACK_TARGET_ID_COLUMN] = ""
        row[SHADOW_POOL_COLUMN] = "false"

    for pool_name, ids in plan["pools"].items():
        for sample_id in ids:
            row_lookup[sample_id][ATTACK_POOL_COLUMN] = pool_name
    for sample_id in plan["shadow_pool"]:
        row_lookup[sample_id][SHADOW_POOL_COLUMN] = "true"
    for target in plan["targets"]:
        row = row_lookup[target["sample_id"]]
        row[ATTACK_TARGET_ROLE_COLUMN] = (
            "T_mem" if target["membership"] == "member" else "T_non"
        )
        row[ATTACK_TARGET_ID_COLUMN] = target["target_id"]


def select_shadow_attack_rows(rows, split_plan, train_spec=None):
    rows_by_id = {row_id(row, index): row for index, row in enumerate(rows)}
    if train_spec is None or train_spec.get("mode", "target") == "target":
        train_ids = split_plan["pools"]["D_mem"]
    else:
        train_ids = shadow_train_ids(split_plan, train_spec)

    return (
        [rows_by_id[sample_id] for sample_id in train_ids],
        [rows_by_id[sample_id] for sample_id in split_plan["validation_ids"]],
        [rows_by_id[sample_id] for sample_id in split_plan["test_ids"]],
    )


def shadow_train_ids(split_plan, train_spec):
    target_id = train_spec["target_id"]
    condition = train_spec["condition"]
    subset_index = int(train_spec["subset_index"])
    in_subset_count = int(split_plan["in_subset_count"])
    if condition == "IN":
        if not 1 <= subset_index <= in_subset_count:
            raise ValueError(f"IN subset_index must be 1..{in_subset_count}: {subset_index}")
        subset_name = f"S_{subset_index:02d}"
        target_sample_id = target_sample_id_by_target_id(split_plan, target_id)
        return list(split_plan["shadow_subsets"][subset_name]) + [target_sample_id]
    if condition == "OUT":
        if not 1 <= subset_index <= in_subset_count:
            raise ValueError(f"OUT subset_index must be 1..{in_subset_count}: {subset_index}")
        subset_name = f"S_{subset_index + in_subset_count:02d}"
        return list(split_plan["shadow_subsets"][subset_name])
    raise ValueError(f"Unsupported shadow train condition: {condition}")


def target_sample_id_by_target_id(split_plan, target_id):
    for target in split_plan["targets"]:
        if target["target_id"] == target_id:
            return target["sample_id"]
    raise ValueError(f"Unknown target_id: {target_id}")


def rows_to_records(rows, config):
    cleaned_root = Path(config["cleaned_data_root"])
    records = []
    missing_path_rows = []
    missing_files = []
    validate_paths = bool(config.get("input", {}).get("validate_paths", True))
    for index, row in enumerate(rows, start=2):
        relpath = select_path_value(row)
        if relpath is None:
            missing_path_rows.append(index)
            continue
        image_path = cleaned_root / relpath
        if validate_paths and not image_path.exists():
            missing_files.append(str(image_path))
        records.append(
            {"image": str(image_path), "label": LABEL_MAP[normalized_label(row)]}
        )

    if missing_path_rows:
        raise ValueError(f"Rows missing usable image path columns: {missing_path_rows[:10]}")
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} manifest image files are missing. First missing paths:\n{preview}"
        )
    return records


def validate_nonempty_splits(records_by_split):
    for split_name, records in records_by_split.items():
        if not records:
            raise ValueError(f"No records found for split '{split_name}'.")


def validate_shadow_attack_plan(plan):
    d_mem = set(plan["pools"]["D_mem"])
    d_non = set(plan["pools"]["D_non"])
    d_aux = set(plan["pools"]["D_aux"])
    shadow_pool = set(plan["shadow_pool"])
    target_ids = {target["sample_id"] for target in plan["targets"]}

    if d_mem & d_non or d_mem & d_aux or d_non & d_aux:
        raise ValueError("Shadow attack pools must be mutually exclusive.")
    if target_ids & shadow_pool:
        raise ValueError("Target samples must not overlap with Pool_shadow.")
    if not shadow_pool <= d_aux:
        raise ValueError("Pool_shadow must be a subset of D_aux.")


def select_path_value(row):
    for column in PATH_COLUMNS:
        if column in row and row[column] is not None and str(row[column]).strip():
            return str(row[column]).strip()
    return None


def normalized_label(row):
    return str(row.get("binary_label", "")).strip().lower()


def row_id(row, index):
    uid = str(row.get("uid", "")).strip()
    if uid:
        return uid
    relpath = select_path_value(row)
    if relpath:
        return relpath
    return f"row_{index}"


def label_by_id(sample_id, rows):
    for index, row in enumerate(rows):
        if row_id(row, index) == sample_id:
            return normalized_label(row)
    raise KeyError(sample_id)


def id_sort_key(sample_id, rows):
    for index, row in enumerate(rows):
        if row_id(row, index) == sample_id:
            return index
    return len(rows)


def read_csv_rows(path):
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        column for column in rows[0].keys() if column not in ATTACK_MANIFEST_DROP_COLUMNS
    ]
    for extra in [
        ATTACK_POOL_COLUMN,
        ATTACK_TARGET_ROLE_COLUMN,
        ATTACK_TARGET_ID_COLUMN,
        SHADOW_POOL_COLUMN,
    ]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: value
                    for column, value in row.items()
                    if column not in ATTACK_MANIFEST_DROP_COLUMNS
                }
            )


def read_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def resolve_output_path(value, config, default_name):
    if value:
        path = Path(value)
    else:
        path = Path(config.get("workdir", ".")) / default_name
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def describe_attack_split(config):
    rows, split_plan = ensure_shadow_attack_split(config)
    pool_labels = {
        pool_name: count_labels(
            row for row in rows if row.get(ATTACK_POOL_COLUMN) == pool_name
        )
        for pool_name in ["D_mem", "D_non", "D_aux"]
    }
    return {
        "rows": len(rows),
        "pool_counts": split_plan["pool_counts"],
        "pool_labels": pool_labels,
        "target_counts": split_plan["target_counts"],
        "shadow_pool_size": split_plan["shadow_pool_size"],
        "shadow_subset_count": split_plan["shadow_subset_count"],
        "shadow_subset_size": split_plan["shadow_subset_size"],
        "validation_count": len(split_plan["validation_ids"]),
        "test_count": len(split_plan["test_ids"]),
    }


def count_labels(rows):
    labels = Counter(normalized_label(row) for row in rows)
    return {
        "total": sum(labels.values()),
        "benign": labels.get("benign", 0),
        "malignant": labels.get("malignant", 0),
    }
