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
CONDITIONAL_DP_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "conditional_dp" / "data"
CONDITIONAL_SAMPLE_ID_COLUMN = "conditional_sample_id"
CONDITIONAL_PATIENT_ID_COLUMN = "conditional_patient_id"
CONDITIONAL_PATIENT_LABEL_COLUMN = "conditional_patient_label"
SOURCE_SPLIT_COLUMN = "source_model_split"
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
    if mode == "conditional_dp":
        return load_conditional_dp_records(config)
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


def load_conditional_dp_records(config):
    rows, split_plan = ensure_conditional_dp_split(config)
    train_rows, validation_rows, test_rows = select_conditional_dp_rows(
        rows,
        split_plan,
        config,
    )
    records_by_split = {
        "train": rows_to_records(train_rows, config),
        "validation": rows_to_records(validation_rows, config),
        "test": rows_to_records(test_rows, config),
    }
    validate_nonempty_splits(records_by_split)
    return records_by_split


def ensure_conditional_dp_split(config):
    input_config = config.get("input", {})
    manifest_path = resolve_conditional_output_path(
        input_config.get("conditional_manifest")
        or input_config.get("derived_manifest"),
        "derived_manifest.csv",
    )
    split_plan_path = resolve_conditional_output_path(
        input_config.get("split_plan"),
        "split_plan.json",
    )
    regenerate = bool(input_config.get("regenerate", False))

    if not regenerate and manifest_path.exists() and split_plan_path.exists():
        rows = read_csv_rows(manifest_path)
        split_plan = read_json(split_plan_path)
        if conditional_split_plan_matches_config(split_plan, config):
            if ensure_conditional_run_metadata(rows, split_plan, config):
                write_json(split_plan_path, split_plan)
            return rows, split_plan

    rows = read_source_manifest(config)
    row_ids = unique_row_ids(rows)
    patient_groups = build_patient_groups(rows, row_ids, config)
    split_plan = build_conditional_dp_split_plan(patient_groups, config)
    apply_conditional_dp_plan(rows, row_ids, patient_groups, split_plan)
    validate_conditional_dp_plan(rows, split_plan, config)
    write_conditional_manifest_rows(manifest_path, rows)
    write_json(split_plan_path, split_plan)
    return rows, split_plan


def resolve_conditional_output_path(value, default_name):
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    return CONDITIONAL_DP_DEFAULT_DIR / default_name


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


def unique_row_ids(rows):
    row_ids = []
    seen = {}
    duplicates = []
    for index, row in enumerate(rows):
        sample_id = row_id(row, index)
        if sample_id in seen:
            duplicates.append(
                {
                    "sample_id": sample_id,
                    "first_row": seen[sample_id] + 2,
                    "duplicate_row": index + 2,
                }
            )
        seen[sample_id] = index
        row_ids.append(sample_id)
    if duplicates:
        raise ValueError(
            "Conditional DP requires unique sample identifiers. "
            f"First duplicates: {duplicates[:10]}"
        )
    return row_ids


def conditional_patient_id_column(config):
    return str(
        config.get("conditional_dp", {}).get("patient_id_column", "empi_anon")
    ).strip()


def build_patient_groups(rows, row_ids, config):
    patient_id_column = conditional_patient_id_column(config)
    if not patient_id_column:
        raise ValueError("conditional_dp.patient_id_column must be a nonempty string.")
    if patient_id_column not in rows[0]:
        raise ValueError(
            "Conditional DP patient-level splitting requires "
            f"patient_id_column={patient_id_column!r}, but it is missing from the manifest."
        )

    groups = {}
    missing_rows = []
    for index, (row, sample_id) in enumerate(zip(rows, row_ids)):
        patient_id = str(row.get(patient_id_column, "")).strip()
        if not patient_id:
            missing_rows.append(index + 2)
            continue
        group = groups.setdefault(
            patient_id,
            {
                "patient_id": patient_id,
                "first_index": index,
                "sample_ids": [],
                "row_indices": [],
                "row_label_counts": Counter(),
            },
        )
        group["sample_ids"].append(sample_id)
        group["row_indices"].append(index)
        group["row_label_counts"][normalized_label(row)] += 1

    if missing_rows:
        raise ValueError(
            f"Rows missing conditional DP patient IDs in {patient_id_column!r}: "
            f"{missing_rows[:10]}"
        )
    if not groups:
        raise ValueError("Conditional DP split found no patient groups.")

    for group in groups.values():
        group["patient_label"] = patient_label_from_counts(group["row_label_counts"])
        group["row_count"] = len(group["row_indices"])
        group["label_counts"] = dict(group["row_label_counts"])
        del group["row_label_counts"]
    return groups


def patient_label_from_counts(label_counts):
    return "malignant" if label_counts.get("malignant", 0) else "benign"


def parse_split_fractions(config):
    configured = config.get("conditional_dp", {}).get("split_fractions", {})
    fractions = {"train": 0.70, "validation": 0.10, "test": 0.20}
    fractions.update(configured)
    required = {"train", "validation", "test"}
    missing = required - set(fractions)
    if missing:
        raise ValueError(f"conditional_dp.split_fractions missing keys: {sorted(missing)}")

    parsed = {key: float(fractions[key]) for key in ("train", "validation", "test")}
    invalid = {key: value for key, value in parsed.items() if value < 0}
    if invalid:
        raise ValueError(f"conditional_dp.split_fractions must be nonnegative: {invalid}")
    total = sum(parsed.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "conditional_dp.split_fractions must sum to 1.0; "
            f"got {total:.8f}"
        )
    if parsed["train"] <= 0:
        raise ValueError("conditional_dp.split_fractions.train must be > 0.")
    return parsed


def fractional_split_counts(total, fractions):
    split_names = ("train", "validation", "test")
    raw = {name: total * fractions[name] for name in split_names}
    counts = {name: int(raw[name]) for name in split_names}
    remaining = total - sum(counts.values())
    remainders = sorted(
        split_names,
        key=lambda name: (raw[name] - counts[name], fractions[name]),
        reverse=True,
    )
    for name in remainders[:remaining]:
        counts[name] += 1

    positive_splits = [name for name in split_names if fractions[name] > 0]
    if total >= len(positive_splits):
        for name in positive_splits:
            if counts[name] != 0:
                continue
            donors = [split for split in split_names if counts[split] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda split: (counts[split], fractions[split]))
            counts[donor] -= 1
            counts[name] += 1
    return counts


def build_conditional_dp_split_plan(patient_groups, config):
    fractions = parse_split_fractions(config)
    seed = int(config.get("conditional_dp", {}).get("split_seed", config.get("seed", 0)))
    rng = random.Random(seed)

    patient_ids_by_label = {"benign": [], "malignant": []}
    for patient_id, group in sorted(
        patient_groups.items(),
        key=lambda item: item[1]["first_index"],
    ):
        patient_ids_by_label[group["patient_label"]].append(patient_id)

    patients_by_split_label = {
        split: {"benign": [], "malignant": []}
        for split in ("train", "validation", "test")
    }
    for label, patient_ids in patient_ids_by_label.items():
        shuffled = list(patient_ids)
        rng.shuffle(shuffled)
        counts = fractional_split_counts(len(shuffled), fractions)
        offset = 0
        for split in ("train", "validation", "test"):
            count = counts[split]
            patients_by_split_label[split][label] = shuffled[offset: offset + count]
            offset += count

    patient_splits = {
        split: (
            patients_by_split_label[split]["benign"]
            + patients_by_split_label[split]["malignant"]
        )
        for split in ("train", "validation", "test")
    }
    patient_labels = {
        patient_id: group["patient_label"]
        for patient_id, group in patient_groups.items()
    }

    plan = {
        "mode": "conditional_dp",
        "seed": seed,
        "source_manifest": str(Path(config["data_csv"])),
        "patient_id_column": conditional_patient_id_column(config),
        "split_fractions": fractions,
        "selection_unit": "rows",
        "selection_policy": "whole_patients_meet_or_exceed_row_targets",
        "patients": patients_by_split_label,
        "patient_splits": patient_splits,
        "patient_labels": patient_labels,
        "patient_counts": summarize_patient_splits(patient_splits, patient_labels),
        "row_counts": summarize_patient_split_rows(patient_splits, patient_groups),
        "available_train_patients": {
            "benign": len(patients_by_split_label["train"]["benign"]),
            "malignant": len(patients_by_split_label["train"]["malignant"]),
        },
    }
    plan["available_train_rows"] = dict(plan["row_counts"]["train"])
    plan["available_train_rows"].pop("total", None)
    plan["run_selections"] = build_conditional_run_selections(
        plan,
        config,
        patient_groups=patient_groups,
    )
    return plan


def summarize_patient_splits(patient_splits, patient_labels):
    summary = {}
    for split, patient_ids in patient_splits.items():
        labels = Counter(patient_labels[patient_id] for patient_id in patient_ids)
        summary[split] = {
            "total": len(patient_ids),
            "benign": labels.get("benign", 0),
            "malignant": labels.get("malignant", 0),
        }
    return summary


def summarize_patient_split_rows(patient_splits, patient_groups):
    summary = {}
    for split, patient_ids in patient_splits.items():
        label_counts = Counter()
        total = 0
        for patient_id in patient_ids:
            group = patient_groups[patient_id]
            total += group["row_count"]
            label_counts.update(group["label_counts"])
        summary[split] = {
            "total": total,
            "benign": label_counts.get("benign", 0),
            "malignant": label_counts.get("malignant", 0),
        }
    return summary


def conditional_split_plan_matches_config(split_plan, config):
    expected = {
        "mode": "conditional_dp",
        "source_manifest": str(Path(config["data_csv"])),
        "seed": int(config.get("conditional_dp", {}).get("split_seed", config.get("seed", 0))),
        "patient_id_column": conditional_patient_id_column(config),
        "split_fractions": parse_split_fractions(config),
        "selection_unit": "rows",
        "selection_policy": "whole_patients_meet_or_exceed_row_targets",
    }
    return all(split_plan.get(key) == value for key, value in expected.items())


def ensure_conditional_run_metadata(rows, split_plan, config):
    split_plan["selection_unit"] = "rows"
    split_plan["selection_policy"] = "whole_patients_meet_or_exceed_row_targets"
    split_plan["available_train_rows"] = available_train_row_counts(split_plan, rows=rows)
    run_selections = build_conditional_run_selections(split_plan, config, rows=rows)
    if split_plan.get("run_selections") == run_selections:
        return False
    split_plan["run_selections"] = run_selections
    return True


def build_conditional_run_selections(
    split_plan,
    config,
    rows=None,
    patient_groups=None,
):
    train_patients = split_plan["patients"]["train"]
    row_counts_by_patient = conditional_train_patient_row_counts(
        split_plan,
        rows=rows,
        patient_groups=patient_groups,
    )

    selections = {
        "full_train": build_conditional_run_selection(
            train_patients,
            row_counts_by_patient,
            n_plus=None,
            rho=None,
        )
    }
    for n_plus, rho in configured_conditional_run_pairs(config):
        selections[conditional_run_key(n_plus, rho)] = build_conditional_run_selection(
            train_patients,
            row_counts_by_patient,
            n_plus=n_plus,
            rho=rho,
        )
    return selections


def conditional_train_patient_row_counts(split_plan, rows=None, patient_groups=None):
    train_patients = split_plan["patients"]["train"]
    train_patient_ids = set(train_patients["benign"] + train_patients["malignant"])

    if patient_groups is not None:
        return {
            patient_id: dict(patient_groups[patient_id]["label_counts"])
            for patient_id in train_patient_ids
        }

    counts = {patient_id: Counter() for patient_id in train_patient_ids}
    patient_id_column = split_plan["patient_id_column"]
    for row in rows or []:
        if str(row.get("model_split", "")).strip() != "train":
            continue
        patient_id = row_patient_id(row, patient_id_column)
        if patient_id in counts:
            counts[patient_id][normalized_label(row)] += 1
    return {patient_id: dict(label_counts) for patient_id, label_counts in counts.items()}


def available_train_row_counts(split_plan, rows=None, patient_groups=None):
    train_patients = split_plan["patients"]["train"]
    row_counts_by_patient = conditional_train_patient_row_counts(
        split_plan,
        rows=rows,
        patient_groups=patient_groups,
    )
    counts = Counter()
    for patient_id in train_patients["benign"] + train_patients["malignant"]:
        counts.update(row_counts_by_patient.get(patient_id, {}))
    return {
        "benign": counts.get("benign", 0),
        "malignant": counts.get("malignant", 0),
    }


def build_conditional_run_selection(
    train_patients,
    row_counts_by_patient,
    n_plus,
    rho,
):
    malignant_patients = list(train_patients["malignant"])
    benign_patients = list(train_patients["benign"])
    available_patients = {
        "benign": len(benign_patients),
        "malignant": len(malignant_patients),
    }
    available_row_counter = Counter()
    for patient_id in benign_patients + malignant_patients:
        available_row_counter.update(row_counts_by_patient.get(patient_id, {}))
    available_rows = {
        "benign": available_row_counter.get("benign", 0),
        "malignant": available_row_counter.get("malignant", 0),
    }

    if n_plus is None:
        resolved_n_plus = available_rows["malignant"]
        requested_n_minus = available_rows["benign"]
        resolved_n_minus = available_rows["benign"]
        selected = {
            "benign": benign_patients,
            "malignant": malignant_patients,
        }
        return make_available_conditional_selection(
            resolved_n_plus,
            rho,
            requested_n_minus,
            resolved_n_minus,
            available_patients,
            available_rows,
            capped_n_minus=False,
            selected=selected,
            row_counts_by_patient=row_counts_by_patient,
        )

    if n_plus > available_rows["malignant"]:
        requested_n_minus = int(round(n_plus * rho))
        return make_unavailable_conditional_selection(
            n_plus,
            rho,
            requested_n_minus,
            available_patients,
            available_rows,
            reason=(
                f"Requested N_plus={n_plus} exceeds available malignant training "
                f"rows={available_rows['malignant']}."
            ),
        )

    selected_malignant = select_patients_to_meet_row_target(
        malignant_patients,
        row_counts_by_patient,
        "malignant",
        n_plus,
    )
    requested_n_minus = int(round(n_plus * rho))
    feasible_n_minus = sum(
        label_row_count(row_counts_by_patient, patient_id, "benign")
        for patient_id in benign_patients + selected_malignant
    )
    resolved_n_minus = min(requested_n_minus, feasible_n_minus)
    if resolved_n_minus <= 0:
        return make_unavailable_conditional_selection(
            n_plus,
            rho,
            requested_n_minus,
            available_patients,
            available_rows,
            reason="Conditional DP subsampling selected no benign training patients.",
        )

    selected = {
        "benign": select_patients_to_meet_row_target(
            benign_patients,
            row_counts_by_patient,
            "benign",
            resolved_n_minus,
            initial_rows=sum(
                label_row_count(row_counts_by_patient, patient_id, "benign")
                for patient_id in selected_malignant
            ),
        ),
        "malignant": selected_malignant,
    }
    return make_available_conditional_selection(
        n_plus,
        rho,
        requested_n_minus,
        resolved_n_minus,
        available_patients,
        available_rows,
        capped_n_minus=requested_n_minus > feasible_n_minus,
        selected=selected,
        row_counts_by_patient=row_counts_by_patient,
    )


def select_patients_to_meet_row_target(
    patient_ids,
    row_counts_by_patient,
    label,
    target_rows,
    initial_rows=0,
):
    selected = []
    selected_rows = initial_rows
    for patient_id in patient_ids:
        if selected_rows >= target_rows:
            break
        selected.append(patient_id)
        selected_rows += label_row_count(row_counts_by_patient, patient_id, label)
    return selected


def label_row_count(row_counts_by_patient, patient_id, label):
    return int(row_counts_by_patient.get(patient_id, {}).get(label, 0))


def make_available_conditional_selection(
    n_plus,
    rho,
    requested_n_minus,
    n_minus,
    available_patients,
    available_rows,
    capped_n_minus,
    selected,
    row_counts_by_patient,
):
    row_counts = selected_row_counts(selected, row_counts_by_patient)
    return {
        "status": "available",
        "count_unit": "rows",
        "selection_policy": "whole_patients_meet_or_exceed_row_targets",
        "requested_N_plus": n_plus,
        "N_plus": n_plus,
        "actual_N_plus": row_counts["malignant"],
        "rho": json_number(rho) if rho is not None else None,
        "requested_N_minus": requested_n_minus,
        "resolved_N_minus": n_minus,
        "N_minus": n_minus,
        "actual_N_minus": row_counts["benign"],
        "available_train_patients": dict(available_patients),
        "available_train_rows": dict(available_rows),
        "capped_N_minus": capped_n_minus,
        "selected_patients": selected,
        "selected_patient_counts": selected_patient_counts(selected),
        "selected_row_counts": row_counts,
    }


def make_unavailable_conditional_selection(
    n_plus,
    rho,
    requested_n_minus,
    available_patients,
    available_rows,
    reason,
):
    selected = {"benign": [], "malignant": []}
    return {
        "status": "unavailable",
        "reason": reason,
        "count_unit": "rows",
        "selection_policy": "whole_patients_meet_or_exceed_row_targets",
        "requested_N_plus": n_plus,
        "N_plus": n_plus,
        "actual_N_plus": 0,
        "rho": json_number(rho),
        "requested_N_minus": requested_n_minus,
        "resolved_N_minus": 0,
        "N_minus": 0,
        "actual_N_minus": 0,
        "available_train_patients": dict(available_patients),
        "available_train_rows": dict(available_rows),
        "capped_N_minus": requested_n_minus > available_rows["benign"],
        "selected_patients": selected,
        "selected_patient_counts": selected_patient_counts(selected),
        "selected_row_counts": {"total": 0, "benign": 0, "malignant": 0},
    }


def selected_patient_counts(selected):
    benign = len(selected["benign"])
    malignant = len(selected["malignant"])
    return {
        "total": benign + malignant,
        "benign": benign,
        "malignant": malignant,
    }


def selected_row_counts(selected, row_counts_by_patient):
    labels = Counter()
    for patient_id in selected["benign"] + selected["malignant"]:
        labels.update(row_counts_by_patient.get(patient_id, {}))
    return {
        "total": sum(labels.values()),
        "benign": labels.get("benign", 0),
        "malignant": labels.get("malignant", 0),
    }


def configured_conditional_run_pairs(config):
    n_plus_values = conditional_axis_values(
        config,
        ("N_plus_values", "n_plus_values", "N_plus", "n_plus"),
        parse_positive_int,
        "N_plus",
    )
    rho_values = conditional_axis_values(
        config,
        ("rho_values", "rho"),
        parse_positive_float,
        "rho",
    )
    pairs = []
    for n_plus in n_plus_values:
        for rho in rho_values:
            pairs.append((n_plus, rho))

    conditional_config = config.get("conditional_dp", {})
    if any(
        key in conditional_config
        for key in ("active_N_plus", "active_n_plus", "active_rho")
    ):
        active = active_conditional_subset(config)
        pairs.append((active["N_plus"], active["rho"]))
    return unique_conditional_pairs(pairs)


def conditional_axis_values(config, keys, parser, label):
    conditional_config = config.get("conditional_dp", {})
    values = []
    for key in keys:
        if key not in conditional_config:
            continue
        raw_values = conditional_config[key]
        if isinstance(raw_values, (list, tuple)):
            candidates = raw_values
        else:
            candidates = [raw_values]
        for value in candidates:
            values.append(parser(value, f"conditional_dp.{key}", label))
    return unique_values(values)


def parse_positive_int(value, key, label):
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{key} {label} values must be integers.")
    parsed = int(number)
    if parsed <= 0:
        raise ValueError(f"{key} {label} values must be > 0.")
    return parsed


def parse_positive_float(value, key, label):
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{key} {label} values must be > 0.")
    return parsed


def unique_values(values):
    unique = []
    seen = set()
    for value in values:
        key = json_number(value)
        if key in seen:
            continue
        unique.append(value)
        seen.add(key)
    return unique


def unique_conditional_pairs(pairs):
    unique = []
    seen = set()
    for n_plus, rho in pairs:
        key = (n_plus, json_number(rho))
        if key in seen:
            continue
        unique.append((n_plus, rho))
        seen.add(key)
    return unique


def conditional_run_key(n_plus, rho):
    return f"N_plus_{n_plus}_rho_{axis_key_value(rho)}"


def axis_key_value(value):
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace("-", "m").replace(".", "p")


def json_number(value):
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def apply_conditional_dp_plan(rows, row_ids, patient_groups, split_plan):
    patient_id_column = split_plan["patient_id_column"]
    split_by_patient = {}
    for split, patient_ids in split_plan["patient_splits"].items():
        for patient_id in patient_ids:
            split_by_patient[patient_id] = split

    for row, sample_id in zip(rows, row_ids):
        patient_id = str(row.get(patient_id_column, "")).strip()
        if SOURCE_SPLIT_COLUMN not in row:
            row[SOURCE_SPLIT_COLUMN] = row.get("model_split", "")
        row["model_split"] = split_by_patient[patient_id]
        row[CONDITIONAL_SAMPLE_ID_COLUMN] = sample_id
        row[CONDITIONAL_PATIENT_ID_COLUMN] = patient_id
        row[CONDITIONAL_PATIENT_LABEL_COLUMN] = patient_groups[patient_id]["patient_label"]


def validate_conditional_dp_plan(rows, split_plan, config):
    split_sets = {
        split: set(patient_ids)
        for split, patient_ids in split_plan["patient_splits"].items()
    }
    if (
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    ):
        raise ValueError("Conditional DP patient splits must be mutually exclusive.")

    all_split_patients = set().union(*split_sets.values())
    if set(split_plan["patient_labels"]) != all_split_patients:
        raise ValueError("Conditional DP patient_labels do not match patient_splits.")

    row_counts = {
        split: count_labels(row for row in rows if row.get("model_split") == split)
        for split in ("train", "validation", "test")
    }
    empty_splits = [
        split for split, counts in row_counts.items()
        if counts["total"] == 0
    ]
    if empty_splits:
        raise ValueError(f"Conditional DP split produced empty splits: {empty_splits}")
    if row_counts["train"]["benign"] == 0 or row_counts["train"]["malignant"] == 0:
        raise ValueError(
            "Conditional DP training split must include both benign and malignant rows."
        )

    unavailable_without_reason = [
        key
        for key, selection in split_plan.get("run_selections", {}).items()
        if selection.get("status") == "unavailable" and not selection.get("reason")
    ]
    if unavailable_without_reason:
        raise ValueError(
            "Conditional DP run selections are unavailable without reasons: "
            f"{unavailable_without_reason}"
        )


def conditional_axis_scalar(config, primary_key, active_key):
    conditional_config = config.get("conditional_dp", {})
    if active_key in conditional_config:
        return conditional_config[active_key]
    value = conditional_config.get(primary_key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return value[0]
        raise ValueError(
            f"conditional_dp.{primary_key} contains multiple values. "
            f"Set conditional_dp.{active_key} for a single training run."
        )
    return value


def active_conditional_subset(config):
    n_plus = conditional_axis_scalar(config, "N_plus", "active_N_plus")
    if n_plus is None:
        n_plus = conditional_axis_scalar(config, "n_plus", "active_n_plus")
    rho = conditional_axis_scalar(config, "rho", "active_rho")
    if n_plus is None and rho is None:
        return None
    if n_plus is None or rho is None:
        raise ValueError(
            "Conditional DP subsampling requires both N_plus and rho, or neither."
        )
    n_plus = int(n_plus)
    rho = float(rho)
    if n_plus <= 0:
        raise ValueError("conditional_dp.N_plus must be > 0.")
    if rho <= 0:
        raise ValueError("conditional_dp.rho must be > 0.")
    return {"N_plus": n_plus, "rho": rho}


def conditional_train_patient_ids(split_plan, config):
    subset = active_conditional_subset(config)
    train_patients = split_plan["patients"]["train"]
    if subset is None:
        full_train = split_plan.get("run_selections", {}).get("full_train")
        if full_train:
            return selected_patient_id_set(full_train)
        return set(train_patients["benign"] + train_patients["malignant"])

    run_key = conditional_run_key(subset["N_plus"], subset["rho"])
    selection = split_plan.get("run_selections", {}).get(run_key)
    if selection is None:
        raise ValueError(
            "Conditional DP split plan is missing run selection "
            f"{run_key!r}. Regenerate or reload the split plan."
        )
    if selection.get("status") != "available":
        raise ValueError(selection.get("reason", f"Conditional DP run {run_key} is unavailable."))
    return selected_patient_id_set(selection)


def selected_patient_id_set(selection):
    selected = selection["selected_patients"]
    return set(selected["benign"] + selected["malignant"])


def select_conditional_dp_rows(rows, split_plan, config):
    patient_id_column = split_plan["patient_id_column"]
    selected_train_patients = conditional_train_patient_ids(split_plan, config)
    selected = {"train": [], "validation": [], "test": []}

    for row in rows:
        split = str(row.get("model_split", "")).strip()
        if split not in selected:
            continue
        patient_id = row_patient_id(row, patient_id_column)
        if split == "train" and patient_id not in selected_train_patients:
            continue
        selected[split].append(row)

    return selected["train"], selected["validation"], selected["test"]


def row_patient_id(row, patient_id_column):
    patient_id = str(row.get(patient_id_column, "")).strip()
    if patient_id:
        return patient_id
    return str(row.get(CONDITIONAL_PATIENT_ID_COLUMN, "")).strip()


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


def write_conditional_manifest_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for column in row:
            if column not in seen:
                fieldnames.append(column)
                seen.add(column)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
