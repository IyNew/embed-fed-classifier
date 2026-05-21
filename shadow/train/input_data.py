from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from centralized.input_data import (  # noqa: F401,E402
    ATTACK_MANIFEST_DROP_COLUMNS,
    ATTACK_POOL_COLUMN,
    ATTACK_TARGET_ID_COLUMN,
    ATTACK_TARGET_ROLE_COLUMN,
    DEFAULT_POOL_COUNTS,
    DEFAULT_TARGET_COUNTS,
    REQUIRED_MANIFEST_COLUMNS,
    SHADOW_POOL_COLUMN,
    apply_shadow_attack_plan,
    build_shadow_attack_split_plan,
    count_labels,
    describe_attack_split,
    ensure_shadow_attack_split,
    id_sort_key,
    label_by_id,
    load_records,
    load_shadow_attack_records,
    make_targets,
    normalized_label,
    proportional_allocations,
    read_csv_rows,
    read_json,
    read_source_manifest,
    resolve_output_path,
    row_id,
    rows_to_records,
    select_path_value,
    select_shadow_attack_rows,
    shadow_train_ids,
    split_eval_ids,
    stratified_partition,
    stratified_take,
    summarize_records,
    target_sample_id_by_target_id,
    validate_nonempty_splits,
    validate_shadow_attack_plan,
    write_csv_rows,
    write_json,
)
