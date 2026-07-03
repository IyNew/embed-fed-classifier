"""
Evaluate LiRA membership-inference scores for one or more loss CSVs.

This script packages the analysis logic from shadow/lira_eval.ipynb into a
reproducible command-line report. It expects each target sample to have one
target-model loss, 16 IN shadow-model losses, and 16 OUT shadow-model losses by
default.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_FPR_TARGETS = (0.001, 0.01, 0.05, 0.1)
DEFAULT_OUTLIER_THRESHOLD = 50.0
REQUIRED_COLUMNS = {
    "target_id",
    "membership",
    "model_category",
    "model_index",
    "loss",
    "label",
}
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def fpr_value(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed >= 1:
        raise argparse.ArgumentTypeError("FPR values must be between 0 and 1")
    return parsed


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--run must be formatted as NAME=CSV_PATH")
    name, path_text = spec.split("=", 1)
    name = name.strip()
    path_text = path_text.strip()
    if not name or not RUN_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "run names must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens"
        )
    if not path_text:
        raise argparse.ArgumentTypeError("--run CSV_PATH cannot be empty")
    return name, Path(path_text)


def natural_target_key(value: str) -> tuple[str, int | str]:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        return str(value), str(value)
    return str(value)[: match.start()], int(match.group(1))


def finite_exp(value: float) -> float:
    if np.isposinf(value):
        return float("inf")
    if np.isneginf(value):
        return 0.0
    if np.isnan(value):
        return float("nan")
    if value > math.log(np.finfo(float).max):
        return float("inf")
    if value < math.log(np.nextafter(0, 1)):
        return 0.0
    return float(math.exp(value))


def format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, float) and np.isposinf(value):
        return "inf"
    if isinstance(value, float) and np.isneginf(value):
        return "-inf"
    return f"{float(value):.{digits}f}"


def format_percent(value: float | int | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def validate_loss_csv(
    df: pd.DataFrame,
    *,
    run_name: str,
    csv_path: Path,
    expected_targets: int,
    expected_shadow_count: int,
) -> dict[str, object]:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{run_name}: missing required columns: {missing}")

    df["target_id"] = df["target_id"].astype(str).str.strip()
    df["membership"] = df["membership"].astype(str).str.strip().str.lower()
    df["model_category"] = df["model_category"].astype(str).str.strip().str.lower()
    df["model_index"] = df["model_index"].fillna("").astype(str).str.strip()
    df["loss"] = pd.to_numeric(df["loss"], errors="raise")

    expected_rows = expected_targets * (1 + 2 * expected_shadow_count)
    category_counts = df["model_category"].value_counts().to_dict()
    target_count = df["target_id"].nunique()
    errors = []

    if len(df) != expected_rows:
        errors.append(
            f"expected {expected_rows} rows, found {len(df)} in {csv_path}"
        )
    if target_count != expected_targets:
        errors.append(f"expected {expected_targets} targets, found {target_count}")

    expected_category_counts = {
        "target": expected_targets,
        "in": expected_targets * expected_shadow_count,
        "out": expected_targets * expected_shadow_count,
    }
    for category, expected in expected_category_counts.items():
        observed = int(category_counts.get(category, 0))
        if observed != expected:
            errors.append(
                f"expected {expected} {category!r} rows, found {observed}"
            )

    unexpected_categories = sorted(
        set(df["model_category"].dropna()) - set(expected_category_counts)
    )
    if unexpected_categories:
        errors.append(f"unexpected model_category values: {unexpected_categories}")

    key_frame = df[["target_id", "model_category", "model_index"]].copy()
    duplicates = key_frame.duplicated(keep=False)
    duplicate_count = int(duplicates.sum())
    if duplicates.any():
        duplicate_preview = key_frame.loc[duplicates].head(10).to_dict("records")
        errors.append(
            "duplicate (target_id, model_category, model_index) keys, "
            f"first duplicates: {duplicate_preview}"
        )

    for target_id in sorted(df["target_id"].unique(), key=natural_target_key):
        sub = df[df["target_id"] == target_id]
        membership_values = sorted(sub["membership"].dropna().unique())
        if len(membership_values) != 1:
            errors.append(
                f"{target_id}: expected one membership value, found {membership_values}"
            )

        label_values = sorted(str(v) for v in sub["label"].dropna().unique())
        if len(label_values) != 1:
            errors.append(f"{target_id}: expected one label value, found {label_values}")

        target_rows = int((sub["model_category"] == "target").sum())
        in_rows = int((sub["model_category"] == "in").sum())
        out_rows = int((sub["model_category"] == "out").sum())
        if target_rows != 1 or in_rows != expected_shadow_count or out_rows != expected_shadow_count:
            errors.append(
                f"{target_id}: expected target=1, in={expected_shadow_count}, "
                f"out={expected_shadow_count}; found target={target_rows}, "
                f"in={in_rows}, out={out_rows}"
            )

        if sub["loss"].isna().any():
            errors.append(f"{target_id}: loss column contains NaN")

    membership_counts = df[df["model_category"] == "target"][
        "membership"
    ].value_counts().to_dict()
    if set(membership_counts) != {"member", "nonmember"}:
        errors.append(
            "target rows must include both 'member' and 'nonmember' memberships; "
            f"found {membership_counts}"
        )

    if errors:
        preview = "\n  - ".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
        raise ValueError(f"{run_name}: validation failed:\n  - {preview}{suffix}")

    return {
        "run_name": run_name,
        "csv_path": str(csv_path),
        "rows": len(df),
        "targets": target_count,
        "target_rows": int(category_counts.get("target", 0)),
        "in_rows": int(category_counts.get("in", 0)),
        "out_rows": int(category_counts.get("out", 0)),
        "member_targets": int(membership_counts.get("member", 0)),
        "nonmember_targets": int(membership_counts.get("nonmember", 0)),
        "duplicate_keys": duplicate_count,
    }


def fit_gaussian(losses: np.ndarray) -> tuple[float, float]:
    mu, std = norm.fit(losses)
    return float(mu), max(float(std), 1e-6)


def list_to_text(values: Iterable[float]) -> str:
    return ";".join(f"{float(value):.10g}" for value in values)


def compute_lira_scores(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    records = []
    target_ids = sorted(df["target_id"].unique(), key=natural_target_key)

    for target_id in target_ids:
        sub = df[df["target_id"] == target_id]
        target_row = sub[sub["model_category"] == "target"].iloc[0]
        losses_in = sub[sub["model_category"] == "in"]["loss"].to_numpy(dtype=float)
        losses_out = sub[sub["model_category"] == "out"]["loss"].to_numpy(dtype=float)
        loss_target = float(target_row["loss"])

        mu_in, std_in = fit_gaussian(losses_in)
        mu_out, std_out = fit_gaussian(losses_out)
        logpdf_in = float(norm.logpdf(loss_target, loc=mu_in, scale=std_in))
        logpdf_out = float(norm.logpdf(loss_target, loc=mu_out, scale=std_out))
        log_lambda = logpdf_in - logpdf_out

        membership = str(target_row["membership"]).lower()
        label = target_row["label"]
        sample_id = target_row.get("sample_id", "")
        correct = target_row.get("correct", "")

        records.append(
            {
                "run_name": run_name,
                "target_id": target_id,
                "sample_id": sample_id,
                "membership": membership,
                "true_member": int(membership == "member"),
                "label": label,
                "target_correct": correct,
                "loss_target": loss_target,
                "loss_in_mean": float(np.mean(losses_in)),
                "loss_in_std": float(np.std(losses_in, ddof=1)),
                "loss_out_mean": float(np.mean(losses_out)),
                "loss_out_std": float(np.std(losses_out, ddof=1)),
                "mu_in": mu_in,
                "std_in": std_in,
                "mu_out": mu_out,
                "std_out": std_out,
                "logpdf_in": logpdf_in,
                "logpdf_out": logpdf_out,
                "lira_log_lambda": float(log_lambda),
                "lira_lambda": finite_exp(float(log_lambda)),
                "loss_delta": float(np.mean(losses_in) - np.mean(losses_out)),
                "in_count": int(len(losses_in)),
                "out_count": int(len(losses_out)),
                "losses_in": list_to_text(losses_in),
                "losses_out": list_to_text(losses_out),
            }
        )

    return pd.DataFrame(records)


def scores_by_membership(frame: pd.DataFrame, score_col: str) -> tuple[np.ndarray, np.ndarray]:
    member_scores = frame.loc[frame["membership"] == "member", score_col].to_numpy(
        dtype=float
    )
    nonmember_scores = frame.loc[
        frame["membership"] == "nonmember", score_col
    ].to_numpy(dtype=float)
    if len(member_scores) == 0 or len(nonmember_scores) == 0:
        raise ValueError("metrics require at least one member and one nonmember target")
    return member_scores, nonmember_scores


def finite_scores_for_metrics(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).copy()
    if np.all(np.isfinite(scores)):
        return scores

    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        raise ValueError("all metric scores are non-finite")

    finite_min = float(np.min(finite_scores))
    finite_max = float(np.max(finite_scores))
    padding = max(1.0, abs(finite_max - finite_min) * 0.1)
    scores[np.isposinf(scores)] = finite_max + padding
    scores[np.isneginf(scores)] = finite_min - padding
    return scores


def auc_from_member_scores(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    y_true = np.array([1] * len(member_scores) + [0] * len(nonmember_scores))
    y_score = finite_scores_for_metrics(np.concatenate([member_scores, nonmember_scores]))
    return float(roc_auc_score(y_true, y_score))


def compute_tpr_at_fpr(
    member_scores: np.ndarray, nonmember_scores: np.ndarray, fpr_target: float
) -> tuple[float, float, float]:
    threshold = float(np.percentile(nonmember_scores, 100 * (1 - fpr_target)))
    tpr = float(np.mean(member_scores > threshold))
    empirical_fpr = float(np.mean(nonmember_scores > threshold))
    return tpr, threshold, empirical_fpr


def score_summary(prefix: str, scores: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(scores)),
        f"{prefix}_median": float(np.median(scores)),
        f"{prefix}_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        f"{prefix}_min": float(np.min(scores)),
        f"{prefix}_max": float(np.max(scores)),
    }


def summarize_run(
    per_target: pd.DataFrame,
    *,
    run_name: str,
    validation: dict[str, object],
    fpr_targets: tuple[float, ...],
    outlier_threshold: float,
) -> dict[str, object]:
    member_log_scores, nonmember_log_scores = scores_by_membership(
        per_target, "lira_log_lambda"
    )
    member_lambda_scores, nonmember_lambda_scores = scores_by_membership(
        per_target, "lira_lambda"
    )
    member_loss_scores, nonmember_loss_scores = scores_by_membership(
        per_target.assign(raw_loss_member_score=-per_target["loss_target"]),
        "raw_loss_member_score",
    )

    summary: dict[str, object] = {
        **validation,
        "lira_auc": auc_from_member_scores(member_log_scores, nonmember_log_scores),
        "raw_loss_auc": auc_from_member_scores(member_loss_scores, nonmember_loss_scores),
        "outlier_threshold_lambda": outlier_threshold,
    }
    summary.update(score_summary("member_lira_log_lambda", member_log_scores))
    summary.update(score_summary("nonmember_lira_log_lambda", nonmember_log_scores))
    summary.update(score_summary("member_lira_lambda", member_lambda_scores))
    summary.update(score_summary("nonmember_lira_lambda", nonmember_lambda_scores))
    summary.update(score_summary("member_target_loss", -member_loss_scores))
    summary.update(score_summary("nonmember_target_loss", -nonmember_loss_scores))

    for fpr_target in fpr_targets:
        label = f"{fpr_target:g}".replace(".", "p")
        tpr, threshold_log, empirical_fpr = compute_tpr_at_fpr(
            member_log_scores, nonmember_log_scores, fpr_target
        )
        loss_tpr, loss_threshold_score, loss_empirical_fpr = compute_tpr_at_fpr(
            member_loss_scores, nonmember_loss_scores, fpr_target
        )
        summary[f"lira_tpr_at_fpr_{label}"] = tpr
        summary[f"lira_empirical_fpr_at_fpr_{label}"] = empirical_fpr
        summary[f"lira_log_threshold_at_fpr_{label}"] = threshold_log
        summary[f"lira_lambda_threshold_at_fpr_{label}"] = finite_exp(threshold_log)
        summary[f"raw_loss_tpr_at_fpr_{label}"] = loss_tpr
        summary[f"raw_loss_empirical_fpr_at_fpr_{label}"] = loss_empirical_fpr
        summary[f"raw_loss_threshold_at_fpr_{label}"] = -loss_threshold_score

    outlier_log_threshold = math.log(outlier_threshold)
    clean = per_target[per_target["lira_log_lambda"] <= outlier_log_threshold].copy()
    outliers = per_target[per_target["lira_log_lambda"] > outlier_log_threshold]
    summary["outlier_excluded_count"] = int(len(outliers))
    summary["outlier_excluded_member_count"] = int(
        (outliers["membership"] == "member").sum()
    )
    summary["outlier_excluded_nonmember_count"] = int(
        (outliers["membership"] == "nonmember").sum()
    )

    clean_member_scores, clean_nonmember_scores = scores_by_membership(
        clean, "lira_log_lambda"
    )
    summary["lira_auc_outlier_excluded"] = auc_from_member_scores(
        clean_member_scores, clean_nonmember_scores
    )
    summary["outlier_excluded_targets"] = int(len(clean))
    summary["outlier_excluded_member_targets"] = int(len(clean_member_scores))
    summary["outlier_excluded_nonmember_targets"] = int(len(clean_nonmember_scores))
    for fpr_target in fpr_targets:
        label = f"{fpr_target:g}".replace(".", "p")
        tpr, threshold_log, empirical_fpr = compute_tpr_at_fpr(
            clean_member_scores, clean_nonmember_scores, fpr_target
        )
        summary[f"lira_tpr_at_fpr_{label}_outlier_excluded"] = tpr
        summary[f"lira_empirical_fpr_at_fpr_{label}_outlier_excluded"] = empirical_fpr
        summary[f"lira_log_threshold_at_fpr_{label}_outlier_excluded"] = threshold_log
        summary[f"lira_lambda_threshold_at_fpr_{label}_outlier_excluded"] = finite_exp(
            threshold_log
        )

    return summary


def get_y_true_scores(per_target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    member_scores, nonmember_scores = scores_by_membership(per_target, "lira_log_lambda")
    y_true = np.array([1] * len(member_scores) + [0] * len(nonmember_scores))
    y_score = finite_scores_for_metrics(np.concatenate([member_scores, nonmember_scores]))
    return y_true, y_score


def save_roc_curve(
    per_target: pd.DataFrame,
    *,
    run_name: str,
    auc: float,
    fpr_targets: tuple[float, ...],
    output_path: Path,
) -> None:
    y_true, y_score = get_y_true_scores(per_target)
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax in axes:
        ax.plot(fpr_vals, tpr_vals, color="#255f85", linewidth=1.6)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{run_name}: ROC (AUC = {auc:.4f})")

    axes[1].set_xscale("log")
    axes[1].set_xlim(min(fpr_targets), 1)
    axes[1].set_title("Low-FPR Region")

    for fpr_target in fpr_targets:
        idx = min(np.searchsorted(fpr_vals, fpr_target), len(fpr_vals) - 1)
        for ax in axes:
            ax.scatter(fpr_vals[idx], tpr_vals[idx], color="#b6383a", zorder=5, s=28)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_score_distribution(
    per_target: pd.DataFrame,
    *,
    run_name: str,
    auc: float,
    tpr_at_1pct: float | None,
    threshold_log_at_1pct: float | None,
    output_path: Path,
) -> None:
    member_log_scores, nonmember_log_scores = scores_by_membership(
        per_target, "lira_log_lambda"
    )
    member_lambda_scores, nonmember_lambda_scores = scores_by_membership(
        per_target, "lira_lambda"
    )

    finite_member_lambda = member_lambda_scores[np.isfinite(member_lambda_scores)]
    finite_nonmember_lambda = nonmember_lambda_scores[np.isfinite(nonmember_lambda_scores)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = min(20, max(8, len(per_target) // 3))

    axes[0].hist(
        finite_nonmember_lambda,
        bins=bins,
        alpha=0.6,
        label="Non-members",
        density=True,
        color="#3e75a1",
        edgecolor="white",
    )
    axes[0].hist(
        finite_member_lambda,
        bins=bins,
        alpha=0.6,
        label="Members",
        density=True,
        color="#d36c4e",
        edgecolor="white",
    )
    axes[0].set_xlabel("LiRA score lambda")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution")

    axes[1].hist(
        nonmember_log_scores,
        bins=bins,
        alpha=0.6,
        label="Non-members",
        density=True,
        color="#3e75a1",
        edgecolor="white",
    )
    axes[1].hist(
        member_log_scores,
        bins=bins,
        alpha=0.6,
        label="Members",
        density=True,
        color="#d36c4e",
        edgecolor="white",
    )
    axes[1].set_xlabel("log(lambda)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Log Score Distribution")

    if threshold_log_at_1pct is not None:
        threshold_lambda = finite_exp(threshold_log_at_1pct)
        if np.isfinite(threshold_lambda):
            axes[0].axvline(
                threshold_lambda,
                color="#b6383a",
                linestyle="--",
                linewidth=1.4,
                label="1% FPR threshold",
            )
        axes[1].axvline(
            threshold_log_at_1pct,
            color="#b6383a",
            linestyle="--",
            linewidth=1.4,
            label="1% FPR threshold",
        )

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    title = f"{run_name}: LiRA scores (AUC={auc:.3f}"
    if tpr_at_1pct is not None:
        title += f", TPR@1%FPR={tpr_at_1pct:.3f}"
    title += ")"
    fig.suptitle(title, fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_ranked_scores(
    per_target: pd.DataFrame,
    *,
    run_name: str,
    output_path: Path,
) -> None:
    sorted_df = per_target.sort_values("lira_log_lambda", ascending=False)
    colors = [
        "#d36c4e" if membership == "member" else "#3e75a1"
        for membership in sorted_df["membership"]
    ]

    fig_width = max(12, len(sorted_df) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(range(len(sorted_df)), sorted_df["lira_log_lambda"], color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df["target_id"], rotation=90, fontsize=6)
    ax.set_ylabel("log(lambda)")
    ax.set_title(f"{run_name}: LiRA Scores Ranked")
    ax.legend(
        handles=[
            Patch(facecolor="#d36c4e", label="Member"),
            Patch(facecolor="#3e75a1", label="Non-member"),
        ],
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_loss_comparison(
    per_target: pd.DataFrame,
    *,
    run_name: str,
    output_path: Path,
) -> None:
    per_target = per_target.sort_values("target_id", key=lambda s: s.map(natural_target_key))
    x = np.arange(len(per_target))
    width = 0.24

    fig_width = max(14, len(per_target) * 0.24)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(
        x - width,
        per_target["loss_out_mean"],
        width,
        label="OUT shadow mean loss",
        color="#3e75a1",
        alpha=0.85,
    )
    ax.bar(
        x,
        per_target["loss_in_mean"],
        width,
        label="IN shadow mean loss",
        color="#d36c4e",
        alpha=0.85,
    )
    ax.bar(
        x + width,
        per_target["loss_target"],
        width,
        label="Target model loss",
        color="#8a8f93",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
    )
    ax.errorbar(
        x - width,
        per_target["loss_out_mean"],
        yerr=per_target["loss_out_std"],
        fmt="none",
        ecolor="black",
        capsize=1.5,
        linewidth=0.5,
    )
    ax.errorbar(
        x,
        per_target["loss_in_mean"],
        yerr=per_target["loss_in_std"],
        fmt="none",
        ecolor="black",
        capsize=1.5,
        linewidth=0.5,
    )

    for i, membership in enumerate(per_target["membership"]):
        color = "#d36c4e" if membership == "member" else "#3e75a1"
        ax.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.045)

    ax.set_xticks(x)
    ax.set_xticklabels(per_target["target_id"], rotation=90, fontsize=6)
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{run_name}: Per-Target Loss Comparison")
    ax.legend(fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_run_figures(
    per_target: pd.DataFrame,
    *,
    run_name: str,
    summary: dict[str, object],
    output_dir: Path,
    fpr_targets: tuple[float, ...],
) -> dict[str, Path]:
    fpr_1_label = f"{0.01:g}".replace(".", "p")
    paths = {
        "roc_curve": output_dir / f"roc_curve_{run_name}.png",
        "score_distribution": output_dir / f"score_distribution_{run_name}.png",
        "score_rankings": output_dir / f"score_rankings_{run_name}.png",
        "loss_comparison": output_dir / f"loss_comparison_{run_name}.png",
    }
    save_roc_curve(
        per_target,
        run_name=run_name,
        auc=float(summary["lira_auc"]),
        fpr_targets=fpr_targets,
        output_path=paths["roc_curve"],
    )
    save_score_distribution(
        per_target,
        run_name=run_name,
        auc=float(summary["lira_auc"]),
        tpr_at_1pct=summary.get(f"lira_tpr_at_fpr_{fpr_1_label}"),
        threshold_log_at_1pct=summary.get(f"lira_log_threshold_at_fpr_{fpr_1_label}"),
        output_path=paths["score_distribution"],
    )
    save_ranked_scores(per_target, run_name=run_name, output_path=paths["score_rankings"])
    save_loss_comparison(per_target, run_name=run_name, output_path=paths["loss_comparison"])
    return paths


def save_comparison_figures(
    per_target_by_run: dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    output_dir: Path,
    fpr_targets: tuple[float, ...],
) -> dict[str, Path]:
    paths = {
        "roc_curve_comparison": output_dir / "roc_curve_comparison.png",
        "metric_comparison": output_dir / "metric_comparison.png",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#255f85", "#b85c38", "#4f7c45", "#704f9f", "#8a6d3b"]
    for i, (run_name, per_target) in enumerate(per_target_by_run.items()):
        y_true, y_score = get_y_true_scores(per_target)
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_score)
        auc = float(summary_df.loc[summary_df["run_name"] == run_name, "lira_auc"].iloc[0])
        color = colors[i % len(colors)]
        axes[0].plot(fpr_vals, tpr_vals, color=color, linewidth=1.6, label=f"{run_name} ({auc:.3f})")
        axes[1].plot(fpr_vals, tpr_vals, color=color, linewidth=1.6, label=f"{run_name} ({auc:.3f})")

    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("ROC Comparison")
    axes[1].set_xscale("log")
    axes[1].set_xlim(min(fpr_targets), 1)
    axes[1].set_title("Low-FPR Region")
    fig.tight_layout()
    fig.savefig(paths["roc_curve_comparison"], bbox_inches="tight", dpi=150)
    plt.close(fig)

    fpr_1_label = f"{0.01:g}".replace(".", "p")
    metric_cols = [
        ("lira_auc", "LiRA AUC"),
        ("raw_loss_auc", "Raw-loss AUC"),
        (f"lira_tpr_at_fpr_{fpr_1_label}", "LiRA TPR@1%FPR"),
        (f"raw_loss_tpr_at_fpr_{fpr_1_label}", "Raw-loss TPR@1%FPR"),
    ]
    available_metrics = [(col, label) for col, label in metric_cols if col in summary_df]
    x = np.arange(len(available_metrics))
    width = 0.8 / max(1, len(summary_df))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (_, row) in enumerate(summary_df.iterrows()):
        values = [float(row[col]) for col, _ in available_metrics]
        offset = (i - (len(summary_df) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=str(row["run_name"]), color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in available_metrics], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_title("LiRA Comparison Metrics")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["metric_comparison"], bbox_inches="tight", dpi=150)
    plt.close(fig)

    return paths


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 4) -> str:
    headers = [header for _, header in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for col, _ in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(format_float(float(value), digits=digits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_interpretation(summary_df: pd.DataFrame) -> str:
    if len(summary_df) < 2:
        row = summary_df.iloc[0]
        return (
            f"{row['run_name']} has LiRA AUC {format_float(row['lira_auc'])} and "
            f"TPR@1%FPR {format_float(row.get('lira_tpr_at_fpr_0p01'))}."
        )

    names = set(summary_df["run_name"])
    if {"non_dp_60", "dp_eps3"}.issubset(names):
        non_dp = summary_df[summary_df["run_name"] == "non_dp_60"].iloc[0]
        dp = summary_df[summary_df["run_name"] == "dp_eps3"].iloc[0]
        auc_delta = float(dp["lira_auc"] - non_dp["lira_auc"])
        tpr_col = "lira_tpr_at_fpr_0p01"
        tpr_delta = float(dp[tpr_col] - non_dp[tpr_col])
        direction = "lower" if auc_delta < 0 else "higher"
        return (
            "Relative to the non-DP run, dp_eps3 has "
            f"{direction} LiRA AUC by {format_float(abs(auc_delta))} and "
            f"TPR@1%FPR changes by {format_float(tpr_delta)}. "
            "Negative deltas indicate reduced membership signal under the DP run."
        )

    best = summary_df.sort_values("lira_auc", ascending=False).iloc[0]
    worst = summary_df.sort_values("lira_auc", ascending=True).iloc[0]
    return (
        f"{best['run_name']} has the highest LiRA AUC "
        f"({format_float(best['lira_auc'])}); {worst['run_name']} has the lowest "
        f"({format_float(worst['lira_auc'])})."
    )


def write_report(
    *,
    output_path: Path,
    summary_df: pd.DataFrame,
    validation_rows: list[dict[str, object]],
    figure_paths_by_run: dict[str, dict[str, Path]],
    comparison_figure_paths: dict[str, Path],
    fpr_targets: tuple[float, ...],
    outlier_threshold: float,
) -> None:
    validation_df = pd.DataFrame(validation_rows)
    fpr_1_label = f"{0.01:g}".replace(".", "p")
    summary_columns = [
        ("run_name", "Run"),
        ("targets", "Targets"),
        ("member_targets", "Members"),
        ("nonmember_targets", "Nonmembers"),
        ("lira_auc", "LiRA AUC"),
        ("raw_loss_auc", "Raw Loss AUC"),
        (f"lira_tpr_at_fpr_{fpr_1_label}", "LiRA TPR@1%FPR"),
        (f"raw_loss_tpr_at_fpr_{fpr_1_label}", "Raw Loss TPR@1%FPR"),
        ("lira_auc_outlier_excluded", "LiRA AUC, Lambda<=50"),
        (f"lira_tpr_at_fpr_{fpr_1_label}_outlier_excluded", "LiRA TPR@1%FPR, Lambda<=50"),
        ("outlier_excluded_count", "Outliers"),
    ]
    summary_columns = [(col, header) for col, header in summary_columns if col in summary_df]

    validation_columns = [
        ("run_name", "Run"),
        ("rows", "Rows"),
        ("targets", "Targets"),
        ("target_rows", "Target Rows"),
        ("in_rows", "IN Rows"),
        ("out_rows", "OUT Rows"),
        ("member_targets", "Members"),
        ("nonmember_targets", "Nonmembers"),
        ("duplicate_keys", "Duplicate Keys"),
        ("csv_path", "CSV"),
    ]

    low_fpr_columns = [("run_name", "Run")]
    for fpr_target in fpr_targets:
        label = f"{fpr_target:g}".replace(".", "p")
        low_fpr_columns.append(
            (f"lira_tpr_at_fpr_{label}", f"LiRA TPR@{format_percent(fpr_target)}FPR")
        )

    score_columns = [
        ("run_name", "Run"),
        ("member_lira_log_lambda_mean", "Member log mean"),
        ("member_lira_log_lambda_median", "Member log median"),
        ("nonmember_lira_log_lambda_mean", "Nonmember log mean"),
        ("nonmember_lira_log_lambda_median", "Nonmember log median"),
        ("member_target_loss_mean", "Member loss mean"),
        ("nonmember_target_loss_mean", "Nonmember loss mean"),
    ]

    lines = [
        "# LiRA Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Method",
        "",
        "- For each target_id, one target loss, 16 IN losses, and 16 OUT losses were required.",
        "- IN and OUT shadow loss distributions were fit with scipy.stats.norm.fit.",
        "- LiRA log-lambda was computed as logpdf_in(loss_target) - logpdf_out(loss_target).",
        "- Higher LiRA scores mean more likely member. The raw-loss baseline uses -loss_target.",
        f"- Outlier-excluded metrics remove targets with LiRA lambda > {outlier_threshold:g}.",
        f"- Low-FPR TPRs were computed at: {', '.join(format_percent(v) for v in fpr_targets)}.",
        "",
        "## Validation",
        "",
        markdown_table(validation_df, validation_columns, digits=0),
        "",
        "## Summary",
        "",
        markdown_table(summary_df, summary_columns),
        "",
        "## Low-FPR LiRA TPRs",
        "",
        markdown_table(summary_df, low_fpr_columns),
        "",
        "## Score Summaries",
        "",
        markdown_table(summary_df, score_columns),
        "",
        "## Interpretation",
        "",
        build_interpretation(summary_df),
        "",
        "## Figures",
        "",
    ]

    for run_name, paths in figure_paths_by_run.items():
        lines.append(f"### {run_name}")
        lines.append("")
        for label, path in paths.items():
            rel_path = path.relative_to(output_path.parent)
            lines.append(f"- {label}: `{rel_path}`")
        lines.append("")

    if comparison_figure_paths:
        lines.append("### Comparison")
        lines.append("")
        for label, path in comparison_figure_paths.items():
            rel_path = path.relative_to(output_path.parent)
            lines.append(f"- {label}: `{rel_path}`")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LiRA scores for named loss CSVs and write comparison outputs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_spec,
        metavar="NAME=CSV",
        help="Named loss CSV input. May be specified more than once.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for per-target tables, summary, report, and figures.",
    )
    parser.add_argument(
        "--expected-targets",
        default=60,
        type=positive_int,
        help="Expected number of target_ids in each CSV. Default: 60.",
    )
    parser.add_argument(
        "--expected-shadow-count",
        default=16,
        type=positive_int,
        help="Expected IN and OUT shadow losses per target. Default: 16.",
    )
    parser.add_argument(
        "--outlier-threshold",
        default=DEFAULT_OUTLIER_THRESHOLD,
        type=positive_float,
        help="LiRA lambda threshold for outlier-excluded metrics. Default: 50.",
    )
    parser.add_argument(
        "--fpr",
        action="append",
        type=fpr_value,
        help="FPR target for TPR reporting. May be specified more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: list[tuple[str, Path]] = args.run
    run_names = [name for name, _ in runs]
    duplicate_names = sorted({name for name in run_names if run_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate run names: {duplicate_names}")

    fpr_targets = tuple(args.fpr) if args.fpr else DEFAULT_FPR_TARGETS
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_target_by_run: dict[str, pd.DataFrame] = {}
    validation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    figure_paths_by_run: dict[str, dict[str, Path]] = {}

    for run_name, csv_path in runs:
        if not csv_path.exists():
            raise FileNotFoundError(f"{run_name}: loss CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        validation = validate_loss_csv(
            df,
            run_name=run_name,
            csv_path=csv_path,
            expected_targets=args.expected_targets,
            expected_shadow_count=args.expected_shadow_count,
        )
        per_target = compute_lira_scores(df, run_name)
        summary = summarize_run(
            per_target,
            run_name=run_name,
            validation=validation,
            fpr_targets=fpr_targets,
            outlier_threshold=args.outlier_threshold,
        )

        per_target_path = output_dir / f"per_target_lira_{run_name}.csv"
        per_target.to_csv(per_target_path, index=False)

        per_target_by_run[run_name] = per_target
        validation_rows.append(validation)
        summary_rows.append(summary)
        figure_paths_by_run[run_name] = save_run_figures(
            per_target,
            run_name=run_name,
            summary=summary,
            output_dir=output_dir,
            fpr_targets=fpr_targets,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "lira_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    comparison_figure_paths = {}
    if len(per_target_by_run) > 1:
        comparison_figure_paths = save_comparison_figures(
            per_target_by_run, summary_df, output_dir, fpr_targets
        )

    write_report(
        output_path=output_dir / "comparison_report.md",
        summary_df=summary_df,
        validation_rows=validation_rows,
        figure_paths_by_run=figure_paths_by_run,
        comparison_figure_paths=comparison_figure_paths,
        fpr_targets=fpr_targets,
        outlier_threshold=args.outlier_threshold,
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {output_dir / 'comparison_report.md'}")
    for run_name in per_target_by_run:
        print(f"Wrote {output_dir / f'per_target_lira_{run_name}.csv'}")


if __name__ == "__main__":
    main()
