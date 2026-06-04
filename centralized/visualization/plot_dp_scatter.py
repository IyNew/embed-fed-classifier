import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

parser = argparse.ArgumentParser(description="DP scatter plot from results_summary.csv + archive scan")
parser.add_argument(
    "--model",
    type=str,
    default="ConvNeXtTiny",
    help="Model name to filter (default: ConvNeXtTiny). Use 'all' to include all models.",
)
args = parser.parse_args()

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE.parent / "results_summary.csv"
ARCHIVES_ROOT = BASE.parents[1] / "centralized_runs" / "archive"
FIGURES_DIR = BASE / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KEYWORD = args.model.lower().replace(" ", "") if args.model != "all" else None

df = pd.read_csv(CSV_PATH)

fb = df[
    (df["FreezeBackbone"] == False)
    & (df["Status"] == "completed")
].copy()

if args.model != "all":
    fb = fb[fb["Model"] == args.model].copy()
    print(f"Filtered to model: {args.model} ({len(fb)} runs from CSV)")

fb["BestValAUC"] = pd.to_numeric(fb["BestValAUC"], errors="coerce")
fb["FinalEpsilon"] = pd.to_numeric(fb["FinalEpsilon"], errors="coerce")
fb["TargetEpsilon"] = pd.to_numeric(fb["TargetEpsilon"], errors="coerce")
fb["NoiseMultiplier"] = pd.to_numeric(fb["NoiseMultiplier"], errors="coerce")
fb["BestValBalancedAccuracy"] = pd.to_numeric(fb["BestValBalancedAccuracy"], errors="coerce")
fb["FinalTestAUC"] = pd.to_numeric(fb["FinalTestAUC"], errors="coerce")
fb["FinalTestBalancedAccuracy"] = pd.to_numeric(fb["FinalTestBalancedAccuracy"], errors="coerce")
fb["BestValSensitivity"] = pd.to_numeric(fb["BestValSensitivity"], errors="coerce")
fb["BestValSpecificity"] = pd.to_numeric(fb["BestValSpecificity"], errors="coerce")
fb["FinalTestSensitivity"] = pd.to_numeric(fb["FinalTestSensitivity"], errors="coerce")
fb["FinalTestSpecificity"] = pd.to_numeric(fb["FinalTestSpecificity"], errors="coerce")
fb["Dropout"] = pd.to_numeric(fb["Dropout"], errors="coerce")


def build_noise_lookup():
    lookup = {}
    for metrics_path in ARCHIVES_ROOT.rglob("**/metrics.json"):
        try:
            with open(metrics_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        dp_info = data.get("differential_privacy", {})
        if dp_info.get("mode") == "target_epsilon":
            nm = dp_info.get("noise_multiplier")
            if nm is not None:
                lookup[metrics_path.parent.name] = float(nm)
    return lookup


def scan_archive_for_dp_runs(model_keyword, csv_run_names):
    rows = []
    for mpath in ARCHIVES_ROOT.rglob("**/metrics.json"):
        run_dir = mpath.parent
        run_name = run_dir.name
        if model_keyword not in run_name.lower():
            continue
        if run_name in csv_run_names:
            continue
        try:
            with open(mpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        dp_info = data.get("differential_privacy", {})
        if not dp_info.get("enabled") or dp_info.get("mode") != "target_epsilon":
            continue
        target_eps = dp_info.get("target_epsilon")
        if target_eps is None or target_eps < 2 or target_eps > 10:
            continue
        best_val = data.get("best_validation", {})
        if best_val.get("auc", 0) <= 0.5:
            continue
        test_info = data.get("test", {})
        bmt = data.get("best_model_test", {})
        summary = data.get("summary", {})
        row = {
            "RunName": run_name,
            "Model": "ConvNeXtSmall",
            "DPEnabled": True,
            "DPMode": "target_epsilon",
            "TargetEpsilon": target_eps,
            "FinalEpsilon": dp_info.get("final_epsilon", np.nan),
            "NoiseMultiplier": dp_info.get("noise_multiplier", np.nan),
            "MaxGradNorm": dp_info.get("max_grad_norm", np.nan),
            "Epochs": len(data.get("epochs", [])),
            "Dropout": 0.0,
            "BestValAUC": best_val.get("auc", np.nan),
            "BestValBalancedAccuracy": best_val.get("balanced_accuracy", np.nan),
            "BestValSensitivity": best_val.get("sensitivity", np.nan),
            "BestValSpecificity": best_val.get("specificity", np.nan),
            "FinalTestAUC": test_info.get("auc", np.nan),
            "FinalTestBalancedAccuracy": test_info.get("balanced_accuracy", np.nan),
            "FinalTestSensitivity": test_info.get("sensitivity", np.nan),
            "FinalTestSpecificity": test_info.get("specificity", np.nan),
            "Optimizer": "Adam",
            "LrBackbone": np.nan,
            "LrFinetune": np.nan,
            "BatchSize": np.nan,
            "ArchivePath": str(run_dir),
        }
        rows.append(row)
    return pd.DataFrame(rows)


noise_lookup = build_noise_lookup()
print(f"Noise multiplier lookup: {len(noise_lookup)} entries")

# --- Non-DP baseline from CSV ---
baseline = fb[
    (fb["DPEnabled"] == False)
    & (fb["BestValAUC"] > 0.5)
    & (fb["Optimizer"] != "SGD")
    & (~fb["ArchivePath"].str.contains("shadow", na=False))
].copy()
baseline_auc = baseline["BestValAUC"].max()
baseline_auc_min = baseline["BestValAUC"].min()
baseline_run = baseline.loc[baseline["BestValAUC"].idxmax()]
print(f"Non-DP baseline: {len(baseline)} runs, range=[{baseline_auc_min:.4f}, {baseline_auc:.4f}]")

# --- Target epsilon from CSV ---
target_eps_all = fb[
    (fb["DPMode"] == "target_epsilon")
    & (fb["TargetEpsilon"] >= 2)
    & (fb["TargetEpsilon"] <= 10)
    & (fb["BestValAUC"] > 0.5)
].copy()
target_eps_all["Sigma"] = target_eps_all["RunName"].map(noise_lookup)

# --- Supplement with archive DP runs not in CSV ---
if MODEL_KEYWORD:
    csv_names = set(fb["RunName"].values)
    archive_dp = scan_archive_for_dp_runs(MODEL_KEYWORD, csv_names)
    if len(archive_dp) > 0:
        print(f"Found {len(archive_dp)} additional DP runs in archives (not in CSV)")
        archive_dp["Sigma"] = archive_dp["NoiseMultiplier"]
        target_eps_all = pd.concat([target_eps_all, archive_dp], ignore_index=True)

# Split: 05-31 dropout sweep vs other target_epsilon runs
is_sweep = target_eps_all["ArchivePath"].str.contains("2026-05-31", na=False)
sweep_runs = target_eps_all[is_sweep].copy()
target_eps = target_eps_all[~is_sweep].copy()
print(f"Target epsilon: {len(target_eps)} non-sweep + {len(sweep_runs)} sweep runs")

print(f"Baseline BestValAUC: {baseline_auc:.4f}")
print(f"Target epsilon: {len(target_eps)} runs")

fig = go.Figure()

# --- Non-DP baseline range lines ---
fig.add_hline(
    y=baseline_auc,
    line_dash="dash",
    line_color="green",
    annotation_text=f"Non-DP max (AUC={baseline_auc:.4f})",
    annotation_position="right",
)
fig.add_hline(
    y=baseline_auc_min,
    line_dash="dot",
    line_color="green",
    annotation_text=f"Non-DP min (AUC={baseline_auc_min:.4f})",
    annotation_position="right",
)

# --- Non-DP baseline scatter points ---
non_dp_hover_texts = []
for _, row in baseline.iterrows():
    non_dp_hover_texts.append(
        f"<b>{row['RunName']}</b><br>"
        f"Model: {row['Model']}<br>"
        f"Mode: non-DP (no privacy)<br>"
        f"Epochs: {row['Epochs']}<br>"
        f"Optimizer: {row['Optimizer']}<br>"
        f"LrBackbone: {row['LrBackbone']}<br>"
        f"LrFinetune: {row['LrFinetune']}<br>"
        f"BatchSize: {row['BatchSize']}<br>"
        f"Best Val AUC: {row['BestValAUC']:.4f}<br>"
        f"Best Val BalAcc: {row['BestValBalancedAccuracy']:.4f}<br>"
        f"Best Val Sens: {row['BestValSensitivity']:.4f}<br>"
        f"Best Val Spec: {row['BestValSpecificity']:.4f}<br>"
        f"Final Test AUC: {row['FinalTestAUC']:.4f}<br>"
        f"Final Test BalAcc: {row['FinalTestBalancedAccuracy']:.4f}<br>"
        f"Final Test Sens: {row['FinalTestSensitivity']:.4f}<br>"
        f"Final Test Spec: {row['FinalTestSpecificity']:.4f}"
    )

fig.add_trace(go.Scatter(
    x=[10] * len(baseline),
    y=baseline["BestValAUC"],
    mode="markers",
    name=f"Non-DP (n={len(baseline)})",
    marker=dict(size=10, color="green", symbol="diamond", line=dict(width=1, color="black")),
    hovertemplate="%{hovertext}<extra></extra>",
    hovertext=non_dp_hover_texts,
))


def make_hover(row, mode_label):
    parts = [
        f"<b>{row['RunName']}</b>",
        f"Model: {row['Model']}",
        f"Mode: {mode_label}",
        f"σ: {row['Sigma']:.4f}" if pd.notna(row.get("Sigma")) else "σ: N/A",
        f"Target ε: {row['TargetEpsilon']}" if pd.notna(row.get("TargetEpsilon")) else "",
        f"Final ε: {row['FinalEpsilon']:.2f}",
        f"Dropout: {row['Dropout']:.3f}" if pd.notna(row.get("Dropout")) else "",
        f"Clip: {row['MaxGradNorm']:.1f}",
        f"Epochs: {row['Epochs']}",
        f"Best Val AUC: {row['BestValAUC']:.4f}",
        f"Best Val BalAcc: {row['BestValBalancedAccuracy']:.4f}",
        f"Best Val Sens: {row['BestValSensitivity']:.4f}",
        f"Best Val Spec: {row['BestValSpecificity']:.4f}",
        f"Final Test AUC: {row['FinalTestAUC']:.4f}",
        f"Final Test BalAcc: {row['FinalTestBalancedAccuracy']:.4f}",
        f"Final Test Sens: {row['FinalTestSensitivity']:.4f}",
        f"Final Test Spec: {row['FinalTestSpecificity']:.4f}",
    ]
    return "<br>".join(p for p in parts if p)


fig.add_trace(go.Scatter(
    x=target_eps["FinalEpsilon"],
    y=target_eps["BestValAUC"],
    mode="markers+text",
    name=f"Target epsilon mode (n={len(target_eps)})",
    marker=dict(size=10, color="#2166ac", line=dict(width=1, color="black")),
    text=[f"σ={s:.3f}" if pd.notna(s) else "" for s in target_eps["Sigma"]],
    textposition="bottom center",
    textfont=dict(size=7, color="#2166ac"),
    hovertemplate="%{hovertext}<extra></extra>",
    hovertext=[make_hover(row, "target_epsilon") for _, row in target_eps.iterrows()],
))

# --- Dropout sweep (05-31) as distinct trace ---
if len(sweep_runs) > 0:
    fig.add_trace(go.Scatter(
        x=sweep_runs["FinalEpsilon"],
        y=sweep_runs["BestValAUC"],
        mode="markers+text",
        name=f"Dropout sweep 05-31 (n={len(sweep_runs)})",
        marker=dict(
            size=12,
            color=sweep_runs["Dropout"],
            symbol="square",
            line=dict(width=1, color="black"),
            colorscale=[
                [0.0, "#2166ac"],
                [0.04, "#1f77b4"],
                [0.2, "#2ca02c"],
                [0.4, "#ff7f0e"],
                [1.0, "#d62728"],
            ],
            colorbar=dict(
                title="Dropout",
                tickvals=[0, 0.01, 0.05, 0.10, 0.25],
                ticktext=["0", "0.01", "0.05", "0.10", "0.25"],
                len=0.5,
                y=0.75,
            ),
        ),
        text=[f"d={d}" for d in sweep_runs["Dropout"]],
        textposition="top center",
        textfont=dict(size=8, color="#333333"),
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=[
            make_hover(row, f"target_epsilon (dropout sweep)")
            for _, row in sweep_runs.iterrows()
        ],
    ))

model_title = args.model if args.model != "all" else "All Models"
has_dp = len(target_eps) > 0 or len(sweep_runs) > 0
all_aucs = list(target_eps["BestValAUC"]) + list(sweep_runs["BestValAUC"]) + [baseline_auc]
y_max = max(all_aucs) * 1.03 if has_dp else baseline_auc * 1.03

fig.update_layout(
    title=f"DP Training Results: Full-Backbone {model_title}",
    xaxis_title="Final Epsilon",
    yaxis_title="Best Validation AUC",
    xaxis_range=[0, 32],
    yaxis_range=[0.55, y_max],
    legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
    hoverlabel=dict(font_size=12),
)

model_slug = args.model.lower().replace(" ", "_")
html_path = FIGURES_DIR / f"dp_scatter_{model_slug}.html"
fig.write_html(html_path)
print(f"\nSaved interactive HTML: {html_path}")
