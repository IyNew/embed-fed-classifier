import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE.parent / "results_summary.csv"
ARCHIVES_ROOT = BASE.parents[1] / "centralized_runs" / "archive"
FIGURES_DIR = BASE / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

fb = df[
    (df["FreezeBackbone"] == False)
    & (df["Status"] == "completed")
].copy()

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


noise_lookup = build_noise_lookup()
print(f"Noise multiplier lookup: {len(noise_lookup)} entries")

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

target_eps = fb[
    (fb["DPMode"] == "target_epsilon")
    & (fb["TargetEpsilon"] >= 2)
    & (fb["TargetEpsilon"] <= 10)
    & (fb["BestValAUC"] > 0.5)
].copy()
target_eps["Sigma"] = target_eps["RunName"].map(noise_lookup)

noise_mode = fb[
    (fb["DPMode"] == "noise_multiplier")
    & (fb["NoiseMultiplier"] > 0)
    & (fb["FinalEpsilon"] > 0)
    & (fb["FinalEpsilon"] <= 200)
    & (fb["BestValAUC"] > 0.5)
    & (fb["ArchivePath"].str.contains("fullbackbone_dp_diagnostics|fullbackbone_dp_targeteps", na=False))
].copy()
noise_mode = noise_mode.drop_duplicates(subset=["NoiseMultiplier", "MaxGradNorm", "Epochs"])
noise_mode["Sigma"] = noise_mode["NoiseMultiplier"]

print(f"Baseline BestValAUC: {baseline_auc:.4f}")
print(f"Target epsilon: {len(target_eps)} runs")
print(f"Noise multiplier: {len(noise_mode)} runs")

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
    x=[30] * len(baseline),
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
        f"Mode: {mode_label}",
        f"σ: {row['Sigma']:.4f}" if pd.notna(row.get("Sigma")) else "σ: N/A",
        f"Target ε: {row['TargetEpsilon']}" if pd.notna(row.get("TargetEpsilon")) else "",
        f"Final ε: {row['FinalEpsilon']:.2f}",
        f"Clip: {row['MaxGradNorm']}",
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
    text=[f"σ={s:.3f}" for s in target_eps["Sigma"]],
    textposition="bottom center",
    textfont=dict(size=7, color="#2166ac"),
    hovertemplate="%{hovertext}<extra></extra>",
    hovertext=[make_hover(row, "target_epsilon") for _, row in target_eps.iterrows()],
))

fig.add_trace(go.Scatter(
    x=noise_mode["FinalEpsilon"],
    y=noise_mode["BestValAUC"],
    mode="markers+text",
    name=f"Noise multiplier mode (n={len(noise_mode)})",
    marker=dict(size=12, color="#d6604d", symbol="triangle-up", line=dict(width=1, color="black")),
    text=[f"σ={s}" for s in noise_mode["Sigma"]],
    textposition="top center",
    textfont=dict(size=8, color="#d6604d"),
    hovertemplate="%{hovertext}<extra></extra>",
    hovertext=[make_hover(row, "noise_multiplier") for _, row in noise_mode.iterrows()],
))

fig.update_layout(
    title="DP Training Results: Full-Backbone ConvNeXtTiny",
    xaxis_title="Final Epsilon",
    yaxis_title="Best Validation AUC",
    xaxis_range=[0, 32],
    yaxis_range=[0.55, max(
        target_eps["BestValAUC"].max(),
        noise_mode["BestValAUC"].max(),
        baseline_auc,
    ) * 1.03],
    legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
    hoverlabel=dict(font_size=12),
)

html_path = FIGURES_DIR / "dp_scatter_target_eps.html"
fig.write_html(html_path)
print(f"\nSaved interactive HTML: {html_path}")
