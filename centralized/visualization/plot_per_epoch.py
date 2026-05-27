import json
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

if len(sys.argv) > 1:
    metrics_path = Path(sys.argv[1]).resolve()
else:
    # Default: target eps=2, e100 DP run
    metrics_path = Path(
        "centralized_runs/archive/2026-05-25/"
        "fullbackbone_dp_privacy_epoch_sweep_20260525_020716/"
        "adam_fullbackbone_dp_targeteps2_clip200_lrhigherhead_b128_e100_seed42_20260525_020716/"
        "metrics.json"
    ).resolve()

with open(metrics_path) as f:
    data = json.load(f)

epochs_list = data["epochs"]
n_epochs = len(epochs_list)
best_epoch = data["best_epoch"]
run_name = metrics_path.parent.name
has_dp = "differential_privacy" in epochs_list[0]

epoch_nums = [e["epoch"] for e in epochs_list]
train_loss = [e["train"]["loss"] for e in epochs_list]
val_loss = [e["validation"]["loss"] for e in epochs_list]
train_auc = [e["train"]["auc"] for e in epochs_list]
val_auc = [e["validation"]["auc"] for e in epochs_list]
train_balacc = [e["train"]["balanced_accuracy"] for e in epochs_list]
val_balacc = [e["validation"]["balanced_accuracy"] for e in epochs_list]
val_sens = [e["validation"]["sensitivity"] for e in epochs_list]
val_spec = [e["validation"]["specificity"] for e in epochs_list]

if has_dp:
    epsilons = [e["differential_privacy"]["epsilon"] for e in epochs_list]
    dp_info = data["differential_privacy"]
    target_eps = dp_info.get("target_epsilon", "N/A")
    noise_mult = dp_info.get("noise_multiplier", "N/A")
    clip = dp_info.get("max_grad_norm", "N/A")
    title_text = (
        f"Per-Epoch Convergence — {run_name}<br>"
        f"<sup>target ε={target_eps}, σ={noise_mult:.4f}, clip={clip}, epochs={n_epochs}</sup>"
    )
else:
    title_text = (
        f"Per-Epoch Convergence — {run_name}<br>"
        f"<sup>non-DP baseline, epochs={n_epochs}</sup>"
    )

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Loss (Train + Val)",
        "AUC (Train + Val)",
        "Balanced Accuracy (Train + Val)",
        "Sensitivity & Specificity (Val)" + (" + Epsilon" if has_dp else ""),
    ),
    vertical_spacing=0.10,
    horizontal_spacing=0.08,
)

# --- Row 1, Col 1: Loss ---
fig.add_trace(go.Scatter(x=epoch_nums, y=train_loss, mode="lines", name="Train Loss",
                           line=dict(color="#2166ac", width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=epoch_nums, y=val_loss, mode="lines", name="Val Loss",
                           line=dict(color="#d6604d", width=1.5)), row=1, col=1)
fig.add_vline(x=best_epoch, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

# --- Row 1, Col 2: AUC ---
fig.add_trace(go.Scatter(x=epoch_nums, y=train_auc, mode="lines", name="Train AUC",
                           line=dict(color="#2166ac", width=1, dash="dot")), row=1, col=2)
fig.add_trace(go.Scatter(x=epoch_nums, y=val_auc, mode="lines", name="Val AUC",
                           line=dict(color="#d6604d", width=1.5)), row=1, col=2)
fig.add_vline(x=best_epoch, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

# --- Row 2, Col 1: Balanced Accuracy ---
fig.add_trace(go.Scatter(x=epoch_nums, y=train_balacc, mode="lines", name="Train BalAcc",
                           line=dict(color="#2166ac", width=1, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=epoch_nums, y=val_balacc, mode="lines", name="Val BalAcc",
                           line=dict(color="#d6604d", width=1.5)), row=2, col=1)
fig.add_vline(x=best_epoch, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

# --- Row 2, Col 2: Sensitivity, Specificity (+ Epsilon if DP) ---
fig.add_trace(go.Scatter(x=epoch_nums, y=val_sens, mode="lines", name="Val Sensitivity",
                           line=dict(color="#4daf4a", width=1.5)), row=2, col=2)
fig.add_trace(go.Scatter(x=epoch_nums, y=val_spec, mode="lines", name="Val Specificity",
                           line=dict(color="#984ea3", width=1.5)), row=2, col=2)
if has_dp:
    fig.add_trace(go.Scatter(x=epoch_nums, y=epsilons, mode="lines", name="ε (privacy)",
                               line=dict(color="#ff7f00", width=1, dash="dot"),
                               yaxis="y4"), row=2, col=2)
fig.add_vline(x=best_epoch, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=2)

fig.update_layout(
    title=dict(text=title_text, font=dict(size=13)),
    showlegend=True,
    legend=dict(x=1.02, y=1.0, orientation="v"),
    hoverlabel=dict(font_size=11),
    width=1200,
    height=800,
)

fig.update_xaxes(title_text="Epoch", row=2, col=1)
fig.update_xaxes(title_text="Epoch", row=2, col=2)
fig.update_yaxes(title_text="Loss", row=1, col=1)
fig.update_yaxes(title_text="AUC", row=1, col=2)
fig.update_yaxes(title_text="Balanced Accuracy", row=2, col=1)
fig.update_yaxes(title_text="Sens / Spec", row=2, col=2)
if has_dp:
    fig.update_yaxes(title_text="Epsilon", overlaying="y4", side="right", color="#ff7f00", row=2, col=2)

html_path = OUTPUT_DIR / f"per_epoch_{run_name}.html"
fig.write_html(html_path)
print(f"Saved: {html_path}")
