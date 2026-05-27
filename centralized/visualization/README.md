# Visualization Pipeline

Scripts and notebooks for visualizing centralized DP training results, located under `centralized/visualization/`.

## Dependencies

```bash
pip install pandas matplotlib plotly kaleido
```

## Data Sources

| Source | Path | Description |
|---|---|---|
| Results CSV | `centralized/results_summary.csv` | Summary of all completed training runs with metrics |
| Run artifacts | `centralized_runs/archive/YYYY-MM-DD/<sweep>/<run>/metrics.json` | Per-epoch metrics and DP privacy parameters |

## Scripts

### 1. DP Scatter Plot

**Files**: `plot_dp_scatter.py`, `dp_scatter.ipynb`

Generates an interactive Plotly scatter plot comparing DP training modes:
- **Target epsilon mode** (eps 2–10): auto-computed noise multiplier from target epsilon
- **Noise multiplier mode**: manually specified noise multiplier
- **Non-DP baseline**: green star marker with dashed horizontal line

**Outputs**:
- `centralized/visualization/figures/dp_scatter_target_eps.html` — interactive HTML

**Filtering logic**:
- `FreezeBackbone == false` (full-backbone only)
- `Status == completed`
- For target epsilon: `DPMode == 'target_epsilon'`, `TargetEpsilon` in [2, 10], `BestValAUC > 0.5`
- For noise multiplier: `DPMode == 'noise_multiplier'`, `NoiseMultiplier > 0`, `FinalEpsilon <= 200`
- Non-DP baseline: `DPEnabled == false`, highest `BestValAUC`

**Noise multiplier lookup**:
For target epsilon runs, the noise multiplier is auto-computed by the privacy engine and recorded in `differential_privacy.noise_multiplier` in each run's `metrics.json`. The script scans all archives to build a lookup by run name.

**Hover text per point**: run name, mode, σ, target/final ε, clip norm, epochs, Best Val AUC/BalAcc/Sens/Spec, Final Test AUC/BalAcc/Sens/Spec.

**Usage**:
```bash
# Python script
python centralized/visualization/plot_dp_scatter.py

# Jupyter notebook
jupyter notebook centralized/visualization/dp_scatter.ipynb
```

### 2. Per-Epoch Convergence Plot

**File**: `plot_per_epoch.py`

Generates a 4-panel interactive Plotly figure from a single run's `metrics.json`:

| Panel | Content |
|---|---|
| Top-left | Train + Val Loss |
| Top-right | Train + Val AUC |
| Bottom-left | Train + Val Balanced Accuracy |
| Bottom-right | Val Sensitivity + Val Specificity (+ Epsilon if DP) |

A vertical dashed line marks the best epoch.

**Usage**:
```bash
# With arguments: specify metrics.json path
python centralized/visualization/plot_per_epoch.py <path/to/metrics.json>

# Without arguments: defaults to target eps=2, e100 DP run
python centralized/visualization/plot_per_epoch.py
```

**Handles both DP and non-DP runs** — automatically detects whether `differential_privacy` data is present and adjusts the 4th panel accordingly.

## Run Examples

```bash
# DP scatter plot
python centralized/visualization/plot_dp_scatter.py

# Per-epoch plot for a specific run
python centralized/visualization/plot_per_epoch.py \
  centralized_runs/archive/2026-05-25/fullbackbone_dp_privacy_epoch_sweep_20260525_020716/\
  adam_fullbackbone_dp_targeteps2_clip200_lrhigherhead_b128_e100_seed42_20260525_020716/metrics.json

# Per-epoch plot for non-DP baseline
python centralized/visualization/plot_per_epoch.py \
  centralized_runs/archive/2026-05-26/fullbackbone_eps5_801010_e80_sweep_20260525_223650/\
  adam_fullbackbone_nondp_lrhigherhead_801010_b128_e80_seed42_20260525_223650/metrics.json
```

## Output Directory

All figures saved under `centralized/visualization/figures/`:

```
centralized/visualization/figures/
├── dp_scatter_target_eps.html          # Interactive scatter plot
├── per_epoch_adam_fullbackbone_dp_targeteps2_...html   # DP convergence
├── per_epoch_adam_fullbackbone_nondp_...html            # Non-DP baseline
└── ...
```

## Updating the Scatter Plot

When new DP training runs complete and are added to `results_summary.csv`:

1. Ensure results are archived with `metrics.json` (for noise multiplier lookup)
2. Add entry to `centralized/results_summary.csv` via the archive protocol
3. Re-run `python centralized/visualization/plot_dp_scatter.py`
4. The script auto-discovers new runs via the CSV filter and noise lookup from archives

No manual edits needed — the pre-existing sweeps are also refreshed if new metrics.json files appear in archives.
