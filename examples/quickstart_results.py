#!/usr/bin/env python3
"""
Quickstart Results Visualization with Temporal Forecasting

This script demonstrates:
1. Temporal forecasting with autoregressive conditioning (new method)
2. Comparison with independent sampling (old method)
3. Constrained forecasting with predetermined variable paths
4. Visualization of results

Run with: python quickstart_results.py
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import altair as alt
import polars as pl
import pandas as pd
from pathlib import Path

from core import CausalFlowMatcher
from examples.data_fetcher import fetch_all_data


# Load the trained model
model_path = Path(__file__).parent / 'my_model.pt'
cfm = CausalFlowMatcher.load(str(model_path))

X = fetch_all_data()  # polars dataframe with date column

# ========================================
# Generate Temporal Forecasting Paths
# ========================================

print("=" * 60)
print("Temporal Forecasting with C-CFM")
print("=" * 60)

n_paths = 50  # Number of different scenario paths
n_steps = 48  # Number of time steps forward (48 months = 4 years)

# NEW: Use temporal forecasting (autoregressive, maintains temporal dependencies)
print("\n[1/4] Generating temporal forecast paths (autoregressive)...")
temporal_paths = cfm.forecast(
    n_paths=n_paths,
    n_steps=n_steps,
    regime=0,
    noise_scale=0.1  # Innovation noise between steps
)
print(f"  Generated {n_paths} temporal paths with {n_steps} steps each")
print(f"  Shape: {temporal_paths.shape}")

# Use temporal_paths as the primary visualization
paths = temporal_paths

# ========================================
# Generate Constrained Forecast
# ========================================
print("\n[3/4] Generating constrained forecast...")

# Create a constraint: if we have VIX, make it spike mid-forecast
constrained_paths = None
constraint_var = None
constraint_trajectory = None

if 'VIX' in cfm.variable_names:
    constraint_var = 'VIX'
    # VIX spike scenario: stable -> spike -> recovery
    constraint_trajectory = np.concatenate([
        np.full(12, 15),           # 1 year stable
        np.linspace(15, 35, 12),   # 1 year spike
        np.linspace(35, 18, 12),   # 1 year recovery
        np.full(12, 18)            # 1 year normalized
    ])
elif len(cfm.variable_names) > 0:
    constraint_var = cfm.variable_names[0]
    # Generic rising path for first variable
    baseline_sample = cfm.sample(n_samples=100)
    var_mean = baseline_sample[:, 0].mean()
    var_std = baseline_sample[:, 0].std()
    constraint_trajectory = np.linspace(var_mean, var_mean + 2 * var_std, n_steps)

if constraint_var and constraint_trajectory is not None:
    constrained_paths = cfm.forecast_with_constraints(
        n_paths=n_paths,
        n_steps=n_steps,
        constraint_paths={constraint_var: constraint_trajectory},
        regime=0,
        guidance_strength=1.0
    )
    print(f"  Generated constrained paths with {constraint_var} following predetermined path")
    print(f"  Shape: {constrained_paths.shape}")
else:
    print("  Skipping constrained forecast (no suitable variable found)")

# ========================================
# Generate single-step baseline for comparison
# ========================================
print("\n[4/4] Generating baseline samples for distribution comparison...")
baseline = cfm.sample(n_samples=1000)  # numpy array without date column
print(f"  Generated {baseline.shape[0]} baseline samples")

# Convert baseline numpy array to polars DataFrame with variable names
baseline_df = pl.DataFrame(
    baseline,
    schema=cfm.variable_names  # Use the variable names from the model (in causal order)
)

# Add a 'type' column to distinguish between real data and synthetic baseline
baseline_df = baseline_df.with_columns(
    pl.lit('Baseline Synthetic').alias('type')
)

# For real data, we need to reorder columns to match the causal order from the model
# so that we can stack them together
X_without_date = X.drop('date')

# Reorder real data columns to match the causal order from the model
X_reordered = X_without_date.select(cfm.variable_names)

X_reordered = X_reordered.with_columns(
    pl.lit('Real Data').alias('type')
)

# Combine the dataframes (now they have matching column names and order)
combined_df = pl.concat([X_reordered, baseline_df])

# Convert to pandas for Altair (Altair works better with pandas)
combined_pd = combined_df.to_pandas()

# Melt the dataframe for faceted plotting
melted_df = combined_pd.melt(id_vars=['type'], var_name='variable', value_name='value')

# Create faceted box plot
chart = alt.Chart(melted_df).mark_boxplot().encode(
    x='type:N',
    y='value:Q',
    color='type:N'
).properties(
    width=150,
    height=200,
    title=alt.TitleParams(text=alt.expr('datum.variable'), anchor='middle')
).facet(
    column=alt.Column('variable:N', header=alt.Header(labelAngle=0)),
    columns=4  # 4 charts per row
).resolve_scale(
    y='independent'  # Each variable gets its own y-axis scale
)

chart.save('quickstart_results_boxplot.html')
print("  Saved: quickstart_results_boxplot.html")

# ========================================
# Create path visualization (time series)
# ========================================
print("\nCreating path visualizations...")

# Convert paths to long-form DataFrame for Altair
# Shape: (n_paths, n_steps, n_variables) -> DataFrame with columns: path_id, step, variable, value

path_data = []
for path_idx in range(n_paths):
    for step_idx in range(n_steps):
        for var_idx, var_name in enumerate(cfm.variable_names):
            path_data.append({
                'path_id': path_idx,
                'step': step_idx,
                'variable': var_name,
                'value': paths[path_idx, step_idx, var_idx]
            })

paths_df = pd.DataFrame(path_data)

# Create faceted line chart showing all paths
path_chart = alt.Chart(paths_df).mark_line(opacity=0.3, size=1).encode(
    x=alt.X('step:Q', title='Time Step'),
    y=alt.Y('value:Q', title='Value'),
    color=alt.Color('path_id:N', legend=None),  # Different color per path, no legend
    detail='path_id:N'  # Separate line per path
).properties(
    width=250,
    height=150,
    title=alt.TitleParams(text=alt.expr('datum.variable'), anchor='middle')
).facet(
    column=alt.Column('variable:N', header=alt.Header(labelAngle=0)),
    columns=4
).resolve_scale(
    y='independent'
)

path_chart.save('quickstart_results_paths.html')
print("  Saved: quickstart_results_paths.html")

# ========================================
# Create alternative view: Percentile bands
# ========================================

# Calculate percentiles across paths for each step
percentile_data = []
for step_idx in range(n_steps):
    for var_idx, var_name in enumerate(cfm.variable_names):
        values = paths[:, step_idx, var_idx]
        percentile_data.append({
            'step': step_idx,
            'variable': var_name,
            'p10': np.percentile(values, 10),
            'p25': np.percentile(values, 25),
            'p50': np.percentile(values, 50),
            'p75': np.percentile(values, 75),
            'p90': np.percentile(values, 90),
            'mean': np.mean(values)
        })

percentile_df = pd.DataFrame(percentile_data)

# Create confidence band chart
base = alt.Chart(percentile_df).encode(
    x=alt.X('step:Q', title='Time Step')
)

# 80% confidence band (p10-p90)
band_80 = base.mark_area(opacity=0.2, color='blue').encode(
    y=alt.Y('p10:Q', title='Value'),
    y2='p90:Q'
)

# 50% confidence band (p25-p75)
band_50 = base.mark_area(opacity=0.3, color='blue').encode(
    y='p25:Q',
    y2='p75:Q'
)

# Median line
median_line = base.mark_line(color='darkblue', size=2).encode(
    y='p50:Q'
)

# Mean line
mean_line = base.mark_line(color='red', strokeDash=[5, 5], size=2).encode(
    y='mean:Q'
)

# Combine layers
band_chart = (band_80 + band_50 + median_line + mean_line).properties(
    width=250,
    height=150,
    title=alt.TitleParams(text=alt.expr('datum.variable'), anchor='middle')
).facet(
    column=alt.Column('variable:N', header=alt.Header(labelAngle=0)),
    columns=4
).resolve_scale(
    y='independent'
)

band_chart.save('quickstart_results_bands.html')
print("  Saved: quickstart_results_bands.html")

# ========================================
# Create Temporal Coherence Analysis
# ========================================
print("\nAnalyzing temporal coherence...")

# Compute lag-1 autocorrelation to verify temporal dependencies
def compute_autocorr(paths_arr, var_idx, lag=1):
    """Compute average autocorrelation across paths."""
    autocorrs = []
    for path_idx in range(paths_arr.shape[0]):
        series = paths_arr[path_idx, :, var_idx]
        if len(series) > lag:
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrs.append(corr)
    return np.mean(autocorrs) if autocorrs else 0.0

# Create autocorrelation data
autocorr_data = []
for var_idx, var_name in enumerate(cfm.variable_names[:6]):
    temporal_ac = compute_autocorr(temporal_paths, var_idx)
    autocorr_data.append({
        'variable': var_name,
        'autocorrelation': temporal_ac
    })

autocorr_df = pd.DataFrame(autocorr_data)

# Create autocorrelation chart
autocorr_chart = alt.Chart(autocorr_df).mark_bar(color='steelblue').encode(
    x=alt.X('variable:N', title='Variable'),
    y=alt.Y('autocorrelation:Q', title='Lag-1 Autocorrelation')
).properties(
    width=400,
    height=250,
    title='Temporal Coherence: Lag-1 Autocorrelation of Forecast Paths'
)

autocorr_chart.save('quickstart_results_autocorr.html')
print("  Saved: quickstart_results_autocorr.html")

# ========================================
# Create Constrained Forecast Visualization
# ========================================
if constrained_paths is not None and constraint_var is not None:
    print("\nCreating constrained forecast visualization...")

    # Get the index of the constrained variable
    var_idx = cfm.variable_names.index(constraint_var)

    # Build data for constrained paths
    constrained_vis_data = []
    for step_idx in range(n_steps):
        values = constrained_paths[:, step_idx, var_idx]
        constrained_vis_data.append({
            'step': step_idx,
            'p10': float(np.percentile(values, 10)),
            'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)),
            'constraint': float(constraint_trajectory[step_idx])
        })

    constrained_vis_df = pd.DataFrame(constrained_vis_data)

    # Create layered chart
    base_c = alt.Chart(constrained_vis_df).encode(
        x=alt.X('step:Q', title='Time Step (months)')
    )

    band_c = base_c.mark_area(opacity=0.3, color='steelblue').encode(
        y=alt.Y('p10:Q', title=constraint_var),
        y2='p90:Q'
    )

    median_c = base_c.mark_line(color='blue', size=2).encode(
        y='p50:Q'
    )

    constraint_line = base_c.mark_line(color='red', strokeDash=[5, 3], size=2.5).encode(
        y='constraint:Q'
    )

    constrained_chart = (band_c + median_c + constraint_line).properties(
        width=500,
        height=300,
        title=f'Constrained Forecast: {constraint_var} (red=constraint path, blue=model forecast)'
    )

    constrained_chart.save('quickstart_results_constrained.html')
    print("  Saved: quickstart_results_constrained.html")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
Generated Forecasts:
- Temporal Forecast: {n_paths} paths x {n_steps} steps (autoregressive)
- Constrained Forecast: {constraint_var if constraint_var else 'N/A'}
- Variables: {len(cfm.variable_names)}

Key Feature:
  Temporal forecast uses autoregressive conditioning to maintain
  temporal coherence between consecutive time steps.

Visualizations Created:
  1. quickstart_results_boxplot.html    - Distribution comparison
  2. quickstart_results_paths.html      - All individual paths
  3. quickstart_results_bands.html      - Percentile bands over time
  4. quickstart_results_autocorr.html   - Temporal coherence analysis
  5. quickstart_results_constrained.html - Constrained forecast (if applicable)
""")
print("=" * 60)
