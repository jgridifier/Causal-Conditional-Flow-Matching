
# Load my_model.pt
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

from core import CausalFlowMatcher
from examples.data_fetcher import fetch_all_data


cfm = CausalFlowMatcher.load('D:\DEV\Causal-Conditional-Flow-Matching\examples\my_model.pt')

X = fetch_all_data() # polars dataframe with date column

# ========================================
# Generate multi-step forward projections
# ========================================

print("Generating forward-looking paths...")
n_paths = 50  # Number of different scenario paths
n_steps = 48  # Number of time steps forward

# Use the optimized generate_paths method from CausalFlowMatcher
paths = cfm.generate_paths(n_paths=n_paths, n_steps=n_steps, regime=0)
print(f"Generated {n_paths} paths with {n_steps} steps each")

# ========================================
# Also generate single-step baseline for comparison
# ========================================
baseline = cfm.sample(n_samples=1000) # numpy array without date column

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
print("Box plot saved to quickstart_results_boxplot.html")

# ========================================
# Create path visualization (time series)
# ========================================

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
print("Path visualization saved to quickstart_results_paths.html")

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
print("Percentile bands visualization saved to quickstart_results_bands.html")

print("\n" + "="*60)
print("Summary:")
print(f"- Generated {n_paths} paths with {n_steps} time steps each")
print(f"- Variables: {len(cfm.variable_names)}")
print("\nVisualizations created:")
print("  1. quickstart_results_boxplot.html - Distribution comparison")
print("  2. quickstart_results_paths.html - All individual paths")
print("  3. quickstart_results_bands.html - Percentile bands over time")
print("="*60)


