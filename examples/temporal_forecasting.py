#!/usr/bin/env python3
"""
Temporal Forecasting with Causal Conditional Flow Matching

This example demonstrates how to use the CFM framework for multi-step
temporal forecasting. Key features:

1. Autoregressive Forecasting: Each step conditions on the previous state
2. Constrained Forecasting: Specify predetermined paths for certain variables
3. Scenario Analysis: "What if interest rates follow this trajectory?"

The key difference from generate_paths() is that forecast() maintains
temporal dependencies between steps, creating coherent forward projections.

Run with: python temporal_forecasting.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import polars as pl
import pandas as pd
import altair as alt
from pathlib import Path

from core import CausalFlowMatcher
from examples.data_fetcher import fetch_all_data


def main():
    print("=" * 70)
    print("Temporal Forecasting with Causal Conditional Flow Matching")
    print("=" * 70)

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # =========================================================================
    # Step 1: Load or Train Model
    # =========================================================================
    print("\n[1/5] Loading data and model...")

    model_path = Path(__file__).parent / "my_model.pt"

    if model_path.exists():
        print(f"  Loading existing model from {model_path}")
        cfm = CausalFlowMatcher.load(str(model_path))
    else:
        print("  No saved model found. Training new model...")
        X = fetch_all_data()
        cfm = CausalFlowMatcher(hidden_dim=128, n_layers=4)
        cfm.fit(X.drop("date"))
        cfm.train(n_epochs=100, lr=1e-3, batch_size=64)
        cfm.save(str(model_path))

    print(f"  Variables: {cfm.variable_names[:5]}... ({len(cfm.variable_names)} total)")

    # =========================================================================
    # Step 2: Basic Temporal Forecasting
    # =========================================================================
    print("\n[2/5] Generating temporal forecasting paths...")

    # Forecast parameters
    n_paths = 100  # Number of scenario paths
    n_steps = 36   # 36 months = 3 years forward

    # Generate autoregressive forecasting paths
    # Unlike generate_paths(), this maintains temporal dependencies
    forecast_paths = cfm.forecast(
        n_paths=n_paths,
        n_steps=n_steps,
        regime=0,
        noise_scale=0.1  # Controls innovation/diversity between steps
    )

    print(f"  Generated {n_paths} paths with {n_steps} time steps")
    print(f"  Shape: {forecast_paths.shape}")

    # Compare with independent sampling (old method)
    independent_paths = cfm.generate_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        regime=0
    )

    print(f"  For comparison: independent paths shape: {independent_paths.shape}")

    # =========================================================================
    # Step 3: Constrained Forecasting
    # =========================================================================
    print("\n[3/5] Generating constrained forecasting paths...")

    # Define constraint trajectories for specific variables
    # Example: Fed raises rates gradually over 3 years

    # Check if we have interest rate variable
    has_dff = 'DFF' in cfm.variable_names
    has_vix = 'VIX' in cfm.variable_names

    constrained_paths = None
    constraint_paths_dict = {}

    if has_dff:
        # Interest rate path: gradual tightening then plateau
        rate_path = np.concatenate([
            np.linspace(5.0, 5.5, 12),   # First year: gradual rise
            np.linspace(5.5, 5.75, 12),  # Second year: slower rise
            np.full(12, 5.75)            # Third year: hold steady
        ])
        constraint_paths_dict['DFF'] = rate_path
        print(f"  Constraint: DFF follows path from {rate_path[0]:.2f} to {rate_path[-1]:.2f}")

    if has_vix:
        # VIX path: spike mid-period then normalize
        vix_path = np.concatenate([
            np.linspace(15, 15, 12),     # First year: stable
            np.linspace(15, 30, 6),      # Mid-year spike
            np.linspace(30, 18, 6),      # Recovery
            np.linspace(18, 16, 12)      # Normalization
        ])
        constraint_paths_dict['VIX'] = vix_path
        print(f"  Constraint: VIX follows path with spike to {vix_path.max():.1f}")

    if constraint_paths_dict:
        constrained_paths = cfm.forecast_with_constraints(
            n_paths=n_paths,
            n_steps=n_steps,
            constraint_paths=constraint_paths_dict,
            regime=0,
            noise_scale=0.1,
            guidance_strength=1.0
        )
        print(f"  Generated constrained paths shape: {constrained_paths.shape}")
    else:
        print("  Skipping constrained forecasting (DFF/VIX not in variables)")

    # =========================================================================
    # Step 4: Analyze Temporal Coherence
    # =========================================================================
    print("\n[4/5] Analyzing temporal coherence...")

    # Compute autocorrelation of paths to show temporal dependencies
    def compute_path_autocorr(paths, var_idx, lag=1):
        """Compute average autocorrelation across paths for a variable."""
        autocorrs = []
        for path_idx in range(paths.shape[0]):
            series = paths[path_idx, :, var_idx]
            if len(series) > lag:
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(corr)
        return np.mean(autocorrs) if autocorrs else 0.0

    print("\n  Lag-1 Autocorrelation Comparison:")
    print("  " + "-" * 50)
    print(f"  {'Variable':<20} {'Forecast':<15} {'Independent':<15}")
    print("  " + "-" * 50)

    for i, var in enumerate(cfm.variable_names[:5]):
        forecast_ac = compute_path_autocorr(forecast_paths, i)
        independent_ac = compute_path_autocorr(independent_paths, i)
        print(f"  {var:<20} {forecast_ac:>+.3f}          {independent_ac:>+.3f}")

    print("  " + "-" * 50)
    print("  (Higher autocorrelation = more temporal coherence)")

    # =========================================================================
    # Step 5: Visualization
    # =========================================================================
    print("\n[5/5] Creating visualizations...")

    # Prepare data for Altair
    def paths_to_dataframe(paths, var_names, path_type):
        """Convert paths array to long-form DataFrame."""
        records = []
        for path_idx in range(paths.shape[0]):
            for step_idx in range(paths.shape[1]):
                for var_idx, var_name in enumerate(var_names):
                    records.append({
                        'path_id': path_idx,
                        'step': step_idx,
                        'variable': var_name,
                        'value': float(paths[path_idx, step_idx, var_idx]),
                        'type': path_type
                    })
        return pd.DataFrame(records)

    # Select a few variables for visualization
    viz_vars = cfm.variable_names[:4]

    # Create comparison dataframe (temporal vs independent)
    forecast_df = paths_to_dataframe(forecast_paths, cfm.variable_names, 'Temporal Forecast')
    independent_df = paths_to_dataframe(independent_paths, cfm.variable_names, 'Independent Samples')

    # Filter to visualization variables
    forecast_df = forecast_df[forecast_df['variable'].isin(viz_vars)]
    independent_df = independent_df[independent_df['variable'].isin(viz_vars)]

    comparison_df = pd.concat([forecast_df, independent_df])

    # Create path comparison chart
    comparison_chart = alt.Chart(comparison_df).mark_line(opacity=0.15, size=0.5).encode(
        x=alt.X('step:Q', title='Time Step (months)'),
        y=alt.Y('value:Q', title='Value'),
        color=alt.Color('type:N', title='Method'),
        detail='path_id:N'
    ).properties(
        width=300,
        height=150
    ).facet(
        column=alt.Column('variable:N', title=None, header=alt.Header(labelAngle=0)),
        row=alt.Row('type:N', title=None)
    ).resolve_scale(
        y='independent'
    )

    comparison_chart.save('temporal_forecast_comparison.html')
    print("  Saved: temporal_forecast_comparison.html")

    # Create percentile band chart for temporal forecast
    percentile_data = []
    for step_idx in range(n_steps):
        for var_idx, var_name in enumerate(cfm.variable_names):
            values = forecast_paths[:, step_idx, var_idx]
            percentile_data.append({
                'step': step_idx,
                'variable': var_name,
                'p10': float(np.percentile(values, 10)),
                'p25': float(np.percentile(values, 25)),
                'p50': float(np.percentile(values, 50)),
                'p75': float(np.percentile(values, 75)),
                'p90': float(np.percentile(values, 90)),
                'mean': float(np.mean(values))
            })

    percentile_df = pd.DataFrame(percentile_data)
    percentile_df = percentile_df[percentile_df['variable'].isin(viz_vars)]

    # Create layered band chart
    base = alt.Chart(percentile_df).encode(
        x=alt.X('step:Q', title='Time Step (months)')
    )

    band_90 = base.mark_area(opacity=0.2, color='steelblue').encode(
        y=alt.Y('p10:Q', title='Value'),
        y2='p90:Q'
    )

    band_50 = base.mark_area(opacity=0.3, color='steelblue').encode(
        y='p25:Q',
        y2='p75:Q'
    )

    median_line = base.mark_line(color='darkblue', size=2).encode(
        y='p50:Q'
    )

    mean_line = base.mark_line(color='red', strokeDash=[5, 3], size=1.5).encode(
        y='mean:Q'
    )

    forecast_band_chart = (band_90 + band_50 + median_line + mean_line).properties(
        width=250,
        height=150,
        title=alt.TitleParams(text='Temporal Forecast with Confidence Bands', anchor='middle')
    ).facet(
        column=alt.Column('variable:N', header=alt.Header(labelAngle=0)),
        columns=4
    ).resolve_scale(
        y='independent'
    )

    forecast_band_chart.save('temporal_forecast_bands.html')
    print("  Saved: temporal_forecast_bands.html")

    # Create constrained forecast visualization if available
    if constrained_paths is not None:
        constrained_data = []
        for step_idx in range(n_steps):
            for var_idx, var_name in enumerate(cfm.variable_names):
                values = constrained_paths[:, step_idx, var_idx]
                constrained_data.append({
                    'step': step_idx,
                    'variable': var_name,
                    'p10': float(np.percentile(values, 10)),
                    'p50': float(np.percentile(values, 50)),
                    'p90': float(np.percentile(values, 90))
                })

        constrained_df = pd.DataFrame(constrained_data)

        # Add constraint lines
        constraint_line_data = []
        for var_name, path in constraint_paths_dict.items():
            for step_idx, value in enumerate(path):
                constraint_line_data.append({
                    'step': step_idx,
                    'variable': var_name,
                    'constraint_value': float(value)
                })

        constraint_line_df = pd.DataFrame(constraint_line_data)

        # Filter to constrained variables
        constrained_vars = list(constraint_paths_dict.keys())
        constrained_df_filtered = constrained_df[constrained_df['variable'].isin(constrained_vars)]

        # Create chart with constraint overlay
        base_constrained = alt.Chart(constrained_df_filtered).encode(
            x=alt.X('step:Q', title='Time Step (months)')
        )

        band = base_constrained.mark_area(opacity=0.3, color='steelblue').encode(
            y=alt.Y('p10:Q', title='Value'),
            y2='p90:Q'
        )

        median = base_constrained.mark_line(color='blue', size=1.5).encode(
            y='p50:Q'
        )

        constraint_line = alt.Chart(constraint_line_df).mark_line(
            color='red',
            strokeDash=[5, 3],
            size=2
        ).encode(
            x='step:Q',
            y='constraint_value:Q'
        )

        constrained_chart = (band + median + constraint_line).properties(
            width=300,
            height=200,
            title='Constrained Forecast (red=constraint path)'
        ).facet(
            column=alt.Column('variable:N', header=alt.Header(labelAngle=0))
        ).resolve_scale(
            y='independent'
        )

        constrained_chart.save('temporal_forecast_constrained.html')
        print("  Saved: temporal_forecast_constrained.html")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"""
Temporal Forecasting Results:
-----------------------------
- Generated {n_paths} paths over {n_steps} time steps ({n_steps // 12} years)
- Variables: {len(cfm.variable_names)}
- Method: Autoregressive conditioning with noise_scale=0.1

Key Differences from Independent Sampling:
- forecast(): Temporal dependencies preserved (higher autocorrelation)
- generate_paths(): Independent samples at each step

Constrained Forecasting:
- Variables with constraints: {list(constraint_paths_dict.keys()) if constraint_paths_dict else 'None'}
- Guidance enforces predetermined paths while allowing other variables to respond

Files Created:
1. temporal_forecast_comparison.html - Side-by-side comparison
2. temporal_forecast_bands.html - Confidence bands for temporal forecast
3. temporal_forecast_constrained.html - Constrained forecast visualization
""")

    print("=" * 70)
    print("Temporal forecasting example complete!")
    print("=" * 70)

    return cfm, forecast_paths, constrained_paths


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    cfm, forecast_paths, constrained_paths = main()
