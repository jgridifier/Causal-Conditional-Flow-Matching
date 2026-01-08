#!/usr/bin/env python3
"""
Causal Conditional Flow Matching - Quick Start Example

This script demonstrates the basic workflow of C-CFM:
1. Load or generate data
2. Fit the preprocessing pipeline
3. Train the flow matching model
4. Generate baseline and stress scenarios

Run with: python quickstart.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
import torch

from core import CausalFlowMatcher
from examples.data_fetcher import fetch_all_data


def main():
    print("=" * 60)
    print("Causal Conditional Flow Matching - Quick Start")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1/5] Loading data...")

    # Use synthetic data (works without API keys)
    # For real data, use: fetch_all_data() from data_fetcher.py
    # X  = fetch_all_data()
    X = pl.read_ipc("examples/all_data.arrow") # pre loaded for testing quicker

    print(f"  Data shape: {X.shape}")

    # =========================================================================
    # Step 2: Initialize C-CFM
    # =========================================================================
    print("\n[2/5] Initializing Causal Flow Matcher...")

    cfm = CausalFlowMatcher(
        hidden_dim=128,      # Network hidden dimension
        n_layers=4,          # Number of residual blocks
        dropout=0.0          # No dropout for this example
    )

    print(f"  Device: {cfm.device}")
    print(f"  Hidden dim: {cfm.hidden_dim}")

    # =========================================================================
    # Step 3: Fit Preprocessing Pipeline
    # =========================================================================
    print("\n[3/5] Fitting preprocessing pipeline...")

    cfm.fit(  # Fully automatic causal discovery
        X.drop("date")
    )

    print(f"  Number of regimes detected: {cfm.n_regimes}")
    print(f"  Variables in causal order: {cfm.variable_names[:5]}...")

    # =========================================================================
    # Step 4: Train the Model
    # =========================================================================
    print("\n[4/5] Training flow matching model...")

    results = cfm.train(
        n_epochs=1000,        # Reduced for quick demo
        lr=1e-3,
        batch_size=64,
        validate_every=20,
        verbose=True
    )

    print(f"\n  Final validation loss: {results['final_metrics']['val_loss']:.4f}")
    print(f"  Best validation loss: {results['best_loss']:.4f}")

    # =========================================================================
    # Step 5: Generate Scenarios
    # =========================================================================
    print("\n[5/5] Generating scenarios...")

    # Baseline scenarios
    print("\n  Generating 100 baseline scenarios...")
    baseline = cfm.sample(n_samples=100, regime=0)
    print(f"    Shape: {baseline.shape}")
    print(f"    Mean of first variable: {baseline[:, 0].mean():.4f}")

    # Stress scenario: VIX shock
    print("\n  Generating VIX shock scenario (3 std)...")
    vix_idx = cfm.variable_names.index('VIX') if 'VIX' in cfm.variable_names else 0
    stressed = cfm.shock(
        variable=vix_idx,
        magnitude=3.0,      # 3 standard deviations
        n_samples=100,
        regime=0
    )
    print(f"    Shape: {stressed.shape}")
    print(f"    Mean of VIX variable: {stressed[:, vix_idx].mean():.4f}")

    # Multiple shocks
    # Use the processed variable names from the topology
    print("\n  Generating multi-shock scenario...")
    var_names = cfm.variable_names
    multi = cfm.multi_shock(
        shocks={
            var_names[0]: 2.0,   # First variable +2 std
            var_names[1]: -1.5   # Second variable -1.5 std
        },
        n_samples=100,
        regime=0
    )
    print(f"    Shape: {multi.shape}")

    # =========================================================================
    # Step 6: Temporal Forecasting
    # =========================================================================
    print("\n[6/6] Temporal Forecasting...")

    # NOTE: For quickstart demo, we use fixed-step RK4 method since we only
    # trained for 100 epochs. The adaptive Dopri5 method requires well-trained
    # models to avoid numerical issues. For production use with fully trained
    # models, use method='dopri5' with tight tolerances for better accuracy.
    cfm.solver.config.method = 'rk4'  # Fixed-step method (more robust)
    cfm.solver.config.step_size = 0.05  # Step size for RK4
    print("\n  Using RK4 solver for quick demo (more robust for incompletely trained models)")

    # Generate autoregressive temporal forecasts (3 years forward)
    print("\n  Generating temporal forecast paths (36 months forward)...")
    forecast_paths = cfm.forecast(
        n_paths=50,       # 50 scenario paths
        n_steps=36,       # 36 months = 3 years
        regime=0,
        noise_scale=0.1   # Innovation noise between steps
    )
    print(f"    Shape: {forecast_paths.shape}")
    print(f"    This maintains temporal dependencies between time steps")

    # Constrained forecasting example
    print("\n  Generating constrained forecast (with predetermined path)...")
    # Create a simple constraint: first variable follows a rising path
    constraint_path = np.linspace(
        baseline[:, 0].mean(),  # Start at current mean
        baseline[:, 0].mean() + 2 * baseline[:, 0].std(),  # Rise by 2 std
        36  # Over 36 steps
    )

    constrained_forecast = cfm.forecast_with_constraints(
        n_paths=50,
        n_steps=36,
        constraint_paths={var_names[0]: constraint_path},
        regime=0,
        guidance_strength=1.0
    )
    print(f"    Constrained shape: {constrained_forecast.shape}")
    print(f"    Variable '{var_names[0]}' follows predetermined path")

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nBaseline Statistics:")
    for i, var in enumerate(cfm.variable_names[:5]):
        print(f"  {var}: mean={baseline[:, i].mean():.3f}, std={baseline[:, i].std():.3f}")

    print("\nStressed Statistics (VIX shock):")
    for i, var in enumerate(cfm.variable_names[:5]):
        print(f"  {var}: mean={stressed[:, i].mean():.3f}, std={stressed[:, i].std():.3f}")

    print("\nTemporal Forecast Statistics (first 5 vars, step 0 vs step 35):")
    for i, var in enumerate(cfm.variable_names[:5]):
        print(f"  {var}: t=0 mean={forecast_paths[:, 0, i].mean():.3f}, "
              f"t=35 mean={forecast_paths[:, -1, i].mean():.3f}")

    # =========================================================================
    # Save Model (optional)
    # =========================================================================
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_model.pt')
    cfm.save(model_path)
    print(f"\nModel saved to {model_path}")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)

    return cfm, baseline, stressed, forecast_paths


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file

    cfm, baseline, stressed, forecast_paths = main()
