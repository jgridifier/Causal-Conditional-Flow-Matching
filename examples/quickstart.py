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
    X  = fetch_all_data()

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
        n_epochs=100,        # Reduced for quick demo
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

    # =========================================================================
    # Save Model (optional)
    # =========================================================================
    cfm.save('D:\DEV\Causal-Conditional-Flow-Matching\examples\my_model.pt')
    print("\nModel saved to my_model.pt")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)

    return cfm, baseline, stressed


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file

    cfm, baseline, stressed = main()
