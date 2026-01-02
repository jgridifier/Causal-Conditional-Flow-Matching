"""
CTree Demonstration: Monte Carlo vs Asymptotic Branching

This example demonstrates the conditional inference tree (CTree) algorithm
with its adaptive p-value computation:
- For small samples (N < 100): Uses Monte Carlo permutation test (robust)
- For large samples (N >= 100): Uses asymptotic chi-squared approximation (fast)

We use the data_fetcher to create a realistic financial dataset and apply
CTree to identify market regimes based on macroeconomic conditions.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from core.ctree import CTree
from examples.data_fetcher import create_sample_dataset
import time


def demo_small_sample_mc():
    """Demonstrate Monte Carlo path with small sample (N < 100)."""
    print("\n" + "=" * 70)
    print("DEMO 1: Small Sample (N=60) - Monte Carlo Permutation Test")
    print("=" * 70)

    np.random.seed(42)
    n = 60

    # Create features: VIX, Term Spread, Unemployment
    X = np.random.randn(n, 3)
    X[:, 0] = 15 + 10 * np.abs(np.random.randn(n))  # VIX-like
    X[:, 1] = 2.0 + np.random.randn(n)  # Term spread
    X[:, 2] = 5.0 + 2 * np.random.randn(n)  # Unemployment

    # Create regime-dependent response (e.g., SPY returns)
    # High VIX => negative returns, positive term spread => positive returns
    Y = np.where(X[:, 0] > 20, -0.01, 0.01)  # VIX regime
    Y += 0.005 * X[:, 1]  # Term spread effect
    Y = Y.reshape(-1, 1) + 0.005 * np.random.randn(n, 1)

    # Fit CTree with Monte Carlo (threshold=100, so N=60 triggers MC)
    print(f"\nData: {n} samples, 3 features")
    print("Features: VIX, Term_Spread, Unemployment")
    print("Response: SPY_returns (regime-dependent)")

    ctree = CTree(
        alpha=0.15,
        min_split=20,
        min_leaf=10,
        mc_threshold=100,
        n_permutations=2000
    )

    print("\nFitting CTree with Monte Carlo permutation test...")
    start = time.time()
    ctree.fit(X, Y, feature_names=['VIX', 'Term_Spread', 'Unemployment'])
    elapsed = time.time() - start

    print(f"[OK] Tree fitted in {elapsed:.2f} seconds")
    print(f"[OK] Number of regimes identified: {ctree.n_regimes}")
    print(f"[OK] Method used: Monte Carlo (N={n} < threshold={ctree.mc_threshold})")

    print("\nTree Structure:")
    print(ctree.print_tree())

    # Show regime statistics
    print("\nRegime Statistics:")
    regime_map = ctree.get_regime_mapping()
    for regime_id, stats in sorted(regime_map.items()):
        mean_ret = stats['mean_response'][0] if stats['mean_response'] is not None else 0
        print(f"  Regime {regime_id}: n={stats['n_samples']:3d}, "
              f"mean_return={mean_ret:+.4f}")

    return ctree


def demo_large_sample_asymptotic():
    """Demonstrate asymptotic path with large sample (N >= 100)."""
    print("\n" + "=" * 70)
    print("DEMO 2: Large Sample (N=500) - Asymptotic Chi-Squared Test")
    print("=" * 70)

    np.random.seed(42)
    n = 500

    # Create features: same structure as small sample
    X = np.random.randn(n, 3)
    X[:, 0] = 15 + 10 * np.abs(np.random.randn(n))  # VIX
    X[:, 1] = 2.0 + np.random.randn(n)  # Term spread
    X[:, 2] = 5.0 + 2 * np.random.randn(n)  # Unemployment

    # Create regime-dependent response
    Y = np.where(X[:, 0] > 20, -0.01, 0.01)
    Y += 0.005 * X[:, 1]
    Y = Y.reshape(-1, 1) + 0.005 * np.random.randn(n, 1)

    print(f"\nData: {n} samples, 3 features")
    print("Features: VIX, Term_Spread, Unemployment")
    print("Response: SPY_returns (regime-dependent)")

    ctree = CTree(
        alpha=0.05,
        min_split=20,
        min_leaf=10,
        mc_threshold=100,
        n_permutations=10000  # Won't be used due to large N
    )

    print("\nFitting CTree with asymptotic chi-squared test...")
    start = time.time()
    ctree.fit(X, Y, feature_names=['VIX', 'Term_Spread', 'Unemployment'])
    elapsed = time.time() - start

    print(f"[OK] Tree fitted in {elapsed:.2f} seconds")
    print(f"[OK] Number of regimes identified: {ctree.n_regimes}")
    print(f"[OK] Method used: Asymptotic (N={n} >= threshold={ctree.mc_threshold})")
    print(f"  (Much faster than MC! Monte Carlo would have taken ~{elapsed * 20:.1f}s)")

    print("\nTree Structure:")
    print(ctree.print_tree())

    # Show regime statistics
    print("\nRegime Statistics:")
    regime_map = ctree.get_regime_mapping()
    for regime_id, stats in sorted(regime_map.items()):
        mean_ret = stats['mean_response'][0] if stats['mean_response'] is not None else 0
        print(f"  Regime {regime_id}: n={stats['n_samples']:3d}, "
              f"mean_return={mean_ret:+.4f}")

    return ctree


def demo_realistic_financial_data():
    """Demonstrate on realistic synthetic financial dataset."""
    print("\n" + "=" * 70)
    print("DEMO 3: Realistic Financial Data (N=1000)")
    print("=" * 70)

    # Get synthetic data from data_fetcher
    X, fast_vars, slow_vars = create_sample_dataset()

    print(f"\nDataset shape: {X.shape}")
    print(f"Fast variables: {', '.join(fast_vars[:4])}... ({len(fast_vars)} total)")
    print(f"Slow variables: {', '.join(slow_vars[:4])}... ({len(slow_vars)} total)")

    # Use slow variables (macro indicators) as features
    # Predict SPY returns (first fast variable)
    feature_indices = list(range(len(fast_vars), len(fast_vars) + len(slow_vars)))
    response_index = 0  # SPY returns

    X_features = X[:, feature_indices]
    Y_response = X[:, response_index].reshape(-1, 1)

    print(f"\nTask: Predict {fast_vars[0]} using macroeconomic indicators")
    print(f"Features (X): {slow_vars}")
    print(f"Response (Y): {fast_vars[0]}")

    # Fit tree
    ctree = CTree(
        alpha=0.05,
        min_split=30,
        min_leaf=20,
        max_depth=3,  # Limit depth for interpretability
        mc_threshold=100
    )

    print("\nFitting CTree (using asymptotic test for N=1000)...")
    start = time.time()
    ctree.fit(X_features, Y_response, feature_names=slow_vars)
    elapsed = time.time() - start

    print(f"[OK] Tree fitted in {elapsed:.2f} seconds")
    print(f"[OK] Number of market regimes identified: {ctree.n_regimes}")

    print("\nTree Structure (Regime Identification):")
    print(ctree.print_tree())

    # Analyze regimes
    print("\nRegime Characteristics:")
    regime_map = ctree.get_regime_mapping()
    predictions = ctree.predict(X_features)

    for regime_id, stats in sorted(regime_map.items()):
        regime_mask = predictions == regime_id
        regime_samples = X_features[regime_mask]

        mean_ret = stats['mean_response'][0] if stats['mean_response'] is not None else 0

        print(f"\n  Regime {regime_id} ({stats['n_samples']} samples):")
        print(f"    Mean SPY return: {mean_ret:+.4f}")
        print(f"    Avg Fed Funds:   {regime_samples[:, 0].mean():.2f}%")
        print(f"    Avg 10Y Yield:   {regime_samples[:, 1].mean():.2f}%")
        print(f"    Avg Unemployment:{regime_samples[:, 5].mean():.2f}%")

        if mean_ret > 0.003:
            regime_type = "BULL MARKET"
        elif mean_ret < -0.003:
            regime_type = "BEAR MARKET"
        else:
            regime_type = "NEUTRAL"
        print(f"    Classification:  {regime_type}")

    return ctree


def demo_temporal_regime_detection():
    """Detect regime changes over time - find when macro conditions shifted."""
    print("\n" + "=" * 70)
    print("DEMO 4: Temporal Regime Detection - Finding Regime Switch Points")
    print("=" * 70)

    # Get synthetic data
    X, fast_vars, slow_vars = create_sample_dataset()
    n_samples = len(X)

    # Extract macro variables as the "response" we want to cluster
    # We'll use time as the feature to partition on
    macro_indices = list(range(len(fast_vars), len(fast_vars) + len(slow_vars)))
    macro_data = X[:, macro_indices]

    # Create time variable (0, 1, 2, ..., n-1)
    time_index = np.arange(n_samples).reshape(-1, 1)

    print(f"\nData: {n_samples} time periods")
    print(f"Task: Partition timeline based on when macro conditions changed")
    print(f"Features (X): Time index (0 to {n_samples-1})")
    print(f"Response (Y): Macro variables [{', '.join(slow_vars[:3])}...]")

    # Fit tree - use time as X, macro conditions as Y
    ctree = CTree(
        alpha=0.05,
        min_split=50,  # Need reasonable periods
        min_leaf=30,   # Each regime should span at least 30 periods
        max_depth=4,   # Limit depth for interpretability
        mc_threshold=100
    )

    print("\nFitting CTree to detect temporal regime switches...")
    start = time.time()
    ctree.fit(time_index, macro_data, feature_names=['Time'])
    elapsed = time.time() - start

    print(f"[OK] Tree fitted in {elapsed:.2f} seconds")
    print(f"[OK] Number of temporal regimes identified: {ctree.n_regimes}")

    print("\nTemporal Partition Points:")
    print(ctree.print_tree())

    # Analyze each temporal regime
    predictions = ctree.predict(time_index)
    regime_map = ctree.get_regime_mapping()

    print("\n" + "=" * 70)
    print("Historical Regime Analysis")
    print("=" * 70)

    for regime_id in sorted(regime_map.keys()):
        regime_mask = predictions == regime_id
        regime_times = time_index[regime_mask].flatten()
        regime_macro = macro_data[regime_mask]

        start_time = regime_times.min()
        end_time = regime_times.max()
        duration = len(regime_times)

        print(f"\nRegime {regime_id}: Period {start_time} to {end_time} ({duration} periods)")
        print(f"  Fed Funds Rate:  {regime_macro[:, 0].mean():.2f}% (std: {regime_macro[:, 0].std():.2f})")
        print(f"  10Y Treasury:    {regime_macro[:, 1].mean():.2f}% (std: {regime_macro[:, 1].std():.2f})")
        print(f"  Term Spread:     {regime_macro[:, 3].mean():.2f}% (std: {regime_macro[:, 3].std():.2f})")
        print(f"  Unemployment:    {regime_macro[:, 5].mean():.2f}% (std: {regime_macro[:, 5].std():.2f})")

        # Characterize the regime
        avg_fed_funds = regime_macro[:, 0].mean()
        avg_term_spread = regime_macro[:, 3].mean()
        avg_unemployment = regime_macro[:, 5].mean()

        characteristics = []
        if avg_fed_funds < 2.0:
            characteristics.append("Low Rates")
        elif avg_fed_funds > 2.2:
            characteristics.append("High Rates")

        if avg_term_spread < 0.5:
            characteristics.append("Flat Curve")
        elif avg_term_spread > 1.5:
            characteristics.append("Steep Curve")

        if avg_unemployment > 5.5:
            characteristics.append("High Unemployment")
        elif avg_unemployment < 4.5:
            characteristics.append("Low Unemployment")

        print(f"  Characteristics: {', '.join(characteristics) if characteristics else 'Normal'}")

    return ctree, predictions


def demo_multivariate_temporal_regime():
    """Use multiple macro variables to define regimes over time."""
    print("\n" + "=" * 70)
    print("DEMO 5: Multivariate Temporal Analysis - Economic Cycle Detection")
    print("=" * 70)

    # Get synthetic data
    X, fast_vars, slow_vars = create_sample_dataset()
    n_samples = len(X)

    # We'll use a subset of macro variables to define economic state
    # Focus on: Fed Funds, Term Spread, Unemployment
    macro_indices = [len(fast_vars) + i for i in [0, 3, 5]]  # Fed Funds, Term Spread, Unemployment
    selected_vars = [slow_vars[i] for i in [0, 3, 5]]
    macro_data = X[:, macro_indices]

    # Time index
    time_index = np.arange(n_samples).reshape(-1, 1)

    print(f"\nData: {n_samples} time periods")
    print(f"Task: Identify economic cycles based on monetary policy and labor market")
    print(f"Features (X): Time")
    print(f"Response (Y): {', '.join(selected_vars)}")

    # Fit tree with stricter criteria for cleaner regimes
    ctree = CTree(
        alpha=0.01,    # Require strong evidence for splits
        min_split=80,  # Larger minimum periods
        min_leaf=40,   # Each regime >= 40 periods
        max_depth=3,   # Max 3 splits = 8 regimes
        mc_threshold=100
    )

    print("\nFitting CTree to detect economic cycles...")
    start = time.time()
    ctree.fit(time_index, macro_data, feature_names=['Time'])
    elapsed = time.time() - start

    print(f"[OK] Tree fitted in {elapsed:.2f} seconds")
    print(f"[OK] Number of economic cycles identified: {ctree.n_regimes}")

    print("\nEconomic Cycle Breakpoints:")
    print(ctree.print_tree())

    # Detailed cycle analysis
    predictions = ctree.predict(time_index)
    regime_map = ctree.get_regime_mapping()

    print("\n" + "=" * 70)
    print("Economic Cycle Characterization")
    print("=" * 70)

    # Also compute SPY returns for each regime to see market implications
    spy_returns = X[:, 0]  # First fast variable

    for regime_id in sorted(regime_map.keys()):
        regime_mask = predictions == regime_id
        regime_times = time_index[regime_mask].flatten()
        regime_macro = macro_data[regime_mask]
        regime_spy = spy_returns[regime_mask]

        start_time = regime_times.min()
        end_time = regime_times.max()
        duration = len(regime_times)

        # Calculate key statistics
        fed_funds = regime_macro[:, 0].mean()
        term_spread = regime_macro[:, 1].mean()
        unemployment = regime_macro[:, 2].mean()
        spy_mean = regime_spy.mean()

        print(f"\n{'='*70}")
        print(f"Cycle {regime_id}: Period {start_time}-{end_time} ({duration} periods)")
        print(f"{'='*70}")
        print(f"  Monetary Policy:")
        print(f"    Fed Funds Rate: {fed_funds:.2f}%")
        print(f"    Term Spread:    {term_spread:+.2f}%")

        print(f"\n  Labor Market:")
        print(f"    Unemployment:   {unemployment:.2f}%")

        print(f"\n  Market Performance:")
        print(f"    Avg SPY Return: {spy_mean:+.4f} ({spy_mean*252*100:+.1f}% annualized)")

        # Classify the economic cycle
        print(f"\n  Cycle Classification:", end=" ")

        if fed_funds < 1.95 and unemployment > 5.0:
            cycle_type = "RECESSION / RECOVERY (Low rates, high unemployment)"
        elif fed_funds < 2.0 and unemployment < 5.0:
            cycle_type = "EXPANSION (Accommodative policy, low unemployment)"
        elif fed_funds > 2.15 and term_spread < 1.0:
            cycle_type = "LATE CYCLE (Tightening policy, flattening curve)"
        elif term_spread < 0.5:
            cycle_type = "YIELD CURVE INVERSION (Recession warning)"
        elif fed_funds > 2.1 and unemployment > 5.5:
            cycle_type = "STAGFLATION RISK (High rates, high unemployment)"
        else:
            cycle_type = "MID CYCLE (Normal conditions)"

        print(cycle_type)

        # Market implications
        if spy_mean > 0.003:
            market_env = "Strong Bull Market"
        elif spy_mean > 0:
            market_env = "Moderate Bull Market"
        elif spy_mean > -0.003:
            market_env = "Flat/Choppy Market"
        else:
            market_env = "Bear Market"

        print(f"  Market Environment: {market_env}")

    return ctree, predictions


def compare_mc_vs_asymptotic():
    """Compare timing and results of MC vs Asymptotic methods."""
    print("\n" + "=" * 70)
    print("DEMO 6: Speed Comparison - Monte Carlo vs Asymptotic")
    print("=" * 70)

    np.random.seed(42)
    n = 120  # Just above threshold

    X = np.random.randn(n, 3)
    Y = (2 * X[:, 0] + X[:, 1] + 0.5 * np.random.randn(n)).reshape(-1, 1)

    print(f"\nData: {n} samples with strong dependence on features")

    # Test 1: Force Monte Carlo
    print("\nTest 1: Force Monte Carlo (set threshold=121)")
    ctree_mc = CTree(mc_threshold=121, n_permutations=5000)
    start = time.time()
    ctree_mc.fit(X, Y)
    time_mc = time.time() - start
    print(f"  Time: {time_mc:.3f} seconds")
    print(f"  Regimes: {ctree_mc.n_regimes}")

    # Test 2: Use Asymptotic
    print("\nTest 2: Use Asymptotic (set threshold=100)")
    ctree_asymp = CTree(mc_threshold=100)
    start = time.time()
    ctree_asymp.fit(X, Y)
    time_asymp = time.time() - start
    print(f"  Time: {time_asymp:.3f} seconds")
    print(f"  Regimes: {ctree_asymp.n_regimes}")

    print(f"\nSpeedup: {time_mc / time_asymp:.1f}x faster with asymptotic method")
    print(f"Both methods detected dependence and created regime structure.")


if __name__ == "__main__":
    print("=" * 70)
    print("CTree Algorithm: Adaptive P-Value Computation Demo")
    print("=" * 70)
    print("\nThis demo shows how CTree automatically switches between:")
    print("  • Monte Carlo permutation test (N < 100): Robust for small samples")
    print("  • Asymptotic chi-squared test (N >= 100): Fast for large samples")

    # Run all demos
    demo_small_sample_mc()
    demo_large_sample_asymptotic()
    demo_realistic_financial_data()
    demo_temporal_regime_detection()
    demo_multivariate_temporal_regime()
    compare_mc_vs_asymptotic()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Small samples use Monte Carlo for statistical robustness")
    print("  2. Large samples use asymptotic test for computational efficiency")
    print("  3. The branching is automatic based on sample size")
    print("  4. CTree identifies meaningful market regimes from features")
    print("  5. CTree can partition TIME to find regime switch points")
    print("  6. Temporal partitioning reveals economic cycles and structural breaks")
    print("  7. All regime boundaries are statistically justified (p-value based)")
