"""
Demo: Non-Gaussianity Verification for LiNGAM

This example demonstrates the new non-Gaussianity checking feature
that verifies LiNGAM identifiability assumptions before causal discovery.

LiNGAM (Linear Non-Gaussian Acyclic Model) requires non-Gaussian error
distributions to uniquely identify causal directions. This feature warns
users when this assumption is violated.
"""

import numpy as np
import sys
import warnings

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, '..')

from core.etl import DataProcessor


def example_1_gaussian_warning():
    """Example 1: Warning triggered for Gaussian data."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Gaussian Data (Warning Triggered)")
    print("="*70)

    print("\n📊 Scenario: Financial data with Gaussian-like distributions")
    print("   (e.g., highly diversified portfolio returns)\n")

    # Generate Gaussian data that might occur in practice
    np.random.seed(42)
    n_samples = 200
    n_vars = 6

    # Simulate Gaussian market data
    X = np.random.randn(n_samples, n_vars) * 0.02  # 2% daily volatility

    print(f"Generated {n_samples} samples of {n_vars} variables")
    print(f"Variable names: ['SPX', 'VIX', 'DXY', 'GDP', 'CPI', 'UNEMP']")

    processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=20)

    print("\n⚙️  Fitting data processor...")
    print("   (Watch for LiNGAM identifiability warning)\n")

    # Capture and display warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        topology = processor.fit_transform(
            X,
            fast_vars=[0, 1, 2],  # SPX, VIX, DXY
            slow_vars=[3, 4, 5]   # GDP, CPI, UNEMP
        )

        # Display warnings
        for warning in w:
            if "LiNGAM" in str(warning.message):
                print("⚠️  WARNING RAISED:")
                print(f"   {warning.message}\n")

    print("✓ Processing completed despite warning")
    print(f"  Final dimensions: {topology.X_processed.shape}")
    print(f"  Causal order established (may be unreliable due to Gaussianity)")


def example_2_non_gaussian_success():
    """Example 2: Non-Gaussian data passes verification."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Non-Gaussian Data (Verification Passed)")
    print("="*70)

    print("\n📊 Scenario: Financial returns with fat tails and skewness")
    print("   (typical real-world financial data)\n")

    np.random.seed(42)
    n_samples = 200
    n_vars = 6

    # Simulate realistic fat-tailed financial data using Student-t
    from scipy.stats import t as student_t
    X = student_t.rvs(df=5, size=(n_samples, n_vars)) * 0.02

    print(f"Generated {n_samples} samples from Student-t(df=5) distribution")
    print(f"Variable names: ['SPX_ret', 'VIX_chg', 'Credit_spread', 'GDP_growth', 'CPI_yoy', 'UnempRate']")

    processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=20)

    print("\n⚙️  Fitting data processor...")
    print("   (Should NOT trigger warning - data is non-Gaussian)\n")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        topology = processor.fit_transform(
            X,
            fast_vars=[0, 1, 2],    # Returns, VIX, Spreads
            slow_vars=[3, 4, 5]     # Macro variables
        )

        # Check for LiNGAM warnings
        lingam_warnings = [warning for warning in w if "LiNGAM" in str(warning.message)]

        if not lingam_warnings:
            print("✓ NO WARNING: Data passed non-Gaussianity verification")
        else:
            print("⚠️  Unexpected warning (data should be non-Gaussian)")

    print(f"\n✓ Processing completed successfully")
    print(f"  Final dimensions: {topology.X_processed.shape}")
    print(f"  Causal order: {topology.causal_order}")
    print(f"  Number of regimes: {topology.n_regimes}")


def example_3_manual_check():
    """Example 3: Manually check non-Gaussianity of your data."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Manual Non-Gaussianity Check")
    print("="*70)

    print("\n📊 Scenario: Check your data before fitting\n")

    # Create mixed data
    np.random.seed(42)
    n = 500
    X = np.zeros((n, 4))

    # Variable 0: Gaussian (problematic!)
    X[:, 0] = np.random.randn(n)

    # Variable 1: Gaussian (problematic!)
    X[:, 1] = np.random.randn(n)

    # Variable 2: Non-Gaussian (good!)
    X[:, 2] = np.random.exponential(1.0, n)

    # Variable 3: Non-Gaussian (good!)
    from scipy.stats import t as student_t
    X[:, 3] = student_t.rvs(df=3, size=n)

    processor = DataProcessor()

    print("Running Jarque-Bera test on each variable...")
    all_ng, gaussian_idx, p_vals = processor._test_non_gaussianity(X, alpha=0.05)

    print(f"\nResults:")
    print(f"  All non-Gaussian: {all_ng}")
    print(f"\n  Per-variable analysis:")

    var_names = ['Var_0', 'Var_1', 'Var_2', 'Var_3']
    for i, (name, p_val) in enumerate(zip(var_names, p_vals)):
        status = "❌ Gaussian" if i in gaussian_idx else "✅ Non-Gaussian"
        print(f"    {name}: {status} (p={p_val:.4f})")

    print(f"\n  Variables requiring attention: {[var_names[i] for i in gaussian_idx]}")

    if gaussian_idx:
        print("\n💡 Recommendations for Gaussian variables:")
        print("   1. Apply transformations: log, Box-Cox, square root")
        print("   2. Use returns instead of levels")
        print("   3. Consider variance-based ordering if transformations don't help")
        print("   4. Remove or exclude from causal discovery")


def example_4_comparison():
    """Example 4: Compare Gaussian vs Non-Gaussian data."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Side-by-Side Comparison")
    print("="*70)

    np.random.seed(42)
    n = 1000

    # Gaussian data
    X_gaussian = np.random.randn(n, 3)

    # Non-Gaussian data (mixture of normals - fat tails)
    X_nongaussian = np.zeros((n, 3))
    for j in range(3):
        # 90% normal, 10% 5x volatility (simulates crisis events)
        regime = np.random.rand(n) < 0.9
        X_nongaussian[regime, j] = np.random.randn(regime.sum()) * 1.0
        X_nongaussian[~regime, j] = np.random.randn((~regime).sum()) * 5.0

    processor = DataProcessor()

    print("\n📊 Testing Gaussian Data:")
    all_ng_g, gauss_idx_g, p_vals_g = processor._test_non_gaussianity(X_gaussian)
    print(f"   Non-Gaussian variables: {3 - len(gauss_idx_g)}/3")
    print(f"   Gaussian variables: {len(gauss_idx_g)}/3")
    print(f"   Mean p-value: {np.mean(p_vals_g):.4f}")
    print(f"   → {'✓ PASSES' if all_ng_g else '✗ FAILS'} for LiNGAM")

    print("\n📊 Testing Non-Gaussian Data (Fat-Tailed):")
    all_ng_ng, gauss_idx_ng, p_vals_ng = processor._test_non_gaussianity(X_nongaussian)
    print(f"   Non-Gaussian variables: {3 - len(gauss_idx_ng)}/3")
    print(f"   Gaussian variables: {len(gauss_idx_ng)}/3")
    print(f"   Mean p-value: {np.mean(p_vals_ng):.4e}")
    print(f"   → {'✓ PASSES' if all_ng_ng else '✗ FAILS'} for LiNGAM")

    print("\n💡 Key Insight:")
    print("   Financial data with crisis events (fat tails) is typically")
    print("   non-Gaussian and suitable for LiNGAM causal discovery.")


def main():
    """Run all examples."""
    print("\n" + "#"*70)
    print("# Non-Gaussianity Verification for LiNGAM - Demonstrations")
    print("#"*70)
    print("\nThis demo shows the new feature that verifies whether your data")
    print("satisfies the non-Gaussianity assumption required for LiNGAM.")

    try:
        example_1_gaussian_warning()
    except Exception as e:
        print(f"\n❌ Error in Example 1: {e}")

    try:
        example_2_non_gaussian_success()
    except Exception as e:
        print(f"\n❌ Error in Example 2: {e}")

    try:
        example_3_manual_check()
    except Exception as e:
        print(f"\n❌ Error in Example 3: {e}")

    try:
        example_4_comparison()
    except Exception as e:
        print(f"\n❌ Error in Example 4: {e}")

    print("\n" + "#"*70)
    print("# Summary")
    print("#"*70)
    print("\n✅ Feature Benefits:")
    print("   1. Automatic verification of LiNGAM assumptions")
    print("   2. Clear warnings when assumptions are violated")
    print("   3. Actionable recommendations for data transformation")
    print("   4. Prevents unreliable causal inference")
    print("\n📚 For more information, see:")
    print("   - Shimizu et al. (2006): LiNGAM paper")
    print("   - Jarque & Bera (1980): Normality test paper")
    print()


if __name__ == "__main__":
    main()
