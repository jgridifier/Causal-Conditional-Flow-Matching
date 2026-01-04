"""
Standalone test script for non-Gaussianity verification.

This script demonstrates the new non-Gaussianity checking functionality
added to verify LiNGAM identifiability assumptions.
"""

import numpy as np
import sys
import warnings

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path
sys.path.insert(0, '.')

from core.etl import DataProcessor


def test_gaussian_detection():
    """Test detection of Gaussian data."""
    print("\n" + "="*60)
    print("TEST 1: Gaussian Data Detection")
    print("="*60)

    np.random.seed(42)

    # Generate purely Gaussian data
    print("\nGenerating 1000 samples of 3 Gaussian variables...")
    X = np.random.randn(1000, 3)

    processor = DataProcessor()
    all_ng, gaussian_idx, p_vals = processor._test_non_gaussianity(X, alpha=0.05)

    print(f"\nResults:")
    print(f"  All non-Gaussian: {all_ng}")
    print(f"  Gaussian variable indices: {gaussian_idx}")
    print(f"  P-values: {[f'{p:.4f}' for p in p_vals]}")

    if not all_ng:
        print(f"\n✓ PASS: Correctly detected {len(gaussian_idx)} Gaussian variables")
    else:
        print(f"\n✗ FAIL: Should have detected Gaussian variables")

    return not all_ng


def test_non_gaussian_detection():
    """Test detection of non-Gaussian data."""
    print("\n" + "="*60)
    print("TEST 2: Non-Gaussian Data Detection (Heavy-Tailed)")
    print("="*60)

    np.random.seed(42)

    # Generate heavy-tailed Student-t data
    print("\nGenerating 1000 samples from Student-t(df=3) distribution...")
    from scipy.stats import t as student_t
    X = student_t.rvs(df=3, size=(1000, 3))

    processor = DataProcessor()
    all_ng, gaussian_idx, p_vals = processor._test_non_gaussianity(X, alpha=0.05)

    print(f"\nResults:")
    print(f"  All non-Gaussian: {all_ng}")
    print(f"  Gaussian variable indices: {gaussian_idx}")
    print(f"  P-values: {[f'{p:.4e}' for p in p_vals]}")

    if len(gaussian_idx) < 3:
        print(f"\n✓ PASS: Correctly identified non-Gaussian structure")
    else:
        print(f"\n✗ FAIL: Should have detected non-Gaussianity")

    return len(gaussian_idx) < 3


def test_warning_integration():
    """Test that warning is issued during LiNGAM fitting."""
    print("\n" + "="*60)
    print("TEST 3: Warning Integration with LiNGAM")
    print("="*60)

    np.random.seed(42)

    # Create Gaussian data (will trigger warning)
    print("\nGenerating 100 samples of Gaussian data...")
    n = 100
    X = np.random.randn(n, 6)

    processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=10)

    print("\nFitting data processor (should trigger warning)...")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        topology = processor.fit_transform(
            X,
            fast_vars=[0, 1, 2],
            slow_vars=[3, 4, 5]
        )

        # Check if warning was raised
        lingam_warnings = [warning for warning in w
                          if "LiNGAM Identifiability" in str(warning.message)]

        if lingam_warnings:
            print(f"\n✓ PASS: Warning raised successfully")
            print(f"\nWarning message:")
            print(f"  {lingam_warnings[0].message}")
        else:
            print(f"\n✗ FAIL: Expected warning not raised")
            print(f"\nAll warnings raised: {[str(warning.message) for warning in w]}")

        return len(lingam_warnings) > 0


def test_mixed_data():
    """Test with mixed Gaussian and non-Gaussian data."""
    print("\n" + "="*60)
    print("TEST 4: Mixed Gaussian/Non-Gaussian Data")
    print("="*60)

    np.random.seed(42)

    # Create mixed data
    print("\nGenerating mixed data:")
    print("  - Variables 0-1: Gaussian")
    print("  - Variables 2-3: Non-Gaussian (exponential)")

    n = 1000
    X = np.zeros((n, 4))

    # Gaussian variables
    X[:, 0] = np.random.randn(n)
    X[:, 1] = np.random.randn(n)

    # Non-Gaussian variables
    X[:, 2] = np.random.exponential(1.0, n)
    X[:, 3] = np.random.choice([-1, 1], n) * np.random.exponential(1.0, n)

    processor = DataProcessor()
    all_ng, gaussian_idx, p_vals = processor._test_non_gaussianity(X, alpha=0.05)

    print(f"\nResults:")
    print(f"  All non-Gaussian: {all_ng}")
    print(f"  Gaussian variable indices: {gaussian_idx}")
    print(f"  P-values: {[f'{p:.4e}' for p in p_vals]}")

    if not all_ng and len(gaussian_idx) >= 1:
        print(f"\n✓ PASS: Correctly identified mixed distribution")
    else:
        print(f"\n✗ FAIL: Should have detected mixed Gaussian/non-Gaussian")

    return not all_ng and len(gaussian_idx) >= 1


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Non-Gaussianity Verification Test Suite")
    print("#"*60)

    results = []

    try:
        results.append(("Gaussian Detection", test_gaussian_detection()))
    except Exception as e:
        print(f"\n✗ ERROR in Gaussian Detection: {e}")
        results.append(("Gaussian Detection", False))

    try:
        results.append(("Non-Gaussian Detection", test_non_gaussian_detection()))
    except Exception as e:
        print(f"\n✗ ERROR in Non-Gaussian Detection: {e}")
        results.append(("Non-Gaussian Detection", False))

    try:
        results.append(("Warning Integration", test_warning_integration()))
    except Exception as e:
        print(f"\n✗ ERROR in Warning Integration: {e}")
        results.append(("Warning Integration", False))

    try:
        results.append(("Mixed Data", test_mixed_data()))
    except Exception as e:
        print(f"\n✗ ERROR in Mixed Data: {e}")
        results.append(("Mixed Data", False))

    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
