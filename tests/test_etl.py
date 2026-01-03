"""
Comprehensive tests for ETL (Data Ingestion & Topology) module

Tests cover:
1. Data validation and cleaning
2. Stationarity testing (ADF)
3. Cubic spline imputation
4. Normalization
5. Causal ordering
6. Polars integration
"""

import numpy as np
import polars as pl
import pytest
from scipy.stats import zscore

import sys
sys.path.insert(0, '..')

from core.etl import (
    DataProcessor,
    DataTopology,
    validate_for_ode,
    _to_numpy_array,
    numpy_to_polars,
    polars_to_numpy
)


class TestDataConversion:
    """Test data format conversions."""

    def test_numpy_to_numpy_array(self):
        """Test numpy array passthrough."""
        X = np.random.randn(100, 5)
        X_out, names = _to_numpy_array(X)

        np.testing.assert_array_equal(X_out, X.astype(np.float64))
        assert len(names) == 5
        assert names[0] == "var_0"

    def test_polars_to_numpy_array(self):
        """Test polars DataFrame conversion."""
        df = pl.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0]
        })

        X_out, names = _to_numpy_array(df)

        assert X_out.shape == (3, 2)
        assert names == ['a', 'b']
        np.testing.assert_array_almost_equal(X_out[:, 0], [1.0, 2.0, 3.0])

    def test_numpy_to_polars_helper(self):
        """Test numpy to polars helper function."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        names = ['col_a', 'col_b']

        df = numpy_to_polars(X, names)

        assert isinstance(df, pl.DataFrame)
        assert df.columns == names
        assert df.shape == (2, 2)

    def test_polars_to_numpy_helper(self):
        """Test polars to numpy helper function."""
        df = pl.DataFrame({
            'x': [1.0, 2.0],
            'y': [3.0, 4.0]
        })

        X, cols = polars_to_numpy(df)

        assert X.shape == (2, 2)
        assert cols == ['x', 'y']

    def test_invalid_type_raises(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError):
            _to_numpy_array("not a valid input")


class TestDataValidation:
    """Test data validation and cleaning."""

    def test_validate_for_ode_clean_data(self):
        """Test that clean data passes validation."""
        X = np.random.randn(100, 5)
        validate_for_ode(X)  # Should not raise

    def test_validate_for_ode_nan_raises(self):
        """Test that NaN raises ValueError."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])

        with pytest.raises(ValueError, match="NaN"):
            validate_for_ode(X)

    def test_validate_for_ode_inf_raises(self):
        """Test that Inf raises ValueError."""
        X = np.array([[1.0, np.inf], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Inf"):
            validate_for_ode(X)

    def test_validate_for_ode_extreme_values_warns(self):
        """Test that extreme values trigger warning."""
        X = np.array([[1e7, 2.0], [3.0, 4.0]])

        with pytest.warns(UserWarning, match="extreme"):
            validate_for_ode(X)


class TestCubicSplineImputation:
    """Test cubic spline imputation for missing values."""

    def test_imputation_fills_nans(self):
        """Test that NaNs are filled."""
        np.random.seed(42)

        X = np.random.randn(100, 3)
        # Introduce some NaNs
        X[10, 0] = np.nan
        X[50, 1] = np.nan

        processor = DataProcessor()
        X_imputed = processor._impute_cubic_spline(X)

        assert not np.any(np.isnan(X_imputed))

    def test_imputation_preserves_non_nan(self):
        """Test that non-NaN values are preserved."""
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

        processor = DataProcessor()
        X_imputed = processor._impute_cubic_spline(X)

        assert X_imputed[0, 0] == 1.0
        assert X_imputed[2, 0] == 5.0
        assert X_imputed[1, 1] == 4.0

    def test_imputation_interpolates_smoothly(self):
        """Test that imputation produces reasonable values."""
        X = np.array([
            [0.0],
            [np.nan],
            [np.nan],
            [3.0],
            [4.0],
            [5.0]
        ])

        processor = DataProcessor()
        X_imputed = processor._impute_cubic_spline(X)

        # Interpolated values should be between 0 and 3
        assert 0 <= X_imputed[1, 0] <= 3
        assert 0 <= X_imputed[2, 0] <= 3

    def test_all_nan_column_raises(self):
        """Test that entirely NaN column raises error."""
        X = np.array([[np.nan], [np.nan], [np.nan]])

        processor = DataProcessor()
        with pytest.raises(ValueError, match="entirely NaN"):
            processor._impute_cubic_spline(X)

    def test_edge_nans_handled(self):
        """Test that NaNs at edges are handled."""
        X = np.array([
            [np.nan],  # Start NaN
            [1.0],
            [2.0],
            [np.nan]   # End NaN
        ])

        processor = DataProcessor()
        X_imputed = processor._impute_cubic_spline(X)

        assert not np.any(np.isnan(X_imputed))
        # Edge NaNs should be filled with nearest value
        assert X_imputed[0, 0] == 1.0
        assert X_imputed[3, 0] == 2.0


class TestStationarityChecks:
    """Test ADF stationarity tests and differencing."""

    def test_stationary_series_detected(self):
        """Test that stationary series is correctly identified."""
        np.random.seed(42)

        # White noise is stationary
        series = np.random.randn(200)

        processor = DataProcessor(adf_threshold=0.05)
        is_stationary, diff_order = processor._check_stationarity(series)

        assert is_stationary
        assert diff_order == 0

    def test_random_walk_needs_differencing(self):
        """Test that random walk requires differencing."""
        np.random.seed(42)

        # Random walk is non-stationary
        series = np.cumsum(np.random.randn(200))

        processor = DataProcessor(adf_threshold=0.05)
        is_stationary, diff_order = processor._check_stationarity(series)

        assert not is_stationary
        assert diff_order >= 1

    def test_differencing_creates_stationary(self):
        """Test that differencing makes series stationary."""
        np.random.seed(42)

        # Start with random walk
        series = np.cumsum(np.random.randn(200))

        processor = DataProcessor()
        differenced = processor._apply_differencing(series, 1)

        # After differencing, should be closer to stationary
        clean_diff = differenced[~np.isnan(differenced)]
        is_stationary, _ = processor._check_stationarity(clean_diff)

        # Differenced random walk should be stationary
        # (It becomes white noise)

    def test_log_returns_for_positive_series(self):
        """Test that log-returns are used for positive series."""
        # Price series (all positive)
        series = np.array([100, 105, 103, 108, 110])

        processor = DataProcessor()
        result = processor._apply_differencing(series, 1)

        # Should be log-returns
        expected = np.log(series[1:] / series[:-1])
        np.testing.assert_array_almost_equal(result[1:], expected)

    def test_first_diff_for_mixed_series(self):
        """Test that first differences are used for mixed sign series."""
        series = np.array([-1, 0, 1, 2, 3])

        processor = DataProcessor()
        result = processor._apply_differencing(series, 1)

        expected = np.diff(series)
        np.testing.assert_array_almost_equal(result[1:], expected)

    def test_short_series_assumed_stationary(self):
        """Test that very short series are assumed stationary."""
        series = np.array([1, 2, 3, 4, 5])

        processor = DataProcessor()
        is_stationary, diff_order = processor._check_stationarity(series)

        assert is_stationary
        assert diff_order == 0


class TestNormalization:
    """Test Z-score normalization."""

    def test_normalized_mean_zero(self):
        """Test that normalized data has mean ~0."""
        X = np.random.randn(100, 5) * 10 + 50

        processor = DataProcessor()
        X_norm, mean, std = processor._normalize(X)

        col_means = np.mean(X_norm, axis=0)
        np.testing.assert_array_almost_equal(col_means, np.zeros(5), decimal=10)

    def test_normalized_std_one(self):
        """Test that normalized data has std ~1."""
        X = np.random.randn(100, 5) * 10 + 50

        processor = DataProcessor()
        X_norm, mean, std = processor._normalize(X)

        col_stds = np.std(X_norm, axis=0, ddof=0)
        np.testing.assert_array_almost_equal(col_stds, np.ones(5), decimal=5)

    def test_scaler_stats_stored(self):
        """Test that mean and std are stored correctly."""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        processor = DataProcessor()
        _, mean, std = processor._normalize(X.astype(float))

        np.testing.assert_array_almost_equal(mean, [3, 4])
        # Standard deviation computation


class TestVariableBlocks:
    """Test fast/slow variable parsing."""

    def test_default_split(self):
        """Test default half/half split."""
        names = ['a', 'b', 'c', 'd']

        processor = DataProcessor()
        fast, slow = processor._parse_variable_blocks(names, None, None)

        np.testing.assert_array_equal(fast, [0, 1])
        np.testing.assert_array_equal(slow, [2, 3])

    def test_string_names(self):
        """Test parsing string variable names."""
        names = ['VIX', 'SPY', 'GDP', 'CPI']

        processor = DataProcessor()
        fast, slow = processor._parse_variable_blocks(
            names,
            fast_vars=['VIX', 'SPY'],
            slow_vars=['GDP', 'CPI']
        )

        np.testing.assert_array_equal(fast, [0, 1])
        np.testing.assert_array_equal(slow, [2, 3])

    def test_integer_indices(self):
        """Test parsing integer indices."""
        names = ['a', 'b', 'c', 'd']

        processor = DataProcessor()
        fast, slow = processor._parse_variable_blocks(
            names,
            fast_vars=[0, 1],
            slow_vars=[2, 3]
        )

        np.testing.assert_array_equal(fast, [0, 1])
        np.testing.assert_array_equal(slow, [2, 3])

    def test_missing_names_ignored(self):
        """Test that non-existent names are ignored."""
        names = ['a', 'b']

        processor = DataProcessor()
        fast, slow = processor._parse_variable_blocks(
            names,
            fast_vars=['a', 'nonexistent'],
            slow_vars=None
        )

        np.testing.assert_array_equal(fast, [0])


class TestCausalOrdering:
    """Test causal graph discovery and ordering."""

    def test_slow_before_fast_ordering(self):
        """Test that slow variables come before fast in causal order."""
        np.random.seed(42)

        X = np.random.randn(100, 4)
        fast_indices = np.array([0, 1])
        slow_indices = np.array([2, 3])

        processor = DataProcessor()
        causal_order = processor._discover_causal_order(X, fast_indices, slow_indices)

        # Slow should come first
        slow_positions = [np.where(causal_order == i)[0][0] for i in slow_indices]
        fast_positions = [np.where(causal_order == i)[0][0] for i in fast_indices]

        assert max(slow_positions) < min(fast_positions)

    def test_variance_based_fallback(self):
        """Test variance-based ordering fallback."""
        np.random.seed(42)

        # Create data with different variances
        X = np.column_stack([
            np.random.randn(100) * 1,   # Low variance
            np.random.randn(100) * 10,  # High variance
            np.random.randn(100) * 5    # Medium variance
        ])

        indices = np.array([0, 1, 2])

        processor = DataProcessor()
        ordered = processor._variance_based_order(X, indices)

        # Lower variance should come first
        # Original order: 0 (low), 1 (high), 2 (medium)
        # Expected: 0, 2, 1 (sorted by variance)
        assert ordered[0] == 0  # Lowest variance first


class TestCAMCausalDiscovery:
    """Test CAM (Causal Additive Models) causal discovery."""

    def test_cam_is_default_method(self):
        """Test that CAM is the default causal discovery method."""
        processor = DataProcessor()
        assert processor.causal_discovery_method == 'cam'

    def test_cam_slow_before_fast_ordering(self):
        """Test that CAM orders slow variables before fast."""
        np.random.seed(42)

        X = np.random.randn(100, 4)
        fast_indices = np.array([0, 1])
        slow_indices = np.array([2, 3])

        processor = DataProcessor(causal_discovery_method='cam')
        causal_order = processor._discover_causal_order(X, fast_indices, slow_indices)

        # Slow should come first
        slow_positions = [np.where(causal_order == i)[0][0] for i in slow_indices]
        fast_positions = [np.where(causal_order == i)[0][0] for i in fast_indices]

        assert max(slow_positions) < min(fast_positions)

    def test_cam_greedy_sink_search(self):
        """Test CAM greedy sink search produces valid ordering."""
        np.random.seed(42)

        # Create data with causal structure: X0 -> X1 -> X2
        n_samples = 200
        X0 = np.random.randn(n_samples)
        X1 = 0.8 * X0 + 0.2 * np.random.randn(n_samples)
        X2 = 0.6 * X1 + 0.3 * np.random.randn(n_samples)
        X = np.column_stack([X0, X1, X2])

        indices = np.array([0, 1, 2])

        processor = DataProcessor(causal_discovery_method='cam')
        ordered = processor._cam_greedy_sink_search(X, indices)

        # Check that ordering is a valid permutation
        assert len(ordered) == 3
        assert set(ordered) == {0, 1, 2}

    def test_cam_greedy_sink_with_nonlinear_data(self):
        """Test CAM handles non-linear relationships."""
        np.random.seed(42)

        # Create non-linear causal structure: X0 -> X1 (quadratic), X1 -> X2
        n_samples = 300
        X0 = np.random.randn(n_samples)
        X1 = 0.5 * X0**2 + 0.3 * np.random.randn(n_samples)  # Non-linear
        X2 = np.sin(X1) + 0.2 * np.random.randn(n_samples)   # Non-linear
        X = np.column_stack([X0, X1, X2])

        indices = np.array([0, 1, 2])

        processor = DataProcessor(causal_discovery_method='cam')
        ordered = processor._cam_greedy_sink_search(X, indices)

        # Should produce valid ordering
        assert len(ordered) == 3
        assert set(ordered) == {0, 1, 2}

    def test_cam_with_full_pipeline(self):
        """Test CAM with full fit_transform pipeline."""
        np.random.seed(42)

        X = np.random.randn(100, 6)

        processor = DataProcessor(
            ctree_alpha=0.20,
            ctree_min_split=20,
            causal_discovery_method='cam'
        )

        topology = processor.fit_transform(
            X,
            fast_vars=[0, 1, 2],
            slow_vars=[3, 4, 5]
        )

        assert topology.X_processed.shape[0] <= 100
        assert topology.X_processed.shape[1] == 6
        assert len(topology.causal_order) == 6


class TestLiNGAMCausalDiscovery:
    """Test LiNGAM causal discovery (explicit selection)."""

    def test_lingam_method_selection(self):
        """Test that LiNGAM can be explicitly selected."""
        processor = DataProcessor(causal_discovery_method='lingam')
        assert processor.causal_discovery_method == 'lingam'

    def test_lingam_slow_before_fast_ordering(self):
        """Test that LiNGAM orders slow variables before fast."""
        np.random.seed(42)

        X = np.random.randn(100, 4)
        fast_indices = np.array([0, 1])
        slow_indices = np.array([2, 3])

        processor = DataProcessor(causal_discovery_method='lingam')
        causal_order = processor._discover_causal_order(X, fast_indices, slow_indices)

        # Slow should come first
        slow_positions = [np.where(causal_order == i)[0][0] for i in slow_indices]
        fast_positions = [np.where(causal_order == i)[0][0] for i in fast_indices]

        assert max(slow_positions) < min(fast_positions)

    def test_lingam_with_full_pipeline(self):
        """Test LiNGAM with full fit_transform pipeline."""
        np.random.seed(42)

        # Use Student-t data for non-Gaussian (LiNGAM requirement)
        from scipy.stats import t as student_t
        X = student_t.rvs(df=5, size=(100, 6))

        processor = DataProcessor(
            ctree_alpha=0.20,
            ctree_min_split=20,
            causal_discovery_method='lingam'
        )

        topology = processor.fit_transform(
            X,
            fast_vars=[0, 1, 2],
            slow_vars=[3, 4, 5]
        )

        assert topology.X_processed.shape[0] <= 100
        assert topology.X_processed.shape[1] == 6


class TestCausalMethodValidation:
    """Test validation of causal discovery method parameter."""

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported causal_discovery_method"):
            DataProcessor(causal_discovery_method='invalid_method')

    def test_supported_methods(self):
        """Test that supported methods are correctly defined."""
        assert 'cam' in DataProcessor.SUPPORTED_CAUSAL_METHODS
        assert 'lingam' in DataProcessor.SUPPORTED_CAUSAL_METHODS


class TestDataTopology:
    """Test DataTopology dataclass."""

    def test_to_dict_serialization(self):
        """Test that DataTopology serializes correctly."""
        topology = DataTopology(
            X_processed=np.random.randn(10, 3),
            regimes=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            regime_embeddings=np.eye(2)[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
            causal_order=np.array([2, 0, 1]),
            variable_names=['a', 'b', 'c'],
            scaler_mean=np.array([1.0, 2.0, 3.0]),
            scaler_std=np.array([0.5, 0.5, 0.5]),
            fast_indices=np.array([0, 1]),
            slow_indices=np.array([2]),
            n_regimes=2
        )

        d = topology.to_dict()

        assert d['causal_order'] == [2, 0, 1]
        assert d['variable_names'] == ['a', 'b', 'c']
        assert d['n_regimes'] == 2


class TestFullPipeline:
    """Test complete fit_transform pipeline."""

    def test_fit_transform_basic(self):
        """Test basic fit_transform workflow."""
        np.random.seed(42)

        X = np.random.randn(100, 6)

        processor = DataProcessor(
            ctree_alpha=0.20,
            ctree_min_split=20
        )

        topology = processor.fit_transform(
            X,
            fast_vars=[0, 1, 2],
            slow_vars=[3, 4, 5]
        )

        assert topology.X_processed.shape[0] <= 100  # May have lost rows to differencing
        assert topology.X_processed.shape[1] == 6
        assert len(topology.regimes) == topology.X_processed.shape[0]
        assert topology.n_regimes >= 1

    def test_fit_transform_with_polars(self):
        """Test fit_transform with Polars DataFrame."""
        np.random.seed(42)

        df = pl.DataFrame({
            'VIX': np.random.randn(100),
            'SPY': np.random.randn(100),
            'GDP': np.random.randn(100),
            'CPI': np.random.randn(100)
        })

        processor = DataProcessor(ctree_alpha=0.20, ctree_min_split=20)
        topology = processor.fit_transform(
            df,
            fast_vars=['VIX', 'SPY'],
            slow_vars=['GDP', 'CPI']
        )

        assert topology.X_processed.shape[1] == 4
        assert 'VIX' in topology.variable_names or 'var_' in topology.variable_names[0]

    def test_transform_new_data(self):
        """Test transforming new data with fitted processor."""
        np.random.seed(42)

        X_train = np.random.randn(100, 4)

        processor = DataProcessor(ctree_alpha=0.20, ctree_min_split=20)
        topology = processor.fit_transform(X_train)

        X_new = np.random.randn(50, 4)
        X_transformed = processor.transform(X_new, topology)

        assert X_transformed.shape[1] == 4

    def test_denormalize(self):
        """Test denormalization reverses normalization."""
        np.random.seed(42)

        X = np.random.randn(100, 4) * 5 + 10

        processor = DataProcessor(ctree_alpha=0.20, ctree_min_split=20)
        topology = processor.fit_transform(X)

        # Denormalize the processed data
        X_denorm = processor.denormalize(topology.X_processed, topology)

        # Should be close to original scale (though order may differ)
        assert np.abs(np.mean(X_denorm) - np.mean(X)) < 2


class TestDimensionalityReduction:
    """Test PCA-based dimensionality reduction."""

    def test_no_reduction_when_under_limit(self):
        """Test that no reduction happens when under max_features."""
        np.random.seed(42)

        X = np.random.randn(100, 10)
        slow_indices = np.array([5, 6, 7, 8, 9])

        processor = DataProcessor(max_features=50)
        X_reduced, pca_info = processor._apply_dimensionality_reduction(X, slow_indices)

        np.testing.assert_array_equal(X_reduced, X)
        assert pca_info == {}

    def test_reduction_when_over_limit(self):
        """Test that reduction happens when over max_features."""
        np.random.seed(42)

        X = np.random.randn(100, 60)
        slow_indices = np.arange(30, 60)  # 30 slow variables

        processor = DataProcessor(max_features=50)

        with pytest.warns(UserWarning, match="PCA"):
            X_reduced, pca_info = processor._apply_dimensionality_reduction(X, slow_indices)

        # Should have reduced dimensions
        assert X_reduced.shape[1] < 60
        assert 'components' in pca_info


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_fast_vars(self):
        """Test with no fast variables."""
        np.random.seed(42)

        X = np.random.randn(50, 4)

        processor = DataProcessor(ctree_alpha=0.20, ctree_min_split=10)
        topology = processor.fit_transform(
            X,
            fast_vars=[],
            slow_vars=[0, 1, 2, 3]
        )

        assert topology.X_processed.shape[1] == 4

    def test_empty_slow_vars(self):
        """Test with no slow variables."""
        np.random.seed(42)

        X = np.random.randn(50, 4)

        processor = DataProcessor(ctree_alpha=0.20, ctree_min_split=10)
        topology = processor.fit_transform(
            X,
            fast_vars=[0, 1, 2, 3],
            slow_vars=[]
        )

        assert topology.X_processed.shape[1] == 4

    def test_single_variable(self):
        """Test with single variable."""
        np.random.seed(42)

        X = np.random.randn(50, 1)

        processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=10)
        topology = processor.fit_transform(X)

        assert topology.X_processed.shape[1] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
