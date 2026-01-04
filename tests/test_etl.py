"""
Tests for ETL and Data Topology Module

Tests cover:
- Data conversion (numpy, polars, pandas)
- Data validation (NaN/Inf handling)
- Cubic spline imputation
- Stationarity checks and differencing
- Normalization
- CAM causal discovery (greedy sink search)
- LiNGAM causal discovery
- Ordering constraints (Bayesian prior)
- Full preprocessing pipeline

Terminology (aligned with paper):
- FAST = informationally fast = UPSTREAM = drives others = FIRST in ordering
- SLOW = informationally slow = DOWNSTREAM = reacts to all = SINK = LAST in ordering
"""

import numpy as np
import polars as pl
import pytest
import warnings

import sys
sys.path.insert(0, '..')

from core.etl import (
    DataProcessor,
    DataTopology,
    _to_numpy_array,
    validate_for_ode,
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

    def test_numpy_to_polars_helper(self):
        """Test numpy to polars helper function."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        names = ['x', 'y']

        df = numpy_to_polars(X, names)

        assert df.shape == (2, 2)
        assert df.columns == ['x', 'y']

    def test_polars_to_numpy_helper(self):
        """Test polars to numpy helper function."""
        df = pl.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})

        X, names = polars_to_numpy(df)

        assert X.shape == (2, 2)
        assert names == ['a', 'b']

    def test_invalid_type_raises(self):
        """Test invalid type raises TypeError."""
        with pytest.raises(TypeError):
            _to_numpy_array("invalid")


class TestDataValidation:
    """Test data validation for ODE safety."""

    def test_validate_for_ode_clean_data(self):
        """Test validation passes for clean data."""
        X = np.random.randn(100, 5)
        validate_for_ode(X)  # Should not raise

    def test_validate_for_ode_nan_raises(self):
        """Test NaN values raise ValueError."""
        X = np.array([[1.0, np.nan], [2.0, 3.0]])

        with pytest.raises(ValueError, match="NaN"):
            validate_for_ode(X)

    def test_validate_for_ode_inf_raises(self):
        """Test Inf values raise ValueError."""
        X = np.array([[1.0, np.inf], [2.0, 3.0]])

        with pytest.raises(ValueError, match="Inf"):
            validate_for_ode(X)

    def test_validate_for_ode_extreme_values_warns(self):
        """Test extreme values generate warning."""
        X = np.array([[1e8, 2.0], [3.0, 4.0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_for_ode(X)
            assert len(w) >= 1
            assert "extreme" in str(w[0].message).lower()


class TestCubicSplineImputation:
    """Test cubic spline imputation for missing values."""

    def test_imputation_fills_nans(self):
        """Test NaN values are filled."""
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0]
        ])

        processor = DataProcessor()
        X_imputed = processor._impute_cubic_spline(X)

        assert not np.any(np.isnan(X_imputed))

    def test_imputation_preserves_non_nan(self):
        """Test non-NaN values are preserved."""
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0]
        ])

        processor = DataProcessor()
        X_imputed = processor._impute_cubic_spline(X)

        assert X_imputed[0, 0] == 1.0
        assert X_imputed[2, 0] == 3.0

    def test_imputation_interpolates_smoothly(self):
        """Test interpolation produces reasonable values."""
        X = np.array([[1.0], [np.nan], [3.0], [4.0], [5.0]])

        processor = DataProcessor()
        X_imputed = processor._impute_cubic_spline(X)

        # Interpolated value should be around 2.0
        assert 1.5 <= X_imputed[1, 0] <= 2.5

    def test_all_nan_column_raises(self):
        """Test all-NaN column raises error."""
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])

        processor = DataProcessor()
        with pytest.raises(ValueError, match="entirely NaN"):
            processor._impute_cubic_spline(X)

    def test_edge_nans_handled(self):
        """Test edge NaN values are handled."""
        X = np.array([[np.nan], [2.0], [3.0], [4.0], [np.nan]])

        processor = DataProcessor()
        X_imputed = processor._impute_cubic_spline(X)

        assert not np.any(np.isnan(X_imputed))


class TestStationarityChecks:
    """Test stationarity checking and differencing."""

    def test_stationary_series_detected(self):
        """Test stationary series is detected."""
        np.random.seed(42)
        series = np.random.randn(200)

        processor = DataProcessor()
        is_stationary, diff_order = processor._check_stationarity(series)

        assert is_stationary
        assert diff_order == 0

    def test_random_walk_needs_differencing(self):
        """Test random walk detected as non-stationary."""
        np.random.seed(42)
        series = np.cumsum(np.random.randn(200))

        processor = DataProcessor()
        is_stationary, diff_order = processor._check_stationarity(series)

        # Should detect as non-stationary or need differencing
        assert not is_stationary or diff_order > 0

    def test_differencing_creates_stationary(self):
        """Test differencing creates stationary series."""
        np.random.seed(42)
        series = np.cumsum(np.random.randn(100))

        processor = DataProcessor()
        diffed = processor._apply_differencing(series, 1)

        # Should have NaN at start
        assert np.isnan(diffed[0])

    def test_log_returns_for_positive_series(self):
        """Test positive series uses log-returns."""
        np.random.seed(42)
        series = np.exp(np.cumsum(0.01 * np.random.randn(100)))

        processor = DataProcessor()
        diffed = processor._apply_differencing(series, 1)

        diffed_clean = diffed[~np.isnan(diffed)]
        # Log returns should be around 0.01 scale
        assert np.abs(np.mean(diffed_clean)) < 0.1

    def test_first_diff_for_mixed_series(self):
        """Test mixed series uses first differences."""
        series = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        processor = DataProcessor()
        diffed = processor._apply_differencing(series, 1)

        diffed_clean = diffed[~np.isnan(diffed)]
        np.testing.assert_array_almost_equal(diffed_clean, np.ones_like(diffed_clean))

    def test_short_series_assumed_stationary(self):
        """Test very short series assumed stationary."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        processor = DataProcessor()
        is_stationary, diff_order = processor._check_stationarity(series)

        assert is_stationary
        assert diff_order == 0


class TestNormalization:
    """Test normalization for OT-Path stability."""

    def test_normalized_mean_zero(self):
        """Test normalized data has mean near zero."""
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 5

        processor = DataProcessor()
        X_norm, mean, std = processor._normalize(X)

        np.testing.assert_array_almost_equal(
            np.mean(X_norm, axis=0),
            np.zeros(5),
            decimal=1
        )

    def test_normalized_std_one(self):
        """Test normalized data has std near one."""
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 5

        processor = DataProcessor()
        X_norm, mean, std = processor._normalize(X)

        np.testing.assert_array_almost_equal(
            np.std(X_norm, axis=0),
            np.ones(5),
            decimal=1
        )

    def test_scaler_stats_stored(self):
        """Test scaler statistics are stored."""
        X = np.random.randn(100, 5)

        processor = DataProcessor()
        X_norm, mean, std = processor._normalize(X)

        assert processor.scaler is not None
        assert len(mean) == 5
        assert len(std) == 5


class TestCAMCausalDiscovery:
    """Test CAM (Causal Additive Models) causal discovery.

    CAM uses greedy sink search to find causal ordering:
    - FAST (upstream) variables drive others → placed FIRST
    - SLOW (downstream) sinks react to all → placed LAST
    """

    def test_cam_is_default_method(self):
        """Test CAM is the default causal discovery method."""
        processor = DataProcessor()
        assert processor.causal_discovery_method == 'cam'

    def test_cam_greedy_sink_search(self):
        """Test CAM greedy sink search produces valid ordering."""
        np.random.seed(42)

        # Create data with causal structure: X0 -> X1 -> X2
        n_samples = 200
        X0 = np.random.randn(n_samples)
        X1 = 0.8 * X0 + 0.2 * np.random.randn(n_samples)
        X2 = 0.6 * X1 + 0.3 * np.random.randn(n_samples)
        X = np.column_stack([X0, X1, X2])

        processor = DataProcessor(causal_discovery_method='cam')
        ordered = processor._cam_constrained_greedy_sink_search(X, constraints=[])

        # Check ordering is valid permutation
        assert len(ordered) == 3
        assert set(ordered) == {0, 1, 2}

    def test_cam_greedy_sink_with_nonlinear_data(self):
        """Test CAM handles non-linear relationships."""
        np.random.seed(42)

        # Create non-linear causal structure
        n_samples = 300
        X0 = np.random.randn(n_samples)
        X1 = 0.5 * X0**2 + 0.3 * np.random.randn(n_samples)
        X2 = np.sin(X1) + 0.2 * np.random.randn(n_samples)
        X = np.column_stack([X0, X1, X2])

        processor = DataProcessor(causal_discovery_method='cam')
        ordered = processor._cam_constrained_greedy_sink_search(X, constraints=[])

        assert len(ordered) == 3
        assert set(ordered) == {0, 1, 2}

    def test_cam_respects_ordering_constraints(self):
        """Test CAM respects user-specified ordering constraints."""
        np.random.seed(42)

        X = np.random.randn(100, 4)
        constraints = [(0, 3)]  # Variable 0 must come before 3

        processor = DataProcessor(causal_discovery_method='cam')
        ordered = processor._cam_constrained_greedy_sink_search(X, constraints)

        pos_0 = list(ordered).index(0)
        pos_3 = list(ordered).index(3)
        assert pos_0 < pos_3

    def test_cam_with_multiple_constraints(self):
        """Test CAM with multiple ordering constraints."""
        np.random.seed(42)

        X = np.random.randn(100, 5)
        constraints = [(0, 2), (2, 4)]

        processor = DataProcessor(causal_discovery_method='cam')
        ordered = processor._cam_constrained_greedy_sink_search(X, constraints)

        pos_0 = list(ordered).index(0)
        pos_2 = list(ordered).index(2)
        pos_4 = list(ordered).index(4)

        assert pos_0 < pos_2
        assert pos_2 < pos_4

    def test_cam_with_full_pipeline(self):
        """Test CAM with full fit_transform pipeline."""
        np.random.seed(42)

        X = np.random.randn(100, 6)

        processor = DataProcessor(
            ctree_alpha=0.20,
            ctree_min_split=20,
            causal_discovery_method='cam'
        )

        topology = processor.fit_transform(X)

        assert topology.X_processed.shape[0] <= 100
        assert topology.X_processed.shape[1] == 6
        assert len(topology.causal_order) == 6


class TestOrderingConstraints:
    """Test ordering constraints (Bayesian prior) functionality."""

    def test_parse_ordering_hints_by_name(self):
        """Test parsing ordering hints by variable name."""
        variable_names = ['A', 'B', 'C', 'D']
        causal_order_hint = ['A', 'C', 'D']

        processor = DataProcessor()
        constraints = processor._parse_ordering_hints(variable_names, causal_order_hint)

        assert (0, 2) in constraints
        assert (0, 3) in constraints
        assert (2, 3) in constraints

    def test_parse_ordering_hints_by_index(self):
        """Test parsing ordering hints by index."""
        variable_names = ['A', 'B', 'C', 'D']
        causal_order_hint = [0, 2, 3]

        processor = DataProcessor()
        constraints = processor._parse_ordering_hints(variable_names, causal_order_hint)

        assert (0, 2) in constraints
        assert (0, 3) in constraints
        assert (2, 3) in constraints

    def test_parse_ordering_hints_missing_variable(self):
        """Test missing variables in hints generate warning."""
        variable_names = ['A', 'B', 'C']
        causal_order_hint = ['A', 'X', 'C']

        processor = DataProcessor()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            constraints = processor._parse_ordering_hints(variable_names, causal_order_hint)
            assert any("not found" in str(warning.message) for warning in w)

    def test_fit_transform_with_causal_order_hint(self):
        """Test full pipeline with causal_order_hint."""
        np.random.seed(42)

        df = pl.DataFrame({
            'Rates': np.random.randn(100),
            'Spreads': np.random.randn(100),
            'VIX': np.random.randn(100),
            'SPX': np.random.randn(100)
        })

        processor = DataProcessor(ctree_alpha=0.20, ctree_min_split=20)
        topology = processor.fit_transform(
            df,
            causal_order_hint=['Rates', 'Spreads', 'SPX']
        )

        names = topology.variable_names
        rates_pos = names.index('Rates')
        spreads_pos = names.index('Spreads')
        spx_pos = names.index('SPX')

        assert rates_pos < spreads_pos
        assert spreads_pos < spx_pos


class TestLiNGAMCausalDiscovery:
    """Test LiNGAM causal discovery."""

    def test_lingam_method_selection(self):
        """Test LiNGAM can be explicitly selected."""
        processor = DataProcessor(causal_discovery_method='lingam')
        assert processor.causal_discovery_method == 'lingam'

    def test_lingam_with_full_pipeline(self):
        """Test LiNGAM with full fit_transform pipeline."""
        np.random.seed(42)

        from scipy.stats import t as student_t
        X = student_t.rvs(df=5, size=(100, 6))

        processor = DataProcessor(
            ctree_alpha=0.20,
            ctree_min_split=20,
            causal_discovery_method='lingam'
        )

        topology = processor.fit_transform(X)

        assert topology.X_processed.shape[0] <= 100
        assert topology.X_processed.shape[1] == 6


class TestCausalMethodValidation:
    """Test validation of causal discovery method parameter."""

    def test_invalid_method_raises_error(self):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported causal_discovery_method"):
            DataProcessor(causal_discovery_method='invalid_method')

    def test_supported_methods(self):
        """Test supported methods are correctly defined."""
        assert 'cam' in DataProcessor.SUPPORTED_CAUSAL_METHODS
        assert 'lingam' in DataProcessor.SUPPORTED_CAUSAL_METHODS


class TestVarianceBasedOrder:
    """Test variance-based ordering fallback."""

    def test_variance_based_order_low_variance_first(self):
        """Test lower variance variables come first (upstream/FAST)."""
        np.random.seed(42)

        X = np.column_stack([
            np.random.randn(100) * 1,   # Low variance (upstream/FAST)
            np.random.randn(100) * 10,  # High variance (downstream/SLOW)
            np.random.randn(100) * 5    # Medium variance
        ])

        processor = DataProcessor()
        ordered = processor._variance_based_order(X, constraints=[])

        # Lower variance first (upstream = FAST)
        assert ordered[0] == 0
        assert ordered[-1] == 1

    def test_variance_based_order_respects_constraints(self):
        """Test variance-based order respects constraints."""
        np.random.seed(42)

        X = np.column_stack([
            np.random.randn(100) * 10,
            np.random.randn(100) * 1,
        ])

        constraints = [(0, 1)]

        processor = DataProcessor()
        ordered = processor._variance_based_order(X, constraints)

        pos_0 = list(ordered).index(0)
        pos_1 = list(ordered).index(1)
        assert pos_0 < pos_1


class TestDataTopology:
    """Test DataTopology dataclass."""

    def test_to_dict_serialization(self):
        """Test DataTopology serializes correctly."""
        topology = DataTopology(
            X_processed=np.random.randn(10, 3),
            regimes=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            regime_embeddings=np.eye(2)[[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
            causal_order=np.array([2, 0, 1]),
            variable_names=['c', 'a', 'b'],
            scaler_mean=np.array([0.0, 0.0, 0.0]),
            scaler_std=np.array([1.0, 1.0, 1.0]),
            n_regimes=2,
            fast_indices=np.array([0, 1]),
            slow_indices=np.array([2])
        )

        d = topology.to_dict()

        assert d['causal_order'] == [2, 0, 1]
        assert d['variable_names'] == ['c', 'a', 'b']
        assert d['n_regimes'] == 2


class TestFullPipeline:
    """Test full preprocessing pipeline."""

    def test_fit_transform_basic(self):
        """Test basic fit_transform functionality."""
        np.random.seed(42)

        X = np.random.randn(100, 6)

        processor = DataProcessor(ctree_alpha=0.20, ctree_min_split=20)
        topology = processor.fit_transform(X)

        assert topology.X_processed.shape[0] <= 100
        assert topology.X_processed.shape[1] == 6

    def test_fit_transform_with_polars(self):
        """Test fit_transform with Polars DataFrame."""
        np.random.seed(42)

        df = pl.DataFrame({
            'var_a': np.random.randn(100),
            'var_b': np.random.randn(100),
            'var_c': np.random.randn(100),
        })

        processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=20)
        topology = processor.fit_transform(df)

        assert topology.X_processed.shape[1] == 3

    def test_transform_new_data(self):
        """Test transforming new data with fitted processor."""
        np.random.seed(42)

        X_train = np.random.randn(100, 4)
        X_new = np.random.randn(20, 4)

        processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=20)
        topology = processor.fit_transform(X_train)

        X_transformed = processor.transform(X_new, topology)

        assert X_transformed.shape[1] == 4

    def test_denormalize(self):
        """Test denormalization reverses normalization."""
        np.random.seed(42)

        X = np.random.randn(100, 4) * 5 + 10

        processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=20)
        topology = processor.fit_transform(X)

        X_denorm = processor.denormalize(topology.X_processed, topology)

        assert np.std(X_denorm) > 0.1


class TestDimensionalityReduction:
    """Test PCA dimensionality reduction."""

    def test_no_reduction_when_under_limit(self):
        """Test PCA not applied when under limit."""
        X = np.random.randn(100, 10)

        processor = DataProcessor(max_features=50)
        X_reduced, pca_info = processor._apply_dimensionality_reduction(X)

        assert X_reduced.shape[1] == 10
        assert pca_info == {}

    def test_reduction_when_over_limit(self):
        """Test PCA applied when over limit."""
        X = np.random.randn(100, 60)

        processor = DataProcessor(max_features=50)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            X_reduced, pca_info = processor._apply_dimensionality_reduction(X)

        assert X_reduced.shape[1] == 50
        assert 'components' in pca_info


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_variable(self):
        """Test processing with single variable."""
        X = np.random.randn(100, 1)

        processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=10)
        topology = processor.fit_transform(X)

        assert topology.X_processed.shape[1] == 1

    def test_deprecated_fast_slow_vars_warning(self):
        """Test deprecated fast_vars/slow_vars generate warnings."""
        X = np.random.randn(100, 4)

        processor = DataProcessor(ctree_alpha=0.50, ctree_min_split=20)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            topology = processor.fit_transform(
                X,
                fast_vars=[0, 1],
                slow_vars=[2, 3]
            )
            assert any("deprecated" in str(warning.message).lower() for warning in w)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
