"""
Data Ingestion & Topology Module for Causal Conditional Flow Matching

Handles the complete preprocessing pipeline for time series data:
1. Validation and cleaning (NaN/Inf removal, Cubic Spline imputation)
2. Stationarity enforcement (ADF tests, differencing, log-returns)
3. Normalization (StandardScaler for OT-Path stability)
4. Regime classification (CTree-Lite based partitioning)
5. Causal graph discovery (LiNGAM on "Fast" block)
6. Variable reordering based on causal hierarchy

Critical Notes:
- ODE solvers are extremely sensitive to NaN/Inf - validation is mandatory
- StandardScaler is required for stable gradients in the OT-Path
- The causal_permutation_list must be saved with the model artifacts

Note: This module uses Polars for DataFrame operations (not pandas).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import interpolate
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings

from .ctree import CTree


@dataclass
class DataTopology:
    """Container for processed data and associated metadata.

    This structure contains all information needed for training
    and must be saved alongside model weights for reproducibility.

    Attributes:
        X_processed: Normalized, causally-ordered feature matrix
        regimes: Regime labels from CTree classification
        regime_embeddings: One-hot or learned embeddings for regimes
        causal_order: Permutation vector for variable ordering
        variable_names: Names of variables in causal order
        scaler_mean: Mean values for denormalization
        scaler_std: Standard deviations for denormalization
        fast_indices: Indices of "fast" (market) variables
        slow_indices: Indices of "slow" (macro) variables
        n_regimes: Number of detected regimes
        differenced_cols: Columns that were differenced for stationarity
        pca_components: PCA components if dimensionality reduction was applied
    """
    X_processed: np.ndarray
    regimes: np.ndarray
    regime_embeddings: np.ndarray
    causal_order: np.ndarray
    variable_names: List[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    fast_indices: np.ndarray
    slow_indices: np.ndarray
    n_regimes: int
    differenced_cols: List[int] = field(default_factory=list)
    pca_components: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize topology to dictionary for saving."""
        return {
            'causal_order': self.causal_order.tolist(),
            'variable_names': self.variable_names,
            'scaler_mean': self.scaler_mean.tolist(),
            'scaler_std': self.scaler_std.tolist(),
            'fast_indices': self.fast_indices.tolist(),
            'slow_indices': self.slow_indices.tolist(),
            'n_regimes': self.n_regimes,
            'differenced_cols': self.differenced_cols,
            'pca_components': self.pca_components.tolist() if self.pca_components is not None else None,
            'pca_mean': self.pca_mean.tolist() if self.pca_mean is not None else None
        }


def _to_numpy_array(data: Union[np.ndarray, pl.DataFrame, "pd.DataFrame"]) -> Tuple[np.ndarray, List[str]]:
    """Convert input data to numpy array with column names.

    Supports numpy arrays, Polars DataFrames, and pandas DataFrames (for compatibility).

    Args:
        data: Input data in various formats

    Returns:
        X: Numpy array of shape (n_samples, n_features)
        column_names: List of column names
    """
    if isinstance(data, np.ndarray):
        return data.astype(np.float64), [f"var_{i}" for i in range(data.shape[1])]
    elif isinstance(data, pl.DataFrame):
        return data.to_numpy().astype(np.float64), data.columns
    else:
        # Fallback for pandas DataFrame (for backward compatibility)
        try:
            return data.values.astype(np.float64), list(data.columns)
        except AttributeError:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Expected numpy.ndarray, polars.DataFrame, or pandas.DataFrame"
            )


class DataProcessor:
    """Complete data preprocessing pipeline for C-CFM.

    Implements the full ETL workflow from raw time series to
    normalized, causally-ordered, regime-labeled data.

    Args:
        adf_threshold: P-value threshold for ADF stationarity test (default 0.05)
        max_diff_order: Maximum differencing order for stationarity (default 2)
        ctree_alpha: Significance threshold for CTree regime detection (default 0.05)
        ctree_min_split: Minimum samples for CTree split (default 20)
        max_features: Maximum features before PCA reduction (default 50)
        pca_variance_ratio: Variance to retain in PCA (default 0.95)

    Example:
        >>> processor = DataProcessor()
        >>> topology = processor.fit_transform(
        ...     X_raw,
        ...     fast_vars=['VIX', 'SPX_ret'],
        ...     slow_vars=['GDP', 'CPI']
        ... )
    """

    def __init__(
        self,
        adf_threshold: float = 0.05,
        max_diff_order: int = 2,
        ctree_alpha: float = 0.05,
        ctree_min_split: int = 20,
        max_features: int = 50,
        pca_variance_ratio: float = 0.95
    ):
        self.adf_threshold = adf_threshold
        self.max_diff_order = max_diff_order
        self.ctree_alpha = ctree_alpha
        self.ctree_min_split = ctree_min_split
        self.max_features = max_features
        self.pca_variance_ratio = pca_variance_ratio

        self.scaler: Optional[StandardScaler] = None
        self.ctree: Optional[CTree] = None
        self.pca: Optional[PCA] = None
        self._is_fitted: bool = False

    def fit_transform(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        fast_vars: Optional[List[str]] = None,
        slow_vars: Optional[List[str]] = None,
        regime_response_vars: Optional[List[str]] = None
    ) -> DataTopology:
        """Full preprocessing pipeline: validate, clean, order, label.

        Args:
            X: Raw time series data of shape (T, D)
               - If Polars DataFrame, column names are used for variable identification
               - If ndarray, provide fast_vars and slow_vars as indices
            fast_vars: Names/indices of "fast" market variables
                       (e.g., VIX, daily returns, spreads)
            slow_vars: Names/indices of "slow" macro variables
                       (e.g., GDP, CPI, unemployment)
            regime_response_vars: Variables to use as response for regime detection
                                  (defaults to fast_vars if not specified)

        Returns:
            DataTopology containing all processed data and metadata
        """
        # Convert to numpy array with column names
        X_array, variable_names = _to_numpy_array(X)
        n_vars = len(variable_names)

        # Parse fast/slow variable specifications
        fast_indices, slow_indices = self._parse_variable_blocks(
            variable_names, fast_vars, slow_vars
        )

        # Step A: Validation and Cleaning
        X_clean, differenced_cols = self._validate_and_clean(X_array)

        # Step B: Dimensionality Reduction (if needed)
        X_reduced, pca_info = self._apply_dimensionality_reduction(
            X_clean, slow_indices
        )

        # Step C: Normalize data
        X_normalized, scaler_mean, scaler_std = self._normalize(X_reduced)

        # Step D: Regime Classification with CTree
        if regime_response_vars is None:
            # Use fast variables as regime indicators by default
            regime_features = X_normalized[:, fast_indices] if len(fast_indices) > 0 else X_normalized
        else:
            regime_idx = [variable_names.index(v) for v in regime_response_vars if v in variable_names]
            regime_features = X_normalized[:, regime_idx] if regime_idx else X_normalized

        regimes, n_regimes = self._classify_regimes(X_normalized, regime_features)

        # Step E: Causal Graph Discovery on Fast Block
        causal_order = self._discover_causal_order(
            X_normalized, fast_indices, slow_indices
        )

        # Reorder data according to causal structure
        X_ordered = X_normalized[:, causal_order]
        ordered_names = [variable_names[i] for i in causal_order]

        # Update fast/slow indices after reordering
        fast_indices_new = np.array([
            np.where(causal_order == i)[0][0]
            for i in fast_indices
            if i in causal_order
        ])
        slow_indices_new = np.array([
            np.where(causal_order == i)[0][0]
            for i in slow_indices
            if i in causal_order
        ])

        # Create regime embeddings (one-hot)
        regime_embeddings = np.eye(n_regimes)[regimes]

        self._is_fitted = True

        return DataTopology(
            X_processed=X_ordered,
            regimes=regimes,
            regime_embeddings=regime_embeddings,
            causal_order=causal_order,
            variable_names=ordered_names,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            fast_indices=fast_indices_new,
            slow_indices=slow_indices_new,
            n_regimes=n_regimes,
            differenced_cols=differenced_cols,
            pca_components=pca_info.get('components'),
            pca_mean=pca_info.get('mean')
        )

    def _parse_variable_blocks(
        self,
        variable_names: List[str],
        fast_vars: Optional[List[str]],
        slow_vars: Optional[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse fast and slow variable specifications."""
        n_vars = len(variable_names)

        if fast_vars is None and slow_vars is None:
            # Default: first half fast, second half slow
            mid = n_vars // 2
            fast_indices = np.arange(mid)
            slow_indices = np.arange(mid, n_vars)
        else:
            # Parse string names or indices
            if fast_vars is not None:
                if all(isinstance(v, str) for v in fast_vars):
                    fast_indices = np.array([
                        variable_names.index(v) for v in fast_vars
                        if v in variable_names
                    ])
                else:
                    fast_indices = np.array(fast_vars)
            else:
                fast_indices = np.array([], dtype=int)

            if slow_vars is not None:
                if all(isinstance(v, str) for v in slow_vars):
                    slow_indices = np.array([
                        variable_names.index(v) for v in slow_vars
                        if v in variable_names
                    ])
                else:
                    slow_indices = np.array(slow_vars)
            else:
                slow_indices = np.array([], dtype=int)

        return fast_indices, slow_indices

    def _validate_and_clean(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """Validate data, impute NaNs, and enforce stationarity.

        Critical: ODE solvers fail on any NaN/Inf values.

        Returns:
            X_clean: Clean numpy array with no missing values
            differenced_cols: List of columns that were differenced
        """
        X = X.astype(np.float64)
        n_samples, n_features = X.shape
        differenced_cols = []

        # Step 1: Check for Inf values (these must be removed or clipped)
        inf_mask = ~np.isfinite(X)
        if np.any(inf_mask):
            n_inf = np.sum(inf_mask)
            warnings.warn(f"Found {n_inf} Inf values. Replacing with NaN for imputation.")
            X[inf_mask] = np.nan

        # Step 2: Cubic Spline Imputation for NaN values
        nan_count = np.sum(np.isnan(X))
        if nan_count > 0:
            X = self._impute_cubic_spline(X)

        # Step 3: Stationarity Tests and Differencing
        for col in range(n_features):
            series = X[:, col]
            is_stationary, diff_order = self._check_stationarity(series)

            if not is_stationary and diff_order > 0:
                # Apply differencing
                X[:, col] = self._apply_differencing(series, diff_order)
                differenced_cols.append(col)

        # Final NaN check after differencing (first values become NaN)
        # Trim rows with NaN from the beginning
        valid_start = 0
        for i in range(n_samples):
            if not np.any(np.isnan(X[i, :])):
                valid_start = i
                break
        X = X[valid_start:, :]

        # Final validation
        if np.any(~np.isfinite(X)):
            raise ValueError(
                "Data still contains NaN/Inf after cleaning. "
                "Please check input data quality."
            )

        return X, differenced_cols

    def _impute_cubic_spline(self, X: np.ndarray) -> np.ndarray:
        """Apply cubic spline interpolation to fill NaN values.

        Uses scipy's CubicSpline for smooth interpolation.
        Edge NaNs are filled with nearest valid value.
        """
        X_imputed = X.copy()
        n_samples, n_features = X.shape
        time_idx = np.arange(n_samples)

        for col in range(n_features):
            series = X[:, col]
            nan_mask = np.isnan(series)

            if not np.any(nan_mask):
                continue

            if np.all(nan_mask):
                raise ValueError(f"Column {col} is entirely NaN - cannot impute.")

            # Get valid points
            valid_mask = ~nan_mask
            valid_idx = time_idx[valid_mask]
            valid_vals = series[valid_mask]

            if len(valid_vals) < 4:
                # Not enough points for cubic spline, use linear
                interp_func = interpolate.interp1d(
                    valid_idx, valid_vals,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(valid_vals[0], valid_vals[-1])
                )
            else:
                # Cubic spline interpolation
                try:
                    cs = interpolate.CubicSpline(
                        valid_idx, valid_vals,
                        bc_type='natural'
                    )
                    interp_func = cs
                except ValueError:
                    # Fall back to linear if spline fails
                    interp_func = interpolate.interp1d(
                        valid_idx, valid_vals,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(valid_vals[0], valid_vals[-1])
                    )

            # Interpolate missing values
            nan_idx = time_idx[nan_mask]
            X_imputed[nan_mask, col] = interp_func(nan_idx)

        return X_imputed

    def _check_stationarity(
        self,
        series: np.ndarray
    ) -> Tuple[bool, int]:
        """Check stationarity using Augmented Dickey-Fuller test.

        Returns:
            is_stationary: Whether series is stationary at current level
            diff_order: Recommended differencing order (0 if stationary)
        """
        # Remove any NaN for the test
        clean_series = series[~np.isnan(series)]

        if len(clean_series) < 20:
            # Not enough data for reliable ADF test
            return True, 0

        for diff_order in range(self.max_diff_order + 1):
            test_series = clean_series.copy()

            # Apply differencing
            for _ in range(diff_order):
                test_series = np.diff(test_series)

            if len(test_series) < 10:
                break

            try:
                adf_result = adfuller(test_series, autolag='AIC')
                p_value = adf_result[1]

                if p_value < self.adf_threshold:
                    return diff_order == 0, diff_order
            except Exception:
                # ADF test failed, assume stationary
                return True, 0

        # If still non-stationary after max differencing, use max order
        return False, self.max_diff_order

    def _apply_differencing(
        self,
        series: np.ndarray,
        order: int
    ) -> np.ndarray:
        """Apply differencing to achieve stationarity.

        For financial data, uses log-returns for positive series,
        otherwise uses first differences.
        """
        result = series.copy()

        for _ in range(order):
            # Check if all positive (can use log-returns)
            if np.all(result[~np.isnan(result)] > 0):
                # Log-returns: r_t = log(P_t) - log(P_{t-1})
                log_vals = np.log(result)
                diff_vals = np.diff(log_vals)
                result = np.concatenate([[np.nan], diff_vals])
            else:
                # First differences
                diff_vals = np.diff(result)
                result = np.concatenate([[np.nan], diff_vals])

        return result

    def _apply_dimensionality_reduction(
        self,
        X: np.ndarray,
        slow_indices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply PCA to slow block if dimensionality is too high.

        Per spec: If D > 50, apply PCA to slow block only,
        keeping fast variables explicit.
        """
        n_features = X.shape[1]
        pca_info: Dict[str, Any] = {}

        if n_features <= self.max_features:
            return X, pca_info

        if len(slow_indices) == 0:
            return X, pca_info

        # Extract slow block
        slow_block = X[:, slow_indices]
        n_slow = slow_block.shape[1]

        # Determine number of components to keep
        target_reduction = n_features - self.max_features
        n_components = max(1, n_slow - target_reduction)
        n_components = min(n_components, n_slow)

        # Fit PCA on slow block
        self.pca = PCA(n_components=n_components)
        slow_reduced = self.pca.fit_transform(slow_block)

        # Store PCA info for reconstruction
        pca_info['components'] = self.pca.components_
        pca_info['mean'] = self.pca.mean_
        pca_info['explained_variance'] = self.pca.explained_variance_ratio_.sum()

        # Reconstruct X with reduced slow block
        # Fast variables remain explicit
        fast_indices = np.setdiff1d(np.arange(n_features), slow_indices)
        fast_block = X[:, fast_indices]

        X_reduced = np.hstack([fast_block, slow_reduced])

        warnings.warn(
            f"Applied PCA to slow block: {n_slow} -> {n_components} dims "
            f"(explained variance: {pca_info['explained_variance']:.2%})"
        )

        return X_reduced, pca_info

    def _normalize(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Z-score normalization for OT-Path stability.

        Critical: The Optimal Transport path assumes noise ~ N(0,1)
        connects to data. High variance data causes gradient instability.
        """
        self.scaler = StandardScaler()
        X_normalized = self.scaler.fit_transform(X)

        return X_normalized, self.scaler.mean_, self.scaler.scale_

    def _classify_regimes(
        self,
        X: np.ndarray,
        Y_response: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Classify regimes using CTree-Lite.

        Uses statistical significance rather than impurity reduction
        to avoid overfitting noise in financial data.
        """
        self.ctree = CTree(
            alpha=self.ctree_alpha,
            min_split=self.ctree_min_split
        )

        try:
            self.ctree.fit(X, Y_response)
            regimes = self.ctree.predict(X)
            n_regimes = self.ctree.n_regimes
        except Exception as e:
            # If CTree fails, fall back to single regime
            warnings.warn(f"CTree failed ({e}), using single regime")
            regimes = np.zeros(len(X), dtype=int)
            n_regimes = 1

        return regimes, n_regimes

    def _discover_causal_order(
        self,
        X: np.ndarray,
        fast_indices: np.ndarray,
        slow_indices: np.ndarray
    ) -> np.ndarray:
        """Discover causal ordering using LiNGAM on residuals.

        Strategy:
        1. Slow variables come first (macro drives markets)
        2. Within fast block, use LiNGAM to determine microstructure
        3. LiNGAM assumes linear non-Gaussian relationships
        """
        n_features = X.shape[1]

        if len(fast_indices) < 2:
            # Not enough fast variables for causal discovery
            # Order: slow first, then fast
            result = np.concatenate([slow_indices, fast_indices])
            return result.astype(int) if len(result) > 0 else np.arange(n_features, dtype=int)

        # Extract fast block for LiNGAM
        fast_block = X[:, fast_indices]

        try:
            # Import LiNGAM dynamically to allow graceful fallback
            import lingam

            # Fit DirectLiNGAM to discover causal order
            model = lingam.DirectLiNGAM()
            model.fit(fast_block)

            # Get causal order (indices into fast_indices)
            fast_causal_order = model.causal_order_

            # Map back to original indices
            fast_ordered = fast_indices[fast_causal_order]

        except ImportError:
            warnings.warn("lingam not available. Using variance-based ordering.")
            fast_ordered = self._variance_based_order(fast_block, fast_indices)

        except Exception as e:
            warnings.warn(f"LiNGAM failed ({e}). Using variance-based ordering.")
            fast_ordered = self._variance_based_order(fast_block, fast_indices)

        # Final order: slow variables first (upstream), then fast (downstream)
        causal_order = np.concatenate([slow_indices, fast_ordered])

        return causal_order.astype(int)

    def _variance_based_order(
        self,
        X: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Fallback ordering: higher variance = faster reaction.

        Intuition: Market variables with higher variance tend to react
        faster to shocks and are more downstream in the causal chain.
        """
        variances = np.var(X, axis=0)
        # Lower variance first (slow), higher variance last (fast)
        order = np.argsort(variances)
        return indices[order]

    def transform(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        topology: DataTopology
    ) -> np.ndarray:
        """Transform new data using fitted preprocessing.

        Args:
            X: New data to transform
            topology: Previously fitted topology

        Returns:
            X_processed: Transformed and ordered data
        """
        # Convert to numpy array
        X_array, _ = _to_numpy_array(X)
        X_array = X_array.astype(np.float64)

        # Apply differencing to same columns
        for col in topology.differenced_cols:
            if col < X_array.shape[1]:
                X_array[:, col] = self._apply_differencing(X_array[:, col], 1)

        # Remove NaN rows from differencing
        valid_mask = ~np.any(np.isnan(X_array), axis=1)
        X_array = X_array[valid_mask]

        # Normalize using stored statistics
        X_array = (X_array - topology.scaler_mean) / topology.scaler_std

        # Reorder according to causal order
        X_array = X_array[:, topology.causal_order]

        return X_array

    def denormalize(
        self,
        X_normalized: np.ndarray,
        topology: DataTopology
    ) -> np.ndarray:
        """Reverse normalization to get original scale.

        Args:
            X_normalized: Normalized data in causal order
            topology: Topology containing scaler statistics

        Returns:
            X_original: Data in original scale (still in causal order)
        """
        # Reverse the causal ordering to align with scaler stats
        inverse_order = np.argsort(topology.causal_order)
        X_reordered = X_normalized[:, inverse_order]

        # Denormalize
        X_original = X_reordered * topology.scaler_std + topology.scaler_mean

        return X_original


def validate_for_ode(X: np.ndarray) -> None:
    """Validate that data is safe for ODE integration.

    Critical: Any NaN or Inf will cause the ODE solver to collapse.

    Raises:
        ValueError: If data contains NaN or Inf
    """
    if np.any(np.isnan(X)):
        nan_count = np.sum(np.isnan(X))
        raise ValueError(
            f"Data contains {nan_count} NaN values. "
            "ODE solver will fail. Apply imputation first."
        )

    if np.any(np.isinf(X)):
        inf_count = np.sum(np.isinf(X))
        raise ValueError(
            f"Data contains {inf_count} Inf values. "
            "ODE solver will fail. Check data range."
        )

    # Check for extreme values that might cause numerical issues
    max_val = np.max(np.abs(X))
    if max_val > 1e6:
        warnings.warn(
            f"Data contains extreme values (max abs: {max_val:.2e}). "
            "Consider normalization for numerical stability."
        )


def numpy_to_polars(X: np.ndarray, column_names: Optional[List[str]] = None) -> pl.DataFrame:
    """Convert numpy array to Polars DataFrame.

    Helper function for users who want to work with Polars DataFrames.

    Args:
        X: Numpy array of shape (n_samples, n_features)
        column_names: Optional list of column names

    Returns:
        df: Polars DataFrame
    """
    if column_names is None:
        column_names = [f"var_{i}" for i in range(X.shape[1])]

    return pl.DataFrame(
        {name: X[:, i] for i, name in enumerate(column_names)}
    )


def polars_to_numpy(df: pl.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Convert Polars DataFrame to numpy array with column names.

    Args:
        df: Polars DataFrame

    Returns:
        X: Numpy array
        columns: List of column names
    """
    return df.to_numpy().astype(np.float64), df.columns
