"""
Data Ingestion & Topology Module for Causal Conditional Flow Matching

Handles the complete preprocessing pipeline for time series data:
1. Validation and cleaning (NaN/Inf removal, Cubic Spline imputation)
2. Stationarity enforcement (ADF tests, differencing, log-returns)
3. Normalization (StandardScaler for OT-Path stability)
4. Regime classification (CTree-Lite based partitioning)
5. Causal graph discovery (CAM or LiNGAM on "Fast" block)
6. Variable reordering based on causal hierarchy

Causal Discovery Methods:
- CAM (Causal Additive Models): Default method. Handles non-linear relationships
  using additive non-parametric regression. Uses greedy sink search for
  topological ordering. Suitable for financial data with convexity and thresholds.
- LiNGAM: Alternative for linear systems. Assumes linear non-Gaussian relationships.
  Use for testing or when linearity assumption holds.

Critical Notes:
- ODE solvers are extremely sensitive to NaN/Inf - validation is mandatory
- StandardScaler is required for stable gradients in the OT-Path
- The causal_permutation_list must be saved with the model artifacts

Note: This module uses Polars for DataFrame operations (not pandas).

References:
- Bühlmann et al. (2014): CAM - Causal Additive Models
- Shimizu et al. (2006): LiNGAM - Linear Non-Gaussian Acyclic Model
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
        causal_discovery_method: Method for causal discovery ('cam' or 'lingam').
            Default is 'cam' (Causal Additive Models) which handles non-linear
            relationships. Use 'lingam' for linear systems testing.

    Example:
        >>> processor = DataProcessor()
        >>> topology = processor.fit_transform(
        ...     X_raw,
        ...     fast_vars=['VIX', 'SPX_ret'],
        ...     slow_vars=['GDP', 'CPI']
        ... )

        >>> # For linear systems, use LiNGAM:
        >>> processor = DataProcessor(causal_discovery_method='lingam')
    """

    SUPPORTED_CAUSAL_METHODS = ('cam', 'lingam')

    def __init__(
        self,
        adf_threshold: float = 0.05,
        max_diff_order: int = 2,
        ctree_alpha: float = 0.05,
        ctree_min_split: int = 20,
        max_features: int = 50,
        pca_variance_ratio: float = 0.95,
        causal_discovery_method: str = 'cam'
    ):
        if causal_discovery_method not in self.SUPPORTED_CAUSAL_METHODS:
            raise ValueError(
                f"Unsupported causal_discovery_method: {causal_discovery_method}. "
                f"Supported methods: {self.SUPPORTED_CAUSAL_METHODS}"
            )

        self.adf_threshold = adf_threshold
        self.max_diff_order = max_diff_order
        self.ctree_alpha = ctree_alpha
        self.ctree_min_split = ctree_min_split
        self.max_features = max_features
        self.pca_variance_ratio = pca_variance_ratio
        self.causal_discovery_method = causal_discovery_method

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
        """Discover causal ordering using configured method.

        Dispatches to CAM (default) or LiNGAM based on causal_discovery_method.

        Strategy:
        1. Slow variables come first (macro drives markets)
        2. Within fast block, use causal discovery to determine microstructure
        3. CAM handles non-linear relationships via additive models
        4. LiNGAM assumes linear non-Gaussian relationships

        Args:
            X: Normalized data matrix
            fast_indices: Indices of fast (market) variables
            slow_indices: Indices of slow (macro) variables

        Returns:
            causal_order: Permutation of variable indices in causal order
        """
        n_features = X.shape[1]

        if len(fast_indices) < 2:
            # Not enough fast variables for causal discovery
            # Order: slow first, then fast
            result = np.concatenate([slow_indices, fast_indices])
            return result.astype(int) if len(result) > 0 else np.arange(n_features, dtype=int)

        # Extract fast block for causal discovery
        fast_block = X[:, fast_indices]

        # Dispatch to appropriate method
        if self.causal_discovery_method == 'cam':
            fast_ordered = self._discover_causal_order_cam(fast_block, fast_indices)
        else:  # lingam
            fast_ordered = self._discover_causal_order_lingam(fast_block, fast_indices)

        # Final order: slow variables first (upstream), then fast (downstream)
        causal_order = np.concatenate([slow_indices, fast_ordered])

        return causal_order.astype(int)

    def _discover_causal_order_cam(
        self,
        X: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Discover causal ordering using CAM (Causal Additive Models).

        Uses greedy sink search algorithm:
        1. For each variable, fit additive non-parametric regression on all others
        2. Identify "sink" (variable best explained by others = lowest residual variance)
        3. Place sink last in ordering, remove from set, repeat
        4. Output is topological sort from root (most upstream) to sink (most downstream)

        This method handles non-linear relationships (convexity, thresholds) common
        in financial data that violate LiNGAM's linearity assumption.

        Args:
            X: Data matrix for fast block (n_samples, n_fast_vars)
            indices: Original indices of variables in full data

        Returns:
            ordered_indices: Original indices reordered by causal structure
        """
        try:
            # Try to use causal-learn CAM implementation
            from causallearn.search.FCMBased.lingam import CAM_UV
            return self._cam_causallearn(X, indices)
        except ImportError:
            pass

        # Fall back to our implementation of greedy sink search with GAM
        try:
            return self._cam_greedy_sink_search(X, indices)
        except Exception as e:
            warnings.warn(
                f"CAM failed ({e}). Using variance-based ordering. "
                "Consider installing 'causal-learn' for better CAM support."
            )
            return self._variance_based_order(X, indices)

    def _cam_greedy_sink_search(
        self,
        X: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Greedy sink search algorithm for CAM ordering.

        Implements the algorithm from Bühlmann et al. (2014):
        - Iteratively identify the "sink" (most downstream variable)
        - Sink is the variable best explained by all others (lowest residual variance)
        - Build ordering from sinks (last) to roots (first)

        Uses generalized additive models (GAM) via sklearn's SplineTransformer
        for non-parametric regression.

        Args:
            X: Data matrix (n_samples, n_vars)
            indices: Original indices of variables

        Returns:
            ordered_indices: Indices in causal order (roots first, sinks last)
        """
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline

        n_samples, n_vars = X.shape
        remaining = list(range(n_vars))
        ordering = []

        # Determine spline parameters based on sample size
        n_knots = min(5, max(2, n_samples // 50))

        while len(remaining) > 1:
            best_sink = None
            best_score = -np.inf  # We want highest R² (lowest residual variance)

            for i in remaining:
                # Get predictors (all other remaining variables)
                predictor_idx = [j for j in remaining if j != i]
                X_pred = X[:, predictor_idx]
                y = X[:, i]

                try:
                    # Fit GAM: y_i = sum_j f_j(x_j) + epsilon
                    # Using additive spline regression
                    model = make_pipeline(
                        SplineTransformer(
                            n_knots=n_knots,
                            degree=3,
                            include_bias=False
                        ),
                        Ridge(alpha=1.0)
                    )
                    model.fit(X_pred, y)

                    # Calculate R² score (higher = better explained = more sink-like)
                    score = model.score(X_pred, y)

                except Exception:
                    # If fitting fails, use simple correlation-based score
                    correlations = [np.abs(np.corrcoef(X[:, j], y)[0, 1])
                                    for j in predictor_idx]
                    score = np.mean([c for c in correlations if np.isfinite(c)])

                if score > best_score:
                    best_score = score
                    best_sink = i

            # Add sink to ordering (will be reversed at end)
            ordering.append(best_sink)
            remaining.remove(best_sink)

        # Add the last remaining variable (it's the root)
        if remaining:
            ordering.append(remaining[0])

        # Reverse: we built from sinks to roots, but want roots first
        ordering = ordering[::-1]

        # Map back to original indices
        return indices[np.array(ordering)]

    def _cam_causallearn(
        self,
        X: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Use causal-learn library's CAM implementation.

        Args:
            X: Data matrix (n_samples, n_vars)
            indices: Original indices of variables

        Returns:
            ordered_indices: Indices in causal order
        """
        try:
            # causal-learn CAM returns adjacency matrix, extract ordering
            from causallearn.search.ScoreBased.GES import ges
            from causallearn.search.FCMBased.lingam import DirectLiNGAM

            # Use GES (Greedy Equivalence Search) with BIC score as a proxy
            # since CAM implementation varies by version
            record = ges(X, score_func='local_score_BIC')
            adj_matrix = record['G'].graph

            # Convert adjacency to topological order
            return self._adjacency_to_order(adj_matrix, indices)

        except Exception as e:
            warnings.warn(f"causal-learn CAM failed ({e}), using greedy sink search.")
            return self._cam_greedy_sink_search(X, indices)

    def _adjacency_to_order(
        self,
        adj_matrix: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Convert adjacency matrix to topological ordering.

        Uses Kahn's algorithm for topological sort.

        Args:
            adj_matrix: Adjacency matrix where adj[i,j]=1 means i->j
            indices: Original indices of variables

        Returns:
            ordered_indices: Indices in topological order
        """
        n = len(adj_matrix)

        # Compute in-degrees
        # Handle different adjacency matrix formats
        # adj[i,j] = 1 means edge from i to j, so in-degree of j is sum of column j
        in_degree = np.zeros(n, dtype=int)

        for j in range(n):
            for i in range(n):
                if i != j and adj_matrix[i, j] != 0:
                    # Check for directed edge i -> j
                    # In CPDAG notation: adj[i,j]=1 and adj[j,i]=0 means i->j
                    if adj_matrix[j, i] == 0:
                        in_degree[j] += 1

        # Kahn's algorithm
        order = []
        queue = [i for i in range(n) if in_degree[i] == 0]

        while queue:
            node = queue.pop(0)
            order.append(node)

            for j in range(n):
                if node != j and adj_matrix[node, j] != 0 and adj_matrix[j, node] == 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        # If not all nodes added (cycle detected), fall back to variance-based
        if len(order) != n:
            warnings.warn("Graph has cycles, using variance-based ordering.")
            return self._variance_based_order(
                np.zeros((1, n)),  # Dummy, won't be used
                indices
            )

        return indices[np.array(order)]

    def _discover_causal_order_lingam(
        self,
        X: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """Discover causal ordering using LiNGAM.

        LiNGAM (Linear Non-Gaussian Acyclic Model) assumes:
        - Linear structural equations: x = Bx + e
        - Non-Gaussian error distributions
        - Acyclic causal graph

        Best suited for linear systems where these assumptions hold.
        For non-linear financial data, consider using 'cam' method instead.

        Args:
            X: Data matrix for fast block (n_samples, n_fast_vars)
            indices: Original indices of variables in full data

        Returns:
            ordered_indices: Original indices reordered by causal structure
        """
        try:
            # Import LiNGAM dynamically to allow graceful fallback
            import lingam

            # Fit DirectLiNGAM to discover causal order
            model = lingam.DirectLiNGAM()
            model.fit(X)

            # Get causal order (indices into fast_indices)
            fast_causal_order = model.causal_order_

            # Map back to original indices
            return indices[fast_causal_order]

        except ImportError:
            warnings.warn("lingam not available. Using variance-based ordering.")
            return self._variance_based_order(X, indices)

        except Exception as e:
            warnings.warn(f"LiNGAM failed ({e}). Using variance-based ordering.")
            return self._variance_based_order(X, indices)

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
