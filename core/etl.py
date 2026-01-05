"""
Data Ingestion & Topology Module for Causal Conditional Flow Matching

Handles the complete preprocessing pipeline for time series data:
1. Validation and cleaning (NaN/Inf removal, Cubic Spline imputation)
2. Stationarity enforcement (ADF tests, differencing, log-returns)
3. Normalization (StandardScaler for OT-Path stability)
4. Regime classification (CTree-Lite based partitioning)
5. Causal graph discovery (CAM or LiNGAM on ALL variables)
6. Variable reordering based on causal hierarchy

Terminology (aligned with paper):
- FAST (informationally fast) = UPSTREAM = drives other variables = FIRST in ordering
  Example: Interest rates, Fed policy decisions
- SLOW (informationally slow) = DOWNSTREAM = reacts to all others = SINK = LAST in ordering
  Example: S&P 500 returns, VIX (reacts to everything)

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
            (FAST/upstream variables first, SLOW/downstream sinks last)
        variable_names: Names of variables in causal order
        scaler_mean: Mean values for denormalization
        scaler_std: Standard deviations for denormalization
        n_regimes: Number of detected regimes
        differenced_cols: Columns that were differenced for stationarity
        pca_components: PCA components if dimensionality reduction was applied
        upstream_indices: Indices of upstream (fast) variables after reordering
        downstream_indices: Indices of downstream (slow/sink) variables after reordering
    """
    X_processed: np.ndarray
    regimes: np.ndarray
    regime_embeddings: np.ndarray
    causal_order: np.ndarray
    variable_names: List[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    n_regimes: int
    differenced_cols: List[int] = field(default_factory=list)
    pca_components: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None
    # Deprecated aliases for backward compatibility
    fast_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    slow_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

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

    Terminology (aligned with paper Section 3-4):
    - FAST = informationally fast = UPSTREAM = drives other variables = FIRST
    - SLOW = informationally slow = DOWNSTREAM = reacts to all = SINK = LAST

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
        >>> # Automatic causal discovery on all variables
        >>> processor = DataProcessor()
        >>> topology = processor.fit_transform(X_raw)

        >>> # With optional ordering hints (Bayesian prior)
        >>> topology = processor.fit_transform(
        ...     X_raw,
        ...     causal_order_hint=['Rates', 'Spreads', 'VIX']  # Rates before Spreads before VIX
        ... )
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
        causal_order_hint: Optional[List[Union[str, int]]] = None,
        regime_response_vars: Optional[List[str]] = None,
        # Deprecated parameters for backward compatibility
        fast_vars: Optional[List[str]] = None,
        slow_vars: Optional[List[str]] = None
    ) -> DataTopology:
        """Full preprocessing pipeline: validate, clean, order, label.

        CAM discovers the causal ordering across ALL variables automatically.
        Optionally, you can provide ordering hints (Bayesian prior) to constrain
        the discovery while still testing against all variables.

        Args:
            X: Raw time series data of shape (T, D)
               - If Polars DataFrame, column names are used for variable identification
               - If ndarray, variable names are auto-generated as var_0, var_1, ...
            causal_order_hint: Optional list of variable names/indices specifying
                a partial ordering constraint. Variables listed will maintain their
                relative order in the final result.
                Example: ['Rates', 'Spreads', 'VIX'] means Rates must come before
                Spreads, which must come before VIX. Other variables are placed
                based on CAM discovery.
            regime_response_vars: Variables to use as response for regime detection
                (defaults to downstream/sink variables if not specified)
            fast_vars: DEPRECATED - use causal_order_hint instead
            slow_vars: DEPRECATED - use causal_order_hint instead

        Returns:
            DataTopology containing all processed data and metadata
        """
        # Convert to numpy array with column names
        X_array, variable_names = _to_numpy_array(X)
        n_vars = len(variable_names)

        # Handle deprecated parameters
        if fast_vars is not None or slow_vars is not None:
            warnings.warn(
                "fast_vars and slow_vars are deprecated. Use causal_order_hint instead. "
                "CAM now discovers ordering across ALL variables automatically.",
                DeprecationWarning
            )
            # Convert old-style to new-style hint (fast_vars should come first)
            if causal_order_hint is None and fast_vars is not None:
                causal_order_hint = list(fast_vars)

        # Parse ordering hints
        ordering_constraints = self._parse_ordering_hints(
            variable_names, causal_order_hint
        )

        # Step A: Validation and Cleaning
        X_clean, differenced_cols = self._validate_and_clean(X_array)

        # Step B: Dimensionality Reduction (if needed)
        X_reduced, pca_info = self._apply_dimensionality_reduction(X_clean)

        # Step C: Normalize data
        X_normalized, scaler_mean, scaler_std = self._normalize(X_reduced)

        # Step D: Causal Graph Discovery on ALL variables
        causal_order = self._discover_causal_order(
            X_normalized, ordering_constraints
        )

        # Reorder data according to causal structure
        X_ordered = X_normalized[:, causal_order]
        ordered_names = [variable_names[i] for i in causal_order]

        # Step E: Regime Classification with CTree
        # Use downstream (sink) variables as regime indicators by default
        if regime_response_vars is None:
            # Last 1/3 of variables are most downstream (sinks)
            n_downstream = max(1, n_vars // 3)
            regime_features = X_ordered[:, -n_downstream:]
        else:
            regime_idx = [ordered_names.index(v) for v in regime_response_vars if v in ordered_names]
            regime_features = X_ordered[:, regime_idx] if regime_idx else X_ordered

        regimes, n_regimes = self._classify_regimes(X_ordered, regime_features)

        # Create regime embeddings (one-hot)
        regime_embeddings = np.eye(n_regimes)[regimes]

        self._is_fitted = True

        # For backward compatibility, set fast/slow indices
        # Fast = first half (upstream), Slow = second half (downstream)
        mid = n_vars // 2
        fast_indices = np.arange(mid)
        slow_indices = np.arange(mid, n_vars)

        return DataTopology(
            X_processed=X_ordered,
            regimes=regimes,
            regime_embeddings=regime_embeddings,
            causal_order=causal_order,
            variable_names=ordered_names,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            n_regimes=n_regimes,
            differenced_cols=differenced_cols,
            pca_components=pca_info.get('components'),
            pca_mean=pca_info.get('mean'),
            fast_indices=fast_indices,
            slow_indices=slow_indices
        )

    def _parse_ordering_hints(
        self,
        variable_names: List[str],
        causal_order_hint: Optional[List[Union[str, int]]]
    ) -> List[Tuple[int, int]]:
        """Parse user-provided ordering hints into pairwise constraints.

        Args:
            variable_names: List of all variable names
            causal_order_hint: User-provided partial ordering

        Returns:
            constraints: List of (i, j) tuples meaning variable i must come before j
        """
        if causal_order_hint is None or len(causal_order_hint) < 2:
            return []

        # Convert to indices
        indices = []
        for v in causal_order_hint:
            if isinstance(v, str):
                if v in variable_names:
                    indices.append(variable_names.index(v))
                else:
                    warnings.warn(f"Variable '{v}' not found in data, ignoring constraint.")
            else:
                if 0 <= v < len(variable_names):
                    indices.append(v)
                else:
                    warnings.warn(f"Index {v} out of range, ignoring constraint.")

        # Create pairwise constraints: each element must come before all following elements
        constraints = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                constraints.append((indices[i], indices[j]))

        return constraints

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
        X: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply PCA if dimensionality is too high.

        Per spec: If D > 50, apply PCA to reduce dimensions.
        """
        n_features = X.shape[1]
        pca_info: Dict[str, Any] = {}

        if n_features <= self.max_features:
            return X, pca_info

        # Fit PCA
        n_components = self.max_features
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)

        # Store PCA info for reconstruction
        pca_info['components'] = self.pca.components_
        pca_info['mean'] = self.pca.mean_
        pca_info['explained_variance'] = self.pca.explained_variance_ratio_.sum()

        warnings.warn(
            f"Applied PCA: {n_features} -> {n_components} dims "
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
        constraints: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Discover causal ordering using configured method on ALL variables.

        The ordering follows the paper's terminology:
        - FIRST (low index) = FAST = informationally fast = upstream = drives others
        - LAST (high index) = SLOW = informationally slow = downstream = sinks

        Args:
            X: Normalized data matrix (all variables)
            constraints: Pairwise ordering constraints from user hints

        Returns:
            causal_order: Permutation of variable indices
                (upstream/fast first, downstream/slow sinks last)
        """
        n_features = X.shape[1]

        if n_features < 2:
            return np.arange(n_features, dtype=int)

        # Dispatch to appropriate method
        if self.causal_discovery_method == 'cam':
            return self._discover_causal_order_cam(X, constraints)
        else:  # lingam
            return self._discover_causal_order_lingam(X, constraints)

    def _discover_causal_order_cam(
        self,
        X: np.ndarray,
        constraints: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Discover causal ordering using CAM (Causal Additive Models).

        Uses CONSTRAINED greedy sink search algorithm:
        1. For each variable, fit additive non-parametric regression on all others
        2. Identify "sink" (variable best explained by others = highest R²)
        3. Check sink doesn't violate user constraints
        4. Place sink LAST in ordering, remove from set, repeat
        5. Output is topological sort: FAST (upstream) first, SLOW (sinks) last

        Args:
            X: Data matrix (n_samples, n_features)
            constraints: Pairwise (i, j) constraints where i must come before j

        Returns:
            causal_order: Variable indices in causal order
        """
        try:
            return self._cam_constrained_greedy_sink_search(X, constraints)
        except Exception as e:
            warnings.warn(
                f"CAM failed ({e}). Using variance-based ordering. "
                "Consider installing 'causal-learn' for better CAM support."
            )
            return self._variance_based_order(X, constraints)

    def _cam_constrained_greedy_sink_search(
        self,
        X: np.ndarray,
        constraints: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Constrained greedy sink search algorithm for CAM ordering.

        Implements the algorithm from Bühlmann et al. (2014) with added
        support for user-specified ordering constraints (Bayesian prior).

        The algorithm respects constraints by only allowing a variable to be
        selected as a sink if all variables that should come AFTER it (per
        constraints) have already been placed.

        Args:
            X: Data matrix (n_samples, n_vars)
            constraints: List of (i, j) tuples meaning i must come before j

        Returns:
            ordered_indices: Indices in causal order (roots first, sinks last)
        """
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline

        n_samples, n_vars = X.shape
        remaining = set(range(n_vars))
        ordering = []  # Will be built from sinks to roots, then reversed

        # Build constraint lookup: for each variable, which variables must come AFTER it?
        must_come_after = {i: set() for i in range(n_vars)}
        for (before, after) in constraints:
            must_come_after[before].add(after)

        # Determine spline parameters based on sample size
        n_knots = min(5, max(2, n_samples // 50))

        while len(remaining) > 1:
            best_sink = None
            best_score = -np.inf

            # Find valid sink candidates
            # A variable can be a sink only if all variables that must come AFTER it
            # have already been placed (i.e., are not in remaining)
            valid_candidates = []
            for i in remaining:
                # Check if all "must come after" variables have been placed
                unplaced_after = must_come_after[i] & remaining
                if len(unplaced_after) == 0:
                    valid_candidates.append(i)

            if not valid_candidates:
                # Constraint conflict - fall back to ignoring constraints
                warnings.warn(
                    "Ordering constraints are inconsistent. Ignoring constraints."
                )
                valid_candidates = list(remaining)

            for i in valid_candidates:
                # Get predictors (all other remaining variables)
                remaining_list = list(remaining)
                predictor_idx = [remaining_list.index(j) for j in remaining if j != i]
                X_remaining = X[:, list(remaining)]
                X_pred = X_remaining[:, predictor_idx]
                y = X[:, i]

                if len(predictor_idx) == 0:
                    score = 0.0
                else:
                    try:
                        # Fit GAM: y_i = sum_j f_j(x_j) + epsilon
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
                        # If fitting fails, use correlation-based score
                        correlations = []
                        for j_idx, j in enumerate(predictor_idx):
                            j_orig = remaining_list[j]
                            corr = np.abs(np.corrcoef(X[:, j_orig], y)[0, 1])
                            if np.isfinite(corr):
                                correlations.append(corr)
                        score = np.mean(correlations) if correlations else 0.0

                if score > best_score:
                    best_score = score
                    best_sink = i

            if best_sink is None:
                # Should not happen, but fallback
                best_sink = list(remaining)[0]

            # Add sink to ordering (will be reversed at end)
            ordering.append(best_sink)
            remaining.remove(best_sink)

        # Add the last remaining variable (it's the root - most upstream)
        if remaining:
            ordering.append(list(remaining)[0])

        # Reverse: we built from sinks to roots, but want roots (FAST) first
        ordering = ordering[::-1]

        return np.array(ordering, dtype=int)

    def _discover_causal_order_lingam(
        self,
        X: np.ndarray,
        constraints: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Discover causal ordering using LiNGAM.

        LiNGAM (Linear Non-Gaussian Acyclic Model) assumes:
        - Linear structural equations: x = Bx + e
        - Non-Gaussian error distributions
        - Acyclic causal graph

        Best suited for linear systems where these assumptions hold.
        For non-linear financial data, consider using 'cam' method instead.

        Args:
            X: Data matrix (n_samples, n_features)
            constraints: Pairwise constraints (used for fallback only)

        Returns:
            causal_order: Variable indices in causal order
        """
        try:
            import lingam

            model = lingam.DirectLiNGAM()
            model.fit(X)

            return np.array(model.causal_order_, dtype=int)

        except ImportError:
            warnings.warn("lingam not available. Using variance-based ordering.")
            return self._variance_based_order(X, constraints)

        except Exception as e:
            warnings.warn(f"LiNGAM failed ({e}). Using variance-based ordering.")
            return self._variance_based_order(X, constraints)

    def _variance_based_order(
        self,
        X: np.ndarray,
        constraints: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Fallback ordering based on variance.

        Intuition from Section 3 of paper:
        - Variables that react to everything (sinks) have HIGHER variance
        - Fundamental drivers have LOWER variance (more stable)
        - Therefore: LOW variance first (upstream/FAST), HIGH variance last (downstream/SLOW)

        Args:
            X: Data matrix
            constraints: Pairwise constraints (respected if possible)

        Returns:
            ordered_indices: Indices in causal order
        """
        n_vars = X.shape[1]
        variances = np.var(X, axis=0)

        # Start with variance-based ordering
        # Low variance = upstream (FAST) = first
        # High variance = downstream (SLOW/sink) = last
        base_order = np.argsort(variances)

        if not constraints:
            return base_order

        # Adjust for constraints using topological sort
        # Build a graph and do constrained sorting
        order_list = list(base_order)

        for (before, after) in constraints:
            before_pos = order_list.index(before)
            after_pos = order_list.index(after)
            if before_pos > after_pos:
                # Constraint violated, swap
                order_list.remove(before)
                order_list.insert(after_pos, before)

        return np.array(order_list, dtype=int)

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

    # Backward compatibility methods
    def _parse_variable_blocks(
        self,
        variable_names: List[str],
        fast_vars: Optional[List[str]],
        slow_vars: Optional[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """DEPRECATED: Parse fast and slow variable specifications."""
        warnings.warn(
            "_parse_variable_blocks is deprecated. CAM now discovers ordering automatically.",
            DeprecationWarning
        )
        n_vars = len(variable_names)
        mid = n_vars // 2
        return np.arange(mid), np.arange(mid, n_vars)


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
