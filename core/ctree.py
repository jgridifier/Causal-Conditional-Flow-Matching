"""
CTree-Lite: Conditional Inference Tree Implementation

A recursive partitioning algorithm that selects splits based on statistical
significance (p-values) rather than impurity reduction. This avoids the bias
of standard CART/sklearn trees which tend to overfit continuous financial data.

Reference:
    Hothorn, T., Hornik, K., & Zeileis, A. (2006).
    Unbiased Recursive Partitioning: A Conditional Inference Framework.
    Journal of Computational and Graphical Statistics, 15(3), 651-674.

Mathematical Framework (Strasser-Weber):
    We test H0: Independence between input variable X_j and response Y.

    1. Transform response and input to influence functions (ranks)
    2. Compute linear statistic S_j = sum(Y * h(X_j))
    3. Standardize using conditional expectation and covariance
    4. Use chi-squared asymptotic approximation for p-values
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, rankdata
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple
import warnings


@dataclass
class LeafNode:
    """Terminal node containing regime assignment.

    Attributes:
        values: The Y values that fall into this leaf
        regime_id: Assigned regime identifier (set during tree construction)
        n_samples: Number of samples in this leaf
        mean_response: Mean of Y values (for regression interpretation)
    """
    values: np.ndarray
    regime_id: int = -1

    def __post_init__(self):
        self.n_samples = len(self.values)
        self.mean_response = np.mean(self.values, axis=0) if len(self.values) > 0 else None

    def predict(self, x: np.ndarray) -> int:
        """Return the regime ID for this leaf."""
        return self.regime_id

    def __repr__(self) -> str:
        return f"LeafNode(regime={self.regime_id}, n={self.n_samples})"


@dataclass
class DecisionNode:
    """Internal decision node with statistical split criterion.

    Attributes:
        var_idx: Index of the variable used for splitting
        cut: Threshold value for the split (left <= cut, right > cut)
        p_value: P-value from independence test (lower = more significant)
        left: Left child node (x <= cut)
        right: Right child node (x > cut)
        var_name: Optional variable name for interpretability
    """
    var_idx: int
    cut: float
    p_value: float
    left: Union[LeafNode, DecisionNode]
    right: Union[LeafNode, DecisionNode]
    var_name: Optional[str] = None

    def predict(self, x: np.ndarray) -> int:
        """Traverse tree to find regime ID."""
        if x[self.var_idx] <= self.cut:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def __repr__(self) -> str:
        name = self.var_name or f"X[{self.var_idx}]"
        return f"DecisionNode({name} <= {self.cut:.4f}, p={self.p_value:.4f})"


class CTree:
    """Conditional Inference Tree for Regime Classification.

    Implements the Strasser-Weber framework for unbiased recursive partitioning.
    Uses statistical significance (p-values) for variable selection and stopping,
    rather than greedy impurity minimization.

    This approach is particularly suited for financial time series where:
    - Standard trees tend to overfit noise
    - We need interpretable, statistically justified regime boundaries
    - The relationship between variables may be weak but meaningful

    Attributes:
        alpha: Significance level for stopping (default 0.05)
        min_split: Minimum samples required to attempt a split
        min_leaf: Minimum samples required in each leaf
        max_depth: Maximum tree depth (None for unlimited)
        n_quantiles: Number of quantile-based split candidates to evaluate

    Example:
        >>> ctree = CTree(alpha=0.05, min_split=20)
        >>> ctree.fit(X_features, Y_response)
        >>> regimes = ctree.predict(X_new)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_split: int = 20,
        min_leaf: int = 10,
        max_depth: Optional[int] = None,
        n_quantiles: int = 20
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if min_split < 2:
            raise ValueError(f"min_split must be >= 2, got {min_split}")
        if min_leaf < 1:
            raise ValueError(f"min_leaf must be >= 1, got {min_leaf}")
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")

        self.alpha = alpha
        self.min_split = min_split
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.n_quantiles = n_quantiles

        self.tree: Optional[Union[LeafNode, DecisionNode]] = None
        self.n_regimes: int = 0
        self.feature_names: Optional[List[str]] = None
        self._regime_counter: int = 0

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "CTree":
        """Fit the conditional inference tree.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
               E.g., VIX, Interest Rates, Credit Spreads
            Y: Response matrix of shape (n_samples,) or (n_samples, n_targets)
               E.g., Future GDP growth, SPX returns
            feature_names: Optional names for interpretability

        Returns:
            self: Fitted tree instance
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if len(X) != len(Y):
            raise ValueError(f"X and Y must have same length: {len(X)} != {len(Y)}")
        if len(X) < self.min_split:
            raise ValueError(f"Need at least {self.min_split} samples, got {len(X)}")

        # Check for NaN/Inf
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(Y)):
            raise ValueError("X and Y must not contain NaN or Inf values")

        self.feature_names = feature_names
        self._regime_counter = 0

        # Build tree recursively
        self.tree = self._recursive_split(X, Y, depth=0)
        self.n_regimes = self._regime_counter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime labels for input samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            regimes: Array of regime labels of shape (n_samples,)
        """
        if self.tree is None:
            raise RuntimeError("Tree not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        regimes = np.array([self.tree.predict(x) for x in X])
        return regimes

    def _recursive_split(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        depth: int
    ) -> Union[LeafNode, DecisionNode]:
        """Recursively partition the data based on significance testing.

        Args:
            X: Feature subset for current node
            Y: Response subset for current node
            depth: Current tree depth

        Returns:
            node: Either a LeafNode (terminal) or DecisionNode (internal)
        """
        n_samples = len(Y)

        # Check stopping conditions
        if n_samples < self.min_split:
            return self._make_leaf(Y)

        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(Y)

        # Variable Selection via Independence Testing
        # Test H0: X_j independent of Y for each feature j
        p_values = np.array([
            self._test_independence(X[:, j], Y)
            for j in range(X.shape[1])
        ])

        best_var = np.argmin(p_values)
        min_p = p_values[best_var]

        # Stopping criterion: If no significant association, create leaf
        if min_p > self.alpha:
            return self._make_leaf(Y)

        # Split Selection: Find optimal cut point for selected variable
        cut_point, can_split = self._find_best_cut(X[:, best_var], Y)

        if not can_split:
            return self._make_leaf(Y)

        # Create split masks
        left_mask = X[:, best_var] <= cut_point
        right_mask = ~left_mask

        # Verify minimum leaf size
        if np.sum(left_mask) < self.min_leaf or np.sum(right_mask) < self.min_leaf:
            return self._make_leaf(Y)

        # Get variable name if available
        var_name = None
        if self.feature_names is not None and best_var < len(self.feature_names):
            var_name = self.feature_names[best_var]

        # Recurse on children
        return DecisionNode(
            var_idx=best_var,
            cut=cut_point,
            p_value=min_p,
            left=self._recursive_split(X[left_mask], Y[left_mask], depth + 1),
            right=self._recursive_split(X[right_mask], Y[right_mask], depth + 1),
            var_name=var_name
        )

    def _make_leaf(self, Y: np.ndarray) -> LeafNode:
        """Create a leaf node and assign regime ID."""
        leaf = LeafNode(values=Y.copy(), regime_id=self._regime_counter)
        self._regime_counter += 1
        return leaf

    def _test_independence(self, x_col: np.ndarray, Y: np.ndarray) -> float:
        """Test independence between a single feature and response using Strasser-Weber.

        Implements the linear rank statistic with asymptotic chi-squared approximation.

        Mathematical Details:
            1. Convert x to ranks h(x) = rank(x) for robustness
            2. Compute linear statistic: T = Y^T @ h(x)
            3. Under H0 (independence), T is asymptotically normal
            4. Compute standardized statistic S = (T - E[T])^T @ Var[T]^-1 @ (T - E[T])
            5. S ~ chi^2(df=dim(Y)) asymptotically

        Args:
            x_col: Single feature column of shape (n,)
            Y: Response matrix of shape (n, d)

        Returns:
            p_value: P-value from chi-squared test (smaller = more significant)
        """
        n = len(x_col)

        # Handle edge cases
        if n < 3:
            return 1.0

        # Check for zero variance
        if np.var(x_col) < 1e-10:
            return 1.0
        if np.all(np.var(Y, axis=0) < 1e-10):
            return 1.0

        # Step 1: Rank transformation of X (influence function)
        # Using average ranks for ties
        ranks = rankdata(x_col, method='average')

        # Center the ranks
        h = ranks - np.mean(ranks)

        # Step 2: Compute linear statistic T = Y^T @ h
        # For multivariate Y, this gives a vector
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Center Y for the statistic
        Y_centered = Y - np.mean(Y, axis=0)

        # Linear statistic: sum of Y weighted by transformed X
        T = Y_centered.T @ h  # Shape: (d,)

        # Step 3: Compute conditional expectation and covariance under H0
        # Under permutation null:
        # E[T | ranks, Y] = 0 (since Y is centered)
        # Var[T | ranks, Y] = (1/(n-1)) * sum(h^2) * Cov(Y)

        # Variance of the influence function
        h_var = np.sum(h ** 2)

        # Covariance matrix of Y
        # Using (n-1) denominator for unbiased estimate
        Y_cov = (Y_centered.T @ Y_centered) / (n - 1)

        # Conditional covariance of T under H0
        # Var(T) = h_var * Y_cov / (n-1) * n/(n-1)
        # Simplified: Var(T) ≈ h_var * Y_cov * n / (n-1)^2
        scale_factor = n / ((n - 1) ** 2)
        T_cov = h_var * Y_cov * scale_factor

        # Step 4: Compute test statistic
        # S = T^T @ Var(T)^{-1} @ T
        d = Y.shape[1]

        try:
            if d == 1:
                # Univariate case: simple standardization
                T_var = T_cov[0, 0]
                if T_var < 1e-10:
                    return 1.0
                S = (T[0] ** 2) / T_var
            else:
                # Multivariate case: need to invert covariance
                # Add small regularization for numerical stability
                T_cov_reg = T_cov + np.eye(d) * 1e-8
                T_cov_inv = np.linalg.inv(T_cov_reg)
                S = T @ T_cov_inv @ T
        except np.linalg.LinAlgError:
            # If matrix is singular, return non-significant
            return 1.0

        # Step 5: P-value from chi-squared distribution
        # Degrees of freedom = dimension of Y
        p_value = 1.0 - chi2.cdf(S, df=d)

        # Ensure valid p-value
        return np.clip(p_value, 0.0, 1.0)

    def _find_best_cut(
        self,
        x_col: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, bool]:
        """Find optimal split point using exhaustive search over quantiles.

        Uses a two-sample test approach: for each candidate split,
        compute the between-group variance (separation criterion).

        Args:
            x_col: Single feature column to split on
            Y: Response values

        Returns:
            cut_point: Optimal threshold value
            can_split: Whether a valid split was found
        """
        n = len(x_col)

        # Get unique sorted values for candidate splits
        unique_vals = np.unique(x_col)

        if len(unique_vals) < 2:
            return 0.0, False

        # Use quantile-based candidates for efficiency
        if len(unique_vals) > self.n_quantiles:
            # Sample quantiles to reduce computation
            quantiles = np.linspace(0, 1, self.n_quantiles + 2)[1:-1]
            candidates = np.quantile(x_col, quantiles)
            candidates = np.unique(candidates)
        else:
            # Use midpoints between unique values
            candidates = (unique_vals[:-1] + unique_vals[1:]) / 2

        if len(candidates) == 0:
            return 0.0, False

        # Evaluate each candidate using between-group sum of squares
        best_score = -np.inf
        best_cut = candidates[0]

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        overall_mean = np.mean(Y, axis=0)

        for cut in candidates:
            left_mask = x_col <= cut
            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            # Enforce minimum leaf size
            if n_left < self.min_leaf or n_right < self.min_leaf:
                continue

            # Between-group sum of squares (BGSS)
            # Measures separation between groups
            left_mean = np.mean(Y[left_mask], axis=0)
            right_mean = np.mean(Y[right_mask], axis=0)

            # Weighted sum of squared deviations from overall mean
            bgss = (
                n_left * np.sum((left_mean - overall_mean) ** 2) +
                n_right * np.sum((right_mean - overall_mean) ** 2)
            )

            if bgss > best_score:
                best_score = bgss
                best_cut = cut

        if best_score == -np.inf:
            return 0.0, False

        return best_cut, True

    def get_regime_mapping(self) -> dict:
        """Get mapping of regime IDs to their statistics.

        Returns:
            mapping: Dict mapping regime_id to {n_samples, mean_response}
        """
        if self.tree is None:
            raise RuntimeError("Tree not fitted. Call fit() first.")

        mapping = {}
        self._collect_leaves(self.tree, mapping)
        return mapping

    def _collect_leaves(
        self,
        node: Union[LeafNode, DecisionNode],
        mapping: dict
    ) -> None:
        """Recursively collect leaf node statistics."""
        if isinstance(node, LeafNode):
            mapping[node.regime_id] = {
                'n_samples': node.n_samples,
                'mean_response': node.mean_response
            }
        else:
            self._collect_leaves(node.left, mapping)
            self._collect_leaves(node.right, mapping)

    def print_tree(self, node: Optional[Union[LeafNode, DecisionNode]] = None, indent: str = "") -> str:
        """Get string representation of tree structure."""
        if node is None:
            node = self.tree
        if node is None:
            return "Tree not fitted"

        lines = []
        if isinstance(node, LeafNode):
            lines.append(f"{indent}└── Regime {node.regime_id} (n={node.n_samples})")
        else:
            name = node.var_name or f"X[{node.var_idx}]"
            lines.append(f"{indent}├── {name} <= {node.cut:.4f} (p={node.p_value:.4f})")
            lines.append(self.print_tree(node.left, indent + "│   "))
            lines.append(f"{indent}├── {name} > {node.cut:.4f}")
            lines.append(self.print_tree(node.right, indent + "│   "))

        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.tree is None:
            return "CTree(not fitted)"
        return f"CTree(n_regimes={self.n_regimes}, alpha={self.alpha})"
