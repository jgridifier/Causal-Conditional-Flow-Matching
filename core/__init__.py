"""
Causal Conditional Flow Matching (C-CFM) Engine

A generative time-series framework based on Continuous Normalizing Flows
with causal structure enforcement for economic scenario generation.

Core Components:
    - CTree: Conditional inference tree for regime classification
    - DataProcessor: ETL pipeline with causal graph discovery
    - VelocityNetwork: Masked neural network with FiLM conditioning
    - FlowMatchingTrainer: Simulation-free CFM training
    - ODESolver: Adaptive ODE integration with stress testing

Quick Start:
    >>> from core import CausalFlowMatcher
    >>>
    >>> # Initialize with data
    >>> cfm = CausalFlowMatcher()
    >>> cfm.fit(X, fast_vars=['VIX', 'SPX_ret'], slow_vars=['GDP', 'CPI'])
    >>>
    >>> # Generate scenarios
    >>> scenarios = cfm.sample(n_samples=1000)
    >>>
    >>> # Stress test
    >>> stressed = cfm.shock('VIX', magnitude=3.0)

References:
    - Lipman et al. (2022): Flow Matching for Generative Modeling
    - Hothorn et al. (2006): Unbiased Recursive Partitioning
    - Shimizu et al. (2006): LiNGAM - Linear Non-Gaussian Acyclic Model
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Union, Dict, List, Any
from pathlib import Path
import warnings
import json

# Core modules
from .ctree import CTree, LeafNode, DecisionNode
from .etl import DataProcessor, DataTopology, validate_for_ode
from .network import (
    VelocityNetwork,
    VelocityNetworkUnconstrained,
    MaskedLinear,
    FiLMLayer,
    SinusoidalTimeEmbedding,
    create_causal_mask
)
from .trainer import (
    FlowMatchingTrainer,
    TrainingConfig,
    TrainingState,
    CFMLoss,
    create_trainer
)
from .solver import (
    ODESolver,
    SolverConfig,
    ScenarioGenerator,
    VelocityFieldWrapper,
    GuidedVelocityField
)


__version__ = "1.0.0"
__author__ = "C-CFM Team"


# Public API
__all__ = [
    # High-level interface
    "CausalFlowMatcher",

    # Tree module
    "CTree",
    "LeafNode",
    "DecisionNode",

    # ETL module
    "DataProcessor",
    "DataTopology",
    "validate_for_ode",

    # Network module
    "VelocityNetwork",
    "VelocityNetworkUnconstrained",
    "MaskedLinear",
    "FiLMLayer",
    "SinusoidalTimeEmbedding",
    "create_causal_mask",

    # Trainer module
    "FlowMatchingTrainer",
    "TrainingConfig",
    "TrainingState",
    "CFMLoss",
    "create_trainer",

    # Solver module
    "ODESolver",
    "SolverConfig",
    "ScenarioGenerator",
    "VelocityFieldWrapper",
    "GuidedVelocityField",
]


class CausalFlowMatcher:
    """High-level API for Causal Conditional Flow Matching.

    This class provides a unified interface for the complete C-CFM pipeline:
    data preprocessing, model training, and scenario generation.

    The workflow enforces causal structure in generated scenarios:
    1. "Slow" macro variables (GDP, CPI) are upstream
    2. "Fast" market variables (VIX, returns) are downstream
    3. Shocks to fast variables propagate to slow (and to other fast)
    4. Shocks to slow variables only affect themselves

    Attributes:
        processor: Data preprocessing pipeline
        topology: Processed data topology
        model: Velocity network
        trainer: Training handler
        solver: ODE solver for generation
        generator: High-level scenario generator

    Args:
        hidden_dim: Hidden dimension for velocity network (default 256)
        n_layers: Number of residual blocks (default 4)
        device: Computation device (default 'auto')

    Example:
        >>> cfm = CausalFlowMatcher(hidden_dim=256)
        >>>
        >>> # Fit to data
        >>> cfm.fit(
        ...     X,
        ...     fast_vars=['VIX', 'SPX_ret', 'Credit_Spread'],
        ...     slow_vars=['GDP_Growth', 'CPI', 'Unemployment']
        ... )
        >>>
        >>> # Train the model
        >>> cfm.train(n_epochs=500)
        >>>
        >>> # Generate baseline scenarios
        >>> baseline = cfm.sample(1000)
        >>>
        >>> # Generate stress scenarios
        >>> stressed = cfm.shock('VIX', magnitude=3.0, n_samples=1000)
        >>>
        >>> # Save for later use
        >>> cfm.save('model_checkpoint.pt')
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        regime_embed_dim: int = 32,
        dropout: float = 0.0,
        device: str = 'auto'
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.time_embed_dim = time_embed_dim
        self.regime_embed_dim = regime_embed_dim
        self.dropout = dropout

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Components (initialized during fit/load)
        self.processor: Optional[DataProcessor] = None
        self.topology: Optional[DataTopology] = None
        self.model: Optional[VelocityNetwork] = None
        self.trainer: Optional[FlowMatchingTrainer] = None
        self.solver: Optional[ODESolver] = None
        self.generator: Optional[ScenarioGenerator] = None

        self._is_fitted: bool = False
        self._is_trained: bool = False

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        fast_vars: Optional[List[str]] = None,
        slow_vars: Optional[List[str]] = None,
        regime_response_vars: Optional[List[str]] = None,
        adf_threshold: float = 0.05,
        ctree_alpha: float = 0.05,
        ctree_min_split: int = 20
    ) -> "CausalFlowMatcher":
        """Fit the preprocessing pipeline and initialize the model.

        This performs:
        1. Data validation and cleaning (NaN imputation)
        2. Stationarity enforcement (ADF test + differencing)
        3. Regime classification (CTree-Lite)
        4. Causal graph discovery (LiNGAM on fast block)
        5. Model initialization with causal masking

        Args:
            X: Raw time series data of shape (T, D)
            fast_vars: Names of fast (market) variables
            slow_vars: Names of slow (macro) variables
            regime_response_vars: Variables for regime detection
            adf_threshold: P-value threshold for stationarity test
            ctree_alpha: Significance threshold for regime splits
            ctree_min_split: Minimum samples per regime

        Returns:
            self: Fitted instance
        """
        # Initialize processor
        self.processor = DataProcessor(
            adf_threshold=adf_threshold,
            ctree_alpha=ctree_alpha,
            ctree_min_split=ctree_min_split
        )

        # Process data
        self.topology = self.processor.fit_transform(
            X,
            fast_vars=fast_vars,
            slow_vars=slow_vars,
            regime_response_vars=regime_response_vars
        )

        # Get dimensions
        state_dim = self.topology.X_processed.shape[1]
        n_regimes = self.topology.n_regimes

        print(f"Data processed:")
        print(f"  State dimension: {state_dim}")
        print(f"  Number of regimes: {n_regimes}")
        print(f"  Samples: {len(self.topology.X_processed)}")
        print(f"  Fast variables: {len(self.topology.fast_indices)}")
        print(f"  Slow variables: {len(self.topology.slow_indices)}")

        # Initialize model
        self.model = VelocityNetwork(
            state_dim=state_dim,
            hidden_dim=self.hidden_dim,
            n_regimes=n_regimes,
            n_layers=self.n_layers,
            time_embed_dim=self.time_embed_dim,
            regime_embed_dim=self.regime_embed_dim,
            causal_order=self.topology.causal_order,
            dropout=self.dropout
        ).to(self.device)

        self._is_fitted = True

        return self

    def train(
        self,
        n_epochs: int = 1000,
        lr: float = 1e-4,
        batch_size: int = 256,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        validate_every: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train the flow matching model.

        Uses the simulation-free CFM objective with OT-path interpolation.

        Args:
            n_epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            weight_decay: L2 regularization
            warmup_epochs: Linear warmup epochs
            validate_every: Validation frequency
            verbose: Whether to print progress

        Returns:
            results: Training history and final metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before train()")

        # Create training config
        config = TrainingConfig(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            n_epochs=n_epochs,
            warmup_epochs=warmup_epochs,
            validate_every=validate_every,
            device=self.device
        )

        # Initialize trainer
        self.trainer = FlowMatchingTrainer(
            self.model,
            self.topology,
            config
        )

        # Train
        results = self.trainer.train()

        # Initialize solver and generator
        self.solver = ODESolver(self.model, self.topology)
        self.generator = ScenarioGenerator(self.solver, self.topology)

        self._is_trained = True

        return results

    def sample(
        self,
        n_samples: int,
        regime: int = 0,
        denormalize: bool = True
    ) -> np.ndarray:
        """Generate samples from the learned distribution.

        Args:
            n_samples: Number of samples to generate
            regime: Regime label
            denormalize: Whether to return in original scale

        Returns:
            samples: Generated samples of shape (n_samples, dim)
        """
        if not self._is_trained:
            raise RuntimeError("Must call train() before sample()")

        return self.generator.baseline(n_samples, regime, denormalize)

    def shock(
        self,
        variable: Union[str, int],
        magnitude: float,
        n_samples: int = 1000,
        regime: int = 0,
        guidance_strength: float = 1.0,
        denormalize: bool = True
    ) -> np.ndarray:
        """Generate stress test scenarios with a single shock.

        The shock is applied in normalized (standard deviation) units.
        A magnitude of 3.0 means 3 standard deviations above mean.

        Args:
            variable: Variable name or index to shock
            magnitude: Shock size in standard deviations
            n_samples: Number of scenarios
            regime: Regime label
            guidance_strength: How strongly to apply the shock
            denormalize: Whether to return in original scale

        Returns:
            scenarios: Shocked scenarios
        """
        if not self._is_trained:
            raise RuntimeError("Must call train() before shock()")

        return self.generator.shock(
            variable, magnitude, n_samples, regime,
            guidance_strength, denormalize
        )

    def multi_shock(
        self,
        shocks: Dict[str, float],
        n_samples: int = 1000,
        regime: int = 0,
        guidance_strength: float = 1.0,
        denormalize: bool = True
    ) -> np.ndarray:
        """Generate scenarios with multiple simultaneous shocks.

        Args:
            shocks: Dict mapping variable names to shock magnitudes
            n_samples: Number of scenarios
            regime: Regime label
            guidance_strength: Guidance strength
            denormalize: Whether to return in original scale

        Returns:
            scenarios: Multi-shocked scenarios
        """
        if not self._is_trained:
            raise RuntimeError("Must call train() before multi_shock()")

        return self.generator.multi_shock(
            shocks, n_samples, regime, guidance_strength, denormalize
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the complete model to disk.

        Saves model weights, topology, and configuration.
        The causal ordering is preserved for correct inference.

        Args:
            path: Path to save the checkpoint
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        checkpoint = {
            'version': __version__,
            'model_state': self.model.state_dict(),
            'topology': self.topology.to_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'n_layers': self.n_layers,
                'time_embed_dim': self.time_embed_dim,
                'regime_embed_dim': self.regime_embed_dim,
                'dropout': self.dropout,
            },
            'is_trained': self._is_trained,
        }

        if self.trainer is not None:
            checkpoint['training_state'] = {
                'epoch': self.trainer.state.epoch,
                'best_loss': self.trainer.state.best_loss,
            }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = 'auto') -> "CausalFlowMatcher":
        """Load a saved model from disk.

        Args:
            path: Path to the checkpoint
            device: Device to load to

        Returns:
            cfm: Loaded CausalFlowMatcher instance
        """
        checkpoint = torch.load(path, map_location='cpu')

        config = checkpoint['config']
        cfm = cls(
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            time_embed_dim=config['time_embed_dim'],
            regime_embed_dim=config['regime_embed_dim'],
            dropout=config['dropout'],
            device=device
        )

        # Reconstruct topology
        topo_dict = checkpoint['topology']
        cfm.topology = DataTopology(
            X_processed=np.array([]),  # Dummy, not needed for inference
            regimes=np.array([]),
            regime_embeddings=np.array([]),
            causal_order=np.array(topo_dict['causal_order']),
            variable_names=topo_dict['variable_names'],
            scaler_mean=np.array(topo_dict['scaler_mean']),
            scaler_std=np.array(topo_dict['scaler_std']),
            fast_indices=np.array(topo_dict['fast_indices']),
            slow_indices=np.array(topo_dict['slow_indices']),
            n_regimes=topo_dict['n_regimes'],
            differenced_cols=topo_dict.get('differenced_cols', []),
            pca_components=np.array(topo_dict['pca_components']) if topo_dict.get('pca_components') else None,
            pca_mean=np.array(topo_dict['pca_mean']) if topo_dict.get('pca_mean') else None
        )

        # Initialize model
        state_dim = len(cfm.topology.causal_order)
        cfm.model = VelocityNetwork(
            state_dim=state_dim,
            hidden_dim=config['hidden_dim'],
            n_regimes=cfm.topology.n_regimes,
            n_layers=config['n_layers'],
            time_embed_dim=config['time_embed_dim'],
            regime_embed_dim=config['regime_embed_dim'],
            causal_order=cfm.topology.causal_order,
            dropout=config['dropout']
        ).to(cfm.device)

        # Load weights
        cfm.model.load_state_dict(checkpoint['model_state'])

        # Initialize solver
        cfm.solver = ODESolver(cfm.model, cfm.topology)
        cfm.generator = ScenarioGenerator(cfm.solver, cfm.topology)

        cfm._is_fitted = True
        cfm._is_trained = checkpoint.get('is_trained', True)

        print(f"Model loaded from {path}")

        return cfm

    @property
    def variable_names(self) -> List[str]:
        """Get variable names in causal order."""
        if self.topology is None:
            return []
        return self.topology.variable_names

    @property
    def n_regimes(self) -> int:
        """Get number of detected regimes."""
        if self.topology is None:
            return 0
        return self.topology.n_regimes

    def __repr__(self) -> str:
        status = "not fitted"
        if self._is_fitted and self._is_trained:
            status = "trained"
        elif self._is_fitted:
            status = "fitted (not trained)"

        return f"CausalFlowMatcher(hidden_dim={self.hidden_dim}, status={status})"
