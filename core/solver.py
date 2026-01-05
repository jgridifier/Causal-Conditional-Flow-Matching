"""
ODE Solver for Causal Conditional Flow Matching

Implements the inference engine that generates scenarios by integrating
the learned velocity field using adaptive ODE solvers.

Key Features:
    - Dopri5 (Dormand-Prince 5th order) adaptive step solver
    - Guided integration for stress testing (Vector Field Steering)
    - Compatibility with torchdiffeq for robust integration
    - Scenario generation from noise prior

Vector Field Steering:
    For stress testing, we can guide specific variables to target values
    by adding a correction term to the velocity field:

    v_guided(t, x) = v_nominal(t, x) + λ * (target - x[i]) / (1 - t + ε)

    This steers variable i toward the target while the masked network
    ensures all other variables update consistently with the causal shock.

References:
    - Chen et al. (2018): Neural Ordinary Differential Equations
    - Dormand & Prince (1980): A family of embedded Runge-Kutta formulae
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable, Union, Dict, Any, List
from dataclasses import dataclass
import warnings

try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    warnings.warn("torchdiffeq not available. Install with: pip install torchdiffeq")

from .network import VelocityNetwork
from .etl import DataTopology, validate_for_ode


@dataclass
class SolverConfig:
    """Configuration for the ODE solver.

    Attributes:
        method: Integration method ('dopri5', 'euler', 'rk4', 'midpoint')
        rtol: Relative tolerance for adaptive solvers (default 1e-5)
        atol: Absolute tolerance for adaptive solvers (default 1e-6)
        step_size: Fixed step size for non-adaptive methods (default 0.01)
        n_steps: Number of integration steps if using fixed stepping (default 100)
        use_adjoint: Whether to use adjoint method for memory efficiency
        max_num_steps: Maximum number of ODE solver steps (default 10000)
    """
    method: str = 'dopri5'
    rtol: float = 1e-5
    atol: float = 1e-6
    step_size: float = 0.01
    n_steps: int = 100
    use_adjoint: bool = False
    max_num_steps: int = 10000


class VelocityFieldWrapper(nn.Module):
    """Wrapper to make VelocityNetwork compatible with torchdiffeq.

    torchdiffeq expects: func(t, x) -> dx/dt
    VelocityNetwork expects: forward(x, t, regime) -> v

    This wrapper handles the interface conversion and regime conditioning.

    Args:
        model: The VelocityNetwork to wrap
        regime: Fixed regime label(s) for integration
    """

    def __init__(
        self,
        model: VelocityNetwork,
        regime: torch.Tensor
    ):
        super().__init__()
        self.model = model
        self.regime = regime

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute velocity at state x, time t.

        Args:
            t: Scalar time value
            x: State tensor of shape (batch, dim)

        Returns:
            v: Velocity tensor of shape (batch, dim)
        """
        batch_size = x.shape[0]

        # Ensure t is properly shaped for the model
        if t.dim() == 0:
            t_batch = t.expand(batch_size)
        else:
            t_batch = t

        # Expand regime if needed
        if self.regime.shape[0] == 1 and batch_size > 1:
            regime = self.regime.expand(batch_size)
        else:
            regime = self.regime

        return self.model(x, t_batch, regime)


class TemporalGuidedVelocityField(nn.Module):
    """Velocity field with time-varying guidance for temporal forecasting.

    Supports guiding multiple variables along predefined trajectories
    during integration. This is useful for constrained forecasting where
    certain variables follow known or prescribed paths.

    The correction term for each constrained variable at each time is:
        correction[i] = guidance_strength * (target_trajectory[i](t) - x[i]) / (1 - t + eps)

    Args:
        model: The VelocityNetwork
        regime: Regime conditioning
        target_trajectories: Dict mapping variable index to callable (t -> target_value)
                            or to a fixed value (same target for all t)
        guidance_strength: Strength of guidance (default 1.0)
        eps: Small constant to prevent division by zero (default 1e-4)
    """

    def __init__(
        self,
        model: VelocityNetwork,
        regime: torch.Tensor,
        target_trajectories: Dict[int, Union[float, Callable[[float], float]]],
        guidance_strength: float = 1.0,
        eps: float = 1e-4
    ):
        super().__init__()
        self.model = model
        self.regime = regime
        self.target_trajectories = target_trajectories
        self.guidance_strength = guidance_strength
        self.eps = eps

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute guided velocity field with time-varying targets.

        Args:
            t: Scalar time value
            x: State tensor of shape (batch, dim)

        Returns:
            v_guided: Guided velocity tensor of shape (batch, dim)
        """
        batch_size = x.shape[0]

        if t.dim() == 0:
            t_batch = t.expand(batch_size)
            t_scalar = t.item()
        else:
            t_batch = t
            t_scalar = t[0].item() if t.shape[0] > 0 else 0.0

        regime = self.regime
        if regime.shape[0] == 1 and batch_size > 1:
            regime = regime.expand(batch_size)

        # Compute nominal velocity
        v_guided = self.model(x, t_batch, regime).clone()

        # Apply guidance to each constrained variable
        remaining_time = 1.0 - t_scalar + self.eps
        for idx, target_spec in self.target_trajectories.items():
            # Get target value at current time
            if callable(target_spec):
                target_value = target_spec(t_scalar)
            else:
                target_value = target_spec

            gap = target_value - x[:, idx]
            correction = self.guidance_strength * gap / remaining_time
            v_guided[:, idx] = v_guided[:, idx] + correction

        return v_guided


class GuidedVelocityField(nn.Module):
    """Velocity field with guidance for stress testing.

    Implements vector field steering to guide specific variables
    toward target values during integration.

    The correction term is:
        correction[i] = guidance_strength * (target - x[i]) / (1 - t + eps)

    This creates a velocity that pushes variable i toward the target,
    with increasing urgency as t approaches 1.

    Args:
        model: The VelocityNetwork
        regime: Regime conditioning
        target_idx: Index of variable to guide
        target_value: Target value for the guided variable
        guidance_strength: Strength of guidance (default 1.0)
        eps: Small constant to prevent division by zero (default 1e-4)
    """

    def __init__(
        self,
        model: VelocityNetwork,
        regime: torch.Tensor,
        target_idx: int,
        target_value: float,
        guidance_strength: float = 1.0,
        eps: float = 1e-4
    ):
        super().__init__()
        self.model = model
        self.regime = regime
        self.target_idx = target_idx
        self.target_value = target_value
        self.guidance_strength = guidance_strength
        self.eps = eps

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute guided velocity field.

        Args:
            t: Scalar time value
            x: State tensor of shape (batch, dim)

        Returns:
            v_guided: Guided velocity tensor of shape (batch, dim)
        """
        batch_size = x.shape[0]

        # Ensure proper shapes
        if t.dim() == 0:
            t_batch = t.expand(batch_size)
            t_scalar = t
        else:
            t_batch = t
            t_scalar = t[0] if t.dim() > 0 else t

        if self.regime.shape[0] == 1 and batch_size > 1:
            regime = self.regime.expand(batch_size)
        else:
            regime = self.regime

        # Compute nominal velocity
        v_nominal = self.model(x, t_batch, regime)

        # Compute guidance correction
        # The correction velocity is proportional to the gap
        # and inversely proportional to remaining time
        remaining_time = 1.0 - t_scalar + self.eps
        gap = self.target_value - x[:, self.target_idx]

        # Correction velocity for the target variable
        correction_velocity = self.guidance_strength * gap / remaining_time

        # Apply correction only to the target variable
        v_guided = v_nominal.clone()
        v_guided[:, self.target_idx] = v_guided[:, self.target_idx] + correction_velocity

        return v_guided


class ODESolver:
    """ODE Solver for generating scenarios from the flow model.

    Integrates the learned velocity field from t=0 (noise) to t=1 (data)
    to generate realistic economic scenarios.

    Features:
    - Adaptive step size with Dopri5 (recommended)
    - Guided integration for stress testing
    - Multiple regime support
    - Trajectory visualization

    Args:
        model: Trained VelocityNetwork
        topology: Data topology with normalization stats
        config: Solver configuration

    Example:
        >>> solver = ODESolver(model, topology)
        >>> # Generate unconditioned samples
        >>> samples = solver.sample(n_samples=100, regime=0)
        >>> # Generate stress scenario
        >>> stressed = solver.sample_guided(
        ...     n_samples=100,
        ...     target_idx=0,  # VIX index
        ...     target_value=2.5,  # 2.5 std above mean
        ...     regime=1  # Crisis regime
        ... )
    """

    def __init__(
        self,
        model: VelocityNetwork,
        topology: DataTopology,
        config: Optional[SolverConfig] = None
    ):
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError(
                "torchdiffeq is required for ODE solving. "
                "Install with: pip install torchdiffeq"
            )

        self.model = model
        self.topology = topology
        self.config = config or SolverConfig()

        # Get device from model
        self.device = next(model.parameters()).device

        # Integration endpoints
        self.t_span = torch.tensor([0.0, 1.0], device=self.device)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        regime: Union[int, torch.Tensor],
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples by integrating from noise to data.

        Args:
            n_samples: Number of samples to generate
            regime: Regime label (int) or tensor of labels (batch,)
            return_trajectory: Whether to return full integration trajectory

        Returns:
            samples: Generated samples of shape (n_samples, dim)
            trajectory: (optional) Full trajectory of shape (n_steps, n_samples, dim)
        """
        self.model.eval()
        dim = self.model.state_dim

        # Sample from noise prior
        x_0 = torch.randn(n_samples, dim, device=self.device)

        # Prepare regime tensor
        if isinstance(regime, int):
            regime_tensor = torch.full((n_samples,), regime, dtype=torch.long, device=self.device)
        else:
            regime_tensor = regime.to(self.device)

        # Create velocity field wrapper
        velocity_fn = VelocityFieldWrapper(self.model, regime_tensor)

        # Choose integrator
        integrator = odeint_adjoint if self.config.use_adjoint else odeint

        # Define time points
        if return_trajectory:
            t_eval = torch.linspace(0, 1, self.config.n_steps + 1, device=self.device)
        else:
            t_eval = self.t_span

        # Integrate
        trajectory = integrator(
            velocity_fn,
            x_0,
            t_eval,
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
            options={'max_num_steps': self.config.max_num_steps}
        )

        if return_trajectory:
            return trajectory[-1], trajectory
        else:
            return trajectory[-1]

    @torch.no_grad()
    def sample_guided(
        self,
        n_samples: int,
        target_idx: int,
        target_value: float,
        regime: Union[int, torch.Tensor],
        guidance_strength: float = 1.0,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate stress test scenarios with guided integration.

        Steers a specific variable toward a target value while
        ensuring all other variables respond causally.

        Args:
            n_samples: Number of samples to generate
            target_idx: Index of variable to guide (in causal order)
            target_value: Target value (in normalized space)
            regime: Regime label
            guidance_strength: How strongly to guide (default 1.0)
            return_trajectory: Whether to return full trajectory

        Returns:
            samples: Stressed samples of shape (n_samples, dim)
            trajectory: (optional) Integration trajectory
        """
        self.model.eval()
        dim = self.model.state_dim

        # Sample from noise prior
        x_0 = torch.randn(n_samples, dim, device=self.device)

        # Prepare regime
        if isinstance(regime, int):
            regime_tensor = torch.full((n_samples,), regime, dtype=torch.long, device=self.device)
        else:
            regime_tensor = regime.to(self.device)

        # Create guided velocity field
        velocity_fn = GuidedVelocityField(
            self.model,
            regime_tensor,
            target_idx=target_idx,
            target_value=target_value,
            guidance_strength=guidance_strength
        )

        # Choose integrator
        integrator = odeint_adjoint if self.config.use_adjoint else odeint

        # Time points
        if return_trajectory:
            t_eval = torch.linspace(0, 1, self.config.n_steps + 1, device=self.device)
        else:
            t_eval = self.t_span

        # Integrate
        trajectory = integrator(
            velocity_fn,
            x_0,
            t_eval,
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
            options={'max_num_steps': self.config.max_num_steps}
        )

        if return_trajectory:
            return trajectory[-1], trajectory
        else:
            return trajectory[-1]

    @torch.no_grad()
    def sample_conditional(
        self,
        n_samples: int,
        conditions: Dict[int, float],
        regime: Union[int, torch.Tensor],
        guidance_strength: float = 1.0
    ) -> torch.Tensor:
        """Generate samples conditioned on multiple variable targets.

        Applies guidance to multiple variables simultaneously.

        Args:
            n_samples: Number of samples
            conditions: Dict mapping variable index to target value
            regime: Regime label
            guidance_strength: Guidance strength

        Returns:
            samples: Conditioned samples of shape (n_samples, dim)
        """
        if len(conditions) == 0:
            return self.sample(n_samples, regime)

        # For single condition, use sample_guided
        if len(conditions) == 1:
            idx, val = next(iter(conditions.items()))
            return self.sample_guided(
                n_samples, idx, val, regime,
                guidance_strength=guidance_strength
            )

        # For multiple conditions, create a custom velocity field
        self.model.eval()
        dim = self.model.state_dim

        x_0 = torch.randn(n_samples, dim, device=self.device)

        if isinstance(regime, int):
            regime_tensor = torch.full((n_samples,), regime, dtype=torch.long, device=self.device)
        else:
            regime_tensor = regime.to(self.device)

        # Custom multi-target guided field
        class MultiGuidedField(nn.Module):
            def __init__(self, model, regime, conditions, strength, eps=1e-4):
                super().__init__()
                self.model = model
                self.regime = regime
                self.conditions = conditions
                self.strength = strength
                self.eps = eps

            def forward(self, t, x):
                batch_size = x.shape[0]
                t_batch = t.expand(batch_size) if t.dim() == 0 else t
                t_scalar = t.item() if t.dim() == 0 else t[0].item()

                regime = self.regime
                if regime.shape[0] == 1 and batch_size > 1:
                    regime = regime.expand(batch_size)

                v = self.model(x, t_batch, regime)

                remaining = 1.0 - t_scalar + self.eps
                for idx, target in self.conditions.items():
                    gap = target - x[:, idx]
                    correction = self.strength * gap / remaining
                    v[:, idx] = v[:, idx] + correction

                return v

        velocity_fn = MultiGuidedField(
            self.model, regime_tensor, conditions,
            guidance_strength
        )

        trajectory = odeint(
            velocity_fn,
            x_0,
            self.t_span,
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
            options={'max_num_steps': self.config.max_num_steps}
        )

        return trajectory[-1]

    def denormalize(
        self,
        samples: torch.Tensor
    ) -> np.ndarray:
        """Convert samples from normalized to original scale.

        Args:
            samples: Normalized samples of shape (n_samples, dim)

        Returns:
            denormalized: Samples in original scale
        """
        samples_np = samples.cpu().numpy()

        # Reverse causal ordering
        inverse_order = np.argsort(self.topology.causal_order)
        samples_reordered = samples_np[:, inverse_order]

        # Denormalize
        denormalized = (
            samples_reordered * self.topology.scaler_std +
            self.topology.scaler_mean
        )

        return denormalized

    def get_variable_name(self, idx: int) -> str:
        """Get variable name for an index (in causal order)."""
        if idx < len(self.topology.variable_names):
            return self.topology.variable_names[idx]
        return f"var_{idx}"

    @torch.no_grad()
    def forecast_step(
        self,
        x_prev: torch.Tensor,
        regime: Union[int, torch.Tensor],
        noise_scale: float = 0.1,
        constraints: Optional[Dict[int, float]] = None,
        guidance_strength: float = 1.0
    ) -> torch.Tensor:
        """Generate one forecasting step conditioned on previous state.

        This implements autoregressive forecasting by using the previous
        state as the starting point (with noise) and integrating forward.

        Args:
            x_prev: Previous state tensor of shape (n_samples, dim) in NORMALIZED space
            regime: Regime label
            noise_scale: Scale of noise to add to x_prev before integrating (default 0.1)
            constraints: Optional dict mapping variable index to target value
            guidance_strength: Strength of guidance for constraints

        Returns:
            x_next: Next state tensor of shape (n_samples, dim) in normalized space
        """
        self.model.eval()
        n_samples = x_prev.shape[0]

        # Add noise to previous state for diversity (this creates the "innovation")
        # The noise scale controls how much the forecast can deviate from conditioning
        x_0 = x_prev + noise_scale * torch.randn_like(x_prev)

        # Prepare regime
        if isinstance(regime, int):
            regime_tensor = torch.full((n_samples,), regime, dtype=torch.long, device=self.device)
        else:
            regime_tensor = regime.to(self.device)

        # Choose velocity field based on whether we have constraints
        if constraints and len(constraints) > 0:
            velocity_fn = TemporalGuidedVelocityField(
                self.model,
                regime_tensor,
                target_trajectories=constraints,
                guidance_strength=guidance_strength
            )
        else:
            velocity_fn = VelocityFieldWrapper(self.model, regime_tensor)

        # Integrate
        integrator = odeint_adjoint if self.config.use_adjoint else odeint
        trajectory = integrator(
            velocity_fn,
            x_0,
            self.t_span,
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
            options={'max_num_steps': self.config.max_num_steps}
        )

        return trajectory[-1]

    @torch.no_grad()
    def forecast_paths(
        self,
        n_paths: int,
        n_steps: int,
        regime: Union[int, torch.Tensor],
        noise_scale: float = 0.1,
        constraint_trajectories: Optional[Dict[int, np.ndarray]] = None,
        guidance_strength: float = 1.0,
        initial_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate multi-step autoregressive forecasting paths.

        This is the core temporal forecasting method. It generates paths by:
        1. Starting from either a provided initial state or sampling from noise
        2. At each step, using the previous state as conditioning
        3. Optionally enforcing constraint trajectories on specific variables

        Args:
            n_paths: Number of scenario paths to generate
            n_steps: Number of time steps to forecast
            regime: Regime label (can change per step if tensor provided)
            noise_scale: Innovation noise scale (higher = more diversity between steps)
            constraint_trajectories: Optional dict mapping variable index to
                                    array of target values of shape (n_steps,)
            guidance_strength: Strength of guidance for constraints
            initial_state: Optional initial state (n_paths, dim) in normalized space.
                          If None, samples from the learned distribution.

        Returns:
            paths: Tensor of shape (n_paths, n_steps, dim) in normalized space

        Example:
            >>> # Unconstrained forecast
            >>> paths = solver.forecast_paths(n_paths=100, n_steps=36, regime=0)
            >>>
            >>> # Constrained forecast (VIX follows a rising path)
            >>> vix_trajectory = np.linspace(0, 2.5, 36)  # Rising from 0 to 2.5 std
            >>> paths = solver.forecast_paths(
            ...     n_paths=100, n_steps=36, regime=0,
            ...     constraint_trajectories={vix_idx: vix_trajectory}
            ... )
        """
        self.model.eval()
        dim = self.model.state_dim

        # Initialize paths storage
        paths = torch.empty((n_paths, n_steps, dim), device=self.device)

        # Generate initial state
        if initial_state is not None:
            x_current = initial_state.to(self.device)
        else:
            # Sample initial state from the model's learned distribution
            x_current = self.sample(n_paths, regime if isinstance(regime, int) else regime[0])

        # Prepare regime handling
        if isinstance(regime, int):
            regime_per_step = [regime] * n_steps
        elif isinstance(regime, (list, np.ndarray)):
            regime_per_step = regime
        else:
            regime_per_step = [regime] * n_steps

        # Generate each step autoregressively
        for step in range(n_steps):
            current_regime = regime_per_step[step] if step < len(regime_per_step) else regime_per_step[-1]

            # Get constraints for this step if any
            step_constraints = None
            if constraint_trajectories:
                step_constraints = {}
                for idx, trajectory in constraint_trajectories.items():
                    if step < len(trajectory):
                        step_constraints[idx] = float(trajectory[step])

            # Generate next state
            x_next = self.forecast_step(
                x_current,
                current_regime,
                noise_scale=noise_scale,
                constraints=step_constraints,
                guidance_strength=guidance_strength
            )

            # Store and update
            paths[:, step, :] = x_next
            x_current = x_next

        return paths

    @torch.no_grad()
    def sample_temporal_guided(
        self,
        n_samples: int,
        target_trajectories: Dict[int, Union[float, Callable[[float], float]]],
        regime: Union[int, torch.Tensor],
        guidance_strength: float = 1.0,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples with time-varying guidance on multiple variables.

        This allows guiding variables along continuous trajectories during
        a single ODE integration step. Useful for smooth transitions.

        Args:
            n_samples: Number of samples
            target_trajectories: Dict mapping variable index to either:
                                - float: constant target throughout integration
                                - callable: function of t returning target value
            regime: Regime label
            guidance_strength: Guidance strength
            return_trajectory: Whether to return integration trajectory

        Returns:
            samples: Guided samples of shape (n_samples, dim)
            trajectory: (optional) Full trajectory of shape (n_steps, n_samples, dim)
        """
        self.model.eval()
        dim = self.model.state_dim

        # Sample from noise prior
        x_0 = torch.randn(n_samples, dim, device=self.device)

        # Prepare regime
        if isinstance(regime, int):
            regime_tensor = torch.full((n_samples,), regime, dtype=torch.long, device=self.device)
        else:
            regime_tensor = regime.to(self.device)

        # Create temporal guided velocity field
        velocity_fn = TemporalGuidedVelocityField(
            self.model,
            regime_tensor,
            target_trajectories=target_trajectories,
            guidance_strength=guidance_strength
        )

        # Integrate
        integrator = odeint_adjoint if self.config.use_adjoint else odeint
        if return_trajectory:
            t_eval = torch.linspace(0, 1, self.config.n_steps + 1, device=self.device)
        else:
            t_eval = self.t_span

        trajectory = integrator(
            velocity_fn,
            x_0,
            t_eval,
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
            options={'max_num_steps': self.config.max_num_steps}
        )

        if return_trajectory:
            return trajectory[-1], trajectory
        else:
            return trajectory[-1]


class ScenarioGenerator:
    """High-level interface for generating economic scenarios.

    Provides convenient methods for common scenario generation tasks.

    Args:
        solver: ODESolver instance
        topology: Data topology

    Example:
        >>> generator = ScenarioGenerator(solver, topology)
        >>> # Generate baseline scenarios
        >>> baseline = generator.baseline(n_scenarios=1000)
        >>> # Generate VIX shock scenario
        >>> shocked = generator.shock(
        ...     variable='VIX',
        ...     magnitude=3.0,  # 3 std shock
        ...     n_scenarios=1000
        ... )
    """

    def __init__(
        self,
        solver: ODESolver,
        topology: DataTopology
    ):
        self.solver = solver
        self.topology = topology

        # Build variable name to index mapping
        self.var_to_idx = {
            name: i for i, name in enumerate(topology.variable_names)
        }

    def baseline(
        self,
        n_scenarios: int,
        regime: Optional[int] = None,
        denormalize: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Generate baseline (unshocked) scenarios.

        Args:
            n_scenarios: Number of scenarios
            regime: Regime label (None = sample from prior)
            denormalize: Whether to return in original scale

        Returns:
            scenarios: Generated scenarios
        """
        if regime is None:
            regime = 0

        samples = self.solver.sample(n_scenarios, regime)

        if denormalize:
            return self.solver.denormalize(samples)
        return samples

    def shock(
        self,
        variable: Union[str, int],
        magnitude: float,
        n_scenarios: int,
        regime: Optional[int] = None,
        guidance_strength: float = 1.0,
        denormalize: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Generate shock scenarios for a specific variable.

        Args:
            variable: Variable name or index to shock
            magnitude: Shock size in standard deviations
            n_scenarios: Number of scenarios
            regime: Regime label
            guidance_strength: How strongly to apply shock
            denormalize: Whether to return in original scale

        Returns:
            scenarios: Shocked scenarios
        """
        if isinstance(variable, str):
            if variable not in self.var_to_idx:
                raise ValueError(f"Unknown variable: {variable}")
            idx = self.var_to_idx[variable]
        else:
            idx = variable

        if regime is None:
            regime = 0

        samples = self.solver.sample_guided(
            n_scenarios,
            target_idx=idx,
            target_value=magnitude,  # In normalized space, this is std devs
            regime=regime,
            guidance_strength=guidance_strength
        )

        if denormalize:
            return self.solver.denormalize(samples)
        return samples

    def multi_shock(
        self,
        shocks: Dict[str, float],
        n_scenarios: int,
        regime: Optional[int] = None,
        guidance_strength: float = 1.0,
        denormalize: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Generate scenarios with multiple simultaneous shocks.

        Args:
            shocks: Dict mapping variable names to shock magnitudes
            n_scenarios: Number of scenarios
            regime: Regime label
            guidance_strength: Guidance strength
            denormalize: Whether to denormalize

        Returns:
            scenarios: Multi-shocked scenarios
        """
        conditions = {}
        for var, magnitude in shocks.items():
            if isinstance(var, str):
                if var not in self.var_to_idx:
                    raise ValueError(f"Unknown variable: {var}")
                idx = self.var_to_idx[var]
            else:
                idx = var
            conditions[idx] = magnitude

        if regime is None:
            regime = 0

        samples = self.solver.sample_conditional(
            n_scenarios,
            conditions=conditions,
            regime=regime,
            guidance_strength=guidance_strength
        )

        if denormalize:
            return self.solver.denormalize(samples)
        return samples

    def stress_test(
        self,
        scenarios: List[Dict[str, Any]],
        n_per_scenario: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Run multiple stress test scenarios.

        Args:
            scenarios: List of scenario specifications
                Each scenario is a dict with:
                - 'name': Scenario name
                - 'shocks': Dict of variable -> magnitude
                - 'regime': (optional) Regime label

        Returns:
            results: Dict mapping scenario names to generated samples
        """
        results = {}

        for spec in scenarios:
            name = spec['name']
            shocks = spec.get('shocks', {})
            regime = spec.get('regime', 0)

            if len(shocks) == 0:
                samples = self.baseline(n_per_scenario, regime)
            else:
                samples = self.multi_shock(
                    shocks, n_per_scenario, regime
                )

            results[name] = samples

        return results
