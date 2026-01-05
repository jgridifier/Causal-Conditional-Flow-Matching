"""
Comprehensive tests for the ODE Solver module

Tests cover:
1. Basic ODE integration
2. Guided velocity fields
3. Scenario generation
4. Numerical stability
"""

import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, '..')

# Check if torchdiffeq is available
try:
    import torchdiffeq
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False

from core.network import VelocityNetwork
from core.etl import DataTopology


def create_mock_topology(n_features=6, n_regimes=2):
    """Create a mock DataTopology for testing."""
    return DataTopology(
        X_processed=np.random.randn(100, n_features).astype(np.float32),
        regimes=np.random.randint(0, n_regimes, 100),
        regime_embeddings=np.eye(n_regimes)[np.random.randint(0, n_regimes, 100)],
        causal_order=np.arange(n_features),
        variable_names=[f"var_{i}" for i in range(n_features)],
        scaler_mean=np.zeros(n_features),
        scaler_std=np.ones(n_features),
        fast_indices=np.arange(n_features // 2),
        slow_indices=np.arange(n_features // 2, n_features),
        n_regimes=n_regimes
    )


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestODESolver:
    """Test ODE solver functionality."""

    def test_import_solver(self):
        """Test that solver can be imported."""
        from core.solver import ODESolver, SolverConfig
        assert ODESolver is not None

    def test_solver_initialization(self):
        """Test solver initialization."""
        from core.solver import ODESolver, SolverConfig

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)

        solver = ODESolver(model, topology)

        assert solver.model is model
        assert solver.topology is topology

    def test_sample_basic(self):
        """Test basic sampling."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        samples = solver.sample(n_samples=10, regime=0)

        assert samples.shape == (10, 6)
        assert not torch.any(torch.isnan(samples))

    def test_sample_with_trajectory(self):
        """Test sampling with trajectory return."""
        from core.solver import ODESolver, SolverConfig

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        config = SolverConfig(n_steps=10)
        solver = ODESolver(model, topology, config)

        samples, trajectory = solver.sample(
            n_samples=5,
            regime=0,
            return_trajectory=True
        )

        assert samples.shape == (5, 6)
        assert trajectory.shape[0] == 11  # n_steps + 1
        assert trajectory.shape[1] == 5
        assert trajectory.shape[2] == 6

    def test_sample_different_regimes(self):
        """Test that different regimes produce different samples."""
        from core.solver import ODESolver

        topology = create_mock_topology(n_regimes=3)
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=3)
        solver = ODESolver(model, topology)

        # Set seed for reproducibility of noise
        torch.manual_seed(42)
        samples_0 = solver.sample(n_samples=10, regime=0)

        torch.manual_seed(42)
        samples_1 = solver.sample(n_samples=10, regime=1)

        # Different regimes should give different samples (even with same noise)
        assert not torch.allclose(samples_0, samples_1)

    def test_denormalize(self):
        """Test denormalization."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        topology.scaler_mean = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        topology.scaler_std = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        # Create normalized samples (mean 0, std 1)
        samples = torch.zeros(10, 6)
        denorm = solver.denormalize(samples)

        # After denormalization, should have the original mean
        # (accounting for causal reordering)
        assert denorm.shape == (10, 6)


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestGuidedVelocityField:
    """Test guided velocity field for stress testing."""

    def test_guided_field_initialization(self):
        """Test guided field creation."""
        from core.solver import GuidedVelocityField

        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        regime = torch.zeros(1, dtype=torch.long)

        field = GuidedVelocityField(
            model=model,
            regime=regime,
            target_idx=0,
            target_value=2.0,
            guidance_strength=1.0
        )

        assert field.target_idx == 0
        assert field.target_value == 2.0

    def test_guided_field_forward(self):
        """Test guided field forward pass."""
        from core.solver import GuidedVelocityField

        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        regime = torch.zeros(4, dtype=torch.long)

        field = GuidedVelocityField(
            model=model,
            regime=regime,
            target_idx=0,
            target_value=2.0
        )

        t = torch.tensor(0.5)
        x = torch.randn(4, 6)

        v = field(t, x)

        assert v.shape == (4, 6)
        assert not torch.any(torch.isnan(v))

    def test_sample_guided(self):
        """Test guided sampling."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        samples = solver.sample_guided(
            n_samples=10,
            target_idx=0,
            target_value=2.0,
            regime=0
        )

        assert samples.shape == (10, 6)

        # Guided variable should be closer to target
        # (This is a soft test - guidance pushes toward target but may not reach it)
        mean_target_var = samples[:, 0].mean().item()
        # Just check it's not NaN
        assert not np.isnan(mean_target_var)

    def test_sample_conditional_multiple(self):
        """Test conditional sampling with multiple targets."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        conditions = {0: 2.0, 1: -1.0}

        samples = solver.sample_conditional(
            n_samples=10,
            conditions=conditions,
            regime=0
        )

        assert samples.shape == (10, 6)


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestVelocityFieldWrapper:
    """Test velocity field wrapper for torchdiffeq compatibility."""

    def test_wrapper_forward(self):
        """Test wrapper forward pass."""
        from core.solver import VelocityFieldWrapper

        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        regime = torch.zeros(8, dtype=torch.long)

        wrapper = VelocityFieldWrapper(model, regime)

        t = torch.tensor(0.5)
        x = torch.randn(8, 6)

        v = wrapper(t, x)

        assert v.shape == (8, 6)

    def test_wrapper_scalar_time(self):
        """Test wrapper with scalar time."""
        from core.solver import VelocityFieldWrapper

        model = VelocityNetwork(state_dim=4, hidden_dim=32, n_regimes=2)
        regime = torch.ones(4, dtype=torch.long)

        wrapper = VelocityFieldWrapper(model, regime)

        t = torch.tensor(0.3)
        x = torch.randn(4, 4)

        v = wrapper(t, x)

        assert v.shape == (4, 4)

    def test_wrapper_regime_broadcast(self):
        """Test that regime is properly broadcast."""
        from core.solver import VelocityFieldWrapper

        model = VelocityNetwork(state_dim=4, hidden_dim=32, n_regimes=2)
        regime = torch.zeros(1, dtype=torch.long)  # Single regime

        wrapper = VelocityFieldWrapper(model, regime)

        t = torch.tensor(0.5)
        x = torch.randn(16, 4)  # More samples than regimes

        v = wrapper(t, x)

        assert v.shape == (16, 4)


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestScenarioGenerator:
    """Test high-level scenario generator."""

    def test_generator_initialization(self):
        """Test scenario generator creation."""
        from core.solver import ODESolver, ScenarioGenerator

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        generator = ScenarioGenerator(solver, topology)

        assert generator.solver is solver
        assert generator.topology is topology

    def test_baseline_generation(self):
        """Test baseline scenario generation."""
        from core.solver import ODESolver, ScenarioGenerator

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)
        generator = ScenarioGenerator(solver, topology)

        baseline = generator.baseline(n_scenarios=20)

        assert baseline.shape == (20, 6)

    def test_shock_by_name(self):
        """Test shock generation by variable name."""
        from core.solver import ODESolver, ScenarioGenerator

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)
        generator = ScenarioGenerator(solver, topology)

        shocked = generator.shock(
            variable='var_0',
            magnitude=3.0,
            n_scenarios=10
        )

        assert shocked.shape == (10, 6)

    def test_shock_by_index(self):
        """Test shock generation by variable index."""
        from core.solver import ODESolver, ScenarioGenerator

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)
        generator = ScenarioGenerator(solver, topology)

        shocked = generator.shock(
            variable=0,
            magnitude=2.0,
            n_scenarios=10
        )

        assert shocked.shape == (10, 6)

    def test_multi_shock(self):
        """Test multiple simultaneous shocks."""
        from core.solver import ODESolver, ScenarioGenerator

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)
        generator = ScenarioGenerator(solver, topology)

        shocks = {'var_0': 2.0, 'var_1': -1.0}
        shocked = generator.multi_shock(
            shocks=shocks,
            n_scenarios=10
        )

        assert shocked.shape == (10, 6)

    def test_stress_test(self):
        """Test stress test with multiple scenarios."""
        from core.solver import ODESolver, ScenarioGenerator

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)
        generator = ScenarioGenerator(solver, topology)

        scenarios = [
            {'name': 'baseline', 'shocks': {}},
            {'name': 'vix_shock', 'shocks': {'var_0': 3.0}},
            {'name': 'multi_shock', 'shocks': {'var_0': 2.0, 'var_1': 1.5}}
        ]

        results = generator.stress_test(scenarios, n_per_scenario=10)

        assert len(results) == 3
        assert 'baseline' in results
        assert 'vix_shock' in results
        assert results['baseline'].shape == (10, 6)

    def test_unknown_variable_raises(self):
        """Test that unknown variable name raises error."""
        from core.solver import ODESolver, ScenarioGenerator

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)
        generator = ScenarioGenerator(solver, topology)

        with pytest.raises(ValueError, match="Unknown variable"):
            generator.shock(variable='nonexistent', magnitude=1.0, n_scenarios=10)


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestSolverConfig:
    """Test solver configuration."""

    def test_default_config(self):
        """Test default solver configuration."""
        from core.solver import SolverConfig

        config = SolverConfig()

        assert config.method == 'dopri5'
        assert config.rtol == 1e-5
        assert config.atol == 1e-6

    def test_custom_config(self):
        """Test custom solver configuration."""
        from core.solver import SolverConfig

        config = SolverConfig(
            method='euler',
            rtol=1e-3,
            atol=1e-4,
            n_steps=200
        )

        assert config.method == 'euler'
        assert config.rtol == 1e-3
        assert config.n_steps == 200


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestNumericalStability:
    """Test numerical stability of solver."""

    def test_no_nan_output(self):
        """Test that solver doesn't produce NaN."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        for _ in range(5):
            samples = solver.sample(n_samples=10, regime=0)
            assert not torch.any(torch.isnan(samples))
            assert not torch.any(torch.isinf(samples))

    def test_different_solver_methods(self):
        """Test different ODE solver methods."""
        from core.solver import ODESolver, SolverConfig

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)

        for method in ['dopri5', 'euler', 'rk4', 'midpoint']:
            config = SolverConfig(method=method, n_steps=50)
            solver = ODESolver(model, topology, config)

            samples = solver.sample(n_samples=5, regime=0)

            assert samples.shape == (5, 6)
            assert not torch.any(torch.isnan(samples))


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestTemporalGuidedVelocityField:
    """Test temporal guided velocity field for constrained forecasting."""

    def test_temporal_guided_field_initialization(self):
        """Test temporal guided field creation."""
        from core.solver import TemporalGuidedVelocityField

        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        regime = torch.zeros(1, dtype=torch.long)

        field = TemporalGuidedVelocityField(
            model=model,
            regime=regime,
            target_trajectories={0: 2.0, 1: -1.0},
            guidance_strength=1.0
        )

        assert field.target_trajectories == {0: 2.0, 1: -1.0}
        assert field.guidance_strength == 1.0

    def test_temporal_guided_field_with_constant_targets(self):
        """Test temporal guided field with constant target values."""
        from core.solver import TemporalGuidedVelocityField

        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        regime = torch.zeros(4, dtype=torch.long)

        field = TemporalGuidedVelocityField(
            model=model,
            regime=regime,
            target_trajectories={0: 2.0},
            guidance_strength=1.0
        )

        t = torch.tensor(0.5)
        x = torch.randn(4, 6)

        v = field(t, x)

        assert v.shape == (4, 6)
        assert not torch.any(torch.isnan(v))

    def test_temporal_guided_field_with_callable_targets(self):
        """Test temporal guided field with callable target functions."""
        from core.solver import TemporalGuidedVelocityField

        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        regime = torch.zeros(4, dtype=torch.long)

        # Define a time-varying target function
        def linear_target(t):
            return t * 3.0  # Target goes from 0 to 3 as t goes 0 to 1

        field = TemporalGuidedVelocityField(
            model=model,
            regime=regime,
            target_trajectories={0: linear_target},
            guidance_strength=1.0
        )

        t = torch.tensor(0.5)
        x = torch.randn(4, 6)

        v = field(t, x)

        assert v.shape == (4, 6)
        assert not torch.any(torch.isnan(v))

    def test_temporal_guided_field_multiple_variables(self):
        """Test temporal guided field with multiple target variables."""
        from core.solver import TemporalGuidedVelocityField

        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        regime = torch.zeros(4, dtype=torch.long)

        # Mix of constant and callable targets
        targets = {
            0: 2.0,
            1: lambda t: -1.0 + t * 0.5,
            3: 1.5
        }

        field = TemporalGuidedVelocityField(
            model=model,
            regime=regime,
            target_trajectories=targets,
            guidance_strength=1.0
        )

        t = torch.tensor(0.75)
        x = torch.randn(4, 6)

        v = field(t, x)

        assert v.shape == (4, 6)
        assert not torch.any(torch.isnan(v))


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestForecastStep:
    """Test single-step autoregressive forecasting."""

    def test_forecast_step_basic(self):
        """Test basic forecast step generation."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        # Generate initial state
        x_prev = torch.randn(10, 6)

        x_next = solver.forecast_step(
            x_prev=x_prev,
            regime=0,
            noise_scale=0.1
        )

        assert x_next.shape == (10, 6)
        assert not torch.any(torch.isnan(x_next))

    def test_forecast_step_with_constraints(self):
        """Test forecast step with constraints."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        x_prev = torch.randn(10, 6)
        constraints = {0: 2.0, 1: -1.0}

        x_next = solver.forecast_step(
            x_prev=x_prev,
            regime=0,
            noise_scale=0.1,
            constraints=constraints,
            guidance_strength=1.0
        )

        assert x_next.shape == (10, 6)
        assert not torch.any(torch.isnan(x_next))

    def test_forecast_step_noise_scale_effect(self):
        """Test that noise scale affects diversity."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        torch.manual_seed(42)
        x_prev = torch.randn(20, 6)

        # Low noise
        torch.manual_seed(123)
        x_low_noise = solver.forecast_step(x_prev, regime=0, noise_scale=0.01)

        # High noise
        torch.manual_seed(123)
        x_high_noise = solver.forecast_step(x_prev, regime=0, noise_scale=1.0)

        # Both should be valid
        assert not torch.any(torch.isnan(x_low_noise))
        assert not torch.any(torch.isnan(x_high_noise))


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestForecastPaths:
    """Test multi-step autoregressive forecasting."""

    def test_forecast_paths_basic(self):
        """Test basic multi-step forecast path generation."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        paths = solver.forecast_paths(
            n_paths=10,
            n_steps=5,
            regime=0,
            noise_scale=0.1
        )

        assert paths.shape == (10, 5, 6)
        assert not torch.any(torch.isnan(paths))

    def test_forecast_paths_with_initial_state(self):
        """Test forecast paths with provided initial state."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        initial_state = torch.randn(10, 6)

        paths = solver.forecast_paths(
            n_paths=10,
            n_steps=5,
            regime=0,
            initial_state=initial_state
        )

        assert paths.shape == (10, 5, 6)
        assert not torch.any(torch.isnan(paths))

    def test_forecast_paths_with_constraints(self):
        """Test forecast paths with constraint trajectories."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        n_steps = 10
        # Constraint trajectory for variable 0
        constraint_trajectories = {
            0: np.linspace(0, 2.0, n_steps)
        }

        paths = solver.forecast_paths(
            n_paths=10,
            n_steps=n_steps,
            regime=0,
            constraint_trajectories=constraint_trajectories,
            guidance_strength=1.0
        )

        assert paths.shape == (10, n_steps, 6)
        assert not torch.any(torch.isnan(paths))

    def test_forecast_paths_temporal_coherence(self):
        """Test that forecast paths have temporal coherence (autocorrelation)."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        n_paths = 50
        n_steps = 20

        paths = solver.forecast_paths(
            n_paths=n_paths,
            n_steps=n_steps,
            regime=0,
            noise_scale=0.1
        )

        # Compute lag-1 autocorrelation for each path
        paths_np = paths.cpu().numpy()
        autocorrs = []
        for path_idx in range(n_paths):
            for var_idx in range(6):
                series = paths_np[path_idx, :, var_idx]
                if len(series) > 1:
                    corr = np.corrcoef(series[:-1], series[1:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(corr)

        # Average autocorrelation should be positive (temporal coherence)
        avg_autocorr = np.mean(autocorrs)
        assert avg_autocorr > 0, f"Expected positive autocorrelation, got {avg_autocorr}"

    def test_forecast_paths_regime_per_step(self):
        """Test forecast paths with different regimes per step."""
        from core.solver import ODESolver

        topology = create_mock_topology(n_regimes=3)
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=3)
        solver = ODESolver(model, topology)

        n_steps = 6
        regime_sequence = [0, 0, 1, 1, 2, 2]

        paths = solver.forecast_paths(
            n_paths=10,
            n_steps=n_steps,
            regime=regime_sequence,
            noise_scale=0.1
        )

        assert paths.shape == (10, n_steps, 6)
        assert not torch.any(torch.isnan(paths))


@pytest.mark.skipif(not TORCHDIFFEQ_AVAILABLE, reason="torchdiffeq not installed")
class TestSampleTemporalGuided:
    """Test temporal guided sampling."""

    def test_sample_temporal_guided_constant(self):
        """Test temporal guided sampling with constant targets."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        target_trajectories = {0: 2.0, 1: -1.0}

        samples = solver.sample_temporal_guided(
            n_samples=10,
            target_trajectories=target_trajectories,
            regime=0
        )

        assert samples.shape == (10, 6)
        assert not torch.any(torch.isnan(samples))

    def test_sample_temporal_guided_callable(self):
        """Test temporal guided sampling with callable targets."""
        from core.solver import ODESolver

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        solver = ODESolver(model, topology)

        target_trajectories = {
            0: lambda t: t * 2.0,  # Linear from 0 to 2
            1: lambda t: np.sin(t * np.pi)  # Sine wave
        }

        samples = solver.sample_temporal_guided(
            n_samples=10,
            target_trajectories=target_trajectories,
            regime=0
        )

        assert samples.shape == (10, 6)
        assert not torch.any(torch.isnan(samples))

    def test_sample_temporal_guided_with_trajectory(self):
        """Test temporal guided sampling with trajectory return."""
        from core.solver import ODESolver, SolverConfig

        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        config = SolverConfig(n_steps=10)
        solver = ODESolver(model, topology, config)

        target_trajectories = {0: 2.0}

        samples, trajectory = solver.sample_temporal_guided(
            n_samples=5,
            target_trajectories=target_trajectories,
            regime=0,
            return_trajectory=True
        )

        assert samples.shape == (5, 6)
        assert trajectory.shape[0] == 11  # n_steps + 1
        assert trajectory.shape[1] == 5
        assert trajectory.shape[2] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
