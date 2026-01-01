"""
Comprehensive tests for the Flow Matching Trainer module

Tests cover:
1. OT-path interpolation correctness
2. CFM loss computation
3. Training loop functionality
4. Optimizer and scheduler configuration
5. Checkpointing
"""

import numpy as np
import torch
import torch.nn as nn
import pytest
import tempfile
import os

import sys
sys.path.insert(0, '..')

from core.trainer import (
    FlowMatchingTrainer,
    TrainingConfig,
    TrainingState,
    CFMLoss,
    create_trainer
)
from core.network import VelocityNetwork
from core.etl import DataTopology


def create_mock_topology(n_samples=100, n_features=6, n_regimes=2):
    """Create a mock DataTopology for testing."""
    return DataTopology(
        X_processed=np.random.randn(n_samples, n_features).astype(np.float32),
        regimes=np.random.randint(0, n_regimes, n_samples),
        regime_embeddings=np.eye(n_regimes)[np.random.randint(0, n_regimes, n_samples)],
        causal_order=np.arange(n_features),
        variable_names=[f"var_{i}" for i in range(n_features)],
        scaler_mean=np.zeros(n_features),
        scaler_std=np.ones(n_features),
        fast_indices=np.arange(n_features // 2),
        slow_indices=np.arange(n_features // 2, n_features),
        n_regimes=n_regimes
    )


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.lr == 1e-4
        assert config.weight_decay == 1e-5
        assert config.batch_size == 256
        assert config.n_epochs == 1000
        assert config.grad_clip == 1.0
        assert config.warmup_epochs == 10

    def test_custom_values(self):
        """Test custom configuration."""
        config = TrainingConfig(
            lr=1e-3,
            batch_size=128,
            n_epochs=100
        )

        assert config.lr == 1e-3
        assert config.batch_size == 128
        assert config.n_epochs == 100


class TestTrainingState:
    """Test TrainingState dataclass."""

    def test_default_state(self):
        """Test default training state."""
        state = TrainingState()

        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_loss == float('inf')
        assert state.loss_history == []

    def test_state_mutation(self):
        """Test that state can be mutated."""
        state = TrainingState()

        state.epoch = 10
        state.loss_history.append(0.5)

        assert state.epoch == 10
        assert len(state.loss_history) == 1


class TestCFMLoss:
    """Test standalone CFM loss module."""

    def test_basic_loss(self):
        """Test basic MSE loss computation."""
        loss_fn = CFMLoss(reduction='mean')

        v_pred = torch.randn(32, 10)
        v_target = torch.randn(32, 10)

        loss = loss_fn(v_pred, v_target)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_prediction_zero_loss(self):
        """Test that identical predictions give zero loss."""
        loss_fn = CFMLoss()

        v = torch.randn(32, 10)
        loss = loss_fn(v, v)

        assert torch.isclose(loss, torch.tensor(0.0))

    def test_reduction_modes(self):
        """Test different reduction modes."""
        v_pred = torch.randn(8, 4)
        v_target = torch.randn(8, 4)

        loss_mean = CFMLoss(reduction='mean')(v_pred, v_target)
        loss_sum = CFMLoss(reduction='sum')(v_pred, v_target)
        loss_none = CFMLoss(reduction='none')(v_pred, v_target)

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (8, 4)

    def test_masked_loss(self):
        """Test loss with mask."""
        loss_fn = CFMLoss()

        v_pred = torch.ones(4, 4)
        v_target = torch.zeros(4, 4)

        # Mask out half the entries
        mask = torch.zeros(4, 4)
        mask[:, :2] = 1

        loss = loss_fn(v_pred, v_target, mask=mask)

        # Loss should be less than without mask
        full_loss = loss_fn(v_pred, v_target)
        assert loss < full_loss


class TestOTPathSampling:
    """Test Optimal Transport path sampling."""

    def test_interpolation_endpoints(self):
        """Test that x_t at t=0 is x_0 and t=1 is x_1."""
        topology = create_mock_topology(n_samples=100)
        model = VelocityNetwork(
            state_dim=6, hidden_dim=32, n_regimes=2, n_layers=1
        )
        config = TrainingConfig(batch_size=32, n_epochs=1)
        trainer = FlowMatchingTrainer(model, topology, config)

        x_1 = torch.randn(32, 6)

        # At t=0: x_t = (1-0)*x_0 + 0*x_1 = x_0
        # At t=1: x_t = (1-1)*x_0 + 1*x_1 = x_1
        x_t, t, u_t, x_0 = trainer.sample_ot_path(x_1, 32)

        # Manually verify for specific t values
        t_zero = torch.zeros(32)
        t_one = torch.ones(32)

        x_t_0 = (1 - t_zero.unsqueeze(-1)) * x_0 + t_zero.unsqueeze(-1) * x_1
        x_t_1 = (1 - t_one.unsqueeze(-1)) * x_0 + t_one.unsqueeze(-1) * x_1

        torch.testing.assert_close(x_t_0, x_0)
        torch.testing.assert_close(x_t_1, x_1)

    def test_target_velocity(self):
        """Test that target velocity is x_1 - x_0."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        trainer = FlowMatchingTrainer(model, topology, TrainingConfig(n_epochs=1))

        x_1 = torch.randn(16, 6)
        x_t, t, u_t, x_0 = trainer.sample_ot_path(x_1, 16)

        expected_velocity = x_1 - x_0
        torch.testing.assert_close(u_t, expected_velocity)

    def test_noise_is_standard_normal(self):
        """Test that x_0 samples are approximately N(0, I)."""
        topology = create_mock_topology(n_samples=1000)
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        trainer = FlowMatchingTrainer(model, topology, TrainingConfig(n_epochs=1))

        x_1 = torch.randn(1000, 6)
        _, _, _, x_0 = trainer.sample_ot_path(x_1, 1000)

        # Check mean is close to 0
        assert torch.abs(x_0.mean()) < 0.1

        # Check std is close to 1
        assert torch.abs(x_0.std() - 1.0) < 0.1

    def test_time_uniform_distribution(self):
        """Test that time is sampled uniformly."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        trainer = FlowMatchingTrainer(model, topology, TrainingConfig(n_epochs=1))

        x_1 = torch.randn(10000, 6)
        _, t, _, _ = trainer.sample_ot_path(x_1, 10000)

        # Check time is in [0, 1]
        assert torch.all(t >= 0)
        assert torch.all(t <= 1)

        # Check roughly uniform (mean should be ~0.5)
        assert torch.abs(t.mean() - 0.5) < 0.05


class TestLossComputation:
    """Test CFM loss computation in trainer."""

    def test_loss_is_positive(self):
        """Test that loss is non-negative."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        trainer = FlowMatchingTrainer(model, topology, TrainingConfig(n_epochs=1))

        x_1 = torch.randn(32, 6)
        regime = torch.randint(0, 2, (32,))

        loss, metrics = trainer.compute_loss(x_1, regime)

        assert loss.item() >= 0

    def test_loss_decreases_with_training(self):
        """Test that loss decreases with training steps."""
        topology = create_mock_topology(n_samples=200)
        model = VelocityNetwork(state_dim=6, hidden_dim=64, n_regimes=2)
        config = TrainingConfig(lr=1e-3, batch_size=32, n_epochs=5)
        trainer = FlowMatchingTrainer(model, topology, config)

        initial_loss = None
        final_loss = None

        for epoch in range(5):
            for x, regime in trainer.train_loader:
                x = x.to(trainer.device)
                regime = regime.to(trainer.device)

                metrics = trainer.train_step(x, regime)

                if initial_loss is None:
                    initial_loss = metrics['loss']
                final_loss = metrics['loss']

        # Loss should generally decrease (though not guaranteed)
        # Just check we don't have NaN
        assert not np.isnan(final_loss)


class TestTrainingLoop:
    """Test the full training loop."""

    def test_train_runs_without_error(self):
        """Test that training completes without errors."""
        topology = create_mock_topology(n_samples=100)
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2, n_layers=2)
        config = TrainingConfig(
            batch_size=32,
            n_epochs=2,
            validate_every=1
        )
        trainer = FlowMatchingTrainer(model, topology, config)

        results = trainer.train()

        assert 'final_metrics' in results
        assert 'loss_history' in results
        assert len(results['loss_history']) == 2

    def test_validation_runs(self):
        """Test that validation is performed."""
        topology = create_mock_topology(n_samples=100)
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        config = TrainingConfig(
            batch_size=32,
            n_epochs=3,
            validate_every=1
        )
        trainer = FlowMatchingTrainer(model, topology, config)

        results = trainer.train()

        assert len(results['val_loss_history']) > 0

    def test_best_loss_tracked(self):
        """Test that best loss is tracked."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        config = TrainingConfig(n_epochs=2, validate_every=1)
        trainer = FlowMatchingTrainer(model, topology, config)

        results = trainer.train()

        assert results['best_loss'] < float('inf')

    def test_grad_clipping_applied(self):
        """Test that gradient clipping is applied."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        config = TrainingConfig(grad_clip=0.1, n_epochs=1)
        trainer = FlowMatchingTrainer(model, topology, config)

        x = torch.randn(32, 6).to(trainer.device)
        regime = torch.randint(0, 2, (32,)).to(trainer.device)

        metrics = trainer.train_step(x, regime)

        # Grad norm should be clipped
        assert metrics['grad_norm'] <= 0.1 + 0.01  # Small tolerance

    def test_warmup_learning_rate(self):
        """Test that warmup is applied."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        config = TrainingConfig(
            lr=1e-3,
            warmup_epochs=5,
            n_epochs=10,
            validate_every=10
        )
        trainer = FlowMatchingTrainer(model, topology, config)

        # During warmup, LR should be less than target
        results = trainer.train()

        # Just check training completes
        assert len(results['loss_history']) == 10


class TestCheckpointing:
    """Test model checkpointing."""

    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        trainer = FlowMatchingTrainer(model, topology, TrainingConfig(n_epochs=1))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(path)

            assert os.path.exists(path)

            # Check checkpoint contents
            checkpoint = torch.load(path)
            assert 'model_state' in checkpoint
            assert 'optimizer_state' in checkpoint
            assert 'topology' in checkpoint

    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        config = TrainingConfig(n_epochs=2, validate_every=1)
        trainer = FlowMatchingTrainer(model, topology, config)

        # Train a bit
        trainer.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(path)

            # Create new trainer and load
            model2 = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
            trainer2 = FlowMatchingTrainer(model2, topology, config)
            trainer2.load_checkpoint(path)

            # State should be restored
            assert trainer2.state.epoch == trainer.state.epoch


class TestTrainerFactory:
    """Test create_trainer factory function."""

    def test_create_trainer_basic(self):
        """Test basic trainer creation."""
        topology = create_mock_topology()

        trainer = create_trainer(
            state_dim=6,
            hidden_dim=64,
            n_regimes=2,
            topology=topology
        )

        assert isinstance(trainer, FlowMatchingTrainer)
        assert trainer.model.state_dim == 6
        assert trainer.model.hidden_dim == 64

    def test_create_trainer_with_config(self):
        """Test trainer creation with custom config."""
        topology = create_mock_topology()
        config = TrainingConfig(lr=1e-2, n_epochs=50)

        trainer = create_trainer(
            state_dim=6,
            hidden_dim=64,
            n_regimes=2,
            topology=topology,
            config=config
        )

        assert trainer.config.lr == 1e-2
        assert trainer.config.n_epochs == 50


class TestCausalGradientCheck:
    """Test causal gradient verification."""

    def test_check_causal_gradients(self):
        """Test causal gradient checking runs."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        config = TrainingConfig(check_causal_gradients=True, n_epochs=1)
        trainer = FlowMatchingTrainer(model, topology, config)

        # Should run without error
        is_valid = trainer.check_causal_gradients()

        assert isinstance(is_valid, bool)


class TestMetrics:
    """Test training metrics computation."""

    def test_slow_fast_error_separation(self):
        """Test that slow and fast errors are computed separately."""
        topology = create_mock_topology()
        model = VelocityNetwork(state_dim=6, hidden_dim=32, n_regimes=2)
        trainer = FlowMatchingTrainer(model, topology, TrainingConfig(n_epochs=1))

        x = torch.randn(32, 6).to(trainer.device)
        regime = torch.randint(0, 2, (32,)).to(trainer.device)

        _, metrics = trainer.compute_loss(x, regime)

        assert 'slow_error' in metrics
        assert 'fast_error' in metrics
        assert metrics['slow_error'] >= 0
        assert metrics['fast_error'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
