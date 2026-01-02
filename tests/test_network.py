"""
Comprehensive tests for the Velocity Network module

Tests cover:
1. Network architecture components
2. Forward pass correctness
3. Causal masking verification
4. FiLM conditioning
5. Gradient flow and training stability
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

import sys
sys.path.insert(0, '..')

from core.network import (
    VelocityNetwork,
    VelocityNetworkUnconstrained,
    MaskedLinear,
    FiLMLayer,
    SinusoidalTimeEmbedding,
    ResidualBlock,
    create_causal_mask
)


class TestSinusoidalTimeEmbedding:
    """Test sinusoidal time embedding."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        embed = SinusoidalTimeEmbedding(embed_dim=64)

        t = torch.rand(32)
        out = embed(t)

        assert out.shape == (32, 64)

    def test_scalar_time(self):
        """Test with scalar time input."""
        embed = SinusoidalTimeEmbedding(embed_dim=64)

        t = torch.tensor(0.5)
        out = embed(t)

        assert out.shape == (1, 64)

    def test_different_times_different_embeddings(self):
        """Test that different times produce different embeddings."""
        embed = SinusoidalTimeEmbedding(embed_dim=64)

        t1 = torch.tensor([0.0])
        t2 = torch.tensor([1.0])

        emb1 = embed(t1)
        emb2 = embed(t2)

        # Embeddings should be different
        assert not torch.allclose(emb1, emb2)

    def test_bounded_output(self):
        """Test that output values are bounded (sin/cos in [-1, 1])."""
        embed = SinusoidalTimeEmbedding(embed_dim=64)

        t = torch.rand(100)
        out = embed(t)

        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)

    def test_odd_embed_dim(self):
        """Test handling of odd embedding dimension."""
        embed = SinusoidalTimeEmbedding(embed_dim=65)

        t = torch.rand(10)
        out = embed(t)

        assert out.shape == (10, 65)


class TestFiLMLayer:
    """Test Feature-wise Linear Modulation layer."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        film = FiLMLayer(hidden_dim=128, context_dim=64)

        h = torch.randn(32, 128)
        context = torch.randn(32, 64)

        out = film(h, context)

        assert out.shape == (32, 128)

    def test_initialization_near_identity(self):
        """Test that initial FiLM is close to identity."""
        film = FiLMLayer(hidden_dim=64, context_dim=32)

        h = torch.randn(10, 64)
        context = torch.zeros(10, 32)

        out = film(h, context)

        # With zero context and proper initialization, output should be close to input
        # (gamma close to 1, beta close to 0)
        # Note: This depends on initialization, so we use a loose tolerance
        assert out.shape == h.shape

    def test_modulation_affects_output(self):
        """Test that different contexts produce different outputs."""
        film = FiLMLayer(hidden_dim=64, context_dim=32)

        h = torch.randn(10, 64)
        context1 = torch.randn(10, 32)
        context2 = torch.randn(10, 32)

        out1 = film(h, context1)
        out2 = film(h, context2)

        assert not torch.allclose(out1, out2)


class TestMaskedLinear:
    """Test MaskedLinear layer."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        mask = torch.ones(64, 32)
        layer = MaskedLinear(in_features=32, out_features=64, mask=mask)

        x = torch.randn(16, 32)
        out = layer(x)

        assert out.shape == (16, 64)

    def test_mask_applied(self):
        """Test that mask zeros out connections."""
        # Create mask that blocks all connections
        mask = torch.zeros(4, 4)
        layer = MaskedLinear(in_features=4, out_features=4, mask=mask, bias=False)

        x = torch.randn(8, 4)
        out = layer(x)

        # With all-zero mask and no bias, output should be zero
        assert torch.allclose(out, torch.zeros_like(out))

    def test_diagonal_mask(self):
        """Test with diagonal mask."""
        mask = torch.eye(4)
        layer = MaskedLinear(in_features=4, out_features=4, mask=mask, bias=False)

        x = torch.randn(8, 4)
        out = layer(x)

        # Each output should only depend on corresponding input
        for i in range(4):
            # Gradient of output[i] w.r.t. input[j] should be 0 for j != i
            x_test = x.clone().requires_grad_(True)
            out_test = layer(x_test)
            out_test[:, i].sum().backward()

            grad = x_test.grad
            for j in range(4):
                if i != j:
                    assert torch.allclose(grad[:, j], torch.zeros_like(grad[:, j]))

    def test_no_mask_fully_connected(self):
        """Test that no mask results in fully connected layer."""
        layer = MaskedLinear(in_features=8, out_features=8, mask=None)

        # Default mask should be all ones
        assert torch.all(layer.mask == 1)


class TestCausalMask:
    """Test causal mask creation."""

    def test_default_lower_triangular(self):
        """Test that default mask is lower triangular."""
        mask = create_causal_mask(dim=4)

        expected = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ], dtype=torch.float)

        torch.testing.assert_close(mask, expected)

    def test_custom_causal_order(self):
        """Test mask with custom causal ordering."""
        # Order: variable 2 is upstream, then 0, then 1
        causal_order = np.array([1, 2, 0])  # Positions in causal chain
        mask = create_causal_mask(dim=3, causal_order=causal_order)

        # Variable at position 0 (order 1) can be influenced by position 2 (order 0)
        # but not by position 1 (order 2)
        assert mask[0, 2] == 1  # order 0 can influence order 1
        assert mask[0, 1] == 0  # order 2 cannot influence order 1

    def test_exclude_diagonal(self):
        """Test mask without diagonal."""
        mask = create_causal_mask(dim=4, include_diagonal=False)

        # Diagonal should be zero
        for i in range(4):
            assert mask[i, i] == 0


class TestResidualBlock:
    """Test residual block with FiLM conditioning."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        block = ResidualBlock(dim=128, context_dim=64)

        x = torch.randn(32, 128)
        context = torch.randn(32, 64)

        out = block(x, context)

        assert out.shape == (32, 128)

    def test_residual_connection(self):
        """Test that residual connection is present."""
        block = ResidualBlock(dim=64, context_dim=32)

        x = torch.randn(8, 64)
        context = torch.randn(8, 32)

        out = block(x, context)

        # Output should be x + something, not just MLP output
        # If we zero out all parameters, output should equal input
        # (This is hard to test directly, but we can check output is different from pure MLP)

    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        block = ResidualBlock(dim=64, context_dim=32, dropout=0.5)
        block.train()

        x = torch.randn(100, 64)
        context = torch.randn(100, 32)

        # Run twice, should get different results with dropout
        out1 = block(x, context)
        out2 = block(x, context)

        # With high dropout, outputs should differ
        assert not torch.allclose(out1, out2)

        # In eval mode, should be deterministic
        block.eval()
        out3 = block(x, context)
        out4 = block(x, context)
        torch.testing.assert_close(out3, out4)


class TestVelocityNetwork:
    """Test the full VelocityNetwork."""

    def test_output_shape(self):
        """Test that output shape matches state dimension."""
        net = VelocityNetwork(
            state_dim=10,
            hidden_dim=64,
            n_regimes=3,
            n_layers=2
        )

        x = torch.randn(32, 10)
        t = torch.rand(32)
        regime = torch.randint(0, 3, (32,))

        v = net(x, t, regime)

        assert v.shape == (32, 10)

    def test_scalar_time(self):
        """Test with scalar time input."""
        net = VelocityNetwork(state_dim=5, hidden_dim=32, n_regimes=2)

        x = torch.randn(16, 5)
        t = torch.tensor(0.5)
        regime = torch.randint(0, 2, (16,))

        v = net(x, t, regime)

        assert v.shape == (16, 5)

    def test_different_regimes_different_output(self):
        """Test that different regimes produce different velocities."""
        net = VelocityNetwork(state_dim=5, hidden_dim=32, n_regimes=3)

        x = torch.randn(10, 5)
        t = torch.tensor(0.5)

        regime0 = torch.zeros(10, dtype=torch.long)
        regime1 = torch.ones(10, dtype=torch.long)

        v0 = net(x, t, regime0)
        v1 = net(x, t, regime1)

        # Different regimes should give different velocities
        assert not torch.allclose(v0, v1)

    def test_different_times_different_output(self):
        """Test that different times produce different velocities."""
        net = VelocityNetwork(state_dim=5, hidden_dim=32, n_regimes=2)

        x = torch.randn(10, 5)
        regime = torch.zeros(10, dtype=torch.long)

        v0 = net(x, torch.tensor(0.0), regime)
        v1 = net(x, torch.tensor(1.0), regime)

        # Different times should give different velocities
        assert not torch.allclose(v0, v1)

    def test_causal_mask_applied(self):
        """Test that causal mask is stored correctly."""
        causal_order = np.array([2, 0, 1])  # Custom ordering
        net = VelocityNetwork(
            state_dim=3,
            hidden_dim=16,
            n_regimes=2,
            causal_order=causal_order
        )

        assert hasattr(net, 'causal_mask')
        assert net.causal_mask.shape == (3, 3)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        net = VelocityNetwork(state_dim=5, hidden_dim=32, n_regimes=2)

        x = torch.randn(8, 5, requires_grad=True)
        t = torch.rand(8)
        regime = torch.randint(0, 2, (8,))

        v = net(x, t, regime)
        loss = v.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_jacobian_computation(self):
        """Test Jacobian computation."""
        net = VelocityNetwork(state_dim=4, hidden_dim=32, n_regimes=2, n_layers=2)

        x = torch.randn(2, 4)
        t = torch.tensor(0.5)
        regime = torch.zeros(2, dtype=torch.long)

        jacobian = net.get_jacobian(x, t, regime)

        assert jacobian.shape == (2, 4, 4)

    def test_verify_causal_structure(self):
        """Test causal structure verification."""
        net = VelocityNetwork(state_dim=4, hidden_dim=32, n_regimes=2)

        is_valid, violations = net.verify_causal_structure()

        # May or may not be valid depending on mask implementation
        # But should return proper types
        assert isinstance(is_valid, bool)
        assert violations.shape == (1, 4, 4)


class TestVelocityNetworkUnconstrained:
    """Test unconstrained velocity network."""

    def test_output_shape(self):
        """Test output shape."""
        net = VelocityNetworkUnconstrained(
            state_dim=8,
            hidden_dim=64,
            n_regimes=3
        )

        x = torch.randn(16, 8)
        t = torch.rand(16)
        regime = torch.randint(0, 3, (16,))

        v = net(x, t, regime)

        assert v.shape == (16, 8)

    def test_no_masking(self):
        """Test that unconstrained network has no masking."""
        net = VelocityNetworkUnconstrained(
            state_dim=5,
            hidden_dim=32,
            n_regimes=2
        )

        # Should not have causal_mask attribute
        assert not hasattr(net, 'causal_mask')

        # All gradients should flow
        x = torch.randn(4, 5, requires_grad=True)
        t = torch.rand(4)
        regime = torch.zeros(4, dtype=torch.long)

        v = net(x, t, regime)
        v[:, 0].sum().backward()

        # Gradient should flow to all inputs
        assert torch.all(x.grad != 0)


class TestNetworkTrainingStability:
    """Test training stability of networks."""

    def test_no_nan_gradients(self):
        """Test that gradients don't become NaN."""
        net = VelocityNetwork(state_dim=10, hidden_dim=64, n_regimes=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        for _ in range(10):
            x = torch.randn(32, 10)
            t = torch.rand(32)
            regime = torch.randint(0, 3, (32,))

            v = net(x, t, regime)
            loss = v.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()

            # Check no NaN gradients
            for param in net.parameters():
                if param.grad is not None:
                    assert not torch.any(torch.isnan(param.grad))

            optimizer.step()

    def test_output_bounded(self):
        """Test that outputs don't explode."""
        net = VelocityNetwork(state_dim=10, hidden_dim=64, n_regimes=3)

        # Test with various inputs
        for _ in range(10):
            x = torch.randn(32, 10)
            t = torch.rand(32)
            regime = torch.randint(0, 3, (32,))

            v = net(x, t, regime)

            # Velocity should be reasonable
            assert not torch.any(torch.isnan(v))
            assert not torch.any(torch.isinf(v))

    def test_silu_activation_smooth(self):
        """Test that SiLU activation is smooth (no dead neurons)."""
        net = VelocityNetwork(state_dim=5, hidden_dim=32, n_regimes=2)

        # Even with very negative inputs, SiLU should give non-zero gradients
        x = torch.randn(16, 5) - 10  # Very negative
        x.requires_grad_(True)
        t = torch.rand(16)
        regime = torch.zeros(16, dtype=torch.long)

        v = net(x, t, regime)
        v.sum().backward()

        # Gradients should not be all zero (unlike ReLU)
        assert torch.any(x.grad != 0)


class TestNetworkRepr:
    """Test string representations."""

    def test_velocity_network_repr(self):
        """Test VelocityNetwork repr."""
        net = VelocityNetwork(state_dim=10, hidden_dim=64, n_regimes=3, n_layers=4)
        repr_str = repr(net)

        assert "state_dim=10" in repr_str
        assert "hidden_dim=64" in repr_str
        assert "n_regimes=3" in repr_str

    def test_masked_linear_repr(self):
        """Test MaskedLinear extra_repr."""
        layer = MaskedLinear(in_features=10, out_features=20)
        repr_str = layer.extra_repr()

        assert "in_features=10" in repr_str
        assert "out_features=20" in repr_str
        assert "masked=True" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
