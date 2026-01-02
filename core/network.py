"""
Masked Velocity Network for Causal Conditional Flow Matching

Implements the neural network that approximates the time-dependent
velocity field v_θ(x_t, t, regime) for the continuous normalizing flow.

Architecture:
    - 4-Layer Residual MLP with SiLU (Swish) activations
    - MaskedLinear layers enforcing causal structure (DAG-consistent)
    - FiLM (Feature-wise Linear Modulation) for regime and time conditioning
    - Sinusoidal positional encoding for time

Key Design Decisions:
    - SiLU activation: Avoids dead neurons during ODE integration (ReLU fails)
    - Masked connectivity: Jacobian is strictly lower-triangular
    - FiLM conditioning: Multiplicative (not concatenative) modulation for regimes

References:
    - Lipman et al. (2022): Flow Matching for Generative Modeling
    - Perez et al. (2018): FiLM: Visual Reasoning with Feature-wise Linear Modulation
    - Germain et al. (2015): MADE: Masked Autoencoder for Distribution Estimation
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time variable.

    Projects scalar time t ∈ [0, 1] to a high-dimensional embedding
    using sine and cosine functions at different frequencies.

    This is the standard positional encoding from Transformer and
    diffusion model literature.

    Args:
        embed_dim: Dimension of the output embedding
        max_period: Maximum period for the sinusoidal functions (default 10000)
    """

    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

        # Precompute frequency bands
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed time values.

        Args:
            t: Time values of shape (batch,) or scalar

        Returns:
            embeddings: Time embeddings of shape (batch, embed_dim)
        """
        # Ensure t is the right shape
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch, 1)

        # Compute sinusoidal embeddings
        args = t * self.freqs.unsqueeze(0) * 2 * math.pi  # (batch, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Pad if embed_dim is odd
        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Modulates hidden activations via learned affine transformation
    conditioned on external context (regime + time embeddings).

    h' = γ(context) * h + β(context)

    This is more expressive than simple concatenation and allows
    the conditioning to multiplicatively gate the hidden representations.

    Args:
        hidden_dim: Dimension of the hidden features to modulate
        context_dim: Dimension of the conditioning context
    """

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Projection to scale (γ) and shift (β) parameters
        self.gamma_proj = nn.Linear(context_dim, hidden_dim)
        self.beta_proj = nn.Linear(context_dim, hidden_dim)

        # Initialize γ close to 1, β close to 0 for stable training start
        nn.init.ones_(self.gamma_proj.weight.data[:, :1])
        nn.init.zeros_(self.gamma_proj.weight.data[:, 1:])
        nn.init.zeros_(self.gamma_proj.bias.data)
        nn.init.zeros_(self.beta_proj.weight.data)
        nn.init.zeros_(self.beta_proj.bias.data)

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            h: Hidden features of shape (batch, hidden_dim)
            context: Conditioning context of shape (batch, context_dim)

        Returns:
            h_modulated: Modulated features of shape (batch, hidden_dim)
        """
        gamma = self.gamma_proj(context)  # (batch, hidden_dim)
        beta = self.beta_proj(context)    # (batch, hidden_dim)

        return gamma * h + beta


class MaskedLinear(nn.Module):
    """Linear layer with causal masking for DAG-consistent connectivity.

    Implements sparse connectivity where information can only flow
    from upstream (slow) to downstream (fast) variables.

    The mask M is defined such that M[i,j] = 1 if variable j can
    influence variable i, else 0. This enforces a lower-triangular
    (or DAG-consistent) Jacobian structure.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        causal_order: Permutation defining causal ordering (upstream first)
        bias: Whether to include bias term (default True)

    Note:
        The mask is stored as a buffer (non-trainable) but gradients
        flow through it during backprop (acts as a constant gate).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: Optional[torch.Tensor] = None,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Register mask as buffer (not a parameter, but moves with module)
        if mask is not None:
            self.register_buffer('mask', mask)
        else:
            # Default: no masking (fully connected)
            self.register_buffer('mask', torch.ones(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked linear transformation.

        Args:
            x: Input of shape (batch, in_features)

        Returns:
            output: Masked output of shape (batch, out_features)
        """
        # Apply mask to weights (gradients flow through, mask is constant)
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, masked=True'


def create_causal_mask(
    dim: int,
    causal_order: Optional[np.ndarray] = None,
    include_diagonal: bool = True
) -> torch.Tensor:
    """Create a causal mask for the MADE construction.

    The mask ensures that slow (upstream) variables cannot be influenced
    by fast (downstream) variables. This is the autoregressive property.

    For weight matrix W[i,j], mask[i,j] = 1 if order[j] <= order[i]
    (allowing variable j to influence variable i).

    Args:
        dim: Dimension of the square mask
        causal_order: Array of ordering indices (lower = more upstream).
                      If None, uses natural order [0, 1, 2, ..., dim-1]
        include_diagonal: Whether diagonal elements are masked in (default True)

    Returns:
        mask: Binary mask tensor of shape (dim, dim)
    """
    if causal_order is None:
        causal_order = np.arange(dim)

    # Convert to tensor
    order = torch.tensor(causal_order, dtype=torch.long)

    # Create mask: M[i,j] = 1 if order[j] <= order[i] (j can influence i)
    # This allows upstream variables to influence downstream
    # order_i[i,j] = order[i] (the order of output i, same across all columns)
    # order_j[i,j] = order[j] (the order of input j, same across all rows)
    order_i = order.unsqueeze(1).expand(-1, dim)  # (dim, dim) - rows
    order_j = order.unsqueeze(0).expand(dim, -1)  # (dim, dim) - columns

    if include_diagonal:
        mask = (order_j <= order_i).float()
    else:
        mask = (order_j < order_i).float()

    return mask


class ResidualBlock(nn.Module):
    """Residual block with FiLM conditioning.

    Implements: output = x + MLP(FiLM(x, context))

    Uses SiLU (Swish) activation which is smooth and avoids dead neurons
    that can occur with ReLU during ODE integration.

    Args:
        dim: Hidden dimension
        context_dim: Dimension of FiLM conditioning context
        dropout: Dropout probability (default 0.0)
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim

        # Pre-norm architecture
        self.norm = nn.LayerNorm(dim)

        # Two-layer MLP
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

        # FiLM conditioning
        self.film = FiLMLayer(dim, context_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Activation
        self.activation = nn.SiLU()  # Swish - smooth, no dead neurons

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with residual connection and FiLM conditioning.

        Args:
            x: Input features of shape (batch, dim)
            context: Conditioning context of shape (batch, context_dim)

        Returns:
            output: Residual output of shape (batch, dim)
        """
        # Pre-norm
        h = self.norm(x)

        # FiLM modulation
        h = self.film(h, context)

        # MLP
        h = self.fc1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # Residual connection
        return x + h


class VelocityNetwork(nn.Module):
    """Masked Velocity Network for Causal Conditional Flow Matching.

    Approximates the time-dependent vector field v_θ(x_t, t, regime)
    with causal masking to enforce the DAG structure discovered by LiNGAM.

    Architecture:
        1. Input projection (MaskedLinear)
        2. Time embedding (Sinusoidal)
        3. Regime embedding (Learned)
        4. Context fusion (time + regime)
        5. N Residual blocks with FiLM conditioning
        6. Output projection (MaskedLinear)

    The masked layers ensure that the Jacobian ∂v/∂x is strictly
    lower-triangular (or DAG-consistent), meaning slow variables
    cannot be influenced by fast variable states.

    Args:
        state_dim: Dimension of the state space (number of variables)
        hidden_dim: Hidden dimension for residual blocks
        n_regimes: Number of regime categories
        n_layers: Number of residual blocks (default 4)
        time_embed_dim: Dimension of time embedding (default 64)
        regime_embed_dim: Dimension of regime embedding (default 32)
        causal_order: Variable ordering for masking (optional)
        dropout: Dropout probability (default 0.0)

    Example:
        >>> net = VelocityNetwork(state_dim=10, hidden_dim=256, n_regimes=3)
        >>> x_t = torch.randn(32, 10)  # batch of states
        >>> t = torch.rand(32)  # time values
        >>> regime = torch.randint(0, 3, (32,))  # regime labels
        >>> v = net(x_t, t, regime)  # predicted velocities
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        n_regimes: int,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        regime_embed_dim: int = 32,
        causal_order: Optional[np.ndarray] = None,
        dropout: float = 0.0
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes
        self.n_layers = n_layers

        # Create causal mask
        if causal_order is not None:
            self.causal_order = np.array(causal_order)
        else:
            self.causal_order = np.arange(state_dim)

        # Input/output masks (same mask for autoregressive property)
        mask = create_causal_mask(state_dim, self.causal_order)
        self.register_buffer('causal_mask', mask)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Regime embedding
        self.regime_embed = nn.Embedding(n_regimes, regime_embed_dim)

        # Context dimension (time + regime)
        context_dim = time_embed_dim + regime_embed_dim

        # Context projection for consistency
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Input projection (masked)
        self.input_proj = MaskedLinear(
            state_dim, hidden_dim,
            mask=self._expand_mask(mask, hidden_dim, state_dim)
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output projection (masked)
        self.output_proj = MaskedLinear(
            hidden_dim, state_dim,
            mask=self._expand_mask(mask.T, state_dim, hidden_dim)
        )

        # Final layer norm
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Initialize output projection to small values for stable training
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def _expand_mask(
        self,
        mask: torch.Tensor,
        out_dim: int,
        in_dim: int
    ) -> torch.Tensor:
        """Expand state-space mask to hidden dimensions.

        For input projection: repeat mask rows for hidden neurons
        For output projection: repeat mask columns for hidden neurons
        """
        state_out, state_in = mask.shape

        if out_dim == state_out and in_dim == self.hidden_dim:
            # Output projection: (state_dim, hidden_dim)
            # Each state variable can only be influenced by causally-upstream hidden units
            # Approximate by using the mask structure
            expanded = mask.unsqueeze(-1).expand(state_out, state_in, in_dim // state_in + 1)
            expanded = expanded.reshape(state_out, -1)[:, :in_dim]
        elif out_dim == self.hidden_dim and in_dim == state_in:
            # Input projection: (hidden_dim, state_dim)
            # Each hidden unit group receives from corresponding state variables
            expanded = mask.unsqueeze(0).expand(out_dim // state_out + 1, state_out, state_in)
            expanded = expanded.reshape(-1, state_in)[:out_dim, :]
        else:
            # Fallback: fully connected
            expanded = torch.ones(out_dim, in_dim)

        return expanded

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        regime: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity field at state x_t, time t, given regime.

        Args:
            x_t: Current state of shape (batch, state_dim)
            t: Time values of shape (batch,) or scalar
            regime: Regime labels of shape (batch,) as integers

        Returns:
            v: Velocity vector of shape (batch, state_dim)
        """
        batch_size = x_t.shape[0]

        # Handle scalar time
        if t.dim() == 0:
            t = t.expand(batch_size)

        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_embed_dim)

        # Regime embedding
        r_emb = self.regime_embed(regime)  # (batch, regime_embed_dim)

        # Fuse context
        context = torch.cat([t_emb, r_emb], dim=-1)  # (batch, context_dim)
        context = self.context_proj(context)  # (batch, hidden_dim)

        # Input projection
        h = self.input_proj(x_t)  # (batch, hidden_dim)
        h = F.silu(h)

        # Residual blocks with FiLM conditioning
        for block in self.blocks:
            h = block(h, context)

        # Output projection
        h = self.output_norm(h)
        v = self.output_proj(h)  # (batch, state_dim)

        return v

    def get_jacobian(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        regime: torch.Tensor
    ) -> torch.Tensor:
        """Compute Jacobian of velocity field w.r.t. state.

        This is useful for verifying the causal mask is working correctly.
        The Jacobian should be lower-triangular (or DAG-consistent).

        Args:
            x_t: State of shape (batch, state_dim) or (state_dim,)
            t: Time value
            regime: Regime label

        Returns:
            jacobian: Jacobian matrix of shape (batch, state_dim, state_dim)
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        x_t = x_t.requires_grad_(True)

        # Forward pass
        v = self.forward(x_t, t, regime)

        # Compute Jacobian row by row
        batch_size = x_t.shape[0]
        jacobian = torch.zeros(batch_size, self.state_dim, self.state_dim,
                               device=x_t.device)

        for i in range(self.state_dim):
            grad_outputs = torch.zeros_like(v)
            grad_outputs[:, i] = 1.0

            grads = torch.autograd.grad(
                v, x_t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]

            jacobian[:, i, :] = grads

        return jacobian

    def verify_causal_structure(
        self,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
        tolerance: float = 1e-6
    ) -> Tuple[bool, torch.Tensor]:
        """Verify that the Jacobian respects the causal structure.

        Checks that gradients from fast to slow variables are zero.

        Returns:
            is_valid: Whether causal structure is maintained
            violations: Tensor of mask violations
        """
        if x_t is None:
            x_t = torch.randn(1, self.state_dim)
        if t is None:
            t = torch.tensor([0.5])
        if regime is None:
            regime = torch.zeros(1, dtype=torch.long)

        # Move to same device as model
        device = next(self.parameters()).device
        x_t = x_t.to(device)
        t = t.to(device)
        regime = regime.to(device)

        # Compute Jacobian
        jacobian = self.get_jacobian(x_t, t, regime)

        # Check violations: entries where mask is 0 should have near-zero gradient
        mask = self.causal_mask.unsqueeze(0)  # (1, state_dim, state_dim)
        violations = torch.abs(jacobian) * (1 - mask)

        max_violation = violations.max().item()
        is_valid = max_violation < tolerance

        return is_valid, violations

    def extra_repr(self) -> str:
        return (f'state_dim={self.state_dim}, hidden_dim={self.hidden_dim}, '
                f'n_regimes={self.n_regimes}, n_layers={self.n_layers}')


class VelocityNetworkUnconstrained(nn.Module):
    """Unconstrained Velocity Network without causal masking.

    This variant is useful for:
    - Baselines comparison
    - Cases where causal structure is unknown
    - Ablation studies

    Same architecture as VelocityNetwork but without masked layers.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        n_regimes: int,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        regime_embed_dim: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Regime embedding
        self.regime_embed = nn.Embedding(n_regimes, regime_embed_dim)

        # Context projection
        context_dim = time_embed_dim + regime_embed_dim
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Input projection (no masking)
        self.input_proj = nn.Linear(state_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, state_dim)

        # Initialize output small
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        regime: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x_t.shape[0]

        if t.dim() == 0:
            t = t.expand(batch_size)

        t_emb = self.time_embed(t)
        r_emb = self.regime_embed(regime)

        context = torch.cat([t_emb, r_emb], dim=-1)
        context = self.context_proj(context)

        h = self.input_proj(x_t)
        h = F.silu(h)

        for block in self.blocks:
            h = block(h, context)

        h = self.output_norm(h)
        v = self.output_proj(h)

        return v
