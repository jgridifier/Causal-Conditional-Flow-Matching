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


def assign_degrees(
    hidden_dim: int,
    state_dim: int,
    min_degree: int = 0,
    max_degree: Optional[int] = None
) -> np.ndarray:
    """Assign degrees to hidden units for MADE masking.

    Each hidden layer must contain all degrees to ensure full connectivity
    (MADE requirement from Germain et al., 2015). Degrees determine which
    state variables each hidden unit can "see" in the autoregressive ordering.

    Args:
        hidden_dim: Number of hidden units
        state_dim: Dimension of state space
        min_degree: Minimum degree (default 0)
        max_degree: Maximum degree (default state_dim-1)

    Returns:
        degrees: Array of degrees for each hidden unit

    Raises:
        ValueError: If hidden_dim is too small to accommodate all degrees
    """
    if max_degree is None:
        max_degree = state_dim - 1

    num_degrees = max_degree - min_degree + 1

    # Ensure we have at least one unit per degree
    if hidden_dim < num_degrees:
        raise ValueError(
            f"Hidden dim {hidden_dim} too small for state dim {state_dim}. "
            f"Need at least {num_degrees} hidden units to represent all degrees "
            f"[{min_degree}, {max_degree}]."
        )

    # Distribute degrees evenly across hidden units
    # This ensures each degree appears at least once (MADE requirement)
    degrees = np.zeros(hidden_dim, dtype=np.int32)
    for i in range(hidden_dim):
        degrees[i] = min_degree + (i % num_degrees)

    return degrees


def create_hidden_mask(
    degrees_in: np.ndarray,
    degrees_out: np.ndarray
) -> torch.Tensor:
    """Create mask for hidden-to-hidden connections in MADE architecture.

    Implements the autoregressive constraint: a hidden unit can only receive
    information from units with equal or lower degree. This ensures that
    the Jacobian remains lower-triangular throughout the network.

    Mask[i,j] = 1 if degree_in[j] <= degree_out[i] (unit j can influence unit i)

    Args:
        degrees_in: Degrees of input units (which variables they can see)
        degrees_out: Degrees of output units

    Returns:
        mask: Binary mask of shape (len(degrees_out), len(degrees_in))
              where mask[i,j]=1 allows connection from input j to output i
    """
    degrees_in = torch.tensor(degrees_in, dtype=torch.long)
    degrees_out = torch.tensor(degrees_out, dtype=torch.long)

    # Broadcast to create mask
    # mask[i,j] = 1 if degrees_in[j] <= degrees_out[i]
    # This allows lower-degree (upstream) units to influence higher-degree units
    mask = (degrees_in.unsqueeze(0) <= degrees_out.unsqueeze(1)).float()

    return mask


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
    """Residual block with FiLM conditioning and optional causal masking.

    Implements: output = x + MLP(FiLM(x, context))

    Uses SiLU (Swish) activation which is smooth and avoids dead neurons
    that can occur with ReLU during ODE integration.

    For MADE-compliant causal masking, pass hidden_degrees to enforce
    autoregressive constraints in the MLP layers.

    Args:
        dim: Hidden dimension
        context_dim: Dimension of FiLM conditioning context
        hidden_degrees: Optional degree assignment for hidden units (for MADE).
                       If provided, fc1 and fc2 will use MaskedLinear layers.
        expansion_factor: Expansion factor for MLP (default 4)
        dropout: Dropout probability (default 0.0)
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_degrees: Optional[np.ndarray] = None,
        expansion_factor: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.expansion_factor = expansion_factor
        expanded_dim = dim * expansion_factor

        # NOTE: We do NOT use LayerNorm here for MADE-compliant networks because
        # LayerNorm mixes information across all hidden units, violating the
        # autoregressive property required for causal structure.

        # FiLM conditioning
        self.film = FiLMLayer(dim, context_dim)

        # Two-layer MLP with optional masking
        if hidden_degrees is not None:
            # Assign degrees to expanded hidden layer
            # The expanded layer gets a finer degree distribution
            expanded_degrees = assign_degrees(
                expanded_dim,
                len(np.unique(hidden_degrees)),
                min_degree=hidden_degrees.min(),
                max_degree=hidden_degrees.max()
            )

            # Create masks for fc1 (dim -> expanded_dim) and fc2 (expanded_dim -> dim)
            mask_fc1 = create_hidden_mask(hidden_degrees, expanded_degrees)
            mask_fc2 = create_hidden_mask(expanded_degrees, hidden_degrees)

            self.fc1 = MaskedLinear(dim, expanded_dim, mask=mask_fc1)
            self.fc2 = MaskedLinear(expanded_dim, dim, mask=mask_fc2)
        else:
            # Fallback: fully connected (for unconstrained network)
            self.fc1 = nn.Linear(dim, expanded_dim)
            self.fc2 = nn.Linear(expanded_dim, dim)

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
        # FiLM modulation (no LayerNorm to preserve causal structure)
        h = self.film(x, context)

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

        # Create causal mask and degree assignments
        if causal_order is not None:
            self.causal_order = np.array(causal_order)
        else:
            self.causal_order = np.arange(state_dim)

        # Store causal mask for verification
        mask = create_causal_mask(state_dim, self.causal_order)
        self.register_buffer('causal_mask', mask)

        # Assign degrees to hidden units (MADE requirement)
        # Each hidden unit is assigned a degree indicating which state variables it can see
        self.hidden_degrees = assign_degrees(hidden_dim, state_dim)

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

        # Input projection mask: state variables -> hidden units
        # State variable i has degree = causal_order[i]
        # Hidden unit j can only see state variables with degree <= hidden_degrees[j]
        input_mask = create_hidden_mask(self.causal_order, self.hidden_degrees)

        # Input projection (masked)
        self.input_proj = MaskedLinear(
            state_dim, hidden_dim,
            mask=input_mask
        )

        # Residual blocks with masked connections
        self.blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim,
                hidden_dim,
                hidden_degrees=self.hidden_degrees,  # Enable MADE masking
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output projection mask: hidden units -> state variables
        # Transpose of input logic: state variable i can only be influenced by
        # hidden units with degree <= causal_order[i]
        output_mask = create_hidden_mask(self.hidden_degrees, self.causal_order)

        # Output projection (masked)
        self.output_proj = MaskedLinear(
            hidden_dim, state_dim,
            mask=output_mask
        )

        # NOTE: We do NOT use LayerNorm before output projection because it would
        # mix information across degrees, violating the causal structure.
        # LayerNorm normalizes across the entire hidden dimension, creating
        # dependencies between all hidden units, which breaks MADE's autoregressive property.

        # Initialize output projection to small values for stable training
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)

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

        # Output projection (no LayerNorm to preserve causal structure)
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
