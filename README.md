# Causal Conditional Flow Matching (C-CFM)

A generative time-series framework based on **Continuous Normalizing Flows** with causal structure enforcement for economic scenario generation and stress testing.

## Overview

C-CFM combines modern flow-based generative models with causal inference to produce realistic economic scenarios that respect the underlying causal structure of financial markets. The key innovation is enforcing that "slow" macro variables (GDP, CPI) cannot be influenced by "fast" market variables (VIX, returns), while allowing appropriate causal propagation in the opposite direction.

## Key Features

- **CTree-Lite**: Custom conditional inference tree for regime classification based on statistical significance (not impurity reduction)
- **Causal Graph Discovery**: LiNGAM-based discovery of market microstructure
- **Masked Velocity Network**: Neural network with DAG-consistent connectivity via MADE-style masking
- **FiLM Conditioning**: Feature-wise Linear Modulation for regime and time conditioning
- **Simulation-Free Training**: Optimal Transport path interpolation for efficient training
- **Guided ODE Integration**: Vector field steering for stress testing and scenario generation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from core import CausalFlowMatcher
import numpy as np

# Load your time series data (T timesteps, D variables)
X = np.load('economic_data.npy')

# Initialize and fit
cfm = CausalFlowMatcher(hidden_dim=256)
cfm.fit(
    X,
    fast_vars=['VIX', 'SPX_ret', 'Credit_Spread'],
    slow_vars=['GDP_Growth', 'CPI', 'Unemployment']
)

# Train the model
cfm.train(n_epochs=500)

# Generate baseline scenarios
baseline = cfm.sample(n_samples=1000)

# Generate stress scenarios (VIX shock of 3 standard deviations)
stressed = cfm.shock('VIX', magnitude=3.0, n_samples=1000)

# Save the model
cfm.save('trained_model.pt')
```

## Architecture

### 1. Data Topology Module (`core/etl.py`)
- Stationarity enforcement via ADF tests
- Cubic spline imputation for missing values
- CTree-Lite regime classification
- LiNGAM causal graph discovery
- Z-score normalization for OT-path stability

### 2. CTree-Lite (`core/ctree.py`)
- Strasser-Weber framework for independence testing
- P-value based variable selection
- Statistical significance stopping criterion
- Robust to overfitting on financial noise

### 3. Velocity Network (`core/network.py`)
- 4-layer Residual MLP with SiLU activation
- MaskedLinear layers for causal structure
- Sinusoidal time embeddings
- FiLM modulation for regime conditioning

### 4. Flow Trainer (`core/trainer.py`)
- Conditional Flow Matching objective
- Optimal Transport interpolation paths
- AdamW optimizer with Cosine Annealing
- Causal gradient verification

### 5. ODE Solver (`core/solver.py`)
- Dopri5 adaptive step integration
- Guided velocity fields for stress testing
- Multi-target scenario conditioning
- Denormalization utilities

## Mathematical Background

### Conditional Flow Matching

The training objective minimizes:
```
L = E_{t,x_0,x_1} [ ||v_θ(x_t, t) - u_t||² ]
```
where:
- `x_t = (1-t)*x_0 + t*x_1` (OT interpolation)
- `u_t = x_1 - x_0` (target velocity)
- `x_0 ~ N(0, I)` (noise prior)
- `x_1 ~ p_data` (real data)

### Causal Masking

The Jacobian of the velocity field is constrained to be DAG-consistent:
```
∂v_i/∂x_j = 0 if j is downstream of i
```
This ensures shocks propagate correctly through the causal graph.

### Guided Integration

For stress testing, we modify the velocity field:
```
v_guided = v_nominal + λ * (target - x[i]) / (1 - t + ε)
```
This steers variable `i` toward the target while maintaining causal consistency.

## References

- Lipman et al. (2022): Flow Matching for Generative Modeling
- Hothorn et al. (2006): Unbiased Recursive Partitioning: A Conditional Inference Framework
- Shimizu et al. (2006): LiNGAM - A Linear Non-Gaussian Acyclic Model
- Perez et al. (2018): FiLM: Visual Reasoning with Feature-wise Linear Modulation
- Germain et al. (2015): MADE: Masked Autoencoder for Distribution Estimation

## License

MIT License
