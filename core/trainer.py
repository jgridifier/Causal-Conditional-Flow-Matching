"""
Flow Trainer for Causal Conditional Flow Matching

Implements the simulation-free training objective using Optimal Transport paths.
The key insight is that we don't need to solve the ODE during training - we can
directly regress the velocity field against the analytical target.

Training Objective (CFM):
    L = E_{t,x_0,x_1} [ ||v_θ(x_t, t) - u_t(x_t | x_0, x_1)||² ]

    where:
    - x_t = (1-t)*x_0 + t*x_1  (OT interpolation path)
    - u_t = x_1 - x_0          (target velocity for OT path)
    - x_0 ~ N(0, I)            (noise prior)
    - x_1 ~ p_data             (real data)

References:
    - Lipman et al. (2022): Flow Matching for Generative Modeling
    - Tong et al. (2023): Improving and Generalizing Flow-Based Generative Models
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple, Any, Callable
from dataclasses import dataclass, field
import warnings

from .network import VelocityNetwork
from .etl import DataTopology


@dataclass
class TrainingConfig:
    """Configuration for CFM training.

    Attributes:
        lr: Learning rate (default 1e-4)
        weight_decay: AdamW weight decay (default 1e-5)
        batch_size: Batch size for training (default 256)
        n_epochs: Number of training epochs (default 1000)
        grad_clip: Gradient clipping norm (default 1.0)
        warmup_epochs: Linear warmup epochs (default 10)
        min_lr: Minimum learning rate for cosine annealing (default 1e-6)
        validate_every: Validation frequency in epochs (default 10)
        checkpoint_every: Checkpoint frequency in epochs (default 100)
        check_causal_gradients: Verify causal structure periodically (default True)
        device: Training device (default 'cuda' if available)
    """
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    n_epochs: int = 1000
    grad_clip: float = 1.0
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    validate_every: int = 10
    checkpoint_every: int = 100
    check_causal_gradients: bool = True
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TrainingState:
    """Mutable training state for checkpointing.

    Attributes:
        epoch: Current epoch
        global_step: Global training step
        best_loss: Best validation loss seen
        loss_history: History of training losses
        val_loss_history: History of validation losses
        grad_norms: History of gradient norms
    """
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    loss_history: list = field(default_factory=list)
    val_loss_history: list = field(default_factory=list)
    grad_norms: list = field(default_factory=list)


class FlowMatchingTrainer:
    """Trainer for Causal Conditional Flow Matching.

    Implements the simulation-free CFM objective with OT-path interpolation.
    Training regresses the velocity network against analytical target velocities.

    Key Features:
    - Optimal Transport interpolation paths
    - AdamW optimizer with Cosine Annealing schedule
    - Causal gradient verification (ensures mask is working)
    - Gradient clipping for stability
    - Checkpointing and early stopping support

    Args:
        model: VelocityNetwork to train
        topology: DataTopology containing processed data and metadata
        config: TrainingConfig with hyperparameters

    Example:
        >>> trainer = FlowMatchingTrainer(model, topology, config)
        >>> trainer.train()
        >>> # Access trained model
        >>> model = trainer.model
    """

    def __init__(
        self,
        model: VelocityNetwork,
        topology: DataTopology,
        config: Optional[TrainingConfig] = None
    ):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        # Model
        self.model = model.to(self.device)

        # Data
        self.topology = topology
        self._prepare_data()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.n_epochs,
            eta_min=self.config.min_lr
        )

        # Training state
        self.state = TrainingState()

        # Hooks for custom callbacks
        self.callbacks: list = []

    def _prepare_data(self) -> None:
        """Prepare data loaders from topology."""
        X = torch.tensor(self.topology.X_processed, dtype=torch.float32)
        regimes = torch.tensor(self.topology.regimes, dtype=torch.long)

        # Split into train/val (90/10)
        n_samples = len(X)
        n_val = max(1, int(0.1 * n_samples))
        indices = torch.randperm(n_samples)

        train_idx = indices[:-n_val]
        val_idx = indices[-n_val:]

        train_dataset = TensorDataset(X[train_idx], regimes[train_idx])
        val_dataset = TensorDataset(X[val_idx], regimes[val_idx])

        # Only drop last batch if we'd still have at least one batch
        # This avoids zero batches when batch_size > dataset_size
        effective_drop_last = len(train_idx) > self.config.batch_size

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=effective_drop_last
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        self.n_train = len(train_idx)
        self.n_val = len(val_idx)

    def sample_ot_path(
        self,
        x1: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample points along the Optimal Transport path.

        The OT path connects noise x_0 ~ N(0,I) to data x_1 via:
            x_t = (1 - t) * x_0 + t * x_1

        The target velocity for this path is simply:
            u_t = x_1 - x_0

        Args:
            x1: Data samples of shape (batch, dim)
            batch_size: Number of samples

        Returns:
            x_t: Interpolated points of shape (batch, dim)
            t: Time values of shape (batch,)
            u_t: Target velocities of shape (batch, dim)
            x_0: Noise samples of shape (batch, dim)
        """
        dim = x1.shape[-1]

        # Sample noise prior
        x_0 = torch.randn_like(x1)

        # Sample time uniformly
        t = torch.rand(batch_size, device=x1.device)

        # OT interpolation: x_t = (1-t)*x_0 + t*x_1
        t_expanded = t.unsqueeze(-1)  # (batch, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x1

        # Target velocity: dx_t/dt = x_1 - x_0
        u_t = x1 - x_0

        return x_t, t, u_t, x_0

    def compute_loss(
        self,
        x1: torch.Tensor,
        regime: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the CFM loss for a batch.

        Loss = E[ ||v_θ(x_t, t, regime) - u_t||² ]

        Args:
            x1: Data batch of shape (batch, dim)
            regime: Regime labels of shape (batch,)

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of additional metrics
        """
        batch_size = x1.shape[0]

        # Sample OT path
        x_t, t, u_t, x_0 = self.sample_ot_path(x1, batch_size)

        # Predict velocity
        v_pred = self.model(x_t, t, regime)

        # MSE loss
        loss = F.mse_loss(v_pred, u_t)

        # Compute per-dimension errors for diagnostics
        with torch.no_grad():
            per_dim_error = (v_pred - u_t).pow(2).mean(dim=0)

            # Separate slow and fast variable errors
            slow_idx = torch.tensor(self.topology.slow_indices, device=x1.device)
            fast_idx = torch.tensor(self.topology.fast_indices, device=x1.device)

            slow_error = per_dim_error[slow_idx].mean().item() if len(slow_idx) > 0 else 0.0
            fast_error = per_dim_error[fast_idx].mean().item() if len(fast_idx) > 0 else 0.0

        metrics = {
            'loss': loss.item(),
            'slow_error': slow_error,
            'fast_error': fast_error,
        }

        return loss, metrics

    def train_step(
        self,
        x1: torch.Tensor,
        regime: torch.Tensor
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            x1: Data batch
            regime: Regime labels

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Compute loss
        loss, metrics = self.compute_loss(x1, regime)

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )
        metrics['grad_norm'] = grad_norm.item()

        # Optimizer step
        self.optimizer.step()

        self.state.global_step += 1

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_slow_error = 0.0
        total_fast_error = 0.0
        n_batches = 0

        for x1, regime in self.val_loader:
            x1 = x1.to(self.device)
            regime = regime.to(self.device)

            _, metrics = self.compute_loss(x1, regime)

            total_loss += metrics['loss']
            total_slow_error += metrics['slow_error']
            total_fast_error += metrics['fast_error']
            n_batches += 1

        # Guard against zero batches (batch_size > dataset_size)
        if n_batches == 0:
            return {
                'val_loss': float('nan'),
                'val_slow_error': float('nan'),
                'val_fast_error': float('nan'),
            }

        return {
            'val_loss': total_loss / n_batches,
            'val_slow_error': total_slow_error / n_batches,
            'val_fast_error': total_fast_error / n_batches,
        }

    def check_causal_gradients(self) -> bool:
        """Verify that causal structure is maintained in gradients.

        The gradients of slow variables w.r.t. fast inputs must be exactly 0.
        This is a sanity check that the masking is working correctly.

        Returns:
            is_valid: Whether causal structure is maintained
        """
        if not hasattr(self.model, 'verify_causal_structure'):
            return True

        # Create test inputs
        x = torch.randn(1, self.model.state_dim, device=self.device)
        t = torch.tensor([0.5], device=self.device)
        regime = torch.zeros(1, dtype=torch.long, device=self.device)

        is_valid, violations = self.model.verify_causal_structure(x, t, regime)

        if not is_valid:
            max_violation = violations.max().item()
            warnings.warn(
                f"Causal structure violation detected! Max gradient: {max_violation:.2e}. "
                f"Check mask implementation."
            )

        return is_valid

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            results: Dictionary containing training history and final metrics
        """
        print(f"Starting training on {self.device}")
        print(f"  Train samples: {self.n_train}")
        print(f"  Val samples: {self.n_val}")
        print(f"  Epochs: {self.config.n_epochs}")
        print(f"  Batch size: {self.config.batch_size}")

        # Initial causal check
        if self.config.check_causal_gradients:
            self.check_causal_gradients()

        for epoch in range(self.config.n_epochs):
            self.state.epoch = epoch

            # Warmup learning rate
            if epoch < self.config.warmup_epochs:
                warmup_factor = (epoch + 1) / self.config.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config.lr * warmup_factor

            # Training epoch
            epoch_metrics = self._train_epoch()
            self.state.loss_history.append(epoch_metrics['loss'])
            self.state.grad_norms.append(epoch_metrics['grad_norm'])

            # Learning rate step (after warmup)
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # Validation
            if (epoch + 1) % self.config.validate_every == 0:
                val_metrics = self.validate()
                self.state.val_loss_history.append(val_metrics['val_loss'])

                # Track best
                if val_metrics['val_loss'] < self.state.best_loss:
                    self.state.best_loss = val_metrics['val_loss']

                # Log progress
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch+1}/{self.config.n_epochs} | "
                    f"Train: {epoch_metrics['loss']:.4f} | "
                    f"Val: {val_metrics['val_loss']:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Grad: {epoch_metrics['grad_norm']:.2f}"
                )

            # Periodic causal check
            if self.config.check_causal_gradients and (epoch + 1) % 100 == 0:
                self.check_causal_gradients()

            # Callbacks
            for callback in self.callbacks:
                callback(self, epoch, epoch_metrics)

        # Final validation
        final_metrics = self.validate()
        print(f"\nTraining complete!")
        print(f"  Final val loss: {final_metrics['val_loss']:.4f}")
        print(f"  Best val loss: {self.state.best_loss:.4f}")

        return {
            'final_metrics': final_metrics,
            'loss_history': self.state.loss_history,
            'val_loss_history': self.state.val_loss_history,
            'best_loss': self.state.best_loss,
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            metrics: Averaged metrics for the epoch
        """
        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0

        for x1, regime in self.train_loader:
            x1 = x1.to(self.device)
            regime = regime.to(self.device)

            metrics = self.train_step(x1, regime)

            total_loss += metrics['loss']
            total_grad_norm += metrics['grad_norm']
            n_batches += 1

        # Guard against zero batches (batch_size > dataset_size)
        if n_batches == 0:
            return {
                'loss': float('nan'),
                'grad_norm': float('nan'),
            }

        return {
            'loss': total_loss / n_batches,
            'grad_norm': total_grad_norm / n_batches,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_state': {
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_loss': self.state.best_loss,
                'loss_history': self.state.loss_history,
                'val_loss_history': self.state.val_loss_history,
            },
            'topology': self.topology.to_dict(),
            'config': {
                'lr': self.config.lr,
                'weight_decay': self.config.weight_decay,
                'batch_size': self.config.batch_size,
                'n_epochs': self.config.n_epochs,
            }
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        state = checkpoint['training_state']
        self.state.epoch = state['epoch']
        self.state.global_step = state['global_step']
        self.state.best_loss = state['best_loss']
        self.state.loss_history = state['loss_history']
        self.state.val_loss_history = state['val_loss_history']


class CFMLoss(nn.Module):
    """Standalone CFM loss module for custom training loops.

    Can be used independently of the Trainer class for integration
    with other training frameworks.

    Example:
        >>> loss_fn = CFMLoss()
        >>> v_pred = model(x_t, t, regime)
        >>> loss = loss_fn(v_pred, target_velocity)
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute CFM loss.

        Args:
            v_pred: Predicted velocities of shape (batch, dim)
            v_target: Target velocities of shape (batch, dim)
            mask: Optional mask for selective loss computation

        Returns:
            loss: Scalar loss value
        """
        diff = v_pred - v_target
        squared_error = diff.pow(2)

        if mask is not None:
            squared_error = squared_error * mask

        if self.reduction == 'mean':
            return squared_error.mean()
        elif self.reduction == 'sum':
            return squared_error.sum()
        else:
            return squared_error


def create_trainer(
    state_dim: int,
    hidden_dim: int,
    n_regimes: int,
    topology: DataTopology,
    causal_order: Optional[np.ndarray] = None,
    config: Optional[TrainingConfig] = None,
    **model_kwargs
) -> FlowMatchingTrainer:
    """Factory function to create a trainer with model.

    Args:
        state_dim: Dimension of state space
        hidden_dim: Hidden dimension for network
        n_regimes: Number of regime categories
        topology: Processed data topology
        causal_order: Causal ordering of variables
        config: Training configuration
        **model_kwargs: Additional arguments for VelocityNetwork

    Returns:
        trainer: Configured FlowMatchingTrainer
    """
    # Use topology's causal order if not provided
    if causal_order is None:
        causal_order = topology.causal_order

    model = VelocityNetwork(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        n_regimes=n_regimes,
        causal_order=causal_order,
        **model_kwargs
    )

    return FlowMatchingTrainer(model, topology, config)
