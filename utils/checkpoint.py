"""
Checkpoint Utilities

Provides utilities for saving and loading model checkpoints.
"""

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path

from .distributed import is_main_process, get_rank


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    save_dir: Optional[str] = None
):
    """
    Save training checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
        save_dir: Optional directory to save additional copies
    """
    if not is_main_process():
        return

    # Create directory
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save checkpoint
    torch.save(state, filepath)

    # Save best model separately
    if is_best and save_dir:
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(state, best_path)


def load_checkpoint(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load checkpoint to

    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if model is not None:
        model_state = checkpoint.get('model_state', checkpoint.get('model'))
        if model_state is not None:
            model.load_state_dict(model_state)

    # Load optimizer state
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    return checkpoint


def get_checkpoint_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading full model.

    Args:
        filepath: Path to checkpoint file

    Returns:
        Dictionary with checkpoint metadata
    """
    if not os.path.exists(filepath):
        return {'exists': False}

    checkpoint = torch.load(filepath, map_location='cpu')
    return {
        'exists': True,
        'epoch': checkpoint.get('epoch', None),
        'step': checkpoint.get('step', None),
        'best_metric': checkpoint.get('best_metric', None),
        'metrics': checkpoint.get('metrics', {}),
    }


def get_latest_checkpoint(checkpoint_dir: str, pattern: str = 'checkpoint_*.pt') -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 5,
    keep_best: bool = True
):
    """
    Clean up old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
        keep_best: Whether to always keep best_model.pt
    """
    if not is_main_process():
        return

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    # Get all checkpoint files (excluding best_model.pt)
    checkpoints = [
        p for p in checkpoint_dir.glob('checkpoint_*.pt')
        if p.name != 'best_model.pt'
    ]

    if len(checkpoints) <= keep_last:
        return

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Delete old checkpoints
    for checkpoint in checkpoints[keep_last:]:
        checkpoint.unlink()


class CheckpointManager:
    """
    Manager for handling checkpoint saving and loading.

    Automatically manages checkpoint rotation and best model tracking.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last: int = 5,
        keep_best: bool = True,
        save_every: int = 1
    ):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last: Number of recent checkpoints to keep
            keep_best: Whether to keep best model
            save_every: Save checkpoint every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.save_every = save_every
        self.best_metric = float('inf') if keep_best else None

        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def should_save(self, epoch: int) -> bool:
        """Check if we should save a checkpoint this epoch."""
        return (epoch + 1) % self.save_every == 0

    def is_best(self, metric: float) -> bool:
        """Check if current metric is the best."""
        if self.best_metric is None:
            return False
        return metric < self.best_metric

    def save(
        self,
        epoch: int,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch
            step: Current training step
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save (optional)
            metrics: Training metrics to save
        """
        if not is_main_process():
            return

        # Check if this is the best model
        metric_value = None
        is_best = False
        if metrics and 'loss' in metrics:
            metric_value = metrics['loss']
            is_best = self.is_best(metric_value)
            if is_best:
                self.best_metric = metric_value

        # Build checkpoint state
        state = {
            'epoch': epoch,
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics or {},
        }

        if scheduler is not None:
            state['scheduler_state'] = scheduler.state_dict()

        if metric_value is not None:
            state['metric_value'] = metric_value
            state['best_metric'] = self.best_metric

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        save_checkpoint(state, str(checkpoint_path), is_best, str(self.checkpoint_dir))

        # Clean up old checkpoints
        if epoch > 0:
            cleanup_old_checkpoints(str(self.checkpoint_dir), self.keep_last, self.keep_best)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint."""
        latest = get_latest_checkpoint(str(self.checkpoint_dir))
        if latest is None:
            return None
        return torch.load(latest, map_location='cpu')

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        if not best_path.exists():
            return None
        return torch.load(best_path, map_location='cpu')
