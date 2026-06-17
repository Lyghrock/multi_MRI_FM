"""
Metrics Utilities

Provides utilities for computing training and evaluation metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None

from .distributed import reduce_tensor, get_world_size


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metrics_list: List[str] = None
) -> Dict[str, float]:
    """
    Compute various metrics between predictions and targets.

    Args:
        predictions: Prediction tensor
        targets: Target tensor
        metrics_list: List of metrics to compute. If None, compute all.

    Returns:
        Dictionary of metric names and values
    """
    if metrics_list is None:
        metrics_list = ['mse', 'mae', 'correlation']

    results = {}

    # Move to CPU and numpy
    pred = predictions.detach().cpu().numpy()
    tgt = targets.detach().cpu().numpy()

    # Flatten if multi-dimensional
    if pred.ndim > 1:
        pred = pred.flatten()
        tgt = tgt.flatten()

    # MSE
    if 'mse' in metrics_list:
        results['mse'] = float(np.mean((pred - tgt) ** 2))

    # MAE
    if 'mae' in metrics_list:
        results['mae'] = float(np.mean(np.abs(pred - tgt)))

    # RMSE
    if 'rmse' in metrics_list:
        results['rmse'] = float(np.sqrt(np.mean((pred - tgt) ** 2)))

    # Pearson Correlation
    if 'correlation' in metrics_list or 'pearson' in metrics_list:
        if len(pred) > 1:
            if pearsonr is not None:
                corr, p_value = pearsonr(pred, tgt)
            else:
                corr = np.corrcoef(pred, tgt)[0, 1]
                p_value = np.nan
            results['pearson'] = float(corr)
            results['pearson_pvalue'] = float(p_value)
        else:
            results['pearson'] = 0.0
            results['pearson_pvalue'] = 1.0

    # Spearman Correlation
    if 'spearman' in metrics_list:
        if len(pred) > 1:
            if spearmanr is None:
                raise ImportError("scipy is required to compute spearman metrics")
            corr, p_value = spearmanr(pred, tgt)
            results['spearman'] = float(corr)
            results['spearman_pvalue'] = float(p_value)
        else:
            results['spearman'] = 0.0
            results['spearman_pvalue'] = 1.0

    # R-squared
    if 'r2' in metrics_list or 'r_squared' in metrics_list:
        ss_res = np.sum((pred - tgt) ** 2)
        ss_tot = np.sum((tgt - np.mean(tgt)) ** 2)
        if ss_tot > 0:
            results['r2'] = float(1 - ss_res / ss_tot)
        else:
            results['r2'] = 0.0

    # Explained Variance
    if 'explained_variance' in metrics_list:
        var_pred = np.var(pred)
        var_target = np.var(tgt)
        if var_target > 0:
            results['explained_variance'] = float(var_pred / var_target)
        else:
            results['explained_variance'] = 0.0

    return results


def compute_reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute metrics for reconstruction tasks.

    Args:
        pred: Predicted values
        target: Target values
        mask: Optional mask (1 for valid, 0 for masked/ignore)

    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        # Apply mask
        valid_mask = mask.flatten().bool()
        pred = pred.flatten()[valid_mask]
        target = target.flatten()[valid_mask]

    return compute_metrics(pred, target)


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for classification tasks.

    Args:
        predictions: Prediction logits or probabilities
        targets: Ground truth labels

    Returns:
        Dictionary of metrics
    """
    results = {}

    pred_probs = torch.softmax(predictions, dim=-1)
    pred_labels = torch.argmax(pred_probs, dim=-1)

    # Accuracy
    correct = (pred_labels == targets).sum().item()
    total = targets.numel()
    results['accuracy'] = correct / total

    # Per-class accuracy
    for c in range(predictions.shape[-1]):
        class_mask = (targets == c)
        if class_mask.sum() > 0:
            class_acc = (pred_labels[class_mask] == c).sum().item() / class_mask.sum().item()
            results[f'accuracy_class_{c}'] = class_acc

    # AUC (if binary or can compute)
    if predictions.shape[-1] == 2:
        if roc_auc_score is None or average_precision_score is None:
            raise ImportError("scikit-learn is required to compute AUC/AP metrics")
        probs = pred_probs[:, 1].cpu().numpy()
        labels = targets.cpu().numpy()
        results['auc'] = roc_auc_score(labels, probs)
        results['ap'] = average_precision_score(labels, probs)

    return results


def compute_vicreg_metrics(
    representations: torch.Tensor,
    representations_target: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute metrics for VICReg-style representations.

    Args:
        representations: Batch of representations (B, N, D)
        representations_target: Optional target representations

    Returns:
        Dictionary of metrics
    """
    results = {}

    # Variance per dimension
    var = torch.var(representations, dim=0)  # (N, D)
    results['var_mean'] = var.mean().item()
    results['var_min'] = var.min().item()
    results['var_std'] = var.std().item()

    # Norm statistics
    norms = torch.norm(representations, dim=-1)  # (B, N)
    results['norm_mean'] = norms.mean().item()
    results['norm_std'] = norms.std().item()

    # Invariance (if target provided)
    if representations_target is not None:
        invariance = torch.mean((representations - representations_target) ** 2)
        results['invariance_loss'] = invariance.item()

    return results


def reduce_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Reduce metrics across all processes in distributed setting.

    Args:
        metrics: Dictionary of metrics from one process

    Returns:
        Reduced metrics (valid on all processes)
    """
    world_size = get_world_size()
    if world_size == 1:
        return metrics

    reduced_metrics = {}

    for name, value in metrics.items():
        tensor = torch.tensor(value, dtype=torch.float32)
        reduced = reduce_tensor(tensor)
        reduced_metrics[name] = reduced.item()

    return reduced_metrics


class MetricTracker:
    """
    Track metrics over training.
    """

    def __init__(self):
        self.history = []
        self.current_epoch = []

    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        self.current_epoch.append(metrics)

    def epoch_end(self) -> Dict[str, float]:
        """Compute epoch averages and reset."""
        if not self.current_epoch:
            return {}

        # Average metrics over epoch
        avg_metrics = {}
        for key in self.current_epoch[0].keys():
            values = [m[key] for m in self.current_epoch if key in m]
            if values:
                avg_metrics[key] = np.mean(values)

        self.history.append(avg_metrics)
        self.current_epoch = []

        return avg_metrics

    def get_history(self, key: Optional[str] = None) -> List[float]:
        """Get history for a specific metric."""
        if key is None:
            return self.history
        return [h.get(key, 0) for h in self.history]


def compute_connectivity_metrics(
    pred_sc: torch.Tensor,
    target_sc: torch.Tensor,
    diagonal_value: float = 0.0
) -> Dict[str, float]:
    """
    Compute metrics for structural connectivity matrices.

    Args:
        pred_sc: Predicted SC matrix (N, N)
        target_sc: Target SC matrix (N, N)
        diagonal_value: Value to set on diagonal

    Returns:
        Dictionary of connectivity metrics
    """
    # Remove diagonal
    N = pred_sc.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool)

    pred_upper = pred_sc[mask].cpu().numpy()
    target_upper = target_sc[mask].cpu().numpy()

    # Set diagonal to zero
    pred_upper = pred_upper[pred_upper != diagonal_value]
    target_upper = target_upper[target_upper != diagonal_value]

    metrics = compute_metrics(
        torch.from_numpy(pred_upper),
        torch.from_numpy(target_upper)
    )

    # Add edge prediction metrics
    metrics['edge_mse'] = metrics.get('mse', 0)
    metrics['edge_mae'] = metrics.get('mae', 0)
    metrics['edge_pearson'] = metrics.get('pearson', 0)

    return metrics
