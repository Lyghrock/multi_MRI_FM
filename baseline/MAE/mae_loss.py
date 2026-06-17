"""
Multi-Modal MAE Loss Functions

Includes reconstruction loss and VICReg regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class VICRegLoss(nn.Module):
    """
    VICReg: Variance, Invariance, Covariance Regularization.

    Prevents representation collapse and encourages dimension decorrelation.

    Reference: https://arxiv.org/abs/2105.04906
    """

    def __init__(
        self,
        lambda_inv: float = 1.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4
    ):
        """
        Args:
            lambda_inv: Weight for invariance loss
            lambda_var: Weight for variance loss
            lambda_cov: Weight for covariance loss
            gamma: Target variance threshold
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VICReg loss.

        Args:
            z1: (B, N, d) or (B, d) representation 1
            z2: (B, N, d) or (B, d) representation 2

        Returns:
            total_loss, metrics_dict
        """
        metrics = {}

        # Invariance loss (MSE)
        if self.lambda_inv > 0:
            L_inv = F.mse_loss(z1, z2)
            metrics['L_inv'] = L_inv.item()
        else:
            L_inv = torch.tensor(0.0, device=z1.device)
            metrics['L_inv'] = 0.0

        # Variance loss (prevents collapse)
        if self.lambda_var > 0:
            L_var = self._variance_loss(z1) + self._variance_loss(z2)
            metrics['L_var'] = L_var.item()
        else:
            L_var = torch.tensor(0.0, device=z1.device)
            metrics['L_var'] = 0.0

        # Covariance loss (decorrelation)
        if self.lambda_cov > 0:
            L_cov = self._covariance_loss(z1) + self._covariance_loss(z2)
            metrics['L_cov'] = L_cov.item()
        else:
            L_cov = torch.tensor(0.0, device=z1.device)
            metrics['L_cov'] = 0.0

        # Total loss
        L_total = (
            self.lambda_inv * L_inv +
            self.lambda_var * L_var +
            self.lambda_cov * L_cov
        )

        metrics['L_vicreg'] = L_total.item()

        return L_total, metrics

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute variance regularization.

        Encourages each dimension to have variance around gamma.
        """
        # Flatten if needed
        if z.dim() > 2:
            z = z.reshape(-1, z.shape[-1])

        # Standard deviation per dimension
        std = torch.sqrt(z.var(dim=0) + self.eps)

        # Pull towards gamma
        L_var = F.relu(self.gamma - std).mean()

        return L_var

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance regularization.

        Encourages dimensions to be uncorrelated.
        """
        # Flatten if needed
        if z.dim() > 2:
            z = z.reshape(-1, z.shape[-1])

        B = z.shape[0]
        denom = max(B - 1, 1)

        # Center the features
        z = z - z.mean(dim=0)

        # Covariance matrix
        cov = (z.T @ z) / denom

        # Off-diagonal elements should be zero
        off_diag = cov.flatten()[1:]  # Skip diagonal
        # Create mask for off-diagonal
        mask = torch.ones_like(off_diag)
        # Zero out the elements we want to keep (diagonal)
        diag_mask = torch.zeros_like(off_diag)

        # Simple approach: just minimize sum of squared off-diagonal
        L_cov = (cov ** 2).sum() - (cov.diag() ** 2).sum()

        return L_cov


class MultiModalMAELoss(nn.Module):
    """
    Multi-Modal MAE Loss.

    Combines reconstruction loss with optional VICReg regularization.
    """

    def __init__(
        self,
        lambda_sMRI: float = 1.0,
        lambda_fMRI: float = 1.0,
        lambda_dMRI: float = 1.0,
        use_vicreg: bool = True,
        lambda_vicreg: float = 0.5,
        loss_type: str = 'mse'
    ):
        """
        Args:
            lambda_sMRI: Weight for sMRI reconstruction
            lambda_fMRI: Weight for fMRI reconstruction
            lambda_dMRI: Weight for dMRI reconstruction
            use_vicreg: Whether to use VICReg regularization
            lambda_vicreg: Weight for VICReg loss
            loss_type: 'mse' or 'smooth_l1'
        """
        super().__init__()

        self.lambda_sMRI = lambda_sMRI
        self.lambda_fMRI = lambda_fMRI
        self.lambda_dMRI = lambda_dMRI
        self.use_vicreg = use_vicreg
        self.lambda_vicreg = lambda_vicreg
        self.loss_type = loss_type

        if use_vicreg:
            self.vicreg = VICRegLoss(lambda_var=25.0, lambda_cov=1.0, gamma=1.0)
        else:
            self.vicreg = None

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        H: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MAE loss.

        Args:
            predictions: Dictionary of predictions per modality
            targets: Dictionary of target tensors per modality
            mask: (B, N) MAE mask tensor (1 = masked)
            H: (B, N, d) latent representation for VICReg

        Returns:
            total_loss, metrics_dict
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        metrics = {}

        # sMRI Reconstruction Loss
        if 'sMRI' in predictions and 'sMRI' in targets:
            pred = predictions['sMRI']
            target = targets['sMRI']
            if pred.shape != target.shape:
                target = target.reshape_as(pred)

            if self.loss_type == 'mse':
                L_sMRI = F.mse_loss(pred, target)
            else:
                L_sMRI = F.smooth_l1_loss(pred, target)

            total_loss = total_loss + self.lambda_sMRI * L_sMRI
            metrics['L_sMRI'] = L_sMRI.item()

        # fMRI Reconstruction Loss
        if 'fMRI' in predictions and 'fMRI' in targets:
            pred = predictions['fMRI']
            target = targets['fMRI']
            if target.dim() == 3 and target.shape[-1] == target.shape[-2]:
                from utils.batch_utils import upper_triangular_from_matrix
                target = upper_triangular_from_matrix(target, target.shape[-1])
            if pred.shape != target.shape:
                target = target.reshape_as(pred)

            if self.loss_type == 'mse':
                L_fMRI = F.mse_loss(pred, target)
            else:
                L_fMRI = F.smooth_l1_loss(pred, target)

            total_loss = total_loss + self.lambda_fMRI * L_fMRI
            metrics['L_fMRI'] = L_fMRI.item()

        # dMRI Reconstruction Loss
        if 'dMRI' in predictions and 'dMRI' in targets:
            pred = predictions['dMRI']
            target = targets['dMRI']
            if target.dim() == 3 and target.shape[-1] == target.shape[-2]:
                from utils.batch_utils import upper_triangular_from_matrix
                target = upper_triangular_from_matrix(target, target.shape[-1])
            if pred.shape != target.shape:
                target = target.reshape_as(pred)

            if self.loss_type == 'mse':
                L_dMRI = F.mse_loss(pred, target)
            else:
                L_dMRI = F.smooth_l1_loss(pred, target)

            total_loss = total_loss + self.lambda_dMRI * L_dMRI
            metrics['L_dMRI'] = L_dMRI.item()

        # VICReg Regularization
        if self.use_vicreg and self.vicreg is not None and H is not None:
            # Create a slightly augmented version
            H_aug = H + torch.randn_like(H) * 0.01

            L_vicreg, vicreg_metrics = self.vicreg(H, H_aug)
            total_loss = total_loss + self.lambda_vicreg * L_vicreg

            metrics['L_vicreg'] = L_vicreg.item()
            metrics.update({f'vicreg_{k}': v for k, v in vicreg_metrics.items()})

        metrics['loss_total'] = total_loss.item()

        return total_loss, metrics


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = 'mse'
) -> torch.Tensor:
    """
    Compute reconstruction loss with optional masking.

    Args:
        pred: Prediction tensor
        target: Target tensor
        mask: Optional mask (1 = compute loss, 0 = ignore)
        loss_type: 'mse' or 'smooth_l1'

    Returns:
        Loss tensor
    """
    if mask is not None and mask.sum() > 0:
        # Apply mask
        if pred.dim() > target.dim():
            mask = mask.unsqueeze(-1)
        loss = F.mse_loss(pred, target, reduction='none')
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    else:
        if loss_type == 'mse':
            loss = F.mse_loss(pred, target)
        else:
            loss = F.smooth_l1_loss(pred, target)

    return loss
