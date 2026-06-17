"""Loss functions for JEPA training."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.MAE.mae_loss import VICRegLoss


def soft_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Smooth-L1 latent prediction loss, optionally only on masked tokens."""
    loss = F.smooth_l1_loss(pred, target.detach(), reduction='none')
    if mask is None or mask.sum() == 0:
        return loss.mean()

    token_loss = loss.mean(dim=-1)
    return (token_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


class JEPALoss(nn.Module):
    """Soft-L1 latent prediction plus optional VICReg regularization."""

    def __init__(
        self,
        lambda_vicreg: float = 0.5,
        vicreg_var: float = 25.0,
        vicreg_cov: float = 1.0,
    ):
        super().__init__()
        self.lambda_vicreg = lambda_vicreg
        self.vicreg = VICRegLoss(lambda_var=vicreg_var, lambda_cov=vicreg_cov)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        jepa = soft_l1_loss(pred, target, mask)
        total = jepa
        metrics: Dict[str, float] = {'L_jepa': jepa.item()}

        if self.lambda_vicreg > 0 and context is not None:
            vicreg, vicreg_metrics = self.vicreg(context, target.detach())
            total = total + self.lambda_vicreg * vicreg
            metrics['L_vicreg'] = vicreg.item()
            metrics.update({f'vicreg_{k}': v for k, v in vicreg_metrics.items()})
        else:
            metrics['L_vicreg'] = 0.0

        metrics['loss_total'] = total.item()
        return total, metrics
