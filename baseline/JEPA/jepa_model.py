"""Multi-modal JEPA model."""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from utils.batch_utils import canonicalize_brain_batch
from .jepa_loss import JEPALoss
from .masking import random_token_mask
from .predictor import create_predictor


class MultiModalJEPA(nn.Module):
    """
    JEPA training wrapper around the BrainFoundationModel.

    Online foundation receives gradients. Target foundation is an EMA copy.
    The predictor maps masked online latent tokens to target latent tokens.
    """

    def __init__(
        self,
        foundation_model: nn.Module,
        d_model: int = 256,
        predictor_type: str = 'mlp',
        predictor_hidden_dim: Optional[int] = None,
        predictor_n_heads: int = 4,
        predictor_n_layers: int = 2,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        ema_beta: float = 0.99,
        mask_ratio: float = 0.30,
        lambda_vicreg: float = 0.5,
        n_rois: int = 200,
    ):
        super().__init__()
        self.foundation = foundation_model
        self.target_foundation = copy.deepcopy(foundation_model)
        self.target_foundation.eval()
        for param in self.target_foundation.parameters():
            param.requires_grad = False

        self.predictor = create_predictor(
            predictor_type=predictor_type,
            d_model=d_model,
            hidden_dim=predictor_hidden_dim,
            n_heads=predictor_n_heads,
            n_layers=predictor_n_layers,
            n_experts=n_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.ema_beta = ema_beta
        self.mask_ratio = mask_ratio
        self.n_rois = n_rois
        self.loss_fn = JEPALoss(lambda_vicreg=lambda_vicreg)

    @torch.no_grad()
    def update_target_encoder(self, beta: Optional[float] = None):
        """EMA update target foundation from online foundation."""
        beta = self.ema_beta if beta is None else beta
        online_state = dict(self.foundation.named_parameters())
        for name, target_param in self.target_foundation.named_parameters():
            source_param = online_state[name]
            target_param.data.mul_(beta).add_(source_param.data, alpha=1.0 - beta)

        online_buffers = dict(self.foundation.named_buffers())
        for name, target_buffer in self.target_foundation.named_buffers():
            if name in online_buffers and target_buffer.is_floating_point():
                target_buffer.data.mul_(beta).add_(online_buffers[name].data, alpha=1.0 - beta)
            elif name in online_buffers:
                target_buffer.data.copy_(online_buffers[name].data)

    def _mask_context(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_token = self.mask_token.to(dtype=H.dtype, device=H.device)
        return torch.where(mask.unsqueeze(-1), mask_token.expand_as(H), H)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mask_ratio: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        inputs = canonicalize_brain_batch(batch, n_rois=self.n_rois)
        H = self.foundation(inputs)

        if mask is None:
            ratio = self.mask_ratio if mask_ratio is None else mask_ratio
            mask = random_token_mask((H.shape[0], H.shape[1]), ratio, device=H.device, min_masks=1)

        context = self._mask_context(H, mask)
        pred = self.predictor(context)

        with torch.no_grad():
            target = self.target_foundation(inputs)

        loss = None
        metrics = {}
        if return_loss:
            loss, metrics = self.loss_fn(pred, target, mask, context=H)

        return {
            'pred': pred,
            'target': target,
            'context': context,
            'H': H,
            'mask': mask,
            'loss': loss,
            'metrics': metrics,
        }

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = canonicalize_brain_batch(batch, n_rois=self.n_rois)
        return self.foundation(inputs)
