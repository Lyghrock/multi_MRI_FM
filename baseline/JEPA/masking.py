"""Masking utilities for JEPA latent prediction."""

from dataclasses import dataclass
from typing import Dict

import torch


def random_token_mask(
    shape: tuple[int, int],
    mask_ratio: float,
    device=None,
    min_masks: int = 1,
) -> torch.Tensor:
    """Create a boolean token mask of shape (B, N). True means masked."""
    B, N = shape
    mask_ratio = float(max(0.0, min(mask_ratio, 1.0)))
    if mask_ratio <= 0:
        return torch.zeros(B, N, dtype=torch.bool, device=device)

    n_mask = max(min_masks, int(round(N * mask_ratio)))
    n_mask = min(n_mask, N)
    noise = torch.rand(B, N, device=device)
    ids = noise.argsort(dim=1)[:, :n_mask]
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask.scatter_(1, ids, True)
    return mask


@dataclass
class JEPAMaskConfig:
    total_epochs: int = 500
    mask_type: str = 'intra'
    intra_mask_ratio: float = 0.3
    inter_mask_ratio: float = 0.2
    mask_warmup: float = 0.0
    mask_alignment: float = 0.15
    mask_refinement: float = 0.30


class JEPACurriculumMasker:
    """Epoch-aware latent token masking schedule."""

    def __init__(self, config: JEPAMaskConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: Dict) -> 'JEPACurriculumMasker':
        train_cfg = config.get('train', {})
        jepa_cfg = config.get('jepa', {})
        return cls(JEPAMaskConfig(
            total_epochs=train_cfg.get('epochs', 500),
            mask_type=jepa_cfg.get('mask_type', 'intra'),
            intra_mask_ratio=jepa_cfg.get('intra_mask_ratio', jepa_cfg.get('mask_refinement', 0.3)),
            inter_mask_ratio=jepa_cfg.get('inter_mask_ratio', 0.2),
            mask_warmup=jepa_cfg.get('mask_warmup', 0.0),
            mask_alignment=jepa_cfg.get('mask_alignment', 0.15),
            mask_refinement=jepa_cfg.get('mask_refinement', 0.30),
        ))

    def get_ratio(self, epoch: int) -> float:
        total = max(self.config.total_epochs, 1)
        warmup_end = int(total * 0.2)
        align_end = int(total * 0.6)

        if epoch < warmup_end:
            return self.config.mask_warmup
        if epoch < align_end:
            t = (epoch - warmup_end) / max(align_end - warmup_end, 1)
            return self.config.mask_warmup * (1 - t) + self.config.mask_alignment * t
        return self.config.mask_refinement

    def mask(self, x: torch.Tensor, epoch: int) -> torch.Tensor:
        return random_token_mask(
            (x.shape[0], x.shape[1]),
            self.get_ratio(epoch),
            device=x.device,
            min_masks=1,
        )
