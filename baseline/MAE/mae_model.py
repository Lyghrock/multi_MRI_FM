"""
Multi-Modal MAE Model

Masked Autoencoder for brain MRI with modality-specific decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .decoders.decoder import SMRI_Decoder, FMRI_Decoder, DMRI_Decoder
from .mae_loss import MultiModalMAELoss
from utils.batch_utils import canonicalize_brain_batch, make_mae_targets


class MAEMasking:
    """
    MAE Masking utility for latent representation.
    """

    def __init__(self, mask_ratio: float = 0.75):
        """
        Args:
            mask_ratio: Ratio of tokens to mask (default: 0.75)
        """
        self.mask_ratio = mask_ratio

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random masking to input.

        Args:
            x: (B, N, d) input tensor

        Returns:
            x_masked: (B, N, d) masked input
            mask: (B, N) binary mask (1 = masked)
            ids_restore: (B, N) indices to restore original order
        """
        B, N, d = x.shape

        # Number of visible tokens
        n_visible = int(N * (1 - self.mask_ratio))

        # Generate random noise
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get ids_shuffle
        ids_shuffle = torch.argsort(noise, dim=1)

        # Sort to get ids_restore (original indices)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Get indices of visible tokens
        ids_keep = ids_shuffle[:, :n_visible]

        # Create masked input
        B_idx = torch.arange(B, device=x.device).unsqueeze(1)
        x_masked = x[B_idx, ids_keep]

        # Create mask (1 for masked, 0 for visible)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_visible] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Pad masked positions
        x_masked_full = torch.zeros(B, N, d, device=x.device)
        x_masked_full[B_idx, ids_keep] = x_masked

        return x_masked_full, mask, ids_restore

    def unmask(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Restore masked positions to original order.

        Args:
            x: (B, N, d) tensor with masked positions filled
            ids_restore: (B, N) indices to restore order

        Returns:
            (B, N, d) restored tensor
        """
        B, N, d = x.shape
        B_idx = torch.arange(B, device=x.device).unsqueeze(1)
        return x[B_idx, ids_restore]


class MultiModalMAE(nn.Module):
    """
    Multi-Modal Masked Autoencoder.

    Architecture:
        Input (sMRI, fMRI, dMRI)
            │
            ▼
        Foundation Model (Encoder + Latent Hub)
            │
            ▼
        MAE Masking (75% tokens masked)
            │
            ▼
        Modality-Specific Decoders
            │
            ▼
        Reconstructed Outputs
    """

    def __init__(
        self,
        foundation_model: nn.Module,
        d_model: int = 256,
        hidden_dim: int = 256,
        n_rois: int = 200,
        n_features: int = 3,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
        decoder_hidden_dim: Optional[int] = None,
        include_sMRI: bool = True,
        include_fMRI: bool = True,
        include_dMRI: bool = True,
        lambda_sMRI: float = 1.0,
        lambda_fMRI: float = 1.0,
        lambda_dMRI: float = 1.0,
        lambda_vicreg: float = 0.5,
    ):
        """
        Args:
            foundation_model: BrainFoundationModel for encoding
            d_model: Latent dimension
            hidden_dim: Decoder hidden dimension
            n_rois: Number of ROIs
            n_features: Features per ROI for sMRI
            mask_ratio: MAE masking ratio
            dropout: Dropout rate
            decoder_hidden_dim: Override decoder hidden dim
            include_sMRI/fMRI/dMRI: Which modalities to include
        """
        super().__init__()

        self.foundation = foundation_model
        self.mask_ratio = mask_ratio
        self.n_rois = n_rois
        self.n_features = n_features
        self.loss_fn = MultiModalMAELoss(
            lambda_sMRI=lambda_sMRI,
            lambda_fMRI=lambda_fMRI,
            lambda_dMRI=lambda_dMRI,
            lambda_vicreg=lambda_vicreg,
        )

        # Masking utility
        self.masking = MAEMasking(mask_ratio)

        # Determine decoder hidden dim
        if decoder_hidden_dim is None:
            decoder_hidden_dim = hidden_dim

        # Modality-specific decoders
        self.include_sMRI = include_sMRI
        self.include_fMRI = include_fMRI
        self.include_dMRI = include_dMRI

        if include_sMRI:
            self.sMRI_decoder = SMRI_Decoder(
                d_model=d_model,
                hidden_dim=decoder_hidden_dim,
                n_rois=n_rois,
                n_features=n_features,
                dropout=dropout
            )

        if include_fMRI:
            self.fMRI_decoder = FMRI_Decoder(
                d_model=d_model,
                hidden_dim=decoder_hidden_dim,
                n_rois=n_rois,
                dropout=dropout
            )

        if include_dMRI:
            self.dMRI_decoder = DMRI_Decoder(
                d_model=d_model,
                hidden_dim=decoder_hidden_dim * 2,  # Larger for SC
                n_rois=n_rois,
                dropout=dropout
            )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        apply_mask: bool = True,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MAE.

        Args:
            batch: Dictionary containing modality data
            apply_mask: Whether to apply MAE masking
            return_loss: Whether to compute reconstruction loss

        Returns:
            Dictionary containing:
                - predictions: decoded outputs
                - loss: total loss (if return_loss=True)
                - metrics: loss components
                - mask: masking tensor
        """
        # 1. Normalize nested dataset batches and encode
        model_inputs = canonicalize_brain_batch(batch, n_rois=self.n_rois)
        H = self.foundation(model_inputs)  # (B, n_latent, d_model)

        # 2. MAE masking (if enabled)
        if apply_mask and self.training:
            H_masked, mask, ids_restore = self.masking.mask(H)
        else:
            H_masked = H
            mask = torch.zeros(H.shape[0], H.shape[1], device=H.device)
            ids_restore = None

        # 3. Decode each modality
        predictions = {}

        if self.include_sMRI and 'sMRI' in model_inputs:
            pred_sMRI = self.sMRI_decoder(H_masked)
            predictions['sMRI'] = pred_sMRI

        if self.include_fMRI and 'fMRI' in model_inputs:
            pred_fMRI = self.fMRI_decoder(H_masked)
            predictions['fMRI'] = pred_fMRI

        if self.include_dMRI and 'dMRI' in model_inputs:
            pred_dMRI = self.dMRI_decoder(H_masked)
            predictions['dMRI'] = pred_dMRI

        # 4. Compute loss
        if return_loss:
            targets = make_mae_targets(batch, n_rois=self.n_rois, device=H.device, dtype=H.dtype)
            loss, metrics = self.compute_loss(predictions, targets, mask, H)
        else:
            loss = None
            metrics = {}

        return {
            'predictions': predictions,
            'loss': loss,
            'metrics': metrics,
            'mask': mask,
            'H': H,
            'ids_restore': ids_restore
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        H: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MAE reconstruction loss.

        Args:
            predictions: Dictionary of predictions per modality
            batch: Original input data
            mask: MAE mask tensor

        Returns:
            total_loss, metrics_dict
        """
        return self.loss_fn(predictions, batch, mask, H)

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode input to latent representation (no decoding).

        Args:
            batch: Input data

        Returns:
            H: (B, n_latent, d_model) latent representation
        """
        return self.foundation(batch)

    def decode(
        self,
        H: torch.Tensor,
        modalities: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent to modality outputs (no masking).

        Args:
            H: (B, n_latent, d_model) latent representation
            modalities: List of modalities to decode

        Returns:
            Dictionary of decoded outputs
        """
        predictions = {}

        if modalities is None:
            modalities = []
            if self.include_sMRI:
                modalities.append('sMRI')
            if self.include_fMRI:
                modalities.append('fMRI')
            if self.include_dMRI:
                modalities.append('dMRI')

        if 'sMRI' in modalities and hasattr(self, 'sMRI_decoder'):
            predictions['sMRI'] = self.sMRI_decoder(H)

        if 'fMRI' in modalities and hasattr(self, 'fMRI_decoder'):
            predictions['fMRI'] = self.fMRI_decoder(H)

        if 'dMRI' in modalities and hasattr(self, 'dMRI_decoder'):
            predictions['dMRI'] = self.dMRI_decoder(H)

        return predictions

    def get_parameters(self) -> dict:
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.foundation.parameters())
        decoder = total - encoder
        return {
            'total': total,
            'encoder': encoder,
            'decoder': decoder
        }


class MAEWrapper(nn.Module):
    """
    Wrapper for MAE with optional freeze encoder during fine-tuning.
    """

    def __init__(self, mae_model: MultiModalMAE, freeze_encoder: bool = True):
        super().__init__()
        self.mae = mae_model
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            self.freeze_foundation()

    def freeze_foundation(self):
        """Freeze the foundation model encoder."""
        for param in self.mae.foundation.parameters():
            param.requires_grad = False

    def unfreeze_foundation(self):
        """Unfreeze the foundation model encoder."""
        for param in self.mae.foundation.parameters():
            param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.mae(*args, **kwargs)
