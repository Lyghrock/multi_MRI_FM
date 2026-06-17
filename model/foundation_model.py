"""
Brain Foundation Model

The main Foundation Model that combines encoders and fusion for multi-modal MRI.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoders.sMRI_encoder import sMRIEncoderWithPatch
from .encoders.fMRI_encoder import fMRIEncoder
from .encoders.dMRI_encoder import dMRIEncoder
from .fusion.latent_hub import SimplifiedLatentHub, CrossAttentionLatentHub
from utils.batch_utils import canonicalize_brain_batch, first_tensor


class BrainFoundationModel(nn.Module):
    """
    Multi-Modal Brain Foundation Model.

    Combines sMRI, fMRI, and dMRI encoders with a shared Latent Hub
    to produce a unified brain representation.

    Architecture:
        Input(sMRI, fMRI, dMRI)
            │
            ▼
        ┌─────────────────────────────┐
        │  Modality Encoders         │
        │  ├─ sMRI Encoder          │
        │  ├─ fMRI Encoder          │
        │  └─ dMRI Encoder          │
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Shared Latent Hub          │
        │  (Cross-Attention Fusion)   │
        └─────────────────────────────┘
            │
            ▼
        H ∈ R^(B×64×d)  # Unified Brain Representation
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 4,
        n_rois: int = 200,
        n_latent: int = 64,
        dropout: float = 0.1,
        fusion_type: str = 'simplified',
        positional_encoding: str = 'learnable',
        use_flash: bool = True,
        **kwargs
    ):
        """
        Initialize the Foundation Model.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ffn: Feed-forward dimension
            n_layers: Number of transformer layers per encoder
            n_rois: Number of ROIs (Schaefer-200)
            n_latent: Number of latent slots in the hub
            dropout: Dropout rate
            fusion_type: Type of fusion ('simplified', 'cross_attention')
        """
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.n_latent = n_latent

        # Modality-specific encoders
        self.sMRI_encoder = sMRIEncoderWithPatch(
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            n_layers=n_layers,
            n_rois=n_rois,
            n_features=3,  # GM_mean, GM_volume, GM_max
            dropout=dropout,
            use_positional_encoding=True,
            positional_encoding=positional_encoding,
            use_flash=use_flash
        )

        self.fMRI_encoder = fMRIEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            n_layers=n_layers,
            n_rois=n_rois,
            dropout=dropout,
            use_positional_encoding=True,
            positional_encoding=positional_encoding,
            use_flash=use_flash
        )

        self.dMRI_encoder = dMRIEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            n_layers=n_layers,
            n_rois=n_rois,
            dropout=dropout,
            use_positional_encoding=True,
            positional_encoding=positional_encoding,
            use_flash=use_flash
        )

        # Shared Latent Hub
        if fusion_type == 'simplified':
            self.latent_hub = SimplifiedLatentHub(
                d_model=d_model,
                n_latent=n_latent,
                n_modalities=3,
                dropout=dropout,
                use_flash=use_flash
            )
        elif fusion_type == 'cross_attention':
            self.latent_hub = CrossAttentionLatentHub(
                d_model=d_model,
                n_latent=n_latent,
                n_heads=n_heads,
                n_layers=2,
                dropout=dropout,
                use_flash=use_flash
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through the Foundation Model.

        Args:
            batch: Dictionary containing:
                - sMRI: (B, n_rois, 3) ROI features
                - fMRI: (B, T, n_rois) time-series or (B, n_rois, n_rois) FC matrix
                - dMRI: (B, n_rois, n_rois) SC matrix

        Returns:
            H: (B, n_latent, d_model) unified brain representation
        """
        batch = canonicalize_brain_batch(batch, n_rois=self.n_rois)
        if not batch:
            raise ValueError("No supported modality tensors found in batch")

        ref_tensor = first_tensor(batch)
        assert ref_tensor is not None

        B = ref_tensor.shape[0]
        dtype = ref_tensor.dtype if ref_tensor.is_floating_point() else torch.float32
        device = ref_tensor.device

        # Encode each modality. Patch tensors can now contribute even when
        # the canonical ROI/matrix tensor is missing from a partial dataset.
        if 'sMRI' in batch or 'sMRI_patches' in batch:
            smri = batch.get('sMRI')
            if smri is None:
                smri = torch.zeros(B, self.n_rois, 3, device=device, dtype=dtype)
            E_s = self.sMRI_encoder(smri, patches=batch.get('sMRI_patches'))  # (B, n_rois, d)
        else:
            E_s = torch.zeros(B, self.n_rois, self.d_model, device=device, dtype=dtype)

        if 'fMRI' in batch or 'fMRI_fc_patches' in batch:
            fmri = batch.get('fMRI')
            if fmri is None:
                fmri = torch.zeros(B, 1, self.n_rois, device=device, dtype=dtype)
            E_f = self.fMRI_encoder(
                fmri,
                fc_patches=batch.get('fMRI_fc_patches')
            )  # (B, n_rois, d)
        else:
            E_f = torch.zeros(B, self.n_rois, self.d_model, device=device, dtype=dtype)

        has_dmri_patches = any(
            key in batch for key in ('dMRI_FA_patches', 'dMRI_MD_patches', 'dMRI_SC_patches')
        )
        if 'dMRI' in batch or has_dmri_patches:
            dmri = batch.get('dMRI')
            if dmri is None:
                dmri = torch.zeros(B, self.n_rois, self.n_rois, device=device, dtype=dtype)
            E_d = self.dMRI_encoder(
                dmri,
                sc_patches=batch.get('dMRI_SC_patches'),
                fa_patches=batch.get('dMRI_FA_patches'),
                md_patches=batch.get('dMRI_MD_patches'),
            )  # (B, n_rois, d)
        else:
            E_d = torch.zeros(B, self.n_rois, self.d_model, device=device, dtype=dtype)

        # Fuse through Latent Hub
        H = self.latent_hub(E_s, E_f, E_d)  # (B, n_latent, d)

        # Output projection
        H = self.output_proj(H)

        return H

    def get_encoder(self, modality: str) -> nn.Module:
        """
        Get encoder for a specific modality.

        Args:
            modality: 'sMRI', 'fMRI', or 'dMRI'

        Returns:
            The corresponding encoder module
        """
        if modality == 'sMRI':
            return self.sMRI_encoder
        elif modality == 'fMRI':
            return self.fMRI_encoder
        elif modality == 'dMRI':
            return self.dMRI_encoder
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def encode_modality(
        self,
        modality: str,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a single modality.

        Args:
            modality: 'sMRI', 'fMRI', or 'dMRI'
            x: Input tensor

        Returns:
            Encoded representation
        """
        encoder = self.get_encoder(modality)
        return encoder(x)

    @classmethod
    def from_config(cls, config: dict) -> 'BrainFoundationModel':
        """
        Create model from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            BrainFoundationModel instance
        """
        model_config = config.get('model', {})
        atlas_config = config.get('atlas', {})

        return cls(
            d_model=model_config.get('d_model', 256),
            n_heads=model_config.get('n_heads', 8),
            d_ffn=model_config.get('d_ffn', 1024),
            n_layers=model_config.get('n_layers', 4),
            n_rois=atlas_config.get('n_rois', model_config.get('n_rois', 200)),
            n_latent=model_config.get('n_latent', 64),
            dropout=model_config.get('dropout', 0.1),
            fusion_type=model_config.get('fusion_type', 'simplified'),
            positional_encoding=model_config.get('positional_encoding', 'learnable'),
            use_flash=model_config.get('use_flash', True),
        )

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoders(self, modalities: list = None):
        """
        Freeze encoder weights.

        Args:
            modalities: List of modalities to freeze. If None, freeze all.
        """
        if modalities is None:
            modalities = ['sMRI', 'fMRI', 'dMRI']

        for modality in modalities:
            encoder = self.get_encoder(modality)
            for param in encoder.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def create_foundation_model(config: dict) -> BrainFoundationModel:
    """
    Factory function to create Foundation Model.

    Args:
        config: Configuration dictionary

    Returns:
        BrainFoundationModel instance
    """
    return BrainFoundationModel.from_config(config)
