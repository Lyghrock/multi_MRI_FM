"""
sMRI Encoder Implementation

Encodes structural MRI (T1-weighted) data into latent representations.
Uses PyTorch native Flash Attention (SDPA) for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..backbone.transformer import TransformerEncoder, CrossModalAttention
from ..backbone.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
    get_positional_encoding,
)


class sMRIEncoder(nn.Module):
    """
    Encoder for sMRI ROI features.

    Encodes ROI-based features (GM volume, intensity, etc.) into latent space.
    Uses PyTorch native SDPA (Flash Attention) for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 4,
        n_rois: int = 200,
        n_features: int = 3,  # GM_mean, GM_volume, GM_max
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        positional_encoding: str = 'learnable',
        use_bias: bool = True,
        use_flash: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ffn: Feed-forward dimension
            n_layers: Number of transformer layers
            n_rois: Number of ROIs (default: 200 for Schaefer atlas)
            n_features: Number of features per ROI
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding
            use_bias: Whether to use bias in linear layers
            use_flash: Use PyTorch native Flash Attention (SDPA)
        """
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.n_features = n_features
        self.use_flash = use_flash

        # Input projection: n_features -> d_model
        self.input_proj = nn.Linear(n_features, d_model, bias=use_bias)

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = get_positional_encoding(
                positional_encoding,
                d_model=d_model,
                max_len=n_rois,
                n_rois=n_rois,
                dropout=dropout
            )
        else:
            self.pos_encoder = None

        # Transformer encoder (with PyTorch native SDPA)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            use_flash=use_flash
        )

        # Output projection: d_model -> d_model
        self.output_proj = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_rois, n_features) or (B, n_rois) with single feature
            attention_mask: (B, n_rois), True for valid, False for masked

        Returns:
            (B, n_rois, d_model)
        """
        B, N, F = x.shape

        # Handle single feature input
        if F == 1:
            x = x.squeeze(-1)

        # Project input features
        x = self.input_proj(x)  # (B, n_rois, d_model)

        # Add positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        # Transformer encoding (with SDPA)
        x = self.transformer(x, attention_mask)

        # Output projection
        x = self.output_proj(x)

        return x


class sMRIEncoderWithPatch(nn.Module):
    """
    sMRI Encoder with 3D patch support.

    For encoding both ROI features and 3D volumetric data.
    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 4,
        n_rois: int = 200,
        n_features: int = 3,
        patch_size: int = 16,
        use_positional_encoding: bool = True,
        positional_encoding: str = 'learnable',
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.patch_size = patch_size
        self.use_flash = use_flash

        # ROI encoder
        self.roi_encoder = sMRIEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            n_layers=n_layers,
            n_rois=n_rois,
            n_features=n_features,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            positional_encoding=positional_encoding,
            use_flash=use_flash
        )

        # Lightweight 3D patch encoder. Each voxel patch is summarized by
        # mean/std/max before projection to avoid expensive Conv3d over many
        # high-resolution patches.
        self.patch_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)

    def _encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode voxel patches or a full volume into one subject-level vector."""
        if patches.dim() == 4:
            # Backward-compatible full-volume input: (B, D, H, W)
            flat = patches.flatten(1)
            stats = torch.stack(
                [flat.mean(dim=1), flat.std(dim=1, unbiased=False), flat.amax(dim=1)],
                dim=-1,
            )
            return self.patch_encoder(stats)

        if patches.dim() != 5:
            raise ValueError(f"Expected sMRI patches as (B,K,D,H,W), got {tuple(patches.shape)}")

        B, K = patches.shape[:2]
        flat = patches.reshape(B, K, -1)
        stats = torch.stack(
            [flat.mean(dim=-1), flat.std(dim=-1, unbiased=False), flat.amax(dim=-1)],
            dim=-1,
        )
        return self.patch_encoder(stats).mean(dim=1)

    def forward(
        self,
        roi_features: torch.Tensor,
        volume: Optional[torch.Tensor] = None,
        patches: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            roi_features: (B, n_rois, n_features)
            volume: (B, D, H, W) optional 3D volume
            patches: (B, K, D, H, W) optional voxel patches

        Returns:
            (B, n_rois, d_model)
        """
        # Encode ROI features
        roi_out = self.roi_encoder(roi_features)  # (B, n_rois, d_model)

        patch_source = patches if patches is not None else volume
        if patch_source is not None:
            patch_out = self._encode_patches(patch_source)
            patch_out = patch_out.unsqueeze(1).expand(-1, roi_out.shape[1], -1)

            # Fuse
            fused = torch.cat([roi_out, patch_out], dim=-1)
            out = self.fusion(fused)
        else:
            out = roi_out

        return out


class PerceiverEncoder(nn.Module):
    """
    Perceiver-style encoder for sMRI.

    Uses cross-attention between small latent array and input.
    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_latent: int = 64,
        n_heads: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 4,
        n_rois: int = 200,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latent = n_latent
        self.use_flash = use_flash

        # Input projection
        self.input_proj = nn.Linear(3, d_model)  # 3 features per ROI

        # Latent array
        self.latent = nn.Parameter(torch.randn(1, n_latent, d_model) * 0.02)

        # Cross-attention layers (using PyTorch native)
        self.cross_attns = nn.ModuleList([
            CrossModalAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_flash=use_flash
            )
            for _ in range(n_layers)
        ])

        # Self-attention on latent (using PyTorch native)
        self.self_attns = nn.ModuleList([
            TransformerEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=1,
                dropout=dropout,
                use_flash=use_flash
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_rois, 3)

        Returns:
            (B, n_latent, d_model)
        """
        B = x.shape[0]

        # Project input
        x = self.input_proj(x)  # (B, n_rois, d_model)

        # Initialize latent
        latent = self.latent.expand(B, -1, -1)  # (B, n_latent, d_model)

        # Cross-attention and self-attention layers
        for cross_attn, self_attn in zip(self.cross_attns, self.self_attns):
            # Cross-attention: latent attends to input
            latent = cross_attn(latent, x)

            # Self-attention on latent
            latent = self_attn(latent)

        return self.norm(latent)
