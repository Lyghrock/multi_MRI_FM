"""
Cross-Attention Fusion Module

Uses PyTorch native CrossAttention for efficiency:
- torch.nn.functional.scaled_dot_product_attention (SDPA)
- Flash Attention support
- Memory-efficient backprop
"""

import torch
import torch.nn as nn
from typing import Optional

from ..backbone.transformer import CrossAttention, TransformerEncoder, CrossModalAttention


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion using PyTorch native cross-attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Cross-attention layers (using PyTorch native)
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_flash=use_flash
            )
            for _ in range(n_layers)
        ])

        # Self-attention for refinement (using native encoder)
        self.self_attn = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=1,
            dropout=dropout,
            use_flash=use_flash
        )

        # Norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse query with context using cross-attention.

        Args:
            query: (B, N_q, d) query tensor
            context: (B, N_c, d) context tensor
            mask: (B, N_c) optional mask for context

        Returns:
            (B, N_q, d)
        """
        x = query

        for cross_attn in self.cross_attns:
            x = cross_attn(x, context, mask)

        x = self.self_attn(x)
        x = self.norm(x)

        return x


class BiDirectionalFusion(nn.Module):
    """
    Bidirectional cross-attention fusion using PyTorch native attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model

        # Forward cross-attention (a attends to b)
        self.cross_attn_ab = CrossModalFusion(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_flash=use_flash
        )

        # Backward cross-attention (b attends to a)
        self.cross_attn_ba = CrossModalFusion(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_flash=use_flash
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """
        Bidirectional fusion of two modalities.

        Args:
            a: (B, N_a, d)
            b: (B, N_b, d)

        Returns:
            (B, N, d) fused representation
        """
        # a attends to b
        fused_a = self.cross_attn_ab(a, b)

        # b attends to a
        fused_b = self.cross_attn_ba(b, a)

        # Fuse both directions: concatenate and project
        fused = torch.cat([fused_a, fused_b], dim=-1)  # (B, N, 2d)
        out = self.fusion(fused)
        out = self.norm(out)

        return out


class TriModalFusion(nn.Module):
    """
    Fusion for three modalities using PyTorch native attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        # Pairwise fusion
        self.fusion_smri_fmri = BiDirectionalFusion(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_flash=use_flash
        )

        self.fusion_result_dmri = CrossModalFusion(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_flash=use_flash
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        e_sMRI: torch.Tensor,
        e_fMRI: torch.Tensor,
        e_dMRI: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse three modalities.

        Args:
            e_sMRI: (B, N, d)
            e_fMRI: (B, N, d)
            e_dMRI: (B, N, d)

        Returns:
            (B, N, d)
        """
        # Fuse sMRI and fMRI
        fused_smri_fmri = self.fusion_smri_fmri(e_sMRI, e_fMRI)

        # Fuse result with dMRI
        fused_all = self.fusion_result_dmri(fused_smri_fmri, e_dMRI)

        # Project
        out = self.final_fusion(fused_all)
        out = self.norm(out)

        return out


class SimplifiedFusion(nn.Module):
    """
    Simplified fusion with single cross-attention layer.
    Good for quick experiments.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        # Single cross-attention for each pair
        self.cross_s_f = CrossAttention(d_model, n_heads, dropout, use_flash)
        self.cross_s_d = CrossAttention(d_model, n_heads, dropout, use_flash)
        self.cross_f_d = CrossAttention(d_model, n_heads, dropout, use_flash)

        # Aggregation
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 3),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        e_sMRI: torch.Tensor,
        e_fMRI: torch.Tensor,
        e_dMRI: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            e_sMRI: (B, N, d)
            e_fMRI: (B, N, d)
            e_dMRI: (B, N, d)

        Returns:
            (B, N, d)
        """
        # Cross-attention in each direction
        f_from_s = self.cross_s_f(e_fMRI, e_sMRI)  # fMRI queries sMRI
        d_from_s = self.cross_s_d(e_dMRI, e_sMRI)   # dMRI queries sMRI
        d_from_f = self.cross_f_d(e_dMRI, e_fMRI)   # dMRI queries fMRI

        # Stack and gate
        combined = torch.cat([f_from_s, d_from_s, d_from_f], dim=-1)  # (B, N, 3d)
        gate = self.gate(combined)
        out = (f_from_s * gate[:, :, :1] + d_from_s * gate[:, :, 1:2] + d_from_f * gate[:, :, 2:3])

        return self.norm(out)
