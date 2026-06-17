"""
Latent Hub - Multi-modal Fusion Module

Uses PyTorch native CrossAttention for efficiency:
- torch.nn.functional.scaled_dot_product_attention (SDPA)
- Flash Attention support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..backbone.transformer import TransformerEncoder, CrossAttention, CrossModalAttention


class LatentHub(nn.Module):
    """
    Base Latent Hub class.
    """

    def forward(self, *representations) -> torch.Tensor:
        raise NotImplementedError


class SimplifiedLatentHub(nn.Module):
    """
    Simplified Latent Hub for multi-modal fusion.

    Concatenates all modality representations and projects to common space.
    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_latent: int = 64,
        n_modalities: int = 3,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            n_latent: Number of latent slots
            n_modalities: Number of modalities (default: 3 for sMRI, fMRI, dMRI)
            dropout: Dropout rate
            use_flash: Use PyTorch native Flash Attention (SDPA)
        """
        super().__init__()

        self.d_model = d_model
        self.n_latent = n_latent
        self.n_modalities = n_modalities
        self.use_flash = use_flash

        # Concatenation projection
        total_dim = d_model * n_modalities
        self.concat_proj = nn.Sequential(
            nn.Linear(total_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # Latent queries (learnable)
        self.latent_queries = nn.Parameter(
            torch.randn(1, n_latent, d_model) * 0.02
        )

        # Cross-attention to latent slots (using PyTorch native)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            n_heads=8,
            dropout=dropout,
            use_flash=use_flash
        )

        # Output norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        e_sMRI: torch.Tensor,
        e_fMRI: torch.Tensor,
        e_dMRI: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-modal representations.

        Args:
            e_sMRI: (B, N_s, d) sMRI representation
            e_fMRI: (B, N_f, d) fMRI representation
            e_dMRI: (B, N_d, d) dMRI representation

        Returns:
            (B, n_latent, d) fused representation
        """
        B = e_sMRI.shape[0]

        # Average pooling each modality to fixed size
        def pool_to_n(x, n):
            if x.shape[1] == n:
                return x
            elif x.shape[1] > n:
                x = x.transpose(1, 2)  # (B, d, N)
                x = F.adaptive_avg_pool1d(x, n)
                x = x.transpose(1, 2)  # (B, n, d)
                return x
            else:
                return F.interpolate(
                    x.transpose(1, 2),
                    size=n,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

        # Pool to same size
        s_pooled = pool_to_n(e_sMRI, self.n_latent)
        f_pooled = pool_to_n(e_fMRI, self.n_latent)
        d_pooled = pool_to_n(e_dMRI, self.n_latent)

        # Concatenate along feature dimension
        concat = torch.cat([s_pooled, f_pooled, d_pooled], dim=-1)  # (B, n_latent, 3d)

        # Project to common space
        projected = self.concat_proj(concat)  # (B, n_latent, d)

        # Initialize latent queries
        latent = self.latent_queries.expand(B, -1, -1)  # (B, n_latent, d)

        # Cross-attention between latent and projected (with SDPA)
        fused = self.cross_attn(latent, projected)  # (B, n_latent, d)

        # Norm
        fused = self.norm(fused)

        return fused


class CrossAttentionLatentHub(nn.Module):
    """
    Cross-Attention based Latent Hub.

    Uses iterative cross-attention to fuse modalities.
    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_latent: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latent = n_latent
        self.use_flash = use_flash

        # Learnable latent slots
        self.latent = nn.Parameter(torch.randn(1, n_latent, d_model) * 0.02)

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

        # Self-attention on latent (using PyTorch native)
        self.self_attn = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=1,
            dropout=dropout,
            use_flash=use_flash
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
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
            (B, n_latent, d)
        """
        B = e_sMRI.shape[0]

        # Pool each modality to fixed size
        def pool_to_n(x, n):
            if x.shape[1] == n:
                return x
            x = x.transpose(1, 2)
            x = F.adaptive_avg_pool1d(x, n)
            return x.transpose(1, 2)

        s = pool_to_n(e_sMRI, self.n_latent)
        f = pool_to_n(e_fMRI, self.n_latent)
        d = pool_to_n(e_dMRI, self.n_latent)

        # Initialize latent
        latent = self.latent.expand(B, -1, -1)

        # Iterative cross-attention (with SDPA)
        for cross_attn in self.cross_attns:
            # Cross-attend to each modality
            latent = cross_attn(latent, s)
            latent = cross_attn(latent, f)
            latent = cross_attn(latent, d)

        # Self-attention on latent (with SDPA)
        latent = self.self_attn(latent)

        # Output
        out = self.norm(self.output_proj(latent))

        return out


class GatingLatentHub(nn.Module):
    """
    Gating-based Latent Hub with learned modality weighting.

    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_latent: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latent = n_latent

        # Gating networks for each modality
        self.gate_sMRI = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )
        self.gate_fMRI = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )
        self.gate_dMRI = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
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
            (B, N, d) or (B, n_latent, d)
        """
        # Pool to common size
        s = F.adaptive_avg_pool1d(e_sMRI.transpose(1, 2), self.n_latent).transpose(1, 2)
        f = F.adaptive_avg_pool1d(e_fMRI.transpose(1, 2), self.n_latent).transpose(1, 2)
        d = F.adaptive_avg_pool1d(e_dMRI.transpose(1, 2), self.n_latent).transpose(1, 2)

        # Concatenate for gating
        concat = torch.cat([s, f, d], dim=-1)  # (B, n, 3d)

        # Compute gates
        gate_s = self.gate_sMRI(concat)  # (B, n, d)
        gate_f = self.gate_fMRI(concat)
        gate_d = self.gate_dMRI(concat)

        # Gated combination
        gated = gate_s * s + gate_f * f + gate_d * d

        # Output
        out = self.norm(self.output_proj(gated))

        return out


class AttentionWeightedHub(nn.Module):
    """
    Attention-weighted fusion with learnable weights.

    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_latent: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latent = n_latent
        self.use_flash = use_flash

        # Query for attention
        self.query = nn.Parameter(torch.randn(1, n_latent, d_model) * 0.02)

        # Key/Value projections for each modality
        self.key_proj = nn.ModuleDict({
            'sMRI': nn.Linear(d_model, d_model),
            'fMRI': nn.Linear(d_model, d_model),
            'dMRI': nn.Linear(d_model, d_model),
        })
        self.value_proj = nn.ModuleDict({
            'sMRI': nn.Linear(d_model, d_model),
            'fMRI': nn.Linear(d_model, d_model),
            'dMRI': nn.Linear(d_model, d_model),
        })

        # Output
        self.output_proj = nn.Linear(d_model, d_model)
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
            (B, n_latent, d)
        """
        B = e_sMRI.shape[0]

        # Pool each modality
        s = F.adaptive_avg_pool1d(e_sMRI.transpose(1, 2), self.n_latent).transpose(1, 2)
        f = F.adaptive_avg_pool1d(e_fMRI.transpose(1, 2), self.n_latent).transpose(1, 2)
        d = F.adaptive_avg_pool1d(e_dMRI.transpose(1, 2), self.n_latent).transpose(1, 2)

        # Query
        q = self.query.expand(B, -1, -1)

        # Keys and values
        k_s = self.key_proj['sMRI'](s)
        k_f = self.key_proj['fMRI'](f)
        k_d = self.key_proj['dMRI'](d)

        v_s = self.value_proj['sMRI'](s)
        v_f = self.value_proj['fMRI'](f)
        v_d = self.value_proj['dMRI'](d)

        # Concatenate keys and values
        k_all = torch.stack([k_s, k_f, k_d], dim=2)  # (B, n, 3, d)
        v_all = torch.stack([v_s, v_f, v_d], dim=2)  # (B, n, 3, d)

        k_all = k_all.reshape(B, self.n_latent, -1)  # (B, n, 3d)
        v_all = v_all.reshape(B, self.n_latent, -1)  # (B, n, 3d)

        # Compute attention scores
        scale = self.d_model ** -0.5
        scores = torch.bmm(q, k_all.transpose(1, 2)) * scale  # (B, n, 3)
        attn = F.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.bmm(attn, v_all)  # (B, n, d)

        # Output
        out = self.norm(self.output_proj(out))

        return out
