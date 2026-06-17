"""
fMRI Encoder Implementation

Encodes functional MRI (resting-state or task) data into latent representations.
Uses PyTorch native Flash Attention (SDPA) for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..backbone.transformer import TransformerEncoder
from ..backbone.positional_encoding import get_positional_encoding


class fMRIEncoder(nn.Module):
    """
    Encoder for fMRI ROI time-series.

    Encodes ROI-based time-series into latent space.
    Uses PyTorch native SDPA (Flash Attention) for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 4,
        n_rois: int = 200,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        positional_encoding: str = 'learnable',
        use_flash: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ffn: Feed-forward dimension
            n_layers: Number of transformer layers
            n_rois: Number of ROIs
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding
            use_flash: Use PyTorch native Flash Attention (SDPA)
        """
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.n_heads = n_heads
        self.use_flash = use_flash

        # Input projection: 1 -> d_model (treating each ROI as a token)
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding for time
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

        # QKV projections for temporal attention (using SDPA)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.temporal_dropout = nn.Dropout(dropout)

        # Transformer encoder for ROI interactions (with SDPA)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            use_flash=use_flash
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Optional FC matrix-patch encoder. The dataset stores FC patches as
        # (B, K, P, P); this path summarizes them and fuses the summary back
        # into every ROI token.
        self.fc_patch_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.patch_fusion = nn.Linear(d_model * 2, d_model)

    def _encode_fc_patches(self, fc_patches: torch.Tensor) -> torch.Tensor:
        """Encode FC patches into one subject-level vector."""
        if fc_patches.dim() == 3:
            fc_patches = fc_patches.unsqueeze(0)
        if fc_patches.dim() != 4:
            raise ValueError(f"Expected fMRI FC patches as (B,K,P,P), got {tuple(fc_patches.shape)}")

        B, K = fc_patches.shape[:2]
        flat = fc_patches.reshape(B, K, -1)
        stats = torch.stack(
            [flat.mean(dim=-1), flat.std(dim=-1, unbiased=False), flat.amax(dim=-1)],
            dim=-1,
        )
        return self.fc_patch_encoder(stats).mean(dim=1)

    def _temporal_attention(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Temporal self-attention using PyTorch native SDPA.

        Args:
            x: (B*N, T, d_model)

        Returns:
            (B*N, d_model)
        """
        B_N, T, d = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)  # (B*N, T, 3d)
        qkv = qkv.reshape(B_N, T, 3, self.n_heads, d // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, H, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SDPA with Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            if self.use_flash and q.is_cuda:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True
                ):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.temporal_dropout.p if self.training else 0.0
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.temporal_dropout.p if self.training else 0.0
                )
        else:
            scale = (d // self.n_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            attn = self.temporal_dropout(attn)
            out = attn @ v

        # Reshape and project
        out = out.transpose(1, 2).contiguous()  # (B*N, T, d_head*H)
        out = out.view(B_N, T, d)
        out = self.out_proj(out)

        # Average over time
        out = out.mean(dim=1)  # (B*N, d_model)

        return out

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fc_patches: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_rois) or (B, T*n_rois) flattened
            attention_mask: (B, T) or (B, n_rois)
            fc_patches: (B, K, P, P) optional FC matrix patches

        Returns:
            (B, n_rois, d_model)
        """
        B, T, N = x.shape

        # Project each ROI's time series
        x = x.unsqueeze(-1)  # (B, T, N, 1)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, 1)
        x = x.reshape(B * N, T, 1)  # (B*N, T, 1)

        # Input projection
        x = self.input_proj(x)  # (B*N, T, d_model)

        # Temporal attention (with SDPA)
        x = self._temporal_attention(x)  # (B*N, d_model)

        # Reshape and apply ROI transformer
        x = x.view(B, N, self.d_model)  # (B, N, d_model)

        # Add ROI positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        # ROI interactions via transformer (with SDPA)
        x = self.transformer(x, attention_mask)

        # Output projection
        x = self.output_proj(x)

        if fc_patches is not None:
            patch_context = self._encode_fc_patches(fc_patches)
            patch_context = patch_context.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = self.patch_fusion(torch.cat([x, patch_context], dim=-1))

        return x


class fMRIEncoderWithTemporal(nn.Module):
    """
    fMRI Encoder with explicit temporal and spatial attention.

    Processes time-series and dynamic FC together.
    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 4,
        n_rois: int = 200,
        n_windows: int = 20,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.n_windows = n_windows
        self.use_flash = use_flash

        # Time-series encoder
        self.timeseries_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )

        # FC encoder (processes upper triangular of FC matrix)
        n_fc_patches = n_rois * (n_rois - 1) // 2
        self.fc_encoder = nn.Sequential(
            nn.Linear(n_fc_patches, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Temporal transformer (with SDPA)
        self.temporal_transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            d_ffn=d_ffn,
            dropout=dropout,
            use_flash=use_flash
        )

        # Fusion and output
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        timeseries: torch.Tensor,
        fc_matrices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            timeseries: (B, T, n_rois)
            fc_matrices: (B, n_windows, n_rois, n_rois) optional

        Returns:
            (B, n_rois, d_model)
        """
        B, T, N = timeseries.shape

        # Encode time-series
        # Process each ROI separately then aggregate
        ts_encoded = []
        for i in range(N):
            roi_ts = timeseries[:, :, i:i+1]  # (B, T, 1)
            encoded = self.timeseries_encoder(roi_ts)  # (B, T, d_model)
            encoded = encoded.mean(dim=1)  # (B, d_model)
            ts_encoded.append(encoded)

        ts_out = torch.stack(ts_encoded, dim=1)  # (B, N, d_model)

        # Encode FC matrices if provided
        if fc_matrices is not None:
            B, W, N, _ = fc_matrices.shape

            # Extract upper triangular
            upper_indices = torch.triu_indices(N, N, offset=1, device=fc_matrices.device)
            fc_flat = fc_matrices[:, :, upper_indices[0], upper_indices[1]]  # (B, W, n_upper)

            # Encode and aggregate across windows
            fc_encoded = self.fc_encoder(fc_flat)  # (B, W, d_model)
            fc_out = fc_encoded.mean(dim=1)  # (B, d_model)

            # Expand for all ROIs
            fc_out = fc_out.unsqueeze(1).expand(-1, N, -1)  # (B, N, d_model)

            # Fuse
            fused = torch.cat([ts_out, fc_out], dim=-1)  # (B, N, 2*d_model)
            out = self.fusion(fused)  # (B, N, d_model)
        else:
            out = ts_out

        # Apply transformer (with SDPA)
        out = self.temporal_transformer(out)

        # Output projection
        out = self.output_proj(out)

        return out


class DynamicFCEncoder(nn.Module):
    """
    Encoder for dynamic Functional Connectivity matrices.

    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_rois: int = 200,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.use_flash = use_flash

        # Patch embedding for FC matrix
        patch_size = 20
        n_patches = (n_rois + patch_size - 1) // patch_size

        self.patch_embed = nn.Conv2d(
            1, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches * n_patches, d_model) * 0.02)

        # Transformer for patches (with SDPA)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_flash=use_flash
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fc_matrix: (B, n_rois, n_rois)

        Returns:
            (B, n_patches^2, d_model)
        """
        B = fc_matrix.shape[0]

        # Add channel dimension
        x = fc_matrix.unsqueeze(1)  # (B, 1, N, N)

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, n_patches, n_patches)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches^2, d_model)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer (with SDPA)
        x = self.transformer(x)

        # Output projection
        x = self.output_proj(x)

        return x
