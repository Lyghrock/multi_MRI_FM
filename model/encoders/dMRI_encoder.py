"""
dMRI Encoder Implementation

Encodes diffusion MRI data into latent representations.
Uses PyTorch native Flash Attention (SDPA) for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..backbone.transformer import TransformerEncoder
from ..backbone.positional_encoding import get_positional_encoding


class dMRIEncoder(nn.Module):
    """
    Encoder for dMRI Structural Connectivity (SC) matrices.

    Encodes SC matrix into latent space.
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
        self.use_flash = use_flash

        # Input projection: 1 -> d_model
        self.input_proj = nn.Linear(1, d_model)

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

        # Transformer encoder (with SDPA)
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

        # Optional patch encoders. SC matrix patches and FA/MD voxel patches
        # are summarized at subject level and fused into ROI tokens.
        self.sc_patch_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.voxel_patch_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.patch_fusion = nn.Linear(d_model * 3, d_model)

    def _encode_matrix_patches(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() == 3:
            patches = patches.unsqueeze(0)
        if patches.dim() != 4:
            raise ValueError(f"Expected dMRI SC patches as (B,K,P,P), got {tuple(patches.shape)}")

        B, K = patches.shape[:2]
        flat = patches.reshape(B, K, -1)
        stats = torch.stack(
            [flat.mean(dim=-1), flat.std(dim=-1, unbiased=False), flat.amax(dim=-1)],
            dim=-1,
        )
        return self.sc_patch_encoder(stats).mean(dim=1)

    def _encode_voxel_patches(self, *patch_groups: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        summaries = []
        for patches in patch_groups:
            if patches is None:
                continue
            if patches.dim() == 4:
                patches = patches.unsqueeze(0)
            if patches.dim() != 5:
                raise ValueError(f"Expected dMRI voxel patches as (B,K,D,H,W), got {tuple(patches.shape)}")

            B, K = patches.shape[:2]
            flat = patches.reshape(B, K, -1)
            stats = torch.stack(
                [flat.mean(dim=-1), flat.std(dim=-1, unbiased=False), flat.amax(dim=-1)],
                dim=-1,
            )
            summaries.append(self.voxel_patch_encoder(stats).mean(dim=1))

        if not summaries:
            return None
        return torch.stack(summaries, dim=0).mean(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sc_patches: Optional[torch.Tensor] = None,
        fa_patches: Optional[torch.Tensor] = None,
        md_patches: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_rois, n_rois) SC matrix
            attention_mask: (B, n_rois)
            sc_patches: (B, K, P, P) optional SC matrix patches
            fa_patches: (B, K, D, H, W) optional FA voxel patches
            md_patches: (B, K, D, H, W) optional MD voxel patches

        Returns:
            (B, n_rois, d_model)
        """
        B, N, _ = x.shape

        # Process upper triangular (excluding diagonal) for efficiency
        # Take mean across connectivity for each ROI
        x_row_mean = x.mean(dim=-1, keepdim=True)  # (B, n_rois, 1)

        # Project
        x = self.input_proj(x_row_mean)  # (B, n_rois, d_model)

        # Add positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        # Transformer encoding (with SDPA)
        x = self.transformer(x, attention_mask)

        # Output projection
        x = self.output_proj(x)

        matrix_context = self._encode_matrix_patches(sc_patches) if sc_patches is not None else None
        voxel_context = self._encode_voxel_patches(fa_patches, md_patches)

        if matrix_context is not None or voxel_context is not None:
            if matrix_context is None:
                matrix_context = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
            if voxel_context is None:
                voxel_context = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
            matrix_context = matrix_context.to(device=x.device, dtype=x.dtype).unsqueeze(1).expand(-1, N, -1)
            voxel_context = voxel_context.to(device=x.device, dtype=x.dtype).unsqueeze(1).expand(-1, N, -1)
            x = self.patch_fusion(torch.cat([x, matrix_context, voxel_context], dim=-1))

        return x


class dMRIEncoderWithPatches(nn.Module):
    """
    dMRI Encoder with patch-based processing.

    Processes SC matrix as patches.
    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 4,
        n_rois: int = 200,
        dropout: float = 0.1,
        patch_size: int = 20,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.patch_size = patch_size
        self.use_flash = use_flash
        n_patches = (n_rois + patch_size - 1) // patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            1, d_model // 2,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Upper triangular extractor
        self.triu_indices = torch.triu_indices(n_rois, n_rois, offset=1)
        self.n_upper = self.triu_indices.shape[0]

        # Upper triangular encoder
        self.triu_encoder = nn.Sequential(
            nn.Linear(self.n_upper, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Transformer for ROI interactions (with SDPA)
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

        # Register upper triangular indices as buffer
        self.register_buffer('triu_idx', self.triu_indices)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_rois, n_rois) SC matrix

        Returns:
            (B, n_rois, d_model)
        """
        B = x.shape[0]

        # Extract upper triangular
        triu = x[:, self.triu_idx[0], self.triu_idx[1]]  # (B, n_upper)

        # Encode upper triangular
        triu_encoded = self.triu_encoder(triu)  # (B, d_model)

        # Expand for each ROI
        triu_expanded = triu_encoded.unsqueeze(1).expand(-1, self.n_rois, -1)  # (B, n_rois, d_model)

        # Transformer (with SDPA)
        out = self.transformer(triu_expanded, attention_mask)

        # Output projection
        out = self.output_proj(out)

        return out


class SCGraphEncoder(nn.Module):
    """
    Graph-based SC encoder using message passing.

    Uses PyTorch native SDPA for efficiency.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_rois: int = 200,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.use_flash = use_flash

        # Node embedding
        self.node_embedding = nn.Linear(1, d_model)

        # Graph convolution layers
        self.gc_layers = nn.ModuleList([
            GraphConvLayer(d_model, dropout=dropout, use_flash=use_flash)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        sc_matrix: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sc_matrix: (B, n_rois, n_rois) SC matrix
            edge_weights: (B, n_edges) optional edge weights

        Returns:
            (B, n_rois, d_model)
        """
        # Normalize SC matrix (degree normalization)
        degree = sc_matrix.sum(dim=-1, keepdim=True) + 1e-8
        adj_normalized = sc_matrix / degree

        # Initial node features from diagonal (self-connectivity)
        x = sc_matrix.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)  # (B, n_rois, 1)

        # Embed
        x = self.node_embedding(x)  # (B, n_rois, d_model)

        # Graph convolution
        for gc_layer in self.gc_layers:
            x = gc_layer(x, adj_normalized)

        # Output projection
        x = self.output_proj(x)

        return x


class GraphConvLayer(nn.Module):
    """
    Graph convolution layer with optional SDPA.

    Uses PyTorch native operations for efficiency.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, use_flash: bool = True):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_flash = use_flash

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) node features
            adj: (B, N, N) adjacency matrix

        Returns:
            (B, N, d_model)
        """
        # Graph convolution: aggregate neighbor features
        x_agg = torch.bmm(adj, x)  # (B, N, d_model)

        # Linear transformation
        x_agg = self.linear(x_agg)

        # Residual connection with norm
        x = self.norm(x + self.dropout(x_agg))

        return x
