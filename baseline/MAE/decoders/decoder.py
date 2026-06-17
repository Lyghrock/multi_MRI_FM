"""
Modality-Specific Decoders for MAE

Each decoder reconstructs the original input from the shared latent representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SMRI_Decoder(nn.Module):
    """
    Decoder for sMRI (structural MRI) ROI features.

    Reconstructs: (B, n_rois, 3) -> ROI features [mean, volume, max]
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 256,
        n_rois: int = 200,
        n_features: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Latent dimension
            hidden_dim: Hidden layer dimension
            n_rois: Number of ROIs
            n_features: Number of features per ROI (default: 3)
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois
        self.n_features = n_features

        # Input projection
        self.input_proj = nn.Linear(d_model, hidden_dim)

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output projection to ROI features
        self.output_proj = nn.Linear(hidden_dim, n_rois * n_features)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, n_latent, d_model) latent representation

        Returns:
            (B, n_rois, n_features) reconstructed ROI features
        """
        B = H.shape[0]

        # Project and decode
        x = self.input_proj(H)  # (B, n_latent, hidden)
        x = self.decoder(x)  # (B, n_latent, hidden)

        # Pool across latent dimension
        x = x.mean(dim=1)  # (B, hidden)

        # Project to output space
        x = self.output_proj(x)  # (B, n_rois * n_features)

        # Reshape to (B, n_rois, n_features)
        x = x.view(B, self.n_rois, self.n_features)

        # Apply positive constraint for physical features
        # sMRI features should be non-negative
        x = F.softplus(x)

        return x


class FMRI_Decoder(nn.Module):
    """
    Decoder for fMRI (functional MRI) representations.

    Reconstructs: (B, n_latent, d) -> functional representations
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 256,
        n_rois: int = 200,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Latent dimension
            hidden_dim: Hidden layer dimension
            n_rois: Number of ROIs
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois

        # Input projection
        self.input_proj = nn.Linear(d_model, hidden_dim)

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output projection for FC matrix
        # Upper triangular of FC matrix: n_rois * (n_rois - 1) // 2
        n_upper = n_rois * (n_rois - 1) // 2
        self.output_proj = nn.Linear(hidden_dim, n_upper)
        self.n_upper = n_upper

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, n_latent, d_model) latent representation

        Returns:
            (B, n_upper) flattened upper triangular FC
        """
        B = H.shape[0]

        # Project and decode
        x = self.input_proj(H)
        x = self.decoder(x)

        # Pool across latent dimension
        x = x.mean(dim=1)  # (B, hidden)

        # Project to FC space
        x = self.output_proj(x)  # (B, n_upper)

        return x

    def reconstruct_fc_matrix(self, flat_fc: torch.Tensor) -> torch.Tensor:
        """
        Convert flattened upper triangular to symmetric FC matrix.

        Args:
            flat_fc: (B, n_upper) flattened upper triangular

        Returns:
            (B, n_rois, n_rois) symmetric FC matrix
        """
        B = flat_fc.shape[0]
        fc = torch.zeros(B, self.n_rois, self.n_rois, device=flat_fc.device)

        # Fill upper triangular
        i, j = torch.triu_indices(self.n_rois, self.n_rois, offset=1)
        fc[:, i, j] = flat_fc
        fc[:, j, i] = flat_fc  # Symmetric

        return fc


class DMRI_Decoder(nn.Module):
    """
    Decoder for dMRI (diffusion MRI) Structural Connectivity (SC).

    Reconstructs: (B, n_latent, d) -> SC matrix
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 512,
        n_rois: int = 200,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Latent dimension
            hidden_dim: Hidden layer dimension
            n_rois: Number of ROIs
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.n_rois = n_rois

        # Input projection
        self.input_proj = nn.Linear(d_model, hidden_dim)

        # Decoder MLP (larger hidden dim for SC output)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output projection for SC matrix (upper triangular)
        n_upper = n_rois * (n_rois - 1) // 2
        self.output_proj = nn.Linear(hidden_dim // 2, n_upper)
        self.n_upper = n_upper

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, n_latent, d_model) latent representation

        Returns:
            (B, n_upper) flattened upper triangular SC
        """
        B = H.shape[0]

        # Project and decode
        x = self.input_proj(H)
        x = self.decoder(x)

        # Pool across latent dimension
        x = x.mean(dim=1)  # (B, hidden)

        # Project to SC space
        x = self.output_proj(x)  # (B, n_upper)

        # Apply ReLU for non-negative connectivity
        x = F.relu(x)

        return x

    def reconstruct_sc_matrix(self, flat_sc: torch.Tensor) -> torch.Tensor:
        """
        Convert flattened upper triangular to symmetric SC matrix.

        Args:
            flat_sc: (B, n_upper) flattened upper triangular

        Returns:
            (B, n_rois, n_rois) symmetric SC matrix
        """
        B = flat_sc.shape[0]
        sc = torch.zeros(B, self.n_rois, self.n_rois, device=flat_sc.device)

        # Fill upper triangular
        i, j = torch.triu_indices(self.n_rois, self.n_rois, offset=1)
        sc[:, i, j] = flat_sc
        sc[:, j, i] = flat_sc  # Symmetric

        # Add small diagonal for numerical stability
        sc = sc + torch.eye(self.n_rois, device=sc.device) * 1e-6

        return sc


class MAE_Decoder(nn.Module):
    """
    Unified MAE Decoder combining all modality-specific decoders.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 256,
        n_rois: int = 200,
        n_features: int = 3,
        dropout: float = 0.1,
        include_sMRI: bool = True,
        include_fMRI: bool = True,
        include_dMRI: bool = True
    ):
        super().__init__()

        self.include_sMRI = include_sMRI
        self.include_fMRI = include_fMRI
        self.include_dMRI = include_dMRI

        if include_sMRI:
            self.sMRI_decoder = SMRI_Decoder(
                d_model=d_model,
                hidden_dim=hidden_dim,
                n_rois=n_rois,
                n_features=n_features,
                dropout=dropout
            )

        if include_fMRI:
            self.fMRI_decoder = FMRI_Decoder(
                d_model=d_model,
                hidden_dim=hidden_dim,
                n_rois=n_rois,
                dropout=dropout
            )

        if include_dMRI:
            self.dMRI_decoder = DMRI_Decoder(
                d_model=d_model,
                hidden_dim=hidden_dim * 2,  # Larger for SC
                n_rois=n_rois,
                dropout=dropout
            )

    def forward(self, H: torch.Tensor) -> dict:
        """
        Args:
            H: (B, n_latent, d_model) latent representation

        Returns:
            Dictionary of decoded outputs for each modality
        """
        outputs = {}

        if self.include_sMRI and hasattr(self, 'sMRI_decoder'):
            outputs['sMRI'] = self.sMRI_decoder(H)

        if self.include_fMRI and hasattr(self, 'fMRI_decoder'):
            outputs['fMRI'] = self.fMRI_decoder(H)

        if self.include_dMRI and hasattr(self, 'dMRI_decoder'):
            outputs['dMRI'] = self.dMRI_decoder(H)

        return outputs
