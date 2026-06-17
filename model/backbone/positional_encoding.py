"""
Positional Encoding Implementations for Brain MRI

Provides positional encoding strategies specifically designed for brain ROI data.
Based on Schaefer-200 atlas functional networks and anatomical organization.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List
from enum import Enum


class NetworkType(Enum):
    """Schaefer-200 functional network types."""
    VISUAL = 0           # ROI 1-28
    SOMATOMOTOR = 1      # ROI 29-56
    DORSAL_ATTENTION = 2  # ROI 57-78
    VENTRAL_ATTENTION = 3 # ROI 79-92
    LIMBIC = 4           # ROI 93-100
    FRONTO_PARIENTAL = 5 # ROI 101-142
    DEFAULT_MODE = 6      # ROI 143-200


# Schaefer-200 网络边界 (200 ROIs, 7 networks)
NETWORK_BOUNDARIES = {
    'Visual': (0, 27),
    'SomatoMotor': (28, 55),
    'DorsalAttention': (56, 77),
    'VentralAttention': (78, 91),
    'Limbic': (92, 99),
    'FrontoParietal': (100, 141),
    'DefaultMode': (142, 199),
}

# 每个 ROI 的半球归属 (L=0, R=1)
# 左半球: ROI 0-99, 右半球: ROI 100-199 (大致)
HEMISPHERE_BOUNDARY = 100


def get_roi_network(roi_idx: int) -> str:
    """Get the network name for a given ROI index."""
    for network, (start, end) in NETWORK_BOUNDARIES.items():
        if start <= roi_idx <= end:
            return network
    return "Unknown"


def get_roi_hemisphere(roi_idx: int) -> int:
    """Get hemisphere (0=left, 1=right) for a given ROI index."""
    return 1 if roi_idx >= HEMISPHERE_BOUNDARY else 0


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for 1D sequences.

    From "Attention Is All You Need" (Vaswani et al., 2017).

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)

        Returns:
            (B, N, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.

    Each position has a trainable embedding.
    Simple but may overfit on small datasets.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)

        Returns:
            (B, N, d_model)
        """
        pe = self.pe[:, :x.size(1)]
        x = x + pe
        return self.dropout(x)


class BrainAwarePositionalEncoding(nn.Module):
    """
    Brain-aware Positional Encoding for Schaefer-200 atlas.

    Encodes three key pieces of information:
    1. ROI sequential position (within network)
    2. Network membership (7 networks)
    3. Hemisphere (left/right)

    This is specifically designed for the Schaefer-200 functional parcellation,
    where ROIs are ordered by network and hemisphere.
    """

    def __init__(
        self,
        d_model: int,
        n_rois: int = 200,
        dropout: float = 0.1,
        learnable_network: bool = True,
        learnable_hemisphere: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            n_rois: Number of ROIs (default 200 for Schaefer-200)
            dropout: Dropout rate
            learnable_network: Use learnable network embeddings
            learnable_hemisphere: Use learnable hemisphere embeddings
        """
        super().__init__()
        self.d_model = d_model
        self.n_rois = n_rois
        self.learnable_network = learnable_network
        self.learnable_hemisphere = learnable_hemisphere

        # 1. Within-network position encoding (sinusoidal)
        max_in_network_pos = max((end - start + 1) for start, end in NETWORK_BOUNDARIES.values())
        self.pos_encoder = SinusoidalPositionalEncoding(d_model // 3, max_in_network_pos, dropout=0)

        # 2. Network embedding (7 networks)
        self.n_networks = 7
        if learnable_network:
            self.network_emb = nn.Embedding(self.n_networks, d_model // 3)
        else:
            self.network_emb = self._create_network_one_hot(d_model // 3)

        # 3. Hemisphere embedding (2 hemispheres)
        if learnable_hemisphere:
            self.hemisphere_emb = nn.Embedding(2, d_model // 3)
        else:
            self.hemisphere_emb = self._create_hemisphere_one_hot(d_model // 3)

        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute ROI metadata
        self.register_buffer('roi_networks', self._get_roi_networks())
        self.register_buffer('roi_hemispheres', self._get_roi_hemispheres())
        self.register_buffer('roi_positions', self._get_roi_positions())

    def _create_network_one_hot(self, dim: int) -> nn.Module:
        """Create fixed network one-hot embeddings."""
        network_emb = torch.zeros(self.n_networks, dim)
        for i in range(self.n_networks):
            network_emb[i, i % dim] = 1.0
        emb = nn.Embedding(self.n_networks, dim)
        emb.weight.data = network_emb
        emb.weight.requires_grad = False
        return emb

    def _create_hemisphere_one_hot(self, dim: int) -> nn.Module:
        """Create fixed hemisphere one-hot embeddings."""
        hem_emb = torch.tensor([[1., 0.], [0., 1.]])
        emb = nn.Embedding(2, dim)
        emb.weight.data = hem_emb[:2, :dim]
        emb.weight.requires_grad = False
        return emb

    def _get_roi_networks(self) -> torch.Tensor:
        """Get network index for each ROI (0-199)."""
        networks = torch.zeros(self.n_rois, dtype=torch.long)
        for network_idx, (network_name, (start, end)) in enumerate(NETWORK_BOUNDARIES.items()):
            for roi_idx in range(start, end + 1):
                if roi_idx < self.n_rois:
                    networks[roi_idx] = network_idx
        return networks

    def _get_roi_hemispheres(self) -> torch.Tensor:
        """Get hemisphere index for each ROI (0=left, 1=right)."""
        hemispheres = torch.zeros(self.n_rois, dtype=torch.long)
        for roi_idx in range(self.n_rois):
            hemispheres[roi_idx] = 1 if roi_idx >= HEMISPHERE_BOUNDARY else 0
        return hemispheres

    def _get_roi_positions(self) -> torch.Tensor:
        """Get within-network position for each ROI."""
        positions = torch.zeros(self.n_rois, dtype=torch.long)
        for network_name, (start, end) in NETWORK_BOUNDARIES.items():
            for local_pos, roi_idx in enumerate(range(start, end + 1)):
                if roi_idx < self.n_rois:
                    positions[roi_idx] = local_pos
        return positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model), N should be <= n_rois

        Returns:
            (B, N, d_model) with brain-aware positional encoding
        """
        B, N, _ = x.shape

        # Truncate metadata if needed
        roi_networks = self.roi_networks[:N]
        roi_hemispheres = self.roi_hemispheres[:N]
        roi_positions = self.roi_positions[:N]

        # 1. Within-network position encoding
        pos_enc = self.pos_encoder.pe[:, roi_positions]  # (1, N, d//3)

        # 2. Network encoding
        if self.learnable_network:
            net_enc = self.network_emb(roi_networks)  # (N, d//3)
        else:
            net_enc = self.network_emb(roi_networks)

        # 3. Hemisphere encoding
        if self.learnable_hemisphere:
            hem_enc = self.hemisphere_emb(roi_hemispheres)  # (N, d//3)
        else:
            hem_enc = self.hemisphere_emb(roi_hemispheres)

        net_enc = net_enc.unsqueeze(0)
        hem_enc = hem_enc.unsqueeze(0)

        # Concatenate all encodings
        pe = torch.cat([pos_enc, net_enc, hem_enc], dim=-1)  # (1, N, d)
        pe = pe.expand(B, -1, -1)  # (B, N, d)

        # Pad if d_model is not divisible by 3
        if pe.shape[-1] < self.d_model:
            pe = torch.cat([pe, torch.zeros(B, N, self.d_model - pe.shape[-1], device=pe.device)], dim=-1)

        x = x + pe[:, :N, :self.d_model]
        return self.dropout(x)


class FunctionalNetworkEncoding(nn.Module):
    """
    Functional Network Encoding based on Schaefer-200.

    Encodes each ROI with its network membership using a structured embedding
    that captures the hierarchical organization of brain networks.

    Network Hierarchy:
        Level 1: Primary (Vis, SomMot) vs Association (DorsAttn, VentAttn, FpnCont, Default)
        Level 2: Specific network identity
        Level 3: Within-network position
    """

    def __init__(
        self,
        d_model: int,
        n_rois: int = 200,
        n_networks: int = 7,
        dropout: float = 0.1,
        use_hierarchy: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_rois = n_rois
        self.n_networks = n_networks
        self.use_hierarchy = use_hierarchy

        # Primary cortex type (sensory vs association)
        self.primary_type_emb = nn.Embedding(2, d_model // 4)

        # Network-level embedding
        self.network_emb = nn.Embedding(n_networks, d_model // 4)

        # Within-network position (sinusoidal)
        max_pos = 50
        self.register_buffer('div_term',
            torch.exp(torch.arange(0, d_model // 4, 2, dtype=torch.float) *
                      (-math.log(10000.0) / (d_model // 4))))

        # Hemisphere embedding
        self.hemisphere_emb = nn.Embedding(2, d_model // 4)

        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute ROI metadata
        self._setup_roi_metadata()

    def _setup_roi_metadata(self):
        """Pre-compute network and type for each ROI."""
        network_idx = []
        primary_type = []  # 0=sensory, 1=association
        hemisphere = []
        in_network_pos = []

        sensory_networks = {'Visual', 'SomatoMotor'}

        for roi_idx in range(self.n_rois):
            network_name = get_roi_network(roi_idx)
            network_id = list(NETWORK_BOUNDARIES.keys()).index(network_name)
            network_idx.append(network_id)

            # Primary vs association cortex
            primary_type.append(0 if network_name in sensory_networks else 1)

            # Hemisphere
            hemisphere.append(1 if roi_idx >= HEMISPHERE_BOUNDARY else 0)

            # Position within network
            start, _ = NETWORK_BOUNDARIES[network_name]
            in_network_pos.append(roi_idx - start)

        self.register_buffer('roi_network_idx', torch.tensor(network_idx))
        self.register_buffer('roi_primary_type', torch.tensor(primary_type))
        self.register_buffer('roi_hemisphere', torch.tensor(hemisphere))
        self.register_buffer('roi_in_network_pos', torch.tensor(in_network_pos))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)

        Returns:
            (B, N, d_model)
        """
        B, N, _ = x.shape

        # Truncate to sequence length
        network_idx = self.roi_network_idx[:N]
        primary_type = self.roi_primary_type[:N]
        hemisphere = self.roi_hemisphere[:N]
        in_network_pos = self.roi_in_network_pos[:N].float()

        # 1. Primary type encoding
        primary_enc = self.primary_type_emb(primary_type)  # (N, d//4)

        # 2. Network encoding
        network_enc = self.network_emb(network_idx)  # (N, d//4)

        # 3. Within-network position (sinusoidal)
        pos_enc = torch.zeros(N, self.d_model // 4, device=x.device)
        pos_enc[:, 0::2] = torch.sin(in_network_pos.unsqueeze(1) * self.div_term)
        pos_enc[:, 1::2] = torch.cos(in_network_pos.unsqueeze(1) * self.div_term[:pos_enc.shape[1] // 2 + 1])

        # 4. Hemisphere encoding
        hem_enc = self.hemisphere_emb(hemisphere)  # (N, d//4)

        # Concatenate
        pe = torch.cat([primary_enc, network_enc, pos_enc, hem_enc], dim=-1)  # (N, d)

        # Adjust dimension if needed
        if pe.shape[-1] < self.d_model:
            pe = torch.cat([pe, torch.zeros(N, self.d_model - pe.shape[-1], device=pe.device)], dim=-1)
        pe = pe[:self.d_model]

        x = x + pe.unsqueeze(0).expand(B, -1, -1)
        return self.dropout(x)


class AnatomicalPositionalEncoding(nn.Module):
    """
    Anatomical positional encoding using 3D MNI coordinates.

    Projects 3D coordinates of each ROI into the embedding space,
    preserving the spatial relationships between brain regions.

    Note: Requires pre-computed MNI coordinates for each ROI.
    """

    def __init__(
        self,
        d_model: int,
        roi_coordinates: torch.Tensor,  # (N_rois, 3) MNI coordinates
        dropout: float = 0.1,
        use_learnable: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            roi_coordinates: (N_rois, 3) MNI coordinates for each ROI
            dropout: Dropout rate
            use_learnable: Use learnable coordinate projection
        """
        super().__init__()
        self.d_model = d_model
        self.roi_coordinates = roi_coordinates
        self.n_rois = roi_coordinates.shape[0]

        # Normalize coordinates
        coords_min = roi_coordinates.min(dim=0)[0]
        coords_max = roi_coordinates.max(dim=0)[0]
        self.register_buffer('coords_min', coords_min)
        self.register_buffer('coords_max', coords_max)

        if use_learnable:
            # Learnable projection
            self.coord_proj = nn.Linear(3, d_model)
        else:
            # Sinusoidal projection
            self._create_sinusoidal_projection()

        self.dropout = nn.Dropout(p=dropout)

    def _create_sinusoidal_projection(self):
        """Create sinusoidal encoding for 3D coordinates."""
        # Normalize coordinates
        coords_norm = (self.roi_coordinates - self.coords_min) / (
            self.coords_max - self.coords_min + 1e-8
        )

        pe = torch.zeros(self.n_rois, self.d_model)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 3, dtype=torch.float) *
            (-math.log(10000.0) / self.d_model)
        )

        for i in range(0, self.d_model, 3):
            if i + 2 < self.d_model:
                pe[:, i] = torch.sin(coords_norm[:, 0] * div_term[i // 3])
                pe[:, i+1] = torch.sin(coords_norm[:, 1] * div_term[i // 3])
                pe[:, i+2] = torch.sin(coords_norm[:, 2] * div_term[i // 3])

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, roi_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)
            roi_indices: (B, N) indices of ROIs (optional)

        Returns:
            (B, N, d_model)
        """
        B, N, _ = x.shape

        if hasattr(self, 'coord_proj'):
            if roi_indices is not None:
                coords = self.roi_coordinates[roi_indices]  # (B, N, 3)
            else:
                coords = self.roi_coordinates[:N].unsqueeze(0).expand(B, -1, -1)
            pe = self.coord_proj(coords)
        else:
            if roi_indices is not None:
                pe = self.pe[roi_indices]
            else:
                pe = self.pe[:N].unsqueeze(0).expand(B, -1, -1)

        x = x + pe
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Reference: https://arxiv.org/abs/2104.09864

    Rotates query and key vectors to encode position information,
    allowing for efficient relative position encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Pre-compute rotation angles
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)

        Returns:
            (B, N, d_model) with rotary embeddings applied
        """
        seq_len = x.size(1)

        # Create position indices
        position = torch.arange(seq_len, device=x.device, dtype=torch.float)
        position = position.unsqueeze(1)

        # Compute angles
        angles = position * self.inv_freq[:seq_len // 2].unsqueeze(0)

        # Compute cos and sin
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return x * cos + self._rotate_half(x) * sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


class HybridPositionalEncoding(nn.Module):
    """
    Hybrid Positional Encoding combining multiple strategies.

    Combines:
    1. Functional network encoding (brain-aware)
    2. Anatomical coordinate encoding (spatial)
    3. Learnable position encoding (residual)
    """

    def __init__(
        self,
        d_model: int,
        n_rois: int = 200,
        roi_coordinates: Optional[torch.Tensor] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Functional network encoding
        self.functional_enc = FunctionalNetworkEncoding(
            d_model=d_model // 3,
            n_rois=n_rois,
            dropout=0  # We'll apply dropout once at the end
        )

        # Anatomical encoding (if coordinates provided)
        if roi_coordinates is not None:
            self.anatomical_enc = AnatomicalPositionalEncoding(
                d_model=d_model // 3,
                roi_coordinates=roi_coordinates,
                dropout=0
            )
        else:
            self.anatomical_enc = None

        # Learnable residual position encoding
        self.learnable_enc = LearnablePositionalEncoding(
            d_model=d_model - 2 * (d_model // 3),
            max_len=n_rois,
            dropout=0
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)

        Returns:
            (B, N, d_model)
        """
        parts = []

        # 1. Functional encoding
        func_enc = self.functional_enc(x[:, :, :self.d_model // 3])
        parts.append(func_enc)

        # 2. Anatomical encoding (if available)
        if self.anatomical_enc is not None:
            anat_enc = self.anatomical_enc(x[:, :, self.d_model // 3:2 * self.d_model // 3])
            parts.append(anat_enc)
        else:
            parts.append(x[:, :, self.d_model // 3:2 * self.d_model // 3])

        # 3. Learnable encoding
        learn_enc = self.learnable_enc(x[:, :, 2 * self.d_model // 3:])
        parts.append(learn_enc)

        # Concatenate and apply dropout
        pe = torch.cat(parts, dim=-1)

        # Adjust dimension if needed
        if pe.shape[-1] < self.d_model:
            pe = torch.cat([pe, torch.zeros(*pe.shape[:-1], self.d_model - pe.shape[-1], device=pe.device)], dim=-1)

        return self.dropout(pe)


def get_positional_encoding(
    encoding_type: str,
    d_model: int,
    max_len: int = 5000,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create positional encoding.

    Args:
        encoding_type: Type of encoding:
            - 'sinusoidal': Standard sinusoidal
            - 'learnable': Trainable embeddings
            - 'brain_aware': Schaefer-200 functional network encoding
            - 'functional': Hierarchical network encoding
            - 'anatomical': 3D MNI coordinate encoding
            - 'rotary': Rotary position embedding
            - 'hybrid': Combination of multiple strategies
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
        **kwargs: Additional arguments:
            - n_rois: Number of ROIs (for brain-aware encodings)
            - roi_coordinates: MNI coordinates (for anatomical)

    Returns:
        Positional encoding module
    """
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(d_model, max_len, dropout)

    elif encoding_type == 'learnable':
        return LearnablePositionalEncoding(d_model, max_len, dropout)

    elif encoding_type == 'brain_aware':
        n_rois = kwargs.get('n_rois', 200)
        return BrainAwarePositionalEncoding(d_model, n_rois, dropout)

    elif encoding_type == 'functional':
        n_rois = kwargs.get('n_rois', 200)
        n_networks = kwargs.get('n_networks', 7)
        return FunctionalNetworkEncoding(d_model, n_rois, n_networks, dropout)

    elif encoding_type == 'anatomical':
        roi_coords = kwargs.get('roi_coordinates')
        if roi_coords is None:
            raise ValueError("roi_coordinates required for anatomical encoding")
        return AnatomicalPositionalEncoding(d_model, roi_coords, dropout)

    elif encoding_type == 'rotary':
        return RotaryPositionalEncoding(d_model, max_len)

    elif encoding_type == 'hybrid':
        n_rois = kwargs.get('n_rois', 200)
        roi_coords = kwargs.get('roi_coordinates', None)
        return HybridPositionalEncoding(d_model, n_rois, roi_coords, dropout)

    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
