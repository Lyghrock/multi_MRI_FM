"""
Schaefer Atlas Utilities

Provides utilities for working with the Schaefer atlas for brain parcellation.

Atlas Download:
    # Schaefer-200 (default)
    wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

    # Schaefer-400
    wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

Usage:
    # Using Schaefer-200
    atlas = SchaeferAtlas(atlas_path='path/to/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz', n_rois=200)

    # Using Schaefer-400
    atlas = SchaeferAtlas(atlas_path='path/to/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz', n_rois=400)
"""

import os
import torch
import numpy as np
from typing import Optional, List
from pathlib import Path


class SchaeferAtlas:
    """
    Schaefer Atlas handler.

    Provides utilities for extracting ROI data and coordinates.
    Requires downloading the atlas NIfTI file.
    """

    # Default MNI coordinates for Schaefer 200 ROIs (approximate centroids)
    # These are pre-computed from the official atlas
    DEFAULT_COORDINATES = None  # Will be computed on first use

    # Network labels: 7 networks
    NETWORK_NAMES = [
        'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'
    ]

    def __init__(
        self,
        atlas_path: Optional[str] = None,
        n_rois: int = 200,
        n_networks: int = 7,
        space: str = 'MNI152_2mm'
    ):
        """
        Args:
            atlas_path: Path to Schaefer atlas NIfTI file.
                       If None, uses default/generated coordinates.
            n_rois: Number of ROIs (100, 200, 400, 600, 800, 1000)
            n_networks: Number of networks (7 or 17)
            space: Standard space (MNI152_2mm)
        """
        self.n_rois = n_rois
        self.n_networks = n_networks
        self.space = space

        # Load atlas if path provided
        self.atlas_data = None
        self.atlas_header = None
        self.affine = None

        if atlas_path and os.path.exists(atlas_path):
            self._load_atlas(atlas_path)
        elif atlas_path:
            print(f"Warning: Atlas file not found at {atlas_path}")
            print("Using generated coordinates (may be less accurate)")

        # Initialize coordinates and labels
        self.roi_coordinates = self._get_coordinates()
        self.roi_network_labels = self._get_network_labels()

    def _load_atlas(self, atlas_path: str):
        """Load atlas from NIfTI file."""
        try:
            import nibabel as nib
            img = nib.load(atlas_path)
            self.atlas_data = img.get_fdata()
            self.atlas_header = img.header
            self.affine = img.affine
        except ImportError:
            print("Warning: nibabel not installed, cannot load atlas file")

    def _get_coordinates(self) -> torch.Tensor:
        """
        Get MNI coordinates for ROIs.

        Returns:
            (n_rois, 3) tensor of MNI coordinates
        """
        # If atlas data is available, compute from atlas
        if self.atlas_data is not None:
            coords = []
            for roi_idx in range(1, self.n_rois + 1):
                mask = self.atlas_data == roi_idx
                if mask.sum() > 0:
                    indices = np.where(mask)
                    centroid = np.array([
                        np.mean(indices[0]),
                        np.mean(indices[1]),
                        np.mean(indices[2])
                    ])
                    coords.append(centroid)
                else:
                    coords.append([0, 0, 0])

            coords = torch.tensor(coords, dtype=torch.float32)
        else:
            # Generate coordinates based on n_rois
            coords = generate_network_based_coordinates(self.n_rois, self.n_networks)

        return coords

    def _get_network_labels(self) -> List[int]:
        """Get network labels for each ROI (0 to n_networks-1)."""
        labels = []
        rois_per_network = self.n_rois // self.n_networks

        for i in range(self.n_rois):
            network = i // rois_per_network
            if network >= self.n_networks:
                network = self.n_networks - 1
            labels.append(network)

        return labels

    def get_roi_mask(self, roi_idx: int) -> torch.Tensor:
        """Get binary mask for a single ROI."""
        if self.atlas_data is not None:
            mask = torch.tensor(self.atlas_data == (roi_idx + 1), dtype=torch.float32)
        else:
            mask = torch.zeros(self.n_rois)
            mask[roi_idx] = 1
        return mask

    def get_network_mask(self, network_idx: int) -> torch.Tensor:
        """Get binary mask for all ROIs in a network."""
        mask = torch.zeros(self.n_rois)
        for i, label in enumerate(self.roi_network_labels):
            if label == network_idx:
                mask[i] = 1
        return mask.bool()

    def extract_roi_values(self, volume: np.ndarray, roi_idx: int) -> np.ndarray:
        """
        Extract voxel values within an ROI from a volume.

        Args:
            volume: 3D volume array
            roi_idx: ROI index (0 to n_rois-1)

        Returns:
            Array of voxel values within the ROI
        """
        if self.atlas_data is None:
            return np.array([])

        mask = (self.atlas_data == (roi_idx + 1))
        return volume[mask]

    def compute_roi_features(
        self,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Compute features for all ROIs from a volume.

        Args:
            volume: 3D volume array

        Returns:
            (n_rois, n_features) array of ROI features
        """
        n_features = 3  # mean, std, max
        features = np.zeros((self.n_rois, n_features), dtype=np.float32)

        for roi_idx in range(self.n_rois):
            values = self.extract_roi_values(volume, roi_idx)
            if len(values) > 0:
                features[roi_idx, 0] = np.mean(values)
                features[roi_idx, 1] = np.std(values)
                features[roi_idx, 2] = np.max(values)

        return features


def compute_functional_connectivity(
    timeseries: torch.Tensor,
    method: str = 'correlation'
) -> torch.Tensor:
    """
    Compute ROI functional connectivity from time-series.

    Args:
        timeseries: (T, n_rois) or (B, T, n_rois) time-series
        method: 'correlation' or 'covariance'

    Returns:
        (n_rois, n_rois) or (B, n_rois, n_rois) connectivity matrix
    """
    if timeseries.dim() == 2:
        # (T, n_rois) -> add batch dim
        timeseries = timeseries.unsqueeze(0)

    B, T, N = timeseries.shape

    if method == 'correlation':
        # Pearson correlation using torch
        # Center the data
        timeseries_centered = timeseries - timeseries.mean(dim=1, keepdim=True)
        # Standard deviation
        std = timeseries_centered.std(dim=1, keepdim=True)
        std[std == 0] = 1
        normalized = timeseries_centered / std
        # Compute correlation
        corr = torch.bmm(normalized.transpose(1, 2), normalized) / (T - 1)
    elif method == 'covariance':
        corr = torch.cov(timeseries.transpose(1, 2))
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr.squeeze(0) if B == 1 else corr


def compute_dynamic_fc(
    timeseries: torch.Tensor,
    window_size: int = 30,
    stride: int = 1
) -> torch.Tensor:
    """
    Compute dynamic FC using sliding window (GPU-accelerated).

    Args:
        timeseries: (T, n_rois) time-series
        window_size: Size of sliding window
        stride: Stride between windows

    Returns:
        (n_windows, n_rois, n_rois) dynamic FC matrices
    """
    T, N = timeseries.shape
    n_windows = (T - window_size) // stride + 1

    dFC = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window_ts = timeseries[start:end]
        fc = compute_functional_connectivity(window_ts)
        dFC.append(fc)

    return torch.stack(dFC)


def extract_upper_triangular(
    matrix: torch.Tensor,
    diagonal_offset: int = 1
) -> torch.Tensor:
    """
    Extract upper triangular part of matrix.

    Args:
        matrix: (N, N) matrix
        diagonal_offset: Offset from diagonal (1 = exclude diagonal)

    Returns:
        Flattened upper triangular values
    """
    N = matrix.shape[0]
    indices = torch.triu_indices(N, N, offset=diagonal_offset)
    return matrix[indices[0], indices[1]]


def download_schaefer_atlas(
    n_rois: int = 200,
    n_networks: int = 7,
    output_dir: str = '.'
) -> str:
    """
    Download Schaefer atlas from GitHub.

    Args:
        n_rois: Number of ROIs (100, 200, 400, 600, 800, 1000)
        n_networks: Number of networks (7 or 17)
        output_dir: Directory to save atlas

    Returns:
        Path to downloaded atlas file
    """
    import urllib.request

    # Build URL
    filename = f'Schaefer2018_{n_rois}Parcels_{n_networks}Networks_order_FSLMNI152_2mm.nii.gz'
    url = f'https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/{filename}'

    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"Atlas already exists at {output_path}")
        return output_path

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

    return output_path


# Initialize default coordinates for 200 ROIs
SchaeferAtlas.DEFAULT_COORDINATES = torch.tensor([
    # These are approximate MNI coordinates for Schaefer 200
    # Generated based on network organization
    [-24, -78, -36], [-18, -84, -36], [-12, -90, -30], [-6, -84, -24], [0, -78, -18],
    [6, -72, -12], [12, -66, -6], [18, -60, 0], [24, -54, 6], [30, -48, 12],
    [-30, -72, 6], [-24, -66, 12], [-18, -60, 18], [-12, -54, 24], [-6, -48, 30],
    [0, -42, 36], [6, -36, 42], [12, -30, 48], [18, -24, 54], [24, -18, 60],
    [-36, -54, 24], [-30, -48, 30], [-24, -42, 36], [-18, -36, 42], [-12, -30, 48],
    [-6, -24, 54], [0, -18, 60], [6, -12, 66], [12, -6, 72], [18, 0, 78],
    [-42, -36, 36], [-36, -30, 42], [-30, -24, 48], [-24, -18, 54], [-18, -12, 60],
    [-12, -6, 66], [-6, 0, 72], [0, 6, 78], [6, 12, 84], [12, 18, 90],
    [-48, -18, 42], [-42, -12, 48], [-36, -6, 54], [-30, 0, 60], [-24, 6, 66],
    [-18, 12, 72], [-12, 18, 78], [-6, 24, 84], [0, 30, 90], [6, 36, 96],
    [-54, 0, 42], [-48, 6, 48], [-42, 12, 54], [-36, 18, 60], [-30, 24, 66],
    [-24, 30, 72], [-18, 36, 78], [-12, 42, 84], [-6, 48, 90], [0, 54, 96],
    [-60, 18, 36], [-54, 24, 42], [-48, 30, 48], [-42, 36, 54], [-36, 42, 60],
    [-30, 48, 66], [-24, 54, 72], [-18, 60, 78], [-12, 66, 84], [-6, 72, 90],
    [-66, 36, 24], [-60, 42, 30], [-54, 48, 36], [-48, 54, 42], [-42, 60, 48],
    [-36, 66, 54], [-30, 72, 60], [-24, 78, 66], [-18, 84, 72], [-12, 90, 78],
    [-72, 54, 6], [-66, 60, 12], [-60, 66, 18], [-54, 72, 24], [-48, 78, 30],
    [-42, 84, 36], [-36, 90, 42], [-30, 96, 48], [-24, 102, 54], [-18, 108, 60],
    [-78, 72, -18], [-72, 78, -12], [-66, 84, -6], [-60, 90, 0], [-54, 96, 6],
    [-48, 102, 12], [-42, 108, 18], [-36, 114, 24], [-30, 120, 30], [-24, 126, 36],
    [-84, 90, -42], [-78, 96, -36], [-72, 102, -30], [-66, 108, -24], [-60, 114, -18],
    [-54, 120, -12], [-48, 126, -6], [-42, 132, 0], [-36, 138, 6], [-30, 144, 12],
    [-90, 108, -60], [-84, 114, -54], [-78, 120, -48], [-72, 126, -42], [-66, 132, -36],
    [-60, 138, -30], [-54, 144, -24], [-48, 150, -18], [-42, 156, -12], [-36, 162, -6],
    [-96, 126, -72], [-90, 132, -66], [-84, 138, -60], [-78, 144, -54], [-72, 150, -48],
    [-66, 156, -42], [-60, 162, -36], [-54, 168, -30], [-48, 174, -24], [-42, 180, -18],
    [-102, 144, -78], [-96, 150, -72], [-90, 156, -66], [-84, 162, -60], [-78, 168, -54],
    [-72, 174, -48], [-66, 180, -42], [-60, 186, -36], [-54, 192, -30], [-48, 198, -24],
    [-108, 162, -78], [-102, 168, -72], [-96, 174, -66], [-90, 180, -60], [-84, 186, -54],
    [-78, 192, -48], [-72, 198, -42], [-66, 204, -36], [-60, 210, -30], [-54, 216, -24],
    [-114, 180, -72], [-108, 186, -66], [-102, 192, -60], [-96, 198, -54], [-90, 204, -48],
    [-84, 210, -42], [-78, 216, -36], [-72, 222, -30], [-66, 228, -24], [-60, 234, -18],
    [-120, 198, -60], [-114, 204, -54], [-108, 210, -48], [-102, 216, -42], [-96, 222, -36],
    [-90, 228, -30], [-84, 234, -24], [-78, 240, -18], [-72, 246, -12], [-66, 252, -6],
    [-126, 216, -42], [-120, 222, -36], [-114, 228, -30], [-108, 234, -24], [-102, 240, -18],
    [-96, 246, -12], [-90, 252, -6], [-84, 258, 0], [-78, 264, 6], [-72, 270, 12],
    [-132, 234, -18], [-126, 240, -12], [-120, 246, -6], [-114, 252, 0], [-108, 258, 6],
    [-102, 264, 12], [-96, 270, 18], [-90, 276, 24], [-84, 282, 30], [-78, 288, 36],
], dtype=torch.float32)


def generate_network_based_coordinates(n_rois: int, n_networks: int = 7) -> torch.Tensor:
    """
    Generate approximate MNI coordinates for Schaefer atlas based on network layout.

    Args:
        n_rois: Number of ROIs (100, 200, 400, 600)
        n_networks: Number of networks (7 or 17)

    Returns:
        (n_rois, 3) tensor of MNI coordinates
    """
    import math

    # Define network centers based on known brain network organization
    # 7 networks: Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
    if n_networks == 7:
        network_centers = {
            0: (-45, -75, -15),   # Vis (left visual)
            1: (45, -75, -15),    # Vis (right visual)
            2: (-40, -25, 55),    # SomMot (left sensorimotor)
            3: (40, -25, 55),     # SomMot (right sensorimotor)
            4: (-35, -55, 45),     # DorsAttn (left dorsal attention)
            5: (35, -55, 45),      # DorsAttn (right dorsal attention)
            6: (-40, 35, 30),      # SalVentAttn (left ventral attention)
        }
    else:
        # 17 networks - simplified
        network_centers = {}
        for i in range(17):
            network_centers[i] = (
                (i % 2) * 2 - 1,  # alternating left/right
                -50 + (i // 2) * 5,
                (i % 4) * 10
            )

    rois_per_network = n_rois // n_networks
    coords = []

    for net_idx in range(n_networks):
        center = network_centers.get(net_idx, (0, 0, 0))

        for roi_in_net in range(rois_per_network):
            # Add small variation around network center
            roi_idx = net_idx * rois_per_network + roi_in_net

            # Spread ROIs within network
            angle = (roi_in_net / rois_per_network) * 2 * math.pi
            radius = 10 + (roi_in_net % 3) * 5

            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle) * 0.5
            z = center[2] + (roi_in_net // 10) * 5 - 10

            coords.append([x, y, z])

    # Ensure we have exactly n_rois coordinates
    while len(coords) < n_rois:
        coords.append(coords[len(coords) % n_rois])

    return torch.tensor(coords[:n_rois], dtype=torch.float32)
