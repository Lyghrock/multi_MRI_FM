"""
Data package for Multi-Modal Brain MRI Dataset

Atlas:
    Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

Download:
    wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

Data Format:
    Processed data is saved as .pt (float16) with patches:
        - sMRI: voxel patches (N, 32, 32, 32) + ROI features (200, 3)
        - fMRI: time patches (N, T_patch, 200) + FC patches (N, P, P)
        - dMRI: FA/MD patches (N, 32, 32, 32) + SC patches (N, P, P)
"""

from .brain_dataset import (
    BrainMRIDataset,
    BrainMRIDatasetPreprocessed,
    PairedBrainMRIDataset,
    BrainMRIDatasetMemory,
    PatchedSampleCollator,
    collate_brain_mri,
)
from .data_loader import create_dataloaders, create_distributed_dataloaders
from .atlas import SchaeferAtlas, compute_functional_connectivity, download_schaefer_atlas

__all__ = [
    # Dataset
    'BrainMRIDataset',
    'BrainMRIDatasetPreprocessed',
    'PairedBrainMRIDataset',
    'BrainMRIDatasetMemory',
    'PatchedSampleCollator',
    'collate_brain_mri',
    # DataLoader
    'create_dataloaders',
    'create_distributed_dataloaders',
    # Atlas
    'SchaeferAtlas',
    'compute_functional_connectivity',
    'download_schaefer_atlas',
]
