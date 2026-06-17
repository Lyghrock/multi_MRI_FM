"""
Brain MRI Dataset Classes with Patching Support

Dataset classes for multi-modal brain MRI data with patch extraction.
Loads preprocessed .pt files.
"""

import os
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class BrainMRIDatasetPreprocessed(Dataset):
    """
    Dataset for preprocessed brain MRI data (.pt files) with patches.

    Resolution-aware patch sizes:
        - 1mm MNI (91, 109, 91): voxel patches = 16³
        - 2mm MNI (182, 218, 182): voxel patches = 32³

    Expected file structure (metric mode):
        data_root/
        ├── train/
        │   ├── sub-001_sMRI_patches.pt      # (N_patches, 16/32, 16/32, 16/32)
        │   ├── sub-001_sMRI_roi.pt          # (200, 3)
        │   ├── sub-001_fMRI_time_patches.pt # (N_time_patches, T_patch, 200)
        │   ├── sub-001_fMRI_fc_patches.pt   # (N_fc_patches, P, P)
        │   ├── sub-001_dMRI_FA_patches.pt   # (N_patches, 16/32, 16/32, 16/32)
        │   ├── sub-001_dMRI_MD_patches.pt  # (N_patches, 16/32, 16/32, 16/32)
        │   └── sub-001_dMRI_SC_patches.pt   # (N_patches, P, P)
        └── test/ (optional)

    Expected file structure (raw mode):
        data_root/
        ├── train/
        │   ├── sub-001_sMRI.pt   # (D, H, W)
        │   ├── sub-001_fMRI.pt   # (D, H, W, T)
        │   ├── sub-001_FA.pt    # (D, H, W)
        │   └── sub-001_MD.pt    # (D, H, W)
        └── test/
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        n_rois: int = 200,
        mode: str = 'metric',
        load_modalities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        cache_in_memory: bool = False,
        max_patches_per_modality: Optional[int] = None,
        split_dir: Optional[str] = None,
    ):
        """
        Args:
            data_root: Root directory containing preprocessed data
            split: 'train', 'val', or 'test'
            n_rois: Number of ROIs
            mode: 'metric' (with patches) or 'raw' (volumes only)
            load_modalities: List of modalities to load
            transform: Optional transform to apply
            cache_in_memory: Whether to cache all data in memory
            max_patches_per_modality: Limit number of patches per modality (for memory)
        """
        self.data_root = data_root
        self.split = split
        self.n_rois = n_rois
        self.mode = mode
        self.load_modalities = load_modalities
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.max_patches = max_patches_per_modality

        self.split_dir = split_dir if split_dir is not None else os.path.join(data_root, split)
        self.subject_ids = self._find_subjects()

        self._cache = {} if cache_in_memory else None

    def _find_subjects(self) -> List[str]:
        """Find all subject IDs in the split directory."""
        if not os.path.exists(self.split_dir):
            return []

        subject_ids = set()

        for filename in os.listdir(self.split_dir):
            if filename.endswith('.pt'):
                parts = filename.split('_')
                if parts and parts[0].startswith('sub-'):
                    subject_ids.add(parts[0])

        return sorted(list(subject_ids))

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _get_file_path(self, subject_id: str, modality_suffix: str) -> Optional[str]:
        """Get file path for a subject's modality."""
        filename = f'{subject_id}_{modality_suffix}.pt'
        path = os.path.join(self.split_dir, filename)
        return path if os.path.exists(path) else None

    def _load_tensor(self, path: str) -> Optional[torch.Tensor]:
        """Load a tensor from file."""
        if self._cache is not None and path in self._cache:
            return self._cache[path]

        try:
            tensor = torch.load(path, map_location='cpu')
            if tensor.dtype != torch.float16 and tensor.dtype != torch.float32:
                tensor = tensor.half()

            if self._cache is not None:
                self._cache[path] = tensor

            return tensor
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single subject's data with patches.

        Returns:
            Dictionary with:
                - subject_id: str
                - sMRI:
                    - patches: (N_voxel, 32, 32, 32)
                    - roi: (200, 3)
                - fMRI:
                    - time_patches: (N_time, T_patch, 200)
                    - fc_patches: (N_fc, P, P)
                - dMRI:
                    - FA_patches: (N_voxel, 32, 32, 32)
                    - MD_patches: (N_voxel, 32, 32, 32)
                    - SC_patches: (N_patches, P, P)
        """
        subject_id = self.subject_ids[idx]

        if self.load_modalities is None:
            modalities = ['sMRI', 'fMRI', 'dMRI']
        else:
            modalities = self.load_modalities

        data = {'subject_id': subject_id, 'mode': self.mode}

        if self.mode == 'metric':
            # ========== sMRI ==========
            if 'sMRI' in modalities:
                sMRI_data = {}

                # Voxel patches
                patches_path = self._get_file_path(subject_id, 'sMRI_patches')
                if patches_path:
                    patches = self._load_tensor(patches_path)
                    if patches is not None and self.max_patches:
                        # Random sample or first N
                        if patches.shape[0] > self.max_patches:
                            indices = torch.randperm(patches.shape[0])[:self.max_patches]
                            patches = patches[indices]
                    sMRI_data['patches'] = patches

                # ROI features
                roi_path = self._get_file_path(subject_id, 'sMRI_roi')
                if roi_path:
                    roi = self._load_tensor(roi_path)
                    sMRI_data['roi'] = roi

                if sMRI_data:
                    data['sMRI'] = sMRI_data

            # ========== fMRI ==========
            if 'fMRI' in modalities:
                fMRI_data = {}

                # Time patches
                time_path = self._get_file_path(subject_id, 'fMRI_time_patches')
                if time_path:
                    time_patches = self._load_tensor(time_path)
                    if time_patches is not None and self.max_patches:
                        if time_patches.shape[0] > self.max_patches:
                            indices = torch.randperm(time_patches.shape[0])[:self.max_patches]
                            time_patches = time_patches[indices]
                    fMRI_data['time_patches'] = time_patches

                # FC patches
                fc_path = self._get_file_path(subject_id, 'fMRI_fc_patches')
                if fc_path:
                    fc_patches = self._load_tensor(fc_path)
                    if fc_patches is not None and self.max_patches:
                        if fc_patches.shape[0] > self.max_patches:
                            indices = torch.randperm(fc_patches.shape[0])[:self.max_patches]
                            fc_patches = fc_patches[indices]
                    fMRI_data['fc_patches'] = fc_patches

                if fMRI_data:
                    data['fMRI'] = fMRI_data

            # ========== dMRI ==========
            if 'dMRI' in modalities:
                dMRI_data = {}

                # FA patches
                fa_path = self._get_file_path(subject_id, 'dMRI_FA_patches')
                if fa_path:
                    fa_patches = self._load_tensor(fa_path)
                    if fa_patches is not None and self.max_patches:
                        if fa_patches.shape[0] > self.max_patches:
                            indices = torch.randperm(fa_patches.shape[0])[:self.max_patches]
                            fa_patches = fa_patches[indices]
                    dMRI_data['FA_patches'] = fa_patches

                # MD patches
                md_path = self._get_file_path(subject_id, 'dMRI_MD_patches')
                if md_path:
                    md_patches = self._load_tensor(md_path)
                    if md_patches is not None and self.max_patches:
                        if md_patches.shape[0] > self.max_patches:
                            indices = torch.randperm(md_patches.shape[0])[:self.max_patches]
                            md_patches = md_patches[indices]
                    dMRI_data['MD_patches'] = md_patches

                # SC matrix (preferred for model input and MAE target)
                sc_matrix_path = (
                    self._get_file_path(subject_id, 'dMRI_SC_matrix')
                    or self._get_file_path(subject_id, 'dMRI_SC')
                    or self._get_file_path(subject_id, 'dMRI_SC_placeholder')
                )
                if sc_matrix_path:
                    sc_matrix = self._load_tensor(sc_matrix_path)
                    if sc_matrix is not None:
                        dMRI_data['SC_matrix'] = sc_matrix

                # SC patches (kept for patch-based extensions)
                sc_path = self._get_file_path(subject_id, 'dMRI_SC_patches')
                if sc_path:
                    sc_patches = self._load_tensor(sc_path)
                    if sc_patches is not None and self.max_patches:
                        if sc_patches.shape[0] > self.max_patches:
                            indices = torch.randperm(sc_patches.shape[0])[:self.max_patches]
                            sc_patches = sc_patches[indices]
                    dMRI_data['SC_patches'] = sc_patches

                if dMRI_data:
                    data['dMRI'] = dMRI_data

        else:
            # Raw mode - just load volumes
            if 'sMRI' in modalities:
                path = self._get_file_path(subject_id, 'sMRI')
                if path:
                    data['sMRI'] = self._load_tensor(path)

            if 'fMRI' in modalities:
                path = self._get_file_path(subject_id, 'fMRI')
                if path:
                    data['fMRI'] = self._load_tensor(path)

            if 'dMRI' in modalities:
                path = self._get_file_path(subject_id, 'FA')
                if path:
                    data['FA'] = self._load_tensor(path)

                path = self._get_file_path(subject_id, 'MD')
                if path:
                    data['MD'] = self._load_tensor(path)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'num_subjects': len(self.subject_ids),
            'n_rois': self.n_rois,
            'split': self.split,
            'mode': self.mode,
        }


class BrainMRIDataset(Dataset):
    """
    Dataset for raw (non-preprocessed) brain MRI data.

    This loads raw NIfTI files directly.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        n_rois: int = 200,
        load_modalities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        split_dir: Optional[str] = None,
        **kwargs
    ):
        self.data_root = data_root
        self.split = split
        self.n_rois = n_rois
        self.load_modalities = load_modalities
        self.transform = transform

        self.split_dir = split_dir if split_dir is not None else os.path.join(data_root, split)
        self.subject_ids = self._find_subjects()

    def _find_subjects(self) -> List[str]:
        if not os.path.exists(self.split_dir):
            return []

        subject_ids = set()

        for filename in os.listdir(self.split_dir):
            if filename.endswith(('.nii', '.nii.gz')):
                parts = filename.split('_')
                if parts and parts[0].startswith('sub-'):
                    subject_ids.add(parts[0])

        return sorted(list(subject_ids))

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict:
        subject_id = self.subject_ids[idx]

        if self.load_modalities is None:
            modalities = ['sMRI', 'fMRI', 'dMRI']
        else:
            modalities = self.load_modalities

        data = {'subject_id': subject_id}

        for modality in modalities:
            path = self._get_file_path(subject_id, modality)
            if path is not None:
                try:
                    import nibabel as nib
                    img = nib.load(path)
                    data[modality] = torch.from_numpy(img.get_fdata())
                except:
                    pass

        if self.transform is not None:
            data = self.transform(data)

        return data

    def _get_file_path(self, subject_id: str, modality: str) -> Optional[str]:
        patterns = {
            'sMRI': ['_T1w.nii', '_T1w.nii.gz'],
            'fMRI': ['_bold.nii', '_bold.nii.gz', '_task-rest_bold.nii.gz'],
            'dMRI': ['_FA.nii', '_FA.nii.gz', '_dMRI.nii.gz'],
        }

        for ext in patterns.get(modality, []):
            path = os.path.join(self.split_dir, f'{subject_id}{ext}')
            if os.path.exists(path):
                return path
        return None


class PairedBrainMRIDataset(Dataset):
    """Dataset for paired brain MRI data (multi-site studies)."""

    def __init__(
        self,
        data_root1: str,
        data_root2: str,
        split: str = 'train',
        n_rois: int = 200,
        transform: Optional[Callable] = None
    ):
        self.dataset1 = BrainMRIDatasetPreprocessed(
            data_root1, split, n_rois
        )
        self.dataset2 = BrainMRIDatasetPreprocessed(
            data_root2, split, n_rois
        )

        subjects1 = set(self.dataset1.subject_ids)
        subjects2 = set(self.dataset2.subject_ids)
        common_subjects = sorted(subjects1 & subjects2)

        self.dataset1.subject_ids = common_subjects
        self.dataset2.subject_ids = common_subjects

        self.subject_ids = common_subjects
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict:
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]

        data = {
            'subject_id': data1['subject_id'],
            'data1': data1,
            'data2': data2,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


def collate_brain_mri(batch: List[Dict]) -> Dict:
    """
    Collate function for BrainMRIDataset.

    For patched data, returns list of samples.
    """
    return batch[0] if len(batch) == 1 else batch


class BrainMRIDatasetMemory(Dataset):
    """
    In-memory dataset for fast training.

    Loads all data into memory at initialization.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        n_rois: int = 200,
        mode: str = 'metric',
        device: str = 'cpu'
    ):
        print(f"Loading data into memory from {data_root}/{split}...")

        base_dataset = BrainMRIDatasetPreprocessed(
            data_root=data_root,
            split=split,
            n_rois=n_rois,
            mode=mode,
            cache_in_memory=False
        )

        self.data = []
        for i in range(len(base_dataset)):
            sample = base_dataset[i]

            for key in ['sMRI', 'fMRI', 'dMRI']:
                if key in sample and isinstance(sample[key], dict):
                    for subkey in sample[key]:
                        if sample[key][subkey] is not None:
                            sample[key][subkey] = sample[key][subkey].to(device)

            self.data.append(sample)

        print(f"Loaded {len(self.data)} subjects into memory")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


class PatchedSampleCollator:
    """
    Collator that handles variable-sized patches.

    Supports both 1mm (16³) and 2mm (32³) voxel patches.
    Can sample a fixed number of patches per batch.
    """

    def __init__(
        self,
        max_voxel_patches: int = 16,
        max_fc_patches: int = 32,
        max_time_patches: int = 8,
    ):
        self.max_voxel_patches = max_voxel_patches
        self.max_fc_patches = max_fc_patches
        self.max_time_patches = max_time_patches

    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate a batch of samples with patch sampling."""
        if len(batch) == 1:
            return self._process_single(batch[0])

        # Sample patches for each sample in batch
        collated = {
            'subject_ids': [s['subject_id'] for s in batch],
            'mode': batch[0]['mode']
        }

        for modality in ['sMRI', 'fMRI', 'dMRI']:
            if modality in batch[0] and batch[0][modality] is not None:
                collated[modality] = {}
                mod_data = batch[0][modality]

                if 'patches' in mod_data and mod_data['patches'] is not None:
                    collated[modality]['patches'] = self._sample_voxel_patches(
                        [s[modality]['patches'] for s in batch if modality in s and 'patches' in s[modality]]
                    )

                if 'roi' in mod_data and mod_data['roi'] is not None:
                    collated[modality]['roi'] = torch.stack([
                        s[modality]['roi'] for s in batch
                        if modality in s and 'roi' in s[modality] and s[modality]['roi'] is not None
                    ])

                if 'time_patches' in mod_data and mod_data['time_patches'] is not None:
                    collated[modality]['time_patches'] = self._sample_time_patches(
                        [s[modality]['time_patches'] for s in batch if modality in s and 'time_patches' in s[modality]]
                    )

                if 'fc_patches' in mod_data and mod_data['fc_patches'] is not None:
                    collated[modality]['fc_patches'] = self._sample_fc_patches(
                        [s[modality]['fc_patches'] for s in batch if modality in s and 'fc_patches' in s[modality]]
                    )

                if 'SC_matrix' in mod_data and mod_data['SC_matrix'] is not None:
                    collated[modality]['SC_matrix'] = torch.stack([
                        s[modality]['SC_matrix'] for s in batch
                        if modality in s and 'SC_matrix' in s[modality] and s[modality]['SC_matrix'] is not None
                    ])

                if 'SC_patches' in mod_data and mod_data['SC_patches'] is not None:
                    collated[modality]['SC_patches'] = self._sample_fc_patches(
                        [s[modality]['SC_patches'] for s in batch if modality in s and 'SC_patches' in s[modality]]
                    )

        return collated

    def _sample_voxel_patches(self, patch_lists: List[torch.Tensor]) -> torch.Tensor:
        """Sample a fixed number of voxel patches from each sample."""
        sampled = []
        for patches in patch_lists:
            if patches is None or patches.shape[0] == 0:
                # Detect patch size from first valid sample or use default 16³
                patch_size = 16  # default for 1mm
                sampled.append(torch.zeros(self.max_voxel_patches, patch_size, patch_size, patch_size,
                                          dtype=torch.float16))
                continue

            # Detect actual patch size from first dimension
            patch_size = patches.shape[1]  # Could be 16 (1mm) or 32 (2mm)
            n = min(patches.shape[0], self.max_voxel_patches)
            indices = torch.randperm(patches.shape[0])[:n]
            sampled.append(patches[indices])

            if n < self.max_voxel_patches:
                pad_size = self.max_voxel_patches - n
                pad = torch.zeros(pad_size, patch_size, patch_size, patch_size, dtype=torch.float16)
                sampled[-1] = torch.cat([sampled[-1], pad], dim=0)

        return torch.stack(sampled)

    def _sample_time_patches(self, patch_lists: List[torch.Tensor]) -> torch.Tensor:
        """Sample a fixed number of time patches from each sample."""
        sampled = []
        for patches in patch_lists:
            if patches is None or patches.shape[0] == 0:
                sampled.append(torch.zeros(self.max_time_patches, 50, 200,
                                          dtype=torch.float16))
                continue

            n = min(patches.shape[0], self.max_time_patches)
            indices = torch.randperm(patches.shape[0])[:n]
            sampled.append(patches[indices])

            if n < self.max_time_patches:
                pad_size = self.max_time_patches - n
                pad = torch.zeros(pad_size, *patches.shape[1:], dtype=torch.float16)
                sampled[-1] = torch.cat([sampled[-1], pad], dim=0)

        return torch.stack(sampled)

    def _sample_fc_patches(self, patch_lists: List[torch.Tensor]) -> torch.Tensor:
        """Sample a fixed number of FC patches from each sample."""
        sampled = []
        for patches in patch_lists:
            if patches is None or patches.shape[0] == 0:
                sampled.append(torch.zeros(self.max_fc_patches, 20, 20,
                                          dtype=torch.float16))
                continue

            n = min(patches.shape[0], self.max_fc_patches)
            indices = torch.randperm(patches.shape[0])[:n]
            sampled.append(patches[indices])

            if n < self.max_fc_patches:
                pad_size = self.max_fc_patches - n
                pad = torch.zeros(pad_size, *patches.shape[1:], dtype=torch.float16)
                sampled[-1] = torch.cat([sampled[-1], pad], dim=0)

        return torch.stack(sampled)

    def _process_single(self, sample: Dict) -> Dict:
        """Process a single sample."""
        return sample
