#!/usr/bin/env python3
"""
Data Preprocessing Pipeline with Patching

Multi-process preprocessing with patch extraction for multi-modal brain MRI.
Supports CPU parallelization for large datasets.

Patch Design:
    - sMRI: VBM voxel-patch + ROI features
    - fMRI: Time-patch ROI time-series + FC patches (or dFC)
    - dMRI: FA voxel-patch + FA/MD voxel-patch + SC patches

Usage:
    # Sequential (single process)
    python -m data.preprocess --input_dir /path/to/raw --output_dir /path/to/output

    # Parallel (multi-process)
    python -m data.preprocess --input_dir /path/to/raw --output_dir /path/to/output --num_workers 4
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from glob import glob
from functools import partial
import time

import numpy as np
import torch


# Optional: nibabel for NIfTI files
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed. Raw preprocessing will be limited.")


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess multi-modal brain MRI data')

    # Paths
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Explicit output split name. Defaults to auto-detecting train/val/test from input_dir.'
    )
    parser.add_argument(
        '--flat_output',
        action='store_true',
        help='Save files directly under output_dir instead of output_dir/{mode}/{split}.'
    )

    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        default='metric',
        choices=['metric', 'raw'],
        help='Preprocessing mode: metric (ROI features + patches) or raw (3D volumes)'
    )

    # Parallelism
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, use 0 for auto)'
    )

    # Atlas
    parser.add_argument('--atlas_path', type=str, default=None, help='Atlas path')
    parser.add_argument('--n_rois', type=int, default=200, help='Number of ROIs')

    # Patch parameters (will be auto-scaled based on volume shape)
    parser.add_argument('--voxel_patch_size', type=int, default=None,
                       help='3D voxel patch size (auto: 16 for 1mm, 32 for 2mm)')
    parser.add_argument('--voxel_patch_stride', type=int, default=None,
                       help='3D voxel patch stride (auto: 8 for 1mm, 16 for 2mm)')
    parser.add_argument('--time_patch_length', type=int, default=50,
                       help='fMRI time-patch length')
    parser.add_argument('--time_patch_stride', type=int, default=25,
                       help='fMRI time-patch stride')
    parser.add_argument('--fc_patch_size', type=int, default=20,
                       help='FC/SC matrix patch size')
    parser.add_argument('--fc_patch_stride', type=int, default=10,
                       help='FC/SC matrix patch stride')
    parser.add_argument('--space', type=str, default=None,
                       help='MNI space: 1mm or 2mm (auto-detect if None)')

    return parser.parse_args()


def get_resolution_config(volume_shape: Tuple[int, int, int], args) -> Dict:
    """
    Auto-detect resolution and return appropriate patch config.

    MNI volumes:
        - 1mm: shape (91, 109, 91)
        - 2mm: shape (182, 218, 182)

    Returns patch config dict with auto-scaled values.
    """
    D, H, W = volume_shape

    # Auto-detect resolution based on volume size
    # 1mm MNI: ~91-109 in any dimension
    # 2mm MNI: ~182-218 in any dimension (roughly 2x of 1mm)
    is_1mm = D < 130 and H < 130 and W < 130

    if args.space:
        # Override with explicit setting
        is_1mm = '1mm' in args.space.lower()

    # Scale patch size based on resolution
    # Target: ~16³ voxels per patch regardless of resolution
    if is_1mm:
        default_voxel_size = 16
        default_voxel_stride = 8
    else:
        default_voxel_size = 32
        default_voxel_stride = 16

    # Use args if provided, otherwise use defaults
    voxel_size = args.voxel_patch_size if args.voxel_patch_size else default_voxel_size
    voxel_stride = args.voxel_patch_stride if args.voxel_patch_stride else default_voxel_stride

    return {
        'is_1mm': is_1mm,
        'volume_shape': volume_shape,
        'voxel_patch_size': voxel_size,
        'voxel_patch_stride': voxel_stride,
        'time_patch_length': args.time_patch_length,
        'time_patch_stride': args.time_patch_stride,
        'fc_patch_size': args.fc_patch_size,
        'fc_patch_stride': args.fc_patch_stride,
    }


class PatchExtractor:
    """Extract patches from various data types."""

    @staticmethod
    def extract_3d_patches(
        volume: np.ndarray,
        patch_size: int = 32,
        stride: int = 16
    ) -> List[np.ndarray]:
        """
        Extract 3D patches from a volume.

        Args:
            volume: 3D array (D, H, W)
            patch_size: Size of cubic patch
            stride: Stride between patches

        Returns:
            List of 3D patches, each (patch_size, patch_size, patch_size)
        """
        D, H, W = volume.shape
        patches = []

        for d in range(0, D - patch_size + 1, stride):
            for h in range(0, H - patch_size + 1, stride):
                for w in range(0, W - patch_size + 1, stride):
                    patch = volume[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                    patches.append(patch)

        return patches

    @staticmethod
    def extract_2d_matrix_patches(
        matrix: np.ndarray,
        patch_size: int = 20,
        stride: int = 10
    ) -> List[np.ndarray]:
        """
        Extract patches from a 2D matrix.

        Args:
            matrix: 2D array (N, N), should be symmetric
            patch_size: Size of patch
            stride: Stride between patches

        Returns:
            List of 2D patches, each (patch_size, patch_size)
        """
        N = matrix.shape[0]
        patches = []

        # Extract from upper triangle (avoiding diagonal)
        for i in range(0, N - patch_size + 1, stride):
            for j in range(i, N - patch_size + 1, stride):
                patch = matrix[i:i+patch_size, j:j+patch_size]
                patches.append(patch)

        return patches

    @staticmethod
    def extract_time_patches(
        timeseries: np.ndarray,
        patch_length: int = 50,
        stride: int = 25
    ) -> List[np.ndarray]:
        """
        Extract time patches from time-series.

        Args:
            timeseries: 2D array (T, N_rois)
            patch_length: Number of time points per patch
            stride: Stride between patches

        Returns:
            List of time patches, each (patch_length, N_rois)
        """
        T, N = timeseries.shape
        patches = []

        for t in range(0, T - patch_length + 1, stride):
            patch = timeseries[t:t+patch_length, :]
            patches.append(patch)

        return patches


class Atlas:
    """Atlas class for ROI extraction."""

    def __init__(self, atlas_path: str, n_rois: int = 200):
        self.n_rois = n_rois
        self.atlas_data = None
        self.affine = None

        if atlas_path and os.path.exists(atlas_path) and HAS_NIBABEL:
            img = nib.load(atlas_path)
            self.atlas_data = img.get_fdata()
            self.affine = img.affine

    def extract_roi_features(self, volume: np.ndarray) -> torch.Tensor:
        """Extract ROI features from volume."""
        features = np.zeros((self.n_rois, 3), dtype=np.float32)

        if self.atlas_data is None:
            features[:, 0] = np.mean(volume)
            features[:, 1] = np.sum(volume > 0) / self.n_rois
            features[:, 2] = np.max(volume)
            return torch.from_numpy(features).half()

        for roi_idx in range(1, self.n_rois + 1):
            mask = self.atlas_data == roi_idx
            if mask.sum() > 0:
                values = volume[mask]
                features[roi_idx - 1, 0] = np.mean(values)
                features[roi_idx - 1, 1] = np.sum(mask)
                features[roi_idx - 1, 2] = np.max(values)

        return torch.from_numpy(features).half()

    def extract_roi_timeseries(self, volume4d: np.ndarray) -> torch.Tensor:
        """Extract ROI time-series from 4D volume."""
        T = volume4d.shape[-1]
        timeseries = np.zeros((T, self.n_rois), dtype=np.float32)

        if self.atlas_data is None:
            return torch.zeros(T, self.n_rois, dtype=torch.float16)

        for roi_idx in range(1, self.n_rois + 1):
            mask = self.atlas_data == roi_idx
            if mask.sum() > 0:
                roi_data = volume4d[mask, :]
                timeseries[:, roi_idx - 1] = np.mean(roi_data, axis=0)

        # Z-score
        mean = timeseries.mean(axis=0, keepdims=True)
        std = timeseries.std(axis=0, keepdims=True)
        std[std == 0] = 1
        timeseries = (timeseries - mean) / std

        return torch.from_numpy(timeseries).half()


def get_subject_files(input_dir: str) -> dict:
    """Get files for each subject."""
    subjects = {}

    # sMRI
    for pattern in ['sub-*_T1w.nii.gz', 'sub-*_T1w.nii']:
        for f in glob(os.path.join(input_dir, pattern)):
            sid = os.path.basename(f).split('_')[0]
            subjects.setdefault(sid, {})['sMRI'] = f

    # fMRI
    for pattern in ['sub-*_bold.nii.gz', 'sub-*_bold.nii', 'sub-*_task-rest_bold.nii.gz']:
        for f in glob(os.path.join(input_dir, pattern)):
            sid = os.path.basename(f).split('_')[0]
            subjects.setdefault(sid, {})['fMRI'] = f

    # dMRI - FA and/or MD
    for pattern in ['sub-*_FA.nii.gz', 'sub-*_FA.nii']:
        for f in glob(os.path.join(input_dir, pattern)):
            sid = os.path.basename(f).split('_')[0]
            subjects.setdefault(sid, {})['FA'] = f

    for pattern in ['sub-*_MD.nii.gz', 'sub-*_MD.nii']:
        for f in glob(os.path.join(input_dir, pattern)):
            sid = os.path.basename(f).split('_')[0]
            subjects.setdefault(sid, {})['MD'] = f

    return subjects


def preprocess_subject_metric(
    subject_id: str,
    files: dict,
    atlas: Atlas,
    output_dir: str,
    patch_config: Dict
) -> dict:
    """
    Preprocess a single subject with patching.

    Args:
        patch_config: Dict with keys:
            - is_1mm: bool
            - voxel_patch_size: int
            - voxel_patch_stride: int
            - time_patch_length: int
            - time_patch_stride: int
            - fc_patch_size: int
            - fc_patch_stride: int

    Outputs:
        {subject_id}_sMRI_patches.pt   - Voxel patches (16³ for 1mm, 32³ for 2mm)
        {subject_id}_sMRI_roi.pt       - ROI features (200, 3)
        {subject_id}_fMRI_time_patches.pt - Time patches
        {subject_id}_fMRI_fc_patches.pt  - FC patches
        {subject_id}_dMRI_FA_patches.pt  - FA voxel patches
        {subject_id}_dMRI_MD_patches.pt  - MD voxel patches
        {subject_id}_dMRI_SC_patches.pt  - SC patches
    """
    result = {'subject_id': subject_id, 'success': True, 'errors': []}

    try:
        os.makedirs(output_dir, exist_ok=True)

        voxel_size = patch_config['voxel_patch_size']
        voxel_stride = patch_config['voxel_patch_stride']

        # ========== sMRI ==========
        if 'sMRI' in files and HAS_NIBABEL:
            img = nib.load(files['sMRI'])
            data = img.get_fdata()  # (D, H, W)

            # Voxel patches
            voxel_patches = PatchExtractor.extract_3d_patches(
                data,
                patch_size=voxel_size,
                stride=voxel_stride
            )
            voxel_patches_tensor = torch.stack([
                torch.from_numpy(p).half() for p in voxel_patches
            ]) if voxel_patches else torch.zeros(0, voxel_size, voxel_size, voxel_size,
                                                  dtype=torch.float16)
            torch.save(voxel_patches_tensor,
                      os.path.join(output_dir, f'{subject_id}_sMRI_patches.pt'))

            # ROI features
            roi_features = atlas.extract_roi_features(data)
            torch.save(roi_features,
                      os.path.join(output_dir, f'{subject_id}_sMRI_roi.pt'))

        # ========== fMRI ==========
        if 'fMRI' in files and HAS_NIBABEL:
            img = nib.load(files['fMRI'])
            data = img.get_fdata()  # (D, H, W, T)

            # Extract ROI time-series
            ts = atlas.extract_roi_timeseries(data)  # (T, 200)
            ts_np = ts.cpu().numpy()

            # Time patches
            time_patches = PatchExtractor.extract_time_patches(
                ts_np,
                patch_length=patch_config['time_patch_length'],
                stride=patch_config['time_patch_stride']
            )
            time_patches_tensor = torch.stack([
                torch.from_numpy(p).half() for p in time_patches
            ]) if time_patches else torch.zeros(0, patch_config['time_patch_length'], atlas.n_rois,
                                                 dtype=torch.float16)
            torch.save(time_patches_tensor,
                      os.path.join(output_dir, f'{subject_id}_fMRI_time_patches.pt'))

            # Static ROI FC matrix: corrcoef expects variables in rows.
            fc = torch.corrcoef(ts.float().T)  # (n_rois, n_rois)
            fc = torch.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            fc = fc.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
            fc_z = torch.atanh(fc.float())
            fc_z = fc_z - torch.diag(torch.diag(fc_z))
            fc_np = fc_z.cpu().numpy()

            # FC patches
            fc_patches = PatchExtractor.extract_2d_matrix_patches(
                fc_np,
                patch_size=patch_config['fc_patch_size'],
                stride=patch_config['fc_patch_stride']
            )
            fc_patches_tensor = torch.stack([
                torch.from_numpy(p).half() for p in fc_patches
            ]) if fc_patches else torch.zeros(0, patch_config['fc_patch_size'],
                                               patch_config['fc_patch_size'], dtype=torch.float16)
            torch.save(fc_patches_tensor,
                      os.path.join(output_dir, f'{subject_id}_fMRI_fc_patches.pt'))

        # ========== dMRI ==========
        # FA
        if 'FA' in files and HAS_NIBABEL:
            img = nib.load(files['FA'])
            data = img.get_fdata()  # (D, H, W)

            voxel_patches = PatchExtractor.extract_3d_patches(
                data,
                patch_size=voxel_size,
                stride=voxel_stride
            )
            voxel_patches_tensor = torch.stack([
                torch.from_numpy(p).half() for p in voxel_patches
            ]) if voxel_patches else torch.zeros(0, voxel_size, voxel_size, voxel_size,
                                                  dtype=torch.float16)
            torch.save(voxel_patches_tensor,
                      os.path.join(output_dir, f'{subject_id}_dMRI_FA_patches.pt'))

        # MD
        if 'MD' in files and HAS_NIBABEL:
            img = nib.load(files['MD'])
            data = img.get_fdata()  # (D, H, W)

            voxel_patches = PatchExtractor.extract_3d_patches(
                data,
                patch_size=voxel_size,
                stride=voxel_stride
            )
            voxel_patches_tensor = torch.stack([
                torch.from_numpy(p).half() for p in voxel_patches
            ]) if voxel_patches else torch.zeros(0, voxel_size, voxel_size, voxel_size,
                                                  dtype=torch.float16)
            torch.save(voxel_patches_tensor,
                      os.path.join(output_dir, f'{subject_id}_dMRI_MD_patches.pt'))

        # SC matrix placeholder (in real implementation, comes from tractography)
        if ('FA' in files or 'MD' in files) and HAS_NIBABEL:
            # Placeholder: real SC from MRtrix3 tractography
            sc_placeholder = torch.zeros(atlas.n_rois, atlas.n_rois, dtype=torch.float16)
            torch.save(sc_placeholder, os.path.join(output_dir, f'{subject_id}_dMRI_SC_matrix.pt'))
            torch.save(sc_placeholder, os.path.join(output_dir, f'{subject_id}_dMRI_SC_placeholder.pt'))

            sc_patches = PatchExtractor.extract_2d_matrix_patches(
                sc_placeholder.cpu().numpy(),
                patch_size=patch_config['fc_patch_size'],
                stride=patch_config['fc_patch_stride']
            )
            sc_patches_tensor = torch.stack([
                torch.from_numpy(p).half() for p in sc_patches
            ]) if sc_patches else torch.zeros(
                0,
                patch_config['fc_patch_size'],
                patch_config['fc_patch_size'],
                dtype=torch.float16
            )
            torch.save(sc_patches_tensor,
                      os.path.join(output_dir, f'{subject_id}_dMRI_SC_patches.pt'))

        result['success'] = True

    except Exception as e:
        result['success'] = False
        result['errors'].append(str(e))
        import traceback
        traceback.print_exc()

    return result


def preprocess_subject_raw(
    subject_id: str,
    files: dict,
    output_dir: str
) -> dict:
    """Preprocess raw data (no patching, just resampling)."""
    result = {'subject_id': subject_id, 'success': True, 'errors': []}

    try:
        os.makedirs(output_dir, exist_ok=True)

        if 'sMRI' in files and HAS_NIBABEL:
            img = nib.load(files['sMRI'])
            data = img.get_fdata().astype(np.float32)
            tensor = torch.from_numpy(data).half()
            torch.save(tensor, os.path.join(output_dir, f'{subject_id}_sMRI.pt'))

        if 'fMRI' in files and HAS_NIBABEL:
            img = nib.load(files['fMRI'])
            data = img.get_fdata().astype(np.float32)
            tensor = torch.from_numpy(data).half()
            torch.save(tensor, os.path.join(output_dir, f'{subject_id}_fMRI.pt'))

        if 'FA' in files and HAS_NIBABEL:
            img = nib.load(files['FA'])
            data = img.get_fdata().astype(np.float32)
            tensor = torch.from_numpy(data).half()
            torch.save(tensor, os.path.join(output_dir, f'{subject_id}_FA.pt'))

        if 'MD' in files and HAS_NIBABEL:
            img = nib.load(files['MD'])
            data = img.get_fdata().astype(np.float32)
            tensor = torch.from_numpy(data).half()
            torch.save(tensor, os.path.join(output_dir, f'{subject_id}_MD.pt'))

    except Exception as e:
        result['success'] = False
        result['errors'].append(str(e))

    return result


def worker_fn(args_tuple):
    """Worker function for multiprocessing."""
    args, subject_id, files, output_dir, patch_config = args_tuple

    atlas = Atlas(args.atlas_path, args.n_rois)

    if args.mode == 'metric':
        return preprocess_subject_metric(subject_id, files, atlas, output_dir, patch_config)
    else:
        return preprocess_subject_raw(subject_id, files, output_dir)


def main():
    args = parse_args()

    # Auto-detect workers
    if args.num_workers == 0:
        args.num_workers = mp.cpu_count()

    print("=" * 60)
    print("Brain MRI Preprocessing with Patching")
    print("=" * 60)
    print(f"Input:       {args.input_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Mode:        {args.mode}")
    print(f"Atlas:       {args.atlas_path}")
    print(f"Workers:     {args.num_workers}")
    print(f"Format:      .pt (float16)")

    # Determine split name
    split_name = args.split
    if split_name is None:
        split_name = 'train'
        if 'val' in args.input_dir.lower():
            split_name = 'val'
        elif 'test' in args.input_dir.lower():
            split_name = 'test'

    output_base = args.output_dir if args.flat_output else os.path.join(args.output_dir, args.mode, split_name)

    # Get all subjects
    subjects = get_subject_files(args.input_dir)
    subject_ids = sorted(subjects.keys())

    if not subject_ids:
        print("No subjects found!")
        return

    # Auto-detect resolution from first subject's sMRI file
    # We'll detect from the actual volume shape
    first_sMRI = None
    for sid, files in subjects.items():
        if 'sMRI' in files and HAS_NIBABEL:
            first_sMRI = files['sMRI']
            break

    if first_sMRI and args.mode == 'metric':
        # Load first volume to detect shape
        img = nib.load(first_sMRI)
        volume_shape = img.shape[:3]  # (D, H, W)
        patch_config = get_resolution_config(volume_shape, args)

        print("-" * 60)
        print(f"Auto-detected Resolution:")
        print(f"  Volume shape: {volume_shape}")
        print(f"  Resolution:   {'1mm MNI' if patch_config['is_1mm'] else '2mm MNI'}")
        print("Patch Settings (auto-scaled):")
        print(f"  Voxel patch size:  {patch_config['voxel_patch_size']}³")
        print(f"  Voxel patch stride:{patch_config['voxel_patch_stride']}")
        print(f"  Time patch length: {patch_config['time_patch_length']}")
        print(f"  Time patch stride: {patch_config['time_patch_stride']}")
        print(f"  FC patch size:     {patch_config['fc_patch_size']}²")
        print(f"  FC patch stride:   {patch_config['fc_patch_stride']}")
        print("=" * 60)
    else:
        # Use args directly for non-metric mode or no sMRI
        patch_config = {
            'voxel_patch_size': args.voxel_patch_size or 32,
            'voxel_patch_stride': args.voxel_patch_stride or 16,
            'time_patch_length': args.time_patch_length,
            'time_patch_stride': args.time_patch_stride,
            'fc_patch_size': args.fc_patch_size,
            'fc_patch_stride': args.fc_patch_stride,
        }
        print("-" * 60)
        print("Patch Settings:")
        print(f"  Voxel patch size:  {patch_config['voxel_patch_size']}³")
        print(f"  Voxel patch stride:{patch_config['voxel_patch_stride']}")
        print("=" * 60)

    print(f"Found {len(subject_ids)} subjects")

    if args.num_workers == 1:
        print("Processing sequentially...")
        atlas = Atlas(args.atlas_path, args.n_rois)

        success = 0
        for i, sid in enumerate(subject_ids):
            if args.mode == 'metric':
                result = preprocess_subject_metric(sid, subjects[sid], atlas, output_base, patch_config)
            else:
                result = preprocess_subject_raw(sid, subjects[sid], output_base)

            if result['success']:
                success += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(subject_ids)} subjects")

        print(f"Done! {success}/{len(subject_ids)} successful")

    else:
        print(f"Processing in parallel with {args.num_workers} workers...")

        work_items = [
            (args, sid, subjects[sid], output_base, patch_config)
            for sid in subject_ids
        ]

        start_time = time.time()

        with mp.Pool(processes=args.num_workers) as pool:
            results = pool.map(worker_fn, work_items)

        elapsed = time.time() - start_time

        success = sum(1 for r in results if r['success'])
        errors = [r for r in results if not r['success']]

        print(f"Done! {success}/{len(subject_ids)} successful in {elapsed:.1f}s")
        if errors:
            print(f"Failed: {[r['subject_id'] for r in errors[:5]]}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
