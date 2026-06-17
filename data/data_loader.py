"""
Data Loader Utilities

Provides functions for creating data loaders with proper distributed sampling.
"""

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Tuple

from .brain_dataset import (
    BrainMRIDataset,
    BrainMRIDatasetPreprocessed,
    PairedBrainMRIDataset,
    PatchedSampleCollator,
    collate_brain_mri
)


def create_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.1,
    use_preprocessed: bool = True,
    max_patches_per_modality: int = 8,
    train_split: str = 'train',
    val_split_name: str = 'val',
    test_split_name: str = 'test',
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_root: Root directory containing the data
                   Structure: data_root/{metric,raw}/{train,val,test}/*.pt
        batch_size: Batch size (number of subjects per batch)
        num_workers: Number of worker processes
        val_split: Fraction of training data to use for validation
        use_preprocessed: Whether to use preprocessed dataset
        max_patches_per_modality: Max patches to load per subject per modality

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_class = BrainMRIDatasetPreprocessed if use_preprocessed else BrainMRIDataset

    full_train_dataset = dataset_class(
        data_root=data_root,
        split=train_split,
        split_dir=train_dir,
        max_patches_per_modality=max_patches_per_modality,
        **dataset_kwargs
    )

    # Use an explicit validation directory when available; otherwise split train.
    if val_dir is not None or os.path.exists(os.path.join(data_root, val_split_name)):
        train_dataset = full_train_dataset
        val_dataset = dataset_class(
            data_root=data_root,
            split=val_split_name,
            split_dir=val_dir,
            max_patches_per_modality=max_patches_per_modality,
            **dataset_kwargs
        )
    else:
        n_train = int(len(full_train_dataset) * (1 - val_split))
        n_val = len(full_train_dataset) - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

    # Test dataset (if exists)
    test_dataset = None
    test_path = test_dir or os.path.join(data_root, test_split_name)
    if os.path.exists(test_path):
        test_dataset = dataset_class(
            data_root=data_root,
            split=test_split_name,
            split_dir=test_dir,
            max_patches_per_modality=max_patches_per_modality,
            **dataset_kwargs
        )

    # Collator for patching (limits patches per batch to save memory)
    collator = PatchedSampleCollator(
        max_voxel_patches=max_patches_per_modality,
        max_fc_patches=max_patches_per_modality * 2,
        max_time_patches=max_patches_per_modality // 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=False
        )

    return train_loader, val_loader, test_loader


def create_distributed_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.1,
    rank: int = 0,
    world_size: int = 1,
    use_preprocessed: bool = True,
    max_patches_per_modality: int = 8,
    train_split: str = 'train',
    val_split_name: str = 'val',
    test_split_name: str = 'test',
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create distributed dataloaders with proper sampling.

    Args:
        data_root: Root directory containing the data
        batch_size: Batch size per GPU
        num_workers: Number of worker processes per GPU
        val_split: Fraction of training data for validation
        rank: Current process rank
        world_size: Total number of processes
        use_preprocessed: Whether to use preprocessed dataset
        max_patches_per_modality: Max patches to load per subject per modality

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_class = BrainMRIDatasetPreprocessed if use_preprocessed else BrainMRIDataset

    full_train_dataset = dataset_class(
        data_root=data_root,
        split=train_split,
        split_dir=train_dir,
        max_patches_per_modality=max_patches_per_modality,
        **dataset_kwargs
    )

    if val_dir is not None or os.path.exists(os.path.join(data_root, val_split_name)):
        train_dataset = full_train_dataset
        val_dataset = dataset_class(
            data_root=data_root,
            split=val_split_name,
            split_dir=val_dir,
            max_patches_per_modality=max_patches_per_modality,
            **dataset_kwargs
        )
    else:
        n_train = int(len(full_train_dataset) * (1 - val_split))
        n_val = len(full_train_dataset) - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

    # Test dataset
    test_dataset = None
    test_path = test_dir or os.path.join(data_root, test_split_name)
    if os.path.exists(test_path):
        test_dataset = dataset_class(
            data_root=data_root,
            split=test_split_name,
            split_dir=test_dir,
            max_patches_per_modality=max_patches_per_modality,
            **dataset_kwargs
        )

    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    # Collator for patching
    collator = PatchedSampleCollator(
        max_voxel_patches=max_patches_per_modality,
        max_fc_patches=max_patches_per_modality * 2,
        max_time_patches=max_patches_per_modality // 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False
    )

    test_loader = None
    if test_dataset is not None:
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=False
        )

    return train_loader, val_loader, test_loader


class DataLoaderWrapper:
    """
    Wrapper for DataLoader with epoch tracking.

    Automatically handles sampler epoch updates for distributed training.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        distributed: bool = False
    ):
        self.dataloader = dataloader
        self.distributed = distributed
        self.epoch = 0

    def __iter__(self):
        if self.distributed and hasattr(self.dataloader.sampler, 'set_epoch'):
            self.dataloader.sampler.set_epoch(self.epoch)

        self.epoch += 1
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset


def get_dataloader_info(dataloader: DataLoader) -> dict:
    """
    Get information about a dataloader.

    Args:
        dataloader: DataLoader to inspect

    Returns:
        Dictionary with dataloader info
    """
    dataset = dataloader.dataset

    info = {
        'num_samples': len(dataset),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'num_workers': dataloader.num_workers,
    }

    if hasattr(dataset, 'get_stats'):
        info.update(dataset.get_stats())

    return info
