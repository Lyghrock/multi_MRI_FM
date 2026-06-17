#!/usr/bin/env python3
"""
train.py - Multi-Modal Brain Foundation Model Training

MAE Training Entry Point with torchrun support.

Usage:
    # Single GPU
    python train.py --config config/default.yaml --exp_type MAE-METRIC

    # Single node, multi-GPU
    torchrun --nproc_per_node=4 train.py --config config/default.yaml --exp_type MAE-METRIC

    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=4 train.py --config config/default.yaml --exp_type MAE-METRIC

    # With custom settings
    torchrun --nproc_per_node=4 train.py \
        --config config/default.yaml \
        --exp_type MAE-METRIC \
        --epochs 1000 \
        --batch_size 16 \
        --lr 5e-5 \
        --seed 42
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model.foundation_model import BrainFoundationModel
from baseline.MAE import (
    MultiModalMAE,
    MAETrainer,
    create_optimizer,
    create_scheduler
)
from baseline.JEPA import MultiModalJEPA, JEPATrainer
from baseline.DINO import MultiModalDINO, DINOTrainer
from data import create_distributed_dataloaders, BrainMRIDatasetPreprocessed, BrainMRIDataset
from utils.distributed import init_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size
from utils.config_utils import load_config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Multi-Modal Brain Foundation Model'
    )

    # Config
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to config file'
    )

    # Distributed training (set by torchrun)
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training'
    )

    # Training
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Experiment type: determines both training method and data type
    # Format: {METHOD}-{DATATYPE}
    # METHOD: MAE, JEPA, or DINO
    # DATATYPE: RAW or METRIC
    parser.add_argument(
        '--exp_type',
        type=str,
        default='MAE-METRIC',
        choices=['MAE-RAW', 'MAE-METRIC', 'JEPA-RAW', 'JEPA-METRIC', 'DINO-RAW', 'DINO-METRIC'],
        help='Experiment type: MAE/JEPA/DINO training with RAW/METRIC processed data'
    )

    # Paths
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Data root directory (should contain metric/ and raw/ subdirs)'
    )
    parser.add_argument('--train_dir', type=str, default=None, help='Explicit train directory override')
    parser.add_argument('--val_dir', type=str, default=None, help='Explicit validation directory override')
    parser.add_argument('--test_dir', type=str, default=None, help='Explicit test directory override')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (auto-generated if not specified)'
    )

    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    # Logging and saving
    parser.add_argument('--eval_every', type=int, default=None, help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=None, help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=None, help='Log every N steps')

    return parser.parse_args()


def load_model_config(config: dict) -> dict:
    """Load model configuration from config dict."""
    model_config = config.get('model', {})
    atlas_config = config.get('atlas', {})

    return {
        'd_model': model_config.get('d_model', 256),
        'n_heads': model_config.get('n_heads', 8),
        'd_ffn': model_config.get('d_ffn', 1024),
        'n_layers': model_config.get('n_layers', 4),
        'n_rois': atlas_config.get('n_rois', 200),
        'n_latent': model_config.get('n_latent', 64),
        'dropout': model_config.get('dropout', 0.1),
        'fusion_type': model_config.get('fusion_type', 'simplified'),
        'positional_encoding': model_config.get('positional_encoding', 'learnable'),
        'use_flash': model_config.get('use_flash', True),
    }


def load_mae_config(config: dict) -> dict:
    """Load MAE configuration from config dict."""
    mae_config = config.get('mae', {})

    return {
        'mask_ratio': mae_config.get('mask_ratio', 0.75),
        'decoder_hidden_dim': mae_config.get('decoder_hidden_dim', 512),
        'lambda_sMRI': mae_config.get('lambda_sMRI', 1.0),
        'lambda_fMRI': mae_config.get('lambda_fMRI', 1.0),
        'lambda_dMRI': mae_config.get('lambda_dMRI', 1.0),
        'lambda_vicreg': mae_config.get('lambda_vicreg', 0.5),
    }


def load_jepa_config(config: dict) -> dict:
    """Load JEPA configuration from config dict."""
    jepa_config = config.get('jepa', {})
    return {
        'predictor_type': jepa_config.get('predictor_type', 'mlp'),
        'predictor_hidden_dim': jepa_config.get(
            'predictor_hidden_dim',
            jepa_config.get('expert_hidden_dim', None)
        ),
        'predictor_n_heads': jepa_config.get('predictor_n_heads', 4),
        'predictor_n_layers': jepa_config.get('predictor_n_layers', 2),
        'n_experts': jepa_config.get('n_experts', 8),
        'top_k': jepa_config.get('top_k', 2),
        'ema_beta': jepa_config.get('ema_beta', 0.99),
        'mask_ratio': jepa_config.get('mask_refinement', jepa_config.get('intra_mask_ratio', 0.3)),
        'lambda_vicreg': jepa_config.get('lambda_vicreg', 0.5),
    }


def load_dino_config(config: dict) -> dict:
    """Load DINO configuration from config dict."""
    dino_config = config.get('dino', {})
    return {
        'out_dim': dino_config.get('out_dim', 4096),
        'hidden_dim': dino_config.get('hidden_dim', 2048),
        'bottleneck_dim': dino_config.get('bottleneck_dim', 256),
        'head_layers': dino_config.get('head_layers', 3),
        'head_dropout': dino_config.get('head_dropout', 0.0),
        'teacher_momentum': dino_config.get('teacher_momentum', 0.996),
        'student_temp': dino_config.get('student_temp', 0.1),
        'teacher_temp': dino_config.get('teacher_temp', 0.04),
        'center_momentum': dino_config.get('center_momentum', 0.9),
        'n_student_views': dino_config.get('n_student_views', 2),
        'noise_std': dino_config.get('noise_std', 0.01),
        'dropout_prob': dino_config.get('dropout_prob', 0.05),
    }


def get_data_type_from_exp_type(exp_type: str) -> str:
    """Extract data type from experiment type."""
    if 'METRIC' in exp_type:
        return 'metric'
    elif 'RAW' in exp_type:
        return 'raw'
    return 'metric'


def main(args):
    """Main training function."""
    # Initialize distributed training (torchrun sets environment variables)
    dist_info = init_distributed(args)

    rank = args.rank
    world_size = args.world_size
    local_rank = args.local_rank
    is_main = rank == 0

    # Set random seed (only on main process)
    if is_main:
        set_seed(args.seed)

    # Synchronize processes
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Parse experiment type
    exp_type = args.exp_type
    method = exp_type.split('-')[0]
    data_type = get_data_type_from_exp_type(exp_type)

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        if is_main:
            print(f"Warning: Config file {args.config} not found, using defaults")
        config = {}

    # Override config with command line arguments
    if args.epochs is not None:
        config.setdefault('train', {})['epochs'] = args.epochs
    if args.batch_size is not None:
        config.setdefault('data', {})['batch_size'] = args.batch_size
    if args.lr is not None:
        config.setdefault('train', {})['lr'] = args.lr
    if args.weight_decay is not None:
        config.setdefault('train', {})['weight_decay'] = args.weight_decay
    if args.data_root is not None:
        config.setdefault('data', {})['data_root'] = args.data_root
    if args.train_dir is not None:
        config.setdefault('data', {})['train_dir'] = args.train_dir
    if args.val_dir is not None:
        config.setdefault('data', {})['val_dir'] = args.val_dir
    if args.test_dir is not None:
        config.setdefault('data', {})['test_dir'] = args.test_dir
    if args.eval_every is not None:
        config['eval_every'] = args.eval_every
    if args.save_every is not None:
        config['save_every'] = args.save_every
    if args.log_every is not None:
        config['log_every'] = args.log_every

    # Set output directory: outputs/{EXP_TYPE}/seed_{SEED}
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    else:
        config['output_dir'] = f"outputs/{exp_type}/seed_{args.seed}"

    # Data root
    data_config = config.get('data', {})
    data_root = data_config.get('data_root', 'data/')

    # Final data path based on data type
    data_root_expanded = os.path.join(data_root, data_type)

    # Print configuration
    if is_main:
        print("=" * 60)
        print("Multi-Modal Brain Foundation Model - Training")
        print("=" * 60)
        print(f"Experiment:    {exp_type}")
        print(f"Method:       {method}")
        print(f"Data Type:    {data_type}")
        print(f"Data Root:    {data_root_expanded}")
        print(f"Output:       {config['output_dir']}")
        print(f"Seed:        {args.seed}")
        print(f"GPUs:        {world_size}")
        print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    # Create model
    if is_main:
        print("Creating Foundation Model...")

    model_config = load_model_config(config)
    mae_config = load_mae_config(config)
    jepa_config = load_jepa_config(config)
    dino_config = load_dino_config(config)

    # Foundation Model (Encoder + Latent Hub)
    foundation = BrainFoundationModel(**model_config)

    if method == 'MAE':
        model = MultiModalMAE(
            foundation_model=foundation,
            d_model=model_config['d_model'],
            hidden_dim=mae_config['decoder_hidden_dim'],
            n_rois=model_config['n_rois'],
            mask_ratio=mae_config['mask_ratio'],
            decoder_hidden_dim=mae_config['decoder_hidden_dim'],
            lambda_sMRI=mae_config['lambda_sMRI'],
            lambda_fMRI=mae_config['lambda_fMRI'],
            lambda_dMRI=mae_config['lambda_dMRI'],
            lambda_vicreg=mae_config['lambda_vicreg'],
        )
        trainer_cls = MAETrainer
    elif method == 'JEPA':
        model = MultiModalJEPA(
            foundation_model=foundation,
            d_model=model_config['d_model'],
            n_rois=model_config['n_rois'],
            dropout=model_config['dropout'],
            **jepa_config,
        )
        trainer_cls = JEPATrainer
    elif method == 'DINO':
        model = MultiModalDINO(
            foundation_model=foundation,
            d_model=model_config['d_model'],
            **dino_config,
        )
        trainer_cls = DINOTrainer
    else:
        raise ValueError(f"Unknown training method: {method}")

    # Print model info
    if is_main:
        if hasattr(model, 'get_parameters'):
            n_params = model.get_parameters()
            print(f"Total parameters: {n_params['total']:,}")
            print(f"  - Encoder: {n_params['encoder']:,}")
            print(f"  - Decoder: {n_params['decoder']:,}")
        else:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {n_params:,}")

    # Move to device
    model = model.to(device)

    # Wrap with DDP
    if world_size > 1:
        ddp_kwargs = {}
        if device.type == 'cuda':
            ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
        model = DDP(
            model,
            **ddp_kwargs,
            find_unused_parameters=True
        )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    num_epochs = config.get('train', {}).get('epochs', 500)
    scheduler = create_scheduler(optimizer, config, num_epochs)

    # Create data loaders
    if is_main:
        print("Loading data...")

    batch_size = data_config.get('batch_size', 16)
    num_workers = data_config.get('num_workers', 4)
    max_patches = data_config.get('max_patches_per_modality', 8)

    # Check if data exists
    train_path = data_config.get('train_dir') or os.path.join(data_root_expanded, data_config.get('train_split', 'train'))
    if not os.path.exists(train_path):
        if is_main:
            print(f"Warning: Data directory {train_path} not found")
            print("Using dummy data loader for testing...")

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100, n_rois=200):
                self.size = size
                self.n_rois = n_rois

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                n = self.n_rois
                return {
                    'subject_id': f'sub_{idx:04d}',
                    'sMRI': {
                        'patches': torch.randn(8, 32, 32, 32).half(),
                        'roi': torch.randn(n, 3).half(),
                    },
                    'fMRI': {
                        'time_patches': torch.randn(4, 50, n).half(),
                        'fc_patches': torch.randn(16, 20, 20).half(),
                    },
                    'dMRI': {
                        'FA_patches': torch.randn(8, 32, 32, 32).half(),
                        'MD_patches': torch.randn(8, 32, 32, 32).half(),
                        'SC_matrix': torch.randn(n, n).half(),
                        'SC_patches': torch.randn(16, 20, 20).half(),
                    },
                    'mode': 'metric',
                }

        train_loader = torch.utils.data.DataLoader(
            DummyDataset(size=100, n_rois=model_config['n_rois']),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            DummyDataset(size=20, n_rois=model_config['n_rois']),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    else:
        # Use distributed dataloaders with proper sampling
        train_loader, val_loader, _ = create_distributed_dataloaders(
            data_root=data_root_expanded,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=0.1,
            rank=rank,
            world_size=world_size,
            use_preprocessed=True,
            n_rois=model_config['n_rois'],
            max_patches_per_modality=max_patches,
            train_split=data_config.get('train_split', 'train'),
            val_split_name=data_config.get('val_split', 'val'),
            test_split_name=data_config.get('test_split', 'test'),
            train_dir=data_config.get('train_dir'),
            val_dir=data_config.get('val_dir'),
            test_dir=data_config.get('test_dir'),
        )

    # Create trainer
    if is_main:
        print("Creating trainer...")

    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        config=config,
        output_dir=config['output_dir'],
        log_every=config.get('log_every', 100),
        save_every=config.get('save_every', 10),
        eval_every=config.get('eval_every', 1),
        clip_grad=config.get('train', {}).get('clip_grad', 1.0),
        use_amp=(device.type == 'cuda')
    )

    # Start training
    if is_main:
        print("Starting training...")
        print("-" * 60)

    start_time = time.time()

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        resume_from=args.resume
    )

    # Training finished
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    if is_main:
        print("-" * 60)
        print(f"Training completed in {total_time / 3600:.2f} hours")
        print(f"Output saved to: {config['output_dir']}")

    # Cleanup distributed training
    cleanup_distributed()


if __name__ == '__main__':
    args = parse_args()
    main(args)
