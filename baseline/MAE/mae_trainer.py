"""
MAE Trainer

Training loop for Multi-Modal MAE with DDP support.
"""

import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict

from .mae_model import MultiModalMAE
from .mae_loss import MultiModalMAELoss
from utils.distributed import (
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    reduce_tensor
)
from utils.checkpoint import CheckpointManager
from utils.logger import Logger, AverageMeter, ProgressTracker
from utils.batch_utils import infer_batch_size, move_to_device


class MAETrainer:
    """
    MAE Trainer with distributed training support.

    Handles training loop, logging, checkpointing, and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        config: Optional[dict] = None,
        output_dir: str = 'outputs',
        log_every: int = 100,
        save_every: int = 1,
        eval_every: int = 1,
        clip_grad: float = 1.0,
        use_amp: bool = True
    ):
        """
        Args:
            model: MultiModalMAE model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Training configuration
            output_dir: Directory for outputs
            log_every: Log every N steps
            save_every: Save checkpoint every N epochs
            eval_every: Evaluate every N epochs
            clip_grad: Gradient clipping value
            use_amp: Use automatic mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.output_dir = output_dir
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.clip_grad = clip_grad
        self.use_amp = use_amp

        # Distributed settings
        self.is_main = is_main_process()
        self.rank = get_rank()
        self.world_size = get_world_size()

        # Create output directory
        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)

        # Logger
        if self.is_main:
            self.logger = Logger(
                log_dir=output_dir,
                name='mae_train',
                use_tensorboard=True,
                rank=self.rank
            )
        else:
            self.logger = None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(output_dir, 'checkpoints'),
            keep_last=5,
            keep_best=True,
            save_every=save_every
        )

        # Grad scaler for AMP
        self.use_amp = use_amp and torch.cuda.is_available() and str(device).startswith('cuda')
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # Progress tracker
        self.train_tracker = ProgressTracker(total_steps=0, log_every=log_every)

    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 500,
        resume_from: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from

        Returns:
            Training history
        """
        # Resume from checkpoint
        if resume_from and self.is_main:
            self._resume_checkpoint(resume_from)

        # Synchronize before starting
        barrier()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }

        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss = None
            if val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_loss = self.validate(val_loader, epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # Logging
            if self.is_main:
                self.logger.log_epoch(
                    epoch=epoch,
                    step=self.global_step,
                    metrics={'train_loss': train_loss, 'val_loss': val_loss or 0},
                    lr=current_lr,
                    elapsed_time=0
                )

                # Save history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss or 0)
                history['lr'].append(current_lr)

                # Save checkpoint
                if (epoch + 1) % self.save_every == 0:
                    self._save_checkpoint(epoch, val_loss)

        if self.is_main and self.logger:
            self.logger.close()

        return history

    def train_epoch(self, loader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()

        # Progress tracker
        self.train_tracker = ProgressTracker(
            total_steps=len(loader),
            log_every=self.log_every
        )

        # Add meters
        self.train_tracker.add_meter('loss')
        self.train_tracker.add_meter('loss/total')
        self.train_tracker.add_meter('loss/sMRI')
        self.train_tracker.add_meter('loss/fMRI')
        self.train_tracker.add_meter('loss/dMRI')
        self.train_tracker.add_meter('loss/vicreg')

        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            batch_start = time.time()

            # Move to device
            batch = self._move_to_device(batch)

            # Forward
            with autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(batch, apply_mask=True, return_loss=True)
                loss = outputs['loss']
                metrics = outputs.get('metrics', {})

            # Backward
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad
                    )

                self.optimizer.step()

            # Update tracker
            self.train_tracker.update(
                loss=loss.item(),
                **{f'loss/{k}': v for k, v in metrics.items()}
            )

            # Logging
            if self.train_tracker.should_log() and self.is_main:
                step_time = time.time() - batch_start
                self.logger.log_step(
                    step=self.global_step,
                    metrics=self.train_tracker.get_metrics(),
                    lr=self.optimizer.param_groups[0]['lr'],
                    step_time=step_time
                )

            self.global_step += 1

        # Epoch summary
        metrics = self.train_tracker.get_metrics()
        epoch_loss = metrics.get('loss/loss_total', metrics.get('loss', 0))

        if self.world_size > 1:
            epoch_loss = reduce_tensor(torch.tensor(epoch_loss)).item()

        return epoch_loss

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> float:
        """Validate the model."""
        self.model.eval()

        val_loss_meter = AverageMeter()

        for batch in loader:
            batch = self._move_to_device(batch)

            outputs = self.model(batch, apply_mask=False, return_loss=True)
            loss = outputs['loss']

            val_loss_meter.update(loss.item(), infer_batch_size(batch))

        return val_loss_meter.avg

    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch to training device."""
        device = self.device

        return move_to_device(batch, device)

    def _save_checkpoint(self, epoch: int, val_loss: Optional[float]):
        """Save training checkpoint."""
        is_best = val_loss is not None and val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss

        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }

        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()

        self.checkpoint_manager.save(
            epoch=epoch,
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics={'loss': val_loss or 0}
        )

    def _resume_checkpoint(self, checkpoint_path: str):
        """Resume from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if self.is_main:
            print(f"Resumed from epoch {self.current_epoch}")


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        model: Model to optimize
        config: Configuration dict

    Returns:
        Optimizer
    """
    train_config = config.get('train', {})

    optimizer_type = train_config.get('optimizer', 'adamw')
    lr = train_config.get('lr', 1e-4)
    weight_decay = train_config.get('weight_decay', 0.05)

    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=train_config.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    num_epochs: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from config.

    Args:
        optimizer: Optimizer
        config: Configuration dict
        num_epochs: Number of training epochs

    Returns:
        Scheduler
    """
    train_config = config.get('train', {})
    scheduler_type = train_config.get('scheduler', 'cosine')
    min_lr = train_config.get('min_lr', 1e-6)
    warmup_epochs = train_config.get('warmup_epochs', 10)

    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == 'cosine_warmup':
        # Cosine with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'step':
        step_size = train_config.get('step_size', 100)
        gamma = train_config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
