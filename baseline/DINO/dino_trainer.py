"""DINO trainer with DDP support."""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from utils.batch_utils import infer_batch_size, move_to_device
from utils.checkpoint import CheckpointManager
from utils.distributed import barrier, get_rank, get_world_size, is_main_process, reduce_tensor
from utils.logger import AverageMeter, Logger, ProgressTracker


class DINOTrainer:
    """Training loop for MultiModalDINO."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        config: Optional[dict] = None,
        output_dir: str = "outputs",
        log_every: int = 100,
        save_every: int = 1,
        eval_every: int = 1,
        clip_grad: float = 1.0,
        use_amp: bool = True,
    ):
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
        self.use_amp = use_amp and torch.cuda.is_available() and str(device).startswith("cuda")

        self.is_main = is_main_process()
        self.rank = get_rank()
        self.world_size = get_world_size()

        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)
            self.logger = Logger(output_dir, name="dino_train", use_tensorboard=True, rank=self.rank)
        else:
            self.logger = None

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            keep_last=5,
            keep_best=True,
            save_every=save_every,
        )
        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.current_epoch = 0
        self.global_step = 0

    def train(self, train_loader, val_loader=None, num_epochs: int = 500, resume_from: Optional[str] = None):
        if resume_from and self.is_main:
            self._resume_checkpoint(resume_from)

        barrier()
        history = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_loader)
            val_loss = None
            if val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_loss = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            if self.is_main:
                self.logger.log_epoch(
                    epoch=epoch,
                    step=self.global_step,
                    metrics={"train_loss": train_loss, "val_loss": val_loss or 0.0},
                    lr=lr,
                    elapsed_time=0,
                )
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss or 0.0)
                history["lr"].append(lr)
                if (epoch + 1) % self.save_every == 0:
                    self._save_checkpoint(epoch, val_loss)

        if self.is_main and self.logger:
            self.logger.close()
        return history

    def train_epoch(self, loader) -> float:
        self.model.train()
        tracker = ProgressTracker(total_steps=len(loader), log_every=self.log_every)
        tracker.add_meter("loss")

        for batch in loader:
            batch_start = time.time()
            batch = move_to_device(batch, self.device)

            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(batch, return_loss=True)
                loss = outputs["loss"]
                metrics = outputs.get("metrics", {})

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            module = self.model.module if hasattr(self.model, "module") else self.model
            module.update_teacher()

            tracker.update(loss=loss.item(), **{f"loss/{k}": v for k, v in metrics.items()})
            if tracker.should_log() and self.is_main:
                self.logger.log_step(
                    step=self.global_step,
                    metrics=tracker.get_metrics(),
                    lr=self.optimizer.param_groups[0]["lr"],
                    step_time=time.time() - batch_start,
                )
            self.global_step += 1

        metrics = tracker.get_metrics()
        epoch_loss = metrics.get("loss/loss_total", metrics.get("loss", 0.0))
        if self.world_size > 1:
            epoch_loss = reduce_tensor(torch.tensor(epoch_loss, device=self.device)).item()
        return epoch_loss

    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        meter = AverageMeter()
        for batch in loader:
            batch = move_to_device(batch, self.device)
            outputs = self.model(batch, return_loss=True)
            meter.update(outputs["loss"].item(), infer_batch_size(batch))
        return meter.avg

    def _save_checkpoint(self, epoch: int, val_loss: Optional[float]):
        metric = val_loss if val_loss is not None else 0.0
        self.checkpoint_manager.save(
            epoch=epoch,
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics={"loss": metric},
        )

    def _resume_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("step", checkpoint.get("global_step", 0))
