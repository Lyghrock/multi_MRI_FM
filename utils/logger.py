"""
Logging Utilities

Provides utilities for logging training progress.
"""

import os
import sys
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

from .distributed import is_main_process, get_rank


def get_logger(
    name: str = 'brain_fm',
    level: int = logging.INFO
) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def setup_logger(
    name: str = 'brain_fm',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_str: Optional custom format string

    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'

    formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class Logger:
    """
    Training logger with both console/file and TensorBoard support.
    """

    def __init__(
        self,
        log_dir: str,
        name: str = 'brain_fm',
        use_tensorboard: bool = True,
        rank: int = 0
    ):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            name: Logger name
            use_tensorboard: Whether to use TensorBoard
            rank: Current process rank
        """
        self.rank = rank
        self.is_main = rank == 0

        if not self.is_main:
            # Only main process logs to file
            self.logger = get_logger(name)
            self.writer = None
            return

        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logger
        log_file = self.log_dir / 'train.log'
        self.logger = setup_logger(name, str(log_file))

        # Set up TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / 'tensorboard'
                self.writer = SummaryWriter(str(tb_dir))
            except Exception as exc:
                self.logger.warning(f"TensorBoard disabled: {exc}")

        self.logger.info(f"Logger initialized. Log directory: {self.log_dir}")

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def log(self, msg: str, level: int = logging.INFO):
        """Log message at specified level."""
        self.logger.log(level, msg)

    def log_epoch(
        self,
        epoch: int,
        step: int,
        metrics: dict,
        lr: float,
        elapsed_time: float
    ):
        """
        Log epoch summary.

        Args:
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            lr: Current learning rate
            elapsed_time: Time elapsed since start
        """
        if not self.is_main:
            return

        msg_parts = [f"Epoch {epoch} | Step {step} | LR {lr:.2e} | Time {elapsed_time:.1f}s"]

        for key, value in metrics.items():
            if isinstance(value, float):
                msg_parts.append(f"{key} {value:.4f}")
            else:
                msg_parts.append(f"{key} {value}")

        msg = " | ".join(msg_parts)
        self.logger.info(msg)

        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'epoch/{key}', value, epoch)
            self.writer.add_scalar('epoch/learning_rate', lr, epoch)
            self.writer.add_scalar('epoch/time', elapsed_time, epoch)

    def log_step(
        self,
        step: int,
        metrics: dict,
        lr: float,
        step_time: float
    ):
        """
        Log training step.

        Args:
            step: Current step
            metrics: Dictionary of metrics
            lr: Current learning rate
            step_time: Time for this step
        """
        if not self.is_main:
            return

        msg_parts = [f"Step {step} | LR {lr:.2e} | {step_time:.3f}s/step"]

        for key, value in metrics.items():
            if isinstance(value, float):
                msg_parts.append(f"{key} {value:.4f}")

        msg = " | ".join(msg_parts)
        self.logger.info(msg)

        # TensorBoard
        if self.writer is not None:
            global_step = step
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'step/{key}', value, global_step)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag: str, image, step: int):
        """Log an image."""
        if self.writer is not None:
            self.writer.add_image(tag, image, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def log_graph(self, model, input_data):
        """Log model graph."""
        if self.writer is not None:
            self.writer.add_graph(model, input_data)

    def close(self):
        """Close logger and writer."""
        if self.writer is not None:
            self.writer.close()


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        if self.name:
            return f'{self.name}: {self.avg:.4f} (current: {self.val:.4f})'
        return f'{self.avg:.4f} (current: {self.val:.4f})'


class ProgressTracker:
    """
    Track progress across distributed training.
    """

    def __init__(self, total_steps: int, log_every: int = 100):
        self.total_steps = total_steps
        self.log_every = log_every
        self.current_step = 0
        self.meters = {}

    def add_meter(self, name: str):
        """Add a meter for tracking."""
        if name not in self.meters:
            self.meters[name] = AverageMeter(name)

    def update(self, **kwargs):
        """Update meter values."""
        for name, value in kwargs.items():
            self.add_meter(name)
            self.meters[name].update(value)
        self.current_step += 1

    def get_metrics(self) -> dict:
        """Get current metric values."""
        return {name: meter.avg for name, meter in self.meters.items()}

    def should_log(self) -> bool:
        """Check if we should log this step."""
        return self.current_step % self.log_every == 0

    def get_progress_str(self) -> str:
        """Get progress string."""
        meter_strs = [str(meter) for meter in self.meters.values()]
        return f"Step {self.current_step}/{self.total_steps} | " + " | ".join(meter_strs)
