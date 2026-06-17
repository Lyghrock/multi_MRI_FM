"""
MAE package - Masked Autoencoder training for Brain MRI
"""

from .mae_model import MultiModalMAE, MAEMasking, MAEWrapper
from .mae_loss import MultiModalMAELoss, VICRegLoss
from .mae_trainer import MAETrainer, create_optimizer, create_scheduler

__all__ = [
    'MultiModalMAE',
    'MAEMasking',
    'MAEWrapper',
    'MultiModalMAELoss',
    'VICRegLoss',
    'MAETrainer',
    'create_optimizer',
    'create_scheduler',
]
