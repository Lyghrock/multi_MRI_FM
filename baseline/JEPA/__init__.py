"""JEPA package for Brain MRI foundation model training."""

from .jepa_model import MultiModalJEPA
from .jepa_loss import JEPALoss, soft_l1_loss
from .jepa_trainer import JEPATrainer
from .masking import JEPACurriculumMasker, JEPAMaskConfig, random_token_mask
from .predictor import MLPPredictor, TransformerPredictor, MoEPredictor, create_predictor

__all__ = [
    'MultiModalJEPA',
    'JEPALoss',
    'soft_l1_loss',
    'JEPATrainer',
    'JEPACurriculumMasker',
    'JEPAMaskConfig',
    'random_token_mask',
    'MLPPredictor',
    'TransformerPredictor',
    'MoEPredictor',
    'create_predictor',
]
