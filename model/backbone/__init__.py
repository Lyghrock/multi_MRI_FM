"""
Backbone package - Shared Transformer components
"""

from .transformer import TransformerEncoder, TransformerBlock
from .positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    BrainAwarePositionalEncoding,
    FunctionalNetworkEncoding,
    AnatomicalPositionalEncoding
)

PositionalEncoding = SinusoidalPositionalEncoding

__all__ = [
    'TransformerEncoder',
    'TransformerBlock',
    'PositionalEncoding',
    'SinusoidalPositionalEncoding',
    'LearnablePositionalEncoding',
    'BrainAwarePositionalEncoding',
    'FunctionalNetworkEncoding',
    'AnatomicalPositionalEncoding',
]
