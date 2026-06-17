"""
Utils package for Multi-Modal Brain Foundation Model
"""

from .distributed import (
    init_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    reduce_tensor,
    sync_model,
)
from .config_utils import load_config, merge_config
from .checkpoint import save_checkpoint, load_checkpoint
from .batch_utils import canonicalize_brain_batch, make_mae_targets, move_to_device


def __getattr__(name):
    if name in {'get_logger', 'setup_logger'}:
        from .logger import get_logger, setup_logger
        return {'get_logger': get_logger, 'setup_logger': setup_logger}[name]
    if name == 'compute_metrics':
        from .metrics import compute_metrics
        return compute_metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'init_distributed',
    'cleanup_distributed',
    'is_main_process',
    'get_rank',
    'get_world_size',
    'barrier',
    'reduce_tensor',
    'sync_model',
    'load_config',
    'merge_config',
    'save_checkpoint',
    'load_checkpoint',
    'get_logger',
    'setup_logger',
    'compute_metrics',
    'canonicalize_brain_batch',
    'make_mae_targets',
    'move_to_device',
]
